"""Evaluation engine: read-only measurement of an existing model state.

Milestone 4. This module measures any *existing* state of a user model —
either the current published ``weights.pt`` or any verified immutable
checkpoint of that model — against one deterministic tokenized dataset split,
using the exact same causal-LM objective as the M3 training engine.

Read-only contract (tested):
  * never modifies model weights, checkpoints, training provenance, datasets,
    tokenizers or any earlier evaluation record
  * the only artifact an evaluation creates is one immutable manifest under
    ``models/<id>/evaluations/eval-<id>/manifest.json`` (no copies of
    weights/data are stored anywhere)
  * corrupted states are refused BEFORE the evaluation loop runs

Semantics (documented, honest):
  * metrics are loss_nats (token-weighted mean cross-entropy over the
    evaluated target tokens, fp32) and perplexity = exp(min(loss, 100));
    lower is better for THIS objective, and these two numbers do NOT define
    overall model quality — comparisons are only meaningful under the same
    dataset/version/split/tokenizer/window objective
  * data policy mirrors M3: the flat token stream is cut into window-length
    sequences in file order (no shuffle); the incomplete tail is dropped;
    final partial batches are handled exactly like M3's evaluator; a
    ``max_eval_tokens`` cap stops deterministically mid-split (the cap counts
    usable *target* tokens; a partial final window contributes exactly the
    remaining targets)
  * M2 stores flat token streams without per-token record boundaries, so for a
    capped (truncated) evaluation ``records_covered`` is None — we never
    estimate record coverage from token counts. Full-split evaluations report
    the exact record count of the split recorded by the M2 version manifest.
  * the evaluation loop is pure (no RNG): dropout is off in eval mode and the
    batch order is fixed, so no seeding is required for determinism; the
    effective seed is still recorded for provenance/result_hash identity
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from . import config as forge_cfg
from .dataset import DatasetEngine
from .hardware import detect_hardware
from .model_builder import build_transformer, content_hash, restore_state
from .schemas import (
    EvalStateKind,
    EvaluationConfig,
    EvaluationRecord,
    EvaluationSplit,
    ModelRecord,
)
from .storage import Storage, atomic_write_json, read_json
from .tokenizer import TokenizerEngine
from .training import TrainingEngine, open_bin

log = forge_cfg.get_logger("evaluation")

EVAL_MANIFEST = "manifest.json"
EVALUATIONS_DIR = "evaluations"
_OVERHEAD_BYTES = 32 * 2**20  # same conservative process-overhead figure as training


def estimate_eval_bytes(param_count: int, batch: int, seq_len: int,
                        hidden: int, n_layers: int) -> int:
    """Rough peak memory for one evaluation forward pass.

    Unlike training there is no optimizer state: fp32 weights (4 B/param) +
    conservative activations + process overhead.
    """
    activations = batch * seq_len * hidden * (n_layers + 2) * 4 * 8
    return _OVERHEAD_BYTES + param_count * 4 + activations


class EvaluationEngine:
    """Everything evaluation-related for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.datasets = DatasetEngine(storage)
        self.tokenizers = TokenizerEngine(storage)
        self.training = TrainingEngine(storage)  # checkpoint verification reuse

    # ------------------------------------------------------------------ #
    # Paths / registry helpers
    # ------------------------------------------------------------------ #

    def _evals_root(self, model_id: str) -> Path:
        return self.storage.model_dir(model_id) / EVALUATIONS_DIR

    def _eval_dir(self, model_id: str, eval_id: str) -> Path:
        return self._evals_root(model_id) / f"eval-{eval_id}"

    def _model_exists(self, model_id: str) -> bool:
        return (self.storage.model_dir(model_id) / "manifest.json").exists()

    def list_evaluations(self, model_id: str) -> list[EvaluationRecord]:
        """Immutable evaluation history of a model (append-only, no pointers).

        Raises FileNotFoundError for an unknown model; returns [] when the
        model has no evaluations yet.
        """
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        root = self._evals_root(model_id)
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith("eval-"):
                continue
            mpath = d / EVAL_MANIFEST
            if mpath.exists():
                try:
                    records.append(EvaluationRecord(**read_json(mpath)))
                except Exception:
                    log.warning("unreadable evaluation manifest %s", mpath)
        records.sort(key=lambda r: (r.created_at, r.eval_id))
        return records

    def get_evaluation(self, model_id: str, eval_id: str) -> EvaluationRecord:
        """Return one persisted immutable evaluation record (never mutates it)."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        path = self._eval_dir(model_id, eval_id) / EVAL_MANIFEST
        if not path.exists():
            raise FileNotFoundError(
                f"evaluation '{eval_id}' not found for model '{model_id}'")
        return EvaluationRecord(**read_json(path))

    # ------------------------------------------------------------------ #
    # M24: per-checkpoint access (read-only)
    # ------------------------------------------------------------------ #

    def list_evaluations_for_checkpoint(
            self, model_id: str, checkpoint_id: str) -> list[EvaluationRecord]:
        """Immutable M4 evaluations recorded under ONE checkpoint of ONE
        model (M24).

        Resolution: unknown model or unregistered checkpoint ->
        FileNotFoundError; ownership is validated through the model's M3
        checkpoint registry (``TrainingEngine.get_checkpoint``) — a
        checkpoint id belonging to another model is not registered under
        this model and raises FileNotFoundError, exactly like an unknown
        one (nothing is inferred from filenames). The result is the
        model's authoritative M4 listing above (the exact engine parse +
        deterministic (created_at, eval_id) ASCENDING order) filtered by
        the persisted ``checkpoint_id`` recorded in each
        EvaluationRecord, so every returned record is a complete
        verbatim EvaluationRecord and records of other checkpoints or
        models never appear. Current-state evaluations keep
        ``checkpoint_id=None`` and therefore never appear. A valid
        checkpoint with no evaluations returns []. Read-only, never
        writes.
        """
        # existence + ownership: raises FileNotFoundError (404 at the API)
        self.training.get_checkpoint(model_id, checkpoint_id)
        return [r for r in self.list_evaluations(model_id)
                if r.checkpoint_id == checkpoint_id]

    def list_evaluations_for_dataset(
            self, model_id: str, dataset_id: str) -> list[EvaluationRecord]:
        """Immutable M4 evaluations of ONE model over ONE dataset (M28).

        Membership comes from the persisted dataset identity ONLY: every
        ``EvaluationRecord`` carries top-level ``dataset_id: str`` and
        ``dataset_version: int`` fields, and an evaluation belongs to
        the request when its persisted ``dataset_id`` equals the
        requested id — never filenames, dataset directory names,
        tokenizer ids, eval ids, checkpoint ids, hashes or timestamps.
        The persisted ``dataset_version`` travels VERBATIM inside each
        returned record (all versions of the dataset are returned, each
        exactly as persisted — versions are neither collapsed, nor
        resolved to the latest, nor rewritten). Resolution: unknown
        model or unknown dataset -> FileNotFoundError; the dataset is
        validated through the M2 registry
        (``DatasetEngine.load_meta`` — the same registry call M4's own
        run preflight uses; datasets are GLOBAL, so the model scoping
        comes from the model's own M4 listing exactly like M24/M27: a
        model never sees another model's evaluations). The result is
        the model's authoritative M4 listing above (the exact engine
        parse + deterministic (created_at, eval_id) ASCENDING order)
        filtered by the persisted dataset identity. A valid dataset
        with no evaluations for the model returns []. Read-only, never
        writes.
        """
        # existence: raises FileNotFoundError (404 at the API)
        self.datasets.load_meta(dataset_id)
        return [r for r in self.list_evaluations(model_id)
                if r.dataset_id == dataset_id]

    def list_evaluations_for_tokenizer(
            self, model_id: str, tokenizer_id: str) -> list[EvaluationRecord]:
        """Immutable M4 evaluations of ONE model measured with ONE
        tokenizer (M30).

        Membership comes from the persisted tokenizer identity ONLY:
        every ``EvaluationRecord`` carries a top-level
        ``tokenizer_id: str`` field, and an evaluation belongs to the
        request when its persisted ``tokenizer_id`` equals the
        requested id — never filenames, eval ids, checkpoint ids,
        dataset identities, the tokenizer currently registered, or a
        latest-tokenizer substitution. The persisted identity is
        matched VERBATIM (tokenizer ids are opaque ids; there is no
        tokenizer versioning in M2/M4 and none is introduced here).
        Resolution: unknown model or unknown tokenizer ->
        FileNotFoundError; the tokenizer is validated through the
        existing tokenizer registry (``TokenizerEngine.load`` — the
        same registry getter ``GET /tokenizers/{id}`` exposes;
        tokenizers are GLOBAL, so the model scoping comes from the
        model's own M4 listing exactly like M28/M29: a model never
        sees another model's evaluations). The result is the model's
        authoritative M4 listing above (the exact engine parse +
        deterministic (created_at, eval_id) ASCENDING order) filtered
        by the persisted tokenizer identity; complete verbatim
        ``EvaluationRecord`` payloads, no rewritten fields. A valid
        tokenizer with no evaluations for the model returns [].
        Read-only, never writes.
        """
        # existence: raises FileNotFoundError (404 at the API)
        self.tokenizers.load(tokenizer_id)
        return [r for r in self.list_evaluations(model_id)
                if r.tokenizer_id == tokenizer_id]

    def list_evaluations_for_split(
            self, model_id: str, split: EvaluationSplit
    ) -> list[EvaluationRecord]:
        """Immutable M4 evaluations of ONE model measured on ONE
        dataset split (M36).

        Membership comes from the persisted split identity ONLY: every
        ``EvaluationRecord`` carries a top-level ``split:
        EvaluationSplit`` (the schema enum train/validation/test,
        persisted verbatim at run time from the evaluation request),
        and an evaluation belongs to the request when its persisted
        ``split`` equals the requested value — never filenames,
        directories, timestamps, dataset names, eval ids or hashes,
        and never resolved or rewritten into another value. Splits
        have NO registry (unlike the M24 checkpoint / M28 dataset /
        M30 tokenizer axes): the enum IS the contract, so an
        unsupported split value is rejected at the API boundary with
        422 (schema-level validation), while an unknown model raises
        FileNotFoundError exactly like the sibling groupings. Each
        evaluation appears EXACTLY ONCE (the authoritative listing
        holds each record exactly once). The result is the model's
        authoritative M4 listing above (the exact engine parse +
        deterministic (created_at, eval_id) ASCENDING order) filtered
        by the persisted split; complete verbatim
        ``EvaluationRecord`` payloads, no rewritten fields. A valid
        split with no evaluations for the model returns []. Splits
        are model-scoped through the listing itself — a model never
        sees another model's evaluations. Read-only, never writes.
        """
        # the authoritative listing validates the model:
        # raises FileNotFoundError (404 at the API)
        return [r for r in self.list_evaluations(model_id)
                if r.split == split]

    def list_evaluations_for_state_kind(
            self, model_id: str, state_kind: EvalStateKind
    ) -> list[EvaluationRecord]:
        """Immutable M4 evaluations of ONE model measuring ONE kind of
        model state (M38).

        Membership comes from the persisted state-kind identity ONLY:
        every ``EvaluationRecord`` carries a top-level ``state_kind:
        EvalStateKind`` (the schema enum current/checkpoint, persisted
        verbatim at run time from the resolved evaluation request —
        ``current`` measured the model's published weights.pt,
        ``checkpoint`` measured one immutable stored checkpoint), and
        an evaluation belongs to the request when its persisted
        ``state_kind`` equals the requested value — never filenames,
        directories, timestamps, eval ids or hashes, and NEVER the
        ``checkpoint_id`` nullability (that nullability is a schema
        consequence of the persisted state kind, not its source; the
        persisted field is the only membership authority and is never
        resolved or rewritten). State kinds have NO registry (unlike
        the M24 checkpoint / M28 dataset / M30 tokenizer axes): the
        enum IS the contract, so an unsupported state-kind value is
        rejected at the API boundary with 422 (schema-level
        validation), while an unknown model raises FileNotFoundError
        exactly like the sibling groupings. Each evaluation appears
        EXACTLY ONCE (the authoritative listing holds each record
        exactly once). The result is the model's authoritative M4
        listing above (the exact engine parse + deterministic
        (created_at, eval_id) ASCENDING order) filtered by the
        persisted state kind; complete verbatim ``EvaluationRecord``
        payloads, no rewritten fields. A valid state kind with no
        evaluations for the model returns []. State kinds are
        model-scoped through the listing itself — a model never sees
        another model's evaluations. Read-only, never writes.
        """
        # the authoritative listing validates the model:
        # raises FileNotFoundError (404 at the API)
        return [r for r in self.list_evaluations(model_id)
                if r.state_kind == state_kind]

    def list_evaluations_for_truncated(
            self, model_id: str, truncated: bool
    ) -> list[EvaluationRecord]:
        """Immutable M4 evaluations of ONE model by persisted
        truncation status (M47).

        Membership comes from the persisted REQUIRED boolean ONLY:
        every ``EvaluationRecord`` carries ``truncated: bool`` (the
        engine's verbatim record of whether ``max_eval_tokens``
        stopped the evaluation before the split ended — False means
        the configured/permitted evaluation stream was consumed
        without the cap cutting it short), and an evaluation belongs
        to the request when that persisted boolean equals the
        requested value — matched VERBATIM, never recalculated and
        never derived from ``records_covered``, token counts, split
        length, the evaluation configuration, timestamps, durations,
        state kinds or any other field. The boolean is a closed
        two-value contract (no registry): True and False are the
        complete value space, so the two groups form a TRUE disjoint
        partition of the listing with no None case; an unsupported
        spelling is rejected at the API boundary with 422
        (schema-level validation), while an unknown model raises
        FileNotFoundError exactly like the sibling groupings. The
        result is the model's authoritative M4 listing (the exact
        engine parse + deterministic (created_at, eval_id) ASCENDING
        order) filtered by the persisted boolean; complete verbatim
        ``EvaluationRecord`` payloads. A model with no evaluations of
        one status returns []. Read-only, never writes.
        """
        # the authoritative listing validates the model:
        # raises FileNotFoundError (404 at the API)
        return [r for r in self.list_evaluations(model_id)
                if r.truncated == truncated]

    # ------------------------------------------------------------------ #
    # The run (preflight everything before creating any artifact)
    # ------------------------------------------------------------------ #

    def run(self, cfg: EvaluationConfig) -> EvaluationRecord:
        start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        spec = detect_hardware()

        # ---- state selection + verification BEFORE anything is loaded -----
        model_record = self._require_model(cfg.model_id)
        model_cfg = model_record.config
        if cfg.checkpoint_id is not None:
            state = self.training.verify_checkpoint(cfg.model_id, cfg.checkpoint_id)
            # verify_checkpoint guarantees content_hash(state) == manifest hash
            state_kind, state_hash = EvalStateKind.CHECKPOINT, content_hash(state)
        else:
            wpath = self.storage.weights_path(cfg.model_id)
            if not wpath.exists():
                raise FileNotFoundError(
                    f"model '{cfg.model_id}' has no weights yet — nothing to evaluate")
            state = self.storage.load_weights(cfg.model_id, device="cpu")
            # load_weights cross-checks the weights.sha256 sidecar and raises
            # RuntimeError (integrity) when the current state is corrupted
            state_kind, state_hash = EvalStateKind.CURRENT, content_hash(state)

        # ---- dataset / version / tokenizer resolution ---------------------
        try:
            meta = self.datasets.load_meta(cfg.dataset_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"dataset '{cfg.dataset_id}' not found") from None
        version = meta.latest_version if cfg.dataset_version is None else cfg.dataset_version

        try:
            self.tokenizers.load(cfg.tokenizer_id)  # existence only
            # deep integrity check of the whole dataset (incl. tokenized bins)
            report = self.datasets.verify(cfg.dataset_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"tokenizer '{cfg.tokenizer_id}' not found") from None
        if report["status"] != "ok":
            raise ValueError(
                f"dataset '{cfg.dataset_id}' failed integrity verification — "
                "evaluation is refused; fix or re-upload the dataset first")
        if not any(v["version"] == version for v in report["versions"]):
            raise FileNotFoundError(
                f"dataset '{cfg.dataset_id}' has no version {version}")

        info, paths = self.datasets.tokenized_artifact(
            cfg.dataset_id, version, cfg.tokenizer_id)
        split = cfg.split.value
        if split not in info.splits or info.splits[split].count < 1:
            raise ValueError(
                f"tokenized split '{split}' is empty (dataset too small for the "
                "deterministic 90/5/5 split); use a larger dataset")

        # exact record count of this split (M2 version manifest); only
        # meaningful for whole-split evaluations
        records_in_split: Optional[int] = None
        try:
            vmanifest = self.datasets.version_manifest(cfg.dataset_id, version)
            records_in_split = vmanifest.splits[split].count
        except Exception:  # pragma: no cover - verified datasets have this
            records_in_split = None

        # ---- window resolution + model compatibility ----------------------
        if cfg.max_seq_len is not None:
            window = cfg.max_seq_len
        else:
            window = model_cfg.context_length  # evaluate at the model's full context
        if window < 2:
            raise ValueError("max_seq_len must be >= 2")
        if window > model_cfg.context_length:
            raise ValueError(
                f"max_seq_len ({window}) exceeds the model's context_length "
                f"({model_cfg.context_length})")
        if window > model_cfg.rope_max_seq() and \
                model_cfg.position_encoding.value == "rope":
            raise ValueError(
                f"max_seq_len ({window}) exceeds the model's RoPE cache "
                f"({model_cfg.rope_max_seq()})")

        # persist the config with the resolved window so the record is
        # self-describing (max_seq_len=None in the request becomes the window)
        config_json = cfg.model_dump(mode="json")
        config_json["max_seq_len"] = window

        stream = open_bin(paths[split], info.dtype)
        max_id = int(stream.array.max())
        if max_id >= model_cfg.vocab_size:
            raise ValueError(
                f"token ids up to {max_id} exceed the model's vocab_size "
                f"({model_cfg.vocab_size}) — the tokenizer and model are "
                "incompatible (train a model with vocab_size >= tokenizer vocab)")
        n_seqs = stream.count // window
        if n_seqs < 1:
            raise ValueError(
                f"split '{split}' yields no complete window of {window} tokens — "
                "reduce max_seq_len or use a larger dataset")

        # ---- memory guard (eval has no optimizer state; batch is user-set) --
        budget = (spec.gpu_vram_bytes or spec.ram_bytes) * 0.6
        if estimate_eval_bytes(model_record.parameter_count, cfg.batch_size,
                               window, model_cfg.hidden_size,
                               model_cfg.n_layers) > budget:
            raise ValueError(
                "this evaluation cannot fit the detected memory "
                f"(budget ~{budget // 2**20} MiB); reduce batch_size or max_seq_len")

        # ---- evaluate ------------------------------------------------------
        device = "cuda" if spec.has_gpu else "cpu"
        model = build_transformer(model_cfg).module
        missing, unexpected = restore_state(model, state)
        if missing or unexpected:
            raise RuntimeError(
                f"evaluated state incompatible with config: {len(missing)} missing, "
                f"{len(unexpected)} unexpected")
        model.to(device)
        model.eval()

        metrics = self._evaluate(model, stream.array, stream.count, window,
                                 cfg.batch_size, cfg.max_eval_tokens, device)
        duration = time.monotonic() - start
        tokenized_sha = info.splits[split].sha256
        if not metrics["truncated"]:
            pass  # records_covered stays the exact split record count
        else:
            records_in_split = None  # capped: never infer record coverage

        record = EvaluationRecord(
            eval_id=uuid.uuid4().hex[:12],
            model_id=cfg.model_id,
            state_kind=state_kind,
            checkpoint_id=cfg.checkpoint_id,
            state_hash=state_hash,
            dataset_id=cfg.dataset_id,
            dataset_version=version,
            split=cfg.split,
            tokenizer_id=cfg.tokenizer_id,
            tokenized_bin_sha256=tokenized_sha,
            loss_nats=round(float(metrics["loss"]), 6),
            perplexity=round(float(metrics["perplexity"]), 6),
            token_count=int(metrics["token_count"]),
            records_covered=records_in_split,
            truncated=bool(metrics["truncated"]),
            config=config_json,
            seed=cfg.effective_seed(),
            result_hash="",  # filled below (needs the rounded metrics)
            hardware=spec.to_dict(),
            created_at=started_at,
            duration_seconds=round(duration, 3),
        )
        record = record.model_copy(update={
            "result_hash": self.result_hash(record)})
        self._persist(record)
        log.info("evaluation %s of model %s (%s): split %s v%s -> loss %.4f "
                 "ppl %.2f (%s tokens%s)", record.eval_id, cfg.model_id,
                 state_kind.value, split, version, record.loss_nats,
                 record.perplexity, record.token_count,
                 ", truncated" if record.truncated else "")
        return record

    def _require_model(self, model_id: str) -> ModelRecord:
        try:
            return self.storage.load_record(model_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"model '{model_id}' not found") from None

    # ------------------------------------------------------------------ #
    # Deterministic windowed evaluation (M3-compatible objective)
    # ------------------------------------------------------------------ #

    def _evaluate(self, model: torch.nn.Module, array: np.ndarray, n_tokens: int,
                  window: int, batch: int, max_eval_tokens: Optional[int],
                  device: str) -> dict[str, Any]:
        """Token-weighted mean causal-LM loss over a flat token stream.

        Mirrors M3 semantics exactly for whole-split evaluation: the stream is
        cut into ``window``-length sequences (incomplete tail dropped), batches
        are walked in order with the last group possibly partial, and each
        sequence contributes (window - 1) target tokens.

        A positive ``max_eval_tokens`` caps the number of evaluated *target*
        tokens: whole windows first, then — when the cap falls inside a window —
        exactly the remaining targets of that window are evaluated as a short
        window, so the token count hits the cap deterministically.
        """
        n_seqs = n_tokens // window
        usable = n_seqs * (window - 1)
        cap = usable if max_eval_tokens is None else min(max_eval_tokens, usable)
        truncated = cap < usable

        full_windows = cap // (window - 1)
        remainder = cap % (window - 1)  # target tokens of the final partial window

        total_loss = 0.0
        total_pairs = 0
        with torch.no_grad():
            for s in range(0, full_windows, batch):
                take = min(batch, full_windows - s)  # last group may be partial
                ids = TrainingEngine._ids_tensor(array, s * window,
                                                 take * window, window, device)
                logits, _ = model(ids)
                loss = TrainingEngine._lm_loss(logits, ids)
                pairs = take * (window - 1)
                total_loss += float(loss.item()) * pairs
                total_pairs += pairs
            if remainder > 0:
                # first `remainder` targets of the next window: the window's
                # tokens 0..remainder suffice (causal, position < remainder)
                start = full_windows * window
                mini_len = remainder + 1
                ids = TrainingEngine._ids_tensor(array, start, mini_len,
                                                 mini_len, device)
                logits, _ = model(ids)
                loss = TrainingEngine._lm_loss(logits, ids)
                total_loss += float(loss.item()) * remainder
                total_pairs += remainder

        loss = total_loss / total_pairs if total_pairs else float("nan")
        return {
            "loss": loss,
            "perplexity": float(np.exp(min(loss, 100.0))),
            "token_count": total_pairs,
            "truncated": truncated,
        }

    # ------------------------------------------------------------------ #
    # Deterministic result hash + persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def result_hash(record: EvaluationRecord) -> str:
        """Deterministic hash of an evaluation's inputs and results.

        Excludes eval_id, created_at, duration, hardware and filesystem paths,
        so two identical evaluations share a result_hash. Inputs: state
        identity (model, kind, checkpoint, canonical state hash), data
        identity (dataset/version/split/tokenizer + exact bin sha256),
        evaluation configuration (window/batch/cap/seed) and the results
        (loss/perplexity/token count/records/truncated).
        """
        payload = {
            "model_id": record.model_id,
            "state_kind": record.state_kind.value,
            "checkpoint_id": record.checkpoint_id,
            "state_hash": record.state_hash,
            "dataset_id": record.dataset_id,
            "dataset_version": record.dataset_version,
            "split": record.split.value,
            "tokenizer_id": record.tokenizer_id,
            "tokenized_bin_sha256": record.tokenized_bin_sha256,
            "config": record.config,
            "seed": record.seed,
            "loss_nats": record.loss_nats,
            "perplexity": record.perplexity,
            "token_count": record.token_count,
            "records_covered": record.records_covered,
            "truncated": record.truncated,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _persist(self, record: EvaluationRecord) -> None:
        """Write one immutable evaluation manifest (atomic; never rewritten)."""
        edir = self._eval_dir(record.model_id, record.eval_id)
        edir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(edir / EVAL_MANIFEST, record.model_dump(mode="json"))
