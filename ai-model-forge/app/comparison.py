"""Comparison engine: evidence-based A/B state comparison (Milestone 5).

Sits ABOVE M3 (checkpoints) and M4 (evaluation records). Answers one
question, honestly:

    given two states of a model, measured under IDENTICAL probe conditions,
    did the loss improve, regress or stay unchanged?

Rules honoured here (tested):
  * the probe is a single shared description: dataset/version/split/
    tokenizer/window/token-cap/batch/seed are identical for A and B — only
    the model state differs; deltas are meaningless otherwise
  * both states are verified with the existing M1/M3 mechanisms BEFORE
    anything is evaluated or persisted (corrupt or missing state -> no
    artifacts)
  * evaluations are never duplicated: an existing immutable M4 evaluation is
    reused when its complete identity matches (model, state kind + id +
    canonical state hash, resolved version, split, tokenizer, window, cap,
    batch, seed); otherwise exactly one new evaluation is created via the M4
    engine
  * a comparison persists exactly one immutable manifest under
    ``models/<id>/comparisons/comp-<id>/manifest.json`` referencing the two
    evaluation records; comparison history is append-only (every request is
    auditable — only *evaluations* are deduplicated, never comparisons)
  * verdicts are loss-only: |loss_B - loss_A| <= tolerance -> unchanged;
    below -tolerance -> improved; above +tolerance -> regressed. Perplexity
    is supporting information. "Improved/regressed" describes measured loss
    on ONE probe — never a universal quality judgment
  * nothing is trained, rolled back or selected automatically

Two engine entry points:
  * ``run(request)`` — canonical flow used by the API (states of one model,
    one shared probe)
  * ``compare_records(a, b, tolerance)`` — guard-level flow: builds a
    comparison over two existing immutable evaluation records, refusing
    mismatched probes or incompatible model configurations (used by ``run``
    after both evaluations resolve, and exercised directly by tests)
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config as forge_cfg
from .dataset import DatasetEngine
from .evaluation import EvaluationEngine
from .model_builder import content_hash
from .schemas import (
    ComparisonRecord,
    ComparisonRequest,
    ComparisonSide,
    ComparisonState,
    ComparisonVerdict,
    EvalStateKind,
    EvaluationConfig,
    EvaluationRecord,
    EvaluationSplit,
    ModelRecord,
)
from .storage import Storage, atomic_write_json, read_json
from .tokenizer import TokenizerEngine
from .training import TrainingEngine

log = forge_cfg.get_logger("comparison")

COMPARISON_MANIFEST = "manifest.json"
COMPARISONS_DIR = "comparisons"


class ComparisonEngine:
    """Comparison orchestration for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.datasets = DatasetEngine(storage)
        self.tokenizers = TokenizerEngine(storage)   # registry validation (M31)
        self.training = TrainingEngine(storage)
        self.evaluation = EvaluationEngine(storage)

    # ------------------------------------------------------------------ #
    # Paths / registry helpers
    # ------------------------------------------------------------------ #

    def _comp_root(self, model_id: str) -> Path:
        return self.storage.model_dir(model_id) / COMPARISONS_DIR

    def _comp_dir(self, model_id: str, comparison_id: str) -> Path:
        return self._comp_root(model_id) / f"comp-{comparison_id}"

    def _model_exists(self, model_id: str) -> bool:
        return (self.storage.model_dir(model_id) / "manifest.json").exists()

    def list_comparisons(self, model_id: str) -> list[ComparisonRecord]:
        """Immutable comparison history (append-only, deterministic order).

        Raises FileNotFoundError for an unknown model; returns [] when the
        model has no comparisons yet.
        """
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        root = self._comp_root(model_id)
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith("comp-"):
                continue
            mpath = d / COMPARISON_MANIFEST
            if mpath.exists():
                try:
                    records.append(ComparisonRecord(**read_json(mpath)))
                except Exception:
                    log.warning("unreadable comparison manifest %s", mpath)
        records.sort(key=lambda r: (r.created_at, r.comparison_id))
        return records

    def get_comparison(self, model_id: str, comparison_id: str) -> ComparisonRecord:
        """One persisted immutable comparison record (never mutates it)."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        path = self._comp_dir(model_id, comparison_id) / COMPARISON_MANIFEST
        if not path.exists():
            raise FileNotFoundError(
                f"comparison '{comparison_id}' not found for model '{model_id}'")
        return ComparisonRecord(**read_json(path))

    # ------------------------------------------------------------------ #
    # M26: per-checkpoint access (read-only)
    # ------------------------------------------------------------------ #

    def list_comparisons_for_checkpoint(
            self, model_id: str, checkpoint_id: str) -> list[ComparisonRecord]:
        """Immutable M5 comparisons involving ONE checkpoint of ONE model
        (M26).

        A comparison involves the checkpoint when EITHER persisted side
        records ``state_kind == "checkpoint"`` with that
        ``checkpoint_id`` (current-state sides keep ``checkpoint_id``
        None and never match — nothing is inferred from hashes,
        timestamps or filenames). Resolution: unknown model or
        unregistered checkpoint -> FileNotFoundError; ownership is
        validated through the model's M3 checkpoint registry
        (``TrainingEngine.get_checkpoint``) — a checkpoint id belonging
        to another model is not registered under this model and raises
        FileNotFoundError, exactly like an unknown one. The result is
        the model's authoritative M5 listing above (the exact engine
        parse + deterministic (created_at, comparison_id) ASCENDING
        order) filtered by the persisted side states; because the
        listing holds each record exactly once, a comparison matching on
        BOTH sides (A = B = checkpoint) appears exactly once — the
        response is a unique list of comparison identities. A valid
        checkpoint with no matching comparisons returns []. Read-only,
        never writes.
        """
        # existence + ownership: raises FileNotFoundError (404 at the API)
        self.training.get_checkpoint(model_id, checkpoint_id)

        def involves(rec: ComparisonRecord) -> bool:
            for side in (rec.state_a, rec.state_b):
                if (side.state_kind == EvalStateKind.CHECKPOINT
                        and side.checkpoint_id == checkpoint_id):
                    return True
            return False

        return [r for r in self.list_comparisons(model_id) if involves(r)]

    def list_comparisons_for_state_kind(
            self, model_id: str, state_kind: EvalStateKind
    ) -> list[ComparisonRecord]:
        """Immutable M5 comparisons of ONE model involving ONE kind of
        model state on EITHER side (M44; the enum-contract sibling of
        M38, with M26's either-side semantics).

        A comparison involves the requested state kind when EITHER
        persisted side records ``state_kind == state_kind`` — every
        ``ComparisonRecord`` persists BOTH sides (``state_a`` /
        ``state_b``, each carrying the schema enum
        ``state_kind: EvalStateKind``: current = the model's published
        weights, checkpoint = an immutable stored checkpoint; matched
        VERBATIM — never inferred from checkpoint ids alone, state
        hashes, losses, evaluation results or filenames, and never
        resolved or rewritten). Because the listing holds each record
        exactly once, a comparison matching on BOTH sides
        (A = B = checkpoint) appears EXACTLY ONCE. State kinds have NO
        registry (the enum IS the contract, exactly like M38): an
        unsupported state-kind value is rejected at the API boundary
        with 422 and never reaches this method, while an unknown model
        raises FileNotFoundError exactly like the sibling groupings
        (the authoritative listing validates the model). The result is
        the model's authoritative M5 listing above (the exact engine
        parse + deterministic (created_at, comparison_id) ASCENDING
        order) filtered by the persisted side state kinds; complete
        verbatim ``ComparisonRecord`` payloads (both sides, losses,
        verdict verbatim — nothing recalculated, no comparison
        re-executed). A valid state kind with no matching comparisons
        returns []. State kinds are model-scoped through the listing
        itself — a model never sees another model's comparisons.
        Read-only, never writes.
        """
        # the authoritative listing validates the model:
        # raises FileNotFoundError (404 at the API); the state kind is
        # matched VERBATIM on EITHER persisted side — never inferred
        # from checkpoint ids alone
        def involves(rec: ComparisonRecord) -> bool:
            for side in (rec.state_a, rec.state_b):
                if side.state_kind == state_kind:
                    return True
            return False

        return [r for r in self.list_comparisons(model_id) if involves(r)]

    def list_comparisons_for_seed(
            self, model_id: str, seed: int) -> list[ComparisonRecord]:
        """Immutable M5 comparisons of ONE model by persisted effective
        seed (M49).

        Membership comes from the persisted REQUIRED integer ONLY:
        every ``ComparisonRecord`` carries ``seed: int`` (the seed of
        the identical-probe A/B measurement, persisted verbatim at
        run time), and a comparison belongs to the request when that
        persisted value equals the requested integer — matched
        VERBATIM, never recalculated, never normalized and never
        derived from the comparison configuration, either side's
        evaluation, state payloads, verdicts, loss deltas, ids,
        timestamps or any other field. The seed is an OPEN integer
        value axis (no registry, no enum): any integer is type-valid,
        so an unmatched seed on a valid model returns [] (a natural
        valid-empty), a non-integer spelling is rejected at the API
        boundary with 422 (schema-level validation), and an unknown
        model raises FileNotFoundError exactly like the sibling
        groupings. Each comparison appears EXACTLY ONCE; the per-seed
        groups form a TRUE disjoint partition of the listing with no
        None case (the field is required). The result is the model's
        authoritative M5 listing (the exact engine parse +
        deterministic (created_at, comparison_id) ASCENDING order)
        filtered by the persisted seed; complete verbatim
        ``ComparisonRecord`` payloads. Read-only, never writes.
        """
        # the authoritative listing validates the model:
        # raises FileNotFoundError (404 at the API)
        return [r for r in self.list_comparisons(model_id)
                if r.seed == seed]

    def list_comparisons_for_dataset(
            self, model_id: str, dataset_id: str) -> list[ComparisonRecord]:
        """Immutable M5 comparisons of ONE model over ONE dataset (M29).

        Membership comes from the persisted shared-probe dataset
        identity ONLY: a comparison is valid only when BOTH sides
        measure the SAME probe, so ``ComparisonRecord`` persists
        exactly ONE top-level ``dataset_id: str`` +
        ``dataset_version: int`` pair (M5 refuses cross-probe requests
        — different dataset/version/split/tokenizer/window/seed — with
        422 before any artifact exists; per-side dataset identities
        cannot occur by construction). A comparison belongs to the
        request when its persisted ``dataset_id`` equals the requested
        id — never filenames, dataset directory names, checkpoint ids,
        state/result hashes, timestamps or eval ids. Every version of
        the dataset is returned, each with its persisted
        ``dataset_version`` VERBATIM inside the record (versions are
        neither collapsed, resolved to the latest, aliased nor
        rewritten). Because the comparison is the unit of grouping and
        the listing above holds each record exactly once, a comparison
        whose two sides measure the requested dataset (the shared
        probe — true for EVERY matching record, including
        same-checkpoint A=B) appears EXACTLY ONCE: dedup by comparison
        identity, never by dataset id, hash, timestamp, side equality
        or path. Resolution: unknown model or unknown dataset ->
        FileNotFoundError; the dataset is validated through the M2
        registry (``DatasetEngine.load_meta`` — the same registry call
        M4's run preflight and M28's by-dataset grouping use; datasets
        are GLOBAL, so the model scoping comes from the model's own M5
        listing — a model never sees another model's comparisons). The
        result keeps the authoritative M5 (created_at, comparison_id)
        ASCENDING order. A valid dataset with no comparisons for the
        model returns []. Read-only, never writes.
        """
        # existence: raises FileNotFoundError (404 at the API)
        self.datasets.load_meta(dataset_id)
        return [r for r in self.list_comparisons(model_id)
                if r.dataset_id == dataset_id]

    def list_comparisons_for_tokenizer(
            self, model_id: str, tokenizer_id: str) -> list[ComparisonRecord]:
        """Immutable M5 comparisons of ONE model measured with ONE
        tokenizer (M31).

        Membership comes from the persisted shared-probe tokenizer
        identity ONLY: a comparison is valid only when BOTH sides
        measure the SAME probe, so ``ComparisonRecord`` persists
        exactly ONE top-level ``tokenizer_id: str`` (per-side
        tokenizer identities cannot occur by construction; M5 refuses
        cross-probe requests — different dataset/version/split/
        tokenizer/window/seed — with 422 before any artifact exists).
        A comparison belongs to the request when its persisted
        ``tokenizer_id`` equals the requested id, matched VERBATIM —
        never filenames, checkpoint ids, dataset identities, hashes or
        the tokenizer currently registered, and never a
        latest-tokenizer substitution (tokenizer ids are opaque ids;
        M2/M5 have no tokenizer versioning and none is introduced
        here). Because the comparison (not the side) is the unit of
        grouping and the listing above holds each record exactly once,
        every matching comparison appears EXACTLY ONCE — including
        same-checkpoint A=B records (both sides measured the requested
        tokenizer by the shared-probe rule). Resolution: unknown model
        or unknown tokenizer -> FileNotFoundError; the tokenizer is
        validated through the existing registry
        (``TokenizerEngine.load`` — the same registry getter
        ``GET /tokenizers/{id}`` exposes; tokenizers are GLOBAL, so
        the model scoping comes from the model's own M5 listing — a
        model never sees another model's comparisons). The result
        keeps the authoritative M5 (created_at, comparison_id)
        ASCENDING order. A valid tokenizer with no comparisons for
        the model returns []. Read-only, never writes.
        """
        # existence: raises FileNotFoundError (404 at the API)
        self.tokenizers.load(tokenizer_id)
        return [r for r in self.list_comparisons(model_id)
                if r.tokenizer_id == tokenizer_id]

    def list_comparisons_for_split(
            self, model_id: str, split: EvaluationSplit
    ) -> list[ComparisonRecord]:
        """Immutable M5 comparisons of ONE model measured on ONE
        dataset split (M37).

        Membership comes from the persisted shared-probe split
        identity ONLY: every ``ComparisonRecord`` carries a top-level
        ``split: EvaluationSplit`` (the schema enum
        train/validation/test — one half of the shared-probe contract:
        a comparison is valid only when BOTH sides measure the SAME
        dataset/version/**split**/tokenizer/window/cap/batch/seed, so
        the split is a property of the comparison itself, never of a
        side), and a comparison belongs to the request when its
        persisted ``split`` equals the requested value — never
        filenames, checkpoint ids, dataset identities, nested
        evaluation records, hashes or timestamps, and never resolved
        or rewritten into another value. Splits have NO registry
        (unlike the M26 checkpoint / M29 dataset / M31 tokenizer
        axes): the enum IS the contract, so an unsupported split value
        is rejected at the API boundary with 422 (schema-level
        validation — it never even reaches this method), while an
        unknown model raises FileNotFoundError exactly like the
        sibling groupings. Each comparison appears EXACTLY ONCE
        (including same-checkpoint A=B records — the split is the
        probe's, not a side's). The result is the model's
        authoritative M5 listing above (the exact engine parse +
        deterministic (created_at, comparison_id) ASCENDING order)
        filtered by the persisted split; complete verbatim
        ``ComparisonRecord`` payloads, no rewritten fields. A valid
        split with no comparisons for the model returns []. Splits
        are model-scoped through the listing itself — a model never
        sees another model's comparisons. Read-only, never writes.
        """
        # existence: raises FileNotFoundError (404 at the API); the
        # split itself needs NO registry lookup — the enum IS the
        # contract (unsupported values are rejected with 422 at the
        # API boundary before this method runs)
        return [r for r in self.list_comparisons(model_id)
                if r.split == split]

    def list_comparisons_for_verdict(
            self, model_id: str, verdict: ComparisonVerdict
    ) -> list[ComparisonRecord]:
        """Immutable M5 comparisons of ONE model with ONE verdict
        (M39).

        Membership comes from the persisted verdict identity ONLY:
        every ``ComparisonRecord`` carries a top-level ``verdict:
        ComparisonVerdict`` (the schema enum improved/regressed/
        unchanged — the immutable loss-only judgment persisted at run
        time by the M5 comparison flow: |loss_B - loss_A| <= tolerance
        -> unchanged, below -> improved, above -> regressed; it
        describes measured loss on ONE probe, never a universal
        quality judgment), and a comparison belongs to the request
        when its persisted ``verdict`` equals the requested value —
        never recalculated from loss deltas, per-side losses,
        tolerances, checkpoint ids or hashes, and never resolved or
        rewritten into another value (the persisted verdict is the
        ONLY authority; this method NEVER calls the comparison
        execution engine or reruns evaluations). Verdicts have NO
        registry (unlike the M26 checkpoint / M29 dataset / M31
        tokenizer axes): the enum IS the contract, so an unsupported
        verdict value is rejected at the API boundary with 422
        (schema-level validation — it never even reaches this
        method), while an unknown model raises FileNotFoundError
        exactly like the sibling groupings. Each comparison appears
        EXACTLY ONCE (the authoritative listing holds each record
        exactly once). The result is the model's authoritative M5
        listing above (the exact engine parse + deterministic
        (created_at, comparison_id) ASCENDING order) filtered by the
        persisted verdict; complete verbatim ``ComparisonRecord``
        payloads, no rewritten fields. A valid verdict with no
        comparisons for the model returns []. Verdicts are
        model-scoped through the listing itself — a model never sees
        another model's comparisons. Read-only, never writes.
        """
        # existence: raises FileNotFoundError (404 at the API); the
        # verdict itself needs NO registry lookup and is NEVER
        # recalculated — the persisted field is filtered verbatim
        return [r for r in self.list_comparisons(model_id)
                if r.verdict == verdict]

    # ------------------------------------------------------------------ #
    # Canonical flow (API): one model, one shared probe, two states
    # ------------------------------------------------------------------ #

    def run(self, request: ComparisonRequest) -> ComparisonRecord:
        start = time.monotonic()
        model_record = self._require_model(request.model_id)

        # resolve the dataset version exactly once (identity for both sides)
        try:
            meta = self.datasets.load_meta(request.dataset_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"dataset '{request.dataset_id}' not found") from None
        version = meta.latest_version if request.dataset_version is None \
            else request.dataset_version

        # verify BOTH states before anything is evaluated or persisted
        hash_a = self._verified_state_hash(request.model_id, request.state_a)
        hash_b = self._verified_state_hash(request.model_id, request.state_b)

        # shared probe window resolved the same way M4 resolves it
        window = request.max_seq_len if request.max_seq_len is not None \
            else model_record.config.context_length

        # single directory scan feeds both exact-match lookups; a freshly
        # created evaluation is appended so an A==B pair reuses one record
        candidates = self.evaluation.list_evaluations(request.model_id)

        eval_a = self._resolve_evaluation(
            model_id=request.model_id, state=request.state_a,
            state_hash=hash_a, dataset_id=request.dataset_id, version=version,
            split=request.split.value, tokenizer_id=request.tokenizer_id,
            max_eval_tokens=request.max_eval_tokens, batch_size=request.batch_size,
            window=window, seed=request.effective_seed(), candidates=candidates)
        candidates.append(eval_a)
        eval_b = self._resolve_evaluation(
            model_id=request.model_id, state=request.state_b,
            state_hash=hash_b, dataset_id=request.dataset_id, version=version,
            split=request.split.value, tokenizer_id=request.tokenizer_id,
            max_eval_tokens=request.max_eval_tokens, batch_size=request.batch_size,
            window=window, seed=request.effective_seed(), candidates=candidates)

        record = self.compare_records(
            eval_a, eval_b, tolerance=request.tolerance, started_at=start)
        log.info("comparison %s on model %s: %s vs %s -> %s (delta loss %.6f)",
                 record.comparison_id, request.model_id,
                 request.state_a.state_kind.value, request.state_b.state_kind.value,
                 record.verdict.value, record.delta_loss_nats)
        return record

    def _require_model(self, model_id: str) -> ModelRecord:
        try:
            return self.storage.load_record(model_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"model '{model_id}' not found") from None

    # Thin public aliases (used by the M6 gate engine, which must reuse this
    # verification/resolution logic instead of duplicating it).
    def verified_state_hash(self, model_id: str,
                            state: ComparisonState) -> str:
        """Verify a state (current weights or checkpoint); return its canonical hash."""
        return self._verified_state_hash(model_id, state)

    def resolve_evaluation(self, model_id: str, state: ComparisonState,
                           state_hash: str, dataset_id: str, version: int,
                           split: str, tokenizer_id: str,
                           max_eval_tokens: Optional[int], batch_size: int,
                           window: int, seed: int,
                           candidates: list[EvaluationRecord]) -> EvaluationRecord:
        """Exact-reuse lookup over immutable evaluations, else run M4 once."""
        return self._resolve_evaluation(
            model_id, state, state_hash, dataset_id, version, split,
            tokenizer_id, max_eval_tokens, batch_size, window, seed, candidates)

    def find_exact_comparison(self, model_id: str, state_a_hash: str,
                              state_b_hash: str, dataset_id: str, version: int,
                              split: str, tokenizer_id: str,
                              max_eval_tokens: Optional[int], batch_size: int,
                              window: int, seed: int,
                              tolerance: float) -> Optional[ComparisonRecord]:
        """Reuse an existing comparison by full identity (state hashes + probe
        + tolerance), so equivalent evidence is never duplicated. Shared by the
        M6 gate engine and M7 compare stages."""
        for rec in self.list_comparisons(model_id):
            if rec.state_a.state_hash != state_a_hash:
                continue
            if rec.state_b.state_hash != state_b_hash:
                continue
            if rec.dataset_id != dataset_id or rec.dataset_version != version:
                continue
            if rec.split.value != split or rec.tokenizer_id != tokenizer_id:
                continue
            if rec.max_eval_tokens != max_eval_tokens:
                continue
            if rec.batch_size != batch_size or rec.max_seq_len != window:
                continue
            if rec.seed != seed or abs(rec.tolerance - tolerance) > 1e-12:
                continue
            return rec
        return None

    def _verified_state_hash(self, model_id: str,
                             state: ComparisonState) -> str:
        """Verify one requested state; return its canonical content hash.

        Missing state -> FileNotFoundError; corrupt state -> RuntimeError
        (integrity) — both raised before any evaluation/persistence.
        """
        if state.state_kind == EvalStateKind.CURRENT:
            wpath = self.storage.weights_path(model_id)
            if not wpath.exists():
                raise FileNotFoundError(
                    f"model '{model_id}' has no weights yet — nothing to compare")
            weights = self.storage.load_weights(model_id, device="cpu")
            return content_hash(weights)  # load_weights checks the sidecar
        if state.checkpoint_id:
            ckpt = self.training.get_checkpoint(model_id, state.checkpoint_id)
            # full verification: load weights + canonical hash vs manifest
            self.training.verify_checkpoint(model_id, state.checkpoint_id)
            return ckpt.weights_sha256
        raise ValueError("state_kind='checkpoint' requires a checkpoint_id")

    # ------------------------------------------------------------------ #
    # Evaluation resolution: reuse exact immutable records, else create one
    # ------------------------------------------------------------------ #

    def _resolve_evaluation(self, model_id: str, state: ComparisonState,
                            state_hash: str, dataset_id: str, version: int,
                            split: str, tokenizer_id: str,
                            max_eval_tokens: Optional[int], batch_size: int,
                            window: int, seed: int,
                            candidates: list[EvaluationRecord]) -> EvaluationRecord:
        """Reuse an existing exact evaluation or run M4 to create one.

        The identity covers every condition that changes the measurement:
        state (kind/id/canonical hash), dataset/version/split/tokenizer,
        window, token cap, batch size and seed. Anything missing -> mismatch.
        """
        existing = self._find_exact_evaluation(
            candidates=candidates, model_id=model_id,
            state_kind=state.state_kind, checkpoint_id=state.checkpoint_id,
            state_hash=state_hash, dataset_id=dataset_id, version=version,
            split=split, tokenizer_id=tokenizer_id,
            max_eval_tokens=max_eval_tokens, batch_size=batch_size,
            window=window, seed=seed)
        if existing is not None:
            return existing
        cfg = EvaluationConfig(
            model_id=model_id, checkpoint_id=state.checkpoint_id,
            dataset_id=dataset_id, dataset_version=version, split=split,
            tokenizer_id=tokenizer_id, max_eval_tokens=max_eval_tokens,
            batch_size=batch_size, max_seq_len=window, seed=seed)
        return self.evaluation.run(cfg)

    @staticmethod
    def _find_exact_evaluation(candidates: list[EvaluationRecord], model_id: str,
                               state_kind: EvalStateKind,
                               checkpoint_id: Optional[str], state_hash: str,
                               dataset_id: str, version: int, split: str,
                               tokenizer_id: str, max_eval_tokens: Optional[int],
                               batch_size: int, window: int,
                               seed: int) -> Optional[EvaluationRecord]:
        """Exact-match lookup over immutable evaluation manifests.

        A stale evaluation is never reused for a different state or probe:
        every identity field must match, including the canonical state hash.
        """
        for rec in candidates:
            if rec.model_id != model_id or rec.state_kind != state_kind:
                continue
            if rec.checkpoint_id != checkpoint_id or rec.state_hash != state_hash:
                continue
            if rec.dataset_id != dataset_id or rec.dataset_version != version:
                continue
            if rec.split.value != split or rec.tokenizer_id != tokenizer_id:
                continue
            cfg = rec.config
            if cfg.get("max_eval_tokens") != max_eval_tokens:
                continue
            if cfg.get("batch_size") != batch_size:
                continue
            if cfg.get("max_seq_len") != window:
                continue
            if rec.seed != seed:
                continue
            return rec
        return None

    # ------------------------------------------------------------------ #
    # Guard-level comparison over two immutable evaluation records
    # ------------------------------------------------------------------ #

    def compare_records(self, a: EvaluationRecord, b: EvaluationRecord,
                        tolerance: float = 1e-4,
                        started_at: Optional[float] = None) -> ComparisonRecord:
        """Compare two verified evaluation records; persist one comparison.

        Refuses (ValueError, nothing persisted) when the records describe
        different probe conditions or incompatible model configurations.
        """
        started_at = started_at if started_at is not None else time.monotonic()
        config_hash = self._guard_architecture(a, b)
        self._guard_probe(a, b)

        delta_loss = b.loss_nats - a.loss_nats
        delta_ppl = b.perplexity - a.perplexity
        verdict = ComparisonEngine._verdict(delta_loss, tolerance)
        comparison_id = uuid.uuid4().hex[:12]

        record = ComparisonRecord(
            comparison_id=comparison_id,
            model_id=a.model_id,
            config_hash=config_hash,
            state_a=self._side(a),
            state_b=self._side(b),
            dataset_id=a.dataset_id,
            dataset_version=a.dataset_version,
            split=a.split,
            tokenizer_id=a.tokenizer_id,
            max_eval_tokens=a.config.get("max_eval_tokens"),
            batch_size=int(a.config.get("batch_size", 1)),
            max_seq_len=int(a.config.get("max_seq_len", 0)),
            seed=a.seed,
            loss_a=round(float(a.loss_nats), 6),
            loss_b=round(float(b.loss_nats), 6),
            perplexity_a=round(float(a.perplexity), 6),
            perplexity_b=round(float(b.perplexity), 6),
            delta_loss_nats=round(float(delta_loss), 6),
            delta_perplexity=round(float(delta_ppl), 6),
            tolerance=tolerance,
            verdict=verdict,
            result_hash="",  # filled below from the rounded record
            created_at=datetime.now(timezone.utc),
            duration_seconds=round(time.monotonic() - started_at, 3),
        )
        record = record.model_copy(update={"result_hash": self.result_hash(record)})
        self._persist(record)
        return record

    @staticmethod
    def _side(rec: EvaluationRecord) -> ComparisonSide:
        return ComparisonSide(
            state_kind=rec.state_kind,
            checkpoint_id=rec.checkpoint_id,
            state_hash=rec.state_hash,
            evaluation_id=rec.eval_id,
            evaluation_result_hash=rec.result_hash,
            loss_nats=rec.loss_nats,
            perplexity=rec.perplexity,
            token_count=rec.token_count,
        )

    def _guard_architecture(self, a: EvaluationRecord,
                            b: EvaluationRecord) -> str:
        """States must belong to models with identical architecture configs."""
        a_model = self._require_model(a.model_id)
        if a.model_id == b.model_id:
            return a_model.config_hash
        b_model = self._require_model(b.model_id)
        if a_model.config != b_model.config:
            raise ValueError(
                "comparison refused: the two states belong to models with "
                "incompatible configurations (loss deltas would be "
                "meaningless) — compare states of one model or of models "
                "with identical architecture")
        return a_model.config_hash

    def _guard_probe(self, a: EvaluationRecord, b: EvaluationRecord) -> None:
        """Both evaluations must share ONE identical probe description."""
        mismatches = []
        for field in ("dataset_id", "dataset_version", "split",
                      "tokenizer_id", "seed"):
            if getattr(a, field) != getattr(b, field):
                mismatches.append(field)
        for field in ("max_eval_tokens", "batch_size", "max_seq_len"):
            if a.config.get(field) != b.config.get(field):
                mismatches.append(field)
        if mismatches:
            raise ValueError(
                "comparison refused: evaluations A and B were measured under "
                f"different probe conditions ({', '.join(mismatches)}) — "
                "comparisons require an identical probe for both states")

    @staticmethod
    def _verdict(delta_loss: float, tolerance: float) -> ComparisonVerdict:
        if abs(delta_loss) <= tolerance:
            return ComparisonVerdict.UNCHANGED
        if delta_loss < 0.0:
            return ComparisonVerdict.IMPROVED
        return ComparisonVerdict.REGRESSED

    # ------------------------------------------------------------------ #
    # Deterministic result hash + persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def result_hash(record: ComparisonRecord) -> str:
        """Deterministic hash over immutable inputs/results only.

        Excludes comparison_id, created_at, duration and paths — identical
        comparisons against unchanged immutable states reproduce it exactly.
        """
        payload = {
            "model_id": record.model_id,
            "config_hash": record.config_hash,
            "state_a": {
                "state_kind": record.state_a.state_kind.value,
                "checkpoint_id": record.state_a.checkpoint_id,
                "state_hash": record.state_a.state_hash,
                "evaluation_result_hash": record.state_a.evaluation_result_hash,
            },
            "state_b": {
                "state_kind": record.state_b.state_kind.value,
                "checkpoint_id": record.state_b.checkpoint_id,
                "state_hash": record.state_b.state_hash,
                "evaluation_result_hash": record.state_b.evaluation_result_hash,
            },
            "dataset_id": record.dataset_id,
            "dataset_version": record.dataset_version,
            "split": record.split.value,
            "tokenizer_id": record.tokenizer_id,
            "max_eval_tokens": record.max_eval_tokens,
            "batch_size": record.batch_size,
            "max_seq_len": record.max_seq_len,
            "seed": record.seed,
            "tolerance": record.tolerance,
            "loss_a": record.loss_a,
            "loss_b": record.loss_b,
            "perplexity_a": record.perplexity_a,
            "perplexity_b": record.perplexity_b,
            "delta_loss_nats": record.delta_loss_nats,
            "delta_perplexity": record.delta_perplexity,
            "verdict": record.verdict.value,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _persist(self, record: ComparisonRecord) -> None:
        """Write one immutable comparison manifest (atomic; never rewritten)."""
        cdir = self._comp_dir(record.model_id, record.comparison_id)
        cdir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(cdir / COMPARISON_MANIFEST,
                          record.model_dump(mode="json"))
