"""Training engine: continued pretraining / SFT on the user's own transformer.

This module is the first real training layer of the forge. Honest scope of
this milestone (see the report + README):

  * both methods run standard causal LM training over the Milestone-2
    tokenized streams (``v<N>/tokenized/<tokenizer_id>/*.bin``); there are NO
    chat templates, role masks, preference labels or method-specific losses
  * sequences are packed by taking the token stream in file order and
    cutting it into ``max_seq_len`` chunks; the incomplete tail is DROPPED
    (documented deterministic policy — no cross-document packing yet, and
    chunks may straddle document boundaries)
  * batch order is fixed (no shuffling) -> a fixed seed yields a fully
    reproducible run on a fixed backend
  * training is synchronous: one run = one call

Storage rules honoured here:
  * checkpoints are immutable once written (``checkpoints/<id>/manifest.json``
    + ``weights.pt``); a later run creates new checkpoints, never rewrites
  * before a checkpoint weight file is written its content hash is computed;
    if an identical artifact already exists for this model, the new file is a
    hard link to it — byte-identical states are never stored twice
  * the model's ``weights.pt`` is replaced only AFTER the run completes
    (preflight failures or crashes never modify it)
  * lineage: every checkpoint records run_id, method, dataset/version,
    metrics and its parent checkpoint; the model manifest records provenance
"""
from __future__ import annotations

import math
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import config as forge_cfg
from .dataset import DatasetEngine
from .hardware import detect_hardware
from .model_builder import build_transformer, content_hash, restore_state
from .schemas import (
    CheckpointDecision,
    CheckpointRecord,
    CheckpointSelection,
    ModelRecord,
    RunProvenance,
    TrainingConfig,
    TrainingReport,
)
from .storage import Storage, atomic_write_json, read_json

log = forge_cfg.get_logger("training")

CKPT_MANIFEST = "manifest.json"
CKPT_WEIGHTS = "weights.pt"
CHECKPOINTS_DIR = "checkpoints"
_OVERHEAD_BYTES = 32 * 2**20  # conservative process overhead in the estimate


# =========================================================================== #
# Memory estimation (conservative, documented) and LR schedules (pure, tested)
# =========================================================================== #


def estimate_training_bytes(param_count: int, batch: int, seq_len: int,
                            hidden: int, n_layers: int, precision_bytes: int = 4) -> int:
    """Rough peak memory: process overhead + optimizer/params + activations.

    fp32 AdamW state: weights+grad+momentum+variance = 16 B/param. Activation
    estimate is intentionally generous (per-layer hidden states + logits).
    """
    activations = batch * seq_len * hidden * (n_layers + 2) * precision_bytes * 8
    return _OVERHEAD_BYTES + param_count * 16 + activations


def fit_batch_size(requested: int, param_count: int, seq_len: int, hidden: int,
                   n_layers: int, budget_bytes: int) -> tuple[int, bool]:
    """Halve the batch until it fits the budget; never raises below 1."""
    batch = max(1, int(requested))
    while batch > 1 and estimate_training_bytes(param_count, batch, seq_len, hidden, n_layers) > budget_bytes:
        batch //= 2
    return batch, batch != requested


def lr_value(step: int, total: int, warmup: int, base: float, schedule: str) -> float:
    """Learning rate at a 0-based optimizer step under a warmup + schedule.

    total == steps in the decay phase; when warmup >= total the schedule stays
    at the peak (decay would have no steps left).
    """
    if total < 1:
        return base
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    # decay phase spans steps warmup..total-1: progress 0 at the warmup step,
    # 1 at the very last step (lr reaches 0 for linear/cosine)
    decay_total = max(1, total - 1 - warmup)
    progress = min(1.0, (step - warmup) / decay_total)
    if schedule == "constant":
        factor = 1.0
    elif schedule == "linear":
        factor = 1.0 - progress
    elif schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:  # pragma: no cover - enum guards this
        raise ValueError(f"unknown lr_schedule '{schedule}'")
    return base * factor


# =========================================================================== #
# Token-stream loader (memory-mapped; zero copies of the bins)
# =========================================================================== #


@dataclass
class TokenStream:
    array: np.ndarray  # memmap view, read-only
    count: int

    def __getitem__(self, item: slice) -> np.ndarray:
        return self.array[item]


def open_bin(path: Path, dtype_name: str) -> TokenStream:
    dtype = np.uint32 if dtype_name == "uint32" else np.uint16
    arr = np.memmap(path, dtype=dtype, mode="r")
    if arr.size == 0:
        raise ValueError(f"token stream is empty: {path}")
    return TokenStream(arr, int(arr.size))


# =========================================================================== #
# Training engine
# =========================================================================== #


class TrainingEngine:
    """Everything training-related for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.datasets = DatasetEngine(storage)

    # ------------------------------------------------------------------ #
    # Paths / registry helpers
    # ------------------------------------------------------------------ #

    def _ckpt_root(self, model_id: str) -> Path:
        return self.storage.model_dir(model_id) / CHECKPOINTS_DIR

    def _ckpt_dir(self, model_id: str, ckpt_id: str) -> Path:
        return self._ckpt_root(model_id) / ckpt_id

    def list_checkpoints(self, model_id: str) -> list[CheckpointRecord]:
        if not (self.storage.model_dir(model_id) / "manifest.json").exists():
            raise FileNotFoundError(f"model '{model_id}' not found")
        root = self._ckpt_root(model_id)
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            mpath = d / CKPT_MANIFEST
            if mpath.exists():
                try:
                    records.append(CheckpointRecord(**read_json(mpath)))
                except Exception:
                    log.warning("unreadable checkpoint manifest %s", mpath)
        records.sort(key=lambda r: (r.step, r.created_at))
        return records

    def get_checkpoint(self, model_id: str, ckpt_id: str) -> CheckpointRecord:
        path = self._ckpt_dir(model_id, ckpt_id) / CKPT_MANIFEST
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint '{ckpt_id}' not found for model '{model_id}'")
        return CheckpointRecord(**read_json(path))

    def list_checkpoints_for_run(self, model_id: str, run_id: str
                                 ) -> list[CheckpointRecord]:
        """Immutable checkpoints of ONE training run (M46 read-only access).

        Ownership: the run must be registered in the model's OWN
        manifest training_provenance (RunProvenance.run_id) — an
        unknown model, an unknown run or a run id that belongs to
        another model raises FileNotFoundError (404 at the API); run
        membership is NEVER inferred from checkpoint directories,
        steps, epochs, timestamps, losses or parent relationships.
        Returns the model's authoritative M3 listing (deterministic
        (step, created_at) ASCENDING order) filtered VERBATIM by each
        checkpoint's own persisted run_id, so every record is a
        complete verbatim CheckpointRecord and checkpoints of other
        runs/models never appear. A registered run with zero
        checkpoints returns []. Read-only, never writes.
        """
        # model + run ownership: raises FileNotFoundError (404 at the API)
        record = self.storage.load_record(model_id)
        if run_id not in {p.run_id for p in record.training_provenance}:
            raise FileNotFoundError(
                f"training run '{run_id}' not found for model "
                f"'{model_id}'")
        return [c for c in self.list_checkpoints(model_id)
                if c.run_id == run_id]

    def select_best_checkpoint(self, model_id: str) -> CheckpointSelection:
        """Deterministic best-checkpoint selection under the persisted
        validation-loss criterion (M52 read-only selection primitive).

        Candidates are the model's own checkpoints as returned by the
        authoritative M3 listing (unknown model -> FileNotFoundError;
        unreadable manifests are already skipped there), minus any whose
        persisted ``validation_loss`` is not finite (never candidates).
        The selection is the MINIMUM persisted ``validation_loss`` —
        read verbatim from the persisted manifest, never recomputed,
        never derived from perplexity, ``decision``, evaluations,
        comparisons, ids or timestamps. Ties on the exact minimum
        resolve by the listing's canonical (step, created_at) ASCENDING
        order — the first checkpoint among equals — and are disclosed
        via ``tied``. Read-only: never writes, never creates an
        evaluation or checkpoint, never touches weights or decisions,
        and persists no selection pointer. "Best" is this criterion
        only, never a claim of overall model quality. A valid model
        with no selectable checkpoints raises FileNotFoundError (404 at
        the API) — nothing is manufactured.
        """
        checkpoints = self.list_checkpoints(model_id)
        candidates = [c for c in checkpoints
                      if math.isfinite(c.validation_loss)]
        if not candidates:
            raise FileNotFoundError(
                f"model '{model_id}' has no selectable checkpoints "
                "(minimum persisted validation-loss criterion)")
        best = min(candidates, key=lambda c: c.validation_loss)
        min_loss = best.validation_loss
        tied = sum(1 for c in candidates
                   if c.validation_loss == min_loss) > 1
        return CheckpointSelection(
            model_id=model_id,
            criterion="minimum_persisted_validation_loss",
            candidate_count=len(candidates),
            tied=tied, checkpoint=best)

    # ------------------------------------------------------------------ #
    # Preflight
    # ------------------------------------------------------------------ #

    def preflight(self, cfg: TrainingConfig) -> dict[str, Any]:
        """Validate the whole job before anything is written.

        Raises FileNotFoundError (missing artifacts) or ValueError
        (incompatible/invalid configuration, failed dataset integrity).
        """
        try:
            model_record = self.storage.load_record(cfg.model_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"model '{cfg.model_id}' not found") from None

        # M55: resume_from_best is a WORKFLOW training-stage declaration —
        # only the workflow engine's single resolver may satisfy it (it
        # pins the M52 selection and hands training a pure M54 explicit
        # id). A DIRECT run declaring best is rejected: name the concrete
        # checkpoint explicitly (GET /checkpoints/best answers it).
        if cfg.resume_from_best:
            raise ValueError(
                "resume_from_best is a workflow training-stage "
                "declaration: the workflow engine's single resolver pins "
                "the M52 best checkpoint before execution — a direct "
                "training run must pass resume_from_checkpoint_id "
                "explicitly (e.g. the answer of GET /checkpoints/best)")

        # M54: validate an explicit resume point BEFORE anything is written —
        # through the EXISTING checkpoint verifier (model-scoped lookup:
        # unknown/foreign checkpoint -> FileNotFoundError; unreadable or
        # content-hash-mismatched weights -> RuntimeError). No second loader.
        if cfg.resume_from_checkpoint_id is not None:
            self.verify_checkpoint(cfg.model_id, cfg.resume_from_checkpoint_id)

        try:
            if cfg.dataset_version is None:
                meta = self.datasets.load_meta(cfg.dataset_id)
                version = meta.latest_version
            else:
                version = cfg.dataset_version
            # existence + integrity verify (never train on an unverified dataset)
            report = self.datasets.verify(cfg.dataset_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"dataset '{cfg.dataset_id}' not found") from None
        if report["status"] != "ok":
            raise ValueError(
                f"dataset '{cfg.dataset_id}' failed integrity verification — "
                "training is refused; fix or re-upload the dataset first")
        version_report = next((v for v in report["versions"] if v["version"] == version), None)
        if version_report is None:
            raise FileNotFoundError(
                f"dataset '{cfg.dataset_id}' has no version {version}")

        if cfg.max_seq_len > model_record.config.context_length:
            raise ValueError(
                f"max_seq_len ({cfg.max_seq_len}) exceeds the model's context_length "
                f"({model_record.config.context_length})")
        if cfg.max_seq_len > model_record.config.rope_max_seq() and \
                model_record.config.position_encoding.value == "rope":
            raise ValueError(
                f"max_seq_len ({cfg.max_seq_len}) exceeds the model's RoPE cache "
                f"({model_record.config.rope_max_seq()}) — reduce max_seq_len or rebuild "
                "the model with a larger context/rope")

        info, paths = self.datasets.tokenized_artifact(cfg.dataset_id, version, cfg.tokenizer_id)
        for split in ("train", "validation"):
            if split not in info.splits or info.splits[split].count < 1:
                raise ValueError(
                    f"tokenized split '{split}' is empty (dataset too small for the "
                    "deterministic 90/5/5 split); use a larger dataset")

        streams = {
            split: open_bin(paths[split], info.dtype) for split in ("train", "validation")
        }
        max_id = max(int(s.array.max()) for s in streams.values())
        if max_id >= model_record.config.vocab_size:
            raise ValueError(
                f"token ids up to {max_id} exceed the model's vocab_size "
                f"({model_record.config.vocab_size}) — the tokenizer and model are "
                "incompatible (train a model with vocab_size >= tokenizer vocab)")

        return {
            "model_record": model_record,
            "version": version,
            "tokenized": info,
            "streams": streams,
            "dataset_verify": report,
        }

    # ------------------------------------------------------------------ #
    # The run
    # ------------------------------------------------------------------ #

    def run(self, cfg: TrainingConfig) -> TrainingReport:
        start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        run_id = uuid.uuid4().hex[:12]
        seed = cfg.effective_seed()
        spec = detect_hardware()
        prepared = self.preflight(cfg)
        model_record: ModelRecord = prepared["model_record"]
        version = prepared["version"]
        model_cfg = model_record.config
        model_id = model_record.id
        train_stream: TokenStream = prepared["streams"]["train"]
        val_stream: TokenStream = prepared["streams"]["validation"]

        # ---- hardware-aware batch (recorded, never silently changed) ----
        budget = (spec.gpu_vram_bytes or spec.ram_bytes) * 0.6
        param_count = model_record.parameter_count
        actual_batch, reduced = fit_batch_size(
            cfg.batch_size, param_count, cfg.max_seq_len,
            model_cfg.hidden_size, model_cfg.n_layers, budget)
        if reduced:
            log.info("memory preflight: batch %s -> %s (estimated peak over budget)",
                     cfg.batch_size, actual_batch)
        if estimate_training_bytes(param_count, 1, cfg.max_seq_len,
                                   model_cfg.hidden_size, model_cfg.n_layers) > budget:
            raise ValueError(
                "this training configuration cannot fit the detected memory even with "
                f"batch_size=1 (budget ~{budget // 2**20} MiB); reduce max_seq_len or "
                "use a smaller model")

        # ---- deterministic seeds ----
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ---- rebuild the model from its config and load the initial weights ----
        # M54: an explicit resume point initializes THIS run from that
        # immutable checkpoint's VERIFIED weights (non-destructive: the
        # model's published weights.pt and latest_checkpoint are NOT touched
        # to prepare the run — publication happens only through the normal
        # training completion semantics below). None = published current
        # weights, exactly as before.
        device = "cuda" if spec.has_gpu else "cpu"
        model = build_transformer(model_cfg).module
        if cfg.resume_from_checkpoint_id is not None:
            state = self.verify_checkpoint(model_id,
                                           cfg.resume_from_checkpoint_id)
        else:
            state = self.storage.load_weights(model_id, device=device)
        missing, unexpected = restore_state(model, state)
        if missing or unexpected:
            raise RuntimeError(
                f"stored weights incompatible with config: {len(missing)} missing, "
                f"{len(unexpected)} unexpected")
        model.to(device)

        # ---- data layout: deterministic chunking, drop the incomplete tail ----
        seq_len = cfg.max_seq_len
        train_seqs = train_stream.count // seq_len
        micros_per_epoch = train_seqs // cfg.batch_size
        steps_per_epoch = micros_per_epoch // cfg.gradient_accumulation_steps
        if steps_per_epoch < 1:
            raise ValueError(
                f"train split yields only {train_seqs} usable sequences of length "
                f"{seq_len} (batch {cfg.batch_size} x accum "
                f"{cfg.gradient_accumulation_steps}) — reduce max_seq_len/batch or add "
                "more data")
        if cfg.epochs is not None:
            total_steps = cfg.epochs * steps_per_epoch
        else:
            total_steps = cfg.steps  # data windows recycle deterministically
        if total_steps < 1:
            raise ValueError("training would run zero optimizer steps")

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate,
                                betas=(cfg.adam_beta1, cfg.adam_beta2),
                                weight_decay=cfg.weight_decay)
        eval_points = sorted(
            set(range(cfg.eval_every_steps, total_steps + 1, cfg.eval_every_steps))
            | {total_steps})
        assert eval_points and eval_points[-1] == total_steps

        # M54: with an explicit resume point the run's TRUE starting state is
        # that checkpoint — recorded as the run's initial checkpoint and as
        # the parent of the run's first checkpoint (existing lineage fields,
        # no new schema); without one, today's semantics (latest lineage).
        initial_checkpoint_id = (cfg.resume_from_checkpoint_id
                                 if cfg.resume_from_checkpoint_id is not None
                                 else model_record.latest_checkpoint)
        parent_checkpoint_id = initial_checkpoint_id  # lineage across runs
        best_val: Optional[float] = None   # best among checkpoints (starts at baseline)
        best_ckpt_id: Optional[str] = None
        checkpoints: list[str] = []
        history: list[float] = []
        first_train_loss: Optional[float] = None
        last_train_loss: Optional[float] = None

        # ---- baseline evaluation BEFORE any update (the comparison anchor) ----
        model.eval()
        baseline_val = self._evaluate(model, val_stream, seq_len, cfg.batch_size, device)
        best_val = baseline_val

        def _checkpoint_and_decide(step: int, train_loss: float, lr_now: float) -> None:
            nonlocal best_val, best_ckpt_id, parent_checkpoint_id
            model.eval()
            val_loss = self._evaluate(model, val_stream, seq_len, cfg.batch_size, device)
            model.train()
            ppl = float(math.exp(min(val_loss, 100.0)))
            improved = val_loss < best_val - 1e-9
            decision = CheckpointDecision.ACCEPT if improved else CheckpointDecision.NOT_BEST
            if improved:
                best_val = val_loss
            ckpt_id = self._save_checkpoint(
                model, model_id, run_id, parent_checkpoint_id=parent_checkpoint_id,
                step=step, epoch=step / steps_per_epoch, cfg=cfg, version=version,
                train_loss=train_loss, val_loss=val_loss, perplexity=ppl,
                lr=lr_now, decision=decision)
            parent_checkpoint_id = ckpt_id  # chain within the run
            if improved:
                best_ckpt_id = ckpt_id
            checkpoints.append(ckpt_id)

        # ---- train loop -------------------------------------------------
        # Token windows depend only on the position inside the steps_per_epoch
        # cycle: window offset (tokens) = ((step % steps_per_epoch) * accum + a)
        # * batch * seq_len. Each epoch recycles the same deterministic order
        # (no shuffling) — reproducibility by construction.
        model.train()
        step = 0
        while step < total_steps:
            in_cycle = step % steps_per_epoch
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for a in range(cfg.gradient_accumulation_steps):
                micro = in_cycle * cfg.gradient_accumulation_steps + a
                start = micro * cfg.batch_size * seq_len
                ids = self._ids_tensor(train_stream.array, start,
                                       cfg.batch_size * seq_len, seq_len, device)
                logits, _ = model(ids)
                loss = self._lm_loss(logits, ids) / cfg.gradient_accumulation_steps
                loss.backward()
                step_loss += float(loss.item())  # mean over the group's micro-batches
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            lr_now = lr_value(step, total_steps, cfg.warmup_steps, cfg.learning_rate,
                              cfg.lr_schedule.value)
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.step()
            step += 1
            history.append(step_loss)
            if first_train_loss is None:
                first_train_loss = step_loss
            last_train_loss = step_loss
            if step in eval_points:
                _checkpoint_and_decide(step, step_loss, lr_now)

        # ---- finalize weights --------------------------------------------
        best_ppl = float(math.exp(min(best_val, 100.0))) if best_val is not None else None
        accepted = best_ckpt_id is not None  # at least one checkpoint beat the baseline
        rolled_back_to: Optional[str] = None
        final_model_version = initial_checkpoint_id
        last_created = checkpoints[-1] if checkpoints else None
        best_id_for_manifest: Optional[str] = None

        if cfg.keep_best:
            if best_ckpt_id is not None:
                # some checkpoint beat the baseline
                if best_ckpt_id == last_created:
                    # the final state IS the best state -> publish it as current
                    self.storage.write_weights(model_id, self._state_on_cpu(model))
                    final_model_version = best_ckpt_id
                else:
                    self._restore_weights_from_checkpoint(model_id, best_ckpt_id)
                    rolled_back_to = best_ckpt_id
                    final_model_version = best_ckpt_id
                best_id_for_manifest = best_ckpt_id
            else:
                # nothing beat the baseline: keep the pre-run weights untouched
                final_model_version = initial_checkpoint_id
        else:
            # adopt the final weights regardless of the comparison
            self.storage.write_weights(model_id, self._state_on_cpu(model))
            final_model_version = last_created

        # ---- update the model manifest (atomic; only after all writes) ----
        provenance = RunProvenance(
            run_id=run_id, method=cfg.method, config=cfg.model_dump(mode="json"),
            seed=seed, dataset_id=cfg.dataset_id, dataset_version=version,
            tokenizer_id=cfg.tokenizer_id, parent_checkpoint_id=initial_checkpoint_id,
            initial_checkpoint_id=initial_checkpoint_id,
            final_checkpoint_id=final_model_version,
            requested_batch_size=cfg.batch_size, actual_batch_size=actual_batch,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            optimizer_steps=step, epochs_run=step / steps_per_epoch,
            baseline_validation_loss=baseline_val,
            best_validation_loss=best_val, best_perplexity=best_ppl,
            initial_train_loss=first_train_loss, final_train_loss=last_train_loss,
            accepted=accepted, rolled_back_to=rolled_back_to,
            hardware=spec.to_dict(), started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_seconds=time.monotonic() - start,
        )
        updated = model_record.model_copy(update={
            "updated_at": datetime.now(timezone.utc),
            "latest_checkpoint": last_created,
            "best_checkpoint": best_id_for_manifest,
            "training_provenance": [*model_record.training_provenance, provenance],
        })
        self.storage.save_record(updated)

        report = TrainingReport(
            run_id=run_id, model_id=model_id, method=cfg.method,
            dataset_id=cfg.dataset_id, dataset_version=version,
            tokenizer_id=cfg.tokenizer_id, seed=seed,
            initial_model_version=initial_checkpoint_id,
            final_model_version=final_model_version,
            optimizer_steps=step, epochs_run=step / steps_per_epoch,
            requested_batch_size=cfg.batch_size, actual_batch_size=actual_batch,
            baseline_validation_loss=baseline_val,
            baseline_perplexity=(float(math.exp(min(baseline_val, 100.0)))
                                 if baseline_val is not None else None),
            initial_train_loss=first_train_loss, final_train_loss=last_train_loss,
            best_validation_loss=best_val, best_perplexity=best_ppl,
            accepted=accepted, rolled_back_to=rolled_back_to,
            train_loss_history=history,
            checkpoints=[self.get_checkpoint(model_id, c).model_dump(mode="json")
                         for c in checkpoints],
            hardware=spec.to_dict(), started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_seconds=time.monotonic() - start,
        )
        log.info("training run %s on model %s (%s): %s steps, baseline val %.4f, "
                 "best val %s, accepted=%s, rollback=%s",
                 run_id, model_id, cfg.method.value, step, baseline_val,
                 f"{best_val:.4f}" if best_val is not None else "n/a",
                 accepted, rolled_back_to)
        return report

    # ------------------------------------------------------------------ #
    # Loss / evaluation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ids_tensor(array: np.ndarray, start_token: int, n_tokens: int,
                    seq_len: int, device: str) -> torch.Tensor:
        """Copy a token window out of the (read-only) memmap into a tensor."""
        chunk = np.asarray(array[start_token:start_token + n_tokens], dtype=np.int64)
        return torch.from_numpy(chunk).view(-1, seq_len).to(device)

    @staticmethod
    def _lm_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Causal LM loss over a full batch: predict ids[..., 1:] from logits[..., :-1]."""
        B, T, V = logits.shape
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, V), ids[:, 1:].reshape(-1))

    @staticmethod
    def _state_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def _evaluate(self, model: torch.nn.Module, stream: TokenStream, seq_len: int,
                  batch: int, device: str) -> float:
        """Mean token cross-entropy over the whole (packed) stream, no grad."""
        n_seqs = stream.count // seq_len
        if n_seqs < 1:
            return float("nan")
        model.eval()
        total_loss = 0.0
        total_pairs = 0
        with torch.no_grad():
            for s in range(0, n_seqs, batch):
                take = min(batch, n_seqs - s)  # the last group may be partial
                ids = self._ids_tensor(stream.array, s * seq_len, take * seq_len,
                                       seq_len, device)
                logits, _ = model(ids)
                loss = self._lm_loss(logits, ids)
                pairs = (ids.shape[1] - 1) * ids.shape[0]
                total_loss += float(loss.item()) * pairs
                total_pairs += pairs
        model.train()
        return total_loss / total_pairs if total_pairs else float("nan")

    # ------------------------------------------------------------------ #
    # Checkpoint I/O (immutable, content-addressed)
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, model: torch.nn.Module, model_id: str, run_id: str,
                         parent_checkpoint_id: Optional[str], step: int, epoch: float,
                         cfg: TrainingConfig, version: int, train_loss: float,
                         val_loss: float, perplexity: float, lr: float,
                         decision: CheckpointDecision) -> str:
        decision = CheckpointDecision(decision)  # accept enum or raw value
        ckpt_id = uuid.uuid4().hex[:12]
        state = self._state_on_cpu(model)
        h = content_hash(state)
        cdir = self._ckpt_dir(model_id, ckpt_id)
        cdir.mkdir(parents=True, exist_ok=False)

        # content-addressing: reuse an identical physical artifact when present
        existing = self._find_artifact(model_id, h)
        target = cdir / CKPT_WEIGHTS
        if existing is not None:
            os.link(existing, target)  # same filesystem -> zero extra bytes
        else:
            tmp = cdir / f".tmp-{uuid.uuid4().hex}.pt"
            try:
                torch.save(state, tmp)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)

        record = CheckpointRecord(
            checkpoint_id=ckpt_id, model_id=model_id, run_id=run_id,
            parent_checkpoint_id=parent_checkpoint_id,
            step=step, epoch=round(epoch, 3), method=cfg.method,
            dataset_id=cfg.dataset_id, dataset_version=version,
            train_loss=round(float(train_loss), 6),
            validation_loss=round(float(val_loss), 6),
            perplexity=round(float(perplexity), 6),
            learning_rate=lr, decision=decision, weights_sha256=h,
            created_at=datetime.now(timezone.utc),
        )
        atomic_write_json(cdir / CKPT_MANIFEST, record.model_dump(mode="json"))
        log.info("checkpoint %s (step %s): train %.4f val %.4f ppl %.2f -> %s",
                 ckpt_id, step, train_loss, val_loss, perplexity, decision.value)
        return ckpt_id

    def _find_artifact(self, model_id: str, weights_sha256: str) -> Optional[Path]:
        """Path of an existing checkpoint weights file with the same content hash."""
        root = self._ckpt_root(model_id)
        if not root.exists():
            return None
        for d in sorted(root.iterdir()):
            mpath = d / CKPT_MANIFEST
            if mpath.exists():
                try:
                    if read_json(mpath).get("weights_sha256") == weights_sha256:
                        wpath = d / CKPT_WEIGHTS
                        if wpath.exists():
                            return wpath
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------ #
    # Integrity + rollback
    # ------------------------------------------------------------------ #

    def verify_checkpoint(self, model_id: str, ckpt_id: str) -> dict[str, torch.Tensor]:
        """Load a checkpoint's weights and confirm they match its manifest hash.

        Returns the state on success; raises RuntimeError when corrupted.
        """
        record = self.get_checkpoint(model_id, ckpt_id)
        wpath = self._ckpt_dir(model_id, ckpt_id) / CKPT_WEIGHTS
        if not wpath.exists():
            raise RuntimeError(f"checkpoint '{ckpt_id}' has no weights file")
        try:
            state = torch.load(wpath, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"checkpoint '{ckpt_id}' weights unreadable: {exc}") from exc
        if content_hash(state) != record.weights_sha256:
            raise RuntimeError(
                f"checkpoint '{ckpt_id}' failed integrity verification (content hash "
                "mismatch) — refusing to use corrupted weights")
        return state

    def rollback(self, model_id: str, ckpt_id: str) -> ModelRecord:
        """Restore a verified checkpoint's weights as the model's current weights."""
        state = self.verify_checkpoint(model_id, ckpt_id)  # raises when corrupted
        self.storage.write_weights(model_id, state)
        record = self.storage.load_record(model_id)
        updated = record.model_copy(update={
            "updated_at": datetime.now(timezone.utc),
            "latest_checkpoint": ckpt_id,
        })
        self.storage.save_record(updated)
        log.info("model %s rolled back to checkpoint %s", model_id, ckpt_id)
        return self.storage.load_record(model_id)

    def _restore_weights_from_checkpoint(self, model_id: str, ckpt_id: str) -> None:
        """Internal best-weights restore (assumes the checkpoint is intact)."""
        state = self.verify_checkpoint(model_id, ckpt_id)
        self.storage.write_weights(model_id, state)

    # ------------------------------------------------------------------ #

    def delete_checkpoints(self, model_id: str) -> None:
        """Drop all checkpoints of a model (weights.pt and manifest untouched)."""
        shutil.rmtree(self._ckpt_root(model_id), ignore_errors=True)
