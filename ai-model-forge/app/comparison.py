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
    ModelRecord,
)
from .storage import Storage, atomic_write_json, read_json
from .training import TrainingEngine

log = forge_cfg.get_logger("comparison")

COMPARISON_MANIFEST = "manifest.json"
COMPARISONS_DIR = "comparisons"


class ComparisonEngine:
    """Comparison orchestration for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.datasets = DatasetEngine(storage)
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
