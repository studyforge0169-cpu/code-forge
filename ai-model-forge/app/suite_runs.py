"""Explicit multi-probe evaluation batches over named probe suites (M10).

A suite run is a THIN ORCHESTRATION LAYER over M4 and nothing else:

    one named ProbeSuite (M9) + one explicit model state (M4 convention)
                │  (each probe independently, canonical suite order)
                ├── Probe 1 ──→ exact M4 evaluation (reused or created)
                ├── Probe 2 ──→ exact M4 evaluation (reused or created)
                └── ...
                │
                ↓
    one immutable SuiteRunRecord (suite-runs/<id>/manifest.json)

Rules honoured here (tested):

  * no aggregation: the run reports per-probe results plus execution
    bookkeeping counts (probe/completed/reused/failed). No average loss,
    no perplexity aggregate, no score, no pass percentage, no ranking.
    A suite is a collection of probes, never a benchmark score
  * evidence reuse uses the EXACT M4 identity (state kind/id/canonical hash,
    dataset/version/split/tokenizer, window, token cap, batch size, seed) via
    ComparisonEngine.resolve_evaluation — the same reuse semantics M5–M9 use.
    Repeated identical runs reuse existing evaluations; only the new run
    manifest is appended
  * every probe executes through the existing EvaluationEngine; loss,
    perplexity, batching, truncation, seeds, state hashing and verification
    stay owned by M4 — nothing here duplicates them
  * deterministic failure semantics: the model, the state and the suite are
    resolved and verified BEFORE anything executes (unknown/corrupt -> clean
    error, nothing persisted). A probe that cannot execute because of an
    invalid/missing underlying artifact is recorded per-probe as ``failed``
    with deterministic text (no fabricated evaluation, no silent skip), the
    run persists with status ``failed``, and remaining probes still execute
    in canonical order
  * deterministic result_hash: sha256 over the semantic execution (model +
    config identity, state incl. canonical hash, suite id + probes hash,
    canonical probe→evaluation mapping, status, failure text) — excluding
    suite_run_id, timestamps, durations, paths and the created/reused
    bookkeeping labels, so two logically identical runs over identical
    immutable evidence reproduce it exactly
  * persistence is one atomic manifest per run; evaluations, weights,
    datasets and tokenizers are never copied
  * read-only per-suite history (M21): ``list_suite_runs_for_suite``
    answers "which immutable suite-run records belong to this model and
    this named suite?" — the model's authoritative listing above,
    filtered by the persisted suite_id, after M9 registry resolution.
    No new store, index, cache or manifest format; never writes
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from .schemas import (
    ModelRecord,
    SuiteRunProbeResult,
    SuiteRunRecord,
    SuiteRunRequest,
    SuiteRunStatus,
    utcnow,
)
from .storage import atomic_write_json, read_json
from .comparison import ComparisonEngine
from .dataset import DatasetEngine
from .evaluation import EvaluationEngine
from .policies import PolicyEngine

SUITE_RUNS_DIR = "suite-runs"

log = logging.getLogger("forge.suite_runs")


class SuiteRunEngine:
    """One named suite against one explicit state, probe by probe (M4 only)."""

    def __init__(self, storage) -> None:
        self.storage = storage
        self.datasets = DatasetEngine(storage)
        self.evaluation = EvaluationEngine(storage)
        self.comparison = ComparisonEngine(storage)
        self.policies = PolicyEngine(storage)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def _runs_root(self) -> Path:
        return self.storage.root / SUITE_RUNS_DIR

    def _run_dir(self, suite_run_id: str) -> Path:
        return self._runs_root() / suite_run_id

    # ------------------------------------------------------------------ #
    # The run
    # ------------------------------------------------------------------ #

    def run(self, request: SuiteRunRequest) -> SuiteRunRecord:
        """Execute every probe of one suite against one state (never writes
        more than the run manifest + any genuinely missing M4 evaluations)."""
        start = time.monotonic()
        model = self._require_model(request.model_id)
        model_id = model.id
        state = request.state

        # ---- preflight: state + suite verified BEFORE anything executes ----
        state_hash = self.comparison.verified_state_hash(model_id, state)
        suite = self._resolve_suite(request.suite_id)
        candidates = self.evaluation.list_evaluations(model_id)
        seen: set[str] = {r.eval_id for r in candidates}

        # ---- per-probe execution in canonical (stored) suite order --------
        results: list[SuiteRunProbeResult] = []
        created = reused = failed = 0
        for probe in suite.probes:
            try:
                cfg = probe.to_evaluation_config(
                    model_id, checkpoint_id=state.checkpoint_id)
                version = self._resolve_version(cfg.dataset_id,
                                                cfg.dataset_version)
                window = (cfg.max_seq_len if cfg.max_seq_len is not None
                          else model.config.context_length)
                seed = cfg.effective_seed()
                rec = self.comparison.resolve_evaluation(
                    model_id=model_id, state=state, state_hash=state_hash,
                    dataset_id=cfg.dataset_id, version=version,
                    split=cfg.split.value, tokenizer_id=cfg.tokenizer_id,
                    max_eval_tokens=cfg.max_eval_tokens,
                    batch_size=cfg.batch_size, window=window, seed=seed,
                    candidates=candidates)
                outcome = "reused" if rec.eval_id in seen else "created"
                seen.add(rec.eval_id)
                candidates.append(rec)
                if outcome == "reused":
                    reused += 1
                else:
                    created += 1
                results.append(SuiteRunProbeResult(
                    probe=probe.model_copy(update={"dataset_version": version}),
                    evaluation_id=rec.eval_id, outcome=outcome))
            except Exception as exc:  # noqa: BLE001 - one broken probe never
                failed += 1            # hides the others or kills the run
                results.append(SuiteRunProbeResult(
                    probe=probe, evaluation_id=None, outcome="failed",
                    error=f"{type(exc).__name__}: {exc}"))

        status = (SuiteRunStatus.FAILED if failed else SuiteRunStatus.COMPLETED)
        record = SuiteRunRecord(
            suite_run_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            config_hash=model.config_hash,
            state=state,
            state_hash=state_hash,
            suite_id=suite.suite_id,
            suite_probes_hash=suite.probes_hash,
            status=status,
            probe_count=len(suite.probes),
            completed_count=created + reused,
            reused_count=reused,
            failed_count=failed,
            results=results,
            result_hash="",  # filled below from the record without itself
            created_at=utcnow(),
            duration_seconds=round(time.monotonic() - start, 3),
        )
        record = record.model_copy(
            update={"result_hash": self.result_hash(record)})
        self._persist(record)
        log.info("suite run %s on model %s, suite '%s' (%d probes): %s "
                 "(%d created, %d reused, %d failed)", record.suite_run_id,
                 model_id, suite.suite_id, record.probe_count, status.value,
                 created, reused, failed)
        return record

    # ------------------------------------------------------------------ #
    # Listing / retrieval (read-only)
    # ------------------------------------------------------------------ #

    def list_suite_runs(self, model_id: str) -> list[SuiteRunRecord]:
        """All immutable suite runs of one model, oldest first.

        Raises FileNotFoundError for an unknown model; returns [] when the
        model has no suite runs yet. Never writes.
        """
        self._require_model(model_id)
        root = self._runs_root()
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            mpath = d / "manifest.json"
            if not mpath.exists():
                continue
            try:
                rec = SuiteRunRecord(**read_json(mpath))
            except Exception as exc:  # noqa: BLE001
                log.warning("unreadable suite-run manifest %s: %s", mpath,
                            exc)
                continue
            if rec.model_id == model_id:
                records.append(rec)
        records.sort(key=lambda r: (r.created_at, r.suite_run_id))
        return records

    def get_suite_run(self, model_id: str, suite_run_id: str) -> SuiteRunRecord:
        """One persisted immutable suite run (404 for unknown model or run)."""
        self._require_model(model_id)
        mpath = self._run_dir(suite_run_id) / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(
                f"suite run '{suite_run_id}' not found for model "
                f"'{model_id}'")
        rec = SuiteRunRecord(**read_json(mpath))
        if rec.model_id != model_id:
            raise FileNotFoundError(
                f"suite run '{suite_run_id}' not found for model "
                f"'{model_id}'")
        return rec

    # ------------------------------------------------------------------ #
    # M21: per-suite access (read-only)
    # ------------------------------------------------------------------ #

    def list_suite_runs_for_suite(self, model_id: str,
                                  suite_id: str) -> list[SuiteRunRecord]:
        """Immutable M10 suite runs of ONE named suite of ONE model (M21).

        Resolution: unknown model or unregistered suite ->
        FileNotFoundError; the suite is validated through the existing M9
        registry (``PolicyEngine.get_suite``) — suite ids are forge-wide
        and resolved from the persisted suite manifest, never inferred
        from run-directory names. The result is the model's authoritative
        M10 listing (the exact engine parse + deterministic
        (created_at, suite_run_id) ASCENDING order) filtered by the
        persisted ``suite_id`` recorded in each SuiteRunRecord, so every
        returned record is a complete verbatim SuiteRunRecord and runs of
        other suites/models never appear. A valid suite with no runs for
        this model returns []. Read-only, never writes.
        """
        # existence: raises FileNotFoundError (404 at the API)
        self._require_model(model_id)
        # suite registry resolution: raises FileNotFoundError when unknown
        self.policies.get_suite(suite_id)
        return [r for r in self.list_suite_runs(model_id)
                if r.suite_id == suite_id]

    # ------------------------------------------------------------------ #
    # Deterministic hashing
    # ------------------------------------------------------------------ #

    @staticmethod
    def result_hash(record: SuiteRunRecord) -> str:
        """sha256 over the semantic execution only (see module docstring)."""
        payload = {
            "model_id": record.model_id,
            "config_hash": record.config_hash,
            "state": {
                "state_kind": record.state.state_kind.value,
                "checkpoint_id": record.state.checkpoint_id,
                "state_hash": record.state_hash,
            },
            "suite_id": record.suite_id,
            "suite_probes_hash": record.suite_probes_hash,
            "status": record.status.value,
            "results": [
                {
                    "probe": r.probe.model_dump(mode="json"),
                    "evaluation_id": r.evaluation_id,
                    "error": r.error,
                }
                for r in record.results
            ],
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _persist(self, record: SuiteRunRecord) -> None:
        d = self._run_dir(record.suite_run_id)
        d.mkdir(parents=True, exist_ok=False)
        atomic_write_json(d / "manifest.json", record.model_dump(mode="json"))

    def _require_model(self, model_id: str) -> ModelRecord:
        try:
            return self.storage.load_record(model_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"model '{model_id}' not found") from None

    def _resolve_suite(self, suite_id: str):
        """Registry resolution with the M9 error semantics."""
        mpath = self.storage.root / "probe-suites" / suite_id / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"probe suite '{suite_id}' not found")
        try:
            return self.policies.get_suite(suite_id)
        except Exception as exc:  # corrupt/unusable suite definition
            raise RuntimeError(
                f"probe suite '{suite_id}' manifest is corrupt: "
                f"{type(exc).__name__}: {exc}") from exc

    def _resolve_version(self, dataset_id: str,
                         dataset_version: Optional[int]) -> int:
        """Exact M4/M6 dataset-version semantics (None -> latest version)."""
        meta = self.datasets.load_meta(dataset_id)
        return meta.latest_version if dataset_version is None \
            else dataset_version
