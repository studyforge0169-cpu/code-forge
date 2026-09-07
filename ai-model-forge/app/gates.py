"""Stage gates: policy-driven, evidence-based run decisions (Milestone 6).

A stage gate answers one explicit question:

    given this policy (fixed probe + baseline form + tolerance/constraints),
    this candidate model state and this exact evaluation probe — should the
    candidate be accepted?

The answer is a deterministic, auditable ``passed | failed`` decision that
embeds the underlying M5-style verdict (``improved | regressed | unchanged``)
and the full evidence chain:

    GateDecision -> Evaluation(s) -> Comparison (when a baseline state exists)
                 -> model state/checkpoint

Design rules honoured here (tested):
  * the policy is INLINE and embedded in every decision — there is no
    policy registry/CRUD (smallest honest representation; every explicit gate
    request is an auditable decision)
  * candidate + baseline reuse the M5 ``ComparisonState`` convention and the
    M5 engines for state verification, exact evaluation reuse and comparison
    maths — nothing is duplicated
  * baseline forms: checkpoint, current weights, a past evaluation result
    hash (verified to match the policy probe exactly), or an absolute
    minimum-loss threshold (no state comparison)
  * verdict/delta semantics are M5's: delta_loss = candidate - baseline;
    verdict from tolerance. Pass/fail semantics (loss-only):
      improved           -> passed
      unchanged          -> passed (unless an absolute minimum_loss ceiling
                            is violated)
      regressed          -> passed only when max_regression_delta is set and
                            delta <= max_regression_delta (bounded regression),
                            else failed
    an optional absolute ``minimum_loss`` ceiling may ride along with any
    state baseline (candidate_loss must be <= minimum_loss to pass)
  * when a state-based gate fails against a CHECKPOINT baseline the decision
    carries ``suggested_checkpoint_id`` + a hint — the gate NEVER rolls back,
    retrains or selects; the user must call the existing M3 rollback endpoint
  * nothing is persisted when preflight validation or integrity verification
    fails; a repeated identical request reuses existing evaluation/comparison
    evidence and only appends one new decision manifest (gates are
    auditable, evidence is deduplicated)
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
from .comparison import ComparisonEngine
from .dataset import DatasetEngine
from .evaluation import EvaluationEngine
from .policies import PolicyEngine
from .schemas import (
    ComparisonRecord,
    ComparisonSide,
    ComparisonState,
    ComparisonVerdict,
    EvalStateKind,
    EvaluationRecord,
    GateBaselineType,
    GateDecision,
    GateDecisionResult,
    GatePolicy,
    GateRequest,
    ModelRecord,
)
from .storage import Storage, atomic_write_json, read_json
from .training import TrainingEngine

log = forge_cfg.get_logger("gates")

GATE_MANIFEST = "manifest.json"
GATES_DIR = "gates"


class GateEngine:
    """Stage-gate decisions for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.policies = PolicyEngine(storage)
        self.datasets = DatasetEngine(storage)
        self.training = TrainingEngine(storage)
        self.evaluation = EvaluationEngine(storage)
        self.comparison = ComparisonEngine(storage)

    # ------------------------------------------------------------------ #
    # Paths / registry helpers
    # ------------------------------------------------------------------ #

    def _gates_root(self, model_id: str) -> Path:
        return self.storage.model_dir(model_id) / GATES_DIR

    def _gate_dir(self, model_id: str, decision_id: str) -> Path:
        return self._gates_root(model_id) / f"gate-{decision_id}"

    def _model_exists(self, model_id: str) -> bool:
        return (self.storage.model_dir(model_id) / "manifest.json").exists()

    def list_decisions(self, model_id: str) -> list[GateDecision]:
        """Immutable gate history (append-only, deterministic order).

        Raises FileNotFoundError for an unknown model; returns [] when the
        model has no gate decisions yet.
        """
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        root = self._gates_root(model_id)
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith("gate-"):
                continue
            mpath = d / GATE_MANIFEST
            if mpath.exists():
                try:
                    records.append(GateDecision(**read_json(mpath)))
                except Exception:
                    log.warning("unreadable gate manifest %s", mpath)
        records.sort(key=lambda r: (r.created_at, r.decision_id))
        return records

    def get_decision(self, model_id: str, decision_id: str) -> GateDecision:
        """One persisted immutable gate decision (never mutates it)."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        path = self._gate_dir(model_id, decision_id) / GATE_MANIFEST
        if not path.exists():
            raise FileNotFoundError(
                f"gate decision '{decision_id}' not found for model '{model_id}'")
        return GateDecision(**read_json(path))

    # ------------------------------------------------------------------ #
    # M23: per-policy access (read-only)
    # M34: per-comparison access (read-only)
    # ------------------------------------------------------------------ #

    def list_decisions_for_policy(self, model_id: str,
                                  policy_id: str) -> list[GateDecision]:
        """Immutable gate decisions of ONE registered policy of ONE model
        (M23).

        Resolution: unknown model or unregistered policy ->
        FileNotFoundError; the policy is resolved through the existing M9
        registry (``PolicyEngine.get_policy``) — policy identity is the
        persisted definition manifest, never inferred from gate-directory
        names. The result is the model's authoritative M6 listing above
        (the exact engine parse + deterministic (created_at, decision_id)
        ASCENDING order) filtered by the persisted ``policy_id`` recorded
        in each GateDecision, so every returned record is a complete
        verbatim GateDecision and decisions of other policies/models
        never appear. Inline-policy decisions keep ``policy_id=None`` and
        therefore belong to no policy id. A valid policy with no
        decisions for this model returns []. Read-only, never writes.
        """
        # authoritative listing validates the model: FileNotFoundError (404)
        decisions = self.list_decisions(model_id)
        # policy registry resolution: raises FileNotFoundError when unknown
        self.policies.get_policy(policy_id)
        return [d for d in decisions if d.policy_id == policy_id]

    def list_decisions_for_comparison(self, model_id: str,
                                      comparison_id: str) -> list[GateDecision]:
        """Immutable gate decisions that judged ONE M5 comparison of ONE
        model (M34).

        Membership comes from the persisted decision comparison identity
        ONLY: every ``GateDecision`` carries a top-level ``comparison_id``
        (the M5 record it judged — the M6 run persists it verbatim from
        the request), and a decision belongs to the request when its
        persisted ``comparison_id`` equals the requested id, matched
        VERBATIM — never filenames, gate-directory names, checkpoint
        ids, policy ids or hashes, and never re-derived from the
        comparison's current content. Legacy direct-evaluation
        decisions keep ``comparison_id=None`` and therefore belong to
        NO by-comparison group (None never matches any requested id);
        they stay in the generic M6 listing untouched. Each decision
        appears EXACTLY ONCE (the authoritative listing holds each
        record exactly once). Resolution: unknown model or unknown
        comparison -> FileNotFoundError; ownership is validated through
        the model's own M5 registry
        (``ComparisonEngine.get_comparison`` — the same getter
        ``GET /models/{id}/comparisons/{comparison_id}`` exposes); a
        comparison id belonging to another model is not registered
        under this model and raises FileNotFoundError, exactly like an
        unknown one (comparisons are model-scoped). The result keeps
        the authoritative M6 (created_at, decision_id) ASCENDING
        order. A valid comparison with no decisions returns [].
        Read-only, never writes.
        """
        # authoritative listing validates the model: FileNotFoundError (404)
        decisions = self.list_decisions(model_id)
        # M5 registry ownership resolution: raises FileNotFoundError when
        # unknown or belonging to another model
        self.comparison.get_comparison(model_id, comparison_id)
        return [d for d in decisions if d.comparison_id == comparison_id]

    def list_decisions_for_decision(
            self, model_id: str, decision: GateDecisionResult
    ) -> list[GateDecision]:
        """Immutable gate decisions of ONE model with ONE decision
        result (M41).

        Membership comes from the persisted decision identity ONLY:
        every ``GateDecision`` carries a top-level ``decision:
        GateDecisionResult`` (the schema enum passed/failed — the
        immutable policy verdict persisted at run time by the M6 gate
        flow: the policy semantics over the measured evidence, which
        may legitimately DIFFER from the loss-only comparison verdict
        — e.g. an improved candidate still fails a minimum_loss
        ceiling), and a decision belongs to the request when its
        persisted ``decision`` equals the requested value — never
        recalculated from loss deltas, tolerances, policy thresholds,
        gate configuration or comparison results, and never resolved
        or rewritten into another value (the persisted decision is the
        ONLY authority; this method never re-evaluates a gate).
        Decision results have NO registry (unlike the M23 policy / M34
        comparison axes): the enum IS the contract, so an unsupported
        decision value is rejected at the API boundary with 422
        (schema-level validation — it never even reaches this method),
        while an unknown model raises FileNotFoundError exactly like
        the sibling groupings. Each decision appears EXACTLY ONCE (the
        authoritative listing holds each record exactly once). The
        result is the model's authoritative M6 listing above (the
        exact engine parse + deterministic (created_at, decision_id)
        ASCENDING order) filtered by the persisted decision; complete
        verbatim ``GateDecision`` payloads, no rewritten fields. A
        valid decision with no gate decisions for the model returns
        []. Decision results are model-scoped through the listing
        itself — a model never sees another model's decisions.
        Read-only, never writes.
        """
        # the authoritative listing validates the model:
        # raises FileNotFoundError (404 at the API); the decision
        # needs NO registry lookup and is NEVER recalculated — the
        # persisted field is filtered verbatim
        return [d for d in self.list_decisions(model_id)
                if d.decision == decision]

    # ------------------------------------------------------------------ #
    # The gate run
    # ------------------------------------------------------------------ #

    def run(self, request: GateRequest) -> GateDecision:
        start = time.monotonic()
        # One resolution point: an inline policy passes through untouched
        # (exact M6 semantics); a registry policy_id resolves to its
        # immutable definition. Either way `policy` below is the executed
        # GatePolicy embedded in the decision.
        policy, policy_id, config_hash = self.policies.gate_policy_for(
            request.model_id, request.policy, request.policy_id)
        model_record = self._require_model(request.model_id)
        model_id = request.model_id
        window = policy.max_seq_len if policy.max_seq_len is not None \
            else model_record.config.context_length
        seed = policy.effective_seed()

        # ---- resolve the dataset version exactly once --------------------
        try:
            meta = self.datasets.load_meta(policy.dataset_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"dataset '{policy.dataset_id}' not found") from None
        version = meta.latest_version if policy.dataset_version is None \
            else policy.dataset_version

        # ---- verify the candidate state BEFORE anything is persisted -----
        candidate_hash = self.comparison.verified_state_hash(
            model_id, request.candidate)
        candidates = self.evaluation.list_evaluations(model_id)

        # ---- baseline resolution per policy form --------------------------
        baseline: Optional[EvaluationRecord] = None
        baseline_side: Optional[ComparisonSide] = None
        if policy.baseline_type == GateBaselineType.MINIMUM_LOSS:
            pass  # absolute threshold: no baseline state, no comparison
        else:
            baseline = self._resolve_baseline(
                policy, model_id, window, seed, version, candidates)
            candidates.append(baseline)
            baseline_side = self._side_of(baseline)

        # ---- candidate evaluation (exact reuse, else one M4 run) ----------
        cand_eval = self.comparison.resolve_evaluation(
            model_id=model_id, state=request.candidate, state_hash=candidate_hash,
            dataset_id=policy.dataset_id, version=version,
            split=policy.split.value, tokenizer_id=policy.tokenizer_id,
            max_eval_tokens=policy.max_eval_tokens, batch_size=policy.batch_size,
            window=window, seed=seed, candidates=candidates)
        cand_side = self._side_of(cand_eval)

        # ---- comparison (reuse an equivalent record, else create one) -----
        comparison: Optional[ComparisonRecord] = None
        verdict: Optional[ComparisonVerdict] = None
        delta: Optional[float] = None
        if baseline is not None:
            comparison = self._find_exact_comparison(
                model_id=model_id, baseline_state_hash=baseline_side.state_hash,
                candidate_state_hash=cand_side.state_hash,
                dataset_id=policy.dataset_id, version=version,
                split=policy.split.value, tokenizer_id=policy.tokenizer_id,
                max_eval_tokens=policy.max_eval_tokens,
                batch_size=policy.batch_size, window=window, seed=seed,
                tolerance=policy.tolerance)
            if comparison is None:
                # M5 semantics: baseline is A, candidate is B, so
                # delta = candidate - baseline (negative = candidate improved)
                comparison = self.comparison.compare_records(
                    baseline, cand_eval, tolerance=policy.tolerance,
                    started_at=start)
            verdict = comparison.verdict
            delta = comparison.delta_loss_nats

        # ---- decide --------------------------------------------------------
        decision, reason = self._decide(policy, cand_side.loss_nats,
                                        baseline_side, delta, verdict)
        suggested, hint = self._suggestion(policy, baseline_side, decision)

        record = GateDecision(
            decision_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            config_hash=model_record.config_hash,
            policy=policy.model_copy(update={"dataset_version": version}),
            policy_id=policy_id,
            policy_config_hash=config_hash,
            candidate=cand_side,
            baseline=baseline_side,
            comparison_id=comparison.comparison_id if comparison else None,
            comparison_result_hash=comparison.result_hash if comparison else None,
            candidate_loss=cand_side.loss_nats,
            baseline_loss=baseline_side.loss_nats if baseline_side else None,
            delta_loss_nats=delta,
            verdict=verdict,
            tolerance=policy.tolerance,
            max_regression_delta=policy.max_regression_delta,
            minimum_loss=policy.minimum_loss,
            decision=decision,
            reason=reason,
            suggested_checkpoint_id=suggested,
            hint=hint,
            result_hash="",  # filled below from the rounded record
            created_at=datetime.now(timezone.utc),
            duration_seconds=round(time.monotonic() - start, 3),
        )
        record = record.model_copy(update={"result_hash": self.result_hash(record)})
        self._persist(record)
        log.info("gate %s on model %s (%s baseline): %s -> %s (%s)",
                 record.decision_id, model_id, policy.baseline_type.value,
                 request.candidate.state_kind.value, decision.value, reason)
        return record

    def _require_model(self, model_id: str) -> ModelRecord:
        try:
            return self.storage.load_record(model_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"model '{model_id}' not found") from None

    # ------------------------------------------------------------------ #
    # Baseline resolution
    # ------------------------------------------------------------------ #

    def _resolve_baseline(self, policy: GatePolicy, model_id: str, window: int,
                          seed: int, version: int,
                          candidates: list[EvaluationRecord]) -> EvaluationRecord:
        kind = policy.baseline_type
        if kind in (GateBaselineType.CHECKPOINT, GateBaselineType.CURRENT):
            state = ComparisonState(
                state_kind=EvalStateKind.CHECKPOINT if kind == GateBaselineType.CHECKPOINT
                else EvalStateKind.CURRENT,
                checkpoint_id=policy.baseline_checkpoint_id)
            state_hash = self.comparison.verified_state_hash(model_id, state)
            return self.comparison.resolve_evaluation(
                model_id=model_id, state=state, state_hash=state_hash,
                dataset_id=policy.dataset_id, version=version,
                split=policy.split.value, tokenizer_id=policy.tokenizer_id,
                max_eval_tokens=policy.max_eval_tokens,
                batch_size=policy.batch_size, window=window, seed=seed,
                candidates=candidates)
        # EVALUATION_RESULT_HASH: a past immutable evaluation whose result
        # hash matches AND whose full probe/state matches the policy — an
        # unrelated evaluation is never accepted as this baseline.
        found = None
        for rec in candidates:
            if rec.result_hash != policy.baseline_result_hash:
                continue
            if self._evaluation_matches_probe(rec, policy, model_id, version,
                                              window, seed):
                found = rec
                break
        if found is None:
            exists = any(r.result_hash == policy.baseline_result_hash
                         for r in candidates)
            if not exists:
                raise FileNotFoundError(
                    f"no evaluation with result hash "
                    f"'{str(policy.baseline_result_hash)[:16]}…' exists for "
                    f"model '{model_id}'")
            raise ValueError(
                "baseline evaluation result hash exists but does not match "
                "the policy probe (dataset/version/split/tokenizer/window/"
                "cap/batch/seed/state) — an unrelated evaluation cannot be "
                "used as this baseline")
        return found

    @staticmethod
    def _evaluation_matches_probe(rec: EvaluationRecord, policy: GatePolicy,
                                  model_id: str, version: int, window: int,
                                  seed: int) -> bool:
        if rec.model_id != model_id:
            return False
        if rec.dataset_id != policy.dataset_id or rec.dataset_version != version:
            return False
        if rec.split.value != policy.split.value:
            return False
        if rec.tokenizer_id != policy.tokenizer_id:
            return False
        cfg = rec.config
        if cfg.get("max_eval_tokens") != policy.max_eval_tokens:
            return False
        if cfg.get("batch_size") != policy.batch_size:
            return False
        if cfg.get("max_seq_len") != window:
            return False
        if rec.seed != seed:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Comparison reuse (an equivalent comparison is never duplicated)
    # ------------------------------------------------------------------ #

    def _find_exact_comparison(self, model_id: str, baseline_state_hash: str,
                               candidate_state_hash: str, dataset_id: str,
                               version: int, split: str, tokenizer_id: str,
                               max_eval_tokens: Optional[int], batch_size: int,
                               window: int, seed: int,
                               tolerance: float) -> Optional[ComparisonRecord]:
        # Dedupe-by-identity is a comparison-record concern: the shared
        # implementation lives on ComparisonEngine (used by M7 compare stages
        # too). Equivalent comparisons are never duplicated.
        return self.comparison.find_exact_comparison(
            model_id=model_id, state_a_hash=baseline_state_hash,
            state_b_hash=candidate_state_hash, dataset_id=dataset_id,
            version=version, split=split, tokenizer_id=tokenizer_id,
            max_eval_tokens=max_eval_tokens, batch_size=batch_size,
            window=window, seed=seed, tolerance=tolerance)

    # ------------------------------------------------------------------ #
    # Decision semantics (loss-only; explicit, documented, tested)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decide(policy: GatePolicy, candidate_loss: float,
                baseline_side: Optional[ComparisonSide],
                delta: Optional[float],
                verdict: Optional[ComparisonVerdict]
                ) -> tuple[GateDecisionResult, str]:
        threshold_ok = policy.minimum_loss is None or \
            candidate_loss <= policy.minimum_loss

        if baseline_side is None:  # absolute minimum-loss mode
            if threshold_ok:
                return (GateDecisionResult.PASSED,
                        f"candidate loss {candidate_loss:.6f} <= absolute "
                        f"threshold {policy.minimum_loss:g}")
            return (GateDecisionResult.FAILED,
                    f"candidate loss {candidate_loss:.6f} > absolute "
                    f"threshold {policy.minimum_loss:g}")

        assert verdict is not None and delta is not None
        if verdict == ComparisonVerdict.IMPROVED:
            result = (GateDecisionResult.PASSED if threshold_ok
                      else GateDecisionResult.FAILED)
            reason = (f"candidate improved by {-delta:.6f} nats "
                      f"(tolerance {policy.tolerance:g})")
        elif verdict == ComparisonVerdict.UNCHANGED:
            result = (GateDecisionResult.PASSED if threshold_ok
                      else GateDecisionResult.FAILED)
            reason = (f"candidate unchanged within tolerance "
                      f"{policy.tolerance:g} (|delta| {abs(delta):.6f})")
        else:  # regressed
            if policy.max_regression_delta is not None and \
                    delta <= policy.max_regression_delta:
                result = (GateDecisionResult.PASSED if threshold_ok
                          else GateDecisionResult.FAILED)
                reason = (f"candidate regressed by {delta:.6f} nats but within "
                          f"max_regression_delta {policy.max_regression_delta:g}")
            else:
                result = GateDecisionResult.FAILED
                reason = (f"candidate regressed by {delta:.6f} nats beyond "
                          f"tolerance {policy.tolerance:g}"
                          + (f" / max_regression_delta "
                             f"{policy.max_regression_delta:g}"
                             if policy.max_regression_delta is not None else ""))
        if not threshold_ok:
            reason += (f"; FAILS absolute minimum_loss ceiling "
                       f"({candidate_loss:.6f} > {policy.minimum_loss:g})")
        return result, reason

    @staticmethod
    def _suggestion(policy: GatePolicy,
                    baseline_side: Optional[ComparisonSide],
                    decision: GateDecisionResult
                    ) -> tuple[Optional[str], Optional[str]]:
        """Only a failed state gate against a CHECKPOINT baseline suggests
        rollback — and it is only a suggestion; nothing is executed here."""
        if decision == GateDecisionResult.FAILED and baseline_side is not None \
                and policy.baseline_type == GateBaselineType.CHECKPOINT \
                and baseline_side.checkpoint_id:
            return baseline_side.checkpoint_id, "rollback recommended"
        return None, None

    # ------------------------------------------------------------------ #
    # Side construction, result hash, persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def _side_of(rec: EvaluationRecord) -> ComparisonSide:
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

    @staticmethod
    def result_hash(record: GateDecision) -> str:
        """Deterministic hash over semantic evidence/config only.

        Excludes decision_id, created_at, duration and paths — identical gate
        requests against unchanged states reproduce it exactly.
        """
        payload = {
            "model_id": record.model_id,
            "config_hash": record.config_hash,
            "policy": record.policy.model_dump(mode="json"),
            "candidate": {
                "state_kind": record.candidate.state_kind.value,
                "checkpoint_id": record.candidate.checkpoint_id,
                "state_hash": record.candidate.state_hash,
                "evaluation_result_hash": record.candidate.evaluation_result_hash,
            },
            "baseline": None if record.baseline is None else {
                "state_kind": record.baseline.state_kind.value,
                "checkpoint_id": record.baseline.checkpoint_id,
                "state_hash": record.baseline.state_hash,
                "evaluation_result_hash": record.baseline.evaluation_result_hash,
            },
            "comparison_result_hash": record.comparison_result_hash,
            "candidate_loss": record.candidate_loss,
            "baseline_loss": record.baseline_loss,
            "delta_loss_nats": record.delta_loss_nats,
            "verdict": None if record.verdict is None else record.verdict.value,
            "decision": record.decision.value,
            "tolerance": record.tolerance,
            "max_regression_delta": record.max_regression_delta,
            "minimum_loss": record.minimum_loss,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _persist(self, record: GateDecision) -> None:
        """Write one immutable gate decision (atomic; never rewritten)."""
        gdir = self._gate_dir(record.model_id, record.decision_id)
        gdir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(gdir / GATE_MANIFEST, record.model_dump(mode="json"))
