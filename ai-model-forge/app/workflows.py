"""Workflow orchestration over the M3–M6 engines (Milestone 7).

A workflow is an ORDERED, AUDITABLE composition of existing capabilities:

    Dataset → Tokenizer → Training → Evaluation → Comparison → Gate
    → Suite Run (M10 batches; M11 stage) → …

M7 coordinates immutable operations; it never replaces an engine and never
adds training intelligence. One plan (``WorkflowPlan``) = one synchronous
execution = one immutable run manifest at
``models/<model_id>/workflows/workflow-<workflow_id>/manifest.json``.

Rules honoured here (tested):

  * stages execute in declared order, at most once; each stage references its
    inputs EXPLICITLY (literal checkpoint ids or ``from_stage`` pointers to an
    earlier train stage's final checkpoint) — nothing is ever guessed
  * branching is driven by the actual M6 ``GateDecision``: gate passed →
    next stage or ``on_pass``; gate failed → ``on_fail`` stage or STOP.
    A stopped run records the gate's rollback suggestion but NEVER executes
    rollback, retraining or model selection
  * evidence is reused by identity (M4 evaluations, M5 comparisons) exactly
    like the M6 gate engine; gate decisions and workflow runs stay
    append-only because each explicit execution is an auditable event
  * a stage that fails records its error in the run (no fabricated artifact
    ids) and the run persists with status ``failed`` referencing the earlier
    successful artifacts; the original exception is re-raised for API mapping
    (404 missing, 409 corrupt, 422 invalid)
  * ``result_hash`` is deterministic over the semantic execution only (plan,
    status, per-stage evidence hashes/results, transitions) — never
    workflow ids, timestamps or paths
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
from .gates import GateEngine
from .suite_runs import SuiteRunEngine
from .schemas import (
    ArtifactKind,
    ComparisonRequest,
    ComparisonState,
    EvalStateKind,
    EvaluationConfig,
    GateDecision,
    GateDecisionResult,
    GateRequest,
    StageStateRef,
    StageType,
    SuiteRunRequest,
    TrainingConfig,
    WorkflowArtifact,
    WorkflowGateStage,
    WorkflowPlan,
    WorkflowRecord,
    WorkflowRecipeRef,
    WorkflowStage,
    WorkflowSuiteRunStage,
    WorkflowStageResult,
    WorkflowStatus,
    WorkflowTransition,
)
from .storage import Storage, atomic_write_json, read_json
from .training import TrainingEngine

log = forge_cfg.get_logger("workflows")

WORKFLOW_MANIFEST = "manifest.json"
WORKFLOWS_DIR = "workflows"


class WorkflowEngine:
    """Ordered execution of one WorkflowPlan over the existing engines."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.datasets = DatasetEngine(storage)
        self.training = TrainingEngine(storage)
        self.evaluation = EvaluationEngine(storage)
        self.comparison = ComparisonEngine(storage)
        self.gates = GateEngine(storage)
        self.suite_runs = SuiteRunEngine(storage)
        # registry validation (M35); the local import avoids the circular
        # recipes -> workflows module dependency, and injecting ``self``
        # keeps THIS engine the sole workflow executor (no second one is
        # constructed inside the recipe engine)
        from .recipes import RecipeEngine
        self.recipes = RecipeEngine(storage, workflows=self)

    # ------------------------------------------------------------------ #
    # Paths / registry helpers
    # ------------------------------------------------------------------ #

    def _workflows_root(self, model_id: str) -> Path:
        return self.storage.model_dir(model_id) / WORKFLOWS_DIR

    def _workflow_dir(self, model_id: str, workflow_id: str) -> Path:
        return self._workflows_root(model_id) / f"workflow-{workflow_id}"

    def _model_exists(self, model_id: str) -> bool:
        return (self.storage.model_dir(model_id) / "manifest.json").exists()

    def list_workflows(self, model_id: str) -> list[WorkflowRecord]:
        """Immutable workflow history (append-only, deterministic order).

        Raises FileNotFoundError for an unknown model; returns [] when the
        model has no workflow runs yet.
        """
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        root = self._workflows_root(model_id)
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith("workflow-"):
                continue
            mpath = d / WORKFLOW_MANIFEST
            if mpath.exists():
                try:
                    records.append(WorkflowRecord(**read_json(mpath)))
                except Exception:
                    log.warning("unreadable workflow manifest %s", mpath)
        records.sort(key=lambda r: (r.created_at, r.workflow_id))
        return records

    def get_workflow(self, model_id: str, workflow_id: str) -> WorkflowRecord:
        """One persisted immutable workflow run (never mutates it)."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        path = self._workflow_dir(model_id, workflow_id) / WORKFLOW_MANIFEST
        if not path.exists():
            raise FileNotFoundError(
                f"workflow run '{workflow_id}' not found for model '{model_id}'")
        return WorkflowRecord(**read_json(path))

    def list_workflows_for_recipe(self, model_id: str,
                                  recipe_id: str) -> list[WorkflowRecord]:
        """Immutable M11 workflow runs of ONE model executed from ONE
        registered M12/M14 recipe (M35; model-scoped).

        Membership comes from the persisted run recipe identity ONLY:
        every ``WorkflowRecord`` carries a top-level ``recipe_id`` plus
        its matching ``recipe_hash`` provenance (M12 records both
        verbatim when a registered recipe is executed; ad-hoc runs
        keep ``recipe_id=None``), and a run belongs to the request when
        its persisted ``recipe_id`` equals the requested id, matched
        VERBATIM — never filenames, paths, stage ids, stage contents,
        statuses, recipe hashes or the recipe's current definition
        (recipes are immutable, but identity is still the persisted id
        alone; the recorded ``recipe_hash`` provenance is preserved
        exactly and never re-derived). Ad-hoc runs (``recipe_id=None``)
        belong to NO by-recipe group and stay in the generic M11
        listing untouched; no "ad-hoc" pseudo-recipe exists and none
        is introduced. Each run appears EXACTLY ONCE (the
        authoritative listing holds each record exactly once).
        Resolution: unknown model or unknown recipe ->
        FileNotFoundError; the recipe is validated through the existing
        GLOBAL M12/M14 registry (``RecipeEngine.get`` — the same
        resolution ``GET /workflows/recipes/{recipe_id}`` uses), while
        model scoping comes from the model's own M11 listing — a model
        never sees another model's runs (this is the model-scoped
        complement of the GLOBAL cross-model
        ``RecipeEngine.runs``/``GET /workflows/recipes/{id}/runs``
        lineage surface, which stays intact). The result keeps the
        authoritative M11 (created_at, workflow_id) ASCENDING order. A
        VALID registered recipe with no runs for this model returns
        []. Read-only, never writes, never expands or executes
        anything.
        """
        # authoritative listing validates the model: FileNotFoundError (404)
        workflows = self.list_workflows(model_id)
        # M12 registry resolution: raises FileNotFoundError when unknown
        self.recipes.get(recipe_id)
        return [w for w in workflows if w.recipe_id == recipe_id]

    def list_workflows_for_status(self, model_id: str,
                                  status: WorkflowStatus
                                  ) -> list[WorkflowRecord]:
        """Immutable M11 workflow runs of ONE model with ONE terminal
        status (M42; model-scoped).

        Membership comes from the persisted run status identity ONLY:
        every ``WorkflowRecord`` carries a top-level ``status:
        WorkflowStatus`` (the THREE-value schema enum persisted at run
        end by the M7 orchestration: completed = plan executed through
        its last stage; failed = a stage raised a missing/corrupt/
        invalid input; stopped = a gate decision failed and no on_fail
        branch was declared; a mid-flight 'running' state is never
        observable and deliberately not modelled — synchronous
        execution persists the manifest only after the run ends), and
        a run belongs to the request when its persisted ``status``
        equals the requested value, matched VERBATIM — never inferred
        from stage results, failed stage ids, workflow timestamps,
        artifact existence or recipe information, and never
        recalculated or resolved into another value (the persisted
        status is the ONLY authority; this method never re-executes
        or replays anything). Statuses have NO registry (unlike the
        M35 recipe axis): the enum IS the contract, so an unsupported
        status value is rejected at the API boundary with 422
        (schema-level validation — it never even reaches this method),
        while an unknown model raises FileNotFoundError exactly like
        the sibling grouping. Each run appears EXACTLY ONCE (the
        authoritative listing holds each record exactly once). The
        result is the model's authoritative M11 listing above (the
        exact engine parse + deterministic (created_at, workflow_id)
        ASCENDING order) filtered by the persisted status; complete
        verbatim ``WorkflowRecord`` payloads (stages, transitions,
        failed_stage_id, terminal_reason, result_hash and recipe
        provenance included), no rewritten fields. A valid status
        with no matching runs for the model returns []. Statuses are
        model-scoped through the listing itself — a model never sees
        another model's runs. Read-only, never writes.
        """
        # the authoritative listing validates the model:
        # raises FileNotFoundError (404 at the API); the status needs
        # NO registry lookup and is NEVER recalculated — the persisted
        # field is filtered verbatim
        return [w for w in self.list_workflows(model_id)
                if w.status == status]

    # ------------------------------------------------------------------ #
    # The run
    # ------------------------------------------------------------------ #

    def resolve_best_state_refs(self, plan: WorkflowPlan) -> WorkflowPlan:
        """Resolve declarative 'best' state references (M53) into CONCRETE
        checkpoint ids pinned on the plan — the ONE resolution path shared
        by inline runs, recipe runs and the M51 recipe-plan preflight.

        'best' means exactly the M52 selection: the checkpoint with the
        MINIMUM persisted validation_loss over the model's authoritative
        M3 listing (canonical (step, created_at) ASCENDING tie-break, first
        among equals, non-finite values never candidates), invoked through
        the SAME TrainingEngine.select_best_checkpoint the
        GET /checkpoints/best route uses — never a second selection
        implementation. Resolution is computed at request time against the
        checkpoints that exist at that moment (no stored pointer, no
        cross-time lock): the selection runs ONCE per plan and the concrete
        id is pinned into every unresolved 'best' StageStateRef
        (resolved_checkpoint_id), M55 best-resume train config
        (resolved_resume_checkpoint_id) and M56 best-evaluation stage
        (resolved_checkpoint_id), which then flows into the immutable run
        record and its plan_hash — executions that resolved different
        checkpoints have different execution identities. Idempotent: refs
        already carrying a pinned id keep it. A model with no selectable
        checkpoints raises FileNotFoundError BEFORE anything executes or
        persists (404 at the API; no invented state). Plans without
        'best' references are returned unchanged.
        """
        def _unresolved(ref: StageStateRef) -> bool:
            return (ref.state_kind == EvalStateKind.BEST
                    and ref.resolved_checkpoint_id is None)

        def _stage_refs(stage: WorkflowStage) -> list[StageStateRef]:
            if stage.type == StageType.SUITE_RUN and stage.suite_run is not None:
                return [stage.suite_run.state]
            if stage.type == StageType.COMPARE and stage.comparison is not None:
                return [stage.comparison.state_a, stage.comparison.state_b]
            if stage.type == StageType.GATE and stage.gate is not None:
                return [stage.gate.candidate]
            return []

        def _train_unresolved(stage: WorkflowStage) -> bool:
            # M55: a declarative best-resume TRAIN stage whose M52
            # selection has not been pinned yet (the resolver's pinned
            # form keeps resume_from_best=True + the concrete id).
            return (stage.type == StageType.TRAIN
                    and stage.training is not None
                    and stage.training.resume_from_best
                    and stage.training.resolved_resume_checkpoint_id
                    is None)

        def _evaluate_unresolved(stage: WorkflowStage) -> bool:
            # M56: a declarative best-evaluation stage whose M52
            # selection has not been pinned yet (the resolver's pinned
            # form keeps checkpoint_from_best=True + the concrete id).
            return (stage.type == StageType.EVALUATE
                    and stage.evaluation is not None
                    and stage.evaluation.checkpoint_from_best
                    and stage.evaluation.resolved_checkpoint_id is None)

        if not any(_unresolved(ref) for stage in plan.stages
                   for ref in _stage_refs(stage)) \
                and not any(_train_unresolved(stage)
                            for stage in plan.stages) \
                and not any(_evaluate_unresolved(stage)
                            for stage in plan.stages):
            return plan
        # ONE selection per plan through the M52 selector (unknown model or
        # no selectable checkpoints -> FileNotFoundError, nothing persisted)
        best = self.training.select_best_checkpoint(plan.model_id)
        ckpt_id = best.checkpoint.checkpoint_id

        def _pin(ref: StageStateRef) -> StageStateRef:
            if _unresolved(ref):
                return ref.model_copy(
                    update={"resolved_checkpoint_id": ckpt_id})
            return ref

        pinned = []
        for stage in plan.stages:
            if stage.type == StageType.SUITE_RUN and stage.suite_run is not None:
                st = stage.suite_run
                if _unresolved(st.state):
                    stage = stage.model_copy(update={
                        "suite_run": st.model_copy(update={
                            "state": _pin(st.state)})})
            elif stage.type == StageType.COMPARE and stage.comparison is not None:
                cmpst = stage.comparison
                if _unresolved(cmpst.state_a) or _unresolved(cmpst.state_b):
                    stage = stage.model_copy(update={"comparison": cmpst.model_copy(
                        update={"state_a": _pin(cmpst.state_a),
                                "state_b": _pin(cmpst.state_b)})})
            elif stage.type == StageType.GATE and stage.gate is not None:
                gt = stage.gate
                if _unresolved(gt.candidate):
                    stage = stage.model_copy(update={
                        "gate": gt.model_copy(update={
                            "candidate": _pin(gt.candidate)})})
            elif stage.type == StageType.TRAIN and stage.training is not None:
                tc = stage.training
                if _train_unresolved(stage):
                    # M55: pin the SAME single per-plan selection onto the
                    # declarative best-resume TRAIN stage (pinned form:
                    # resume_from_best=True + resolved_resume_checkpoint_
                    # id=<concrete id>) — the record and plan_hash carry
                    # the concrete resolution exactly like M53 state refs.
                    stage = stage.model_copy(update={
                        "training": tc.model_copy(update={
                            "resolved_resume_checkpoint_id": ckpt_id})})
            elif stage.type == StageType.EVALUATE \
                    and stage.evaluation is not None:
                ev = stage.evaluation
                if _evaluate_unresolved(stage):
                    # M56: pin the SAME single per-plan selection onto the
                    # declarative best-evaluation stage (pinned form:
                    # checkpoint_from_best=True + resolved_checkpoint_
                    # id=<concrete id>) — the record and plan_hash carry
                    # the concrete resolution exactly like M53 state refs
                    # and M55 best-resume train stages.
                    stage = stage.model_copy(update={
                        "evaluation": ev.model_copy(update={
                            "resolved_checkpoint_id": ckpt_id})})
            pinned.append(stage)
        # a REAL WorkflowPlan (full validation incl. _plan_consistent)
        return WorkflowPlan(name=plan.name, model_id=plan.model_id,
                            description=plan.description, stages=pinned)

    def run(self, plan: WorkflowPlan, *,
           recipe_id: Optional[str] = None,
           recipe_hash: Optional[str] = None,
           composition: Optional[list[WorkflowRecipeRef]] = None
           ) -> WorkflowRecord:
        """Execute one plan synchronously; persist one immutable run record.

        Validates nothing beyond the plan schema + model existence before
        executing (structural problems -> ValueError/FileNotFoundError with no
        artifact). A failing stage persists a ``failed`` run that references
        the earlier successful artifacts and identifies the failure point, then
        re-raises the stage exception for API mapping. Gate-driven stops are a
        NORMAL outcome: the ``stopped`` record is returned.

        ``recipe_id``/``recipe_hash`` (M12) are optional provenance for runs
        produced by a registered workflow recipe (the RecipeEngine binds the
        model into the plan and delegates here — this engine stays the SOLE
        workflow executor). ``composition`` (M14) is the deterministic
        expansion trace of the referenced recipes for composite recipe runs.
        Inline plan runs leave all three null; they never influence
        ``result_hash`` (semantic execution only). M53: any declarative
        'best' state reference is resolved to a CONCRETE checkpoint id
        (the M52 selection, one call per plan) BEFORE execution — the
        resolved plan is what executes, persists and hashes, so the run
        record always pins the concrete checkpoint it used.
        """
        plan = self.resolve_best_state_refs(plan)
        start = time.monotonic()
        model = self._require_model(plan.model_id)
        model_id = plan.model_id
        n = len(plan.stages)
        stage_ids = [s.stage_id for s in plan.stages]
        index = {sid: i for i, sid in enumerate(stage_ids)}

        results: list[WorkflowStageResult] = [
            WorkflowStageResult(stage_id=s.stage_id, type=s.type)
            for s in plan.stages]
        transitions: list[WorkflowTransition] = []
        artifacts: dict[str, WorkflowArtifact] = {}
        status: WorkflowStatus = WorkflowStatus.COMPLETED
        terminal_reason: Optional[str] = None
        failed_stage_id: Optional[str] = None
        suggested: Optional[str] = None
        hint: Optional[str] = None

        idx = 0
        while idx < n:
            stage = plan.stages[idx]
            try:
                artifact, branch = self._execute_stage(model, stage, artifacts)
            except Exception as exc:  # stage failed: record + persist + re-raise
                results[idx].executed = True
                results[idx].error = str(exc)
                for j in range(idx + 1, n):
                    results[j].skipped = True
                status = WorkflowStatus.FAILED
                failed_stage_id = stage.stage_id
                terminal_reason = f"stage '{stage.stage_id}' failed: {exc}"
                record = self._record(plan, model_id, model.config_hash,
                                      status, terminal_reason, failed_stage_id,
                                      results, transitions, suggested, hint,
                                      start, recipe_id, recipe_hash,
                                      composition)
                self._persist(record)
                log.error("workflow run for model %s failed at stage %s: %s",
                          model_id, stage.stage_id, exc)
                raise
            results[idx].executed = True
            results[idx].artifact = artifact
            artifacts[stage.stage_id] = artifact

            if stage.type != StageType.GATE:
                to_stage = stage_ids[idx + 1] if idx + 1 < n else None
                transitions.append(WorkflowTransition(
                    stage_id=stage.stage_id, decision="next", to_stage=to_stage))
                idx += 1
                continue

            decision: GateDecisionResult = (artifact.gate_decision
                                            if artifact.gate_decision is not None
                                            else GateDecisionResult.PASSED)
            passed = decision == GateDecisionResult.PASSED
            branch_target = stage.on_pass if passed else stage.on_fail
            if passed:
                transitions.append(WorkflowTransition(
                    stage_id=stage.stage_id, decision="passed",
                    to_stage=branch_target if branch_target else
                    (stage_ids[idx + 1] if idx + 1 < n else None)))
                if branch_target:
                    jump = index[branch_target]
                    for j in range(idx + 1, jump):
                        results[j].skipped = True
                    idx = jump
                else:
                    idx += 1
                continue

            # gate failed
            if stage.on_fail is None:
                status = WorkflowStatus.STOPPED
                gate_record = self.gates.get_decision(model_id,
                                                      artifact.artifact_id)
                suggested = gate_record.suggested_checkpoint_id
                hint = gate_record.hint
                terminal_reason = (
                    f"gate '{stage.stage_id}' decision failed; workflow stopped "
                    f"(no on_fail branch)")
                transitions.append(WorkflowTransition(
                    stage_id=stage.stage_id, decision="failed", to_stage=None))
                for j in range(idx + 1, n):
                    results[j].skipped = True
                break
            transitions.append(WorkflowTransition(
                stage_id=stage.stage_id, decision="failed",
                to_stage=stage.on_fail))
            jump = index[stage.on_fail]
            for j in range(idx + 1, jump):
                results[j].skipped = True
            idx = jump

        record = self._record(plan, model_id, model.config_hash, status,
                              terminal_reason, failed_stage_id, results,
                              transitions, suggested, hint, start,
                              recipe_id, recipe_hash, composition)
        self._persist(record)
        log.info("workflow %s on model %s: %s (%d stages)",
                 record.workflow_id, model_id, status.value, n)
        return record

    # ------------------------------------------------------------------ #
    # Per-stage execution (thin orchestration over M3–M6)
    # ------------------------------------------------------------------ #

    def _execute_stage(self, model, stage: WorkflowStage,
                       artifacts: dict[str, WorkflowArtifact]
                       ) -> tuple[WorkflowArtifact, Optional[str]]:
        """Run one stage through its existing engine.

        Returns (artifact summary, branch decision) where branch is only set
        for gate stages ('passed'/'failed').
        """
        model_id = model.id
        if stage.type == StageType.TRAIN:
            cfg: TrainingConfig = stage.training  # type: ignore[assignment]
            if cfg.resume_from_best:
                # M55: the resolver already pinned the M52 selection's
                # concrete checkpoint id. Hand the training engine a PURE
                # M54 explicit-resume config — the workflow layer selects,
                # the training layer receives the explicit id (separation
                # of concerns; no dynamic "best" query inside training).
                # The declarative trace stays in the run record's plan.
                cfg = cfg.model_copy(update={
                    "resume_from_checkpoint_id":
                        cfg.resolved_resume_checkpoint_id,
                    "resume_from_best": False,
                    "resolved_resume_checkpoint_id": None})
            report = self.training.run(cfg)
            summary = self._training_artifact(model_id, report)
            return summary, None

        if stage.type == StageType.EVALUATE:
            payload = stage.evaluation  # type: ignore[assignment]
            cfg: EvaluationConfig = payload.config.model_copy()
            if payload.checkpoint_from_best:
                # M56: the resolver already pinned the M52 selection's
                # concrete checkpoint id. Hand the M4 evaluation path a
                # PURE explicit-checkpoint config (the same conversion
                # pattern as M55 train stages: the workflow layer selects,
                # the evaluation layer receives the explicit id — no
                # dynamic "best" query, no silent fallback to current).
                # The declarative trace (checkpoint_from_best=True + the
                # pin) stays in the run record's plan.
                if payload.resolved_checkpoint_id is None:
                    raise ValueError(
                        "checkpoint_from_best reached execution without a "
                        "pinned M52 selection — the workflow resolver must "
                        "run first (no silent fallback to current weights)")
                cfg = cfg.model_copy(update={
                    "checkpoint_id": payload.resolved_checkpoint_id})
            if payload.checkpoint_from_stage:
                ckpt_id = self._final_checkpoint_of(
                    payload.checkpoint_from_stage, artifacts)
                cfg = cfg.model_copy(update={"checkpoint_id": ckpt_id})
            state = ComparisonState(
                state_kind=EvalStateKind.CHECKPOINT if cfg.checkpoint_id
                else EvalStateKind.CURRENT,
                checkpoint_id=cfg.checkpoint_id)
            state_hash = self.comparison.verified_state_hash(model_id, state)
            version = self._resolve_version(cfg.dataset_id, cfg.dataset_version)
            window = (cfg.max_seq_len if cfg.max_seq_len is not None
                      else model.config.context_length)
            seed = cfg.effective_seed()
            candidates = self.evaluation.list_evaluations(model_id)
            rec = self.comparison.resolve_evaluation(
                model_id=model_id, state=state, state_hash=state_hash,
                dataset_id=cfg.dataset_id, version=version,
                split=cfg.split.value, tokenizer_id=cfg.tokenizer_id,
                max_eval_tokens=cfg.max_eval_tokens, batch_size=cfg.batch_size,
                window=window, seed=seed, candidates=candidates)
            return (WorkflowArtifact(
                kind=ArtifactKind.EVALUATION, artifact_id=rec.eval_id,
                result_hash=rec.result_hash, state_hash=rec.state_hash,
                checkpoint_id=cfg.checkpoint_id, loss_nats=rec.loss_nats), None)

        if stage.type == StageType.COMPARE:
            payload_cmp = stage.comparison  # type: ignore[assignment]
            state_a = self._state_of(payload_cmp.state_a, artifacts)
            state_b = self._state_of(payload_cmp.state_b, artifacts)
            # both states verified BEFORE anything is evaluated (M5 order)
            hash_a = self.comparison.verified_state_hash(model_id, state_a)
            hash_b = self.comparison.verified_state_hash(model_id, state_b)
            version = self._resolve_version(payload_cmp.dataset_id,
                                            payload_cmp.dataset_version)
            window = (payload_cmp.max_seq_len
                      if payload_cmp.max_seq_len is not None
                      else model.config.context_length)
            # seed parity with the M5 engine: derive from the request exactly
            # as M5's run() would (states literal, raw probe fields)
            request = ComparisonRequest(
                model_id=model_id, state_a=state_a, state_b=state_b,
                dataset_id=payload_cmp.dataset_id,
                dataset_version=payload_cmp.dataset_version,
                split=payload_cmp.split, tokenizer_id=payload_cmp.tokenizer_id,
                max_eval_tokens=payload_cmp.max_eval_tokens,
                batch_size=payload_cmp.batch_size,
                max_seq_len=payload_cmp.max_seq_len,
                seed=payload_cmp.seed, tolerance=payload_cmp.tolerance)
            seed = request.effective_seed()
            candidates = self.evaluation.list_evaluations(model_id)
            eval_a = self.comparison.resolve_evaluation(
                model_id=model_id, state=state_a, state_hash=hash_a,
                dataset_id=payload_cmp.dataset_id, version=version,
                split=payload_cmp.split.value,
                tokenizer_id=payload_cmp.tokenizer_id,
                max_eval_tokens=payload_cmp.max_eval_tokens,
                batch_size=payload_cmp.batch_size, window=window, seed=seed,
                candidates=candidates)
            candidates.append(eval_a)
            eval_b = self.comparison.resolve_evaluation(
                model_id=model_id, state=state_b, state_hash=hash_b,
                dataset_id=payload_cmp.dataset_id, version=version,
                split=payload_cmp.split.value,
                tokenizer_id=payload_cmp.tokenizer_id,
                max_eval_tokens=payload_cmp.max_eval_tokens,
                batch_size=payload_cmp.batch_size, window=window, seed=seed,
                candidates=candidates)
            existing = self.comparison.find_exact_comparison(
                model_id=model_id, state_a_hash=hash_a, state_b_hash=hash_b,
                dataset_id=payload_cmp.dataset_id, version=version,
                split=payload_cmp.split.value,
                tokenizer_id=payload_cmp.tokenizer_id,
                max_eval_tokens=payload_cmp.max_eval_tokens,
                batch_size=payload_cmp.batch_size, window=window, seed=seed,
                tolerance=payload_cmp.tolerance)
            if existing is not None:
                record = existing
            else:
                record = self.comparison.compare_records(
                    eval_a, eval_b, tolerance=payload_cmp.tolerance,
                    started_at=None)
            return (WorkflowArtifact(
                kind=ArtifactKind.COMPARISON,
                artifact_id=record.comparison_id,
                result_hash=record.result_hash,
                delta_loss_nats=record.delta_loss_nats,
                verdict=record.verdict), None)

        # SUITE RUN (M11): a thin orchestration adapter over the M10 engine.
        # The stage names ONE suite and ONE explicit state source; the state
        # is resolved through the same StageStateRef provenance rules as M7
        # compare/gate stages (from_stage -> that train stage's final
        # checkpoint; nothing is ever guessed). SuiteRunEngine.run persists
        # the immutable suite-run record and reuses exact M4 evaluations;
        # preflight failures (unknown suite/state/corrupt suite) raise here
        # so the M7 failure path records the stage error and no fabricated
        # suite-run artifact ever exists. A suite run whose own record ended
        # with status 'failed' (per-probe failures) is a REAL persisted
        # artifact and is referenced as such — never converted to success.
        if stage.type == StageType.SUITE_RUN:
            payload_sr: WorkflowSuiteRunStage = stage.suite_run  # type: ignore[assignment]
            state = self._state_of(payload_sr.state, artifacts)
            record = self.suite_runs.run(SuiteRunRequest(
                model_id=model_id, suite_id=payload_sr.suite_id, state=state))
            return (WorkflowArtifact(
                kind=ArtifactKind.SUITE_RUN,
                artifact_id=record.suite_run_id,
                result_hash=record.result_hash,
                state_hash=record.state_hash,
                checkpoint_id=state.checkpoint_id), None)

        # GATE: M6 is the single source of truth for decision semantics.
        # The stage carries either an inline policy or a registry policy_id —
        # GateEngine.run resolves both through PolicyEngine (the inline path
        # stays byte-identical to historical M6 behaviour).
        payload_gate: WorkflowGateStage = stage.gate  # type: ignore[assignment]
        candidate = self._state_of(payload_gate.candidate, artifacts)
        decision_record = self.gates.run(GateRequest(
            model_id=model_id, policy=payload_gate.policy,
            policy_id=payload_gate.policy_id, candidate=candidate))
        return (self._gate_artifact(decision_record),
                decision_record.decision.value)

    @staticmethod
    def _state_of(ref: StageStateRef,
                  artifacts: dict[str, WorkflowArtifact]) -> ComparisonState:
        """Resolve an explicit state source to a ComparisonState.

        A checkpoint ``from_stage`` pointer resolves to that train stage's
        FINAL checkpoint; a stage that produced none raises FileNotFoundError
        (the workflow fails cleanly; no id is fabricated). A ``best``
        reference (M53) uses the concrete checkpoint id pinned by the
        resolver — downstream engines always receive a literal state.
        """
        if ref.state_kind == EvalStateKind.CURRENT:
            return ComparisonState(state_kind=EvalStateKind.CURRENT)
        if ref.state_kind == EvalStateKind.BEST:
            # M53: the resolver already pinned the M52 selection's concrete
            # checkpoint id; downstream engines see a NORMAL explicit
            # checkpoint state (never the moving 'best' label), so their
            # immutable records stay byte-identical in shape to literal-id
            # runs. An unresolved 'best' can only mean a plan that bypassed
            # WorkflowEngine.run/resolution — nothing is invented.
            if not ref.resolved_checkpoint_id:
                raise FileNotFoundError(
                    "state_kind='best' is not resolved — the plan must go "
                    "through the workflow engine's resolver (M52 selection "
                    "pins the concrete checkpoint) before execution")
            return ComparisonState(
                state_kind=EvalStateKind.CHECKPOINT,
                checkpoint_id=ref.resolved_checkpoint_id)
        if ref.checkpoint_id:
            return ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                   checkpoint_id=ref.checkpoint_id)
        artifact = artifacts.get(ref.from_stage) if ref.from_stage else None
        if artifact is None or artifact.kind != ArtifactKind.TRAINING_REPORT \
                or not artifact.checkpoint_id:
            raise FileNotFoundError(
                f"stage '{ref.from_stage}' produced no final checkpoint — "
                f"nothing to reference")
        return ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                               checkpoint_id=artifact.checkpoint_id)

    @staticmethod
    def _final_checkpoint_of(stage_id: str,
                             artifacts: dict[str, WorkflowArtifact]) -> str:
        artifact = artifacts.get(stage_id)
        if artifact is None or artifact.kind != ArtifactKind.TRAINING_REPORT \
                or not artifact.checkpoint_id:
            raise FileNotFoundError(
                f"stage '{stage_id}' produced no final checkpoint — nothing "
                f"to reference")
        return artifact.checkpoint_id

    def _resolve_version(self, dataset_id: str,
                         version: Optional[int]) -> int:
        try:
            meta = self.datasets.load_meta(dataset_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"dataset '{dataset_id}' not found") from None
        return meta.latest_version if version is None else version

    def _training_artifact(self, model_id: str, report) -> WorkflowArtifact:
        """Summarise an M3 TrainingReport (id + outcome + final checkpoint)."""
        summary = WorkflowArtifact(
            kind=ArtifactKind.TRAINING_REPORT, artifact_id=report.run_id,
            checkpoint_id=report.final_model_version,
            accepted=report.accepted, optimizer_steps=report.optimizer_steps,
            epochs_run=report.epochs_run,
            best_validation_loss=report.best_validation_loss,
            final_train_loss=report.final_train_loss)
        if report.final_model_version:
            try:
                ckpt = self.training.get_checkpoint(
                    model_id, report.final_model_version)
                summary.state_hash = ckpt.weights_sha256
                summary.final_validation_loss = ckpt.validation_loss
            except Exception:  # pragma: no cover - report/manifest desync
                log.warning("final checkpoint %s unreadable after training",
                            report.final_model_version)
        return summary

    @staticmethod
    def _gate_artifact(record: GateDecision) -> WorkflowArtifact:
        return WorkflowArtifact(
            kind=ArtifactKind.GATE_DECISION,
            artifact_id=record.decision_id,
            result_hash=record.result_hash,
            state_hash=record.candidate.state_hash,
            checkpoint_id=record.candidate.checkpoint_id,
            loss_nats=record.candidate_loss,
            delta_loss_nats=record.delta_loss_nats,
            verdict=record.verdict,
            gate_decision=record.decision)

    # ------------------------------------------------------------------ #
    # Record assembly, result hash, persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record(plan: WorkflowPlan, model_id: str, config_hash: str,
                status: WorkflowStatus, terminal_reason: Optional[str],
                failed_stage_id: Optional[str],
                results: list[WorkflowStageResult],
                transitions: list[WorkflowTransition],
                suggested: Optional[str], hint: Optional[str],
                start: float,
                recipe_id: Optional[str] = None,
                recipe_hash: Optional[str] = None,
                composition: Optional[list[WorkflowRecipeRef]] = None
                ) -> WorkflowRecord:
        record = WorkflowRecord(
            workflow_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            config_hash=config_hash,
            name=plan.name,
            plan_hash=plan.plan_hash(),
            plan=plan,
            status=status,
            terminal_reason=terminal_reason,
            failed_stage_id=failed_stage_id,
            stages=results,
            transitions=transitions,
            suggested_checkpoint_id=suggested,
            hint=hint,
            recipe_id=recipe_id,
            recipe_hash=recipe_hash,
            composition=composition,
            result_hash="",
            created_at=datetime.now(timezone.utc),
            duration_seconds=round(time.monotonic() - start, 3),
        )
        return record.model_copy(update={"result_hash": WorkflowEngine.result_hash(record)})

    @staticmethod
    def result_hash(record: WorkflowRecord) -> str:
        """Deterministic hash over the semantic execution only.

        Includes the plan, terminal status, per-stage evidence (kind + the
        underlying artifacts' deterministic hashes/losses/verdicts/decisions —
        never their random ids), executed/skipped flags and the transition
        decisions. Excludes workflow_id, created_at, duration, paths and all
        recipe provenance (recipe_id/recipe_hash/composition — M12/M14
        bookkeeping, not semantics), so identical workflows over identical
        immutable inputs reproduce it.
        """
        def semantic(artifact: Optional[WorkflowArtifact]):
            if artifact is None:
                return None
            # JSON form (enum -> value), minus random artifact/checkpoint ids:
            # content hashes and results remain, ids do not.
            dump = artifact.model_dump(mode="json", exclude={
                "artifact_id", "checkpoint_id"})
            return {k: v for k, v in dump.items() if v is not None}

        payload = {
            "model_id": record.model_id,
            "config_hash": record.config_hash,
            "plan": record.plan.model_dump(mode="json"),
            "status": record.status.value,
            "failed_stage_id": record.failed_stage_id,
            "stages": [{
                "stage_id": s.stage_id,
                "type": s.type.value,
                "executed": s.executed,
                "skipped": s.skipped,
                "artifact": semantic(s.artifact),
            } for s in record.stages],
            "transitions": [t.model_dump(mode="json") for t in record.transitions],
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _persist(self, record: WorkflowRecord) -> None:
        """Write one immutable workflow run (atomic; never rewritten)."""
        wdir = self._workflow_dir(record.model_id, record.workflow_id)
        wdir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(wdir / WORKFLOW_MANIFEST,
                          record.model_dump(mode="json"))

    def _require_model(self, model_id: str):
        try:
            return self.storage.load_record(model_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"model '{model_id}' not found") from None
