"""The forge facade: single entry point the API (and later the orchestrator) uses.

Keeps storage + model builder glued together in one small module instead of
scattering creation logic across routes. No business logic lives here that
belongs in storage or the builder — only composition.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional


from . import config as forge_cfg
from .comparison import ComparisonEngine
from .dashboards import DashboardEngine
from .dataset import DatasetEngine
from .evaluation import EvaluationEngine
from .gates import GateEngine
from .hardware import HardwareSpec, detect_hardware
from .model_builder import build_transformer, content_hash
from .policies import PolicyEngine
from .recipes import RecipeEngine
from .sampling import SamplingEngine
from .sample_quality import SampleQualityEngine
from .schemas import (
    ComparisonRecord,
    ComparisonVerdict,
    EvalStateKind,
    EvaluationRecord,
    EvaluationSplit,
    GateBaselineType,
    GateDecisionResult,
    ModelCreateRequest,
    ModelRecord,
    PolicyCreateRequest,
    PolicyDefinition,
    ProbeSuite,
    ProbeSuiteCreateRequest,
    SuiteRunRecord,
    SuiteRunRequest,
    SuiteRunSummary,
    WorkflowRecord,
    WorkflowRecipe,
    WorkflowRecipeCreateRequest,
    WorkflowStatus,
    SampleGenerateRequest,
    SampleEvaluationRecord,
    SampleRecord,
    SampleStrategy,
    TokenizerConfig,
    TransformerConfig,
    utcnow,
)
from .storage import Storage
from .suite_runs import SuiteRunEngine
from .tokenizer import TokenizerEngine
from .training import TrainingEngine
from .workflows import WorkflowEngine

log = forge_cfg.get_logger("forge")


def _config_hash(cfg: TransformerConfig) -> str:
    """Hash over the exact JSON form used inside manifests."""
    import hashlib
    import json

    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class ModelForge:
    def __init__(self, root: Optional[Path] = None):
        root = root or forge_cfg.forge_root()
        self.storage = Storage(root)
        self.storage.initialize()
        self.datasets = DatasetEngine(self.storage)
        self.tokenizers = TokenizerEngine(self.storage)
        self.training = TrainingEngine(self.storage)
        self.evaluation = EvaluationEngine(self.storage)
        self.comparison = ComparisonEngine(self.storage)
        self.gates = GateEngine(self.storage)
        self.workflows = WorkflowEngine(self.storage)
        self.dashboards = DashboardEngine(self.storage)
        self.policies = PolicyEngine(self.storage)
        self.suite_runs = SuiteRunEngine(self.storage)
        self.recipes = RecipeEngine(self.storage, workflows=self.workflows)
        self.samples = SamplingEngine(self.storage)
        self.sample_quality = SampleQualityEngine(self.storage)
        self._hardware: Optional[HardwareSpec] = None
        log.info("forge storage ready at %s", root)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def project_info(self) -> dict[str, Any]:
        info = self.storage.project_info()
        info["model_count"] = len(self.list_models())
        info["dataset_count"] = len(self.datasets.list())
        info["tokenizer_count"] = len(self.tokenizers.list())
        info["app_version"] = forge_cfg.APP_VERSION
        info["schema_version"] = forge_cfg.MANIFEST_SCHEMA_VERSION
        info["storage_root"] = str(self.storage.root)
        return info

    def hardware(self) -> dict[str, Any]:
        if self._hardware is None:
            self._hardware = detect_hardware()
        return self._hardware.to_dict()

    def storage_usage(self) -> dict[str, Any]:
        return self.storage.usage()

    # ------------------------------------------------------------------ #
    # Model registry
    # ------------------------------------------------------------------ #

    def create_model(self, request: ModelCreateRequest) -> tuple[ModelRecord, dict[str, Any]]:
        """Create, persist and verify a fresh model. Returns (record, extra info)."""
        cfg = request.config
        model_id = uuid.uuid4().hex[:12]
        created = utcnow()

        # Prevent accidental duplicate names in one project (ids stay unique).
        for existing in self.list_models():
            if existing.name == cfg.name:
                raise ValueError(
                    f"a model named '{cfg.name}' already exists (id {existing.id}); "
                    "choose a unique name"
                )

        try:
            built = build_transformer(cfg, device="cpu")
        except Exception:
            log.exception("model build failed for '%s'", cfg.name)
            raise

        state = built.state_dict()
        weights_path = self.storage.write_weights(model_id, state)

        record = ModelRecord(
            id=model_id,
            name=cfg.name,
            description=request.description,
            metadata=request.metadata,
            architecture=cfg.architecture,
            created_at=created,
            updated_at=created,
            builder_version=forge_cfg.APP_VERSION,
            manifest_schema_version=forge_cfg.MANIFEST_SCHEMA_VERSION,
            config=cfg,
            config_hash=_config_hash(cfg),
            parameter_count=built.parameter_count,
            weight_bytes=weights_path.stat().st_size,
            dtype=built.dtype,
            initializer=cfg.weight_init.value,
            seed=built.seed,
            state_hash=content_hash(state),
            hardware=self.hardware(),
            param_rows=built.param_rows,
        )
        self.storage.save_record(record)

        # Round-trip verification: what we wrote must be exactly readable.
        loaded = self.storage.load_record(model_id)
        self.storage.load_weights(model_id)  # integrity sidecar check
        if loaded.config != cfg or loaded.state_hash != record.state_hash:
            raise RuntimeError(f"persistence round-trip failed for model {model_id}")

        log.info(
            "created model id=%s name=%r params=%s dtype=%s weights=%sB",
            model_id, cfg.name, f"{built.parameter_count:,}", built.dtype,
            f"{record.weight_bytes:,}",
        )
        extra = {
            "weights_bytes_on_disk": record.weight_bytes,
            "storage_bytes_total": self.storage.usage()["bytes"],
            "hardware": self.hardware(),
        }
        return record, extra

    def get_model(self, model_id: str) -> ModelRecord:
        return self.storage.load_record(model_id)

    def list_models(self) -> list[ModelRecord]:
        records = self.storage.load_all()
        records.sort(key=lambda r: r.created_at)
        return records

    def delete_model(self, model_id: str) -> None:
        self.storage.remove(model_id)
        log.info("deleted model %s", model_id)

    def verify_model(self, model_id: str) -> dict[str, Any]:
        """Full integrity check: manifest parse, state reload, hash sidecar."""
        self.storage.verify_integrity(model_id)
        return {"id": model_id, "integrity": "ok"}

    def weights_archive_path(self, model_id: str) -> Path:
        return self.storage.weights_path(model_id)

    # ------------------------------------------------------------------ #
    # Datasets (thin delegation to the data engine)
    # ------------------------------------------------------------------ #

    def upload_dataset(self, files: list[tuple[str, bytes]], dataset_id: Optional[str] = None,
                       name: Optional[str] = None) -> dict[str, Any]:
        return self.datasets.create_or_append(files, dataset_id=dataset_id, name=name)

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> list[dict[str, Any]]:
        return [d.model_dump(mode="json") for d in self.datasets.list()]

    def verify_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.datasets.verify(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        self.datasets.delete(dataset_id)

    def tokenize_dataset(self, dataset_id: str, tokenizer_id: str,
                         version: Optional[int] = None) -> dict[str, Any]:
        return self.datasets.tokenize(dataset_id, tokenizer_id, version=version)

    # ------------------------------------------------------------------ #
    # Tokenizers (thin delegation to the tokenizer engine)
    # ------------------------------------------------------------------ #

    def train_tokenizer(self, config: TokenizerConfig,
                        files: Optional[list[tuple[str, bytes]]] = None,
                        dataset_id: Optional[str] = None):
        """Train on uploaded files and/or a stored dataset (single source)."""
        from .dataset import extract_records_for_training

        if dataset_id and files:
            raise ValueError("provide either uploaded files OR a dataset_id, not both")
        if not dataset_id and not files:
            raise ValueError("provide uploaded files or a dataset_id to train on")

        if dataset_id:
            meta = self.datasets.load_meta(dataset_id)
            records = [row["text"] for row in
                       self.datasets.iter_version_records(dataset_id, meta.latest_version)]
        else:
            records = extract_records_for_training(files or [])
        if not records:
            raise ValueError("no usable text records to train on")
        return self.tokenizers.train(config, records, trained_on_dataset_id=dataset_id)

    def get_tokenizer(self, tokenizer_id: str):
        return self.tokenizers.load(tokenizer_id)

    def list_tokenizers(self) -> list[dict[str, Any]]:
        return [t.model_dump(mode="json") for t in self.tokenizers.list()]

    def delete_tokenizer(self, tokenizer_id: str) -> None:
        self.tokenizers.delete(tokenizer_id)

    # ------------------------------------------------------------------ #
    # Training (thin delegation to the training engine)
    # ------------------------------------------------------------------ #

    def run_training(self, cfg):
        return self.training.run(cfg)

    def list_checkpoints(self, model_id: str):
        return self.training.list_checkpoints(model_id)

    def get_checkpoint(self, model_id: str, checkpoint_id: str):
        return self.training.get_checkpoint(model_id, checkpoint_id)

    def list_checkpoints_for_run(self, model_id: str, run_id: str):
        return self.training.list_checkpoints_for_run(model_id, run_id)

    def rollback_model(self, model_id: str, checkpoint_id: str):
        return self.training.rollback(model_id, checkpoint_id)

    # ------------------------------------------------------------------ #
    # Evaluation (thin delegation to the evaluation engine; read-only)
    # ------------------------------------------------------------------ #

    def run_evaluation(self, cfg):
        return self.evaluation.run(cfg)

    def list_evaluations(self, model_id: str):
        return self.evaluation.list_evaluations(model_id)

    def get_evaluation(self, model_id: str, eval_id: str):
        return self.evaluation.get_evaluation(model_id, eval_id)

    def list_evaluations_for_checkpoint(self, model_id: str,
                                        checkpoint_id: str):
        """Immutable M4 evaluation records recorded under ONE checkpoint
        (M24 read-only access).

        The checkpoint must be registered in the model's M3 checkpoint
        registry (unknown checkpoint, or a checkpoint id belonging to
        another model -> FileNotFoundError -> 404). Returns the model's
        authoritative M4 listing filtered by the persisted checkpoint_id
        recorded in each record — verbatim records in the M4
        (created_at, eval_id) order; current-state evaluations
        (checkpoint_id None) never appear; a checkpoint without
        evaluations returns []. Read-only, never writes."""
        return self.evaluation.list_evaluations_for_checkpoint(
            model_id, checkpoint_id)

    def list_evaluations_for_dataset(self, model_id: str,
                                     dataset_id: str
                                     ) -> list[EvaluationRecord]:
        """Immutable M4 evaluation records of ONE model over ONE dataset
        (M28 read-only access).

        The dataset must exist in the M2 registry (unknown dataset ->
        FileNotFoundError -> 404; datasets are global — model scoping
        comes from the model's own M4 listing). Returns the model's
        authoritative M4 listing filtered by the persisted dataset
        identity (top-level dataset_id; the persisted dataset_version
        travels VERBATIM inside each record — all versions returned,
        never collapsed/resolved/rewritten) in the M4 (created_at,
        eval_id) order. A valid dataset with no evaluations for the
        model returns []. Read-only, never writes."""
        return self.evaluation.list_evaluations_for_dataset(
            model_id, dataset_id)

    def list_evaluations_for_tokenizer(self, model_id: str,
                                       tokenizer_id: str
                                       ) -> list[EvaluationRecord]:
        """Immutable M4 evaluation records of ONE model measured with
        ONE tokenizer (M30 read-only access).

        The tokenizer must exist in the tokenizer registry (unknown
        tokenizer -> FileNotFoundError -> 404; tokenizers are global —
        model scoping comes from the model's own M4 listing). Returns
        the model's authoritative M4 listing filtered by the persisted
        tokenizer identity (top-level tokenizer_id, matched VERBATIM —
        never inferred from filenames or substituted with the latest
        tokenizer) in the M4 (created_at, eval_id) order, as complete
        verbatim records. A valid tokenizer with no evaluations for
        the model returns []. Read-only, never writes."""
        return self.evaluation.list_evaluations_for_tokenizer(
            model_id, tokenizer_id)

    def list_evaluations_for_split(self, model_id: str,
                                   split: EvaluationSplit
                                   ) -> list[EvaluationRecord]:
        """Immutable M4 evaluation records of ONE model measured on
        ONE dataset split (M36 read-only access).

        The split is the persisted schema enum (train/validation/test)
        — there is NO split registry, so an unsupported value is
        rejected at the API boundary with 422, while an unknown model
        raises FileNotFoundError (404) exactly like the sibling
        groupings. Returns the model's authoritative M4 listing
        filtered by the persisted split (top-level split, matched
        VERBATIM — never inferred from filenames, datasets or
        timestamps, never rewritten) in the M4 (created_at, eval_id)
        order, as complete verbatim records. A valid split with no
        evaluations for the model returns []. Read-only, never
        writes."""
        return self.evaluation.list_evaluations_for_split(model_id, split)

    def list_evaluations_for_state_kind(self, model_id: str,
                                        state_kind: EvalStateKind
                                        ) -> list[EvaluationRecord]:
        """Immutable M4 evaluation records of ONE model measuring ONE
        kind of model state (M38 read-only access).

        The state kind is the persisted schema enum
        (current/checkpoint) — there is NO state-kind registry, so an
        unsupported value is rejected at the API boundary with 422,
        while an unknown model raises FileNotFoundError (404) exactly
        like the sibling groupings. Returns the model's authoritative
        M4 listing filtered by the persisted top-level state_kind
        (matched VERBATIM — never inferred from checkpoint_id
        nullability, filenames or timestamps, never rewritten) in the
        M4 (created_at, eval_id) order, as complete verbatim records.
        A valid state kind with no evaluations for the model returns
        []. Read-only, never writes."""
        return self.evaluation.list_evaluations_for_state_kind(
            model_id, state_kind)

    def list_evaluations_for_truncated(self, model_id: str,
                                       truncated: bool
                                       ) -> list[EvaluationRecord]:
        """Immutable M4 evaluation records of ONE model by persisted
        truncation status (M47 read-only access).

        ``truncated`` is the persisted REQUIRED boolean the engine
        recorded at run time (True = ``max_eval_tokens`` stopped the
        evaluation before the split ended) — a closed two-value
        contract with NO registry, so an unsupported spelling is
        rejected at the API boundary with 422, while an unknown model
        raises FileNotFoundError (404) exactly like the sibling
        groupings. Returns the model's authoritative M4 listing
        filtered by the persisted boolean (matched VERBATIM — never
        recalculated, never derived from records_covered, token
        counts, config or timestamps) in the M4 (created_at, eval_id)
        order, as complete verbatim records. A model with no
        evaluations of one status returns []. Read-only, never
        writes."""
        return self.evaluation.list_evaluations_for_truncated(
            model_id, truncated)

    def list_evaluations_for_seed(self, model_id: str,
                                  seed: int) -> list[EvaluationRecord]:
        """Immutable M4 evaluation records of ONE model by persisted
        effective seed (M48 read-only access).

        ``seed`` is the REQUIRED integer persisted on every record
        (the effective seed used, default derived from the config) —
        an OPEN value axis with NO registry, so a non-integer
        spelling is rejected at the API boundary with 422, while an
        unknown model raises FileNotFoundError (404) exactly like the
        sibling groupings. Returns the model's authoritative M4
        listing filtered by the persisted record value (matched
        VERBATIM — never recalculated, never normalized, never
        derived from the config) in the M4 (created_at, eval_id)
        order, as complete verbatim records. An unmatched seed on a
        valid model returns []. Read-only, never writes."""
        return self.evaluation.list_evaluations_for_seed(model_id, seed)

    # ------------------------------------------------------------------ #
    # Comparison (thin delegation; read-only orchestration over evaluations)
    # ------------------------------------------------------------------ #

    def run_comparison(self, request):
        return self.comparison.run(request)

    def list_comparisons(self, model_id: str):
        return self.comparison.list_comparisons(model_id)

    def get_comparison(self, model_id: str, comparison_id: str):
        return self.comparison.get_comparison(model_id, comparison_id)

    def list_comparisons_for_checkpoint(self, model_id: str,
                                        checkpoint_id: str
                                        ) -> list[ComparisonRecord]:
        """Immutable M5 comparison records involving ONE checkpoint (M26
        read-only access).

        The checkpoint must be registered in the model's M3 checkpoint
        registry (unknown checkpoint, or a checkpoint id belonging to
        another model -> FileNotFoundError -> 404). Returns the model's
        authoritative M5 listing filtered by the persisted side states
        — a comparison is included when EITHER side records
        state_kind "checkpoint" with the requested checkpoint_id
        (current-state sides never match); each record appears exactly
        once even when both sides match (unique comparison identities)
        in the M5 (created_at, comparison_id) order. A checkpoint with
        no matching comparisons returns []. Read-only, never writes."""
        return self.comparison.list_comparisons_for_checkpoint(
            model_id, checkpoint_id)

    def list_comparisons_for_dataset(self, model_id: str,
                                     dataset_id: str
                                     ) -> list[ComparisonRecord]:
        """Immutable M5 comparison records of ONE model over ONE dataset
        (M29 read-only access).

        The dataset must exist in the M2 registry (unknown dataset ->
        FileNotFoundError -> 404; datasets are global — model scoping
        comes from the model's own M5 listing). Returns the model's
        authoritative M5 listing filtered by the persisted shared-probe
        dataset identity (a comparison persists exactly ONE top-level
        dataset_id + dataset_version — both sides measure the same
        probe by construction; per-side dataset identities cannot
        occur), each record exactly once (dedup by comparison
        identity), with the persisted dataset_version VERBATIM, in the
        M5 (created_at, comparison_id) order. A valid dataset with no
        comparisons for the model returns []. Read-only, never
        writes."""
        return self.comparison.list_comparisons_for_dataset(
            model_id, dataset_id)

    def list_comparisons_for_tokenizer(self, model_id: str,
                                       tokenizer_id: str
                                       ) -> list[ComparisonRecord]:
        """Immutable M5 comparison records of ONE model measured with
        ONE tokenizer (M31 read-only access).

        The tokenizer must exist in the tokenizer registry (unknown
        tokenizer -> FileNotFoundError -> 404; tokenizers are global —
        model scoping comes from the model's own M5 listing). Returns
        the model's authoritative M5 listing filtered by the persisted
        shared-probe tokenizer identity (a comparison persists exactly
        ONE top-level tokenizer_id — both sides measure the same probe
        by construction; matched VERBATIM, never substituted with the
        latest tokenizer), each record exactly once, in the M5
        (created_at, comparison_id) order. A valid tokenizer with no
        comparisons for the model returns []. Read-only, never
        writes."""
        return self.comparison.list_comparisons_for_tokenizer(
            model_id, tokenizer_id)

    def list_comparisons_for_split(self, model_id: str,
                                   split: EvaluationSplit
                                   ) -> list[ComparisonRecord]:
        """Immutable M5 comparison records of ONE model measured on
        ONE dataset split (M37 read-only access).

        Splits have NO registry — the EvaluationSplit enum IS the
        contract (unsupported values are rejected with 422 at the API
        boundary; unknown model -> FileNotFoundError -> 404, exactly
        like the sibling groupings). Returns the model's authoritative
        M5 listing filtered by the persisted shared-probe split
        identity (a comparison persists exactly ONE top-level split —
        both sides measure the same probe by construction; matched
        VERBATIM, never inferred from datasets, checkpoints or nested
        evaluations), each record exactly once, in the M5
        (created_at, comparison_id) order. A valid split with no
        comparisons for the model returns []. Read-only, never
        writes."""
        return self.comparison.list_comparisons_for_split(model_id, split)

    def list_comparisons_for_verdict(self, model_id: str,
                                     verdict: ComparisonVerdict
                                     ) -> list[ComparisonRecord]:
        """Immutable M5 comparison records of ONE model with ONE
        verdict (M39 read-only access).

        The verdict is the persisted schema enum (improved/regressed/
        unchanged) — there is NO verdict registry, so an unsupported
        value is rejected at the API boundary with 422, while an
        unknown model raises FileNotFoundError (404) exactly like the
        sibling groupings. Returns the model's authoritative M5
        listing filtered by the persisted top-level verdict (matched
        VERBATIM — NEVER recalculated from loss deltas or per-side
        losses, never rewritten; no comparison is executed and no
        evaluation rerun) in the M5 (created_at, comparison_id)
        order, as complete verbatim records. A valid verdict with no
        comparisons for the model returns []. Read-only, never
        writes."""
        return self.comparison.list_comparisons_for_verdict(
            model_id, verdict)

    def list_comparisons_for_state_kind(self, model_id: str,
                                        state_kind: EvalStateKind):
        """Immutable M5 comparisons of ONE model involving ONE kind of
        model state on EITHER side (M44 read-only access; M26's
        either-side semantics).

        The state kind is the persisted schema enum (current/checkpoint)
        of each comparison side — there is NO state-kind registry, so
        an unsupported value is rejected at the API boundary with 422,
        while an unknown model raises FileNotFoundError (404) exactly
        like the sibling groupings. Returns the model's authoritative
        M5 listing filtered by the persisted side state kinds — a
        comparison belongs to the request when EITHER side records the
        requested kind (a both-sides match appears EXACTLY ONCE;
        matched VERBATIM, never inferred from checkpoint ids or
        hashes; nothing recalculated or re-executed) in the M5
        (created_at, comparison_id) order, as complete verbatim
        records. A valid state kind with no matching comparisons
        returns []. Read-only, never writes."""
        return self.comparison.list_comparisons_for_state_kind(
            model_id, state_kind)

    # ------------------------------------------------------------------ #
    # Stage gates (thin delegation; M6)
    # ------------------------------------------------------------------ #

    def run_gate(self, request):
        return self.gates.run(request)

    def list_gate_decisions(self, model_id: str):
        return self.gates.list_decisions(model_id)

    def get_gate_decision(self, model_id: str, decision_id: str):
        return self.gates.get_decision(model_id, decision_id)

    def list_gate_decisions_for_policy(self, model_id: str, policy_id: str):
        """Immutable gate decisions of ONE registered policy (M23
        read-only access).

        The policy must exist in the M9 registry (unknown policy ->
        FileNotFoundError -> 404), and the model must exist (unknown
        model -> FileNotFoundError -> 404). Returns the model's
        authoritative M6 listing filtered by the persisted policy_id
        recorded in each decision — verbatim records in the M6
        (created_at, decision_id) order; inline-policy decisions
        (policy_id None) never appear; a valid policy with no decisions
        for this model returns []. Read-only, never writes."""
        return self.gates.list_decisions_for_policy(model_id, policy_id)

    def list_gate_decisions_for_comparison(self, model_id: str,
                                           comparison_id: str):
        """Immutable gate decisions that judged ONE M5 comparison (M34
        read-only access).

        The comparison must be registered under THIS model's M5
        registry (unknown comparison, or a comparison id belonging to
        another model -> FileNotFoundError -> 404), and the model must
        exist. Returns the model's authoritative M6 listing filtered by
        the persisted comparison_id recorded in each decision —
        verbatim records in the M6 (created_at, decision_id) order;
        legacy direct-evaluation decisions (comparison_id None) never
        appear; a valid comparison with no decisions returns [].
        Read-only, never writes."""
        return self.gates.list_decisions_for_comparison(
            model_id, comparison_id)

    def list_gate_decisions_for_decision(self, model_id: str,
                                         decision: GateDecisionResult):
        """Immutable gate decisions of ONE model with ONE decision
        result (M41 read-only access).

        The decision is the persisted schema enum (passed/failed) —
        there is NO decision registry, so an unsupported value is
        rejected at the API boundary with 422, while an unknown model
        raises FileNotFoundError (404) exactly like the sibling
        groupings. Returns the model's authoritative M6 listing
        filtered by the persisted top-level decision (matched VERBATIM
        — NEVER recalculated from loss deltas, policy thresholds or
        comparison results; no gate is re-evaluated) in the M6
        (created_at, decision_id) order, as complete verbatim records.
        A valid decision with no gate decisions for the model returns
        []. Read-only, never writes."""
        return self.gates.list_decisions_for_decision(model_id, decision)

    def list_gate_decisions_for_verdict(self, model_id: str,
                                        verdict: ComparisonVerdict):
        """Immutable gate decisions of ONE model with ONE recorded
        comparison verdict (M43 read-only access).

        The verdict is the persisted loss-only comparison enum
        (improved/regressed/unchanged) — there is NO verdict registry,
        so an unsupported value is rejected at the API boundary with
        422, while an unknown model raises FileNotFoundError (404)
        exactly like the sibling groupings. Returns the model's
        authoritative M6 listing filtered by the persisted top-level
        verdict (matched VERBATIM — NEVER recalculated from losses,
        deltas, tolerances or comparison records; no gate is
        re-evaluated) in the M6 (created_at, decision_id) order, as
        complete verbatim records. Threshold-only decisions (verdict
        None) belong to NO by-verdict group. A valid verdict with no
        matching decisions returns []. Read-only, never writes."""
        return self.gates.list_decisions_for_verdict(model_id, verdict)

    def list_gate_decisions_for_baseline_type(self, model_id: str,
                                              baseline_type:
                                              GateBaselineType):
        """Immutable gate decisions of ONE model whose embedded policy
        compared the candidate against ONE kind of baseline (M45
        read-only access; the first nested-field grouping).

        The baseline type is the REQUIRED persisted schema enum
        (checkpoint/current/evaluation_result_hash/minimum_loss)
        embedded verbatim in each decision's policy — there is NO
        baseline-type registry, so an unsupported value is rejected at
        the API boundary with 422, while an unknown model raises
        FileNotFoundError (404) exactly like the sibling groupings.
        Returns the model's authoritative M6 listing filtered by the
        persisted nested policy.baseline_type (matched VERBATIM —
        NEVER derived from checkpoint ids, references, results or
        loss deltas; no gate is re-evaluated) in the M6 (created_at,
        decision_id) order, as complete verbatim records. Because the
        field is required, the groups form a TRUE disjoint partition
        of the listing with no None case; a valid baseline type with
        no matching decisions (e.g. evaluation_result_hash where none
        exist) returns []. Read-only, never writes."""
        return self.gates.list_decisions_for_baseline_type(
            model_id, baseline_type)

    # ------------------------------------------------------------------ #
    # Workflows (thin delegation; ordered orchestration over M3-M6; M7)
    # ------------------------------------------------------------------ #

    def run_workflow(self, plan):
        return self.workflows.run(plan)

    def list_workflows(self, model_id: str):
        return self.workflows.list_workflows(model_id)

    def list_workflows_for_recipe(self, model_id: str, recipe_id: str):
        """Immutable workflow runs of ONE model executed from ONE
        registered recipe (M35 read-only access).

        The recipe must exist in the GLOBAL M12/M14 registry (unknown
        recipe -> FileNotFoundError -> 404) and the model must exist
        (unknown model -> FileNotFoundError -> 404). Returns the
        model's authoritative M11 listing filtered by the persisted
        top-level recipe_id — verbatim records in the M11 (created_at,
        workflow_id) order; ad-hoc runs (recipe_id None) never appear;
        a valid registered recipe with no runs for this model returns
        []. Read-only, never writes, never executes anything."""
        return self.workflows.list_workflows_for_recipe(model_id,
                                                        recipe_id)

    def list_workflows_for_status(self, model_id: str,
                                  status: WorkflowStatus):
        """Immutable workflow runs of ONE model with ONE terminal
        status (M42 read-only access).

        The status is the persisted schema enum (completed/failed/
        stopped) — there is NO status registry, so an unsupported
        value is rejected at the API boundary with 422, while an
        unknown model raises FileNotFoundError (404) exactly like the
        sibling grouping. Returns the model's authoritative M11
        listing filtered by the persisted top-level status (matched
        VERBATIM — NEVER inferred from stage results, failed stage
        ids, timestamps, artifact existence or recipe information;
        nothing is re-executed) in the M11 (created_at, workflow_id)
        order, as complete verbatim records. A valid status with no
        matching runs for the model returns []. Read-only, never
        writes, never executes anything."""
        return self.workflows.list_workflows_for_status(model_id,
                                                        status)

    # ------------------------------------------------------------------ #
    # Dashboard (read-only aggregation over immutable histories; M8)
    # ------------------------------------------------------------------ #

    def get_dashboard(self, model_id: str):
        """One deterministic read-only dashboard summary (never writes)."""
        return self.dashboards.dashboard(model_id)

    def get_workflow(self, model_id: str, workflow_id: str):
        return self.workflows.get_workflow(model_id, workflow_id)

    # ------------------------------------------------------------------ #
    # Stable policy registry + named probe suites (definitions only; M9)
    # ------------------------------------------------------------------ #

    def register_policy(self, request: PolicyCreateRequest) -> PolicyDefinition:
        """Register one immutable named policy definition (never overwrites)."""
        return self.policies.register_policy(request)

    def list_policies(self) -> list[PolicyDefinition]:
        return self.policies.list_policies()

    def get_policy(self, policy_id: str) -> PolicyDefinition:
        return self.policies.get_policy(policy_id)

    def register_probe_suite(self, request: ProbeSuiteCreateRequest) -> ProbeSuite:
        """Register one immutable named probe suite (never overwrites)."""
        return self.policies.register_suite(request)

    def list_probe_suites(self) -> list[ProbeSuite]:
        return self.policies.list_suites()

    def get_probe_suite(self, suite_id: str) -> ProbeSuite:
        return self.policies.get_suite(suite_id)

    # ------------------------------------------------------------------ #
    # Suite runs (explicit multi-probe M4 batches over named suites; M10)
    # ------------------------------------------------------------------ #

    def run_suite(self, request: SuiteRunRequest) -> SuiteRunRecord:
        """Execute every probe of one named suite against one state (M4 only)."""
        return self.suite_runs.run(request)

    def list_suite_runs(self, model_id: str) -> list[SuiteRunRecord]:
        return self.suite_runs.list_suite_runs(model_id)

    def get_suite_run(self, model_id: str, suite_run_id: str) -> SuiteRunRecord:
        return self.suite_runs.get_suite_run(model_id, suite_run_id)

    def list_suite_runs_for_suite(self, model_id: str,
                                  suite_id: str) -> list[SuiteRunRecord]:
        """Immutable M10 suite-run records of ONE named suite (M21
        read-only access).

        The suite must exist in the M9 registry (unknown suite ->
        FileNotFoundError -> 404), and the model must exist (unknown
        model -> FileNotFoundError -> 404). Returns the model's
        authoritative M10 listing filtered by the persisted suite_id
        recorded in each record — verbatim records in the M10
        (created_at, suite_run_id) order; a valid suite with no runs for
        this model returns []. Read-only, never writes."""
        return self.suite_runs.list_suite_runs_for_suite(model_id, suite_id)

    def list_suite_run_summary_for_suite(self, model_id: str,
                                         suite_id: str) -> SuiteRunSummary:
        """Read-only bookkeeping summary of ONE model's suite-run history
        for ONE named suite (M22).

        Reuses the M21 filter exactly (model + M9 suite-registry
        validation, persisted suite_id, deterministic M10 order) and
        derives only identity/counting bookkeeping: total_count, the
        ordered run ids and the earliest/latest recorded run timestamps.
        A valid suite with no runs yields a zero summary (total_count 0,
        empty run_ids, None timestamps), not a 404. Never writes.
        """
        return self.suite_runs.list_suite_run_summary_for_suite(
            model_id, suite_id)

    def list_suite_runs_for_checkpoint(self, model_id: str,
                                       checkpoint_id: str
                                       ) -> list[SuiteRunRecord]:
        """Immutable M10 suite-run records executed against ONE checkpoint
        state (M25 read-only access).

        The checkpoint must be registered in the model's M3 checkpoint
        registry (unknown checkpoint, or a checkpoint id belonging to
        another model -> FileNotFoundError -> 404). Returns the model's
        authoritative M10 listing filtered by the persisted run state
        (state_kind "checkpoint" + the checkpoint id) — verbatim records
        in the M10 (created_at, suite_run_id) order; current-state runs
        (state.checkpoint_id None) never appear; a checkpoint with no
        suite runs returns []. Read-only, never writes."""
        return self.suite_runs.list_suite_runs_for_checkpoint(
            model_id, checkpoint_id)

    # ------------------------------------------------------------------ #
    # Workflow recipes (immutable reusable M7 plans; definitions only; M12)
    # ------------------------------------------------------------------ #

    def register_workflow_recipe(
            self, request: WorkflowRecipeCreateRequest) -> WorkflowRecipe:
        """Register one immutable named workflow recipe (never overwrites).

        M14: recipes may reference other registered recipes via ``recipe``
        stages; registration resolves every reference, rejects unknown
        references/cycles/depth > 32 and validates the fully expanded stage
        list with the shared M7 rules before persisting one manifest with
        reference-oriented composition provenance (no manifest on any
        failure; idempotent identical content; conflicts ValueError -> 409).
        """
        return self.recipes.register(request)

    def list_workflow_recipes(self) -> list[WorkflowRecipe]:
        return self.recipes.list()

    def get_workflow_recipe(self, recipe_id: str) -> WorkflowRecipe:
        return self.recipes.get(recipe_id)

    def run_workflow_recipe(self, recipe_id: str, model_id: str) -> WorkflowRecord:
        """Bind ONE explicit model and execute through the existing
        WorkflowEngine (the sole workflow executor). RecipeEngine never
        executes stages itself and never auto-selects anything. M14: the
        definition is expanded deterministically first (identity for plain
        recipes); the run record carries recipe provenance plus the additive
        expansion trace and stays a single workflow record."""
        return self.recipes.run(recipe_id, model_id)

    def list_recipe_runs(self, recipe_id: str) -> list[WorkflowRecord]:
        """Cross-model lineage of one recipe's workflow runs (read-only;
        unknown recipe -> FileNotFoundError -> 404)."""
        return self.recipes.runs(recipe_id)

    # ------------------------------------------------------------------ #
    # Checkpoint sampling (deterministic text generation; M15)
    # ------------------------------------------------------------------ #

    def generate_sample(
            self, request: SampleGenerateRequest) -> SampleRecord:
        """Generate text from ONE explicit verified immutable checkpoint.

        Inference only: never trains, evaluates, scores or modifies the
        model. One immutable sample manifest is created per successful
        request; every preflight failure (unknown model/checkpoint/tokenizer
        404; corrupt checkpoint integrity error; vocabulary/parameter/
        context violations 422) writes nothing."""
        return self.samples.run(request)

    def list_samples(self, model_id: str) -> list[SampleRecord]:
        """Immutable sample history of one model in (created_at, sample_id)
        order (unknown model -> FileNotFoundError -> 404; read-only)."""
        return self.samples.list_samples(model_id)

    def get_sample(self, model_id: str, sample_id: str) -> SampleRecord:
        """One persisted immutable sample (404 unknown model or sample)."""
        return self.samples.get_sample(model_id, sample_id)

    def list_samples_for_checkpoint(self, model_id: str,
                                    checkpoint_id: str
                                    ) -> list[SampleRecord]:
        """Immutable M15 samples generated from ONE checkpoint (M27
        read-only access).

        The checkpoint must be registered in the model's M3 checkpoint
        registry (unknown checkpoint, or a checkpoint id belonging to
        another model -> FileNotFoundError -> 404). Returns the model's
        authoritative M15 listing filtered by the persisted sample
        identity — every sample carries a required non-nullable
        checkpoint_id and belongs to the request only when that
        persisted id matches (nothing is inferred from filenames,
        timestamps or hashes) — in the M15 (created_at, sample_id)
        order. A checkpoint with no samples returns []. Read-only,
        never writes."""
        return self.samples.list_samples_for_checkpoint(
            model_id, checkpoint_id)

    def list_samples_for_tokenizer(self, model_id: str,
                                   tokenizer_id: str
                                   ) -> list[SampleRecord]:
        """Immutable M15 samples of ONE model generated with ONE
        tokenizer (M32 read-only access).

        The tokenizer must exist in the GLOBAL M2 tokenizer registry
        (unknown tokenizer -> FileNotFoundError -> 404); model scoping
        comes from the model's own M15 listing. Returns the
        authoritative M15 listing filtered by the persisted sample
        tokenizer identity — every sample carries a required
        non-nullable top-level tokenizer_id (plus its matching
        tokenizer_hash, preserved verbatim) and belongs to the request
        only when that persisted id matches VERBATIM (nothing is
        inferred from filenames, checkpoints or hashes; no
        latest-tokenizer substitution) — in the M15 (created_at,
        sample_id) order. A valid tokenizer with no samples for the
        model returns []. Read-only, never writes."""
        return self.samples.list_samples_for_tokenizer(
            model_id, tokenizer_id)

    def list_samples_for_strategy(self, model_id: str,
                                  strategy: SampleStrategy
                                  ) -> list[SampleRecord]:
        """Immutable M15 samples of ONE model generated with ONE
        decoding strategy (M40 read-only access).

        The strategy is the persisted schema enum (greedy/temperature)
        — there is NO strategy registry, so an unsupported value is
        rejected at the API boundary with 422, while an unknown model
        raises FileNotFoundError (404) exactly like the sibling
        groupings. Returns the model's authoritative M15 listing
        filtered by the persisted top-level strategy (matched VERBATIM
        — NEVER recalculated from temperature, seed or any other
        field; no sample is regenerated) in the M15 (created_at,
        sample_id) order, as complete verbatim records. A valid
        strategy with no samples for the model returns []. Read-only,
        never writes."""
        return self.samples.list_samples_for_strategy(model_id,
                                                      strategy)

    # ------------------------------------------------------------------ #
    # Sample quality (per-sample likelihood measurement; M16)
    # ------------------------------------------------------------------ #

    def evaluate_sample(self, model_id: str,
                        sample_id: str) -> SampleEvaluationRecord:
        """Measure ONE immutable M15 sample under its OWN recorded
        checkpoint and tokenizer (sample-driven state resolution; nothing
        is auto-selected). The full prompt + generated sequence is scored
        under the existing M4 causal-LM objective over the generated
        continuation targets only (prompt targets excluded). One immutable
        sample-evaluation manifest per success; unknown model/sample 404;
        corrupt/inconsistent records 409; invalid measurement state
        (context window exceeded, vocabulary violations, ...) 422 — every
        failure writes nothing."""
        return self.sample_quality.run(model_id, sample_id)

    def list_sample_evaluations(self, model_id: str
                                ) -> list[SampleEvaluationRecord]:
        """Immutable per-sample measurement history of one model in
        (created_at, evaluation_id) order (unknown model -> 404; read-only;
        never writes anything)."""
        return self.sample_quality.list_sample_evaluations(model_id)

    def get_sample_evaluation(self, model_id: str,
                              evaluation_id: str) -> SampleEvaluationRecord:
        """One persisted immutable sample-evaluation record (404 unknown
        model or evaluation; read-only)."""
        return self.sample_quality.get_sample_evaluation(model_id,
                                                         evaluation_id)

    def list_sample_evaluations_for_sample(
            self, model_id: str, sample_id: str
    ) -> list[SampleEvaluationRecord]:
        """Immutable M16 measurements of ONE sample (M19 read-only access).

        The sample must exist under samples/<model_id>/ (unknown sample or
        a sample id belonging to another model -> FileNotFoundError ->
        404). Returns the model's authoritative M16 listing filtered by the
        sample's persisted sample_id — verbatim records in the M16
        (created_at, evaluation_id) order; a sample without measurements
        returns []. Read-only, never writes."""
        return self.sample_quality.list_sample_evaluations_for_sample(
            model_id, sample_id)

    def list_sample_evaluations_for_checkpoint(
            self, model_id: str, checkpoint_id: str
    ) -> list[SampleEvaluationRecord]:
        """Immutable M16 measurements recorded under ONE checkpoint (M20
        read-only access).

        The checkpoint must be registered in the model's M3 checkpoint
        registry (unknown checkpoint, or a checkpoint id belonging to
        another model -> FileNotFoundError -> 404). Returns the model's
        authoritative M16 listing filtered by the persisted checkpoint_id
        recorded in each record — verbatim records in the M16
        (created_at, evaluation_id) order; a checkpoint without
        measurements returns []. Read-only, never writes."""
        return self.sample_quality.list_sample_evaluations_for_checkpoint(
            model_id, checkpoint_id)

    def list_sample_evaluations_for_tokenizer(
            self, model_id: str, tokenizer_id: str
    ) -> list[SampleEvaluationRecord]:
        """Immutable M16 sample-quality measurements of ONE model whose
        measured samples were generated with ONE tokenizer (M33
        read-only access).

        The tokenizer must exist in the GLOBAL M2 tokenizer registry
        (unknown tokenizer -> FileNotFoundError -> 404); model scoping
        comes from the model's own M16 listing. Returns the
        authoritative M16 listing filtered by the persisted measurement
        tokenizer identity — every record carries a required
        non-nullable top-level tokenizer_id (the measured sample's
        recorded state) and belongs to the request only when that
        persisted id matches VERBATIM (nothing is inferred from
        filenames, sample or checkpoint ids; no latest-tokenizer
        substitution) — in the M16 (created_at, evaluation_id) order.
        A valid tokenizer with no measurements for the model returns
        []. Read-only, never writes."""
        return self.sample_quality.list_sample_evaluations_for_tokenizer(
            model_id, tokenizer_id)


# --------------------------------------------------------------------------- #
# Module-level singleton (bound to FORGE_ROOT at import time; tests override
# the environment variable before importing the package)
# --------------------------------------------------------------------------- #

_forge: Optional[ModelForge] = None


def get_forge() -> ModelForge:
    global _forge
    if _forge is None:
        _forge = ModelForge()
    return _forge
