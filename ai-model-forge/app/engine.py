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
    ModelCreateRequest,
    ModelRecord,
    PolicyCreateRequest,
    PolicyDefinition,
    ProbeSuite,
    ProbeSuiteCreateRequest,
    SuiteRunRecord,
    SuiteRunRequest,
    WorkflowRecord,
    WorkflowRecipe,
    WorkflowRecipeCreateRequest,
    SampleGenerateRequest,
    SampleEvaluationRecord,
    SampleRecord,
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

    # ------------------------------------------------------------------ #
    # Comparison (thin delegation; read-only orchestration over evaluations)
    # ------------------------------------------------------------------ #

    def run_comparison(self, request):
        return self.comparison.run(request)

    def list_comparisons(self, model_id: str):
        return self.comparison.list_comparisons(model_id)

    def get_comparison(self, model_id: str, comparison_id: str):
        return self.comparison.get_comparison(model_id, comparison_id)

    # ------------------------------------------------------------------ #
    # Stage gates (thin delegation; M6)
    # ------------------------------------------------------------------ #

    def run_gate(self, request):
        return self.gates.run(request)

    def list_gate_decisions(self, model_id: str):
        return self.gates.list_decisions(model_id)

    def get_gate_decision(self, model_id: str, decision_id: str):
        return self.gates.get_decision(model_id, decision_id)

    # ------------------------------------------------------------------ #
    # Workflows (thin delegation; ordered orchestration over M3-M6; M7)
    # ------------------------------------------------------------------ #

    def run_workflow(self, plan):
        return self.workflows.run(plan)

    def list_workflows(self, model_id: str):
        return self.workflows.list_workflows(model_id)

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
