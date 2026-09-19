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
from .comparison import COMPARISONS_DIR, ComparisonEngine
from .dashboards import DashboardEngine
from .dataset import TOKENIZED_DIR, DatasetEngine, _referencing_tokenizers
from .evaluation import EVALUATIONS_DIR, EvaluationEngine
from .gates import GATES_DIR, GateEngine
from .hardware import HardwareSpec, detect_hardware
from .model_builder import build_transformer, content_hash
from .policies import (POLICIES_DIR, SUITES_DIR,
                    PolicyEngine, policy_config_hash,
                    probes_hash)
from .recipes import (RECIPES_DIR, RecipeEngine,
                     recipe_config_hash)
from .sampling import SAMPLES_DIR, SamplingEngine
from .sample_quality import SAMPLE_EVALUATIONS_DIR, SampleQualityEngine
from .schemas import (
    CheckpointDeletionBlocker,
    CheckpointDeletionResult,
    ArtifactDeletionBlocker,
    ArtifactKind,
    ArtifactUsageCategory,
    CheckpointRetentionEntry,
    CheckpointRetentionOverview,
    ComparisonRecord,
    ComparisonVerdict,
    EvalStateKind,
    EvaluationRecord,
    EvaluationSplit,
    GateBaselineType,
    GateDecisionResult,
    ModelCreateRequest,
    DeletionImpactPreview,
    DefinitionDeletionBlocker,
    ImpactBecomesDeletable,
    ImpactBlocker,
    ImpactDependent,
    ProjectFamilyRetention,
    ProjectRetentionOverview,
    DefinitionDeletionResult,
    DefinitionRetentionOverview,
    ModelDeletionBlocker,
    ModelDeletionResult,
    ModelOwnedRecordCategory,
    ModelRecordDeletionBlocker,
    ModelRecordDeletionResult,
    ModelRecordRetentionOverview,
    ModelOwnedRecordUsage,
    ModelRecordsUsageOverview,
    ModelRecord,
    RecordReference,
    ModelRetentionOverview,
    ModelUsageCategory,
    ModelUsageOverview,
    PolicyCreateRequest,
    PolicyDefinition,
    ProbeSuite,
    ProbeSuiteCreateRequest,
    ProjectModelStorageSummary,
    ProjectStorageCategory,
    ProjectStorageOverview,
    DatasetDeletionResult,
    DatasetRetentionOverview,
    DatasetUsageOverview,
    TokenizerDeletionResult,
    TokenizerRetentionOverview,
    TokenizerUsageOverview,
    SuiteRunDeletionResult,
    SuiteRunRecord,
    SuiteRunRequest,
    SuiteRunSummary,
    WorkflowRecord,
    WorkflowRecipe,
    WorkflowRecipeCreateRequest,
    WorkflowRecipeResolution,
    WorkflowStatus,
    SampleGenerateRequest,
    SampleEvaluationDeletionResult,
    SampleEvaluationRecord,
    SampleDeletionResult,
    SampleEvaluationRetentionOverview,
    SampleRetentionOverview,
    SuiteRunRetentionOverview,
    SampleRecord,
    SampleStrategy,
    TokenizerConfig,
    TransformerConfig,
    utcnow,
)
from .storage import Storage, atomic_delete_dir
from .suite_runs import SUITE_RUNS_DIR, SuiteRunEngine
from .tokenizer import TokenizerEngine
from .training import CHECKPOINTS_DIR, TrainingEngine
from .workflows import WORKFLOWS_DIR, WorkflowEngine

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

    def delete_model(self, model_id: str) -> ModelDeletionResult:
        """Explicit VERIFIED model retention (M67): remove ONE model
        — only after proving, live, that nothing OUTSIDE the model
        directory references it. The full guard, in order: (1) scope
        through the registry (``storage.load_record`` — unknown or
        registry-invisible (unparseable manifest) model ->
        FileNotFoundError, nothing deleted); (2) INTEGRITY FIRST (the
        M61/M65 ordering): the existing M2 verifier
        (``storage.verify_integrity`` — manifest parse, state reload,
        weights hash sidecar) must pass; a corrupt, incomplete or
        missing-weights model is REFUSED with RuntimeError (409) —
        deletion never bypasses integrity validation, and no force
        flag or filesystem fallback exists; (3) the LIVE dependency
        analysis (``model_deletion_blockers`` — EXACTLY the EXTERNAL
        root-level references of the ONE M66 usage overview: suite
        runs, samples, sample-quality measurements, model-bound
        workflow recipes and model-bound gate policies; ANY such
        reference -> ValueError listing the ordered typed blockers, so
        deletion can never orphan a persisted record. The INTERNAL
        model-scoped families — training runs, checkpoints,
        workflows, evaluations, comparisons, gate decisions — are
        ownership: they live INSIDE ``models/<id>/`` and are removed
        atomically WITH the model, so they never block); (4) ATOMIC
        removal of the model's OWN directory only
        (``atomic_delete_dir`` — one rename to a hidden sibling, then
        rmtree; no partial model can ever be observed). No cascade,
        no force, no bulk mode — exactly the one explicitly requested
        model. Never touches root-level families (they are protected
        BY the guard). Replaces the M1-era unguarded rmtree."""
        self._scope_data_artifact(self.storage.load_record, model_id,
                                  "model")
        try:
            self.storage.verify_integrity(model_id)
        except Exception as exc:
            # scope already proved the manifest loads, so ANY failure
            # here is an integrity failure (missing/corrupt weights,
            # hash mismatch, empty state) — refuse (409), never a
            # silent rmtree
            raise RuntimeError(
                f"model '{model_id}' could not be verified: "
                f"{exc}") from exc
        blockers = self.model_deletion_blockers(model_id)
        if blockers:
            summary = "; ".join(f"{b.category}: {b.reference_id}"
                                for b in blockers)
            raise ValueError(
                f"model '{model_id}' is referenced and cannot be "
                f"deleted — {summary}")
        model_dir = self.storage.model_dir(model_id)
        files, nbytes = self._artifact_files(model_dir)
        atomic_delete_dir(model_dir)
        log.info("M67 verified model deletion %s (%d files, %d bytes)",
                 model_id, len(files), nbytes)
        return ModelDeletionResult(model_id=model_id,
                                   files_removed=len(files),
                                   bytes_reclaimed=nbytes)

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

    def select_best_checkpoint(self, model_id: str):
        """Read-only selection of ONE model's best checkpoint under the
        persisted validation-loss criterion (M52): the MINIMUM persisted
        validation_loss over the authoritative M3 listing, ties resolved
        by the canonical (step, created_at) ASCENDING order (first among
        equals, disclosed via ``tied``), non-finite values never
        candidates. Never writes, never persists a selection pointer;
        unknown model or no selectable checkpoints -> FileNotFoundError
        (404 at the API)."""
        return self.training.select_best_checkpoint(model_id)

    def best_checkpoint_history(self, model_id: str):
        """M59: read-only chronological history of the M52 best-checkpoint
        selection movements — the running argmin replayed over the
        authoritative listing with the ONE shared winner rule. Computed
        live (zero persistence); the final entry is always the live M52
        answer; unknown model -> FileNotFoundError (404); a valid model
        with no selectable checkpoints has an empty history."""
        return self.training.best_checkpoint_history(model_id)

    def rollback_model(self, model_id: str, checkpoint_id: str):
        return self.training.rollback(model_id, checkpoint_id)

    # ------------------------------------------------------------------ #
    # Explicit verified checkpoint retention (M61): the facade is the one
    # integration point where every record family's engine is visible, so
    # the LIVE reference-safety analysis lives here — computed from the
    # authoritative listings on every call, never stored, never a second
    # registry.
    # ------------------------------------------------------------------ #

    #: canonical deterministic blocker order (the API's 409 detail and the
    #: engine's rejection message list blockers in exactly this order)
    CHECKPOINT_BLOCKER_ORDER = (
        "best",                    # the live M52 selection
        "published",               # model manifest latest_checkpoint
        "manifest_reference",      # model manifest best_checkpoint (weights ref)
        "workflow_reference",      # pinned/explicit ids in workflow records
        "evaluation_reference",    # M4 evaluation records
        "comparison_reference",    # M5 comparison records (either side)
        "gate_reference",          # M6 decisions (candidate/baseline/suggestion)
        "suite_run_reference",     # M10 suite-run records
        "sample_reference",        # M15 samples generated from the checkpoint
        "sample_quality_reference",  # M16 sample-quality measurements
    )

    @staticmethod
    def _workflow_checkpoint_refs(record: WorkflowRecord) -> list[tuple[str, str]]:
        """Every checkpoint id one immutable workflow record depends on
        (M61): the pins and explicit ids in the recorded PLAN (M53 state
        refs, M54/M55 resume, M56 evaluation, M57 baseline, M60 publish)
        plus the checkpoint ids in the recorded stage ARTIFACTS (training
        final checkpoints, evaluated states, publications). Returns
        (checkpoint_id, where) pairs; deterministic in record order."""
        refs: list[tuple[str, str]] = []

        def add(cid, where):
            if cid:
                refs.append((cid, where))

        for stage in record.plan.stages:
            sid = stage.stage_id
            if stage.training is not None:
                add(stage.training.resume_from_checkpoint_id,
                    f"plan.{sid}.resume")
                add(stage.training.resolved_resume_checkpoint_id,
                    f"plan.{sid}.resume(best)")
            if stage.evaluation is not None:
                add(stage.evaluation.config.checkpoint_id,
                    f"plan.{sid}.evaluation")
                add(stage.evaluation.resolved_checkpoint_id,
                    f"plan.{sid}.evaluation(best)")
            if stage.comparison is not None:
                for name, ref in (("state_a", stage.comparison.state_a),
                                  ("state_b", stage.comparison.state_b)):
                    add(ref.checkpoint_id, f"plan.{sid}.{name}")
                    add(ref.resolved_checkpoint_id, f"plan.{sid}.{name}(best)")
            if stage.gate is not None:
                if stage.gate.policy is not None:
                    add(stage.gate.policy.baseline_checkpoint_id,
                        f"plan.{sid}.baseline")
                    add(stage.gate.policy.resolved_baseline_checkpoint_id,
                        f"plan.{sid}.baseline(best)")
                add(stage.gate.candidate.checkpoint_id,
                    f"plan.{sid}.candidate")
                add(stage.gate.candidate.resolved_checkpoint_id,
                    f"plan.{sid}.candidate(best)")
            if stage.suite_run is not None:
                add(stage.suite_run.state.checkpoint_id,
                    f"plan.{sid}.suite_state")
                add(stage.suite_run.state.resolved_checkpoint_id,
                    f"plan.{sid}.suite_state(best)")
            if stage.publish is not None:
                add(stage.publish.checkpoint_id, f"plan.{sid}.publish")
                add(stage.publish.resolved_checkpoint_id,
                    f"plan.{sid}.publish(best)")
        for sr in record.stages:
            if sr.artifact is not None:
                add(sr.artifact.checkpoint_id,
                    f"artifact.{sr.stage_id}")
        return refs

    def checkpoint_blockers(self, model_id: str,
                            ckpt_id: str) -> list[CheckpointDeletionBlocker]:
        """LIVE reference-safety analysis for ONE checkpoint (M61): every
        authoritative state or immutable EVIDENCE record that would be
        invalidated by deleting it. Computed from the authoritative
        listings on every call — the SAME selectors and the SAME
        ``..._for_checkpoint`` filters the read APIs use (one source of
        truth, no stored index, no second registry). Blocking references
        (documented in the M61 report):

        * ``best`` — the live M52 selection (never deletable);
        * ``published`` — the model manifest's ``latest_checkpoint``
          (the live/published state);
        * ``manifest_reference`` — the manifest's ``best_checkpoint``
          stored best-known weights reference;
        * ``workflow_reference`` — pins/explicit ids in immutable M7
          workflow records (plans and stage artifacts — the audit
          trail of what executed);
        * ``evaluation_reference`` / ``comparison_reference`` /
          ``gate_reference`` / ``suite_run_reference`` /
          ``sample_reference`` / ``sample_quality_reference`` — the
          persisted checkpoint ids in the M4/M5/M6/M10/M15/M16
          EVIDENCE record families (which state was measured; via the
          existing read-only ``..._for_checkpoint`` filters).

        Deliberately NOT blocking (inspected and documented): pure
        LINEAGE metadata — a surviving checkpoint's
        ``parent_checkpoint_id`` and the model manifest's run
        provenance (``initial``/``final``/``rolled_back_to``) — because
        the architecture treats them as informational history, not
        resolvable dependencies: nothing loads state through them and
        the M8 reference graph explicitly tolerates a missing target
        (a DashboardDiagnostic, valid artifacts stay visible). They
        also CANNOT block without making retention dead code: M3
        chains checkpoints within every run (each checkpoint's parent
        is its predecessor; the run's final is the provenance final),
        so blocking lineage would protect every checkpoint ever
        created. Likewise the M59 best-history and the M8 dashboard
        are computed views — appearing in them protects nothing.
        Unknown model or a checkpoint not in the authoritative listing
        (including an unreadable manifest, which the listing skips) ->
        FileNotFoundError (404 at the API); deletion is refused, never
        guessed. Read-only, never writes."""
        listing = self.training.list_checkpoints(model_id)
        if ckpt_id not in {c.checkpoint_id for c in listing}:
            raise FileNotFoundError(
                f"checkpoint '{ckpt_id}' not found for model '{model_id}'")
        found: list[tuple[str, str]] = []

        def add(reason: str, detail: str) -> None:
            found.append((reason, detail))

        # best (the ONE selector, computed live)
        try:
            if self.training.select_best_checkpoint(
                    model_id).checkpoint.checkpoint_id == ckpt_id:
                add("best", "the live M52 best-checkpoint selection")
        except FileNotFoundError:
            pass
        model = self.storage.load_record(model_id)
        if model.latest_checkpoint == ckpt_id:
            add("published", "model manifest latest_checkpoint "
                             "(the published/live state)")
        if model.best_checkpoint == ckpt_id:
            add("manifest_reference", "model manifest best_checkpoint "
                                      "(stored best-known weights reference)")
        for record in self.workflows.list_workflows(model_id):
            for cid, where in self._workflow_checkpoint_refs(record):
                if cid == ckpt_id:
                    add("workflow_reference",
                        f"{record.workflow_id} {where}")
        evals = self.evaluation.list_evaluations_for_checkpoint(
            model_id, ckpt_id)
        if evals:
            add("evaluation_reference",
                "evaluations: " + ", ".join(e.eval_id for e in evals))
        comps = self.comparison.list_comparisons_for_checkpoint(
            model_id, ckpt_id)
        if comps:
            add("comparison_reference",
                "comparisons: " + ", ".join(c.comparison_id for c in comps))
        gate_refs: list[str] = []
        for g in self.gates.list_decisions(model_id):
            for side_name, side in (("candidate", g.candidate),
                                    ("baseline", g.baseline)):
                if (side is not None
                        and side.state_kind == EvalStateKind.CHECKPOINT
                        and side.checkpoint_id == ckpt_id):
                    gate_refs.append(f"{g.decision_id}.{side_name}")
            if g.suggested_checkpoint_id == ckpt_id:
                gate_refs.append(f"{g.decision_id}.suggested_checkpoint_id")
        if gate_refs:
            add("gate_reference", "gates: " + ", ".join(gate_refs))
        suites = self.suite_runs.list_suite_runs_for_checkpoint(
            model_id, ckpt_id)
        if suites:
            add("suite_run_reference",
                "suite runs: " + ", ".join(s.suite_run_id for s in suites))
        samples = self.samples.list_samples_for_checkpoint(model_id, ckpt_id)
        if samples:
            add("sample_reference",
                "samples: " + ", ".join(s.sample_id for s in samples))
        sq = self.sample_quality.list_sample_evaluations_for_checkpoint(
            model_id, ckpt_id)
        if sq:
            add("sample_quality_reference",
                "sample evaluations: " + ", ".join(s.evaluation_id for s in sq))
        order = {r: i for i, r in enumerate(self.CHECKPOINT_BLOCKER_ORDER)}
        found.sort(key=lambda item: (order[item[0]], item[1]))
        return [CheckpointDeletionBlocker(reason=r, detail=d)
                for r, d in found]

    def delete_checkpoint(self, model_id: str,
                          ckpt_id: str) -> CheckpointDeletionResult:
        """Explicit VERIFIED checkpoint retention (M61): remove ONE
        checkpoint — only after proving, live, that nothing
        authoritative depends on it. The full guard, in order:
        (1) model/checkpoint scope through the authoritative listing
        (unknown model, or a checkpoint the listing cannot see —
        including an unreadable manifest, which it skips — ->
        FileNotFoundError, nothing deleted);
        (2) INTEGRITY VERIFICATION through the existing M3 verifier
        (``verify_checkpoint``: weights load + content-hash check) — a
        corrupt checkpoint is REFUSED, so deletion can never bypass
        integrity validation;
        (3) the LIVE reference-safety analysis
        (``checkpoint_blockers``) — the M52 best, the published
        ``latest_checkpoint`` and every referenced checkpoint are
        protected deterministically -> ValueError listing the blockers;
        (4) ATOMIC removal (one ``os.rename`` to a hidden sibling the
        listing skips, then rmtree) — no partial checkpoint can ever be
        observed, and no manifest/pointer needs updating because the
        checkpoint registry IS the listing.
        No automatic deletion, no keep-N, no age policy, no bulk mode —
        exactly the one explicitly requested checkpoint. Never creates
        a replacement checkpoint; never touches other checkpoints, the
        published state, or any record family."""
        listing = self.training.list_checkpoints(model_id)
        if ckpt_id not in {c.checkpoint_id for c in listing}:
            raise FileNotFoundError(
                f"checkpoint '{ckpt_id}' not found for model '{model_id}'")
        self.training.verify_checkpoint(model_id, ckpt_id)  # corrupt -> refuse
        blockers = self.checkpoint_blockers(model_id, ckpt_id)
        if blockers:
            summary = "; ".join(f"{b.reason}: {b.detail}" for b in blockers)
            raise ValueError(
                f"checkpoint '{ckpt_id}' of model '{model_id}' is "
                f"protected and cannot be deleted — {summary}")
        files, nbytes = self.training.remove_checkpoint(model_id, ckpt_id)
        return CheckpointDeletionResult(
            model_id=model_id, checkpoint_id=ckpt_id,
            files_removed=files, bytes_reclaimed=nbytes)

    def checkpoint_retention_overview(self, model_id: str
                                      ) -> CheckpointRetentionOverview:
        """Read-only live-computed retention overview of ONE model's
        checkpoint registry (M62): for every checkpoint in the
        authoritative M3 listing (canonical (step, created_at) order)
        the EXACT M61 decision — ``deletable`` is True iff an immediate
        M61 ``delete_checkpoint`` would succeed, i.e. the SAME M3
        integrity verification passes AND the SAME live blocker
        analysis (``checkpoint_blockers`` — the ONE analysis, never a
        second scanner) returns nothing — plus the persisted identity
        (run/step/created_at/validation_loss) and the SAME artifact-set
        measurement the removal result reports. Aggregates are
        deterministic sums over the entries; ``reclaimable`` counts
        ONLY currently-deletable checkpoint artifact sets (never model
        weights, tokenizer, dataset or record storage); ``protected``
        counts every non-deletable checkpoint (referenced OR
        integrity-failed). Zero storage, zero mutation, byte-identical
        over unchanged state; unknown model -> FileNotFoundError (404
        at the API); a valid model with no checkpoints -> an EMPTY
        overview with zeroed totals (the collection convention). A
        checkpoint whose manifest is unreadable is listing-invisible
        and therefore never appears (the established M61 corruption
        semantics)."""
        listing = self.training.list_checkpoints(model_id)
        entries: list[CheckpointRetentionEntry] = []
        for c in listing:
            blockers = self.checkpoint_blockers(model_id, c.checkpoint_id)
            try:
                self.training.verify_checkpoint(model_id, c.checkpoint_id)
                integrity_verified = True
            except (RuntimeError, FileNotFoundError):
                integrity_verified = False
            files, nbytes = self.training.checkpoint_artifact_stats(
                model_id, c.checkpoint_id)
            entries.append(CheckpointRetentionEntry(
                checkpoint_id=c.checkpoint_id,
                run_id=c.run_id,
                step=c.step,
                created_at=c.created_at,
                validation_loss=c.validation_loss,
                files=files,
                size_bytes=nbytes,
                integrity_verified=integrity_verified,
                deletable=integrity_verified and not blockers,
                blockers=blockers))
        deletable = [e for e in entries if e.deletable]
        return CheckpointRetentionOverview(
            model_id=model_id,
            total_checkpoints=len(entries),
            deletable_checkpoints=len(deletable),
            protected_checkpoints=len(entries) - len(deletable),
            total_checkpoint_bytes=sum(e.size_bytes for e in entries),
            reclaimable_checkpoint_bytes=sum(e.size_bytes
                                             for e in deletable),
            checkpoints=entries)

    # Canonical PROJECT STORAGE category order (M63, fixed and
    # deterministic). Every physical project file belongs to EXACTLY
    # ONE category — the family directory constants are imported from
    # the ONE place each family defines its layout (never a second
    # taxonomy), and "unclassified" is the explicit catch-all so no
    # file is ever silently discarded.
    PROJECT_STORAGE_CATEGORIES = (
        "models",              # models/<id>/{manifest,weights.pt,weights.sha256}
        "checkpoints",         # models/<id>/checkpoints/**
        "model_records",       # models/<id>/{evaluations,comparisons,gates,workflows}/**
        "datasets",            # datasets/**
        "tokenizers",          # tokenizers/**
        "suite_runs",          # suite-runs/**
        "samples",             # samples/**
        "sample_evaluations",  # sample-evaluations/**
        "policies",            # policies/**
        "probe_suites",        # probe-suites/**
        "workflow_recipes",    # workflow-recipes/**
        "project",             # project.json
        "unclassified",        # explicit catch-all (never silent)
    )

    def project_storage_overview(self) -> ProjectStorageOverview:
        """Read-only live-computed PHYSICAL storage overview of the
        whole project (M63): totals, category partition and per-model
        rows, plus the project-level retention aggregates.

        Accounting rules (the M63 invariants):

        * ONE physical walk classifies every file under the storage
          root into EXACTLY ONE category (the family directory
          constants above; tmp/ scratch and hidden crash-residue
          entries are outside the boundary), so the categories sum to
          ``total_files``/``total_bytes`` — no file is ever counted
          twice and nothing is silently discarded (unknown layouts
          land in the explicit ``unclassified`` category).
        * checkpoint retention numbers are NEVER recomputed here: each
          model row carries its M62 ``checkpoint_retention_overview``
          aggregates VERBATIM (the ONE M61 blocker analysis and the
          ONE artifact-set measurement), and the project aggregates
          are deterministic sums over the rows. ``reclaimable`` counts
          ONLY currently-deletable checkpoint artifact sets — never
          model weights, tokenizer, dataset or record storage; M61
          deletion is the only operation that ever reclaims them.
        * model rows follow the registry order (``list_models``:
          created_at) and cover record-valid models only; a model
          whose manifest cannot be parsed is skipped (the registry
          convention) while its files still count in the physical
          category totals.
        * ``checkpoints`` category bytes are PHYSICAL (including any
          listing-invisible checkpoint directory of a valid model);
          ``total_checkpoint_bytes`` is the registry-visible M62 sum —
          the two agree exactly on healthy storage.

        Zero storage, zero mutation, deterministic (byte-identical
        over unchanged state); an empty project reports zeroed totals
        with an empty model collection and only the ``project``
        category non-empty."""
        root = self.storage.root
        evidence = (EVALUATIONS_DIR, COMPARISONS_DIR, GATES_DIR,
                    WORKFLOWS_DIR)
        root_families = {
            self.storage.datasets_dir.name: "datasets",
            self.storage.tokenizers_dir.name: "tokenizers",
            SUITE_RUNS_DIR: "suite_runs",
            SAMPLES_DIR: "samples",
            SAMPLE_EVALUATIONS_DIR: "sample_evaluations",
            POLICIES_DIR: "policies",
            SUITES_DIR: "probe_suites",
            RECIPES_DIR: "workflow_recipes",
        }

        # ---- ONE physical walk: relative posix path -> byte size ----
        sizes: dict[str, int] = {}
        if root.exists():
            for p in sorted(root.rglob("*")):
                rel = p.relative_to(root)
                if rel.parts[0] == "tmp":
                    continue          # atomic-write scratch (startup-cleaned)
                if any(part.startswith(".") for part in rel.parts):
                    continue          # hidden crash residue (atomic renames)
                if p.is_file():
                    sizes[rel.as_posix()] = p.stat().st_size

        def category_of(rel: str) -> str:
            parts = rel.split("/")
            if parts[0] == "models":
                if len(parts) == 3:
                    return "models"   # directly in the model dir
                if len(parts) >= 4:
                    if parts[2] == CHECKPOINTS_DIR:
                        return "checkpoints"
                    if parts[2] in evidence:
                        return "model_records"
                return "unclassified"
            if rel == "project.json":
                return "project"
            return root_families.get(parts[0], "unclassified")

        cat_files = {name: 0 for name in self.PROJECT_STORAGE_CATEGORIES}
        cat_bytes = {name: 0 for name in self.PROJECT_STORAGE_CATEGORIES}
        for rel, size in sizes.items():
            cat = category_of(rel)
            cat_files[cat] += 1
            cat_bytes[cat] += size

        # ---- per-model rows: registry order, M62 numbers VERBATIM ----
        rows: list[ProjectModelStorageSummary] = []
        for record in self.list_models():
            mid = record.id
            model_bytes = 0
            records_bytes = 0
            for rel, size in sizes.items():
                parts = rel.split("/")
                if parts[:2] != ["models", mid]:
                    continue
                if len(parts) == 3:
                    model_bytes += size
                elif len(parts) >= 4 and parts[2] in evidence:
                    records_bytes += size
            ov = self.checkpoint_retention_overview(mid)
            rows.append(ProjectModelStorageSummary(
                model_id=mid,
                name=record.name,
                created_at=record.created_at,
                model_bytes=model_bytes,
                records_bytes=records_bytes,
                checkpoint_count=ov.total_checkpoints,
                deletable_checkpoints=ov.deletable_checkpoints,
                protected_checkpoints=ov.protected_checkpoints,
                total_checkpoint_bytes=ov.total_checkpoint_bytes,
                reclaimable_checkpoint_bytes=ov.reclaimable_checkpoint_bytes,
                protected_checkpoint_bytes=sum(
                    e.size_bytes for e in ov.checkpoints if not e.deletable),
                total_model_bytes=(model_bytes + records_bytes
                                   + ov.total_checkpoint_bytes)))

        return ProjectStorageOverview(
            total_files=len(sizes),
            total_bytes=sum(sizes.values()),
            model_count=len(rows),
            checkpoint_count=sum(r.checkpoint_count for r in rows),
            deletable_checkpoints=sum(r.deletable_checkpoints for r in rows),
            protected_checkpoints=sum(r.protected_checkpoints for r in rows),
            total_checkpoint_bytes=sum(r.total_checkpoint_bytes
                                       for r in rows),
            reclaimable_checkpoint_bytes=sum(r.reclaimable_checkpoint_bytes
                                             for r in rows),
            protected_checkpoint_bytes=sum(r.protected_checkpoint_bytes
                                           for r in rows),
            categories=[ProjectStorageCategory(name=name,
                                               files=cat_files[name],
                                               bytes=cat_bytes[name])
                        for name in self.PROJECT_STORAGE_CATEGORIES],
            models=rows)

    # Canonical M64 usage-category orders (fixed and deterministic; the
    # shared evidence categories follow the M61 blocker order, the
    # provenance category leads, artifact-specific categories trail).
    DATASET_USAGE_CATEGORIES = (
        "training_run",       # model manifests' RunProvenance (dataset_id)
        "workflow",           # workflow records' embedded plans (stage configs)
        "evaluation",         # M4 evaluation records (the ONE for-dataset filter)
        "comparison",         # M5 comparison records (the ONE for-dataset filter)
        "suite_run",          # M10 suite-run records (executed probes)
        "tokenizer_training", # tokenizers trained on it (the ONE guard scan)
        "tokenized_version",  # tokenized artifacts under its own versions
    )
    TOKENIZER_USAGE_CATEGORIES = (
        "training_run",       # model manifests' RunProvenance (tokenizer_id)
        "workflow",           # workflow records' embedded plans (stage configs)
        "evaluation",         # M4 evaluation records (the ONE for-tokenizer filter)
        "comparison",         # M5 comparison records (the ONE for-tokenizer filter)
        "suite_run",          # M10 suite-run records (executed probes)
        "sample",             # M15 sample records (the ONE for-tokenizer filter)
        "sample_quality",     # M16 sample-quality records (the ONE filter)
        "tokenized_dataset",  # dataset versions this tokenizer tokenized
    )

    @staticmethod
    def _workflow_data_refs(plan) -> set[tuple[str, str]]:
        """(dataset_id, tokenizer_id) pairs DIRECTLY named by a workflow
        plan's stage configs — train (TrainingConfig), evaluate
        (EvaluationConfig) and compare (WorkflowComparisonStage) stages.
        Suite-run stages reference a suite id (the shared M9 registry),
        never a dataset/tokenizer directly — that indirection is
        deliberately NOT followed here (the suite registry is a
        definition, not per-model evidence). Read-only plan walk, the
        M61 blocker-analysis style."""
        refs: set[tuple[str, str]] = set()
        for stage in plan.stages:
            if stage.training is not None:
                refs.add((stage.training.dataset_id,
                          stage.training.tokenizer_id))
            if stage.evaluation is not None:
                cfg = stage.evaluation.config
                refs.add((cfg.dataset_id, cfg.tokenizer_id))
            if stage.comparison is not None:
                refs.add((stage.comparison.dataset_id,
                          stage.comparison.tokenizer_id))
        return refs

    def _tokenized_tokenizers(self, dataset_id: str, version: int) -> list[str]:
        """Tokenizer ids with a persisted tokenized artifact under ONE
        dataset version — the SAME M2 layout ``DatasetEngine.get``
        exposes (``datasets/<id>/v<N>/tokenized/<tokenizer_id>/``),
        walked read-only."""
        troot = (self.storage.dataset_dir(dataset_id) / f"v{version}"
                 / TOKENIZED_DIR)
        if not troot.exists():
            return []
        return sorted(d.name for d in troot.iterdir()
                      if d.is_dir() and not d.name.startswith("."))

    def dataset_usage_overview(self, dataset_id: str) -> DatasetUsageOverview:
        """Read-only live-computed usage overview of ONE dataset (M64):
        every persisted record that references it, by category. REUSES
        the ONE cross-reference filters (``list_evaluations_for_dataset``,
        ``list_comparisons_for_dataset``), the ONE tokenizer-training
        scan the deletion guard refuses on
        (``_referencing_tokenizers``), the authoritative registries
        (``list_models`` — corrupt manifests skipped, the registry
        convention) and the ONE M2 tokenized layout. Per-record
        references (a workflow/suite-run/training run counts ONCE even
        when several of its stages/probes name the dataset); every
        category appears in the canonical order with its deterministic
        sorted id list. Zero storage, zero mutation, byte-identical
        over unchanged state; unknown dataset -> FileNotFoundError."""
        meta = self.datasets.load_meta(dataset_id)   # the registry getter
        training_runs: list[str] = []
        workflows: list[str] = []
        evaluations: list[str] = []
        comparisons: list[str] = []
        suite_runs: list[str] = []
        for record in self.list_models():
            mid = record.id
            for prov in record.training_provenance:
                if prov.dataset_id == dataset_id:
                    training_runs.append(f"{mid}/{prov.run_id}")
            for wf in self.list_workflows(mid):
                if any(ds == dataset_id
                       for ds, _tok in self._workflow_data_refs(wf.plan)):
                    workflows.append(f"{mid}/{wf.workflow_id}")
            for ev in self.list_evaluations_for_dataset(mid, dataset_id):
                evaluations.append(f"{mid}/{ev.eval_id}")
            for comp in self.list_comparisons_for_dataset(mid, dataset_id):
                comparisons.append(f"{mid}/{comp.comparison_id}")
            for run in self.list_suite_runs(mid):
                if any(r.probe.dataset_id == dataset_id
                       for r in run.results):
                    suite_runs.append(f"{mid}/{run.suite_run_id}")
        tokenizer_training = _referencing_tokenizers(self.storage, dataset_id)
        tokenized_versions = [
            f"v{version}/{tokenizer_id}"
            for version in meta.versions
            for tokenizer_id in self._tokenized_tokenizers(dataset_id,
                                                            version)
        ]
        categories = [
            ("training_run", sorted(training_runs)),
            ("workflow", sorted(workflows)),
            ("evaluation", sorted(evaluations)),
            ("comparison", sorted(comparisons)),
            ("suite_run", sorted(suite_runs)),
            ("tokenizer_training", sorted(tokenizer_training)),
            ("tokenized_version", sorted(tokenized_versions)),
        ]
        total = sum(len(refs) for _, refs in categories)
        return DatasetUsageOverview(
            dataset_id=dataset_id,
            name=meta.name,
            created_at=meta.created_at,
            version_count=len(meta.versions),
            latest_version=meta.latest_version,
            referenced=total > 0,
            total_references=total,
            categories=[ArtifactUsageCategory(category=c, references=r)
                        for c, r in categories])

    def tokenizer_usage_overview(
            self, tokenizer_id: str) -> TokenizerUsageOverview:
        """Read-only live-computed usage overview of ONE tokenizer
        (M64): every persisted record that references it, by category.
        REUSES the ONE cross-reference filters
        (``list_evaluations_for_tokenizer``,
        ``list_comparisons_for_tokenizer``,
        ``list_samples_for_tokenizer``,
        ``list_sample_evaluations_for_tokenizer``), the authoritative
        registries and the ONE M2 tokenized layout (reversed: which
        dataset versions this tokenizer tokenized). The tokenizer's own
        ``trained_on_dataset_id`` travels as identity (provenance).
        Per-record references; every category appears in the canonical
        order with its deterministic sorted id list. Zero storage, zero
        mutation, byte-identical over unchanged state; unknown
        tokenizer -> FileNotFoundError."""
        record = self.tokenizers.load(tokenizer_id)  # the registry getter
        training_runs: list[str] = []
        workflows: list[str] = []
        evaluations: list[str] = []
        comparisons: list[str] = []
        suite_runs: list[str] = []
        samples: list[str] = []
        sample_quality: list[str] = []
        for model in self.list_models():
            mid = model.id
            for prov in model.training_provenance:
                if prov.tokenizer_id == tokenizer_id:
                    training_runs.append(f"{mid}/{prov.run_id}")
            for wf in self.list_workflows(mid):
                if any(tok == tokenizer_id
                       for _ds, tok in self._workflow_data_refs(wf.plan)):
                    workflows.append(f"{mid}/{wf.workflow_id}")
            for ev in self.list_evaluations_for_tokenizer(mid, tokenizer_id):
                evaluations.append(f"{mid}/{ev.eval_id}")
            for comp in self.list_comparisons_for_tokenizer(mid,
                                                            tokenizer_id):
                comparisons.append(f"{mid}/{comp.comparison_id}")
            for run in self.list_suite_runs(mid):
                if any(r.probe.tokenizer_id == tokenizer_id
                       for r in run.results):
                    suite_runs.append(f"{mid}/{run.suite_run_id}")
            for sample in self.list_samples_for_tokenizer(mid, tokenizer_id):
                samples.append(f"{mid}/{sample.sample_id}")
            for sq in self.list_sample_evaluations_for_tokenizer(
                    mid, tokenizer_id):
                sample_quality.append(f"{mid}/{sq.evaluation_id}")
        tokenized_datasets = [
            f"{info.id}/v{version}"
            for info in self.datasets.list()
            for version in info.versions
            if tokenizer_id in self._tokenized_tokenizers(info.id, version)
        ]
        categories = [
            ("training_run", sorted(training_runs)),
            ("workflow", sorted(workflows)),
            ("evaluation", sorted(evaluations)),
            ("comparison", sorted(comparisons)),
            ("suite_run", sorted(suite_runs)),
            ("sample", sorted(samples)),
            ("sample_quality", sorted(sample_quality)),
            ("tokenized_dataset", sorted(tokenized_datasets)),
        ]
        total = sum(len(refs) for _, refs in categories)
        return TokenizerUsageOverview(
            tokenizer_id=tokenizer_id,
            name=record.name,
            created_at=record.created_at,
            requested_vocab_size=record.requested_vocab_size,
            actual_vocab_size=record.actual_vocab_size,
            trained_on_dataset_id=record.trained_on_dataset_id,
            referenced=total > 0,
            total_references=total,
            categories=[ArtifactUsageCategory(category=c, references=r)
                        for c, r in categories])

    def _scope_data_artifact(self, loader, artifact_id: str, kind: str):
        """Resolve ONE data artifact through its registry loader with
        the M61 scope convention: missing -> FileNotFoundError (404);
        unparseable manifest (registry-invisible — the registry
        listings skip it) -> FileNotFoundError with an explicit
        unreadable-manifest message (404, nothing deleted), never a
        500."""
        try:
            return loader(artifact_id)
        except FileNotFoundError:
            raise
        except ValueError as exc:
            raise FileNotFoundError(
                f"{kind} '{artifact_id}' not found (manifest unreadable"
                f" — registry-invisible)") from exc

    @staticmethod
    def _artifact_files(directory) -> tuple[list[str], int]:
        """Ordered (sorted relative posix paths, hidden crash-residue
        entries excluded) file list + total byte size of ONE artifact
        directory — the M63-boundary walk, read-only."""
        files: list[str] = []
        total = 0
        if directory.exists():
            for p in sorted(directory.rglob("*")):
                rel = p.relative_to(directory)
                if any(part.startswith(".") for part in rel.parts):
                    continue
                if p.is_file():
                    files.append(rel.as_posix())
                    total += p.stat().st_size
        return files, total

    def dataset_retention_overview(
            self, dataset_id: str) -> DatasetRetentionOverview:
        """Read-only live-computed retention overview of ONE dataset
        (M65): the deletion-readiness view — identity, the ordered
        artifact files + total bytes of the dataset's OWN directory,
        the M2 integrity-verification outcome, ``deletable`` (True iff
        integrity passes AND the ONE M64 reference analysis finds
        nothing) and the ordered blockers (the SAME list the DELETE
        guard refuses on). A corrupt dataset is never deletable; an
        unknown or registry-invisible dataset -> FileNotFoundError
        (404 at the API). Zero storage, zero mutation, byte-identical
        over unchanged state."""
        meta = self._scope_data_artifact(self.datasets.load_meta,
                                         dataset_id, "dataset")
        files, nbytes = self._artifact_files(
            self.storage.dataset_dir(dataset_id))
        try:
            integrity_verified = (
                self.datasets.verify(dataset_id).get("status") == "ok")
        except Exception:
            integrity_verified = False
        blockers = self.dataset_deletion_blockers(dataset_id)
        return DatasetRetentionOverview(
            dataset_id=dataset_id,
            name=meta.name,
            created_at=meta.created_at,
            version_count=len(meta.versions),
            latest_version=meta.latest_version,
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

    def tokenizer_retention_overview(
            self, tokenizer_id: str) -> TokenizerRetentionOverview:
        """Read-only live-computed retention overview of ONE tokenizer
        (M65): the deletion-readiness view — identity, the ordered
        artifact files + total bytes, the content-hash
        integrity-verification outcome, ``deletable`` and the ordered
        blockers (the SAME list the DELETE guard refuses on). A
        corrupt tokenizer is never deletable; an unknown or
        registry-invisible tokenizer -> FileNotFoundError (404 at the
        API). Zero storage, zero mutation, byte-identical over
        unchanged state."""
        record = self._scope_data_artifact(self.tokenizers.load,
                                           tokenizer_id, "tokenizer")
        files, nbytes = self._artifact_files(
            self.storage.tokenizer_dir(tokenizer_id))
        try:
            integrity_verified = (
                self.tokenizers.verify(tokenizer_id).get("status") == "ok")
        except Exception:
            integrity_verified = False
        blockers = self.tokenizer_deletion_blockers(tokenizer_id)
        return TokenizerRetentionOverview(
            tokenizer_id=tokenizer_id,
            name=record.name,
            created_at=record.created_at,
            requested_vocab_size=record.requested_vocab_size,
            actual_vocab_size=record.actual_vocab_size,
            trained_on_dataset_id=record.trained_on_dataset_id,
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

    # Canonical M66 model usage category order (fixed and
    # deterministic). The first six are INTERNAL model-scoped families
    # (persisted inside models/<id>/ — ownership); the last five are
    # EXTERNAL root-level families persisting the model id OUTSIDE the
    # model directory — exactly the surface the M67 deletion guard
    # refuses on. (M67 inspection proved the M66 surface incomplete by
    # one family: registered gate policies persist ``policy.model_id``
    # at policies/<id>/ and are resolution-checked against it — the
    # same inert-definition-binds-a-model pattern as workflow recipes
    # — so ``policy`` joins the external tail.)
    MODEL_USAGE_CATEGORIES = (
        "training_run",      # the model manifest's own RunProvenance
        "checkpoint",        # models/<id>/checkpoints/
        "workflow",          # models/<id>/workflows/
        "evaluation",        # models/<id>/evaluations/
        "comparison",        # models/<id>/comparisons/
        "gate",              # models/<id>/gates/
        "suite_run",         # suite-runs/<id> (root-level)
        "sample",            # samples/<model_id>/ (root-level)
        "sample_quality",    # sample-evaluations/<model_id>/ (root-level)
        "workflow_recipe",   # workflow-recipes/<id> stage configs (root)
        "policy",            # policies/<id> registry policy (root)
    )
    MODEL_USAGE_INTERNAL_CATEGORIES = frozenset(MODEL_USAGE_CATEGORIES[:6])

    # Canonical M69 model-owned record families (the six INTERNAL
    # model-scoped categories of the M66 usage) and the canonical
    # REFERENCING-family order. "model" = the model manifest's own
    # persisted pointers (latest_checkpoint / best_checkpoint);
    # "checkpoint" = a surviving checkpoint's parent_checkpoint_id
    # (LINEAGE); "run_provenance" = the manifest's run-provenance
    # checkpoint pointers (LINEAGE); "workflow" = M7 stage artifacts
    # and suggestions; evaluation/comparison/gate = the M4/M5/M6
    # evidence records; suite_run/sample/sample_quality = the
    # ROOT-LEVEL families whose records persist model-owned record
    # ids OUTSIDE models/<id>/ (external — they survive model-record
    # deletion, exactly the future per-record blocker surface).
    MODEL_RECORD_CATEGORIES = (
        "training_run",      # the model manifest's own RunProvenance
        "checkpoint",        # models/<id>/checkpoints/
        "workflow",          # models/<id>/workflows/
        "evaluation",        # models/<id>/evaluations/
        "comparison",        # models/<id>/comparisons/
        "gate",              # models/<id>/gates/
    )
    RECORD_REFERENCE_CATEGORIES = (
        "model",
        "checkpoint",
        "run_provenance",
        "workflow",
        "evaluation",
        "comparison",
        "gate",
        "suite_run",
        "sample",
        "sample_quality",
    )
    RECORD_REFERENCE_EXTERNAL = frozenset(
        ("suite_run", "sample", "sample_quality"))
    # The M61-classified LINEAGE edges: informational history that
    # never blocks checkpoint deletion (nothing loads state through
    # them) — reported by M69 under their own categories so the
    # future per-record blocker surface stays obvious.
    RECORD_REFERENCE_LINEAGE = frozenset(
        (("checkpoint", "checkpoint"),     # parent_checkpoint_id
         ("run_provenance", "checkpoint")))  # provenance pointers

    def model_records_usage_overview(
            self, model_id: str) -> ModelRecordsUsageOverview:
        """Read-only live-computed usage overview of ONE model's
        OWNED records (M69): for every record of the six model-scoped
        families (training runs, checkpoints, workflows, evaluations,
        comparisons, gate decisions) every persisted record that
        references it — internal references (the model manifest's
        pointers, checkpoint lineage, run-provenance lineage,
        workflow stage artifacts, and the evaluation / comparison /
        gate evidence records, all inside models/<id>/) plus the
        EXTERNAL root-level references that survive model-record
        deletion (suite-run states and probe results, samples,
        sample-quality measurements — the future per-record blocker
        surface). Computed from the ONE authoritative listings on
        every call (no stored index, no second scanner); references
        are unique (category, reference_id) pairs in the canonical
        category order then id order; records are sorted by id.
        Zero storage, zero mutation, byte-identical over unchanged
        state; unknown or registry-invisible (unparseable manifest)
        model -> FileNotFoundError (404 at the API)."""
        record = self._scope_data_artifact(self.storage.load_record,
                                           model_id, "model")
        refs: dict[tuple[str, str], set[tuple[str, str]]] = {}

        def add(family: str, rid: str, category: str,
                ref_id: str) -> None:
            refs.setdefault((family, rid), set()).add((category, ref_id))

        # the model manifest's own persisted pointers
        if record.latest_checkpoint:
            add("checkpoint", record.latest_checkpoint, "model", model_id)
        if record.best_checkpoint:
            add("checkpoint", record.best_checkpoint, "model", model_id)
        # run-provenance LINEAGE pointers (informational, M61)
        for p in record.training_provenance:
            for cid in (p.parent_checkpoint_id,
                        p.initial_checkpoint_id,
                        p.final_checkpoint_id,
                        p.rolled_back_to):
                if cid:
                    add("checkpoint", cid, "run_provenance", model_id)
        # checkpoints: run origin + parent LINEAGE
        for c in self.training.list_checkpoints(model_id):
            add("training_run", c.run_id, "checkpoint", c.checkpoint_id)
            if c.parent_checkpoint_id:
                add("checkpoint", c.parent_checkpoint_id,
                    "checkpoint", c.checkpoint_id)
        # evaluations: the measured checkpoint state
        for e in self.list_evaluations(model_id):
            if e.checkpoint_id:
                add("checkpoint", e.checkpoint_id,
                    "evaluation", e.eval_id)
        # comparisons: both sides' checkpoint + evaluation ids
        for c in self.list_comparisons(model_id):
            for side in (c.state_a, c.state_b):
                if side.checkpoint_id:
                    add("checkpoint", side.checkpoint_id,
                        "comparison", c.comparison_id)
                add("evaluation", side.evaluation_id,
                    "comparison", c.comparison_id)
        # gates: candidate/baseline sides + the compared comparison
        for g in self.list_gate_decisions(model_id):
            for side in (g.candidate, g.baseline):
                if side is not None:
                    if side.checkpoint_id:
                        add("checkpoint", side.checkpoint_id,
                            "gate", g.decision_id)
                    add("evaluation", side.evaluation_id,
                        "gate", g.decision_id)
            if g.comparison_id:
                add("comparison", g.comparison_id,
                    "gate", g.decision_id)
        # workflows: stage artifacts chain the underlying records
        for w in self.list_workflows(model_id):
            for st in w.stages:
                art = st.artifact
                if art is None:
                    continue
                if art.kind == ArtifactKind.TRAINING_REPORT:
                    add("training_run", art.artifact_id,
                        "workflow", w.workflow_id)
                elif art.kind == ArtifactKind.EVALUATION:
                    add("evaluation", art.artifact_id,
                        "workflow", w.workflow_id)
                elif art.kind == ArtifactKind.COMPARISON:
                    add("comparison", art.artifact_id,
                        "workflow", w.workflow_id)
                elif art.kind == ArtifactKind.GATE_DECISION:
                    add("gate", art.artifact_id,
                        "workflow", w.workflow_id)
                if art.checkpoint_id:
                    add("checkpoint", art.checkpoint_id,
                        "workflow", w.workflow_id)
            if w.suggested_checkpoint_id:
                add("checkpoint", w.suggested_checkpoint_id,
                    "workflow", w.workflow_id)
        # suite runs (EXTERNAL): evaluated state + probe evaluations
        for r in self.list_suite_runs(model_id):
            if r.state.checkpoint_id:
                add("checkpoint", r.state.checkpoint_id,
                    "suite_run", r.suite_run_id)
            for pr in r.results:
                if pr.evaluation_id:
                    add("evaluation", pr.evaluation_id,
                        "suite_run", r.suite_run_id)
        # samples (EXTERNAL): the generating checkpoint
        for s in self.list_samples(model_id):
            add("checkpoint", s.checkpoint_id, "sample", s.sample_id)
        # sample-quality measurements (EXTERNAL): the measured state
        for sq in self.list_sample_evaluations(model_id):
            add("checkpoint", sq.checkpoint_id,
                "sample_quality", sq.evaluation_id)

        order = {c: i for i, c in enumerate(self.RECORD_REFERENCE_CATEGORIES)}
        families = {
            "training_run": sorted(p.run_id for p in
                                   record.training_provenance),
            "checkpoint": sorted(c.checkpoint_id for c in
                                 self.training.list_checkpoints(model_id)),
            "workflow": sorted(w.workflow_id for w in
                               self.list_workflows(model_id)),
            "evaluation": sorted(e.eval_id for e in
                                 self.list_evaluations(model_id)),
            "comparison": sorted(c.comparison_id for c in
                                 self.list_comparisons(model_id)),
            "gate": sorted(g.decision_id for g in
                           self.list_gate_decisions(model_id)),
        }
        categories: list[ModelOwnedRecordCategory] = []
        total_refs = internal_refs = external_refs = 0
        total_records = 0
        for family in self.MODEL_RECORD_CATEGORIES:
            entries: list[ModelOwnedRecordUsage] = []
            for rid in families[family]:
                pairs = sorted(refs.get((family, rid), ()),
                               key=lambda cr: (order[cr[0]], cr[1]))
                rlist = [RecordReference(
                    category=cat, reference_id=ref,
                    external=cat in self.RECORD_REFERENCE_EXTERNAL)
                    for cat, ref in pairs]
                ext = sum(1 for r in rlist if r.external)
                entries.append(ModelOwnedRecordUsage(
                    record_id=rid,
                    references=rlist,
                    total_references=len(rlist),
                    internal_references=len(rlist) - ext,
                    external_references=ext))
                total_refs += len(rlist)
                internal_refs += len(rlist) - ext
                external_refs += ext
            total_records += len(entries)
            categories.append(ModelOwnedRecordCategory(
                category=family, records=entries))
        return ModelRecordsUsageOverview(
            model_id=model_id,
            name=record.name,
            total_records=total_records,
            total_references=total_refs,
            internal_references=internal_refs,
            external_references=external_refs,
            categories=categories)

    # The M70 deletable model-owned record families: the four
    # directory-backed evidence records. Checkpoints keep their M61
    # guard; training runs are manifest entries (not individually
    # deletable records) — both deliberately OUT OF M70 SCOPE.
    MODEL_RECORD_DELETABLE_CATEGORIES = ("workflow", "evaluation",
                                         "comparison", "gate")

    def _deletable_record_scope(self, model_id: str, category: str,
                                record_id: str):
        """Resolve ONE deletable model-owned record (M70 scope step):
        validate the category against the M70 deletable families
        (ValueError otherwise — checkpoints and training runs are out
        of scope), then scope through the family's authoritative
        getter (unknown model, unknown record, or registry-invisible
        (unparseable manifest) record -> FileNotFoundError, nothing
        deleted). Returns (record, family_engine, record_dir,
        result_hash_checker)."""
        if category == "workflow":
            getter = lambda rid: self.workflows.get_workflow(  # noqa: E731
                model_id, rid)
            engine, rdir = self.workflows, self.workflows._workflow_dir(
                model_id, record_id)
            check = self.workflows.result_hash
        elif category == "evaluation":
            getter = lambda rid: self.evaluation.get_evaluation(  # noqa: E731
                model_id, rid)
            engine, rdir = self.evaluation, self.evaluation._eval_dir(
                model_id, record_id)
            check = self.evaluation.result_hash
        elif category == "comparison":
            getter = lambda rid: self.comparison.get_comparison(  # noqa: E731
                model_id, rid)
            engine, rdir = self.comparison, self.comparison._comp_dir(
                model_id, record_id)
            check = self.comparison.result_hash
        elif category == "gate":
            getter = lambda rid: self.gates.get_decision(  # noqa: E731
                model_id, rid)
            engine, rdir = self.gates, self.gates._gate_dir(
                model_id, record_id)
            check = self.gates.result_hash
        else:
            raise ValueError(
                f"record family '{category}' has no M70 deletion "
                f"(deletable families: "
                f"{', '.join(self.MODEL_RECORD_DELETABLE_CATEGORIES)})")
        record = self._scope_data_artifact(getter, record_id, category)
        return record, engine, rdir, check

    def model_record_deletion_blockers(
            self, model_id: str, category: str, record_id: str
    ) -> list[ModelRecordDeletionBlocker]:
        """The M70 deletion guard's ordered blocker list: EXACTLY the
        NON-LINEAGE persisted references of the ONE M69 usage
        analysis — the same categories, the same reference ids, the
        same canonical order (category order, then sorted reference
        ids). The M61-classified LINEAGE edges (checkpoint parents,
        run provenance) never block (and never point at the four M70
        families anyway — they target checkpoints); every other
        reference the M69 overview reports for the record is ONE
        typed blocker: a record is blocked by every persisted
        DEPENDENT that would be orphaned (comparison/gate sides and
        suite-run probe results referencing an evaluation; gate
        decisions and workflow stage artifacts referencing their
        targets). Nothing protected that is not shown, nothing shown
        that is not protected. Zero storage, zero mutation."""
        record, _, _, _ = self._deletable_record_scope(
            model_id, category, record_id)
        overview = self.model_records_usage_overview(model_id)
        entry = next(
            (r for c in overview.categories if c.category == category
             for r in c.records if r.record_id == record_id), None)
        blockers: list[ModelRecordDeletionBlocker] = []
        if entry is not None:
            for ref in entry.references:
                if ref.category in self.RECORD_REFERENCE_LINEAGE:
                    continue  # M61 informational lineage never blocks
                blockers.append(ModelRecordDeletionBlocker(
                    category=ref.category,
                    reference_id=ref.reference_id,
                    detail=self._record_blocker_detail(ref.category,
                                                       ref.reference_id)))
        return blockers

    @staticmethod
    def _record_blocker_detail(category: str, ref_id: str) -> str:
        """Short authoritative identifying detail for ONE M70 blocker
        (the referencing record's family, M67 pattern)."""
        return {
            "model": "model manifest pointer",
            "checkpoint": f"checkpoint '{ref_id}'",
            "run_provenance": "run provenance",
            "workflow": f"workflow '{ref_id}' stage artifact",
            "evaluation": f"evaluation '{ref_id}'",
            "comparison": f"comparison '{ref_id}'",
            "gate": f"gate decision '{ref_id}'",
            "suite_run": f"suite run '{ref_id}'",
            "sample": f"sample '{ref_id}'",
            "sample_quality": f"sample-quality measurement '{ref_id}'",
        }.get(category, f"record '{ref_id}'")

    def model_record_retention_overview(
            self, model_id: str, category: str,
            record_id: str) -> ModelRecordRetentionOverview:
        """Read-only live-computed retention overview of ONE
        model-owned record (M70): the deletion-readiness view —
        identity, the ordered artifact files + total bytes of the
        record's OWN directory, the record's result-hash integrity
        outcome, ``deletable`` (True iff integrity passes AND the ONE
        M69 analysis finds no non-lineage reference) and the ordered
        blockers (the SAME list the DELETE guard refuses on). A
        tampered record is never deletable; an unknown or
        registry-invisible (unparseable manifest) record ->
        FileNotFoundError (404 at the API). Zero storage, zero
        mutation, byte-identical over unchanged state."""
        record, _, rdir, check = self._deletable_record_scope(
            model_id, category, record_id)
        files, nbytes = self._artifact_files(rdir)
        integrity_verified = check(record) == record.result_hash
        blockers = self.model_record_deletion_blockers(
            model_id, category, record_id)
        return ModelRecordRetentionOverview(
            model_id=model_id,
            category=category,
            record_id=record_id,
            created_at=record.created_at,
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

    def delete_model_record(self, model_id: str, category: str,
                            record_id: str) -> ModelRecordDeletionResult:
        """Explicit VERIFIED model-owned record retention (M70):
        remove ONE workflow / evaluation / comparison / gate decision
        — only after proving, live, that nothing depends on it. The
        full guard, in order: (1) scope through the family getter
        (unknown model, unknown record, or registry-invisible
        (unparseable manifest) record -> FileNotFoundError, nothing
        deleted); (2) INTEGRITY FIRST (the M61/M65/M67/M68 ordering):
        the record's persisted ``result_hash`` must reproduce from its
        semantic payload — a tampered/corrupt record is REFUSED with
        RuntimeError (409), never deletable, no force flag; (3) the
        LIVE dependency analysis (``model_record_deletion_blockers``
        — EXACTLY the NON-LINEAGE references of the ONE M69 usage
        overview: comparison/gate sides, workflow stage artifacts and
        suite-run probe results referencing the record; ANY reference
        -> ValueError listing the ordered typed blockers, so deletion
        can never orphan a persisted dependent. The M61 LINEAGE edges
        never block); (4) ATOMIC removal of the record's own
        directory only (the family engine's ``delete`` — measure +
        ``atomic_delete_dir``, the ``DatasetEngine.delete`` pattern).
        No cascade, no force, no bulk mode — exactly the one
        explicitly requested record. Checkpoints keep their M61
        guard; training runs are manifest entries; model deletion
        (M67) and every M68 surface are untouched."""
        record, engine, rdir, check = self._deletable_record_scope(
            model_id, category, record_id)
        if check(record) != record.result_hash:
            raise RuntimeError(
                f"{category} '{record_id}' of model '{model_id}' "
                f"failed integrity verification — refusing to delete "
                f"corrupted storage")
        blockers = self.model_record_deletion_blockers(
            model_id, category, record_id)
        if blockers:
            summary = "; ".join(f"{b.category}: {b.reference_id}"
                                for b in blockers)
            raise ValueError(
                f"{category} '{record_id}' of model '{model_id}' is "
                f"referenced and cannot be deleted — {summary}")
        files, nbytes = engine.delete(model_id, record_id)
        log.info("M70 verified %s deletion %s/%s (%d files, %d bytes)",
                 category, model_id, record_id, files, nbytes)
        return ModelRecordDeletionResult(
            model_id=model_id, category=category, record_id=record_id,
            files_removed=files, bytes_reclaimed=nbytes)

    # ------------------------------------------------------------------ #
    # M71: explicit definition retention (recipes / policies / suites)
    # ------------------------------------------------------------------ #

    # The M71 definition families (root-level registries) and the
    # canonical DEPENDENT categories that block deletion: a workflow
    # run's recipe provenance; a COMPOSITE recipe's composition
    # reference; a gate decision's registry policy provenance; a
    # suite run's suite id. Every blocker derivation reuses the ONE
    # canonical model-scoped filters (M35/M23/M21) plus the recipe
    # registry listing — never a second scanner.
    DEFINITION_DELETABLE_FAMILIES = ("workflow_recipe", "gate_policy",
                                     "probe_suite")
    DEFINITION_DEPENDENT_CATEGORIES = (
        ("workflow_recipe", ("workflow", "workflow_recipe")),
        ("gate_policy", ("gate", "workflow_recipe")),
        ("probe_suite", ("suite_run", "workflow_recipe")),
    )

    def _definition_scope(self, family: str, definition_id: str):
        """M71 scope + integrity resolver: the family's getter through
        the ``_scope_data_artifact`` convention (unknown or
        registry-invisible (unparseable manifest) definition ->
        FileNotFoundError, nothing deleted) plus the record's OWN
        content-hash check as a callable. Returns (record, rdir,
        hash_ok)."""
        if family == "workflow_recipe":
            record = self._scope_data_artifact(self.recipes.get,
                                               definition_id, "recipe")
            rdir = self.recipes._recipe_dir(definition_id)

            def check(r) -> bool:
                if r.composition:
                    pairs = [(ref.recipe_id, ref.config_hash)
                             for ref in r.composition]
                    deps = [ref.config_hash for ref in r.composition]
                    return self.recipes._composite_hash(
                        r.stages, pairs, deps) == r.config_hash
                return recipe_config_hash(r.stages) == r.config_hash
            return record, rdir, check
        if family == "gate_policy":
            record = self._scope_data_artifact(
                self.policies.get_policy, definition_id, "policy")
            rdir = self.policies._policy_dir(definition_id)

            def check(r) -> bool:
                return policy_config_hash(r.policy) == r.config_hash
            return record, rdir, check
        if family == "probe_suite":
            record = self._scope_data_artifact(
                self.policies.get_suite, definition_id, "probe suite")
            rdir = self.policies._suite_dir(definition_id)

            def check(r) -> bool:
                return probes_hash(r.probes) == r.probes_hash
            return record, rdir, check
        raise ValueError(
            f"family '{family}' has no definition lifecycle (M71 "
            f"families: {', '.join(self.DEFINITION_DELETABLE_FAMILIES)})")

    def definition_model_ids(self, family: str,
                             definition_id: str) -> list[str]:
        """The models ONE definition references (M71): a recipe names
        every model its stage configs bind (the ONE M66 analysis,
        ``_recipe_model_ids``); a policy names its target model; a
        probe suite names NONE (a suite binds a model only at run
        time — the M66 distinction). Sorted, unique. Read-only."""
        record, _, check = self._definition_scope(family, definition_id)
        if family == "workflow_recipe":
            return sorted(self._recipe_model_ids(record))
        if family == "gate_policy":
            return [record.policy.model_id]
        return []

    def definition_deletion_blockers(
            self, family: str,
            definition_id: str) -> list[DefinitionDeletionBlocker]:
        """The M71 deletion guard's ordered blocker list: every
        persisted record that references the definition — workflow
        runs with the recipe's provenance id (the ONE M35
        ``list_workflows_for_recipe`` filter, across every model),
        COMPOSITE recipes whose composition references it (the recipe
        registry listing, the M14 reference set), gate decisions with
        the policy's registry provenance id (the ONE M23
        ``list_gate_decisions_for_policy`` filter), suite runs with
        the suite's id (the ONE M21 ``list_suite_runs_for_suite``
        filter) and — the STRUCTURAL directions, persisted recipe
        stage configs — recipes whose gate stage names the policy id
        or whose suite-run stage names the suite id (deleting the
        definition would break every future run of that recipe at
        resolution). Deleting the definition would orphan exactly
        these records' references, so each is ONE typed blocker.
        Deterministic (category order, then sorted reference ids);
        zero storage, zero mutation."""
        record, _, _ = self._definition_scope(family, definition_id)
        found: list[tuple[str, str]] = []
        if family == "workflow_recipe":
            for model_id in sorted(self.storage.model_ids()):
                for w in self.list_workflows_for_recipe(
                        model_id, definition_id):
                    found.append(("workflow", w.workflow_id))
            for r in self.recipes.list():
                if r.recipe_id == definition_id:
                    continue
                if r.composition and any(
                        ref.recipe_id == definition_id
                        for ref in r.composition):
                    found.append(("workflow_recipe", r.recipe_id))
        elif family == "gate_policy":
            for model_id in sorted(self.storage.model_ids()):
                for g in self.list_gate_decisions_for_policy(
                        model_id, definition_id):
                    found.append(("gate", g.decision_id))
            for r in self.recipes.list():
                if r.recipe_id == definition_id:
                    continue
                if any(s.type == "gate" and s.gate is not None
                       and s.gate.policy_id == definition_id
                       for s in r.stages):
                    found.append(("workflow_recipe", r.recipe_id))
        else:  # probe_suite
            for model_id in sorted(self.storage.model_ids()):
                for r in self.list_suite_runs_for_suite(
                        model_id, definition_id):
                    found.append(("suite_run", r.suite_run_id))
            for r in self.recipes.list():
                if r.recipe_id == definition_id:
                    continue
                if any(s.type == "suite_run" and s.suite_run is not None
                       and s.suite_run.suite_id == definition_id
                       for s in r.stages):
                    found.append(("workflow_recipe", r.recipe_id))
        order = {c: i for i, c in enumerate(
            dict(self.DEFINITION_DEPENDENT_CATEGORIES)[family])}
        return [DefinitionDeletionBlocker(
            category=cat, reference_id=rid,
            detail=self._definition_blocker_detail(family, cat, rid))
            for cat, rid in sorted(found, key=lambda cr: (
                order[cr[0]], cr[1]))]

    @staticmethod
    def _definition_blocker_detail(family: str, category: str,
                                   ref_id: str) -> str:
        """Short authoritative identifying detail for ONE M71 blocker
        (the referencing record's family, the M67/M70 pattern). The
        ``workflow_recipe`` category detail names the exact structural
        edge: a COMPOSITE recipe's composition reference (recipes
        family), a gate stage's policy reference (policies family) or
        a suite-run stage's suite reference (suites family)."""
        return {
            ("workflow_recipe", "workflow"):
                f"workflow run '{ref_id}' recipe provenance",
            ("workflow_recipe", "workflow_recipe"):
                f"composite recipe '{ref_id}' composition",
            ("gate_policy", "gate"):
                f"gate decision '{ref_id}' policy provenance",
            ("gate_policy", "workflow_recipe"):
                f"recipe '{ref_id}' gate stage policy reference",
            ("probe_suite", "suite_run"):
                f"suite run '{ref_id}'",
            ("probe_suite", "workflow_recipe"):
                f"recipe '{ref_id}' suite-run stage suite reference",
        }.get((family, category), f"record '{ref_id}'")

    def definition_retention_overview(
            self, family: str,
            definition_id: str) -> DefinitionRetentionOverview:
        """Read-only live-computed retention overview of ONE root-level
        definition (M71): the deletion-readiness view — identity, the
        bound model ids (where applicable), the ordered artifact
        files + total bytes of the definition's OWN directory, the
        content-hash integrity outcome, ``deletable`` (True iff
        integrity passes AND no persisted record references the
        definition) and the ordered blockers (the SAME list the
        DELETE guard refuses on). A tampered definition is never
        deletable; an unknown or registry-invisible definition ->
        FileNotFoundError (404 at the API). Zero storage, zero
        mutation, byte-identical over unchanged state."""
        record, rdir, check = self._definition_scope(family,
                                                     definition_id)
        files, nbytes = self._artifact_files(rdir)
        integrity_verified = check(record)
        blockers = self.definition_deletion_blockers(family,
                                                     definition_id)
        return DefinitionRetentionOverview(
            family=family,
            definition_id=definition_id,
            created_at=record.created_at,
            model_ids=self.definition_model_ids(family, definition_id),
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

    def delete_definition(self, family: str,
                          definition_id: str) -> DefinitionDeletionResult:
        """Explicit VERIFIED definition retention (M71): remove ONE
        root-level definition — a workflow recipe, a gate policy or a
        probe suite — only after proving, live, that nothing
        references it. The full guard, in order: (1) scope through
        the family getter (unknown or registry-invisible definition
        -> FileNotFoundError, nothing deleted); (2) INTEGRITY FIRST
        (the M61/M65/M67/M68/M70 ordering): the definition's
        persisted content hash (``config_hash`` / ``probes_hash``)
        must reproduce from its semantic payload — a tampered or
        corrupt definition is REFUSED with RuntimeError (409), never
        deletable, no force flag; (3) the LIVE dependent analysis
        (``definition_deletion_blockers`` — the ONE canonical
        filters: workflow runs with recipe provenance, composite
        recipes' composition references, gate decisions with policy
        provenance, suite runs with the suite id, recipes whose stage
        configs structurally name the policy/suite; ANY reference ->
        ValueError listing the ordered typed blockers, so deletion
        can never orphan a persisted record); (4) ATOMIC removal of
        the definition's OWN directory only (measure +
        ``atomic_delete_dir``). No cascade, no force, no bulk mode.
        Deleting a recipe/policy also removes the MODEL external
        reference it represents (the M66 ``workflow_recipe`` /
        ``policy`` categories shrink LIVE — the M67 model guard
        reads that analysis on every call, no guard code changes)."""
        record, _, check = self._definition_scope(family, definition_id)
        if not check(record):
            raise RuntimeError(
                f"{family} '{definition_id}' failed integrity "
                f"verification — refusing to delete corrupted storage")
        blockers = self.definition_deletion_blockers(family,
                                                     definition_id)
        if blockers:
            summary = "; ".join(f"{b.category}: {b.reference_id}"
                                for b in blockers)
            raise ValueError(
                f"{family} '{definition_id}' is referenced and cannot "
                f"be deleted — {summary}")
        if family == "workflow_recipe":
            files, nbytes = self.recipes.delete(definition_id)
        elif family == "gate_policy":
            files, nbytes = self.policies.delete_policy(definition_id)
        else:
            files, nbytes = self.policies.delete_suite(definition_id)
        log.info("M71 verified %s deletion %s (%d files, %d bytes)",
                 family, definition_id, files, nbytes)
        return DefinitionDeletionResult(
            family=family, definition_id=definition_id,
            files_removed=files, bytes_reclaimed=nbytes)

    # The canonical M72 project retention family order: every
    # persistable family in the deletion lifecycle matrix, plus the
    # ONE record family without a lifecycle (training_run — the
    # model manifest's own run provenance: ownership, it goes WITH
    # the model; deletion_supported=False).
    PROJECT_RETENTION_FAMILIES = (
        "model",              # M67 (size includes owned records)
        "dataset",            # M65 (includes its tokenized versions)
        "tokenizer",          # M65
        "workflow_recipe",    # M71
        "gate_policy",        # M71
        "probe_suite",        # M71
        "training_run",       # count-only (no deletion lifecycle)
        "checkpoint",         # M61/M62
        "workflow",           # M70
        "evaluation",         # M70
        "comparison",         # M70
        "gate",               # M70
        "suite_run",          # M68 (root-level)
        "sample",             # M68 (root-level)
        "sample_quality",     # M68 (root-level)
    )

    def _project_retention_core(self, skip=frozenset(),
                                deletable_override=None,
                                model_storage_delta=None):
        """The ONE M72 project-retention aggregation over the
        existing per-artifact views (M62/M65/M67/M68/M69/M70/M71),
        parameterized for the M73 impact preview's SHADOW state:
        ``skip`` = (family, artifact_id) keys treated as already
        deleted; ``deletable_override`` = keys flipped to deletable
        (the M73 first-level shadow result); ``model_storage_delta``
        = model_id -> (files, bytes) subtracted from a model's own
        storage when one of its owned artifacts is skipped (the
        record/checkpoint directory is part of the model's). With
        all parameters empty this is EXACTLY the M72 overview
        computation — the public overview and the M73 impact preview
        share ONE code path, ONE scanner and ONE overlap rule; the
        shadow never touches the filesystem. Returns
        (agg, reclaimable_files, reclaimable_bytes)."""
        deletable_override = deletable_override or {}
        model_storage_delta = model_storage_delta or {}

        agg = {f: dict(count=0, files=0, size=0, deletable=0,
                       rfiles=0, rbytes=0)
               for f in self.PROJECT_RETENTION_FAMILIES}

        def add(family, key, files: int, size: int, deletable: bool):
            d = agg[family]
            d["count"] += 1
            d["files"] += files
            d["size"] += size
            if deletable:
                d["deletable"] += 1
                d["rfiles"] += files
                d["rbytes"] += size

        # root-level data artifacts + definitions (global registries)
        for d in self.datasets.list():
            if ("dataset", d.id) in skip:
                continue
            v = self.dataset_retention_overview(d.id)
            add("dataset", d.id, len(v.files), v.size_bytes,
                v.deletable or ("dataset", d.id) in deletable_override)
        for t in self.tokenizers.list():
            if ("tokenizer", t.id) in skip:
                continue
            v = self.tokenizer_retention_overview(t.id)
            add("tokenizer", t.id, len(v.files), v.size_bytes,
                v.deletable or ("tokenizer", t.id) in deletable_override)
        for r in self.recipes.list():
            if ("workflow_recipe", r.recipe_id) in skip:
                continue
            v = self.definition_retention_overview(
                "workflow_recipe", r.recipe_id)
            add("workflow_recipe", r.recipe_id, len(v.files), v.size_bytes,
                v.deletable or ("workflow_recipe", r.recipe_id)
                in deletable_override)
        for pol in self.policies.list_policies():
            if ("gate_policy", pol.policy_id) in skip:
                continue
            v = self.definition_retention_overview(
                "gate_policy", pol.policy_id)
            add("gate_policy", pol.policy_id, len(v.files), v.size_bytes,
                v.deletable or ("gate_policy", pol.policy_id)
                in deletable_override)
        for s in self.policies.list_suites():
            if ("probe_suite", s.suite_id) in skip:
                continue
            v = self.definition_retention_overview(
                "probe_suite", s.suite_id)
            add("probe_suite", s.suite_id, len(v.files), v.size_bytes,
                v.deletable or ("probe_suite", s.suite_id)
                in deletable_override)

        # per model: the model itself + its owned records (the ONE
        # M69 usage listing enumerates the record ids; the ONE M62/
        # M70 views carry the deletion-readiness)
        reclaimable_files = reclaimable_bytes = 0
        reclaimable_files += agg["dataset"]["rfiles"] + \
            agg["tokenizer"]["rfiles"] + agg["workflow_recipe"][
                "rfiles"] + agg["gate_policy"]["rfiles"] + \
            agg["probe_suite"]["rfiles"]
        reclaimable_bytes += agg["dataset"]["rbytes"] + \
            agg["tokenizer"]["rbytes"] + agg["workflow_recipe"][
                "rbytes"] + agg["gate_policy"]["rbytes"] + \
            agg["probe_suite"]["rbytes"]
        record_families = ("checkpoint", "workflow", "evaluation",
                           "comparison", "gate")
        for m in self.list_models():
            if ("model", m.id) in skip:
                continue  # the whole model directory is shadow-gone
            v = self.model_retention_overview(m.id)
            d_files, d_bytes = model_storage_delta.get(m.id, (0, 0))
            m_deletable = v.deletable or ("model", m.id) \
                in deletable_override
            add("model", m.id, len(v.files) - d_files,
                v.size_bytes - d_bytes, m_deletable)
            # this model's OWN record reclaim, isolated via before/
            # after snapshots of the cumulative record-family sums
            pre_rf = sum(agg[f]["rfiles"] for f in record_families)
            pre_rb = sum(agg[f]["rbytes"] for f in record_families)
            usage = self.model_records_usage_overview(m.id)
            cats = {c.category: c.records for c in usage.categories}
            agg["training_run"]["count"] += len(
                cats.get("training_run", []))
            ck = self.checkpoint_retention_overview(m.id)
            for e in ck.checkpoints:
                if ("checkpoint", e.checkpoint_id) in skip:
                    continue
                add("checkpoint", e.checkpoint_id, e.files, e.size_bytes,
                    e.deletable or ("checkpoint", e.checkpoint_id)
                    in deletable_override)
            for category in ("workflow", "evaluation", "comparison",
                             "gate"):
                for r in cats.get(category, []):
                    if (category, r.record_id) in skip:
                        continue
                    rv = self.model_record_retention_overview(
                        m.id, category, r.record_id)
                    add(category, r.record_id, len(rv.files),
                        rv.size_bytes, rv.deletable
                        or (category, r.record_id) in deletable_override)
            # root-level M68 records of this model
            for sr in self.list_suite_runs(m.id):
                if ("suite_run", sr.suite_run_id) in skip:
                    continue
                rv = self.suite_run_retention_overview(
                    m.id, sr.suite_run_id)
                add("suite_run", sr.suite_run_id, len(rv.files),
                    rv.size_bytes, rv.deletable
                    or ("suite_run", sr.suite_run_id) in deletable_override)
            for s in self.list_samples(m.id):
                if ("sample", s.sample_id) in skip:
                    continue
                rv = self.sample_retention_overview(m.id, s.sample_id)
                add("sample", s.sample_id, len(rv.files), rv.size_bytes,
                    rv.deletable or ("sample", s.sample_id)
                    in deletable_override)
            for q in self.list_sample_evaluations(m.id):
                if ("sample_quality", q.evaluation_id) in skip:
                    continue
                rv = self.sample_evaluation_retention_overview(
                    m.id, q.evaluation_id)
                add("sample_quality", q.evaluation_id, len(rv.files),
                    rv.size_bytes, rv.deletable
                    or ("sample_quality", q.evaluation_id)
                    in deletable_override)
            # the EXACT model/record overlap rule: a deletable model
            # contributes its WHOLE directory (its records go with
            # it); a blocked model contributes only its own
            # deletable records
            if m_deletable:
                reclaimable_files += len(v.files) - d_files
                reclaimable_bytes += v.size_bytes - d_bytes
            else:
                reclaimable_files += sum(
                    agg[f]["rfiles"] for f in record_families) - pre_rf
                reclaimable_bytes += sum(
                    agg[f]["rbytes"] for f in record_families) - pre_rb
        # root-level M68 records live OUTSIDE models/ — always add
        reclaimable_files += sum(agg[f]["rfiles"] for f in
                                 ("suite_run", "sample",
                                  "sample_quality"))
        reclaimable_bytes += sum(agg[f]["rbytes"] for f in
                                 ("suite_run", "sample",
                                  "sample_quality"))

        return agg, reclaimable_files, reclaimable_bytes

    def project_retention_overview(self) -> ProjectRetentionOverview:
        """Read-only live-computed PROJECT retention inventory (M72):
        the whole deletion surface in ONE view. Per family, in the
        canonical ``PROJECT_RETENTION_FAMILIES`` order: the artifact
        count, the family's OWN storage (files/bytes from the ONE
        retention views — the model family's size INCLUDES its owned
        records, the M67 ownership semantics), the currently
        deletable vs not-deletable counts (``blocked`` = held by
        references or failed integrity) and what deleting every
        currently-deletable artifact of the family would reclaim.
        Project totals: ``total_files``/``total_size_bytes`` come
        from the ONE M63 physical storage walk (the TRUE storage —
        family sizes overlap by ownership and never sum into it),
        ``total_count``/``total_deletable``/``total_blocked`` are
        deterministic sums over the families (training_run
        contributes counts only), and ``reclaimable_files``/
        ``bytes`` are the EXACT result of deleting every
        currently-deletable artifact: a deletable MODEL contributes
        its WHOLE directory (subsuming its records); a blocked model
        contributes only its own deletable records. Every number is
        read from the EXISTING M62/M63/M65/M67/M68/M69/M70/M71
        analyses — no second scanner, nothing persisted, zero
        mutation, byte-identical over unchanged state."""
        agg, reclaimable_files, reclaimable_bytes = \
            self._project_retention_core()


        families = []
        for family in self.PROJECT_RETENTION_FAMILIES:
            d = agg[family]
            supported = family != "training_run"
            families.append(ProjectFamilyRetention(
                family=family,
                deletion_supported=supported,
                count=d["count"],
                files=d["files"],
                size_bytes=d["size"],
                deletable_count=d["deletable"] if supported else 0,
                blocked_count=(d["count"] - d["deletable"])
                if supported else 0,
                reclaimable_files=d["rfiles"] if supported else 0,
                reclaimable_bytes=d["rbytes"] if supported else 0))
        storage = self.project_storage_overview()
        return ProjectRetentionOverview(
            families=families,
            total_count=sum(f.count for f in families),
            total_files=storage.total_files,
            total_size_bytes=storage.total_bytes,
            total_deletable=sum(f.deletable_count for f in families),
            total_blocked=sum(f.blocked_count for f in families),
            reclaimable_files=reclaimable_files,
            reclaimable_bytes=reclaimable_bytes)

    # ------------------------------------------------------------------ #
    # M73: read-only FIRST-LEVEL deletion impact preview
    # ------------------------------------------------------------------ #

    # The families with a VERIFIED deletion lifecycle (every M72
    # family except the lifecycle-less training_run). Only these are
    # eligible for an impact preview; the route layer passes the
    # family explicitly.
    IMPACT_FAMILIES = ("model", "dataset", "tokenizer",
                       "workflow_recipe", "gate_policy", "probe_suite",
                       "checkpoint", "workflow", "evaluation",
                       "comparison", "gate", "suite_run", "sample",
                       "sample_quality")
    # X family -> the M70 blocker category under which X appears in a
    # record dependent's blockers (the M69 referencing-family name)
    IMPACT_RECORD_CATEGORIES = frozenset(
        ("workflow", "comparison", "gate", "suite_run"))

    def _impact_current(self, family: str, artifact_id: str,
                        model_id: str | None):
        """Scope + the family's EXISTING retention view (M73 current
        state): the ONE per-artifact analysis — never a second
        retention engine — uniformized to (deletable, integrity,
        files, size, blockers). Blockers are a 1:1 projection: the
        M67/M70/M71 families carry (category, reference_id, detail)
        verbatim; the reason+detail families (M61 checkpoints, M65
        data artifacts, M68 root records) map reason -> category and
        leave reference_id unset (their native form joins the ids
        into detail). Unknown or registry-invisible artifact ->
        FileNotFoundError (404 at the API)."""
        def blocked(blockers, reason=False):
            return [(b.reason if reason else b.category,
                     None if reason else b.reference_id, b.detail)
                    for b in blockers]
        if family == "model":
            v = self.model_retention_overview(artifact_id)
            return dict(deletable=v.deletable,
                        integrity=v.integrity_verified,
                        files=len(v.files), size=v.size_bytes,
                        blockers=blocked(v.blockers))
        if family == "dataset":
            v = self.dataset_retention_overview(artifact_id)
            return dict(deletable=v.deletable,
                        integrity=v.integrity_verified,
                        files=len(v.files), size=v.size_bytes,
                        blockers=blocked(v.blockers, reason=True))
        if family == "tokenizer":
            v = self.tokenizer_retention_overview(artifact_id)
            return dict(deletable=v.deletable,
                        integrity=v.integrity_verified,
                        files=len(v.files), size=v.size_bytes,
                        blockers=blocked(v.blockers, reason=True))
        if family in ("workflow_recipe", "gate_policy", "probe_suite"):
            v = self.definition_retention_overview(family, artifact_id)
            return dict(deletable=v.deletable,
                        integrity=v.integrity_verified,
                        files=len(v.files), size=v.size_bytes,
                        blockers=blocked(v.blockers))
        if family == "checkpoint":
            ov = self.checkpoint_retention_overview(model_id)
            entry = next((e for e in ov.checkpoints
                          if e.checkpoint_id == artifact_id), None)
            if entry is None:
                raise FileNotFoundError(
                    f"checkpoint '{artifact_id}' not found for model "
                    f"'{model_id}'")
            return dict(deletable=entry.deletable,
                        integrity=entry.integrity_verified,
                        files=entry.files, size=entry.size_bytes,
                        blockers=blocked(entry.blockers, reason=True))
        if family in ("workflow", "evaluation", "comparison", "gate"):
            v = self.model_record_retention_overview(
                model_id, family, artifact_id)
            return dict(deletable=v.deletable,
                        integrity=v.integrity_verified,
                        files=len(v.files), size=v.size_bytes,
                        blockers=blocked(v.blockers))
        if family == "suite_run":
            v = self.suite_run_retention_overview(model_id, artifact_id)
            return dict(deletable=v.deletable,
                        integrity=v.integrity_verified,
                        files=len(v.files), size=v.size_bytes,
                        blockers=blocked(v.blockers, reason=True))
        if family == "sample":
            v = self.sample_retention_overview(model_id, artifact_id)
            return dict(deletable=v.deletable,
                        integrity=v.integrity_verified,
                        files=len(v.files), size=v.size_bytes,
                        blockers=blocked(v.blockers, reason=True))
        v = self.sample_evaluation_retention_overview(model_id,
                                                      artifact_id)
        return dict(deletable=v.deletable, integrity=v.integrity_verified,
                    files=len(v.files), size=v.size_bytes,
                    blockers=blocked(v.blockers, reason=True))

    def _impact_outbound(self, family: str, artifact_id: str,
                         model_id: str | None) -> list[tuple]:
        """The selected artifact's OUTBOUND references — the persisted
        edges (typed record fields, the ONE M61/M64 helpers) whose
        targets have the selected artifact in their blocker lists.
        Returns ordered unique (dep_family, dep_id, dep_model_id,
        reference_category) candidates; membership is verified per
        candidate by ``_impact_evaluate_dependent``. Deleting the
        selected artifact removes exactly these edges from the
        dependents' blocker analyses. Suite-run STAGE artifacts are
        deliberately absent: the created suite run is ROOT-LEVEL and
        survives a workflow deletion; lineage checkpoint parents never
        block; probe suites bind models only at run time."""
        out: set[tuple[str, str, str | None, str]] = set()

        def push(dep_family, dep_id, dep_model, cat):
            out.add((dep_family, dep_id, dep_model, cat))

        if family == "model":
            m = artifact_id
            for info in self.datasets.list():
                u = self.dataset_usage_overview(info.id)
                for c in u.categories:
                    if c.category in ("training_run", "workflow",
                                      "evaluation", "comparison",
                                      "suite_run") and any(
                            r.startswith(f"{m}/") for r in c.references):
                        push("dataset", info.id, None, c.category)
            for t in self.tokenizers.list():
                u = self.tokenizer_usage_overview(t.id)
                for c in u.categories:
                    if c.category in ("training_run", "workflow",
                                      "evaluation", "comparison",
                                      "suite_run", "sample",
                                      "sample_quality") and any(
                            r.startswith(f"{m}/") for r in c.references):
                        push("tokenizer", t.id, None, c.category)
        elif family == "dataset":
            u = self.dataset_usage_overview(artifact_id)
            for c in u.categories:
                if c.category == "tokenized_version":
                    for ref in c.references:  # "v{n}/{tokenizer_id}"
                        push("tokenizer", ref.split("/", 1)[1], None,
                             "tokenized_dataset")
        elif family == "workflow_recipe":
            rec = self._definition_scope("workflow_recipe",
                                         artifact_id)[0]
            for m in self._recipe_model_ids(rec):
                push("model", m, None, "workflow_recipe")
            # a COMPOSITE recipe's composition refs: the referenced
            # recipes carry THIS recipe in their blocker lists (the
            # M71 composition blocker, mirrored here as dependents)
            for ref in (rec.composition or []):
                push("workflow_recipe", ref.recipe_id, None,
                     "workflow_recipe")
        elif family == "gate_policy":
            rec = self.policies.get_policy(artifact_id)
            push("model", rec.policy.model_id, None, "policy")
        elif family == "workflow":
            rec = self.workflows.get_workflow(model_id, artifact_id)
            for st in rec.stages:
                art = st.artifact
                if art is None:
                    continue
                if art.kind == ArtifactKind.EVALUATION:
                    push("evaluation", art.artifact_id, model_id,
                         "workflow")
                elif art.kind == ArtifactKind.COMPARISON:
                    push("comparison", art.artifact_id, model_id,
                         "workflow")
                elif art.kind == ArtifactKind.GATE_DECISION:
                    push("gate", art.artifact_id, model_id, "workflow")
                # SUITE_RUN artifacts are root-level records that
                # SURVIVE a workflow deletion — not dependents
            for cid, _where in self._workflow_checkpoint_refs(rec):
                push("checkpoint", cid, model_id, "workflow")
            if rec.recipe_id:
                push("workflow_recipe", rec.recipe_id, None, "workflow")
            for ds, tok in self._workflow_data_refs(rec.plan):
                push("dataset", ds, None, "workflow")
                push("tokenizer", tok, None, "workflow")
        elif family == "evaluation":
            rec = self.evaluation.get_evaluation(model_id, artifact_id)
            if rec.checkpoint_id:
                push("checkpoint", rec.checkpoint_id, model_id,
                     "evaluation")
            push("dataset", rec.dataset_id, None, "evaluation")
            push("tokenizer", rec.tokenizer_id, None, "evaluation")
        elif family == "comparison":
            rec = self.comparison.get_comparison(model_id, artifact_id)
            for side in (rec.state_a, rec.state_b):
                push("evaluation", side.evaluation_id, model_id,
                     "comparison")
                if side.state_kind == EvalStateKind.CHECKPOINT \
                        and side.checkpoint_id:
                    push("checkpoint", side.checkpoint_id, model_id,
                         "comparison")
            push("dataset", rec.dataset_id, None, "comparison")
            push("tokenizer", rec.tokenizer_id, None, "comparison")
        elif family == "gate":
            rec = self.gates.get_decision(model_id, artifact_id)
            if rec.comparison_id:
                push("comparison", rec.comparison_id, model_id, "gate")
            for side in (rec.candidate, rec.baseline):
                if side is None:
                    continue
                push("evaluation", side.evaluation_id, model_id, "gate")
                if side.state_kind == EvalStateKind.CHECKPOINT \
                        and side.checkpoint_id:
                    push("checkpoint", side.checkpoint_id, model_id,
                         "gate")
            if rec.suggested_checkpoint_id:
                push("checkpoint", rec.suggested_checkpoint_id, model_id,
                     "gate")
            if rec.policy_id:
                push("gate_policy", rec.policy_id, None, "gate")
            push("dataset", rec.policy.dataset_id, None, "gate")
            push("tokenizer", rec.policy.tokenizer_id, None, "gate")
        elif family == "suite_run":
            rec = self.get_suite_run(model_id, artifact_id)
            push("model", model_id, None, "suite_run")
            push("probe_suite", rec.suite_id, None, "suite_run")
            if rec.state.state_kind == EvalStateKind.CHECKPOINT \
                    and rec.state.checkpoint_id:
                push("checkpoint", rec.state.checkpoint_id, model_id,
                     "suite_run")
            for r in rec.results:
                if r.evaluation_id:
                    push("evaluation", r.evaluation_id, model_id,
                         "suite_run")
                push("dataset", r.probe.dataset_id, None, "suite_run")
                push("tokenizer", r.probe.tokenizer_id, None,
                     "suite_run")
        elif family == "sample":
            rec = self.get_sample(model_id, artifact_id)
            push("model", model_id, None, "sample")
            push("checkpoint", rec.checkpoint_id, model_id, "sample")
            push("tokenizer", rec.tokenizer_id, None, "sample")
        elif family == "sample_quality":
            rec = self.get_sample_evaluation(model_id, artifact_id)
            push("model", model_id, None, "sample_quality")
            push("sample", rec.sample_id, model_id, "sample_quality")
            push("checkpoint", rec.checkpoint_id, model_id,
                 "sample_quality")
            push("tokenizer", rec.tokenizer_id, None, "sample_quality")
        # tokenizer / probe_suite / checkpoint: nothing has them in a
        # blocker list (a tokenizer is referenced, never referencing;
        # suites bind at run time; checkpoint lineage never blocks)
        order = {f: i for i, f in enumerate(
            self.PROJECT_RETENTION_FAMILIES)}
        return sorted(out, key=lambda t: (order[t[0]], t[1], t[3]))

    def _impact_evaluate_dependent(
            self, family: str, artifact_id: str, model_id: str | None,
            dep_family: str, dep_id: str, dep_model: str | None,
            ref_category: str):
        """Verify ONE outbound candidate (is the selected artifact
        REALLY in the dependent's blocker list?) and compute its
        FIRST-LEVEL after-state: (is_dependent, becomes_entry | None).
        The after-state re-evaluates the dependent's blockers over a
        minimal in-memory shadow — the SAME canonical filters and
        views the guards use, minus the selected artifact's edges —
        never touching the filesystem (§7). First level only: a
        dependent that stays blocked after this ONE deletion is
        reported as a dependent but never as becoming deletable."""
        if dep_family in ("workflow", "evaluation", "comparison",
                          "gate"):
            v = self.model_record_retention_overview(
                dep_model, dep_family, dep_id)
            hits = [b for b in v.blockers
                    if b.category == family and b.reference_id ==
                    artifact_id]
            if not hits:
                return False, None
            rest = [b for b in v.blockers
                    if not (b.category == family
                            and b.reference_id == artifact_id)]
            entry = None
            if not rest and v.integrity_verified:
                entry = ImpactBecomesDeletable(
                    family=dep_family, artifact_id=dep_id,
                    model_id=dep_model, files=len(v.files),
                    size_bytes=v.size_bytes)
            return True, entry
        if dep_family == "checkpoint":
            ov = self.checkpoint_retention_overview(dep_model)
            entry = next((e for e in ov.checkpoints
                          if e.checkpoint_id == dep_id), None)
            if entry is None:
                return False, None
            # §7 shadow over the SAME canonical filters M61 uses,
            # minus the selected artifact's edges
            pointers = [b for b in entry.blockers if b.reason in
                        ("best", "published", "manifest_reference")]
            evals = {e.eval_id for e in
                     self.evaluation.list_evaluations_for_checkpoint(
                         dep_model, dep_id)}
            comps = {c.comparison_id for c in
                     self.comparison.list_comparisons_for_checkpoint(
                         dep_model, dep_id)}
            gates = set()
            for g in self.gates.list_decisions(dep_model):
                for side_name, side in (("candidate", g.candidate),
                                        ("baseline", g.baseline)):
                    if (side is not None
                            and side.state_kind ==
                            EvalStateKind.CHECKPOINT
                            and side.checkpoint_id == dep_id):
                        gates.add(f"{g.decision_id}.{side_name}")
                if g.suggested_checkpoint_id == dep_id:
                    gates.add(f"{g.decision_id}.suggested_checkpoint_id")
            suites = {s.suite_run_id for s in
                      self.suite_runs.list_suite_runs_for_checkpoint(
                          dep_model, dep_id)}
            samples = {s.sample_id for s in
                       self.samples.list_samples_for_checkpoint(
                           dep_model, dep_id)}
            sqs = {q.evaluation_id for q in
                   self.sample_quality.
                   list_sample_evaluations_for_checkpoint(
                       dep_model, dep_id)}
            wfs = {w.workflow_id for w in
                   self.workflows.list_workflows(dep_model)
                   if any(cid == dep_id for cid, _ in
                          self._workflow_checkpoint_refs(w))}
            member = False
            if family == "workflow":
                member = artifact_id in wfs
                wfs.discard(artifact_id)
            elif family == "evaluation":
                member = artifact_id in evals
                evals.discard(artifact_id)
            elif family == "comparison":
                member = artifact_id in comps
                comps.discard(artifact_id)
            elif family == "gate":
                member = any(g.startswith(f"{artifact_id}.")
                             for g in gates)
                gates = {g for g in gates
                         if not g.startswith(f"{artifact_id}.")}
            elif family == "suite_run":
                member = artifact_id in suites
                suites.discard(artifact_id)
            elif family == "sample":
                member = artifact_id in samples
                samples.discard(artifact_id)
            else:  # sample_quality
                member = artifact_id in sqs
                sqs.discard(artifact_id)
            if not member:
                return False, None
            entry_out = None
            if (not pointers and not evals and not comps and not gates
                    and not suites and not samples and not sqs
                    and not wfs and entry.integrity_verified):
                entry_out = ImpactBecomesDeletable(
                    family="checkpoint", artifact_id=dep_id,
                    model_id=dep_model, files=entry.files,
                    size_bytes=entry.size_bytes)
            return True, entry_out
        if dep_family == "model":
            v = self.model_retention_overview(dep_id)
            # the dependent's blocker category is the outbound
            # reference_category (the M70 name — "policy" for gate
            # policies, the family name for every other family)
            hits = [b for b in v.blockers
                    if b.category == ref_category
                    and b.reference_id == artifact_id]
            if not hits:
                return False, None
            rest = [b for b in v.blockers
                    if not (b.category == ref_category
                            and b.reference_id == artifact_id)]
            entry = None
            if not rest and v.integrity_verified:
                entry = ImpactBecomesDeletable(
                    family="model", artifact_id=dep_id, model_id=None,
                    files=len(v.files), size_bytes=v.size_bytes)
            return True, entry
        if dep_family in ("dataset", "tokenizer"):
            usage = (self.dataset_usage_overview(dep_id)
                     if dep_family == "dataset"
                     else self.tokenizer_usage_overview(dep_id))
            if family == "dataset":
                # X = dataset: the tokenizer's tokenized_dataset refs
                contributed = [r for c in usage.categories
                               if c.category == "tokenized_dataset"
                               for r in c.references
                               if r.startswith(f"{artifact_id}/")]
            elif family == "model":
                contributed = [r for c in usage.categories
                               for r in c.references
                               if r.startswith(f"{artifact_id}/")]
            else:  # a record of model_id: its ONE qualified id in X's
                   # own category
                qid = f"{model_id}/{artifact_id}"
                contributed = [r for c in usage.categories
                               if c.category == family
                               for r in c.references if r == qid]
            if not contributed:
                return False, None
            rest = any(
                len([r for r in c.references
                     if r not in set(contributed)]) > 0
                for c in usage.categories)
            entry = None
            integrity = (self.dataset_retention_overview(dep_id)
                         .integrity_verified
                         if dep_family == "dataset" else
                         self.tokenizer_retention_overview(dep_id)
                         .integrity_verified)
            if not rest and integrity:
                v = (self.dataset_retention_overview(dep_id)
                     if dep_family == "dataset"
                     else self.tokenizer_retention_overview(dep_id))
                entry = ImpactBecomesDeletable(
                    family=dep_family, artifact_id=dep_id, model_id=None,
                    files=len(v.files), size_bytes=v.size_bytes)
            return True, entry
        if dep_family in ("workflow_recipe", "gate_policy",
                          "probe_suite"):
            v = self.definition_retention_overview(dep_family, dep_id)
            hits = [b for b in v.blockers if b.category == family
                    and b.reference_id == artifact_id]
            if not hits:
                return False, None
            rest = [b for b in v.blockers
                    if not (b.category == family
                            and b.reference_id == artifact_id)]
            entry = None
            if not rest and v.integrity_verified:
                entry = ImpactBecomesDeletable(
                    family=dep_family, artifact_id=dep_id,
                    model_id=None, files=len(v.files),
                    size_bytes=v.size_bytes)
            return True, entry
        # dep_family == "sample": X = sample_quality (the ONE M19
        # listing is the blocker source — M68)
        v = self.sample_retention_overview(dep_model, dep_id)
        listing = [m.evaluation_id for m in
                   self.sample_quality.list_sample_evaluations_for_sample(
                       dep_model, dep_id)]
        if artifact_id not in listing:
            return False, None
        rest = [m for m in listing if m != artifact_id]
        entry = None
        if not rest and v.integrity_verified:
            entry = ImpactBecomesDeletable(
                family="sample", artifact_id=dep_id, model_id=dep_model,
                files=len(v.files), size_bytes=v.size_bytes)
        return True, entry

    def deletion_impact_preview(self, family: str, artifact_id: str,
                                model_id: str | None = None
                                ) -> DeletionImpactPreview:
        """Read-only live-computed FIRST-LEVEL deletion impact preview
        of ONE artifact (M73): what the artifact's VERIFIED deletion
        would unblock if it were performed NOW. Sections: (A) the
        CURRENT retention state — the family's EXISTING view verbatim
        (scope: unknown or registry-invisible artifact ->
        FileNotFoundError, 404 at the API; the blocker projection is
        1:1, never a second blocker engine); (B) the IMMEDIATE
        DEPENDENTS — artifacts whose current blocker lists contain
        the selected artifact (its outbound persisted edges, verified
        against each dependent's canonical blockers); (C) the
        BECOMES-DELETABLE set — FIRST LEVEL ONLY: the dependents
        whose blocker lists become empty in the minimal in-memory
        shadow (the SAME canonical filters minus the selected
        artifact's edges; no recursive cascade — dependents of
        dependents are deliberately absent); (D) the RECLAIMABLE
        impact — the artifact's own files/bytes plus the project
        reclaimable before/after/delta, computed by re-running the
        ONE M72 aggregation core over the shadow state (the selected
        artifact skipped, its model's storage shrunk when it is
        model-owned, the newly-deletable artifacts flipped) so the
        M72 overlap rule holds verbatim (a deletable model
        contributes its WHOLE directory, subsuming its records). A
        blocked or integrity-failed artifact has NO executable
        deletion (the guard would refuse it): ``executable`` is
        false, the becomes-deletable set is empty and the project
        reclaimable does not move — the honest guard semantics,
        never bypassed. Zero storage, zero mutation, deterministic;
        the lifecycle-less ``training_run`` family has no preview
        (ValueError)."""
        if family not in self.IMPACT_FAMILIES:
            raise ValueError(
                f"family '{family}' has no deletion lifecycle (M73 "
                f"families: {', '.join(self.IMPACT_FAMILIES)})")
        cur = self._impact_current(family, artifact_id, model_id)
        before_rb = self._project_retention_core()[2]
        dependents: list[ImpactDependent] = []
        becomes: list[ImpactBecomesDeletable] = []
        # the IMMEDIATE DEPENDENTS are current-state FACTS (who the
        # selected artifact blocks) — reported for blocked artifacts
        # too; only the becomes-deletable set and the reclaim
        # movement are the EXECUTABLE impact, gated on the guard's
        # own decision (§8: never pretend a refused deletion occurs)
        for (dep_family, dep_id, dep_model,
             ref_category) in self._impact_outbound(
                family, artifact_id, model_id):
            is_dep, entry = self._impact_evaluate_dependent(
                family, artifact_id, model_id, dep_family, dep_id,
                dep_model, ref_category)
            if not is_dep:
                continue
            dependents.append(ImpactDependent(
                family=dep_family, artifact_id=dep_id,
                model_id=dep_model, reference_category=ref_category))
            if cur["deletable"] and entry is not None:
                becomes.append(entry)
        # the shadow reclaim: ONE M72 core re-run — the selected
        # artifact removed + the newly-deletable flipped — gated on
        # the guard's own decision (a refused deletion moves nothing)
        if cur["deletable"]:
            skip = {(family, artifact_id)}
            override = {(b.family, b.artifact_id) for b in becomes}
            delta = {}
            if family in ("checkpoint", "workflow", "evaluation",
                          "comparison", "gate"):
                delta[model_id] = (cur["files"], cur["size"])
            after_rb = self._project_retention_core(
                skip=skip, deletable_override=override,
                model_storage_delta=delta)[2]
        else:
            after_rb = before_rb
        order = {f: i for i, f in enumerate(
            self.PROJECT_RETENTION_FAMILIES)}
        dependents.sort(key=lambda d: (order[d.family],
                                       d.artifact_id,
                                       d.reference_category))
        becomes.sort(key=lambda b: (order[b.family], b.artifact_id))
        return DeletionImpactPreview(
            family=family,
            artifact_id=artifact_id,
            model_id=model_id,
            deletion_supported=True,
            deletable=cur["deletable"],
            integrity_verified=cur["integrity"],
            blockers=[ImpactBlocker(category=c, reference_id=r,
                                    detail=d)
                      for c, r, d in cur["blockers"]],
            files=cur["files"],
            size_bytes=cur["size"],
            executable=cur["deletable"],
            immediate_dependents=dependents,
            becomes_deletable=becomes,
            immediate_files=cur["files"],
            immediate_bytes=cur["size"],
            project_reclaimable_before=before_rb,
            project_reclaimable_after=after_rb,
            project_reclaimable_delta=after_rb - before_rb)

    def _recipe_model_ids(self, recipe) -> set[str]:
        """Model ids DIRECTLY named by ONE workflow recipe's stage
        configs (train stage configs, evaluate stage configs and gate
        policies all persist ``model_id``; a gate stage may name the
        policy INLINE or by REGISTRY id — the registry form resolves
        through the ONE policy registry to the policy's target model,
        the M71-completed analysis; suite-run/publish/recipe stages
        never name a model). A recipe is INERT DATA but its
        definition is bound to the models it names — deleting such a
        model would leave the recipe unresolvable — so it is a REAL
        persisted external reference. Read-only scan, the M64
        ``_workflow_data_refs`` pattern."""
        ids: set[str] = set()
        for stage in recipe.stages:
            if stage.training is not None:
                ids.add(stage.training.model_id)
            if stage.evaluation is not None:
                ids.add(stage.evaluation.config.model_id)
            if stage.gate is not None:
                if stage.gate.policy is not None:
                    ids.add(stage.gate.policy.model_id)
                elif stage.gate.policy_id is not None:
                    # REGISTRY policy: the recipe binds the policy's
                    # target model (the M71 guard keeps this id
                    # resolvable — a read-only scan stays robust if a
                    # legacy dangling id ever appears)
                    try:
                        ids.add(self.policies.get_policy(
                            stage.gate.policy_id).policy.model_id)
                    except FileNotFoundError:
                        pass
        return ids

    def model_usage_overview(self, model_id: str) -> ModelUsageOverview:
        """Read-only live-computed usage overview of ONE model (M66):
        every persisted record that references it, by category — the
        internal model-scoped families (the model's own training
        provenance, checkpoints, workflows, evaluations, comparisons,
        gate decisions — all through the ONE authoritative listings)
        plus the EXTERNAL root-level families persisting the model id
        outside the model directory (suite runs, samples, sample-quality
        measurements, model-bound workflow recipes and model-bound
        registered gate policies). Per-category references are the
        listings' record ids,
        sorted and unique; internal/external splits expose exactly what
        a future model-retention guard would need. Zero storage, zero
        mutation, byte-identical over unchanged state; unknown or
        registry-invisible (unparseable manifest) model ->
        FileNotFoundError (404 at the API)."""
        record = self._scope_data_artifact(self.storage.load_record,
                                           model_id, "model")
        categories = [
            ("training_run",
             sorted(p.run_id for p in record.training_provenance)),
            ("checkpoint",
             sorted(c.checkpoint_id
                    for c in self.training.list_checkpoints(model_id))),
            ("workflow",
             sorted(w.workflow_id
                    for w in self.list_workflows(model_id))),
            ("evaluation",
             sorted(e.eval_id
                    for e in self.list_evaluations(model_id))),
            ("comparison",
             sorted(c.comparison_id
                    for c in self.list_comparisons(model_id))),
            ("gate",
             sorted(g.decision_id
                    for g in self.list_gate_decisions(model_id))),
            ("suite_run",
             sorted(r.suite_run_id
                    for r in self.list_suite_runs(model_id))),
            ("sample",
             sorted(s.sample_id
                    for s in self.list_samples(model_id))),
            ("sample_quality",
             sorted(sq.evaluation_id
                    for sq in self.list_sample_evaluations(model_id))),
            ("workflow_recipe",
             sorted(r.recipe_id for r in self.recipes.list()
                    if model_id in self._recipe_model_ids(r))),
            ("policy",
             sorted(d.policy_id for d in self.policies.list_policies()
                    if d.policy.model_id == model_id)),
        ]
        internal = sum(len(refs) for name, refs in categories
                       if name in self.MODEL_USAGE_INTERNAL_CATEGORIES)
        external = sum(len(refs) for name, refs in categories
                       if name not in self.MODEL_USAGE_INTERNAL_CATEGORIES)
        return ModelUsageOverview(
            model_id=model_id,
            name=record.name,
            created_at=record.created_at,
            architecture=record.architecture.value,
            parameter_count=record.parameter_count,
            referenced=internal + external > 0,
            externally_referenced=external > 0,
            total_references=internal + external,
            internal_references=internal,
            external_references=external,
            categories=[ModelUsageCategory(category=c, references=r)
                        for c, r in categories])

    def model_deletion_blockers(
            self, model_id: str) -> list[ModelDeletionBlocker]:
        """The M67 deletion guard's ordered blocker list: EXACTLY the
        EXTERNAL (root-level) references of the ONE M66 usage
        analysis — the same categories, the same persisted reference
        ids, the same canonical order (category order, then sorted
        reference ids). The INTERNAL model-scoped categories
        (training_run / checkpoint / workflow / evaluation /
        comparison / gate) are OWNERSHIP: they live inside
        ``models/<id>/`` and are removed atomically WITH the model, so
        deleting the model leaves nothing unresolved and they never
        block. The EXTERNAL categories (suite_run / sample /
        sample_quality / workflow_recipe / policy) persist the model
        id OUTSIDE the model directory — deleting the model would
        leave those records unresolved — so every such reference is
        ONE typed blocker with short authoritative identifying detail
        from the SAME ONE listings the overview itself uses. Nothing
        protected that is not shown, nothing shown that is not
        protected. Zero storage, zero mutation."""
        overview = self.model_usage_overview(model_id)
        details: dict[tuple[str, str], str] = {}
        for r in self.list_suite_runs(model_id):
            details[("suite_run", r.suite_run_id)] = f"suite '{r.suite_id}'"
        for s in self.list_samples(model_id):
            details[("sample", s.sample_id)] = \
                f"checkpoint '{s.checkpoint_id}'"
        for sq in self.list_sample_evaluations(model_id):
            details[("sample_quality", sq.evaluation_id)] = \
                f"sample '{sq.sample_id}'"
        for r in self.recipes.list():
            if model_id in self._recipe_model_ids(r):
                details[("workflow_recipe", r.recipe_id)] = \
                    f"config_hash '{r.config_hash}'"
        for d in self.policies.list_policies():
            if d.policy.model_id == model_id:
                details[("policy", d.policy_id)] = \
                    f"gate policy '{d.policy.name}'"
        blockers: list[ModelDeletionBlocker] = []
        for cat in overview.categories:
            if cat.category in self.MODEL_USAGE_INTERNAL_CATEGORIES:
                continue  # ownership — removed WITH the model
            for ref in cat.references:
                blockers.append(ModelDeletionBlocker(
                    category=cat.category, reference_id=ref,
                    detail=details.get(
                        (cat.category, ref),
                        f"references model '{model_id}'")))
        return blockers

    def model_retention_overview(
            self, model_id: str) -> ModelRetentionOverview:
        """Read-only live-computed retention overview of ONE model
        (M67): the deletion-readiness view — identity, the ordered
        artifact files + total bytes of the model's OWN directory (its
        whole internal history), the M2 integrity-verification
        outcome, ``deletable`` (True iff integrity passes AND the ONE
        M66 reference analysis finds no EXTERNAL reference) and the
        ordered blockers (the SAME list the DELETE guard refuses on).
        A corrupt model is never deletable; an unknown or
        registry-invisible (unparseable manifest) model ->
        FileNotFoundError (404 at the API). Zero storage, zero
        mutation, byte-identical over unchanged state."""
        record = self._scope_data_artifact(self.storage.load_record,
                                           model_id, "model")
        files, nbytes = self._artifact_files(
            self.storage.model_dir(model_id))
        try:
            integrity_verified = (
                self.verify_model(model_id).get("integrity") == "ok")
        except Exception:
            integrity_verified = False
        blockers = self.model_deletion_blockers(model_id)
        return ModelRetentionOverview(
            model_id=model_id,
            name=record.name,
            created_at=record.created_at,
            architecture=record.architecture.value,
            parameter_count=record.parameter_count,
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

    def dataset_deletion_blockers(
            self, dataset_id: str) -> list[ArtifactDeletionBlocker]:
        """The reference-safety analysis for dataset deletion (M65),
        computed LIVE from the ONE M64 usage overview — never a second
        scanner. Every M64-visible reference category with at least one
        reference becomes one blocker, in the SAME canonical order and
        with the SAME reference ids the ``GET /datasets/{id}/usage``
        overview reports (the guard refuses on EXACTLY what the
        overview shows: nothing protected that is not shown, nothing
        shown that is not protected). Unknown dataset ->
        FileNotFoundError (404 at the API); zero storage, zero
        mutation."""
        overview = self.dataset_usage_overview(dataset_id)
        return [ArtifactDeletionBlocker(reason=c.category,
                                        detail=", ".join(c.references))
                for c in overview.categories if c.references]

    def tokenizer_deletion_blockers(
            self, tokenizer_id: str) -> list[ArtifactDeletionBlocker]:
        """The reference-safety analysis for tokenizer deletion (M65),
        computed LIVE from the ONE M64 usage overview — never a second
        scanner. Every M64-visible reference category with at least one
        reference becomes one blocker, in the SAME canonical order and
        with the SAME reference ids the
        ``GET /tokenizers/{id}/usage`` overview reports. Unknown
        tokenizer -> FileNotFoundError (404 at the API); zero storage,
        zero mutation."""
        overview = self.tokenizer_usage_overview(tokenizer_id)
        return [ArtifactDeletionBlocker(reason=c.category,
                                        detail=", ".join(c.references))
                for c in overview.categories if c.references]

    def delete_dataset(self, dataset_id: str) -> DatasetDeletionResult:
        """Explicit VERIFIED dataset retention (M65): remove ONE
        dataset — only after proving, live, that nothing references it.
        The full guard, in order: (1) scope through the M2 registry
        (``load_meta`` — unknown dataset -> FileNotFoundError, nothing
        deleted); (2) the LIVE reference-safety analysis
        (``dataset_deletion_blockers`` — the ONE M64 analysis: training
        runs, workflows, evaluations, comparisons, suite runs, the
        tokenizers trained on it and the tokenized versions derived
        from it; ANY reference -> ValueError listing the ordered
        blockers, so deletion can never orphan referencing evidence);
        (3) ATOMIC removal of the dataset's OWN directory only
        (``DatasetEngine.delete`` — one rename to a hidden sibling,
        then rmtree; no partial dataset can ever be observed). No
        cascade, no force, no bulk mode, no policies — exactly the one
        explicitly requested dataset. Never touches tokenizers,
        models, checkpoints or any other family's records (they are
        protected BY the guard).
        INTEGRITY FIRST (the M61 ordering): between scope and the
        reference analysis the existing M2 verifier
        (``DatasetEngine.verify`` — records, splits, tokenized
        artifacts) must pass; a corrupt dataset is REFUSED with
        RuntimeError (409) — deletion never bypasses integrity
        validation, and a manifest that cannot be parsed at all is
        registry-invisible (scope -> 404, nothing deleted)."""
        self._scope_data_artifact(self.datasets.load_meta, dataset_id,
                                  "dataset")
        try:
            report = self.datasets.verify(dataset_id)
        except Exception as exc:
            raise RuntimeError(
                f"dataset '{dataset_id}' could not be verified: "
                f"{exc}") from exc
        if report.get("status") != "ok":
            raise RuntimeError(
                f"dataset '{dataset_id}' failed integrity verification"
                f" — refusing to delete corrupted storage")
        blockers = self.dataset_deletion_blockers(dataset_id)
        if blockers:
            summary = "; ".join(f"{b.reason}: {b.detail}" for b in blockers)
            raise ValueError(
                f"dataset '{dataset_id}' is referenced and cannot be "
                f"deleted — {summary}")
        files, nbytes = self.datasets.delete(dataset_id)
        log.info("M65 verified dataset deletion %s (%d files, %d bytes)",
                 dataset_id, files, nbytes)
        return DatasetDeletionResult(dataset_id=dataset_id,
                                     files_removed=files,
                                     bytes_reclaimed=nbytes)

    def delete_tokenizer(self, tokenizer_id: str) -> TokenizerDeletionResult:
        """Explicit VERIFIED tokenizer retention (M65): remove ONE
        tokenizer — only after proving, live, that nothing references
        it. The full guard, in order: (1) scope through the registry
        (``TokenizerEngine.load`` — unknown tokenizer ->
        FileNotFoundError, nothing deleted); (2) the LIVE
        reference-safety analysis (``tokenizer_deletion_blockers`` —
        the ONE M64 analysis: training runs, workflows, evaluations,
        comparisons, suite runs, samples, sample-quality measurements
        and the dataset versions it tokenized; ANY reference ->
        ValueError listing the ordered blockers — closing the pre-M61
        gap this family had: an unguarded rmtree); (3) ATOMIC removal
        of the tokenizer's OWN directory only. No cascade, no force,
        no bulk mode, no policies. Never touches datasets (a
        referenced tokenizer's tokenized artifacts under a dataset
        appear in this analysis as ``tokenized_dataset`` and therefore
        block deletion), models, checkpoints or any other family's
        records.
        INTEGRITY FIRST (the M61 ordering): between scope and the
        reference analysis the tokenizer verifier
        (``TokenizerEngine.verify`` — manifest resolution + the
        tokenizer.json content hash vs the persisted
        ``tokenizer_hash``) must pass; a corrupt tokenizer is REFUSED
        with RuntimeError (409) — deletion never bypasses integrity
        validation, and a manifest that cannot be parsed at all is
        registry-invisible (scope -> 404, nothing deleted)."""
        self._scope_data_artifact(self.tokenizers.load, tokenizer_id,
                                  "tokenizer")
        try:
            report = self.tokenizers.verify(tokenizer_id)
        except Exception as exc:
            raise RuntimeError(
                f"tokenizer '{tokenizer_id}' could not be verified: "
                f"{exc}") from exc
        if report.get("status") != "ok":
            raise RuntimeError(
                f"tokenizer '{tokenizer_id}' failed integrity "
                f"verification — refusing to delete corrupted storage")
        blockers = self.tokenizer_deletion_blockers(tokenizer_id)
        if blockers:
            summary = "; ".join(f"{b.reason}: {b.detail}" for b in blockers)
            raise ValueError(
                f"tokenizer '{tokenizer_id}' is referenced and cannot "
                f"be deleted — {summary}")
        files, nbytes = self.tokenizers.delete(tokenizer_id)
        log.info("M65 verified tokenizer deletion %s (%d files, %d bytes)",
                 tokenizer_id, files, nbytes)
        return TokenizerDeletionResult(tokenizer_id=tokenizer_id,
                                       files_removed=files,
                                       bytes_reclaimed=nbytes)

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

    def list_comparisons_for_seed(self, model_id: str,
                                  seed: int) -> list[ComparisonRecord]:
        """Immutable M5 comparison records of ONE model by persisted
        effective seed (M49 read-only access).

        ``seed`` is the REQUIRED integer persisted on every record
        (the seed of the identical-probe A/B measurement) — an OPEN
        value axis with NO registry, so a non-integer spelling is
        rejected at the API boundary with 422, while an unknown model
        raises FileNotFoundError (404) exactly like the sibling
        groupings. Returns the model's authoritative M5 listing
        filtered by the persisted record value (matched VERBATIM —
        never recalculated, never normalized, never derived from the
        configuration or either side's evaluation) in the M5
        (created_at, comparison_id) order, as complete verbatim
        records. An unmatched seed on a valid model returns [].
        Read-only, never writes."""
        return self.comparison.list_comparisons_for_seed(model_id, seed)

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

    def delete_suite_run(self, model_id: str,
                         suite_run_id: str) -> SuiteRunDeletionResult:
        """Explicit VERIFIED suite-run retention (M68): remove ONE
        suite run — after scope and integrity verification. The guard,
        in order: (1) scope through the family getter (unknown model,
        unknown run, or registry-invisible (unparseable manifest) run
        -> FileNotFoundError, nothing deleted); (2) INTEGRITY FIRST
        (the M61/M65/M67 ordering): the record's persisted
        ``result_hash`` must reproduce from its semantic payload — a
        tampered/corrupt record is REFUSED with RuntimeError (409);
        (3) NO reference guard — a suite run is a LEAF record: nothing
        persists a suite_run_id. Its probe EVALUATIONS are MODEL-OWNED
        (inside models/<id>/evaluations/) and are never touched (no
        cascade — they simply keep existing); (4) ATOMIC removal of
        the run's own directory (``SuiteRunEngine.delete`` — the
        ``DatasetEngine.delete`` pattern). No cascade, no force, no
        bulk mode."""
        record = self._scope_data_artifact(
            lambda sid: self.suite_runs.get_suite_run(model_id, sid),
            suite_run_id, "suite run")
        if self.suite_runs.result_hash(record) != record.result_hash:
            raise RuntimeError(
                f"suite run '{suite_run_id}' failed integrity "
                f"verification — refusing to delete corrupted storage")
        files, nbytes = self.suite_runs.delete(suite_run_id)
        log.info("M68 verified suite-run deletion %s (%d files, %d bytes)",
                 suite_run_id, files, nbytes)
        return SuiteRunDeletionResult(model_id=record.model_id,
                                      suite_run_id=suite_run_id,
                                      files_removed=files,
                                      bytes_reclaimed=nbytes)

    def suite_run_retention_overview(
            self, model_id: str,
            suite_run_id: str) -> SuiteRunRetentionOverview:
        """Read-only live-computed retention overview of ONE suite run
        (M68): the deletion-readiness view — identity, the ordered
        artifact files + total bytes of the run's OWN directory, the
        record's result-hash integrity outcome, ``deletable`` and the
        ordered blockers (the SAME list the DELETE guard refuses on —
        always empty: a suite run is a LEAF record, nothing persists
        a suite_run_id). A tampered/corrupt record is never deletable;
        an unknown or registry-invisible (unparseable manifest) run ->
        FileNotFoundError (404 at the API). Zero storage, zero
        mutation, byte-identical over unchanged state."""
        record = self._scope_data_artifact(
            lambda sid: self.suite_runs.get_suite_run(model_id, sid),
            suite_run_id, "suite run")
        files, nbytes = self._artifact_files(
            self.suite_runs._run_dir(suite_run_id))
        integrity_verified = (
            self.suite_runs.result_hash(record) == record.result_hash)
        blockers: list[ArtifactDeletionBlocker] = []
        return SuiteRunRetentionOverview(
            model_id=record.model_id,
            suite_run_id=suite_run_id,
            suite_id=record.suite_id,
            created_at=record.created_at,
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

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

    def list_suite_runs_for_reused_count(self, model_id: str,
                                         reused_count: int
                                         ) -> list[SuiteRunRecord]:
        """Immutable M10 suite-run records of ONE model by persisted
        reuse count (M50 read-only access).

        ``reused_count`` is the REQUIRED integer persisted on every
        record (probes satisfied by pre-existing evidence — execution
        bookkeeping only, NEVER a score) — an OPEN value axis with NO
        registry, so a non-integer spelling is rejected at the API
        boundary with 422, while an unknown model raises
        FileNotFoundError (404) exactly like the sibling groupings.
        Returns the model's authoritative M10 listing filtered by the
        persisted record value (matched VERBATIM — never recalculated,
        never derived from probe outcomes or any other field) in the
        M10 (created_at, suite_run_id) order, as complete verbatim
        records. An unmatched count on a valid model returns [].
        Read-only, never writes."""
        return self.suite_runs.list_suite_runs_for_reused_count(
            model_id, reused_count)

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

    def run_workflow_recipe_repeated(self, recipe_id: str, model_id: str,
                                     repetitions: int) -> list[WorkflowRecord]:
        """M58: execute ONE registered recipe N times sequentially — a thin
        loop over the existing single-run path (one executor, one resolver
        per iteration, one normal immutable record per iteration). Each
        iteration resolves 'best' independently at its own plan start; a
        'stopped' iteration (gate stop) ends the sequence; a 'failed'
        iteration keeps the existing stage-exception semantics. No retry,
        no rollback, no background execution."""
        return self.recipes.run_repeated(recipe_id, model_id, repetitions)

    def resolve_workflow_recipe(self, recipe_id: str,
                                model_id: str) -> WorkflowRecipeResolution:
        """Read-only resolution of ONE registered recipe against ONE
        explicit model (M51 preflight): the SAME resolution path as
        run_workflow_recipe (recipe lookup -> model validation -> M14
        expansion -> WorkflowPlan construction + full M7 validation),
        WITHOUT executing or persisting anything. Returns the expanded,
        model-bound plan plus recipe provenance and the composition
        trace; error semantics identical to a run request (unknown
        recipe/model -> FileNotFoundError -> 404, binding mismatch ->
        ValueError -> 422)."""
        return self.recipes.resolve(recipe_id, model_id)

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

    def sample_deletion_blockers(
            self, model_id: str, sample_id: str
    ) -> list[ArtifactDeletionBlocker]:
        """The M68 sample deletion guard's ordered blocker list:
        EXACTLY the sample-quality measurements referencing the sample
        — the ONE M19 ``list_sample_evaluations_for_sample`` listing,
        the same persisted evaluation ids, sorted. A measurement
        persists ``sample_id`` OUTSIDE the sample's own directory
        (sample-evaluations/<model_id>/), so deleting the sample would
        leave it referencing a missing record — it blocks. Nothing
        else references a sample. Zero storage, zero mutation."""
        measurements = self.sample_quality \
            .list_sample_evaluations_for_sample(model_id, sample_id)
        if not measurements:
            return []
        ids = ", ".join(sorted(m.evaluation_id for m in measurements))
        return [ArtifactDeletionBlocker(
            reason="sample_quality",
            detail=f"sample-quality measurement(s) '{ids}'")]

    def delete_sample(self, model_id: str,
                      sample_id: str) -> SampleDeletionResult:
        """Explicit VERIFIED sample retention (M68): remove ONE
        sample — only after proving, live, that nothing references it.
        The guard, in order: (1) scope through the family getter
        (unknown model, unknown sample, or registry-invisible
        (unparseable manifest) sample -> FileNotFoundError, nothing
        deleted); (2) INTEGRITY FIRST: the record's persisted
        ``result_hash`` must reproduce — a tampered/corrupt sample is
        REFUSED with RuntimeError (409); (3) the LIVE reference guard
        (``sample_deletion_blockers`` — the ONE M19 listing: ANY
        sample-quality measurement referencing the sample ->
        ValueError listing the ordered blockers, so deletion can never
        orphan a persisted measurement); (4) ATOMIC removal of the
        sample's own directory (``SamplingEngine.delete``). No
        cascade, no force, no bulk mode."""
        record = self._scope_data_artifact(
            lambda sid: self.samples.get_sample(model_id, sid),
            sample_id, "sample")
        if self.samples.result_hash(record) != record.result_hash:
            raise RuntimeError(
                f"sample '{sample_id}' failed integrity verification "
                f"— refusing to delete corrupted storage")
        blockers = self.sample_deletion_blockers(model_id, sample_id)
        if blockers:
            summary = "; ".join(f"{b.reason}: {b.detail}" for b in blockers)
            raise ValueError(
                f"sample '{sample_id}' of model '{model_id}' is "
                f"referenced and cannot be deleted — {summary}")
        files, nbytes = self.samples.delete(model_id, sample_id)
        log.info("M68 verified sample deletion %s/%s (%d files, %d bytes)",
                 model_id, sample_id, files, nbytes)
        return SampleDeletionResult(model_id=model_id, sample_id=sample_id,
                                    files_removed=files,
                                    bytes_reclaimed=nbytes)

    def sample_retention_overview(
            self, model_id: str,
            sample_id: str) -> SampleRetentionOverview:
        """Read-only live-computed retention overview of ONE sample
        (M68): the deletion-readiness view — identity, the ordered
        artifact files + total bytes of the sample's OWN directory,
        the record's result-hash integrity outcome, ``deletable``
        (True iff integrity passes AND the ONE M19 listing finds no
        sample-quality measurement referencing it) and the ordered
        blockers (the SAME list the DELETE guard refuses on). A
        tampered/corrupt sample is never deletable; an unknown or
        registry-invisible (unparseable manifest) sample ->
        FileNotFoundError (404 at the API). Zero storage, zero
        mutation, byte-identical over unchanged state."""
        record = self._scope_data_artifact(
            lambda sid: self.samples.get_sample(model_id, sid),
            sample_id, "sample")
        files, nbytes = self._artifact_files(
            self.samples._sample_dir(model_id, sample_id))
        integrity_verified = (
            self.samples.result_hash(record) == record.result_hash)
        blockers = self.sample_deletion_blockers(model_id, sample_id)
        return SampleRetentionOverview(
            model_id=model_id,
            sample_id=sample_id,
            checkpoint_id=record.checkpoint_id,
            created_at=record.created_at,
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

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

    def delete_sample_evaluation(
            self, model_id: str,
            evaluation_id: str) -> SampleEvaluationDeletionResult:
        """Explicit VERIFIED sample-quality measurement retention
        (M68): remove ONE measurement — after scope and integrity
        verification. (1) scope through the family getter (unknown
        model, unknown measurement, or registry-invisible ->
        FileNotFoundError, nothing deleted); (2) INTEGRITY FIRST: the
        record's persisted ``result_hash`` must reproduce —
        tampered/corrupt -> RuntimeError (409); (3) NO reference
        guard — a measurement is a LEAF record: nothing persists its
        evaluation_id (deleting it is exactly what UNBLOCKS the
        sample it measured, live through the M67 model guard's ONE
        M66 analysis); (4) ATOMIC removal of the measurement's own
        directory (``SampleQualityEngine.delete``). No cascade, no
        force, no bulk mode."""
        record = self._scope_data_artifact(
            lambda eid: self.sample_quality.get_sample_evaluation(
                model_id, eid),
            evaluation_id, "sample evaluation")
        if self.sample_quality.result_hash(record) != record.result_hash:
            raise RuntimeError(
                f"sample evaluation '{evaluation_id}' failed integrity "
                f"verification — refusing to delete corrupted storage")
        files, nbytes = self.sample_quality.delete(model_id, evaluation_id)
        log.info("M68 verified sample-evaluation deletion %s/%s (%d files,"
                 " %d bytes)", model_id, evaluation_id, files, nbytes)
        return SampleEvaluationDeletionResult(
            model_id=model_id, evaluation_id=evaluation_id,
            sample_id=record.sample_id,
            files_removed=files, bytes_reclaimed=nbytes)

    def sample_evaluation_retention_overview(
            self, model_id: str,
            evaluation_id: str) -> SampleEvaluationRetentionOverview:
        """Read-only live-computed retention overview of ONE
        sample-quality measurement (M68): the deletion-readiness view
        — identity, the ordered artifact files + total bytes of the
        measurement's OWN directory, the record's result-hash
        integrity outcome, ``deletable`` and the ordered blockers (the
        SAME list the DELETE guard refuses on — always empty: a
        measurement is a LEAF record, nothing persists its
        evaluation_id). A tampered/corrupt record is never deletable;
        an unknown or registry-invisible (unparseable manifest) record
        -> FileNotFoundError (404 at the API). Zero storage, zero
        mutation, byte-identical over unchanged state."""
        record = self._scope_data_artifact(
            lambda eid: self.sample_quality.get_sample_evaluation(
                model_id, eid),
            evaluation_id, "sample-quality measurement")
        files, nbytes = self._artifact_files(
            self.sample_quality._evaluation_dir(model_id, evaluation_id))
        integrity_verified = (
            self.sample_quality.result_hash(record) == record.result_hash)
        blockers: list[ArtifactDeletionBlocker] = []
        return SampleEvaluationRetentionOverview(
            model_id=model_id,
            sample_id=record.sample_id,
            evaluation_id=evaluation_id,
            files=files,
            size_bytes=nbytes,
            integrity_verified=integrity_verified,
            deletable=integrity_verified and not blockers,
            blockers=blockers)

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
