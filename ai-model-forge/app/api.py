"""AI Model Forge — REST API (FastAPI).

Routes are deliberately few: health/project introspection, system (hardware),
and the model registry (create/list/get/delete/verify + weights download).
Training, data and evaluation routes arrive with their engines in later
milestones — no placeholder endpoints are exposed.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from . import config as forge_cfg
from .engine import ModelForge, get_forge
from .schemas import (
    ComparisonRecord,
    ComparisonRequest,
    ComparisonVerdict,
    EvalStateKind,
    EvaluationConfig,
    EvaluationRecord,
    EvaluationSplit,
    GateDecision,
    GateDecisionResult,
    GateBaselineType,
    GateRequest,
    HealthResponse,
    ModelDashboard,
    ModelCreateRequest,
    PolicyCreateRequest,
    PolicyDefinition,
    ProbeSuite,
    ProbeSuiteCreateRequest,
    SuiteRunRecord,
    SuiteRunRequest,
    SuiteRunSummary,
    ModelRecord,
    ModelUsageOverview,
    ProjectInfo,
    ProjectStorageOverview,
    SampleGenerateRequest,
    SampleEvaluationRecord,
    SampleRecord,
    SampleStrategy,
    BestCheckpointHistory,
    CheckpointDeletionResult,
    CheckpointRetentionOverview,
    CheckpointSelection,
    DatasetDeletionBlocked,
    DatasetDeletionResult,
    DatasetRetentionOverview,
    DatasetUsageOverview,
    RollbackRequest,
    TokenizerConfig,
    TokenizerDeletionBlocked,
    TokenizerDeletionResult,
    TokenizerRetentionOverview,
    TokenizerUsageOverview,
    TrainingConfig,
    TrainingReport,
    TransformerConfig,
    ValidationReport,
    WorkflowPlan,
    WorkflowRecord,
    WorkflowRecipe,
    WorkflowRecipeCreateRequest,
    WorkflowRecipeRunRequest,
    WorkflowRecipeRepetitionRun,
    WorkflowRecipeResolution,
    WorkflowStatus,
)

forge_cfg.init_logging()

app = FastAPI(
    title="AI Model Forge",
    description="Create, train, evaluate and deploy user-owned AI models.",
    version=forge_cfg.APP_VERSION,
)
api = APIRouter(prefix=forge_cfg.API_PREFIX)


def _forge() -> ModelForge:
    return get_forge()


def _conflict_or_422(exc: ValueError) -> HTTPException:
    """409 for name/dependency conflicts, 422 for unusable inputs."""
    detail = str(exc)
    code = 409 if ("already exists" in detail or "delete those tokenizers" in detail) else 422
    return HTTPException(status_code=code, detail=detail)


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", app="ai-model-forge", version=forge_cfg.APP_VERSION)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Minimal self-contained landing page (no external assets)."""
    version = forge_cfg.APP_VERSION
    prefix = forge_cfg.API_PREFIX
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AI Model Forge</title>
<style>
  body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0f1420;color:#e8ecf4;
       margin:0;padding:48px 24px;display:flex;flex-direction:column;align-items:center;}}
  main{{max-width:760px;width:100%;}} h1{{letter-spacing:.5px;}}
  code{{background:#1c2436;padding:2px 6px;border-radius:6px;font-size:.9em;}}
  .card{{background:#171f31;border:1px solid #26314d;border-radius:12px;padding:18px 22px;margin:14px 0;}}
  a{{color:#7fb2ff;}} ul{{line-height:1.8;}} .ok{{color:#5fd47e;font-weight:600;}}
</style></head><body><main>
  <h1>⚒ AI Model Forge <span style="color:#8fa1c0;font-size:.6em">v{version}</span></h1>
  <div class="card"><b>Project status</b><div id="status">Checking…</div></div>
  <div class="card"><b>Milestones live</b>
    <ul>
      <li><b>M1 model foundation</b> — configurable decoder-only Transformer (MHA/GQA, RoPE,
        SwiGLU, trained/learned embeddings), content-hashed immutable weights.</li>
      <li><b>M2 data engine</b> — versioned datasets, sha256 registry, deterministic
        90/5/5 splits, byte-level BPE tokenizer, compressed tokenized streams.</li>
      <li><b>M3 training engine</b> — continued pretraining / SFT (causal LM) on the
        user's own tokenizer+model, AdamW, LR schedules, immutable checkpoints,
        content-addressed storage, evaluation gate, best-rollback.</li>
      <li><b>M4 evaluation engine</b> — read-only measurement of the current weights
        or any verified checkpoint against a tokenized split (loss / perplexity /
        token &amp; record counts), deterministic result hashes, immutable records.</li>
      <li><b>M5 comparison engine</b> — evidence-based A/B: two states of a model
        measured under one identical probe, verdict improved / regressed / unchanged
        (loss-only, tolerance-based), evaluation reuse, immutable comparison history.</li>
      <li><b>M6 stage gates</b> — explicit policy-driven run decisions: given an inline
        policy (fixed probe, baseline form, tolerance / max regression / absolute
        minimum loss) and a candidate state, answer passed / failed with a full
        evidence chain (decision → evaluations → comparison → state). Baselines:
        checkpoint, current weights, a past evaluation result hash, or an absolute
        loss threshold. Immutable append-only decisions; suggests rollback, never
        executes it.</li>
      <li><b>M7 workflows</b> — ordered, auditable orchestration: one inline plan
        (train → evaluate → compare → gate stages with explicit stage references,
        forward branches on gate pass/fail, stop-on-fail with rollback suggestion)
        executed synchronously by the existing M3–M6 engines; one immutable run
        manifest per execution; no automatic rollback/retraining/model choice.</li>
      <li><b>M8 read-only dashboard</b> — deterministic per-model history views over
        the existing immutable manifests: checkpoint lineage, evaluation series by
        exact probe identity, comparison series by state pair + probe, gate decision
        series by recorded policy, workflow run history with status counts, and a
        reference graph (checkpoint → evaluation → comparison → gate → workflow).
        Live recomputation, zero writes, corrupt manifests skipped with diagnostics,
        hash-stable output; never a quality score or leaderboard.</li>
      <li><b>M9 stable policy registry &amp; named probe suites</b> — reusable immutable
        definitions over the existing semantics: a policy is a stable id for one
        exact M6 GatePolicy (config hash), a probe suite is a stable id for a set of
        exact M4 probes (canonical, probes hash). Same id + same content resolves
        idempotently; same id + different content is a 409 — nothing is ever
        overwritten. M6 gates and M7 workflows may reference policies by id; the
        executed policy is still embedded in every decision. No new decision maths,
        no scores, no execution agents.</li>
      <li><b>M10 explicit suite runs</b> — one named probe suite executed against ONE
        explicit model state (current or checkpoint): each probe runs as an
        independent exact M4 evaluation (reused by exact M4 identity when the
        evidence already exists) and one immutable suite-run manifest is appended.
        Per-probe results only — execution bookkeeping counts, never an aggregate
        score, benchmark or ranking. A failed probe is recorded per-probe; nothing
        is fabricated. Suites never trigger comparisons, gates or training.</li>
      <li><b>M11 suite-run stages inside workflows</b> — M7 workflows gain a
        <code>suite_run</code> stage: one named suite executed against one explicitly
        resolved state (literal checkpoint, current weights, or <code>from_stage</code>
        → an earlier train stage's final checkpoint — nothing guessed), through the
        existing M10 engine. The workflow record references the immutable suite-run
        artifact; the suite never becomes a single quality judgment, gates stay
        one-probe, and all M7 control flow (transitions, on_pass/on_fail, failure
        recording) is unchanged.</li>
      <li><b>M12 named workflow recipes</b> — immutable, reusable M7 pipeline
        definitions: one registered ordered <code>WorkflowStage</code> list (same
        validation rules as inline plans, shared validator) with a deterministic
        config hash, invoked only by an explicit run request that binds ONE model
        (never auto-selected). Execution converts the recipe into an existing
        <code>WorkflowPlan</code> and runs the existing WorkflowEngine; the run
        record carries recipe provenance (<code>recipe_id</code>/<code>recipe_hash</code>,
        null for inline runs). Same id + same stages resolves idempotently; same id +
        different stages is 409. Recipes are inert data — no scheduling, no
        automatic execution, no model/checkpoint/suite selection, no aggregation.</li>
      <li><b>M13 read-only observability</b> — zero-write views over the existing
        immutable manifests: the per-model dashboard gains a deterministic
        <code>suite_runs</code> section (the model's M10 suite-run records, scanned
        live from the storage-root <code>suite-runs/</code> family and filtered by
        the persisted <code>model_id</code>; counts of recorded statuses only,
        per-probe evaluation references verbatim; corrupt runs skipped with
        diagnostics) and the artifact graph gains <code>suite_run</code> nodes with
        <code>workflow → suite_run → evaluation</code> edges for recorded references
        only. Recipe runs expose cross-model lineage via
        <code>GET /workflows/recipes/&#123;recipe_id&#125;/runs</code> (live scan,
        <code>(created_at, workflow_id)</code> order, 404 unknown recipe). No suite
        score, aggregate, ranking or recipe-quality number is computed anywhere, and
        M13 never writes.</li>
      <li><b>M14 composable workflow recipes</b> — registered recipes may contain a
        <code>recipe</code> stage referencing other registered recipes (one
        <code>recipe_id</code>, no parameters, no runtime substitution).
        Registration-time validation resolves every referenced recipe against the
        immutable registry, rejects cycles (self, 2-node, longer) and compositions
        deeper than 32 recipes, then deterministically EXPANDS the composition —
        stages spliced in parent order with every id qualified by the dot-joined
        call-stage path (<code>call.stage</code>; nested calls chain:
        <code>a.b.stage</code>) and every INTERNAL <code>from_stage</code> /
        <code>on_pass</code> / <code>on_fail</code> reference rewritten to its
        qualified id — and re-runs the shared M7 validator over the fully expanded
        list (cross-boundary references are rejected, never invented). No manifest
        is written when any check fails; conflict re-registration stays 409 with the
        original bytes untouched. Composite config hashes cover the declared stages
        PLUS the referenced recipes' ids and config hashes, so an immutable
        dependency pins its consumer. Running a composite executes ONE
        WorkflowEngine run (the existing engines — nested recipes never bind or
        select a model; embedded model pins are never rewritten and conflicts are
        rejected pre-run), records ONE workflow manifest whose provenance names the
        TOP-LEVEL recipe plus an additive reference-oriented composition trace, and
        reuses M4/M10 evidence exactly (no duplicate evaluation manifests).
        Lineage (<code>GET .../runs</code>) keeps M13 semantics: only the recipe the
        caller invoked owns the run. Recipe stages never appear in inline plans.</li>
      <li><b>M15 checkpoint sampling</b> — the platform's first inference primitive:
        deterministic text generation from ONE explicitly selected, verified,
        immutable checkpoint (<code>model_id</code> + <code>checkpoint_id</code> +
        <code>tokenizer_id</code> + prompt + strategy, never auto-selected). Exactly
        two strategies — <code>greedy</code> (deterministic argmax, no RNG) and
        seeded <code>temperature</code> sampling (one deterministic RNG stream from
        the request's explicit integer seed) — with a 512-token hard cap and the
        context window enforced at preflight (never silently truncated). The
        checkpoint passes the existing content-hash integrity verification, the
        tokenizer/model vocabulary compatibility follows the exact M3/M4 platform
        rule, and logits are restricted to the tokenizer's vocab so every generated
        token decodes. One immutable sample manifest per request under a new
        <code>samples/</code> family; a deterministic <code>result_hash</code> over
        the semantic payload reproduces identical requests byte-for-byte. Pure
        CPU-first inference under no_grad — no training, no evaluation, no quality
        judgment of any kind.</li>
      <li><b>M16 per-sample quality measurement</b> — the platform's first
        measurement layer over generated text: given one immutable M15 sample
        (<code>model_id</code> + <code>sample_id</code> ONLY), the sample's own
        recorded checkpoint and tokenizer are resolved and verified (content-hash
        integrity + the platform vocabulary convention; nothing is ever
        auto-selected or supplied separately). The full prompt + generated
        sequence is scored in ONE context window under the existing M4 causal-LM
        objective over the generated-continuation target tokens only (prompt
        tokens condition but never count; the first generated token is conditioned
        on the full prompt; every generated target counted exactly once; overlong
        samples rejected 422, never silently truncated). Metrics are exactly
        <code>loss_nats</code> and <code>perplexity = exp(min(loss,100))</code> —
        a likelihood measurement under one objective, explicitly NOT an overall
        quality judgment. One immutable manifest per measurement under a new
        <code>sample-evaluations/</code> family; deterministic
        <code>result_hash</code>; samples/checkpoints/tokenizers/M4 evaluations
        are never modified.</li>
      <li><b>M17 sample-quality observability</b> — the M8/M13 model dashboard
        now exposes a deterministic, read-only <code>sample_quality</code>
        section derived LIVE from the model's immutable M16
        <code>sample-evaluations/</code> manifests: total record count,
        per-sample history references and a newest-first evaluation reference
        list. History structure only — no loss/perplexity values, no
        averages, no rankings and no quality judgment. Zero storage growth:
        the dashboard stays a pure recomputation.</li>

      <li><b>M18 sample-quality record access</b> — one dedicated read-only
        listing route,
        <code>GET /models/&#123;id&#125;/sample-quality/records</code>, returns
        the model's FULL immutable M16 <code>SampleEvaluationRecord</code>
        payloads (the recorded <code>loss_nats</code>/<code>perplexity</code>
        values included) in the exact M16 (created_at, evaluation_id) order —
        a pure pass-through of the M16 engine, no derived statistics, no
        aggregation, zero storage growth; records remain immutable
        likelihood measurements under one causal-LM objective, not an
        overall quality score.</li>

      <li><b>M19 per-sample quality history</b> — one read-only access path,
        <code>GET /models/&#123;id&#125;/sample-quality/by-sample/&#123;sample&#125;</code>,
        answers "what immutable M16 measurements exist for this known
        generated sample?": the model's authoritative M16 listing filtered
        to the sample's persisted identity (full record payloads, exact M16
        ordering; unknown model/sample or a sample of another model ->
        404; sample without measurements -> []). Pure data access — no
        derived fields, no statistics, no quality judgment, zero storage
        growth.</li>

      <li><b>M20 per-checkpoint quality history</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/sample-quality/by-checkpoint/&#123;checkpoint&#125;</code>,
        answers "which immutable sample-quality evaluations belong to this
        checkpoint?": the model's authoritative M16 listing filtered by the
        persisted checkpoint identity recorded in each record (full record
        payloads, exact M16 ordering; unknown model/checkpoint or a
        checkpoint of another model -> 404; checkpoint without measurements
        -> []). Pure data access — no derived fields, no statistics, no
        quality judgment, no checkpoint comparison, zero storage growth.</li>

      <li><b>M21 per-suite suite-run history</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/suite-runs/by-suite/&#123;suite_id&#125;</code>,
        answers "which immutable suite-run records belong to this model and
        this named suite?": the model's authoritative M10 listing filtered
        by the persisted suite_id recorded in each record, after suite
        existence is verified through the M9 registry (full record
        payloads, exact M10 ordering; unknown model or suite -> 404; a
        valid suite without runs for this model -> []). Pure data access —
        no aggregation, no scores, no ranking, zero storage growth.</li>

      <li><b>M22 per-suite bookkeeping summary</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/suite-runs/by-suite/&#123;suite_id&#125;/summary</code>,
        answers "how many suite runs exist for this model and suite, which
        ones, and when did they run?": a pure derived counting view over
        the M21 grouping — total_count, ordered run ids, earliest/latest
        recorded timestamps. A valid suite without runs returns a zero
        summary; unknown model/suite -> 404. No scores, averages, trends
        or judgments; never persisted, zero storage growth.</li>

      <li><b>M23 gate-decision history by policy</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/gates/decisions/by-policy/&#123;policy_id&#125;</code>,
        answers "which immutable gate decisions of this model were produced
        under this registered policy?": the model's authoritative M6
        listing filtered by the persisted policy_id recorded in each
        decision, after policy existence is verified through the M9
        registry (full record payloads, exact M6 ordering; unknown model
        or policy -> 404; a valid policy without decisions for this model
        -> []; inline-policy decisions never appear). Pure data access —
        no aggregation, no verdicts beyond the persisted ones, zero
        storage growth.</li>

      <li><b>M34 gate-decision history by comparison</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/gates/decisions/by-comparison/&#123;comparison_id&#125;</code>,
        answers "which immutable gate decisions of this model judged
        this M5 comparison?": the model's authoritative M6 listing
        filtered by the persisted comparison_id recorded in each
        decision, after comparison ownership is verified through the
        model's own M5 registry (full record payloads incl. verdict,
        decision, losses, delta and reason, exact M6 ordering; unknown
        model or comparison -> 404; a comparison of another model ->
        404; a valid comparison without decisions -> []; legacy
        direct-evaluation decisions with null comparison_id never
        appear). Pure data access — no aggregation, no new verdicts,
        zero storage growth.</li>

      <li><b>M35 workflow history by recipe</b> — one read-only
        MODEL-SCOPED access path, <code>GET /models/&#123;id&#125;/workflows/by-recipe/&#123;recipe_id&#125;</code>,
        answers "which immutable M11 workflow runs of this model were
        executed from this registered recipe?": the model's
        authoritative M11 listing filtered by the persisted top-level
        recipe_id (with its recorded recipe_hash provenance preserved
        verbatim), after the recipe is validated through the GLOBAL
        M12/M14 registry (recipes are global; model scoping from the
        model's own listing; full record payloads, exact M11
        ordering; unknown model or recipe -> 404; a valid recipe with
        no runs for this model -> []; ad-hoc runs with null recipe_id
        never appear). The model-scoped complement of the GLOBAL M12
        recipe-runs lineage surface, which stays unchanged. Pure data
        access — no execution, no aggregation, zero storage
        growth.</li>

      <li><b>M36 evaluation history by split</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/evaluations/by-split/&#123;split&#125;</code>,
        answers "which immutable M4 evaluations of this model measured
        this dataset split?": the model's authoritative M4 listing
        filtered by the persisted top-level split (schema enum
        train/validation/test, matched VERBATIM — never inferred from
        filenames, datasets or timestamps; full record payloads, exact
        M4 ordering; unknown model -> 404; an UNSUPPORTED split value
        -> 422, because splits have no registry — the enum IS the
        contract; a valid split with no evaluations for the model ->
        []). Pure data access — no aggregates, zero storage
        growth.</li>

      <li><b>M38 evaluation history by state kind</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/evaluations/by-state-kind/&#123;state_kind&#125;</code>,
        answers "which immutable M4 evaluations of this model measured
        which kind of model state?": the model's authoritative M4
        listing filtered by the persisted top-level state_kind
        (schema enum current/checkpoint, matched VERBATIM — never
        inferred from filenames, checkpoint_id nullability or
        timestamps; full record payloads, exact M4 ordering; unknown
        model -> 404; an UNSUPPORTED state-kind value -> 422, because
        state kinds have no registry — the enum IS the contract; a
        valid state kind with no evaluations for the model -> []).
        Pure data access — no aggregates, zero storage growth.</li>

      <li><b>M24 evaluation history by checkpoint</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/evaluations/by-checkpoint/&#123;checkpoint&#125;</code>,
        answers "which immutable M4 evaluations measured this checkpoint?":
        the model's authoritative M4 listing filtered by the persisted
        checkpoint identity recorded in each record, after checkpoint
        ownership is validated through the M3 registry (full record
        payloads, exact M4 ordering; unknown model/checkpoint or a
        checkpoint of another model -> 404; a checkpoint without
        evaluations -> []; current-state evaluations never appear). Pure
        data access — no derived statistics, no comparison, zero storage
        growth.</li>

      <li><b>M25 suite-run history by checkpoint</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/suite-runs/by-checkpoint/&#123;checkpoint&#125;</code>,
        answers "which immutable M10 suite runs executed against this
        checkpoint state?": the model's authoritative M10 listing filtered
        by the persisted run state recorded in each record, after
        checkpoint ownership is validated through the M3 registry (full
        record payloads, exact M10 ordering; unknown model/checkpoint or
        a checkpoint of another model -> 404; a valid checkpoint without
        runs -> []; current-state runs never appear). Pure data access —
        no aggregation, zero storage growth.</li>

      <li><b>M26 comparison history by checkpoint</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/comparisons/by-checkpoint/&#123;checkpoint&#125;</code>,
        answers "which immutable M5 comparisons involve this checkpoint on
        either side?": the model's authoritative M5 listing filtered by
        the persisted side states recorded in each record (both sides
        examined; a same-checkpoint A=B comparison appears exactly once;
        current-state sides never match), after checkpoint ownership is
        validated through the M3 registry (full record payloads, exact M5
        ordering; unknown model/checkpoint or a checkpoint of another
        model -> 404; a checkpoint with no matching comparisons -> []).
        Pure data access — no new metrics, zero storage growth.</li>

      <li><b>M27 sample history by checkpoint</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/samples/by-checkpoint/&#123;checkpoint&#125;</code>,
        answers "which immutable M15 samples were generated from this
        checkpoint?": the model's authoritative M15 listing filtered by
        the persisted checkpoint identity recorded in every sample
        (M15 generation always binds one explicit verified checkpoint;
        membership never comes from filenames, timestamps or hashes),
        after checkpoint ownership is validated through the M3 registry
        (full record payloads, exact M15 ordering; unknown
        model/checkpoint or a checkpoint of another model -> 404; a
        checkpoint with no samples -> []). Pure data access — no new
        metrics, zero storage growth.</li>

      <li><b>M28 evaluation history by dataset</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/evaluations/by-dataset/&#123;dataset&#125;</code>,
        answers "which immutable M4 evaluations of this model measured
        this dataset?": the model's authoritative M4 listing filtered by
        the persisted dataset identity recorded in every evaluation
        (top-level dataset_id + dataset_version travelling VERBATIM in
        each record — versions never collapsed or rewritten), after the
        dataset is validated through the M2 registry (full record
        payloads, exact M4 ordering; datasets are global so the model
        scoping comes from the model's own listing; unknown
        model/dataset -> 404; a dataset with no evaluations for the
        model -> []). Pure data access — no aggregates, zero storage
        growth.</li>

      <li><b>M29 comparison history by dataset</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/comparisons/by-dataset/&#123;dataset&#125;</code>,
        answers "which immutable M5 comparisons of this model measured
        this dataset?": the model's authoritative M5 listing filtered by
        the persisted shared-probe dataset identity (a comparison
        persists exactly ONE dataset_id + dataset_version — both sides
        measure the same probe by construction; per-side dataset
        identities cannot occur), each matching comparison EXACTLY
        ONCE (dedup by comparison identity), versions VERBATIM, after
        the dataset is validated through the M2 registry (full record
        payloads, exact M5 ordering; unknown model/dataset -> 404; a
        dataset with no comparisons for the model -> []). Pure data
        access — no aggregates, zero storage growth.</li>

      <li><b>M30 evaluation history by tokenizer</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/evaluations/by-tokenizer/&#123;tokenizer&#125;</code>,
        answers "which immutable M4 evaluations of this model measured
        with this tokenizer?": the model's authoritative M4 listing
        filtered by the persisted tokenizer identity recorded in every
        evaluation (matched VERBATIM — never inferred from filenames
        or substituted with the latest tokenizer), after the tokenizer
        is validated through the existing registry (full record
        payloads, exact M4 ordering; tokenizers are global so the
        model scoping comes from the model's own listing; unknown
        model/tokenizer -> 404; a tokenizer with no evaluations for
        the model -> []). Pure data access — no aggregates, zero
        storage growth.</li>

      <li><b>M31 comparison history by tokenizer</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/comparisons/by-tokenizer/&#123;tokenizer&#125;</code>,
        answers "which immutable M5 comparisons of this model measured
        with this tokenizer?": the model's authoritative M5 listing
        filtered by the persisted shared-probe tokenizer identity (a
        comparison persists exactly ONE tokenizer_id — both sides
        measure the same probe by construction), each matching
        comparison EXACTLY ONCE (dedup by comparison identity), the
        persisted id matched VERBATIM, after the tokenizer is
        validated through the existing registry (full record payloads,
        exact M5 ordering; unknown model/tokenizer -> 404; a tokenizer
        with no comparisons for the model -> []). Pure data access —
        no aggregates, zero storage growth.</li>

      <li><b>M37 comparison history by split</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/comparisons/by-split/&#123;split&#125;</code>,
        answers "which immutable M5 comparisons of this model measured
        this dataset split?": the model's authoritative M5 listing
        filtered by the persisted shared-probe split (a comparison
        persists exactly ONE top-level split — both sides measure the
        same probe by construction), matched VERBATIM — never inferred
        from datasets, checkpoints, nested evaluation records or
        timestamps; full record payloads, exact M5 ordering; unknown
        model -> 404; an UNSUPPORTED split value -> 422, because
        splits have no registry — the EvaluationSplit enum IS the
        contract; a valid split with no comparisons for the model ->
        []). Pure data access — no aggregates, zero storage
        growth.</li>

      <li><b>M39 comparison history by verdict</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/comparisons/by-verdict/&#123;verdict&#125;</code>,
        answers "which immutable M5 comparisons of this model produced
        this verdict?": the model's authoritative M5 listing filtered
        by the persisted top-level verdict (schema enum
        improved/regressed/unchanged — the immutable loss-only
        judgment of ONE probe), matched VERBATIM — NEVER recalculated
        from loss deltas, per-side losses or tolerances, never
        executed; full record payloads, exact M5 ordering; unknown
        model -> 404; an UNSUPPORTED verdict value -> 422, because
        verdicts have no registry — the enum IS the contract; a valid
        verdict with no comparisons for the model -> []). Pure data
        access — no aggregates, no rankings, zero storage
        growth.</li>

      <li><b>M32 sample history by tokenizer</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/samples/by-tokenizer/&#123;tokenizer&#125;</code>,
        answers "which immutable M15 samples of this model were
        generated with this tokenizer?": the model's authoritative M15
        listing filtered by the persisted sample tokenizer identity
        (every sample carries a required non-nullable top-level
        tokenizer_id plus its matching tokenizer_hash, preserved
        verbatim), each matching sample EXACTLY ONCE, the persisted id
        matched VERBATIM, after the tokenizer is validated through the
        existing GLOBAL registry (model scoping from the model's own
        listing; full record payloads, exact M15 ordering; unknown
        model/tokenizer -> 404; a tokenizer with no samples for the
        model -> []). Pure data access — no aggregates, no generation,
        zero storage growth.</li>

      <li><b>M40 sample history by strategy</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/samples/by-strategy/&#123;strategy&#125;</code>,
        answers "which immutable M15 samples of this model were
        generated with this decoding strategy?": the model's
        authoritative M15 listing filtered by the persisted top-level
        strategy (schema enum greedy/temperature, persisted verbatim
        at generation time), matched VERBATIM — NEVER recalculated
        from temperature, seed or any other field, never regenerated;
        full record payloads, exact M15 ordering; unknown model ->
        404; an UNSUPPORTED strategy value -> 422, because strategies
        have no registry — the enum IS the contract; a valid strategy
        with no samples for the model -> []). Pure data access — no
        aggregates, no scoring, zero storage growth.</li>

      <li><b>M41 gate-decision history by decision</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/gates/decisions/by-decision/&#123;decision&#125;</code>,
        answers "which immutable gate decisions of this model produced
        this decision result?": the model's authoritative M6 listing
        filtered by the persisted top-level decision (schema enum
        passed/failed — the immutable policy verdict of the M6 run,
        which may legitimately differ from the loss-only comparison
        verdict), matched VERBATIM — NEVER recalculated from loss
        deltas, policy thresholds or comparison results, never
        re-evaluated; full record payloads, exact M6 ordering; unknown
        model -> 404; an UNSUPPORTED decision value -> 422, because
        decision results have no registry — the enum IS the contract;
        a valid decision with no gate decisions for the model -> []).
        Pure data access — no aggregates, no rankings, zero storage
        growth.</li>

      <li><b>M42 workflow history by status</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/workflows/by-status/&#123;status&#125;</code>,
        answers "which immutable workflow runs of this model ended
        with this terminal status?": the model's authoritative M11
        listing filtered by the persisted top-level status (schema
        enum completed/failed/stopped — the terminal state recorded
        at run end by the M7 orchestration), matched VERBATIM —
        NEVER inferred from stage results, failed stage ids,
        timestamps, artifact existence or recipe information, never
        re-executed; full record payloads, exact M11 ordering;
        unknown model -> 404; an UNSUPPORTED status value -> 422,
        because statuses have no registry — the enum IS the
        contract; a valid status with no matching runs for the model
        -> []). Pure data access — no aggregation, no analytics,
        zero storage growth.</li>

      <li><b>M43 gate-decision history by verdict</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/gates/decisions/by-verdict/&#123;verdict&#125;</code>,
        answers "which immutable gate decisions of this model
        recorded this loss-only comparison verdict?": the model's
        authoritative M6 listing filtered by the persisted top-level
        verdict (the Optional[ComparisonVerdict] enum
        improved/regressed/unchanged, recorded verbatim by the M6 run
        — deliberately DISTINCT from the M41 decision result: an
        improved candidate can still fail a policy), matched VERBATIM
        — NEVER recalculated from losses, deltas, tolerances or
        comparisons, never re-evaluated; full record payloads, exact
        M6 ordering; unknown model -> 404; an UNSUPPORTED verdict
        value -> 422, because verdicts have no registry — the enum IS
        the contract; a valid verdict with no matching decisions ->
        []; threshold-only gates keep verdict null — there is
        deliberately NO route for None, and those decisions belong to
        NO by-verdict group). Pure data access — no aggregates, no
        rankings, zero storage growth.</li>

      <li><b>M44 comparison history by state kind</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/comparisons/by-state-kind/&#123;state_kind&#125;</code>,
        answers "which immutable comparisons of this model involve
        the requested kind of model state on either side?": the
        model's authoritative M5 listing filtered by the persisted
        state kind of the comparison sides (the schema enum
        current/checkpoint on each persisted side — M26's
        either-side semantics: a comparison belongs when EITHER side
        records the kind; a both-sides match appears exactly once),
        matched VERBATIM — NEVER inferred from checkpoint ids or
        hashes, never recalculated, nothing re-executed; full record
        payloads, exact M5 ordering; unknown model -> 404; an
        UNSUPPORTED state-kind value -> 422, because state kinds have
        no registry — the enum IS the contract; a valid state kind
        with no matching comparisons -> []). Pure data access — no
        aggregates, no rankings, zero storage growth.</li>

      <li><b>M45 gate-decision history by baseline type</b> — one
        read-only access path, <code>GET /models/&#123;id&#125;/gates/decisions/by-baseline-type/&#123;baseline_type&#125;</code>,
        answers "which immutable gate decisions of this model
        compared their candidate against this kind of baseline?":
        the model's authoritative M6 listing filtered by the
        persisted NESTED policy field <code>policy.baseline_type</code>
        (the schema enum checkpoint / current /
        evaluation_result_hash / minimum_loss, embedded VERBATIM in
        each decision — the M6 run persists the policy exactly as
        evaluated), matched VERBATIM — NEVER derived from checkpoint
        id presence, references, results, verdicts or loss deltas,
        never re-evaluated; the REQUIRED field makes the groups a
        TRUE disjoint partition with no None case; the enum exposes
        the COMPLETE contract — evaluation_result_hash is a valid
        value that naturally returns [] where no such gates exist;
        full record payloads, exact M6 ordering; unknown model ->
        404; an UNSUPPORTED baseline type -> 422, because baseline
        types have no registry — the enum IS the contract. Pure data
        access — no aggregates, no rankings, zero storage
        growth.</li>

      <li><b>M46 checkpoint history by training run</b> — one
        read-only access path, <code>GET /models/&#123;id&#125;/checkpoints/by-run/&#123;run&#125;</code>,
        answers "which immutable checkpoints did ONE training run of
        this model produce?": the model's authoritative M3 listing
        filtered by each checkpoint's own persisted
        <code>run_id</code>, validated against the model's OWN
        manifest <code>training_provenance</code> (an unknown run or
        a run belonging to another model -> 404; membership NEVER
        inferred from checkpoint directories, steps, epochs,
        timestamps, losses or parent ids); exact (step, created_at)
        ordering; a registered run with zero checkpoints -> 200 [];
        no separate run registry is introduced — the model manifest
        IS the registry. Pure data access — no aggregates, no
        lineage graphs, zero storage growth.</li>

      <li><b>M47 evaluation history by truncation</b> — one
        read-only access path, <code>GET /models/&#123;id&#125;/evaluations/by-truncated/&#123;truncated&#125;</code>,
        answers "which immutable evaluations of this model were
        stopped early by the max_eval_tokens cap?": the model's
        authoritative M4 listing filtered by the persisted REQUIRED
        boolean <code>truncated</code> the engine recorded at run
        time (True = the cap stopped the evaluation before the split
        ended; False = the configured/permitted stream was consumed
        without the cap cutting it short) — matched VERBATIM, never
        recalculated, never derived from records_covered, token
        counts, split length, configuration, timestamps or
        durations; the boolean carries NO quality judgment — it is
        engine metadata about how far the evaluation stream was
        consumed, nothing more. The closed two-value contract makes
        the groups a TRUE disjoint partition with no None case;
        non-boolean spellings -> 422 (schema-level, pre-handler);
        unknown model -> 404; full record payloads, exact M4
        ordering. Pure data access — no aggregates, no re-runs, no
        automatic coverage enforcement, zero storage growth.</li>

      <li><b>M48 evaluation history by seed</b> — one
        read-only access path, <code>GET /models/&#123;id&#125;/evaluations/by-seed/&#123;seed&#125;</code>,
        answers "which immutable evaluations of this model ran with
        this effective seed?": the model's authoritative M4 listing
        filtered by the REQUIRED integer <code>seed</code> persisted
        on each record at run time (the effective seed used, default
        derived from the config) — matched VERBATIM, never
        recalculated, never normalized, never derived from the
        embedded config dict, dataset identity, splits, tokenizers,
        state kinds, losses, ids or timestamps. The seed is
        bookkeeping identity, not a quality metric — no seed is
        better than another. The seed is an OPEN integer value axis
        (no registry, no enum): any integer is type-valid — an
        unmatched seed -> 200 [] — while a non-integer spelling ->
        422 (schema-level, pre-handler); unknown model -> 404; full
        record payloads, exact M4 ordering. Pure data access — no
        aggregates, no seed sweeps, zero storage growth.</li>

      <li><b>M49 comparison history by seed</b> — one
        read-only access path, <code>GET /models/&#123;id&#125;/comparisons/by-seed/&#123;seed&#125;</code>,
        answers "which immutable A/B comparisons of this model ran
        with this effective seed?": the model's authoritative M5
        listing filtered by the REQUIRED integer <code>seed</code>
        persisted on each record at run time (the seed of the
        identical-probe measurement) — matched VERBATIM, never
        recalculated, never normalized, never derived from the
        comparison configuration, either side's evaluation, state
        payloads, verdicts, loss deltas, ids or timestamps. The seed
        is bookkeeping identity, not a quality metric — no seed
        produces better comparisons. The seed is an OPEN integer
        value axis (no registry, no enum): any integer is
        type-valid — an unmatched seed -> 200 [] — while a
        non-integer spelling -> 422 (schema-level, pre-handler);
        unknown model -> 404; full record payloads, exact M5
        ordering. Pure data access — no aggregates, no seed sweeps,
        zero storage growth.</li>

      <li><b>M50 suite-run history by reused count</b> — one
        read-only access path, <code>GET /models/&#123;id&#125;/suite-runs/by-reused/&#123;reused_count&#125;</code>,
        answers "which immutable suite runs of this model satisfied
        this many probes with pre-existing evidence?": the model's
        authoritative M10 listing filtered by the REQUIRED integer
        <code>reused_count</code> persisted on each record at run
        time — matched VERBATIM, never recalculated, never derived
        from probe outcomes, completed/failed/probe counts, suite
        size, status, timestamps, artifact ids, evaluation or
        comparison records or configuration. The count is EXECUTION
        BOOKKEEPING, never a score — reused evidence is not better
        or worse, it is how the immutable evaluation cache satisfied
        the suite. The count is an OPEN integer value axis (no
        registry, no enum): any integer is type-valid — an unmatched
        count -> 200 [] — while a non-integer spelling -> 422
        (schema-level, pre-handler); unknown model -> 404; full
        record payloads, exact M10 ordering. Pure data access — no
        aggregation, no cache statistics, zero storage growth.</li>

      <li><b>M51 recipe resolution preflight</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/workflows/recipes/&#123;recipe&#125;/plan</code>,
        answers "what EXACTLY would this registered recipe execute
        against this model?": the EXACT resolution path of the recipe
        RUN surface (recipe lookup -> model validation -> M14
        deterministic expansion -> WorkflowPlan construction with the
        FULL M7 validation incl. embedded-config model agreement)
        through the SAME RecipeEngine code — one resolution system,
        never a second executor. Returns the expanded, model-bound,
        fully validated <code>WorkflowPlan</code> (its plan_hash
        predicts the executed run's plan_hash) plus recipe provenance
        and the M14 composition trace; composite recipes resolve to
        their spliced, id-qualified expanded stage list. A computed
        view: NEVER persisted — a resolve leaves zero new files.
        Errors identical to a run: unknown recipe/model -> 404,
        binding conflicts -> 422. Execution itself stays where it was:
        the synchronous <code>POST /workflows/recipes/&#123;recipe&#125;/runs</code>
        through the sole WorkflowEngine — no scheduling, no
        background anything.</li>

      <li><b>M52 best-checkpoint selection</b> — one read-only access
        path, <code>GET /models/&#123;id&#125;/checkpoints/best</code>,
        answers "which of this model's checkpoints has the MINIMUM
        persisted <code>validation_loss</code>?": a deterministic
        computed view over the authoritative M3 checkpoint listing —
        the persisted manifest is the source, validation loss is never
        recomputed and never derived from perplexity, decisions,
        evaluations, ids or timestamps, and non-finite persisted values
        are never candidates. Ties on the exact minimum resolve by the
        listing's canonical (step, created_at) ASCENDING order — the
        first checkpoint among equals — and are disclosed via a
        <code>tied</code> flag. Returns the complete verbatim
        <code>CheckpointRecord</code> plus the explicit criterion and
        candidate count. "Best" means exactly this criterion — NOT a
        claim of overall model quality. Read-only, never writes, no
        persisted selection pointer; unknown model or no selectable
        checkpoints -> 404.</li>

      <li><b>M55 declarative best-resume for train stages</b> — a
        workflow/recipe TRAIN stage may declare
        <code>resume_from_best</code>: the SAME resolver that pins M53
        state refs resolves the SAME M52 selection ONCE per plan and
        pins the concrete checkpoint id; the training engine receives a
        PURE M54 explicit resume (the training layer never queries
        "best"). XOR with the explicit id (both -> 422); a direct
        training run declaring best -> 422; the declarative recipe
        manifest is never rewritten while the immutable run record
        pins the concrete id (a different resolution = a different
        plan_hash). The improvement loop — train, evaluate, train from
        best, evaluate, gate — is ONE finite, explicit, re-runnable
        recipe; no automatic repetition.</li>

      <li><b>M56 declarative best-evaluation stages</b> — a
        workflow/recipe EVALUATE stage may declare
        <code>checkpoint_from_best</code>: the SAME resolver (M53)
        resolves the SAME M52 selection ONCE per plan, pins the
        concrete id and the M4 evaluation path receives a PURE
        explicit-checkpoint probe (the evaluation layer never queries
        "best"; existing exact evidence for the concrete checkpoint is
        reused). XOR with the explicit id and with
        <code>checkpoint_from_stage</code> (both -> 422); the direct
        M4 evaluation route has no such field (name the checkpoint
        explicitly); the declarative recipe manifest is never rewritten
        while the immutable run record pins the concrete id. Every
        stage of the canonical loop — train, evaluate(best), train
        from best, evaluate(best), gate — is now declarative; no
        automatic repetition.</li>

      <li><b>M57 declarative best-baseline gate policies</b> — a
        workflow/recipe GATE stage's INLINE policy may declare
        <code>baseline_from_best</code>: the SAME resolver (M53)
        resolves the SAME M52 selection ONCE per plan, pins the
        concrete id (<code>resolved_baseline_checkpoint_id</code>) and
        the gate engine receives a PURE M6 policy with an explicit
        <code>baseline_checkpoint_id</code> (the gate layer never
        queries "best"). XOR with the explicit id and only valid with
        <code>baseline_type='checkpoint'</code> (the best IS a
        checkpoint baseline; both set -> 422); a DIRECT
        <code>POST /gates/evaluate</code> declaring best -> 422;
        best-baseline policies cannot be REGISTERED (the registry is
        immutable/shared — the pin lives in the per-plan workflow
        record). The decision's embedded policy + baseline side carry
        the concrete checkpoint used. The canonical loop — train,
        evaluate(best), gate(candidate vs best), train from best,
        evaluate(best) — is fully declarative; no automatic
        repetition.</li>

      <li><b>M58 bounded finite recipe repetitions</b> — the recipe-run
        request may declare <code>repetitions: N</code> (default 1,
        bounded 1..16): the recipe executes N times SEQUENTIALLY, each
        iteration a FULL independent resolution + execution, so every
        <code>best</code> declaration (M53 refs, M55 resume, M56
        evaluate, M57 gate baseline) re-resolves at THAT iteration's
        plan start and may advance between iterations. N normal
        immutable workflow records (no wrapper record, no second
        executor, no repetition storage); repetitions=1 reproduces
        today's exact behavior and response; repetitions&gt;1 returns
        an ordered batch view (counts + workflow ids + full records).
        An iteration ending <code>stopped</code> (gate stop) or
        <code>failed</code> ends the sequence — no retry, no skip, no
        rollback. Bounded and explicit only: no unbounded, no
        while-improving, no convergence detection.</li>

      <li><b>M59 best-checkpoint improvement history</b> —
        <code>GET /models/&#123;id&#125;/checkpoints/best/history</code>
        shows the chronological sequence of checkpoints that BECAME the
        M52-selected best (only winners; non-winning checkpoints are
        excluded), each with its persisted validation loss, producing
        run, step, timestamp and the improvement delta vs the previous
        best (negative = improvement; the M5/M6 sign convention). The
        selection semantics are EXACTLY M52's one shared winner rule;
        the FINAL entry always equals the live
        <code>GET .../checkpoints/best</code> answer; losses are
        monotonically non-increasing. A pure computed view over
        persisted manifests — read-only, derived live on every call,
        ZERO storage (no cache, no pointer, no history database). It
        makes the improvement trajectory observable (e.g. an M58
        repeated run's successive advances); it decides nothing — no
        automatic stopping, no convergence detection.</li>

      <li><b>M60 declarative best-publication stage</b> — a
        workflow/recipe PUBLISH stage closes the loop's missing
        ACCEPT step: <code>publish_from_best</code> declares
        <code>PUBLISH(best)</code> and the SAME resolver (M53) resolves
        the SAME M52 selection ONCE per plan, pins the concrete
        checkpoint id (<code>resolved_checkpoint_id</code>), and the
        executor runs the EXISTING M3 verified rollback machinery —
        integrity verification, atomic weights restore as current,
        <code>latest_checkpoint</code> update — never a second
        publication mechanism, never a weights copy, never a new
        pointer system, and the immutable source checkpoint is never
        touched. XOR with the explicit <code>checkpoint_id</code>
        (both or neither -> 422); the direct M3 rollback route keeps
        naming the concrete checkpoint ('best' never leaks into M3).
        The canonical acceptance loop — train, evaluate(best),
        gate(best), publish(best) — is now fully declarative; on a
        gate stop the publish stage is simply skipped (nothing is
        published from a rejected state). No new endpoint: the
        existing <code>POST /workflows/run</code>, recipe runs and
        the M51 preflight all resolve through the same path.</li>

      <li><b>M61 explicit verified checkpoint retention</b> —
        <code>DELETE /models/&#123;id&#125;/checkpoints/&#123;ckpt&#125;</code>
        removes exactly ONE checkpoint, only after live proof that
        nothing authoritative depends on it: the guard verifies
        integrity through the existing M3 verifier (a corrupt
        checkpoint is refused — deletion never bypasses integrity
        validation), then scans the authoritative listings for
        blocking references — the live M52 best, the published
        <code>latest_checkpoint</code>, the manifest's stored
        best-known weights reference, and every persisted checkpoint
        id in workflow / evaluation / comparison / gate / suite-run /
        sample / sample-quality records (via the same read-only
        filters the listing routes use). Protected checkpoints are
        refused with 409 and a deterministic ordered blocker list; a
        safe one is removed ATOMICALLY (one rename the registry scans
        never observe as partial) with a small deterministic result
        (files removed, bytes reclaimed). No automatic deletion, no
        keep-N, no age policy, no bulk mode — explicit requests only.
        Pure lineage metadata (checkpoint parents, run provenance) and
        the M59 history / M8 dashboard computed views protect nothing:
        after any deletion they recompute naturally over the surviving
        registry.</li>

      <li><b>M62 read-only retention overview</b> —
        <code>GET /models/&#123;id&#125;/checkpoints/retention</code>
        makes the M61 retention state visible BEFORE any deletion
        attempt: for every checkpoint in the authoritative listing
        (canonical (step, created_at) order) the persisted identity,
        the artifact-set size, the M3 integrity-verification outcome,
        <code>deletable</code> — EXACTLY what an immediate M61 DELETE
        would decide (the SAME verifier + the SAME ONE blocker
        analysis; never "deletable" here then a blocker there) — and
        the SAME ordered blockers the M61 409 reports, plus
        deterministic aggregates (total / deletable / protected
        counts; total / reclaimable checkpoint bytes — reclaimable
        counts only currently-deletable checkpoint artifact sets,
        never model weights or record storage). Visibility only: it
        deletes nothing, retains nothing automatically, runs no
        cleanup and persists no policy — a user inspects the overview
        and then explicitly invokes M61 deletion. Computed live on
        every call: zero storage, zero mutation, byte-identical over
        unchanged state.</li>

      <li><b>M63 read-only project storage overview</b> —
        <code>GET /project/storage</code> lifts storage visibility one
        level (checkpoint &rarr; model &rarr; project): total physical
        files/bytes under the storage root, a complete category
        partition (models / checkpoints / model_records / datasets /
        tokenizers / suite_runs / samples / sample_evaluations /
        policies / probe_suites / workflow_recipes / project /
        unclassified — every physical file in EXACTLY ONE category,
        never counted twice, nothing silently discarded) and one
        compact row per model (its own state + records + checkpoint
        bytes; the retention aggregates are VERBATIM that model's M62
        overview — one analysis, never a second). Project totals are
        deterministic sums over the rows; the physical filesystem is
        the independent byte oracle. <code>reclaimable</code> counts
        ONLY currently-deletable checkpoint artifact sets across all
        models — never weights, datasets, tokenizers or records. M63
        REPORTS storage; M61 PERFORMS deletion — no cleanup, no
        policies, no quotas, no garbage collection. Zero storage, zero
        mutation, byte-identical over unchanged state.</li>

      <li><b>M64 read-only dataset &amp; tokenizer usage overview</b> —
        <code>GET /datasets/&#123;id&#125;/usage</code> and
        <code>GET /tokenizers/&#123;id&#125;/usage</code> answer "what
        still references this dataset/tokenizer?" before any deletion
        decision: every persisted referencing record by category —
        training-run provenance (model manifests), workflow plans
        (train/evaluate/compare stage configs), M4 evaluations, M5
        comparisons, M10 suite-run probes, M15 samples + M16
        sample-quality measurements (tokenizers), the tokenizers
        trained on a dataset (the ONE reference category its deletion
        guard refuses on) and the tokenized-version derivations — each
        category in a canonical order with deterministic sorted ids,
        plus a <code>referenced</code> flag and total count. Computed
        live through the ONE existing cross-reference filters and
        registries (never a second scanner). Visibility only: no
        deletion guards are added or changed here.</li>

      <li><b>M65 explicit verified dataset &amp; tokenizer retention</b> —
        <code>DELETE /datasets/&#123;id&#125;</code> and
        <code>DELETE /tokenizers/&#123;id&#125;</code> become explicit,
        verified and reference-safe (the M61 pattern one family over):
        the guard refuses with 409 + an ORDERED blocker list while ANY
        M64-visible reference exists — the SAME canonical categories
        and reference ids the usage overview reports (nothing
        protected that is not shown, nothing shown that is not
        protected) — and a successful deletion removes the artifact's
        OWN directory ATOMICALLY (one rename to a hidden sibling, then
        rmtree — no partial artifact can ever be observed), returning
        the deterministic files/bytes result. The tokenizer family's
        pre-M65 gap (an unguarded rmtree) is closed; the dataset
        family's single-category guard becomes the full ordered list.
        The M61 guard ORDER is preserved exactly — scope, INTEGRITY
        VERIFICATION (the existing M2 dataset verifier; a new tokenizer
        content-hash verifier), then references, then ONE atomic
        removal — so a corrupt artifact is never deletable (409) and
        an unparseable manifest is registry-invisible (404, nothing
        deleted). Read-only retention views
        (<code>GET /datasets/&#123;id&#125;/retention</code>,
        <code>GET /tokenizers/&#123;id&#125;/retention</code>) expose
        the same decision: ordered artifact files, total bytes,
        integrity outcome, <code>deletable</code> and the ordered
        blockers. No cascade, no force, no bulk, no policies.</li>

      <li><b>M66 read-only model usage overview</b> —
        <code>GET /models/&#123;id&#125;/usage</code> answers "what
        persisted artifacts currently reference this model?" BEFORE any
        future model-retention decision: every referencing record by
        category — the internal model-scoped families (training runs,
        checkpoints, workflows, evaluations, comparisons, gate
        decisions) and the EXTERNAL root-level families that persist
        the model id outside the model directory (suite runs, samples,
        sample-quality measurements, and workflow recipes whose stage
        configs name the model — deleting the model today would orphan
        exactly these). Deterministic category order + sorted record
        ids + internal/external splits
        (<code>externally_referenced</code> is the future-guard
        surface). Computed live through the ONE authoritative listings
        (never a second scanner). Visibility only — no model deletion
        or guard exists or changes here.</li>

      <li><b>M53 best state references</b> — a workflow stage state
        (<code>StageStateRef</code>: suite-run states, comparison sides,
        gate candidates) may now declare
        <code>state_kind: "best"</code>: at execution or M51-preflight
        time the workflow engine's ONE resolver invokes the SAME M52
        selection (minimum persisted
        <code>validation_loss</code>) and pins the CONCRETE checkpoint
        id (<code>resolved_checkpoint_id</code>) into the resolved plan
        — which is what executes, persists and hashes, so every
        immutable run record shows exactly
        <code>best → concrete id</code> and history never silently
        changes meaning when a later checkpoint becomes best. The
        recipe definition stays declarative (never rewritten, no
        mutable pointer); downstream engines receive a NORMAL literal
        checkpoint state; direct comparison/gate/suite-run requests
        still require current/checkpoint (best is a workflow-stage
        reference). No new endpoint — the existing inline
        <code>POST /workflows/run</code>, recipe runs and the M51
        <code>GET .../recipes/&#123;recipe&#125;/plan</code> preflight all
        resolve through the same path.</li>

      <li><b>M33 sample-quality history by tokenizer</b> — one read-only
        access path, <code>GET /models/&#123;id&#125;/sample-quality/by-tokenizer/&#123;tokenizer&#125;</code>,
        answers "which immutable M16 sample-quality measurements of
        this model measured samples generated with this
        tokenizer?": the model's authoritative M16 listing filtered by
        the persisted measurement tokenizer identity (every
        SampleEvaluationRecord carries a required non-nullable
        top-level tokenizer_id — the measured sample's recorded
        state), each matching record EXACTLY ONCE, the persisted id
        matched VERBATIM, after the tokenizer is validated through the
        existing GLOBAL registry (model scoping from the model's own
        listing; full record payloads incl. loss/perplexity, exact M16
        ordering; unknown model/tokenizer -> 404; a tokenizer with no
        measurements for the model -> []). Pure data access — no
        aggregates, no new scoring, zero storage growth.</li>
    </ul>
  </div>
  <div class="card"><b>REST API</b> (interactive docs at <code>/docs</code>)
    <ul>
      <li><code>GET   {prefix}/project</code> — project info &amp; artifact counts</li>
      <li><code>GET   {prefix}/project/storage</code> — read-only PHYSICAL storage overview of the whole project: totals, category partition (no double counting) and per-model rows with the M62 retention aggregates (M63)</li>
      <li><code>GET   {prefix}/datasets/&#123;id&#125;/usage</code> — read-only usage overview of one dataset: every referencing record by category (training runs, workflows, evaluations, comparisons, suite runs, tokenizer training, tokenized versions; M64)</li>
      <li><code>GET   {prefix}/tokenizers/&#123;id&#125;/usage</code> — read-only usage overview of one tokenizer: every referencing record by category (training runs, workflows, evaluations, comparisons, suite runs, samples, sample quality, tokenized datasets; M64)</li>
      <li><code>GET   {prefix}/datasets/&#123;id&#125;/retention</code> — read-only deletion-readiness view of one dataset: artifact files/bytes, integrity outcome, deletable + ordered blockers (M65)</li>
      <li><code>DELETE {prefix}/datasets/&#123;id&#125;</code> — explicit VERIFIED dataset retention: one dataset, only with zero M64-visible references (ordered blockers on 409), atomic removal, deterministic files/bytes result (M65)</li>
      <li><code>GET   {prefix}/tokenizers/&#123;id&#125;/retention</code> — read-only deletion-readiness view of one tokenizer: artifact files/bytes, integrity outcome, deletable + ordered blockers (M65)</li>
      <li><code>DELETE {prefix}/tokenizers/&#123;id&#125;</code> — explicit VERIFIED tokenizer retention: one tokenizer, only with zero M64-visible references (ordered blockers on 409), atomic removal, deterministic files/bytes result (M65)</li>
      <li><code>POST  {prefix}/models</code> — create a model (validated config)</li>
      <li><code>POST  {prefix}/datasets/upload</code> — ingest txt/md/csv/json → versioned dataset</li>
      <li><code>GET   {prefix}/datasets/&#123;id&#125;/verify</code> — integrity check (tamper detection)</li>
      <li><code>POST  {prefix}/datasets/&#123;id&#125;/tokenize</code> — tokenize with a stored tokenizer</li>
      <li><code>POST  {prefix}/tokenizers/train</code> — deterministic byte-level BPE</li>
      <li><code>POST  {prefix}/training/run</code> — train (CPT/SFT), synchronous; optional <code>resume_from_checkpoint_id</code> initializes the run from an immutable checkpoint without publishing it (M54)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/usage</code> — read-only model usage overview: every referencing record by category (internal families + external root-level suite runs / samples / sample quality / recipes; M66)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints</code> — immutable checkpoint store</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints/by-run/&#123;run&#125;</code> — checkpoints of one training run (provenance-validated, M46)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints/best</code> — deterministic selection by MINIMUM persisted validation loss (read-only, criterion explicit, M52)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints/best/history</code> — chronological history of the best-checkpoint selection movements, final entry always the M52 answer (read-only, computed live, zero storage, M59)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints/retention</code> — read-only retention overview: per-checkpoint deletability + ordered blockers (the exact M61 analysis) and reclaimable totals (live-computed, zero storage, M62)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints/&#123;ckpt&#125;</code> — one checkpoint</li>
      <li><code>DELETE {prefix}/models/&#123;id&#125;/checkpoints/&#123;ckpt&#125;</code> — explicit VERIFIED checkpoint retention: one checkpoint, only with zero blocking authoritative references (M52 best / published / parents / provenance / workflow+evaluation+comparison+gate+suite+sample references), atomic removal, ordered blockers on 409 (M61)</li>
      <li><code>POST  {prefix}/models/&#123;id&#125;/rollback</code> — verified restore of a checkpoint</li>
      <li><code>POST  {prefix}/evaluations/run</code> — read-only evaluation (current state or checkpoint)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations</code> — immutable evaluation history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-checkpoint/&#123;ckpt&#125;</code> — evaluations recorded under ONE checkpoint (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-dataset/&#123;ds&#125;</code> — evaluations of ONE model over ONE dataset (read-only, deterministic, versions verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-tokenizer/&#123;tok&#125;</code> — evaluations of ONE model with ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-split/&#123;split&#125;</code> — evaluations of ONE model on ONE dataset split (read-only, deterministic; unsupported split 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-state-kind/&#123;state_kind&#125;</code> — evaluations of ONE model over ONE kind of model state (read-only, deterministic, persisted state_kind verbatim; unsupported state kind 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-truncated/&#123;truncated&#125;</code> — evaluations of ONE model by persisted truncation status (read-only, deterministic, boolean verbatim, no quality judgment; non-boolean 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-seed/&#123;seed&#125;</code> — evaluations of ONE model by persisted effective seed (read-only, deterministic, persisted integer verbatim, open value axis; unmatched -> [], non-integer 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/&#123;eval&#125;</code> — one evaluation record</li>
      <li><code>POST  {prefix}/comparisons/run</code> — A/B comparison (improved / regressed / unchanged)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons</code> — immutable comparison history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-checkpoint/&#123;ckpt&#125;</code> — comparisons involving ONE checkpoint on either side (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-dataset/&#123;ds&#125;</code> — comparisons of ONE model over ONE dataset (read-only, deterministic, versions verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-tokenizer/&#123;tok&#125;</code> — comparisons of ONE model with ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-split/&#123;split&#125;</code> — comparisons of ONE model on ONE dataset split (read-only, deterministic, persisted shared-probe split verbatim; unsupported split 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-verdict/&#123;verdict&#125;</code> — comparisons of ONE model with ONE verdict (read-only, deterministic, persisted verdict verbatim — never recalculated; unsupported verdict 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-state-kind/&#123;state_kind&#125;</code> — comparisons of ONE model involving ONE kind of model state on EITHER side (read-only, deterministic, persisted side state kinds verbatim — never inferred; both-sides match appears once; unsupported state kind 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-seed/&#123;seed&#125;</code> — comparisons of ONE model by persisted effective seed (read-only, deterministic, persisted integer verbatim, open value axis; unmatched -> [], non-integer 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/samples/by-tokenizer/&#123;tok&#125;</code> — samples of ONE model generated with ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/samples/by-strategy/&#123;strategy&#125;</code> — samples of ONE model generated with ONE decoding strategy (read-only, deterministic, persisted strategy verbatim — never recalculated; unsupported strategy 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/sample-quality/by-tokenizer/&#123;tok&#125;</code> — M16 sample-quality measurements of ONE model under ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/&#123;comp&#125;</code> — one comparison record</li>
      <li><code>POST  {prefix}/gates/evaluate</code> — stage gate: policy + candidate → passed / failed decision</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions</code> — immutable gate decision history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/by-policy/&#123;policy_id&#125;</code> — gate decisions of ONE registered policy (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/by-comparison/&#123;comparison_id&#125;</code> — gate decisions that judged ONE M5 comparison (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/by-decision/&#123;decision&#125;</code> — gate decisions of ONE model with ONE decision result (read-only, deterministic, persisted decision verbatim — never recalculated; unsupported decision 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/by-verdict/&#123;verdict&#125;</code> — gate decisions of ONE model with ONE recorded comparison verdict (read-only, deterministic, persisted verdict verbatim — never recalculated; null-verdict threshold-only decisions in no group; unsupported verdict 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/by-baseline-type/&#123;baseline_type&#125;</code> — gate decisions of ONE model whose embedded policy used ONE baseline kind (read-only, deterministic, persisted nested policy.baseline_type verbatim — never derived; true disjoint partition; unsupported baseline type 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/&#123;decision&#125;</code> — one gate decision</li>
      <li><code>POST  {prefix}/workflows/run</code> — execute one inline workflow plan synchronously</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/workflows</code> — immutable workflow run history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/workflows/by-recipe/&#123;recipe_id&#125;</code> — workflow runs of ONE model executed from ONE registered recipe (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/workflows/by-status/&#123;status&#125;</code> — workflow runs of ONE model with ONE terminal status (read-only, deterministic, persisted status verbatim — never recalculated; unsupported status 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/workflows/&#123;run&#125;</code> — one workflow run</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/dashboard</code> — read-only history dashboard incl. sample-quality observability (deterministic, no writes)</li>
      <li><code>POST  {prefix}/policies</code> — register an immutable policy definition (idempotent; conflicts 409)</li>
      <li><code>GET   {prefix}/policies</code> / <code>GET {prefix}/policies/&#123;policy_id&#125;</code> — immutable policy definitions</li>
      <li><code>POST  {prefix}/probe-suites</code> — register an immutable named probe suite (set of exact M4 probes)</li>
      <li><code>GET   {prefix}/probe-suites</code> / <code>GET {prefix}/probe-suites/&#123;suite_id&#125;</code> — immutable suites</li>
      <li><code>POST  {prefix}/suite-runs</code> — execute one named suite against one state (independent M4 evaluations, immutable run record)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/suite-runs</code> / <code>GET {prefix}/models/&#123;id&#125;/suite-runs/&#123;run&#125;</code> — immutable suite-run history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/suite-runs/by-suite/&#123;suite_id&#125;</code> — suite-run records of ONE named suite (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/suite-runs/by-suite/&#123;suite_id&#125;/summary</code> — bookkeeping summary of ONE suite's runs (count, ordered ids, earliest/latest; read-only)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/suite-runs/by-checkpoint/&#123;ckpt&#125;</code> — suite runs executed against ONE checkpoint state (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/suite-runs/by-reused/&#123;reused_count&#125;</code> — suite runs of ONE model by persisted reuse count (read-only, deterministic, bookkeeping never a score; unmatched -> [], non-integer 422)</li>
      <li><code>POST  {prefix}/workflows/recipes</code> — register an immutable workflow recipe (idempotent; conflicts 409)</li>
      <li><code>GET   {prefix}/workflows/recipes</code> / <code>GET {prefix}/workflows/recipes/&#123;recipe_id&#125;</code> — immutable recipes</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/workflows/recipes/&#123;recipe&#125;/plan</code> — resolve ONE registered recipe against ONE model WITHOUT executing (read-only preflight: expanded model-bound plan + provenance, zero persistence; M51)</li>
      <li><code>POST  {prefix}/workflows/recipes/&#123;recipe_id&#125;/runs</code> — execute a recipe against one explicit model (existing M7 engine)</li>
      <li><code>GET   {prefix}/workflows/recipes/&#123;recipe_id&#125;/runs</code> — cross-model run lineage of one recipe (read-only; 404 unknown recipe)</li>
      <li><code>POST  {prefix}/samples/generate</code> — deterministic generation from one explicit verified checkpoint (greedy / seeded temperature; inference only)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/samples</code> / <code>GET {prefix}/models/&#123;id&#125;/samples/&#123;sample&#125;</code> — immutable sample history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/samples/by-checkpoint/&#123;ckpt&#125;</code> — samples generated from ONE checkpoint (read-only, deterministic, persisted identity authoritative)</li>
      <li><code>POST  {prefix}/models/&#123;id&#125;/samples/&#123;sample&#125;/quality</code> — per-sample likelihood measurement (sample-driven state resolution; causal-LM loss over generated targets; no body config)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/sample-quality</code> / <code>GET {prefix}/models/&#123;id&#125;/sample-quality/&#123;eval&#125;</code> — immutable sample-evaluation history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/sample-quality/records</code> — full immutable M16 sample-evaluation records (loss/perplexity included; read-only, deterministic, no statistics)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/sample-quality/by-sample/&#123;sample&#125;</code> — M16 measurements of ONE sample (read-only, deterministic, no statistics)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/sample-quality/by-checkpoint/&#123;checkpoint&#125;</code> — M16 measurements recorded under ONE checkpoint (read-only, deterministic, no statistics)</li>
      <li><code>GET   {prefix}/system</code> — hardware &amp; storage snapshot</li>
    </ul>
  </div>
  <script>
    fetch('{prefix}/project').then(r=>r.json()).then(d=>{{
      document.getElementById('status').innerHTML =
        '<span class="ok">● storage ready</span> — root <code>'+d.storage_root+
        '</code> · '+d.model_count+' model(s) · '+d.dataset_count+' dataset(s) · '+
        d.tokenizer_count+' tokenizer(s)';
    }}).catch(()=>{{document.getElementById('status').textContent='API unreachable';}});
  </script>
</main></body></html>""")


@api.get("/project", response_model=ProjectInfo, tags=["meta"])
def project_info() -> dict[str, Any]:
    return _forge().project_info()


@api.get("/project/storage",
         response_model=ProjectStorageOverview, tags=["meta"])
def project_storage_overview() -> ProjectStorageOverview:
    """Read-only PHYSICAL storage overview of the whole project (M63):
    total files/bytes, a complete category partition (models /
    checkpoints / model_records / datasets / tokenizers / suite_runs /
    samples / sample_evaluations / policies / probe_suites /
    workflow_recipes / project / unclassified — every physical file in
    EXACTLY ONE category, so the categories sum to the totals; nothing
    is silently discarded) and one compact row per model (its own
    state + records + checkpoint bytes, with the retention aggregates
    VERBATIM from that model's M62 overview — one retention analysis,
    never a second). ``reclaimable_checkpoint_bytes`` sums ONLY the
    currently-deletable checkpoint artifact sets across all models —
    never model weights, tokenizer, dataset or record storage. M63
    REPORTS storage; M61 PERFORMS deletion — this view deletes
    nothing, retains nothing automatically, runs no cleanup and
    persists no policy or quota. Computed live from one filesystem
    walk: zero storage, zero mutation, byte-identical over unchanged
    state; an empty project reports zeroed totals with an empty model
    collection."""
    return _forge().project_storage_overview()


@api.get("/system", response_model=dict, tags=["meta"])
def system_info() -> dict[str, Any]:
    """Hardware + storage snapshot (drives later training auto-configuration)."""
    return {"hardware": _forge().hardware(), "storage": _forge().storage_usage()}


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


@api.post(
    "/models",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    tags=["models"],
)
def create_model(request: ModelCreateRequest) -> dict[str, Any]:
    """Build + persist a new configurable transformer model (fresh weights)."""
    try:
        record, extra = _forge().create_model(request)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"model": record.model_dump(mode="json"), **extra}


@api.post("/models/validate", response_model=ValidationReport, tags=["models"])
def validate_config(payload: dict) -> ValidationReport:
    """Validate a model configuration payload without creating anything."""
    try:
        TransformerConfig(**payload)
        return ValidationReport(valid=True)
    except ValidationError as exc:
        return ValidationReport(
            valid=False,
            errors=[f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()],
        )


@api.get("/models", response_model=list[dict], tags=["models"])
def list_models() -> list[dict[str, Any]]:
    return [r.model_dump(mode="json") for r in _forge().list_models()]


@api.get("/models/{model_id}", response_model=ModelRecord, tags=["models"])
def get_model(model_id: str) -> ModelRecord:
    try:
        return _forge().get_model(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found") from exc


@api.get("/models/{model_id}/usage",
         response_model=ModelUsageOverview, tags=["models"])
def model_usage(model_id: str) -> ModelUsageOverview:
    """Read-only usage overview of ONE model (M66): every persisted
    record that references it, by category — the internal model-scoped
    families (training runs, checkpoints, workflows, evaluations,
    comparisons, gate decisions) plus the EXTERNAL root-level families
    persisting the model id outside the model directory (suite runs,
    samples, sample-quality measurements, workflow recipes whose stage
    configs name the model — exactly what a future model-retention
    guard would have to refuse on). Per-category sorted record ids,
    deterministic counts and internal/external splits. Computed live:
    zero storage, zero mutation, byte-identical over unchanged state.
    Visibility only — no deletion semantics exist or change here."""
    try:
        return _forge().model_usage_overview(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/verify", response_model=dict, tags=["models"])
def verify_model(model_id: str) -> dict[str, Any]:
    try:
        return _forge().verify_model(model_id)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/weights", tags=["models"])
def download_weights(model_id: str) -> Response:
    """Download the current weight file (torch state dict, sha256 sidecar)."""
    forge = _forge()
    try:
        record = forge.get_model(model_id)
        path = forge.weights_archive_path(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"weights for '{model_id}' are missing")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{record.name}-{model_id}-weights.pt",
    )


@api.delete("/models/{model_id}", response_model=dict, tags=["models"])
def delete_model(model_id: str) -> dict[str, Any]:
    if model_id not in _forge().storage.model_ids():
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    _forge().delete_model(model_id)
    return {"deleted": model_id}


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Read uploaded files into (name, bytes). Names are base names only."""
    from pathlib import PurePath

    result = []
    for f in files:
        data = await f.read()
        result.append((PurePath(f.filename or "upload.bin").name, data))
    return result


@api.post("/datasets/upload", response_model=dict, status_code=status.HTTP_201_CREATED, tags=["datasets"])
async def upload_dataset(
    files: list[UploadFile] = File(...),
    dataset_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Create a dataset (v1) or append a new version to an existing one.

    Mixed valid/invalid uploads return a per-file report; a dataset/version
    is only created when at least one file yields records.
    """
    try:
        return _forge().upload_dataset(await _read_uploads(files), dataset_id=dataset_id, name=name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise _conflict_or_422(exc) from exc


@api.get("/datasets", response_model=list[dict], tags=["datasets"])
def list_datasets() -> list[dict[str, Any]]:
    return _forge().list_datasets()


@api.get("/datasets/{dataset_id}", response_model=dict, tags=["datasets"])
def get_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        return _forge().get_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/datasets/{dataset_id}/verify", response_model=dict, tags=["datasets"])
def verify_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        return _forge().verify_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/datasets/{dataset_id}/usage",
         response_model=DatasetUsageOverview, tags=["datasets"])
def dataset_usage(dataset_id: str) -> DatasetUsageOverview:
    """Read-only usage overview of ONE dataset (M64): every persisted
    record that references it, by category — training-run provenance,
    workflow plans, evaluations, comparisons, suite-run probes, the
    tokenizers trained on it (the ONE reference category the deletion
    guard refuses on) and the tokenized versions derived from it.
    Computed live: zero storage, zero mutation, byte-identical over
    unchanged state. Visibility only — it deletes nothing and changes
    no deletion semantics."""
    try:
        return _forge().dataset_usage_overview(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:  # malformed manifest -> registry-invisible
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/datasets/{dataset_id}/retention",
         response_model=DatasetRetentionOverview, tags=["datasets"])
def dataset_retention(dataset_id: str) -> DatasetRetentionOverview:
    """Read-only retention overview of ONE dataset (M65): the
    deletion-readiness view — identity, ordered artifact files + total
    bytes, the M2 integrity-verification outcome, ``deletable`` and
    the ordered blockers (the SAME categories/ids the usage overview
    reports and the DELETE guard refuses on). A corrupt dataset is
    never deletable. Zero storage, zero mutation, byte-identical over
    unchanged state."""
    try:
        return _forge().dataset_retention_overview(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/datasets/{dataset_id}/tokenize", response_model=dict, tags=["datasets"])
def tokenize_dataset(dataset_id: str, payload: dict) -> dict[str, Any]:
    """Tokenize one version: JSON body {tokenizer_id: str, version?: int}."""
    tokenizer_id = payload.get("tokenizer_id")
    version = payload.get("version")
    if not isinstance(tokenizer_id, str) or not tokenizer_id:
        raise HTTPException(status_code=422, detail="body must contain 'tokenizer_id'")
    if version is not None and (not isinstance(version, int) or isinstance(version, bool)):
        raise HTTPException(status_code=422, detail="'version' must be an integer")
    try:
        return _forge().tokenize_dataset(dataset_id, tokenizer_id, version=version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@api.delete("/datasets/{dataset_id}",
            response_model=DatasetDeletionResult, tags=["datasets"],
            responses={409: {"model": DatasetDeletionBlocked,
                             "description": "referenced — ordered "
                             "blockers (the M64 usage categories)"}})
def delete_dataset(dataset_id: str) -> DatasetDeletionResult:
    """Explicit VERIFIED dataset retention (M65): delete exactly ONE
    dataset — only with live proof that nothing references it.

    The guard: scope through the M2 registry (unknown dataset -> 404,
    nothing deleted); the LIVE reference-safety analysis — training
    runs, workflows, evaluations, comparisons, suite runs, tokenizers
    trained on it, tokenized versions (ANY reference -> 409 with the
    ordered blocker list, the SAME categories and ids the M64
    ``GET /datasets/{id}/usage`` overview reports); then ONE atomic
    removal of the dataset's own directory. No cascade, no force, no
    bulk mode. The deterministic result reports the removed files and
    reclaimed bytes; every other family's records stay untouched."""
    try:
        return _forge().delete_dataset(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:  # integrity failure -> refuse (M61: 409)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        # protection rejection: re-derive the ordered blocker list for a
        # structured 409 detail (the M61 pattern; the list is diagnostic)
        try:
            blockers = [b.model_dump() for b in
                        _forge().dataset_deletion_blockers(dataset_id)]
        except Exception:
            blockers = []
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "dataset_id": dataset_id,
                    "protected": True, "blockers": blockers}) from exc


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #


@api.post("/tokenizers/train", response_model=dict, status_code=status.HTTP_201_CREATED, tags=["tokenizers"])
async def train_tokenizer(
    config: str = Form(...),
    files: list[UploadFile] = File(None),
    dataset_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Train a deterministic byte-level BPE tokenizer.

    ``config`` is a JSON string: {name, vocab_size, special_tokens?, ...}.
    Training text comes from uploaded files or from a stored dataset
    (``dataset_id``) — exactly one of the two.
    """
    try:
        cfg = TokenizerConfig(**json.loads(config))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid tokenizer config: {exc}") from exc
    try:
        uploads = await _read_uploads(files) if files else None
        record = _forge().train_tokenizer(cfg, files=uploads, dataset_id=dataset_id)
    except ValueError as exc:
        raise _conflict_or_422(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"tokenizer": record.model_dump(mode="json")}


@api.get("/tokenizers", response_model=list[dict], tags=["tokenizers"])
def list_tokenizers() -> list[dict[str, Any]]:
    return _forge().list_tokenizers()


@api.get("/tokenizers/{tokenizer_id}", response_model=dict, tags=["tokenizers"])
def get_tokenizer(tokenizer_id: str) -> dict[str, Any]:
    try:
        return _forge().get_tokenizer(tokenizer_id).model_dump(mode="json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/tokenizers/{tokenizer_id}/usage",
         response_model=TokenizerUsageOverview, tags=["tokenizers"])
def tokenizer_usage(tokenizer_id: str) -> TokenizerUsageOverview:
    """Read-only usage overview of ONE tokenizer (M64): every persisted
    record that references it, by category — training-run provenance,
    workflow plans, evaluations, comparisons, suite-run probes, samples,
    sample-quality measurements and the dataset versions it tokenized.
    Computed live: zero storage, zero mutation, byte-identical over
    unchanged state. Visibility only — it deletes nothing and changes
    no deletion semantics."""
    try:
        return _forge().tokenizer_usage_overview(tokenizer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:  # malformed manifest -> registry-invisible
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/tokenizers/{tokenizer_id}/retention",
         response_model=TokenizerRetentionOverview, tags=["tokenizers"])
def tokenizer_retention(tokenizer_id: str) -> TokenizerRetentionOverview:
    """Read-only retention overview of ONE tokenizer (M65): the
    deletion-readiness view — identity, ordered artifact files + total
    bytes, the content-hash integrity-verification outcome,
    ``deletable`` and the ordered blockers (the SAME categories/ids
    the usage overview reports and the DELETE guard refuses on). A
    corrupt tokenizer is never deletable. Zero storage, zero mutation,
    byte-identical over unchanged state."""
    try:
        return _forge().tokenizer_retention_overview(tokenizer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.delete("/tokenizers/{tokenizer_id}",
            response_model=TokenizerDeletionResult, tags=["tokenizers"],
            responses={409: {"model": TokenizerDeletionBlocked,
                             "description": "referenced — ordered "
                             "blockers (the M64 usage categories)"}})
def delete_tokenizer(tokenizer_id: str) -> TokenizerDeletionResult:
    """Explicit VERIFIED tokenizer retention (M65): delete exactly ONE
    tokenizer — only with live proof that nothing references it.

    The guard: scope through the registry (unknown tokenizer -> 404,
    nothing deleted); the LIVE reference-safety analysis — training
    runs, workflows, evaluations, comparisons, suite runs, samples,
    sample-quality measurements, tokenized dataset versions (ANY
    reference -> 409 with the ordered blocker list, the SAME
    categories and ids the M64 ``GET /tokenizers/{id}/usage``
    overview reports); then ONE atomic removal of the tokenizer's own
    directory. No cascade, no force, no bulk mode."""
    try:
        return _forge().delete_tokenizer(tokenizer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:  # integrity failure -> refuse (M61: 409)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        try:
            blockers = [b.model_dump() for b in
                        _forge().tokenizer_deletion_blockers(tokenizer_id)]
        except Exception:
            blockers = []
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "tokenizer_id": tokenizer_id,
                    "protected": True, "blockers": blockers}) from exc


# --------------------------------------------------------------------------- #
# Training
# Milestone 46 adds the read-only by-run grouping of the checkpoint history
#   (validated against the model's own training_provenance)
# Milestone 52 adds the read-only best-checkpoint selection
#   (minimum persisted validation loss; declared before the generic detail
#    route so "best" can never be captured as a checkpoint id)
# Milestone 54 adds the explicit training resume point
#   (TrainingConfig.resume_from_checkpoint_id: non-destructive per-run
#    initialization from a verified immutable checkpoint; no new route)
# Milestone 55 adds the declarative best-resume for TRAIN stages
#   (resume_from_best: resolved by the SAME M53 resolver through the
#    SAME M52 selection into a concrete M54 resume id; no new route)
# --------------------------------------------------------------------------- #


def _engine_error(exc: Exception) -> HTTPException:
    """Map training/evaluation failures to HTTP codes: 404 missing, 409 integrity, 422 invalid."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, RuntimeError)):
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            return HTTPException(status_code=409, detail=detail)
        return HTTPException(status_code=422, detail=detail)
    return HTTPException(status_code=500, detail=str(exc))


@api.post("/training/run", response_model=TrainingReport, tags=["training"])
def training_run(config: TrainingConfig) -> TrainingReport:
    """Train the user's model (continued pretraining or SFT), synchronously.

    Executes inline — no background queue in this milestone. Returns the full
    training report (checkpoints, losses, decisions, rollback info).

    M54: ``resume_from_checkpoint_id`` (optional, default None = today's
    behavior) initializes THIS run from an immutable checkpoint of the same
    model WITHOUT publishing it first — no rollback, no ``latest_checkpoint``
    mutation to prepare the run; the run's provenance records the concrete
    resume checkpoint (``initial_checkpoint_id``) and its first checkpoint
    descends from it. Model-weight resume only (no optimizer/scheduler
    state). Unknown/foreign checkpoint -> 404; corrupted weights -> 409.
    M55: ``resume_from_best`` is a WORKFLOW training-stage declaration —
    this DIRECT route rejects it (422): the workflow engine's single
    resolver pins the M52 best selection into a concrete
    ``resume_from_checkpoint_id`` before the training engine runs; name
    the checkpoint explicitly here (GET /checkpoints/best answers it).
    """
    try:
        return _forge().run_training(config)
    except Exception as exc:  # preflight + runtime failures mapped to HTTP
        raise _engine_error(exc) from exc


@api.get("/models/{model_id}/checkpoints", response_model=list[dict], tags=["training"])
def list_checkpoints(model_id: str) -> list[dict[str, Any]]:
    try:
        _forge().get_model(model_id)  # 404 for unknown models
        return [c.model_dump(mode="json") for c in _forge().list_checkpoints(model_id)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/checkpoints/by-run/{run_id}",
         response_model=list[dict], tags=["training"])
def list_checkpoints_by_run(model_id: str, run_id: str) -> list[dict[str, Any]]:
    """Immutable checkpoints of ONE training run (M46; run-lineage
    grouping validated against the model's own provenance).

    Read-only per-run grouping: the run must be registered in the
    model's own manifest training_provenance — an unknown model, an
    unknown run or a run id that belongs to another model is 404 (run
    ids are validated against the persisted provenance registry,
    never inferred from checkpoint directories, steps, timestamps,
    losses or parent relationships). Returns the authoritative M3
    listing filtered VERBATIM by each checkpoint's own persisted
    run_id, in the exact (step, created_at) ASCENDING order; a
    registered run with zero checkpoints returns 200 []. No separate
    training-run registry is introduced — the model manifest IS the
    registry. Pure data access, never writes.
    """
    try:
        return [c.model_dump(mode="json")
                for c in _forge().list_checkpoints_for_run(model_id,
                                                           run_id)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/checkpoints/best",
         response_model=CheckpointSelection, tags=["training"])
def select_best_checkpoint(model_id: str) -> CheckpointSelection:
    """Deterministic best-checkpoint SELECTION under the persisted
    validation-loss criterion (M52, read-only).

    Answers: which of this model's checkpoints has the MINIMUM
    persisted validation_loss? Candidates come from the authoritative
    M3 checkpoint listing (the persisted manifest is the source);
    validation loss is read VERBATIM from it — never recomputed, never
    derived from perplexity, gate/checkpoint decisions, evaluations,
    comparisons, ids or timestamps — and non-finite persisted values
    are never candidates. Ties on the exact minimum resolve by the
    listing's canonical (step, created_at) ASCENDING order — the first
    checkpoint among equals — and are disclosed via the ``tied`` flag.
    Returns the complete verbatim CheckpointRecord plus the explicit
    criterion and candidate count. "Best" means exactly this criterion
    — NOT a claim of overall model quality (semantic quality,
    factuality, safety or generalization are NOT established). Pure
    computed view: never writes, never persists a selection pointer.
    Unknown model -> 404; a valid model with no selectable checkpoints
    -> 404 (nothing is manufactured). Declared BEFORE the generic
    {checkpoint_id} detail route so "best" can never be captured as a
    checkpoint id.
    """
    try:
        return _forge().select_best_checkpoint(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/checkpoints/best/history",
         response_model=BestCheckpointHistory, tags=["training"])
def best_checkpoint_history(model_id: str) -> BestCheckpointHistory:
    """Chronological history of the M52 best-checkpoint selection
    movements (M59, read-only computed view).

    Replays the ONE M52 winner rule (minimum persisted validation_loss,
    ties by the canonical (step, created_at) ASCENDING order — first
    among equals; non-finite values never candidates) over the
    authoritative M3 listing in chronological order: an entry appears
    exactly when a checkpoint DISPLACED the running best as the
    registry grew, so the sequence shows the model's real improvement
    trajectory (only winners — non-winning checkpoints are excluded).
    The FINAL entry is always the live M52 answer
    (GET /models/{id}/checkpoints/best); losses are monotonically
    non-increasing; ``delta_loss_nats`` (current - previous; negative
    means improvement) is None on the first entry. Computed live from
    persisted manifests on every call: never writes, never caches,
    never persists a pointer — zero storage. Unknown model -> 404; a
    valid model with no selectable checkpoints -> an EMPTY history
    (consistent with the collection endpoints). Declared before the
    generic {checkpoint_id} detail route so "best" can never be
    captured as a checkpoint id.
    """
    try:
        return _forge().best_checkpoint_history(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/checkpoints/retention",
         response_model=CheckpointRetentionOverview, tags=["training"])
def checkpoint_retention_overview(model_id: str) -> CheckpointRetentionOverview:
    """Read-only retention overview of ONE model's checkpoint registry
    (M62, live-computed view — visibility only).

    For every checkpoint in the authoritative M3 listing (canonical
    (step, created_at) order): the persisted identity (checkpoint id,
    run, step, created_at, validation_loss), the artifact-set size, the
    M3 integrity-verification outcome, ``deletable`` — EXACTLY what an
    immediate M61 DELETE would decide (the SAME verifier and the SAME
    ONE blocker analysis; a user never sees "deletable" here and then
    receives a blocker there) — and the SAME ordered blockers the M61
    409 reports. Plus deterministic aggregates: total / deletable /
    protected counts and total / reclaimable checkpoint bytes
    (reclaimable counts ONLY currently-deletable checkpoint artifact
    sets — never model weights, tokenizer, dataset or record storage).
    The overview deletes nothing, retains nothing automatically, runs
    no cleanup and persists no policy — a user inspects it and then
    explicitly invokes M61 deletion. Computed live on every call:
    zero storage, zero mutation, byte-identical over unchanged state.
    Unknown model -> 404; a valid model with no checkpoints -> an
    EMPTY overview with zeroed totals. Declared before the generic
    {checkpoint_id} detail route so "retention" can never be captured
    as a checkpoint id.
    """
    try:
        return _forge().checkpoint_retention_overview(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/checkpoints/{checkpoint_id}", response_model=dict,
         tags=["training"])
def get_checkpoint(model_id: str, checkpoint_id: str) -> dict[str, Any]:
    try:
        return _forge().get_checkpoint(model_id, checkpoint_id).model_dump(mode="json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.delete("/models/{model_id}/checkpoints/{checkpoint_id}",
            response_model=CheckpointDeletionResult, tags=["training"])
def delete_checkpoint(model_id: str, checkpoint_id: str) -> CheckpointDeletionResult:
    """Explicit VERIFIED checkpoint retention (M61): delete exactly ONE
    checkpoint — only after live proof that nothing authoritative
    depends on it.

    The guard runs in a fixed order: scope through the authoritative
    listing (unknown model/checkpoint, or an unreadable manifest the
    listing skips -> 404, nothing deleted); integrity verification
    through the existing M3 verifier (corrupt/unreadable weights ->
    409 — deletion never bypasses integrity validation); the live
    reference-safety analysis (the M52 best, the published
    ``latest_checkpoint``, the manifest's stored best-known reference,
    and every persisted checkpoint id in workflow/evaluation/
    comparison/gate/suite-run/sample/sample-quality records -> 409
    with the ordered blocker list); then ONE atomic removal (never a
    partial checkpoint). No automatic deletion, no keep-N, no age
    policy, no bulk mode. The deterministic result reports the removed
    files and reclaimed bytes; every surviving state and record stays
    untouched.
    """
    try:
        return _forge().delete_checkpoint(model_id, checkpoint_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        # protection rejection: re-derive the ordered blocker list for a
        # structured 409 detail (the engine re-checked safety internally,
        # so the message alone is authoritative; the list is diagnostic)
        try:
            blockers = [b.model_dump() for b in
                        _forge().checkpoint_blockers(model_id, checkpoint_id)]
        except Exception:
            blockers = []
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "model_id": model_id,
                    "checkpoint_id": checkpoint_id, "protected": True,
                    "blockers": blockers}) from exc


@api.post("/models/{model_id}/rollback", response_model=ModelRecord, tags=["training"])
def rollback_model(model_id: str, request: RollbackRequest) -> ModelRecord:
    """Verify a checkpoint's integrity, then restore its weights as current."""
    try:
        return _forge().rollback_model(model_id, request.checkpoint_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc





# --------------------------------------------------------------------------- #
# Evaluation routes (Milestone 4 — read-only measurement;
# Milestone 28 adds the read-only by-dataset grouping of the history;
# Milestone 30 adds the read-only by-tokenizer grouping of the history;
# Milestone 36 adds the read-only by-split grouping of the same history;
# Milestone 38 adds the read-only by-state-kind grouping of the same history;
# Milestone 47 adds the read-only by-truncated grouping of the same history;
# Milestone 48 adds the read-only by-seed grouping of the same history)
# --------------------------------------------------------------------------- #

@api.post("/evaluations/run", response_model=EvaluationRecord, tags=["evaluation"])
def evaluation_run(config: EvaluationConfig) -> EvaluationRecord:
    """Measure an existing model state against one tokenized split (read-only).

    checkpoint_id = None -> the model's current published weights; otherwise a
    verified immutable checkpoint of that model. Nothing is modified: only one
    immutable evaluation record is created under models/<id>/evaluations/.
    M56: ``checkpoint_from_best`` is a WORKFLOW evaluation-stage declaration
    with no place on this DIRECT route (unknown field -> 422) — the workflow
    engine's single resolver pins the M52 best selection into a concrete
    ``checkpoint_id`` before the M4 engine runs; name the checkpoint
    explicitly here (GET /checkpoints/best answers it).
    """
    try:
        return _forge().run_evaluation(config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


@api.get("/models/{model_id}/evaluations", response_model=list[dict], tags=["evaluation"])
def list_evaluations(model_id: str) -> list[dict[str, Any]]:
    """Immutable evaluation history of a model (404 for unknown models)."""
    try:
        return [r.model_dump(mode="json")
                for r in _forge().list_evaluations(model_id)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/by-checkpoint/{checkpoint_id}",
         response_model=list[EvaluationRecord], tags=["evaluation"])
def list_evaluations_by_checkpoint(model_id: str, checkpoint_id: str
                                   ) -> list[EvaluationRecord]:
    """Immutable M4 evaluation records recorded under ONE checkpoint (M24).

    Read-only per-checkpoint grouping: validates the checkpoint through
    the model's M3 checkpoint registry (an unknown checkpoint, or a
    checkpoint id belonging to another model, is 404 — checkpoint ids
    are model-scoped; nothing is inferred from filenames), then returns
    the model's authoritative M4 listing filtered by the persisted
    checkpoint_id recorded in each EvaluationRecord — complete verbatim
    payloads (loss_nats/perplexity included) in the exact M4
    (created_at, eval_id) order. Current-state evaluations keep
    checkpoint_id null and never appear; a checkpoint with no
    evaluations returns []. Answers only which immutable evaluations
    measured this checkpoint; no derived statistics, no comparison, no
    writes. (Must stay registered before /evaluations/{eval_id}; the
    literal "by-checkpoint" segment is not an evaluation id.)"""
    try:
        return _forge().list_evaluations_for_checkpoint(model_id,
                                                        checkpoint_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/by-dataset/{dataset_id}",
         response_model=list[EvaluationRecord], tags=["evaluation"])
def list_evaluations_by_dataset(model_id: str, dataset_id: str
                                ) -> list[EvaluationRecord]:
    """Immutable M4 evaluations of ONE model over ONE dataset (M28).

    Read-only per-dataset grouping: validates the dataset through the
    M2 registry (unknown dataset -> 404; never a raw filesystem check),
    then returns the model's authoritative M4 listing filtered by the
    persisted dataset identity recorded in each EvaluationRecord
    (top-level dataset_id; membership never comes from filenames,
    tokenizer ids, eval ids, checkpoint ids, hashes or timestamps).
    The persisted dataset_version travels VERBATIM inside every
    returned record — all versions of the dataset are returned, each
    exactly as persisted (versions are neither collapsed, nor resolved
    to the latest, nor rewritten). Complete verbatim payloads
    (loss_nats/perplexity/state identity included) in the exact M4
    (created_at, eval_id) order; datasets are global, so model scoping
    comes from the model's own listing — a model never sees another
    model's evaluations. A valid dataset with no evaluations for the
    model returns []. No aggregates, no writes. (Must stay registered
    before /evaluations/{eval_id}; the literal "by-dataset" segment is
    not an evaluation id.)"""
    try:
        return _forge().list_evaluations_for_dataset(model_id, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/by-tokenizer/{tokenizer_id}",
         response_model=list[EvaluationRecord], tags=["evaluation"])
def list_evaluations_by_tokenizer(model_id: str, tokenizer_id: str
                                  ) -> list[EvaluationRecord]:
    """Immutable M4 evaluations of ONE model measured with ONE
    tokenizer (M30).

    Read-only per-tokenizer grouping: validates the tokenizer through
    the existing tokenizer registry (unknown tokenizer -> 404; never a
    raw filesystem check), then returns the model's authoritative M4
    listing filtered by the persisted tokenizer identity recorded in
    each EvaluationRecord (top-level tokenizer_id; membership never
    comes from filenames, eval ids, checkpoint/dataset identities or a
    latest-tokenizer substitution — the persisted id is matched
    VERBATIM and every other persisted field travels unchanged).
    Complete verbatim payloads (loss_nats/perplexity/state/dataset
    identity included) in the exact M4 (created_at, eval_id) order;
    tokenizers are global, so model scoping comes from the model's own
    listing — a model never sees another model's evaluations. A valid
    tokenizer with no evaluations for the model returns []. No
    aggregates, no writes. (Must stay registered before
    /evaluations/{eval_id}; the literal "by-tokenizer" segment is not
    an evaluation id.)"""
    try:
        return _forge().list_evaluations_for_tokenizer(model_id,
                                                       tokenizer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/by-split/{split}",
         response_model=list[EvaluationRecord], tags=["evaluation"])
def list_evaluations_by_split(model_id: str, split: EvaluationSplit
                              ) -> list[EvaluationRecord]:
    """Immutable M4 evaluations of ONE model measured on ONE dataset
    split (M36).

    Read-only per-split grouping: returns the model's authoritative
    M4 listing filtered by the persisted split recorded in each
    EvaluationRecord (top-level split, matched VERBATIM — membership
    never comes from filenames, directories, timestamps, dataset
    names, eval ids or hashes, and the persisted value is never
    resolved or rewritten). Complete verbatim payloads
    (loss_nats/perplexity/state/dataset identity included) in the
    exact M4 (created_at, eval_id) order; a valid split with no
    evaluations for the model returns []. Splits have NO registry
    (unlike the by-checkpoint/by-dataset/by-tokenizer axes): the
    EvaluationSplit enum IS the contract, so an UNSUPPORTED split
    value is rejected with 422 at the API boundary (schema-level
    validation — never a registry-style 404), while an unknown model
    is 404 exactly like the sibling groupings. No aggregates, no
    writes. (Must stay registered before /evaluations/{eval_id}; the
    literal "by-split" segment is not an evaluation id.)"""
    try:
        return _forge().list_evaluations_for_split(model_id, split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/by-state-kind/{state_kind}",
         response_model=list[EvaluationRecord], tags=["evaluation"])
def list_evaluations_by_state_kind(model_id: str,
                                   state_kind: EvalStateKind
                                   ) -> list[EvaluationRecord]:
    """Immutable M4 evaluations of ONE model measuring ONE kind of
    model state (M38).

    Read-only per-state-kind grouping: returns the model's
    authoritative M4 listing filtered by the persisted state_kind
    recorded in each EvaluationRecord (top-level state_kind, matched
    VERBATIM — membership never comes from filenames, directories,
    timestamps, eval ids or hashes, and NEVER from checkpoint_id
    nullability: that nullability is a schema consequence of the
    persisted state kind, not its source; the persisted value is
    never resolved or rewritten). Complete verbatim payloads
    (loss_nats/perplexity/state/dataset identity included) in the
    exact M4 (created_at, eval_id) order; a valid state kind with no
    evaluations for the model returns []. State kinds have NO
    registry (unlike the by-checkpoint/by-dataset/by-tokenizer
    axes): the EvalStateKind enum IS the contract, so an UNSUPPORTED
    state-kind value is rejected with 422 at the API boundary
    (schema-level validation — never a registry-style 404, and the
    validation fires BEFORE this handler even for an unknown model),
    while an unknown model with a VALID state kind raises
    FileNotFoundError -> 404 exactly like the sibling groupings. No
    aggregates, no writes. (Must stay registered before
    /evaluations/{eval_id}; the literal "by-state-kind" segment is
    not an evaluation id.)"""
    try:
        return _forge().list_evaluations_for_state_kind(model_id,
                                                        state_kind)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/by-truncated/{truncated}",
         response_model=list[EvaluationRecord], tags=["evaluation"])
def list_evaluations_by_truncated(model_id: str,
                                  truncated: bool
                                  ) -> list[EvaluationRecord]:
    """Immutable M4 evaluations of ONE model by persisted truncation
    status (M47).

    Read-only per-status grouping: returns the model's authoritative
    M4 listing filtered by the persisted REQUIRED boolean
    ``truncated`` recorded in each EvaluationRecord (True =
    ``max_eval_tokens`` stopped the evaluation before the split
    ended; False = the configured/permitted evaluation stream was
    consumed without the cap cutting it short) — matched VERBATIM,
    never recalculated, never derived from records_covered, token
    counts, split length, the evaluation configuration, timestamps,
    durations, state kinds or any other field. The boolean carries
    no quality judgment — it is engine metadata about how far the
    evaluation stream was consumed, nothing more. Complete verbatim
    payloads in the exact M4 (created_at, eval_id) order; a model
    with no evaluations of one status returns []. The boolean is a
    closed two-value contract (no registry): non-boolean spellings
    are rejected with 422 at the API boundary (schema-level
    validation — never a silent reinterpretation, and the validation
    fires BEFORE this handler even for an unknown model), while an
    unknown model with a VALID boolean raises FileNotFoundError ->
    404 exactly like the sibling groupings. No aggregates, no
    writes. (Must stay registered before /evaluations/{eval_id};
    the literal "by-truncated" segment is not an evaluation id.)"""
    try:
        return _forge().list_evaluations_for_truncated(model_id,
                                                       truncated)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/by-seed/{seed}",
         response_model=list[EvaluationRecord], tags=["evaluation"])
def list_evaluations_by_seed(model_id: str,
                             seed: int) -> list[EvaluationRecord]:
    """Immutable M4 evaluations of ONE model by persisted effective
    seed (M48).

    Read-only per-seed grouping: returns the model's authoritative
    M4 listing filtered by the REQUIRED integer ``seed`` persisted
    on each EvaluationRecord at run time (the effective seed used,
    default derived from the config) — matched VERBATIM, never
    recalculated, never normalized, never derived from the embedded
    config dict, request parameters, dataset identity, splits,
    tokenizers, state kinds, losses, ids, timestamps or any other
    field. The seed is bookkeeping identity, not a quality metric —
    no seed is better than another. Complete verbatim payloads in
    the exact M4 (created_at, eval_id) order. The seed is an OPEN
    integer value axis (no registry, no enum): any integer is
    type-valid — an unmatched seed on a valid model returns 200 []
    — while a non-integer spelling is rejected with 422 at the API
    boundary (schema-level validation — never a silent
    reinterpretation, and the validation fires BEFORE this handler
    even for an unknown model); an unknown model with a VALID
    integer raises FileNotFoundError -> 404 exactly like the sibling
    groupings. No aggregates, no writes. (Must stay registered
    before /evaluations/{eval_id}; the literal "by-seed" segment is
    not an evaluation id.)"""
    try:
        return _forge().list_evaluations_for_seed(model_id, seed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/evaluations/{eval_id}", response_model=EvaluationRecord,
         tags=["evaluation"])
def get_evaluation(model_id: str, eval_id: str) -> EvaluationRecord:
    """One persisted immutable evaluation record."""
    try:
        return _forge().get_evaluation(model_id, eval_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Comparison routes (Milestone 5 — evidence-based A/B state comparison;
# Milestone 26 adds the read-only by-checkpoint grouping of the immutable history;
# Milestone 29 adds the read-only by-dataset grouping of the immutable history;
# Milestone 31 adds the read-only by-tokenizer grouping of the immutable history;
# Milestone 37 adds the read-only by-split grouping of the same history;
# Milestone 39 adds the read-only by-verdict grouping of the same history;
# Milestone 44 adds the read-only by-state-kind grouping with either-side semantics;
# Milestone 49 adds the read-only by-seed grouping of the same history)
# --------------------------------------------------------------------------- #

@api.post("/comparisons/run", response_model=ComparisonRecord, tags=["comparison"])
def comparison_run(request: ComparisonRequest) -> ComparisonRecord:
    """Compare two states of a model under one identical evaluation probe.

    Both states are verified, then each is evaluated (reusing an exact
    immutable M4 evaluation when one exists). Only one immutable comparison
    manifest is created — nothing is trained or modified.
    """
    try:
        return _forge().run_comparison(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


@api.get("/models/{model_id}/comparisons", response_model=list[dict], tags=["comparison"])
def list_comparisons(model_id: str) -> list[dict[str, Any]]:
    """Immutable comparison history of a model (404 for unknown models)."""
    try:
        return [r.model_dump(mode="json")
                for r in _forge().list_comparisons(model_id)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/by-checkpoint/{checkpoint_id}",
         response_model=list[ComparisonRecord], tags=["comparison"])
def list_comparisons_by_checkpoint(model_id: str, checkpoint_id: str
                                   ) -> list[ComparisonRecord]:
    """Immutable M5 comparisons involving ONE checkpoint (M26).

    Read-only per-checkpoint grouping: validates the checkpoint through
    the model's M3 checkpoint registry (an unknown checkpoint, or a
    checkpoint id belonging to another model, is 404 — checkpoint ids
    are model-scoped; nothing is inferred from filenames), then returns
    the model's authoritative M5 listing filtered by the persisted side
    states — a comparison is included when EITHER side records
    state_kind "checkpoint" with the requested checkpoint_id
    (current-state sides keep checkpoint_id null and never match).
    Complete verbatim payloads (verdict/losses included) in the exact
    M5 (created_at, comparison_id) order; each comparison appears
    exactly once even when BOTH sides match (A = B = checkpoint — the
    response is records, not matching sides). A valid checkpoint with
    no matching comparisons returns []. No new metrics, no writes.
    (Must stay registered before /comparisons/{comparison_id}; the
    literal "by-checkpoint" segment is not a comparison id.)"""
    try:
        return _forge().list_comparisons_for_checkpoint(model_id,
                                                        checkpoint_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/by-dataset/{dataset_id}",
         response_model=list[ComparisonRecord], tags=["comparison"])
def list_comparisons_by_dataset(model_id: str, dataset_id: str
                                ) -> list[ComparisonRecord]:
    """Immutable M5 comparisons of ONE model over ONE dataset (M29).

    Read-only per-dataset grouping: validates the dataset through the
    M2 registry (unknown dataset -> 404; never a raw filesystem
    check), then returns the model's authoritative M5 listing filtered
    by the persisted shared-probe dataset identity recorded in each
    ComparisonRecord — a comparison is valid only when BOTH sides
    measure the SAME probe, so exactly ONE top-level dataset_id +
    dataset_version is persisted per record (M5 refuses cross-probe
    requests with 422; per-side dataset identities cannot occur by
    construction; membership never comes from filenames, checkpoint
    ids or hashes). The persisted dataset_version travels VERBATIM
    inside every returned record — all versions of the dataset are
    returned, each exactly as persisted (never collapsed, resolved or
    rewritten) — and each matching comparison appears EXACTLY ONCE
    (dedup by comparison identity, not side combinations). Complete
    verbatim payloads (verdict/per-side losses included) in the exact
    M5 (created_at, comparison_id) order; datasets are global, so
    model scoping comes from the model's own listing — a model never
    sees another model's comparisons. A valid dataset with no
    comparisons for the model returns []. No aggregates, no writes.
    (Must stay registered before /comparisons/{comparison_id}; the
    literal "by-dataset" segment is not a comparison id.)"""
    try:
        return _forge().list_comparisons_for_dataset(model_id, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/by-tokenizer/{tokenizer_id}",
         response_model=list[ComparisonRecord], tags=["comparison"])
def list_comparisons_by_tokenizer(model_id: str, tokenizer_id: str
                                  ) -> list[ComparisonRecord]:
    """Immutable M5 comparisons of ONE model measured with ONE
    tokenizer (M31).

    Read-only per-tokenizer grouping: validates the tokenizer through
    the existing registry (unknown tokenizer -> 404; never a raw
    filesystem check), then returns the model's authoritative M5
    listing filtered by the persisted shared-probe tokenizer identity
    recorded in each ComparisonRecord — a comparison persists exactly
    ONE top-level tokenizer_id (both sides measure the same probe by
    construction; M5 refuses cross-probe requests with 422; membership
    never comes from filenames, checkpoint ids, dataset identities or
    hashes; the persisted id is matched VERBATIM, never substituted
    with the latest tokenizer). Each matching comparison appears
    EXACTLY ONCE (dedup by comparison identity, not side
    combinations). Complete verbatim payloads (verdict/per-side losses
    included) in the exact M5 (created_at, comparison_id) order;
    tokenizers are global, so model scoping comes from the model's own
    listing — a model never sees another model's comparisons. A valid
    tokenizer with no comparisons for the model returns []. No
    aggregates, no writes. (Must stay registered before
    /comparisons/{comparison_id}; the literal "by-tokenizer" segment
    is not a comparison id.)"""
    try:
        return _forge().list_comparisons_for_tokenizer(model_id,
                                                       tokenizer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/by-split/{split}",
         response_model=list[ComparisonRecord], tags=["comparison"])
def list_comparisons_by_split(model_id: str, split: EvaluationSplit
                              ) -> list[ComparisonRecord]:
    """Immutable M5 comparisons of ONE model measured on ONE dataset
    split (M37).

    Read-only per-split grouping: returns the model's authoritative
    M5 listing filtered by the persisted shared-probe split recorded
    in each ComparisonRecord (top-level split — a comparison is valid
    only when BOTH sides measure the SAME dataset/version/split/
    tokenizer/window/seed probe, so the split is a property of the
    comparison itself; matched VERBATIM — membership never comes from
    filenames, checkpoint ids, dataset identities, nested evaluation
    records or hashes, and the persisted value is never resolved or
    rewritten). Complete verbatim payloads (verdict/per-side losses
    included) in the exact M5 (created_at, comparison_id) order; a
    valid split with no comparisons for the model returns []. Splits
    have NO registry (unlike the by-checkpoint/by-dataset/
    by-tokenizer axes): the EvaluationSplit enum IS the contract, so
    an UNSUPPORTED split value is rejected with 422 at the API
    boundary (schema-level validation — never a registry-style 404,
    and the validation fires BEFORE this handler even for an unknown
    model), while an unknown model with a VALID split raises
    FileNotFoundError -> 404 exactly like the sibling groupings. No
    aggregates, no writes. (Must stay registered before
    /comparisons/{comparison_id}; the literal "by-split" segment is
    not a comparison id.)"""
    try:
        return _forge().list_comparisons_for_split(model_id, split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/by-verdict/{verdict}",
         response_model=list[ComparisonRecord], tags=["comparison"])
def list_comparisons_by_verdict(model_id: str, verdict: ComparisonVerdict
                                ) -> list[ComparisonRecord]:
    """Immutable M5 comparisons of ONE model with ONE verdict (M39).

    Read-only per-verdict grouping: returns the model's authoritative
    M5 listing filtered by the persisted verdict recorded in each
    ComparisonRecord (top-level verdict — the immutable loss-only
    judgment persisted at run time by the M5 comparison flow;
    matched VERBATIM — membership NEVER comes from recalculating
    loss deltas, per-side losses, tolerances, checkpoint ids or
    hashes, no comparison is executed and no evaluation rerun; the
    persisted value is never resolved or rewritten). Complete
    verbatim payloads (verdict/per-side losses included) in the exact
    M5 (created_at, comparison_id) order; a valid verdict with no
    comparisons for the model returns []. Verdicts have NO registry
    (unlike the by-checkpoint/by-dataset/by-tokenizer axes): the
    ComparisonVerdict enum IS the contract, so an UNSUPPORTED verdict
    value is rejected with 422 at the API boundary (schema-level
    validation — never a registry-style 404, and the validation
    fires BEFORE this handler even for an unknown model), while an
    unknown model with a VALID verdict raises FileNotFoundError ->
    404 exactly like the sibling groupings. No aggregates, no
    rankings, no writes. (Must stay registered before
    /comparisons/{comparison_id}; the literal "by-verdict" segment is
    not a comparison id.)"""
    try:
        return _forge().list_comparisons_for_verdict(model_id, verdict)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/by-state-kind/{state_kind}",
         response_model=list[ComparisonRecord], tags=["comparison"])
def list_comparisons_by_state_kind(model_id: str,
                                   state_kind: EvalStateKind
                                   ) -> list[ComparisonRecord]:
    """Immutable comparisons of ONE model involving ONE kind of model
    state on EITHER side (M44).

    Read-only per-state-kind grouping with M26's either-side
    semantics: returns the model's authoritative M5 listing filtered
    by the persisted state kind of the comparison sides — a
    comparison belongs to the request when EITHER persisted side
    (state_a / state_b) records the requested state_kind (the
    schema enum current/checkpoint; matched VERBATIM — membership
    NEVER comes from inferring the kind from checkpoint ids, state
    hashes, losses or evaluation results, nothing is recalculated or
    re-executed; the persisted values are never resolved or
    rewritten). Because the listing holds each record exactly once, a
    comparison matching on BOTH sides appears EXACTLY ONCE. Complete
    verbatim payloads (both sides, losses, verdict verbatim) in the
    exact M5 (created_at, comparison_id) order; a valid state kind
    with no matching comparisons returns []. State kinds have NO
    registry (exactly like M38): the EvalStateKind enum IS the
    contract, so an UNSUPPORTED state-kind value is rejected with 422
    at the API boundary (schema-level validation — never a
    registry-style 404, and the validation fires BEFORE this handler
    even for an unknown model), while an unknown model with a VALID
    state kind raises FileNotFoundError -> 404 exactly like the
    sibling groupings. No aggregates, no rankings, no writes. (Must
    stay registered before /comparisons/{comparison_id}; the literal
    "by-state-kind" segment is not a comparison id.)"""
    try:
        return _forge().list_comparisons_for_state_kind(model_id,
                                                        state_kind)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/by-seed/{seed}",
         response_model=list[ComparisonRecord], tags=["comparison"])
def list_comparisons_by_seed(model_id: str,
                             seed: int) -> list[ComparisonRecord]:
    """Immutable M5 comparisons of ONE model by persisted effective
    seed (M49).

    Read-only per-seed grouping: returns the model's authoritative
    M5 listing filtered by the REQUIRED integer ``seed`` persisted
    on each ComparisonRecord at run time (the seed of the
    identical-probe A/B measurement) — matched VERBATIM, never
    recalculated, never normalized, never derived from the
    comparison configuration, either side's evaluation, state
    payloads, verdicts, loss deltas, ids, timestamps or any other
    field. The seed is bookkeeping identity, not a quality metric —
    no seed produces better comparisons. Complete verbatim payloads
    (both sides, losses, verdict included) in the exact M5
    (created_at, comparison_id) order. The seed is an OPEN integer
    value axis (no registry, no enum): any integer is type-valid —
    an unmatched seed on a valid model returns 200 [] — while a
    non-integer spelling is rejected with 422 at the API boundary
    (schema-level validation — never a silent reinterpretation, and
    the validation fires BEFORE this handler even for an unknown
    model); an unknown model with a VALID integer raises
    FileNotFoundError -> 404 exactly like the sibling groupings. No
    aggregates, no writes. (Must stay registered before
    /comparisons/{comparison_id}; the literal "by-seed" segment is
    not a comparison id.)"""
    try:
        return _forge().list_comparisons_for_seed(model_id, seed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/comparisons/{comparison_id}", response_model=ComparisonRecord,
         tags=["comparison"])
def get_comparison(model_id: str, comparison_id: str) -> ComparisonRecord:
    """One persisted immutable comparison record."""
    try:
        return _forge().get_comparison(model_id, comparison_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# Gate routes (Milestone 6 — policy-driven, evidence-based run decisions;
# Milestone 23 adds the read-only by-policy grouping of the immutable history;
# Milestone 34 adds the read-only by-comparison grouping of the same history;
# Milestone 41 adds the read-only by-decision grouping of the same history;
# Milestone 43 adds the read-only by-verdict grouping of the same history;
# Milestone 45 adds the read-only by-baseline-type grouping (nested policy field))
# --------------------------------------------------------------------------- #

@api.post("/gates/evaluate", response_model=GateDecision, tags=["gates"])
def gates_evaluate(request: GateRequest) -> GateDecision:
    """Evaluate one candidate state against an inline gate policy.

    Returns an immutable ``passed | failed`` decision whose evidence chain is
    explicit (decision -> evaluations -> comparison -> state/checkpoint). The
    gate never trains, rolls back or selects anything — a failed checkpoint
    gate only suggests a rollback target for the caller to use via the M3
    rollback endpoint.
    M57: ``baseline_from_best`` is a WORKFLOW gate-stage declaration — this
    DIRECT route rejects it (422): the workflow engine's single resolver
    pins the M52 best selection into a concrete ``baseline_checkpoint_id``
    before the gate engine runs; name the baseline explicitly here (GET
    /models/{id}/checkpoints/best answers it).
    """
    try:
        return _forge().run_gate(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


@api.get("/models/{model_id}/gates/decisions", response_model=list[dict], tags=["gates"])
def list_gate_decisions(model_id: str) -> list[dict[str, Any]]:
    """Immutable gate decision history of a model (404 for unknown models)."""
    try:
        return [r.model_dump(mode="json")
                for r in _forge().list_gate_decisions(model_id)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/gates/decisions/by-policy/{policy_id}",
         response_model=list[GateDecision], tags=["gates"])
def list_gate_decisions_by_policy(model_id: str, policy_id: str
                                  ) -> list[GateDecision]:
    """Immutable gate decisions of ONE registered policy of one model (M23).

    Read-only per-policy grouping: resolves the policy through the
    existing M9 policy registry (an unknown policy is 404 — policy
    identity is the persisted definition manifest, never inferred from
    gate-directory names) and the model through the existing registry
    (unknown model is 404), then returns the model's authoritative M6
    listing filtered by the persisted policy_id recorded in each
    GateDecision — complete verbatim payloads in the exact M6
    (created_at, decision_id) order. Inline-policy decisions keep
    policy_id null and never appear; a valid policy with no decisions
    for this model returns []. Answers only which immutable decisions
    belong to this model and policy; no aggregation, no verdicts beyond
    the persisted ones, no writes. (Must stay registered before
    /gates/decisions/{decision_id}; the literal "by-policy" segment is
    not a decision id.)"""
    try:
        return _forge().list_gate_decisions_for_policy(model_id, policy_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/gates/decisions/by-comparison/{comparison_id}",
         response_model=list[GateDecision], tags=["gates"])
def list_gate_decisions_by_comparison(model_id: str, comparison_id: str
                                      ) -> list[GateDecision]:
    """Immutable gate decisions that judged ONE comparison (M34).

    Read-only per-comparison grouping: resolves the comparison through
    the model's own M5 comparison registry (an unknown comparison, or a
    comparison id belonging to another model, is 404 — comparisons are
    model-scoped; identity is the persisted comparison manifest, never
    inferred from gate-directory names) and the model through the
    existing registry (unknown model is 404), then returns the model's
    authoritative M6 listing filtered by the persisted comparison_id
    recorded in each GateDecision — complete verbatim payloads
    (verdict, decision, losses, delta, reason included) in the exact
    M6 (created_at, decision_id) order. Legacy direct-evaluation
    decisions keep comparison_id null and never appear; a valid
    comparison with no decisions for this model returns []. Answers
    only which immutable decisions judged this comparison; no
    aggregation, no new verdicts, no re-evaluation, no writes. (Must
    stay registered before /gates/decisions/{decision_id}; the literal
    "by-comparison" segment is not a decision id.)"""
    try:
        return _forge().list_gate_decisions_for_comparison(
            model_id, comparison_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/gates/decisions/by-decision/{decision}",
         response_model=list[GateDecision], tags=["gates"])
def list_gate_decisions_by_decision(model_id: str,
                                    decision: GateDecisionResult
                                    ) -> list[GateDecision]:
    """Immutable gate decisions of ONE model with ONE decision result
    (M41).

    Read-only per-decision grouping: returns the model's authoritative
    M6 listing filtered by the persisted decision recorded in each
    GateDecision (top-level decision — the schema enum passed/failed,
    the immutable policy verdict persisted at run time by the M6 gate
    flow; matched VERBATIM — membership NEVER comes from recalculating
    loss deltas, tolerances, policy thresholds, gate configuration or
    comparison results, no gate is re-evaluated; the persisted value
    is never resolved or rewritten). Complete verbatim payloads
    (verdict/evidence chain/rollback suggestion included) in the exact
    M6 (created_at, decision_id) order; a valid decision with no gate
    decisions for the model returns []. Decision results have NO
    registry (unlike the by-policy/by-comparison axes): the
    GateDecisionResult enum IS the contract, so an UNSUPPORTED
    decision value is rejected with 422 at the API boundary
    (schema-level validation — never a registry-style 404, and the
    validation fires BEFORE this handler even for an unknown model),
    while an unknown model with a VALID decision raises
    FileNotFoundError -> 404 exactly like the sibling groupings. No
    aggregates, no rankings, no writes. (Must stay registered before
    /gates/decisions/{decision_id}; the literal "by-decision" segment
    is not a decision id.)"""
    try:
        return _forge().list_gate_decisions_for_decision(model_id,
                                                         decision)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/gates/decisions/by-verdict/{verdict}",
         response_model=list[GateDecision], tags=["gates"])
def list_gate_decisions_by_verdict(model_id: str,
                                   verdict: ComparisonVerdict
                                   ) -> list[GateDecision]:
    """Immutable gate decisions of ONE model with ONE recorded
    comparison verdict (M43).

    Read-only per-verdict grouping: returns the model's authoritative
    M6 listing filtered by the persisted verdict recorded in each
    GateDecision (top-level verdict — the Optional[ComparisonVerdict]
    loss-only verdict recorded verbatim by the M6 run: improved /
    regressed / unchanged from the measured evidence, deliberately
    DISTINCT from the M41 decision result; matched VERBATIM —
    membership NEVER comes from recalculating losses, deltas,
    tolerances, policies or comparison results, no gate is
    re-evaluated; the persisted value is never resolved or rewritten).
    Complete verbatim payloads (decision/evidence chain/rollback
    suggestion included) in the exact M6 (created_at, decision_id)
    order; a valid verdict with no matching decisions returns [].
    Verdicts have NO registry (exactly like M41): the
    ComparisonVerdict enum IS the contract, so an UNSUPPORTED verdict
    value is rejected with 422 at the API boundary (schema-level
    validation — never a registry-style 404, and the validation fires
    BEFORE this handler even for an unknown model), while an unknown
    model with a VALID verdict raises FileNotFoundError -> 404 exactly
    like the sibling groupings. The field is OPTIONAL and the endpoint
    is for ENUM VALUES ONLY: threshold-only gates keep verdict null,
    there is deliberately NO route representing None, and null-verdict
    decisions belong to NO by-verdict group (they stay listed in the
    generic M6 history). No aggregates, no rankings, no writes. (Must
    stay registered before /gates/decisions/{decision_id}; the literal
    "by-verdict" segment is not a decision id.)"""
    try:
        return _forge().list_gate_decisions_for_verdict(model_id, verdict)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/gates/decisions/by-baseline-type/{baseline_type}",
         response_model=list[GateDecision], tags=["gates"])
def list_gate_decisions_by_baseline_type(model_id: str,
                                         baseline_type: GateBaselineType
                                         ) -> list[GateDecision]:
    """Immutable gate decisions of ONE model whose embedded policy
    compared the candidate against ONE kind of baseline (M45; the
    first nested-field grouping).

    Read-only per-baseline-type grouping over the persisted NESTED
    policy field: every GateDecision embeds its GatePolicy VERBATIM
    (the M6 run persists the policy exactly as evaluated), and every
    policy carries a REQUIRED baseline_type (the schema enum
    checkpoint / current / evaluation_result_hash / minimum_loss);
    the listing is filtered by that persisted nested value — matched
    VERBATIM — membership NEVER comes from deriving the type from
    checkpoint id presence, comparison or evaluation references,
    policy contents outside baseline_type, the gate result, the
    verdict, loss deltas or timestamps, no gate is re-evaluated, and
    the persisted value is never resolved or rewritten. Complete
    verbatim payloads (embedded policy, verdict, evidence chain
    included) in the exact M6 (created_at, decision_id) order.
    Because the field is required, the groups form a TRUE disjoint
    partition of the listing with no None case; the enum exposes the
    COMPLETE contract including evaluation_result_hash — a valid
    value with no matching decisions is a deterministic [] (never
    404, and no route is omitted for empty categories). Baseline
    types have NO registry (exactly like M41/M43): the
    GateBaselineType enum IS the contract, so an UNSUPPORTED baseline
    type is rejected with 422 at the API boundary (schema-level
    validation — never a registry-style 404, and the validation fires
    BEFORE this handler even for an unknown model), while an unknown
    model with a VALID baseline type raises FileNotFoundError -> 404
    exactly like the sibling groupings. No aggregates, no rankings,
    no writes. (Must stay registered before
    /gates/decisions/{decision_id}; the literal "by-baseline-type"
    segment is not a decision id.)"""
    try:
        return _forge().list_gate_decisions_for_baseline_type(
            model_id, baseline_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/gates/decisions/{decision_id}", response_model=GateDecision,
         tags=["gates"])
def get_gate_decision(model_id: str, decision_id: str) -> GateDecision:
    """One persisted immutable gate decision."""
    try:
        return _forge().get_gate_decision(model_id, decision_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Stable policy registry + named probe suites (Milestone 9 — definitions only)
# --------------------------------------------------------------------------- #

@api.post("/policies", response_model=PolicyDefinition,
          status_code=status.HTTP_201_CREATED, tags=["policies"])
def create_policy(request: PolicyCreateRequest) -> PolicyDefinition:
    """Register one immutable policy definition (stable id + M6 semantics).

    Idempotent for the same policy_id with identical semantic configuration
    (returns the existing definition); the same id with any different content
    is a 409 conflict — definitions are never overwritten. The registry adds
    stable identity and reusability only: the GateEngine keeps evaluating the
    exact M6 GatePolicy embedded in the definition.
    M57: a policy declaring ``baseline_from_best`` cannot be registered
    (422) — the M52 best selection is resolved and pinned per workflow
    plan, so best-baseline gates declare their policy INLINE in the gate
    stage.
    """
    try:
        return _forge().register_policy(request)
    except ValueError as exc:
        raise _conflict_or_422(exc) from exc


@api.get("/policies", response_model=list[PolicyDefinition], tags=["policies"])
def list_policies() -> list[PolicyDefinition]:
    """Immutable policy definitions, deterministic order (oldest first)."""
    return _forge().list_policies()


@api.get("/policies/{policy_id}", response_model=PolicyDefinition,
         tags=["policies"])
def get_policy(policy_id: str) -> PolicyDefinition:
    """One immutable policy definition (404 for unknown ids)."""
    try:
        return _forge().get_policy(policy_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/probe-suites", response_model=ProbeSuite,
          status_code=status.HTTP_201_CREATED, tags=["policies"])
def create_probe_suite(request: ProbeSuiteCreateRequest) -> ProbeSuite:
    """Register one immutable named probe suite (a set of exact M4 probes).

    Idempotent for the same suite_id with identical probe contents (returns
    the existing suite); the same id with any different contents is a 409
    conflict. Probes are canonicalized on registration — order carries no
    evaluation meaning and never influences identity or output. A suite is a
    named container only: it computes no score and each probe stays an
    independent M4 evaluation probe.
    """
    try:
        return _forge().register_probe_suite(request)
    except ValueError as exc:
        raise _conflict_or_422(exc) from exc


@api.get("/probe-suites", response_model=list[ProbeSuite], tags=["policies"])
def list_probe_suites() -> list[ProbeSuite]:
    """Immutable probe suites, deterministic order (oldest first)."""
    return _forge().list_probe_suites()


@api.get("/probe-suites/{suite_id}", response_model=ProbeSuite,
         tags=["policies"])
def get_probe_suite(suite_id: str) -> ProbeSuite:
    """One immutable probe suite (404 for unknown ids)."""
    try:
        return _forge().get_probe_suite(suite_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Suite runs (Milestone 10 — explicit multi-probe M4 batches over named suites;
# Milestone 21 adds the read-only by-suite grouping of the immutable history,
# Milestone 22 its bookkeeping summary and Milestone 25 the by-checkpoint
# grouping; Milestone 50 adds the read-only by-reused-count grouping)
# --------------------------------------------------------------------------- #

@api.post("/suite-runs", response_model=SuiteRunRecord, tags=["suite-runs"])
def run_suite(request: SuiteRunRequest) -> SuiteRunRecord:
    """Execute ONE named probe suite against ONE explicit model state.

    Every probe of the suite runs as an independent exact M4 evaluation
    (reused when identical evidence already exists) and one immutable
    suite-run manifest is persisted. The response exposes each probe and its
    evaluation id separately — no aggregation, score or ranking is computed.
    Unknown model/suite -> 404; corrupt suite -> 409; invalid request -> 422.
    A probe that cannot execute is recorded per-probe as failed (the run
    persists with status "failed"); nothing is fabricated or silently skipped.
    """
    try:
        return _forge().run_suite(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


@api.get("/models/{model_id}/suite-runs", response_model=list[SuiteRunRecord],
         tags=["suite-runs"])
def list_suite_runs(model_id: str) -> list[SuiteRunRecord]:
    """Immutable suite-run history of a model, deterministic order (404 for
    unknown models)."""
    try:
        return _forge().list_suite_runs(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/suite-runs/by-suite/{suite_id}",
         response_model=list[SuiteRunRecord], tags=["suite-runs"])
def list_suite_runs_by_suite(model_id: str, suite_id: str
                             ) -> list[SuiteRunRecord]:
    """Immutable suite-run records of ONE named suite of one model (M21).

    Read-only per-suite grouping: resolves the suite through the existing
    M9 probe-suite registry (an unknown suite is 404 — suite identity is
    the persisted suite manifest, never inferred from run-directory
    names) and the model through the existing registry (unknown model is
    404), then returns the model's authoritative M10 listing filtered by
    the persisted suite_id recorded in each SuiteRunRecord — complete
    verbatim payloads in the exact M10 (created_at, suite_run_id) order.
    A valid suite with no runs for this model returns []. Answers only
    which immutable suite-run records belong to this model and suite; no
    aggregation, no scores, no cross-suite comparison, no writes. (Must
    stay registered before /suite-runs/{suite_run_id}; the literal
    "by-suite" segment is not a suite-run id.)"""
    try:
        return _forge().list_suite_runs_for_suite(model_id, suite_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/suite-runs/by-suite/{suite_id}/summary",
         response_model=SuiteRunSummary, tags=["suite-runs"])
def summarize_suite_runs_by_suite(model_id: str, suite_id: str
                                  ) -> SuiteRunSummary:
    """Read-only bookkeeping summary of ONE model's suite-run history for
    ONE named suite (M22).

    Pure derived view over the immutable records: the exact M21 grouping
    (model + M9 suite-registry validation, persisted suite_id filter) is
    summarized with identity/counting bookkeeping only — total_count, the
    run ids in the deterministic M21 ASCENDING (created_at, suite_run_id)
    order, and the earliest/latest recorded run timestamps. A valid suite
    with no runs for this model returns a zero summary (total_count 0,
    empty run_ids, null timestamps), not a 404. No scores, averages,
    trends, verdicts or comparisons; never persisted, never writes."""
    try:
        return _forge().list_suite_run_summary_for_suite(model_id, suite_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/suite-runs/by-checkpoint/{checkpoint_id}",
         response_model=list[SuiteRunRecord], tags=["suite-runs"])
def list_suite_runs_by_checkpoint(model_id: str, checkpoint_id: str
                                  ) -> list[SuiteRunRecord]:
    """Immutable M10 suite runs executed against ONE checkpoint state (M25).

    Read-only per-checkpoint grouping: validates the checkpoint through
    the model's M3 checkpoint registry (an unknown checkpoint, or a
    checkpoint id belonging to another model, is 404 — checkpoint ids
    are model-scoped; nothing is inferred from filenames), then returns
    the model's authoritative M10 listing filtered by the persisted run
    state recorded in each SuiteRunRecord (state_kind "checkpoint" with
    the requested state.checkpoint_id) — complete verbatim payloads in
    the exact M10 (created_at, suite_run_id) order. Current-state runs
    keep state.checkpoint_id null and never appear; a valid checkpoint
    with no suite runs returns []. Answers only which immutable runs
    executed against this checkpoint; no aggregation, no scores, no
    writes. (Must stay registered before /suite-runs/{suite_run_id};
    the literal "by-checkpoint" segment is not a suite-run id.)"""
    try:
        return _forge().list_suite_runs_for_checkpoint(model_id,
                                                       checkpoint_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/suite-runs/by-reused/{reused_count}",
         response_model=list[SuiteRunRecord], tags=["suite-runs"])
def list_suite_runs_by_reused_count(model_id: str,
                                    reused_count: int
                                    ) -> list[SuiteRunRecord]:
    """Immutable M10 suite runs of ONE model by persisted reuse count
    (M50).

    Read-only per-count grouping: returns the model's authoritative
    M10 listing filtered by the REQUIRED integer ``reused_count``
    persisted on each SuiteRunRecord at run time (the number of
    probes satisfied by pre-existing evidence) — matched VERBATIM,
    never recalculated, never derived from probe outcomes,
    completed_count, failed_count, probe_count, suite size, status,
    timestamps, artifact ids, evaluation or comparison records,
    configuration or any other field. The count is EXECUTION
    BOOKKEEPING, never a score, a ranking or a quality signal —
    reused evidence is not better or worse, it is how the immutable
    M4 cache satisfied the suite. Complete verbatim payloads in the
    exact M10 (created_at, suite_run_id) order. The count is an OPEN
    integer value axis (no registry, no enum): any integer is
    type-valid — an unmatched count on a valid model returns 200 []
    — while a non-integer spelling is rejected with 422 at the API
    boundary (schema-level validation — never a silent
    reinterpretation, and the validation fires BEFORE this handler
    even for an unknown model); an unknown model with a VALID
    integer raises FileNotFoundError -> 404 exactly like the sibling
    groupings. No aggregation, no writes. (Must stay registered
    before /suite-runs/{suite_run_id}; the literal "by-reused"
    segment is not a suite-run id.)"""
    try:
        return _forge().list_suite_runs_for_reused_count(
            model_id, reused_count)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/suite-runs/{suite_run_id}",
         response_model=SuiteRunRecord, tags=["suite-runs"])
def get_suite_run(model_id: str, suite_run_id: str) -> SuiteRunRecord:
    """One persisted immutable suite run (404 for unknown model or run)."""
    try:
        return _forge().get_suite_run(model_id, suite_run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# Workflow routes (Milestone 7 — ordered orchestration over M3–M6;
# Milestone 35 adds the read-only model-scoped by-recipe grouping of the history;
# Milestone 42 adds the read-only model-scoped by-status grouping of the history;
# Milestone 51 adds the read-only model-bound recipe RESOLUTION (preflight);
# Milestone 53 adds the 'best' state reference resolved through the M52
#   selection at execution/preflight time (no new route — shared resolver))
# --------------------------------------------------------------------------- #

@api.post("/workflows/run", response_model=WorkflowRecord, tags=["workflows"])
def workflows_run(plan: WorkflowPlan) -> WorkflowRecord:
    """Execute one inline workflow plan synchronously.

    Stages run through the existing M3–M6 engines in declared order with
    explicit references and forward branches on M6 gate decisions. A gate
    failure stops the run (recording the rollback suggestion only) unless an
    on_fail branch is declared. Missing/corrupt/invalid inputs map to
    404/409/422; a failed run is still persisted so earlier successful
    artifacts stay referenced and auditable.
    """
    try:
        return _forge().run_workflow(plan)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


@api.get("/models/{model_id}/workflows", response_model=list[dict], tags=["workflows"])
def list_workflows(model_id: str) -> list[dict[str, Any]]:
    """Immutable workflow run history of a model (404 for unknown models)."""
    try:
        return [r.model_dump(mode="json")
                for r in _forge().list_workflows(model_id)]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/workflows/by-recipe/{recipe_id}",
         response_model=list[WorkflowRecord], tags=["workflows"])
def list_workflows_by_recipe(model_id: str, recipe_id: str
                             ) -> list[WorkflowRecord]:
    """Immutable workflow runs of ONE model from ONE recipe (M35).

    Read-only model-scoped per-recipe grouping: resolves the recipe
    through the GLOBAL M12/M14 recipe registry (an unknown recipe is
    404 — the same resolution GET /workflows/recipes/{recipe_id} uses;
    recipes are global, so model scoping comes from the model's own
    M11 listing and a model never sees another model's runs; nothing
    is inferred from filenames or stage contents) and the model
    through the existing registry (unknown model is 404 — a valid
    recipe never makes an unknown model valid), then returns the
    model's authoritative M11 listing filtered by the persisted
    top-level recipe_id recorded in each WorkflowRecord — complete
    verbatim payloads (status, stages, transitions, result_hash and
    the recorded recipe_id/recipe_hash provenance included) in the
    exact M11 (created_at, workflow_id) order. Ad-hoc runs keep
    recipe_id null and never appear (no ad-hoc pseudo-recipe exists);
    a VALID registered recipe with no runs for this model returns []
    (never 404). This is the model-scoped complement of the GLOBAL M12
    cross-model /workflows/recipes/{recipe_id}/runs lineage surface,
    which stays unchanged. No execution, no aggregation, no writes.
    (Must stay registered before /workflows/{workflow_id}; the literal
    "by-recipe" segment is not a workflow id.)"""
    try:
        return _forge().list_workflows_for_recipe(model_id, recipe_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/workflows/by-status/{status}",
         response_model=list[WorkflowRecord], tags=["workflows"])
def list_workflows_by_status(model_id: str, status: WorkflowStatus
                             ) -> list[WorkflowRecord]:
    """Immutable workflow runs of ONE model with ONE terminal status
    (M42).

    Read-only per-status grouping: returns the model's authoritative
    M11 listing filtered by the persisted status recorded in each
    WorkflowRecord (top-level status — the schema enum completed/
    failed/stopped, the terminal state persisted at run end by the M7
    orchestration; matched VERBATIM — membership NEVER comes from
    inferring or recalculating stage results, failed stage ids,
    timestamps, artifact existence or recipe information, nothing is
    re-executed; the persisted value is never resolved or rewritten).
    Complete verbatim payloads (stages, transitions, terminal_reason,
    result_hash and recipe provenance included) in the exact M11
    (created_at, workflow_id) order; a valid status with no matching
    runs for the model returns []. Statuses have NO registry (unlike
    the by-recipe axis): the WorkflowStatus enum IS the contract, so
    an UNSUPPORTED status value is rejected with 422 at the API
    boundary (schema-level validation — never a registry-style 404,
    and the validation fires BEFORE this handler even for an unknown
    model), while an unknown model with a VALID status raises
    FileNotFoundError -> 404 exactly like the sibling grouping. No
    aggregation, no analytics, no writes. (Must stay registered
    before /workflows/{workflow_id}; the literal "by-status" segment
    is not a workflow id.)"""
    try:
        return _forge().list_workflows_for_status(model_id, status)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/workflows/{workflow_id}", response_model=WorkflowRecord,
         tags=["workflows"])
def get_workflow(model_id: str, workflow_id: str) -> WorkflowRecord:
    """One persisted immutable workflow run."""
    try:
        return _forge().get_workflow(model_id, workflow_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Workflow recipes (Milestone 12 — immutable, reusable M7 pipeline definitions)
# --------------------------------------------------------------------------- #

@api.post("/workflows/recipes", response_model=WorkflowRecipe,
          status_code=status.HTTP_201_CREATED, tags=["workflows"])
def create_workflow_recipe(request: WorkflowRecipeCreateRequest) -> WorkflowRecipe:
    """Register one immutable workflow recipe (an ordered M7 stage list).

    Idempotent for the same recipe_id with identical semantic stage content
    (returns the existing definition); the same id with any different content
    is a 409 conflict — recipes are never overwritten. The stage list is
    validated by the exact same rules as inline M7 plans (shared validator).
    A recipe is inert data: it executes only when an explicit run request
    binds one model.

    ``recipe`` stages (M14) reference other registered recipes: registration
    then additionally requires every referenced recipe to exist (422 with no
    manifest otherwise), rejects cycles and compositions deeper than 32
    recipes (422), expands the composition deterministically and re-runs the
    shared validator over the FULLY expanded stage list before persisting the
    manifest with reference-oriented ``composition`` provenance.
    """
    try:
        return _forge().register_workflow_recipe(request)
    except ValueError as exc:
        raise _conflict_or_422(exc) from exc


@api.get("/workflows/recipes", response_model=list[WorkflowRecipe],
         tags=["workflows"])
def list_workflow_recipes() -> list[WorkflowRecipe]:
    """Immutable workflow recipes, deterministic order (oldest first)."""
    return _forge().list_workflow_recipes()


@api.get("/workflows/recipes/{recipe_id}", response_model=WorkflowRecipe,
         tags=["workflows"])
def get_workflow_recipe(recipe_id: str) -> WorkflowRecipe:
    """One immutable workflow recipe (404 for unknown ids)."""
    try:
        return _forge().get_workflow_recipe(recipe_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/workflows/recipes/{recipe_id}/runs", response_model=list[WorkflowRecord],
         tags=["workflows"])
def list_workflow_recipe_runs(recipe_id: str) -> list[WorkflowRecord]:
    """Cross-model lineage of ONE recipe's workflow runs (read-only, M13).

    Every persisted workflow run generated from this recipe is returned with
    its recorded model_id — the view is recipe-scoped because a recipe may be
    explicitly bound to different models over time. Live scan of the
    existing workflow manifests: no index, no cache, no writes. Unknown
    recipe -> 404; existing recipe with no runs -> []. Ordering is
    (created_at, workflow_id).
    """
    try:
        return _forge().list_recipe_runs(recipe_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/workflows/recipes/{recipe_id}/plan",
         response_model=WorkflowRecipeResolution, tags=["workflows"])
def resolve_workflow_recipe(model_id: str, recipe_id: str
                            ) -> WorkflowRecipeResolution:
    """Resolve ONE registered recipe against ONE explicit model WITHOUT
    executing anything (M51 read-only preflight of the recipe-run
    surface).

    Runs the EXACT resolution path of POST /workflows/recipes/
    {recipe_id}/runs — recipe lookup, model validation, M14
    deterministic expansion, WorkflowPlan construction with the FULL
    M7 validation incl. embedded-config model agreement — through the
    SAME RecipeEngine code (one resolution system, never a second
    executor). Returns the expanded, model-bound, fully validated
    WorkflowPlan the WorkflowEngine WOULD execute (its plan_hash
    predicts the executed run's plan_hash), plus recipe provenance
    (recipe_id + recipe_hash) and the additive M14 composition trace
    (null for plain recipes). A computed view: NEVER persisted, never
    written — a resolve leaves zero new files. Error semantics
    identical to a run request: unknown recipe/model -> 404 with
    nothing persisted; binding/schema conflicts -> 422. No execution,
    no scheduling, no background anything.
    """
    try:
        return _forge().resolve_workflow_recipe(recipe_id, model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@api.post("/workflows/recipes/{recipe_id}/runs",
          response_model=WorkflowRecord | WorkflowRecipeRepetitionRun,
          tags=["workflows"])
def run_workflow_recipe(recipe_id: str,
                        request: WorkflowRecipeRunRequest
                        ) -> WorkflowRecord | WorkflowRecipeRepetitionRun:
    """Execute ONE registered workflow recipe against ONE explicit model.

    The model is the only runtime binding; the recipe's stage list (EXPANDED
    deterministically first for composite recipes — M14) is re-validated as
    an existing M7 WorkflowPlan (same structural rules plus embedded-config
    model agreement) and executed by the existing WorkflowEngine. Unknown
    recipe/model -> 404 with nothing persisted; binding/schema conflicts ->
    422; runtime stage failures persist a failed run record exactly like
    inline plans and map to 404/409/422. Recipe provenance (recipe_id +
    config_hash) plus the additive composition trace (M14) are recorded on
    the run; referenced recipes never claim the run in their lineage.

    M58: ``repetitions`` (default 1, bounded 1..16) executes the recipe N
    times sequentially — each iteration a FULL independent resolution +
    execution, so 'best' references may advance between iterations.
    repetitions=1 (or null) returns exactly today's single WorkflowRecord;
    repetitions>1 returns an ordered batch view (requested/actual counts +
    ordered workflow ids + the full immutable records). An iteration ending
    'stopped' (gate stop) ends the sequence and the batch is returned; a
    'failed' iteration keeps the existing error semantics (the failed
    record persists, the error is raised, later iterations never run). No
    automatic retry, no rollback, no unbounded repetition.
    """
    try:
        if request.repetitions == 1:
            return _forge().run_workflow_recipe(recipe_id, request.model_id)
        records = _forge().run_workflow_recipe_repeated(
            recipe_id, request.model_id, request.repetitions)
        return WorkflowRecipeRepetitionRun(
            recipe_id=recipe_id,
            model_id=request.model_id,
            repetitions=request.repetitions,
            executed=len(records),
            stopped_early=len(records) < request.repetitions,
            workflow_ids=[r.workflow_id for r in records],
            records=records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


# --------------------------------------------------------------------------- #
# Checkpoint sampling (Milestone 15 — deterministic text generation, inference;
# Milestone 27 adds the read-only by-checkpoint grouping of the sample history;
# Milestone 32 adds the read-only by-tokenizer grouping of the sample history;
# Milestone 40 adds the read-only by-strategy grouping of the sample history)
# --------------------------------------------------------------------------- #

@api.post("/samples/generate", response_model=SampleRecord,
          tags=["sampling"])
def generate_sample(request: SampleGenerateRequest) -> SampleRecord:
    """Generate text from ONE explicitly selected verified immutable checkpoint.

    Inference only: the model is never trained, evaluated, scored or modified.
    Every input is explicit (model_id, checkpoint_id, tokenizer_id, prompt,
    strategy, parameters) — nothing is auto-selected; the checkpoint passes
    the existing integrity verification before anything runs; the tokenizer/
    model vocabulary compatibility follows the platform convention
    (model vocab_size >= tokenizer actual vocab, exactly as M3/M4 enforce).
    Greedy is deterministic argmax; temperature sampling uses one seeded
    deterministic RNG stream from the request's explicit integer seed.
    Exactly ONE immutable sample manifest is created per successful request
    (samples/<model_id>/sample-<id>/manifest.json). Unknown model/checkpoint/
    tokenizer -> 404; corrupt checkpoint -> 409; vocabulary/parameter/context
    violations -> 422; every preflight failure writes nothing.
    """
    try:
        return _forge().generate_sample(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


@api.get("/models/{model_id}/samples", response_model=list[SampleRecord],
         tags=["sampling"])
def list_samples(model_id: str) -> list[SampleRecord]:
    """Immutable sample history of one model, deterministic
    (created_at, sample_id) order (404 for unknown models; read-only)."""
    try:
        return _forge().list_samples(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/samples/by-checkpoint/{checkpoint_id}",
         response_model=list[SampleRecord], tags=["sampling"])
def list_samples_by_checkpoint(model_id: str, checkpoint_id: str
                               ) -> list[SampleRecord]:
    """Immutable M15 samples generated from ONE checkpoint (M27).

    Read-only per-checkpoint grouping: validates the checkpoint through
    the model's M3 checkpoint registry (an unknown checkpoint, or a
    checkpoint id belonging to another model, is 404 — checkpoint ids
    are model-scoped; nothing is inferred from filenames), then returns
    the model's authoritative M15 listing filtered by the persisted
    sample identity — every sample carries a required non-nullable
    checkpoint_id (M15 generation always binds one explicit verified
    checkpoint; there is no current-state sample) and belongs to the
    request only when that persisted id matches. Complete verbatim
    payloads (prompt, token ids, output text, strategy, result_hash
    included) in the exact M15 (created_at, sample_id) order; a valid
    checkpoint with no samples returns []. No new metrics, no writes.
    (Must stay registered before /samples/{sample_id}; the literal
    "by-checkpoint" segment is not a sample id.)"""
    try:
        return _forge().list_samples_for_checkpoint(model_id, checkpoint_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/samples/by-tokenizer/{tokenizer_id}",
         response_model=list[SampleRecord], tags=["sampling"])
def list_samples_by_tokenizer(model_id: str, tokenizer_id: str
                              ) -> list[SampleRecord]:
    """Immutable M15 samples generated with ONE tokenizer (M32).

    Read-only per-tokenizer grouping: validates the tokenizer through
    the GLOBAL M2 tokenizer registry (an unknown tokenizer is 404 —
    the same registry getter GET /tokenizers/{id} exposes; model
    scoping comes from the model's own M15 listing, so a model never
    sees another model's samples; nothing is inferred from filenames),
    then returns the model's authoritative M15 listing filtered by the
    persisted sample tokenizer identity — every sample carries a
    required non-nullable top-level tokenizer_id (plus its matching
    tokenizer_hash, preserved verbatim; no M15 schema change) and
    belongs to the request only when that persisted id matches
    VERBATIM (never a latest-tokenizer substitution; M2/M15 have no
    tokenizer versioning). Complete verbatim payloads (prompt, token
    ids, output text, strategy, result_hash included) in the exact M15
    (created_at, sample_id) order; a valid tokenizer with no samples
    for the model returns [] (never 404). No new metrics, no
    generation, no writes. (Must stay registered before
    /samples/{sample_id}; the literal "by-tokenizer" segment is not a
    sample id.)"""
    try:
        return _forge().list_samples_for_tokenizer(model_id, tokenizer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/samples/by-strategy/{strategy}",
         response_model=list[SampleRecord], tags=["sampling"])
def list_samples_by_strategy(model_id: str, strategy: SampleStrategy
                             ) -> list[SampleRecord]:
    """Immutable M15 samples of ONE model generated with ONE decoding
    strategy (M40).

    Read-only per-strategy grouping: returns the model's authoritative
    M15 listing filtered by the persisted strategy recorded in each
    SampleRecord (top-level strategy — the schema enum
    greedy/temperature, persisted verbatim at generation time from the
    explicit request; matched VERBATIM — membership NEVER comes from
    sample ids, prompt text, generated token ids, temperature values,
    seed presence, filenames or manifest paths, and the strategy is
    NEVER recalculated from temperature/seed/other fields; no sample
    is regenerated). Complete verbatim payloads (prompt, token ids,
    output text, temperature, result_hash included) in the exact M15
    (created_at, sample_id) order; a valid strategy with no samples
    for the model returns []. Strategies have NO registry (unlike the
    by-checkpoint/by-tokenizer axes): the SampleStrategy enum IS the
    contract, so an UNSUPPORTED strategy value is rejected with 422 at
    the API boundary (schema-level validation — never a registry-style
    404, and the validation fires BEFORE this handler even for an
    unknown model), while an unknown model with a VALID strategy
    raises FileNotFoundError -> 404 exactly like the sibling
    groupings. No aggregates, no scoring, no writes. (Must stay
    registered before /samples/{sample_id}; the literal "by-strategy"
    segment is not a sample id.)"""
    try:
        return _forge().list_samples_for_strategy(model_id, strategy)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/samples/{sample_id}",
         response_model=SampleRecord, tags=["sampling"])
def get_sample(model_id: str, sample_id: str) -> SampleRecord:
    """One persisted immutable sample (404 for unknown model or sample)."""
    try:
        return _forge().get_sample(model_id, sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Sample quality (Milestone 16 — per-sample likelihood measurement, read-only;
# Milestone 33 adds the read-only by-tokenizer grouping of the measurement history)
# --------------------------------------------------------------------------- #

@api.post("/models/{model_id}/samples/{sample_id}/quality",
          response_model=SampleEvaluationRecord, tags=["sample-quality"])
def evaluate_sample_quality(model_id: str, sample_id: str
                            ) -> SampleEvaluationRecord:
    """Measure ONE immutable M15 sample under its own recorded state.

    The request carries NO checkpoint/tokenizer/metric configuration: the
    sample manifest is the source of truth (model + sample resolve the
    recorded checkpoint and tokenizer, both verified — content-hash
    integrity plus the platform vocabulary convention). The full recorded
    prompt + generated token sequence is scored in ONE context window
    under the existing M4 causal-LM objective over the GENERATED
    continuation targets only (prompt tokens condition, never count;
    first generated token conditioned on the full prompt; every generated
    target counted exactly once). Metrics: loss_nats + perplexity =
    exp(min(loss, 100)) — a likelihood measurement under one objective,
    NOT an overall quality judgment. Exactly ONE immutable manifest is
    created per success (sample-evaluations/<model_id>/evaluation-<id>/
    manifest.json); samples/checkpoints/tokenizers/M4 evaluations are
    never modified. Unknown model/sample/checkpoint/tokenizer -> 404;
    corrupt/inconsistent records -> 409; invalid measurement state
    (sequence exceeds the context window, vocabulary violations) -> 422;
    every preflight failure writes nothing.
    """
    try:
        return _forge().evaluate_sample(model_id, sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


@api.get("/models/{model_id}/sample-quality",
         response_model=list[SampleEvaluationRecord],
         tags=["sample-quality"])
def list_sample_quality(model_id: str) -> list[SampleEvaluationRecord]:
    """Immutable per-sample measurement history of one model in
    deterministic (created_at, evaluation_id) order (404 for unknown
    models; read-only, never writes)."""
    try:
        return _forge().list_sample_evaluations(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/sample-quality/records",
         response_model=list[SampleEvaluationRecord],
         tags=["sample-quality"])
def list_sample_quality_records(model_id: str
                                ) -> list[SampleEvaluationRecord]:
    """Full immutable M16 sample-evaluation records of one model (M18).

    A dedicated read-only access path to the actual measurement records
    (including the recorded loss_nats/perplexity values) behind the M17
    dashboard's history-structure section. Pure pass-through of the M16
    engine: SampleQualityEngine.list_sample_evaluations() is the
    authoritative source, so ordering is the exact M16 convention
    ((created_at, evaluation_id) ASCENDING), model isolation and empty
    history follow the engine, and records are returned verbatim — no
    derived statistics, no aggregation, no interpretation. The route must
    be registered before /sample-quality/{evaluation_id} so the literal
    "records" segment resolves here. Unknown model -> 404; read-only,
    never writes. (Registered first: the M16 reference listing
    GET /models/{id}/sample-quality and the M17 dashboard section are
    unchanged.)"""
    try:
        return _forge().list_sample_evaluations(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/sample-quality/by-sample/{sample_id}",
         response_model=list[SampleEvaluationRecord],
         tags=["sample-quality"])
def list_sample_quality_by_sample(model_id: str, sample_id: str
                                  ) -> list[SampleEvaluationRecord]:
    """Immutable M16 measurements of ONE sample of one model (M19).

    Read-only per-sample access: resolves the sample under the model (an
    unknown sample, or a sample id belonging to another model, is 404 —
    sample ids are model-scoped paths, nothing is inferred from filenames),
    then returns the model's authoritative M16 listing filtered by the
    sample's persisted sample_id — complete verbatim SampleEvaluationRecord
    payloads (loss_nats/perplexity included) in the exact M16
    (created_at, evaluation_id) order. A sample with no measurements
    returns []. No derived fields, no statistics, no quality judgment; the
    M16/M17/M18 surfaces are unchanged; read-only, never writes. (Must stay
    registered before /sample-quality/{evaluation_id}; the literal
    "by-sample" segment is not an evaluation id.)"""
    try:
        return _forge().list_sample_evaluations_for_sample(model_id,
                                                           sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/sample-quality/by-checkpoint/{checkpoint_id}",
         response_model=list[SampleEvaluationRecord],
         tags=["sample-quality"])
def list_sample_quality_by_checkpoint(model_id: str, checkpoint_id: str
                                      ) -> list[SampleEvaluationRecord]:
    """Immutable M16 measurements recorded under ONE checkpoint (M20).

    Read-only per-checkpoint access: validates the checkpoint through the
    model's M3 checkpoint registry (an unknown checkpoint, or a checkpoint
    id belonging to another model, is 404 — checkpoint ids are
    model-scoped; nothing is inferred from filenames), then returns the
    model's authoritative M16 listing filtered by the persisted
    checkpoint_id recorded in each SampleEvaluationRecord — complete
    verbatim payloads (loss_nats/perplexity included) in the exact M16
    (created_at, evaluation_id) order. A checkpoint with no measurements
    returns []. Answers only which immutable evaluations belong to this
    checkpoint; no derived fields, no statistics, no quality judgment, no
    checkpoint comparison; the M16/M17/M18/M19 surfaces are unchanged;
    read-only, never writes. (Must stay registered before
    /sample-quality/{evaluation_id}; the literal "by-checkpoint" segment
    is not an evaluation id.)"""
    try:
        return _forge().list_sample_evaluations_for_checkpoint(
            model_id, checkpoint_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/sample-quality/by-tokenizer/{tokenizer_id}",
         response_model=list[SampleEvaluationRecord],
         tags=["sample-quality"])
def list_sample_quality_by_tokenizer(model_id: str, tokenizer_id: str
                                     ) -> list[SampleEvaluationRecord]:
    """Immutable M16 measurements whose samples used ONE tokenizer (M33).

    Read-only per-tokenizer grouping: validates the tokenizer through
    the GLOBAL M2 tokenizer registry (an unknown tokenizer is 404 —
    the same registry getter GET /tokenizers/{id} exposes; model
    scoping comes from the model's own M16 listing, so a model never
    sees another model's measurements; nothing is inferred from
    filenames), then returns the model's authoritative M16 listing
    filtered by the persisted measurement tokenizer identity — every
    SampleEvaluationRecord carries a required non-nullable top-level
    tokenizer_id (M16 measures an immutable M15 sample under its own
    RECORDED state; the record persists that state's tokenizer
    identity) and belongs to the request only when that persisted id
    matches VERBATIM (never inferred from sample/checkpoint ids or
    hashes; never a latest-tokenizer substitution; M2/M16 have no
    tokenizer versioning). Complete verbatim payloads
    (loss_nats/perplexity included) in the exact M16 (created_at,
    evaluation_id) order; a valid tokenizer with no measurements for
    the model returns [] (never 404). No new metrics, no aggregation,
    no ranking, no writes. (Must stay registered before
    /sample-quality/{evaluation_id}; the literal "by-tokenizer"
    segment is not an evaluation id.)"""
    try:
        return _forge().list_sample_evaluations_for_tokenizer(
            model_id, tokenizer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/sample-quality/{evaluation_id}",
         response_model=SampleEvaluationRecord,
         tags=["sample-quality"])
def get_sample_quality(model_id: str, evaluation_id: str
                       ) -> SampleEvaluationRecord:
    """One persisted immutable per-sample measurement (404 for unknown
    model or evaluation; read-only)."""
    try:
        return _forge().get_sample_evaluation(model_id, evaluation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/models/{model_id}/dashboard", response_model=ModelDashboard,
         tags=["dashboard"])
def get_dashboard(model_id: str) -> ModelDashboard:
    """Read-only regression dashboard over the model's immutable history.

    Deterministic live recomputation from the persisted manifests: checkpoint
    lineage, evaluation series grouped by exact probe identity, comparison
    series by state pair + probe, gate decision series by recorded policy,
    workflow runs with status counts, and the artifact reference graph.
    Never writes; corrupt artifacts are skipped with diagnostics; unknown
    model -> 404. No quality scores or rankings are produced.
    """
    try:
        return _forge().get_dashboard(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        if "integrity" in detail.lower() or "corrupt" in detail.lower():
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=422, detail=detail) from exc


app.include_router(api)
