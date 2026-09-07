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
    EvaluationConfig,
    EvaluationRecord,
    EvaluationSplit,
    GateDecision,
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
    ProjectInfo,
    SampleGenerateRequest,
    SampleEvaluationRecord,
    SampleRecord,
    RollbackRequest,
    TokenizerConfig,
    TrainingConfig,
    TrainingReport,
    TransformerConfig,
    ValidationReport,
    WorkflowPlan,
    WorkflowRecord,
    WorkflowRecipe,
    WorkflowRecipeCreateRequest,
    WorkflowRecipeRunRequest,
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
      <li><code>POST  {prefix}/models</code> — create a model (validated config)</li>
      <li><code>POST  {prefix}/datasets/upload</code> — ingest txt/md/csv/json → versioned dataset</li>
      <li><code>GET   {prefix}/datasets/&#123;id&#125;/verify</code> — integrity check (tamper detection)</li>
      <li><code>POST  {prefix}/datasets/&#123;id&#125;/tokenize</code> — tokenize with a stored tokenizer</li>
      <li><code>POST  {prefix}/tokenizers/train</code> — deterministic byte-level BPE</li>
      <li><code>POST  {prefix}/training/run</code> — train (CPT/SFT), synchronous</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints</code> — immutable checkpoint store</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/checkpoints/&#123;ckpt&#125;</code> — one checkpoint</li>
      <li><code>POST  {prefix}/models/&#123;id&#125;/rollback</code> — verified restore of a checkpoint</li>
      <li><code>POST  {prefix}/evaluations/run</code> — read-only evaluation (current state or checkpoint)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations</code> — immutable evaluation history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-checkpoint/&#123;ckpt&#125;</code> — evaluations recorded under ONE checkpoint (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-dataset/&#123;ds&#125;</code> — evaluations of ONE model over ONE dataset (read-only, deterministic, versions verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-tokenizer/&#123;tok&#125;</code> — evaluations of ONE model with ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/by-split/&#123;split&#125;</code> — evaluations of ONE model on ONE dataset split (read-only, deterministic; unsupported split 422)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/evaluations/&#123;eval&#125;</code> — one evaluation record</li>
      <li><code>POST  {prefix}/comparisons/run</code> — A/B comparison (improved / regressed / unchanged)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons</code> — immutable comparison history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-checkpoint/&#123;ckpt&#125;</code> — comparisons involving ONE checkpoint on either side (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-dataset/&#123;ds&#125;</code> — comparisons of ONE model over ONE dataset (read-only, deterministic, versions verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/by-tokenizer/&#123;tok&#125;</code> — comparisons of ONE model with ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/samples/by-tokenizer/&#123;tok&#125;</code> — samples of ONE model generated with ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/sample-quality/by-tokenizer/&#123;tok&#125;</code> — M16 sample-quality measurements of ONE model under ONE tokenizer (read-only, deterministic, persisted identity verbatim)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/comparisons/&#123;comp&#125;</code> — one comparison record</li>
      <li><code>POST  {prefix}/gates/evaluate</code> — stage gate: policy + candidate → passed / failed decision</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions</code> — immutable gate decision history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/by-policy/&#123;policy_id&#125;</code> — gate decisions of ONE registered policy (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/by-comparison/&#123;comparison_id&#125;</code> — gate decisions that judged ONE M5 comparison (read-only, deterministic, no aggregation)</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/gates/decisions/&#123;decision&#125;</code> — one gate decision</li>
      <li><code>POST  {prefix}/workflows/run</code> — execute one inline workflow plan synchronously</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/workflows</code> — immutable workflow run history</li>
      <li><code>GET   {prefix}/models/&#123;id&#125;/workflows/by-recipe/&#123;recipe_id&#125;</code> — workflow runs of ONE model executed from ONE registered recipe (read-only, deterministic, persisted identity verbatim)</li>
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
      <li><code>POST  {prefix}/workflows/recipes</code> — register an immutable workflow recipe (idempotent; conflicts 409)</li>
      <li><code>GET   {prefix}/workflows/recipes</code> / <code>GET {prefix}/workflows/recipes/&#123;recipe_id&#125;</code> — immutable recipes</li>
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


@api.delete("/datasets/{dataset_id}", response_model=dict, tags=["datasets"])
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        _forge().delete_dataset(dataset_id)
    except ValueError as exc:  # dependency conflict
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": dataset_id}


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


@api.delete("/tokenizers/{tokenizer_id}", response_model=dict, tags=["tokenizers"])
def delete_tokenizer(tokenizer_id: str) -> dict[str, Any]:
    if not _forge().tokenizers.exists_id(tokenizer_id):
        raise HTTPException(status_code=404, detail=f"tokenizer '{tokenizer_id}' not found")
    _forge().delete_tokenizer(tokenizer_id)
    return {"deleted": tokenizer_id}


# --------------------------------------------------------------------------- #
# Training
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


@api.get("/models/{model_id}/checkpoints/{checkpoint_id}", response_model=dict,
         tags=["training"])
def get_checkpoint(model_id: str, checkpoint_id: str) -> dict[str, Any]:
    try:
        return _forge().get_checkpoint(model_id, checkpoint_id).model_dump(mode="json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
# Milestone 30 adds the read-only by-tokenizer grouping of the history)
# --------------------------------------------------------------------------- #

@api.post("/evaluations/run", response_model=EvaluationRecord, tags=["evaluation"])
def evaluation_run(config: EvaluationConfig) -> EvaluationRecord:
    """Measure an existing model state against one tokenized split (read-only).

    checkpoint_id = None -> the model's current published weights; otherwise a
    verified immutable checkpoint of that model. Nothing is modified: only one
    immutable evaluation record is created under models/<id>/evaluations/.
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
# Milestone 36 adds the read-only by-split grouping of the same history)
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
# Milestone 34 adds the read-only by-comparison grouping of the same history)
# --------------------------------------------------------------------------- #

@api.post("/gates/evaluate", response_model=GateDecision, tags=["gates"])
def gates_evaluate(request: GateRequest) -> GateDecision:
    """Evaluate one candidate state against an inline gate policy.

    Returns an immutable ``passed | failed`` decision whose evidence chain is
    explicit (decision -> evaluations -> comparison -> state/checkpoint). The
    gate never trains, rolls back or selects anything — a failed checkpoint
    gate only suggests a rollback target for the caller to use via the M3
    rollback endpoint.
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
# grouping)
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


@api.get("/models/{model_id}/suite-runs/{suite_run_id}",
         response_model=SuiteRunRecord, tags=["suite-runs"])
def get_suite_run(model_id: str, suite_run_id: str) -> SuiteRunRecord:
    """One persisted immutable suite run (404 for unknown model or run)."""
    try:
        return _forge().get_suite_run(model_id, suite_run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# Workflow routes (Milestone 7 — ordered orchestration over M3–M6;
# Milestone 35 adds the read-only model-scoped by-recipe grouping of the history)
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


@api.post("/workflows/recipes/{recipe_id}/runs", response_model=WorkflowRecord,
          tags=["workflows"])
def run_workflow_recipe(recipe_id: str,
                        request: WorkflowRecipeRunRequest) -> WorkflowRecord:
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
    """
    try:
        return _forge().run_workflow_recipe(recipe_id, request.model_id)
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
# Milestone 32 adds the read-only by-tokenizer grouping of the sample history)
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
