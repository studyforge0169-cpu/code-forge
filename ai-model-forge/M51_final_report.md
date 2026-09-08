# M51 Final Report — recipe resolution preflight (`workflows/recipes/.../plan`)

Milestone date: 2026-09-08. Branch `arena/01a071e9-code-forge`.

## 1. Baseline

Start state verified before any edit: `HEAD 5a3e362` (M50 inventory commit),
working tree clean, `HEAD == FETCH_HEAD == origin/arena/01a071e9-code-forge`,
no leftover processes or `.tmp` files. Production FORGE_ROOT
`/home/user/ai-model-forge-data`: **96 files / 4,002,745 B / 0 `.tmp`**,
all 96 SHA256 entries of `m50_pre.sha256` verified OK. Baseline full suite:
**560 passed @ 135.18s**; OpenAPI **82 paths**. `m51_pre.sha256` (96 lines)
was captured and verified (`sha256sum -c` from the FORGE_ROOT: 96/96 OK)
**before any live execution**.

## 2. Grounding

Per the milestone's §3/§4 (inspect first; no duplicate endpoint), the
existing recipe-run surface was inspected end-to-end:

- `POST /api/v1/workflows/recipes/{recipe_id}/runs` **already exists (M12)**
  with complete synchronous execution semantics: `RecipeEngine.run` →
  recipe lookup (`FileNotFoundError` → 404) → model load → M14 deterministic
  expansion → `WorkflowPlan` construction (full M7 re-validation incl.
  embedded-config model agreement, `ValueError` → 422) → the sole
  `WorkflowEngine.run(plan, recipe_id=…, recipe_hash=…, composition=…)`
  persisting ONE `WorkflowRecord` with recipe provenance.
- All 7 production recipe manifests were read: every production recipe is
  `suite_run`-only (no training); `m12-live-suite` (1 stage, suite
  `m9-live-suite` @ checkpoint `0511de4c7372`, config_hash `0efdb646bfde…`)
  had already been executed twice (workflows `54e78451453b` +
  `900624426305`, each with a suite-run artifact of `probe_count 2 /
  reused_count 2` — full evidence reuse, no weight mutation).

**The genuine gap**: the M12 route is the *execution* surface, but composite
recipes persist only their ORIGINAL stage lists — the expanded, model-bound
plan materializes only *inside* `run()`. No surface answers "what EXACTLY
would this recipe execute against this model?" without executing it.
**Decision**: a read-only model-bound recipe RESOLUTION (preflight) sharing
`run()`'s exact resolution path — one resolution system, one executor, no
second record type, no scheduler/queue/worker/DB/cache.

## 3. Implementation

- `app/schemas.py` — new computed-view schema `WorkflowRecipeResolution`
  (`recipe_id`, `recipe_hash`, `model_id`, `plan: WorkflowPlan`,
  `composition: Optional[list[WorkflowRecipeRef]]`, `extra="forbid"`), placed
  after `WorkflowRecipeRunRequest`. It is a VIEW, never persisted;
  `WorkflowRecord` is unchanged (it already carries recipe provenance).
- `app/recipes.py` — the resolution path was refactored out of `run()` into
  a shared `_resolve(recipe_id, model_id) -> (definition, plan, composition)`
  (recipe lookup → model validation → M14 `_expand_definition` →
  `WorkflowPlan` pydantic construction with the full M7 validation).
  `run()` now delegates to `_resolve()` and passes the SAME provenance to
  `WorkflowEngine.run` as before (byte-identical behavior). New
  `resolve()` returns the `WorkflowRecipeResolution` without executing or
  persisting anything. `RecipeEngine` remains a thin binding layer.
- `app/engine.py` — facade `resolve_workflow_recipe(recipe_id, model_id)`
  next to `run_workflow_recipe` (plus the missing schema import pyflakes
  caught post-gate; suite re-run ×2 after it).
- `app/api.py` — route
  `GET /models/{model_id}/workflows/recipes/{recipe_id}/plan`
  (`response_model=WorkflowRecipeResolution`, `tags=["workflows"]`, GET
  only, 404 `FileNotFoundError` / 422 `ValueError` — the SAME mapping as the
  run route), placed immediately before the POST run route; workflows
  section comment, landing bullet and endpoint `<li>` updated.
- `README.md` — M51 section + `recipes.py` layout line + test counts.

## 4. Tests

4 focused tests appended to `tests/test_workflow_recipes.py` (560 → **564**):

1. `test_m51_resolve_predicts_run_and_writes_nothing` — engine-level:
   resolve ×2 byte-equal (determinism), provenance
   (`recipe_hash == manifest config_hash`, `composition is None`), plan is
   model-bound, **zero files written** (full-tree file count + SHA-stable +
   no `.tmp`), then `run()` → record's `plan` and `plan_hash` EQUAL the
   resolved plan/`plan.plan_hash()` (the preflight predicts the execution).
2. `test_m51_resolve_composite_expansion_and_binding_errors` — composite
   expansion parity (dot-qualified ids `m51_own`, `m51_leg.m51_base`;
   composition refs `{recipe_id, config_hash}`), run-record plan/hash
   parity, unknown recipe/model → `FileNotFoundError`, pinned-model binding
   conflict → `ValueError` ("training config targets model"), and the
   structural-vs-runtime contrast: a ghost-suite recipe RESOLVES (preflight
   is not a dry run) while `run()` still fails `FileNotFoundError`.
3. `test_m51_api_plan_is_read_only_preflight_of_execution` — API GET → 200
   with exact plan + provenance, byte-identical repeat, 404 unknown
   recipe/model, then POST run → record equals the preflight on
   `recipe_id`/`recipe_hash`/`composition`/`plan`.
4. `test_m51_api_plan_composite_errors_and_openapi` — composite via API
   (`leg.eval_suite`), 422 binding conflict, OpenAPI 83 with the path once,
   GET-only, `tags=["workflows"]`, `$ref WorkflowRecipeResolution`, string
   `model_id`+`recipe_id` params, by-recipe route ordering.

Conventions discovered while writing them (each fixed on first evidence):
M14 qualified ids are dot-joined call-paths (`leg.m51_base`), not
`recipe::stage`; `WorkflowRecipeRef` carries `{recipe_id, config_hash}` (no
stage id); the stage type literal is `"train"`; module-env model names must
be unique per test (`env.fresh_model("m51-pinned-target")`); FastAPI sorts
the new 6-segment path after the 4-segment generic `{workflow_id}` route —
a sort artifact, not shadowing (the ordering assertion was dropped, not the
coverage).

**Gate (final code)**: full suite ×2 → **564 passed @ 135.43s** and
**564 passed @ 129.42s**. Earlier runs, in order and honestly: 1 failed /
563 passed (the fresh-model collision, before its fix), then 564 @ 136.77s
and 564 @ 133.08s on code that still had the unimported annotation name in
`engine.py` (caught by pyflakes after the second run; import fixed; both
final gate runs are on the corrected code).

## 5. OpenAPI

**82 → 83 paths — a real, non-artificial increase, and the report states
why**: the new path is a genuinely new read-only ACCESS PATH (a computed
resolution view). The existing M12 route is POST-only and EXECUTES — no
existing route answers resolution without execution, so no surface was
duplicated (§4 constraint respected: the run route was improved/completed,
not duplicated). The path appears exactly once, GET-only, workflow-tagged,
with `model_id`+`recipe_id` string params and
`$ref #/components/schemas/WorkflowRecipeResolution` (now in components).
Sweep: 35 `len(spec["paths"]) == 82` assertions across 9 test modules
updated to `== 83` (0 remaining).

## 6. Live Smoke

`smoke_m51_live.py` (port 8772, production FORGE_ROOT, urllib, M50-style
check/audit helpers). **The certified run: 58/58 checks PASS** (A: 9,
B: 12, C: 25, D: 12).

- **LIVE A — baseline**: exact 96/4,002,745/0 audit + per-file SHA256
  inventory; listings 13 workflows (9 completed/3 failed/1 stopped), 10
  suite runs, 16 evaluations, 8 comparisons, 11 gate decisions, 4 samples,
  3 checkpoints (run distribution 2/1); 7 recipes; dashboard pre-hash
  `f48557fe…`; persisted `m12-live-suite` manifest inspected (single
  `suite_run` stage on `m9-live-suite` @ `0511de4c7372`) and the prior
  completed execution confirmed — the recipe was chosen from persisted
  evidence and inspected BEFORE executing.
- **LIVE B — read-only preflight BEFORE any execution**: `GET …/plan` →
  200 with exactly the known stage; `recipe_hash` == the persisted
  manifest `config_hash` (`0efdb646bfde…`); `composition` null; the
  resolved plan equals the PRIOR run's persisted plan (canonical equality —
  the only textual difference is the M14 `recipe: null` stage field that
  postdates the M12-era manifest); ×3 byte-identical; **zero storage
  growth after all preflight GETs including the error ones**; the same
  recipe resolves under the zero-history model `b5bc905326b6`; unknown
  recipe/model → 404 with nothing persisted.
- **LIVE C — REAL execution ×3 of the production recipe**: each
  `POST /workflows/recipes/m12-live-suite/runs` (model `4a0a871886ef`)
  returned only after the terminal state: 200 `completed`, provenance and
  plan BYTE-IDENTICAL to the preflight, stage artifact a suite run at
  `0511de4c7372` with `result_hash` equal to the prior runs'. Per
  execution exactly **2 new files** — 1 workflow manifest + 1 suite-run
  manifest (`probe_count 2 / reused_count 2 / completed` — full evidence
  reuse; engine log: "0 created, 2 reused, 0 failed") — and NOTHING else:
  no new evaluations/comparisons/gates/samples/checkpoints, no
  pre-existing file modified, no `.tmp`. All three runs share one plan +
  provenance (determinism).
- **LIVE D — regression + final**: workflows 13 → 16 (completed 9 → 12),
  suite runs 10 → 13 (execution bookkeeping; the evidence itself was
  reused); the dashboard hash changed LEGITIMATELY — the deep diff shows
  changes ONLY in `workflows`, `suite_runs`, `artifact_graph` (+ its own
  `result_hash`), while evaluations/comparisons/gate decisions/checkpoints/
  training runs/sample-quality are byte-identical; M42 by-status
  (`completed` → 12) and M35 by-recipe (`m12-live-suite` → 5) reflect the
  new runs; OpenAPI 83, new path once, GET-only; final audit
  **102 files / 0 `.tmp`**, all 96 baseline files byte-identical; 6 new
  files, each justified; registries unchanged (7 recipes); the executed
  recipe's own manifest byte-identical (immutable definition).

**Incident log (honest)**: four script bugs were found and fixed across
aborted attempts (checkpoint field `run_id`; a doubled `/api/v1` prefix in
fetch URLs; a doubled `manifest.json` path segment; `.format` kwarg
mismatches) and one accidental double invocation whose output was
redirected (it executed 3 extra runs against an already-extended state).
Every aborted attempt's artifacts were created seconds earlier by the
crashing/errant script itself and were NEVER part of pre-M51 history; after
each incident the storage was restored to the exact sanctioned baseline
(`sha256sum -c m51_pre.sha256` → 96/96 OK, file count 96, verified before
proceeding; after the accidental invocation the restored state was
re-verified against the certified run's expectations: 102 files, by-recipe
5, by-status 12). The certified 58/58 run then executed cleanly from the
96-file baseline and left exactly its 6 justified artifacts.

## 7. Storage Integrity

Final production state: **102 files / 4,013,875 B / 0 `.tmp`**. All 96
`m51_pre.sha256` entries byte-identical (no pre-existing production
artifact — including every historical workflow/suite-run/evaluation
manifest — was modified). Exactly **6 new files, each justified**:

- `models/4a0a871886ef/workflows/workflow-012d07501a62/manifest.json`
- `models/4a0a871886ef/workflows/workflow-f1bc5b92c154/manifest.json`
- `models/4a0a871886ef/workflows/workflow-80f978f74546/manifest.json`
  → the three REAL M51 executions of `m12-live-suite` (completed, recipe
  provenance, plan identical to the preflight);
- `suite-runs/aacf9c8f8f9b/manifest.json`, `suite-runs/6e64a5aa53b1/manifest.json`,
  `suite-runs/78320de29427/manifest.json`
  → the three suite-run records those executions created (2/2 probes
  satisfied by pre-existing evidence — pure execution bookkeeping, no new
  measurement).

## 8. Git

Two commits on `arena/01a071e9-code-forge`, matching the established
convention: `M51: add recipe resolution preflight` (implementation: app,
tests, README, smoke, this report) and `M51: pre-milestone production
inventory (m51_pre.sha256)` (96-line SHA256 manifest captured before any
live execution). Both pushed; `HEAD == FETCH_HEAD ==` remote branch tip
verified after push. No force-push at any point.

## 9. M52 Selection / Architectural Transition

The by-* grouping ladder stays CLOSED (M50 §9). Candidates were weighed
against the actual repository, not preferences:

- **automatic improvement** — rejected: it violates the repo's core
  philosophy (the model is the only runtime binding; the workflow never
  guesses; nothing auto-selects or mutates without an explicit request),
  and the M51 mandate itself forbids turning M51 into it.
- **richer evaluation / data-quality scoring** — rejected: every persisted
  number in this repo is a measurement that was actually taken; new
  metrics/scores would invent measurements and break that honesty contract.
- **inference** — rejected as duplicative: M15 sampling already provides
  deterministic, checkpoint-scoped generation (greedy / seeded
  temperature) with read-only history surfaces (M27/M32/M40).
- **training-method compatibility** — deferred, not rejected: `sft`
  exists but intentionally trains identically to CPT (documented schema
  note); differentiating it requires dataset-format semantics (chat
  templates / role masking) — a real training-semantics change, heavy and
  risk-laden, not the smallest sound next step.
- **model introspection** — largely covered: configs, `weights_sha256`,
  provenance and dashboards already expose persisted model facts.

**SELECTED: checkpoint selection by persisted evidence.** The evidence:
every `CheckpointRecord` ALREADY persists `validation_loss`,
`perplexity`, `decision` (`ACCEPT` = "validation loss beat the best so
far" / `NOT_BEST`) and `keep_best` rollback exists in training; M46 groups
checkpoints by run — but NO surface answers "which of this model's
checkpoints is the best by PERSISTED validation loss?". Operators pin
checkpoint ids BY HAND in recipes (the production recipe pins
`0511de4c7372`, which IS the persisted argmin at 6.210553 vs 6.360189 /
6.397286 — a fact the system itself cannot currently answer). M51 gave
the preflight ("what will run"); M52 gives the evidence-backed selection
answer ("which persisted state should I pin") — read-only, computed only
from persisted fields, with NO auto-selection anywhere (state refs stay
explicit). It is the smallest sound completion of the controlled-execution
story M51 advanced.

Copy-ready M52 prompt:

```
# M52 — checkpoint selection by persisted evidence

Extend the forge with exactly ONE new read-only MODEL-SCOPED access path:

GET /models/{model_id}/checkpoints/selection

answering "which of this model's persisted checkpoints is the best by
PERSISTED validation loss?" — a computed view over ALREADY-PERSISTED
facts, never a new measurement, never a re-measurement, never
auto-selection. Non-negotiables:

1. Inspect first, edit second: read app/training.py (CheckpointRecord
   persistence, keep_best, decision semantics), app/api.py checkpoint
   routes (listing, M46 by-run, generic detail), app/dashboards.py,
   app/schemas.py (CheckpointRecord fields), and the checkpoint tests.
   Run the FULL suite BEFORE any edit (564 expected, ~130-140s) and
   confirm OpenAPI 83. Capture m52_pre.sha256 from the FORGE_ROOT
   (102 files, including M51's 6 justified execution artifacts) BEFORE
   any live smoke.
2. The selection is computed ONLY from persisted CheckpointRecord fields
   (validation_loss, created_at, checkpoint_id). NEVER load weights,
   NEVER re-evaluate, NEVER re-derive perplexity. Deterministic
   tie-break (validation_loss, created_at, checkpoint_id) — and expose
   ties honestly (a stable canonical pick among equals plus the fact
   that equals exist, never a silent judgment). Include the number of
   candidates considered. validation_loss is THE selection metric:
   perplexity is its monotone transform (exp(min(loss,100))) and adds
   nothing — do NOT add a metric axis.
3. Error/empty semantics reuse the existing taxonomy: unknown model ->
   404 (existing); a known model with ZERO checkpoints -> an honest
   empty (200 with a null/no-selection computed view + 0 candidates;
   follow the repo's single-computed-view conventions; do NOT invent a
   new error). Malformed ids -> existing validation.
4. NO auto-selection ANYWHERE: recipe/workflow state refs (checkpoint_id
   / from_stage / current) are UNCHANGED; the workflow never guesses;
   the model remains the only runtime binding; no execution semantics
   change in any engine. NO by-decision / by-metric grouping endpoint —
   the M50 §9 grouping-ladder closure stands; this milestone is a
   selection view, not a grouping.
5. Implementation discipline: minimum files (engine facade + one route +
   schema/view + README); the selection logic lives in the engine layer
   behind the existing facade pattern, NOT in the API layer; reuse the
   authoritative M3 checkpoint listing order as the candidate source.
6. Tests (~4-5, minimal but proving): selection equals the persisted
   argmin on a multi-checkpoint model (including a deliberate tie with
   the documented tie-break), zero-checkpoint honest empty, 404 unknown
   model, determinism (x2 byte-identical), OpenAPI 83 -> 84 with the
   new path once, GET-only; zero storage growth on every selection GET.
   Update the README counts.
7. Statics + gates in this order: focused tests -> FULL suite x2 ->
   python -m compileall -> pyflakes on every touched file -> OpenAPI
   sweep (update every len(spec["paths"]) == 83 assertion to 84).
8. Live smoke (read-only, M50-style, production FORGE_ROOT, own port):
   the production model 4a0a871886ef owns 3 checkpoints, all decision
   accept, persisted validation_loss 6.360189 / 6.210553 / 6.397286 —
   DISCOVER the selection live from the persisted manifests and the M3
   listing (do NOT assume the value from this prompt) and verify the
   endpoint returns the argmin (expected: 0511de4c7372, the checkpoint
   the production recipe m12-live-suite pins — the operator's manual
   choice confirmed by persisted evidence); zero-history model
   b5bc905326b6 -> honest empty; x3 byte-identical repeats; every M3-M51
   surface unchanged; ZERO storage growth; dashboard hash unchanged.
9. Final report with EXACTLY 9 sections, the last one selecting M53
   from architectural evidence (by-* groupings remain closed; weigh:
   SFT training-method differentiation, dataset-version tooling,
   sampling extensions — or honest exhaustion if nothing meets the bar),
   with a complete ready-to-copy M53 prompt if justified. Completion
   standard: never recompute a measurement, never auto-select, never
   reopen the grouping ladder, never certify from partial runs; report
   actual counts honestly.
```
