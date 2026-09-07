# MILESTONE 35 — MODEL-SCOPED WORKFLOW HISTORY BY RECIPE — FINAL REPORT

## 1. M35 Objective

Add exactly ONE small, read-only, MODEL-SCOPED API capability for
inspecting the existing M11 workflow history grouped by the M12/M14
recipe each run was executed from:

`GET /api/v1/models/{model_id}/workflows/by-recipe/{recipe_id}`

* Membership comes from the persisted run recipe identity on the
  authoritative M11 listing ONLY: every `WorkflowRecord` carries a
  top-level `recipe_id` plus its matching `recipe_hash` provenance
  (M12 records both verbatim when a registered recipe is executed),
  matched VERBATIM — never filenames, paths, stage ids, stage
  contents, statuses, recipe hashes alone, or the recipe's current
  definition. The recorded provenance is preserved exactly and never
  re-derived.
* Ad-hoc runs keep `recipe_id=None`, belong to NO by-recipe group,
  and stay in the generic M11 listing untouched (no `by-ad-hoc`
  pseudo-recipe exists). Each matching run appears EXACTLY ONCE;
  exact M11 `(created_at, workflow_id)` ordering inherited; verbatim
  `WorkflowRecord` payloads (status, stages, transitions,
  `result_hash` included).
* The recipe is validated through the GLOBAL M12/M14 registry
  (`RecipeEngine.get` — the same resolution
  `GET /workflows/recipes/{recipe_id}` uses; unknown recipe → 404 —
  never `[]`); model scoping comes from the model's own M11 listing
  (a model never sees another model's runs; a valid recipe never
  makes an unknown model valid). Valid registered recipe + zero runs
  → `200 []`.
* This is the model-scoped complement of the GLOBAL M12 cross-model
  `/workflows/recipes/{recipe_id}/runs` lineage surface, which stays
  unchanged (verified as a regression).
* Route registered after the M11 listing route and BEFORE the generic
  `/workflows/{workflow_id}` detail getter. Strictly read-only: no
  execution, no recipe expansion, no aggregation, no writes.

Implementation chain:
**M7 authoritative workflow registry/listing → M12 recipe validation
→ persisted `WorkflowRecord.recipe_id` filter → workflow engine →
facade → API route**.

## 2. Baseline

Verified at turn start — **no sandbox reset this time** (HEAD
`b7ea51a` == FETCH_HEAD; the M34 implementation `9760975` + inventory
`b7ea51a` both present and pushed; production 96 files; venv OK).
Actual verified baseline before implementation:

* Commit: HEAD **`b7ea51a`** == FETCH_HEAD on
  `arena/01a071e9-code-forge`; tree clean except
  `ai_model_forge.egg-info/`.
* Tests: **480 passed** (86.17 s).
* OpenAPI: **66 paths**.
* Production: **96 files / 4,002,745 B / 0 tmp**; `m35_pre.sha256`
  written (96 entries); diff vs `m34_pre.sha256`: **EMPTY**;
  inventory/hash 96/96 OK (verified again in the final audit).

## 3. Implementation

Exact files changed: `app/workflows.py`, `app/engine.py`,
`app/api.py`, `README.md`, `tests/test_workflow_recipes.py` (+6 test
modules' OpenAPI-count comments/assertions), `smoke_m35_live.py`,
`M35_final_report.md`.

* **Engine** (`app/workflows.py`):
  `list_workflows_for_recipe(model_id, recipe_id)` directly after
  `get_workflow`: the authoritative `list_workflows(model_id)`
  validates the model (unknown → `FileNotFoundError`),
  `self.recipes.get(recipe_id)` validates the recipe through the
  GLOBAL M12/M14 registry (unknown → `FileNotFoundError`), then the
  listing is filtered by the persisted top-level `recipe_id` (`None`
  never matches). Full docstring documents the grouping unit,
  VERBATIM matching + provenance preservation, ad-hoc exclusion,
  global-recipe/model-scoped semantics, ordering, read-only/no-
  execution contract.
* **Composition** (ONE new line in `WorkflowEngine.__init__`):
  `self.recipes = RecipeEngine(storage, workflows=self)` — the
  import is function-local because `recipes.py` imports
  `WorkflowEngine` at module level (circular otherwise), and
  injecting `self` keeps THIS engine the sole workflow executor (no
  second `WorkflowEngine` is constructed inside the recipe engine).
* **Facade** (`app/engine.py`):
  `list_workflows_for_recipe` — thin delegation, no business logic.
* **API** (`app/api.py`): ONE GET route, tag `workflows`,
  `response_model=list[WorkflowRecord]`, 404 mapping identical to
  the M11 siblings; registered **after** the M11 listing route and
  **before** the generic `/workflows/{workflow_id}` getter. Landing-
  page M35 bullet + endpoint `<li>` + workflow-routes header
  comment.
* **Documentation**: README M35 section (read-only, model-scoped,
  M12 validation, persisted-identity filtering, ordering, `200 []`
  vs 404 semantics, no storage writes) + counts 480→485 (2 sites) +
  `workflows.py` layout line `(M35)`.

## 4. Tests

* Focused: **5/5 first run** (3 engine `test_m35_engine_*` + 2 API
  `test_m35_api_*` in `tests/test_workflow_recipes.py`); workflow
  modules together 83/83.
* Full suite: **485 passed** (93.04 s) — exactly the predicted
  480 + 5.
* Second full suite (after stale-`/tmp` cleanup, 5 dirs, 0 active
  processes confirmed): **485 passed** (90.09 s).
* compileall: clean (app, tests, `smoke_m35_live.py`).
* pyflakes: **0 new findings** (pre-existing legacy: 2, confined to
  `smoke_m9_live.py`/`smoke_m11_live.py`). One honest fix: an unused
  `suite2` unpacking in the first API-test draft was flagged by
  pyflakes and removed before the first run.
* Coverage (engine): parity with `list_workflows()` filtered by
  persisted id; `(created_at, workflow_id)` ordering; verbatim
  `get_workflow` parity AND `recipe_hash` provenance == the
  registered definition's `config_hash`; discovery-based partition
  (groups over every non-null recipe cover exactly the attributed
  runs); registered-never-run recipe → `[]`; fresh model → `[]` for
  every recipe (model-scoped empty + disjoint listings); unknown
  model/recipe → `FileNotFoundError`; ×3 identical repeats; workflow
  AND recipe manifest bytes unchanged (read-only).
* Coverage (API): grouping/determinism/empty; 404s (unknown model,
  ghost + malformed recipe, combined — a valid recipe never makes an
  unknown model valid); cross-model `200 []` via a second real
  model; M11 listing/getter + generic ghost workflow id 404 (no
  route capture); M12 regression (GLOBAL recipe-runs surface returns
  the same record incl. `model_id`, unknown 404); M34 regression
  (by-comparison ghost 404); OpenAPI 67 + route order + GET-only +
  schema.

## 5. API / OpenAPI

* Endpoint:
  `GET /api/v1/models/{model_id}/workflows/by-recipe/{recipe_id}` —
  appears **exactly once**, GET-only, tag `workflows`, items `$ref
  WorkflowRecord`.
* Route placement verified (standalone + live): M11 listing
  `/models/{id}/workflows` < **by-recipe** < generic
  `/models/{id}/workflows/{workflow_id}` — `by-recipe` can never be
  captured as a workflow id; the GLOBAL M12
  `/workflows/recipes/{recipe_id}/runs` route still present exactly
  once (no duplicate registration).
* OpenAPI **66 → 67**; all 19 genuine `len(spec["paths"])` assertion
  sites updated with comment ladders (`+ 1 (M35 workflows
  by-recipe)`); unrelated `== 66` occurrences inspected and left
  untouched; grep-verified 0 stale path-count `== 66` remain.
* Semantics: known recipe → `200` + exact runs; valid zero-run
  recipe → `200 []`; unknown model → `404`; unknown recipe → `404`;
  cross-model → `200 []` (never another model's records); ad-hoc
  (`null`) records never returned.

## 6. Live Smoke

`smoke_m35_live.py` on port **8756**: **33/33 PASS first run**;
repeated 3× (exit 0 each). Server started fresh and stopped after
(log clean).

* Distribution DISCOVERED live (not assumed): 13 workflows of
  `4a0a871886ef` → **None → 8, m12-live-suite → 2, m14-comp → 2,
  m12-live-ghost → 1**; recipe registry 7; zero-run recipes
  discovered = {m14-base, m14-chain-a, m14-chain-b, m14-chain-c}.
* Known recipe (m12-live-suite): exact 2 records, parity with the
  filtered M11 listing, verbatim detail-getter payloads,
  `recipe_hash` provenance == registry `config_hash`, ASC order,
  every run exactly once.
* Repeatability: three GETs raw-byte-identical.
* Valid empty: all 4 zero-run recipes → `200 []`.
* Unknown ids: unknown model 404 (with the real recipe id); ghost +
  malformed recipe 404.
* Cross-model: known recipe under `b5bc905326b6` → `200 []`.
* Ad-hoc isolation: null runs in NO group; groups partition the 5
  attributed runs exactly.
* Ordering: every group preserves the authoritative order.
* M12 regression: GLOBAL recipe-runs surface unchanged (2 runs, same
  records incl. `model_id`, unknown 404) + registry list verbatim.
* M34 regression: by-comparison (fc379bfcb50f → 4) + M23 by-policy
  (m9-live-policy → 1) unchanged.
* M24–M33 regressions: evaluations 3/3/3 + 16 by-dataset + 16
  by-tokenizer; suite runs 10 + summary; comparisons 6/5/1 + 8
  by-dataset + 8 by-tokenizer; samples 4/0/0 + 4 by-tokenizer;
  sample-quality 2 by-tokenizer; M11 listing still deterministic.
* Dashboard hash unchanged; M2/M9/M12/M14 registries unchanged;
  OpenAPI exactly 67 with correct route order.

## 7. Storage / Integrity

* Final production state: **96 files / 4,002,745 bytes / 0 tmp** —
  identical before/after.
* Smoke-internal per-file SHA256: **0 changed / 0 missing / 0 new**.
* Final audit `sha256sum -c m35_pre.sha256` from the data root:
  **96/96 OK, 0 non-OK** (0 changed / 0 new / 0 missing). M35 wrote
  NOTHING to production (strictly read-only; all test artifacts went
  to pytest tmp dirs).

## 8. Git / Certification

* Implementation commit: **`M35: add workflow history by recipe`**;
  follow-up inventory commit (convention M21–M34):
  `M35: add pre-milestone storage inventory` tracking
  `m35_pre.sha256`.
* Branch `arena/01a071e9-code-forge`; pushed; after both commits
  **local HEAD == remote HEAD (FETCH_HEAD)**; working tree clean
  except expected `ai_model_forge.egg-info/`. No force-push used.
* Certification verdict: **M35 is CERTIFIED COMPLETE** — every
  checklist item verified (baseline 480/OpenAPI 66 pre-change;
  inventory verified; engine method reusing the authoritative M7
  listing + M12 recipe validation; verbatim persisted-id filtering
  incl. ad-hoc exclusion; model ownership + recipe validity
  enforced; thin facade; exactly one GET route in correct order; all
  semantics; 485 ×2 full suites; compileall; 0 new static findings;
  OpenAPI 67; 33/33 live smoke ×3; M12 recipe-run + M23–M34
  regressions; dashboard/registries unchanged; storage + SHA256
  audits clean; committed + pushed + local == remote).

## 9. NEXT MILESTONE — M36

Selection was made from live inspection at the end of M35. The
history families now fully covered: evaluations (checkpoint/dataset/
tokenizer), comparisons (checkpoint/dataset/tokenizer), gate
decisions (policy/comparison), samples (checkpoint/tokenizer),
sample-quality (sample/checkpoint/tokenizer), workflows (recipe,
model-scoped + the global M12 lineage), suite runs (suite/checkpoint
+ summary). The remaining genuine persisted top-level identity with
natural non-empty AND empty cases is the M4 evaluation's
**`split`** (`EvaluationSplit` enum): production model
`4a0a871886ef` owns 16 evaluations in `(created_at, eval_id)` ASC
order grouped **validation → 14, train → 2** (both groups non-empty
— verified live at M35), `b5bc905326b6` has none, and the split
value is persisted verbatim on every `EvaluationRecord`. Because a
split is a schema enum (not a registry), its unknown-value contract
is 422 (schema-level), unlike the registry 404s of M30–M35 — the
prompt states this precisely.

```
# MILESTONE 36 — EVALUATION HISTORY BY SPLIT

Continue the existing **AI Model Forge** project.

M35 is the current certified milestone.

The goal of M36 is to add one small, read-only API capability for
inspecting the existing M4 evaluation history grouped by the dataset
split each evaluation measured.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M35 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 485 tests passing
* OpenAPI: 67 paths
* M35 workflows by-recipe endpoint working
* M34 gate-decision by-comparison endpoint working
* M33/M32/M31/M30 by-tokenizer endpoints working
* M29/M28/M27/M26/M25/M24 history endpoints working
* M23 by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working

* If a sandbox reset occurred, recover first: verify HEAD/FETCH_HEAD,
  restore production from `code forge.zip`, verify against the
  correct pre-milestone inventory, rebuild the venv
  (`python3 -m venv ~/.venv && ~/.venv/bin/pip install -e
  ai-model-forge[dev] pyflakes`), reconcile git with the
  authoritative remote (never force-push), re-prove the M35 baseline.
* ONE read-only endpoint; ONE engine method; ONE facade; ONE route;
  ZERO new `__init__` composition lines (the evaluation engine
  already owns everything it needs — inspect first).
* No new evaluations, no training/rollback/optimization/HPO, no
  workers, databases, caches, indexes, no new persistence, no
  automatic decisions, no dashboard redesign, no Gemini.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the real `b5bc905326b6` (valid model, empty
  history).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (67 -> 68) and their adjacent stale comments; inspect
  every hard-coded `== 67` and leave unrelated ones untouched.
* Verify the baseline BEFORE changing code; report honestly at all
  times.

---

## 2. M36 OBJECTIVE

Add exactly ONE read-only endpoint:

`GET /api/v1/models/{model_id}/evaluations/by-split/{split}`

answering "which immutable M4 evaluations of this model measured this
dataset split?" and nothing else.

Membership rules (identical philosophy to M24/M28/M30-M35):

* The authoritative source is the model's OWN M4 listing
  (`list_evaluations(model_id)` — the same one `GET /models/{id}/
  evaluations` exposes); never re-derive from filenames, paths,
  dataset ids, tokenizers, checkpoints or hashes.
* Each persisted EvaluationRecord carries a top-level `split`
  (EvaluationSplit enum, persisted verbatim: 'train' or
  'validation'). Match it VERBATIM against the requested split.
* Each evaluation appears EXACTLY ONCE.
* Inherit the exact existing M4 ordering (verified at M35:
  deterministic (created_at, eval_id) ASCENDING; no new sort rule).
* Return verbatim EvaluationRecord payloads (loss/perplexity/
  token counts included).
* The model is validated through the existing contract (unknown
  model -> 404). Splits have NO registry (they are a schema enum),
  so the unknown-value contract is SCHEMA-LEVEL: an unsupported
  split value -> 422 (FastAPI path-param validation via the
  EvaluationSplit enum), NEVER a 404 and NEVER an empty 200 — this
  is the documented deviation from the registry-404 pattern and must
  be stated in the route docstring.
* A valid split with zero evaluations for the model -> `200 []`
  (NEVER 404).
* Register the route with the established prefix-group ordering:
  after the other /evaluations/by-* routes (or directly beside them,
  following the M28/M30 registration order) and BEFORE the generic
  /evaluations/{eval_id} detail getter; M24/M28/M30 routes stay
  intact and unconfused.
* Strictly read-only: no evaluation execution, no new metrics, no
  manifest writes, no index/cache/db/duplicated storage.
* Required chain: M4 authoritative listing -> persisted split filter
  -> thin facade -> route (no registry validation needed — the enum
  IS the contract).

## 3. PRODUCTION FACTS (verified at M35)

* Model `4a0a871886ef` owns 16 evaluations in ASCENDING (created_at,
  eval_id) order; persisted split distribution: validation -> 14,
  train -> 2 (both groups non-empty; discover the exact ids live).
* All 16 measure dataset `ee1a716c4573` v1 with tokenizer
  `99106e3255c5` (M28/M30 grouping axes are orthogonal to split).
* Model `b5bc905326b6` exists with NO evaluations (natural
  model-scoped empty case -> `200 []` for either split).
* Dashboard hash (unchanged since M17, re-verified at M35):
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838.
* Storage: 96 files / 4,002,745 B / 0 `.tmp`.

## 4. REQUIRED BEHAVIOR

1. `200` with the exact filtered array for the known pair
   (`4a0a871886ef` + `validation` -> the 14 records; `train` -> the
   2 records; discover ids live).
2. Exact parity with the authoritative M4 listing filtered locally by
   persisted split (no missing, no extra, no duplicate).
3. Persisted identity travels VERBATIM (no substitution/rewriting).
4. Authoritative (created_at, eval_id) ordering preserved.
5. Deterministic: repeated GETs byte-identical (x3).
6. Valid split + zero evaluations for the model -> `200 []`
   (b5bc905326b6 for both splits).
7. Unknown model -> 404; unsupported split value -> 422 (well-formed
   AND malformed values, e.g. "test-set", "TEST", "train "); the
   422-vs-404 distinction from registry-based milestones must hold.
8. Cross-model isolation: each model's group draws only from its own
   listing.
9. M4 run/listing/getter, M24/M28/M30 evaluation groupings and the
   M26-M35 surfaces byte-identical before and after.
10. Zero production storage growth.

## 5. ENGINE IMPLEMENTATION

* Inspect the evaluation engine FIRST (its listing method and the
  M24/M28/M30 methods for the established pattern).
* Add ONE method `list_evaluations_for_split(model_id, split)` that
  validates the model through the authoritative listing (unknown ->
  FileNotFoundError) and filters by the persisted top-level `split`;
  the split parameter type is the EvaluationSplit enum (the API
  layer enforces 422 for unsupported values); the facade and route
  are thin pass-throughs.
* Docstring must explain the grouping unit, VERBATIM persisted-enum
  matching, the no-registry/422 contract, and the read-only
  guarantee.

## 6. TESTS

* Target: 485 -> 490 tests (~5 new), in the existing evaluation test
  modules, REUSING their fixtures/helpers (never modify production;
  never inflate the corpus).
* Cover: authoritative-filter parity for BOTH splits; verbatim
  identity; exclusion between splits (partition of the listing);
  empty `[]` for a valid split; 404 unknown model; 422 unsupported
  split values (multiple forms); cross-model isolation; ordering;
  determinism; no storage writes; M4/M24/M28/M30 regressions;
  OpenAPI 67 -> 68 + route order + ghost eval id 404.
* Search EVERY hard-coded OpenAPI count (grep `== 68` candidates and
  `len(spec["paths"])`) and update only genuinely affected assertions
  + adjacent stale comments.
* Gates in order: baseline tests -> focused -> full suite -> full
  suite again after stale `/tmp/forge-tests-*` cleanup (verify no
  process first) -> compileall (incl. the new smoke) -> pyflakes
  (report pre-existing findings separately) -> OpenAPI verification
  -> live smoke -> final SHA256 audit; commit only after all gates
  pass.

## 7. LIVE TEST

Write `smoke_m36_live.py` (mirror `smoke_m35_live.py`), port
**8757**, baseline inventory
`/tmp/m36-smoke-baseline-inventory.json`; capture the baseline FIRST
(files/bytes/tmp/SHA256/OpenAPI/test count/registries/evaluation
distribution by split), then checks: A known pair 200 + exact
records (validation 14; train 2 — discovered live); B discover ids;
C listing-parity + verbatim payloads + ordering; D >= 3
byte-identical repeats; E empty `200 []` for b5bc905326b6 on BOTH
splits; F unknown model 404; G unsupported split values 422
(well-formed + malformed, multiple forms); H split partition
(validation + train groups disjoint and covering the listing);
I M24/M28/M30 evaluation groupings unchanged; J M31-M35 regressions
(comparisons/samples/sample-quality by-tokenizer, gate by-comparison,
workflows by-recipe incl. the GLOBAL M12 surface); K dashboard +
registries; L OpenAPI 68 with route order; M storage zero drift
(96/4,002,745/0, SHA256 before/after 0 changed/new/missing;
investigate + report ANY change honestly).

## 8. FINAL AUDIT + RECOVERY

* `sha256sum -c` the pre-milestone inventory from the data root:
  96/96 OK, 0 changed/new/missing.
* Recovery runbook as in section 1; NEVER force-push.

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; no execution, no scoring, no rankings, no
  automatic decisions, no schema change (EvaluationSplit already
  exists), no duplication of the authoritative listing, no
  caches/indexes/databases/workers/queues, no
  training/rollback/optimization/HPO/Gemini, no unrelated
  refactoring.
* Report honestly; never certify from a partial run.

## 10. COMMIT / PUSH / REPORT

* Commit message: `M36: add evaluation history by split`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree (inventory `m36_pre.sha256` tracked in a follow-up commit per
  the M21-M35 convention).
* Write `M36_final_report.md` with EXACTLY 9 sections (objective,
  baseline, implementation, tests, API/OpenAPI, live smoke, storage/
  integrity, git/certification, next milestone) with EXACT measured
  values (eval ids, per-split counts, ordering, determinism, 404/422
  cases, cross-model/scoping, SHA256, dashboard hash, commit/branch/
  sync/tree).
* Section 9 must contain the COMPLETE copy-ready M37 prompt, grounded
  ONLY in facts discovered and verified during M36 (no invented ids,
  counts, or endpoints), defining the next small additive read-only
  capability, including automatic M38 prompt generation.
```
