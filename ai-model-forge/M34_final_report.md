# MILESTONE 34 — GATE-DECISION HISTORY BY COMPARISON — FINAL REPORT

## 1. M34 Objective

Add exactly ONE small, read-only API capability for inspecting the
existing M6 gate-decision history grouped by the M5 comparison each
decision judged:

`GET /api/v1/models/{model_id}/gates/decisions/by-comparison/{comparison_id}`

* Membership comes from the persisted decision comparison identity on
  the authoritative M6 listing ONLY: every `GateDecision` carries a
  top-level `comparison_id` (the M5 record it judged — the M6 run
  persists it verbatim from the request), matched VERBATIM — never
  filenames, gate-directory names, checkpoint ids, policy ids, hashes,
  or re-derivation from the comparison's current content.
* Legacy direct-evaluation decisions keep `comparison_id=None`, belong
  to NO by-comparison group, and stay in the generic M6 listing
  untouched. Each matching decision appears EXACTLY ONCE; exact M6
  `(created_at, decision_id)` ordering inherited; verbatim
  `GateDecision` payloads (verdict, decision, losses, delta, reason
  included).
* The comparison is validated through the model's OWN M5 registry
  (`ComparisonEngine.get_comparison`): unknown comparison, or one
  belonging to another model, → 404 (comparisons are model-scoped).
  Valid comparison + zero decisions → `200 []`, never 404.
* Route registered after the M23 by-policy route and BEFORE the
  generic `/gates/decisions/{decision_id}` detail getter.
* Strictly read-only: no gate evaluation, no comparison/evaluation
  execution, no aggregation/ranking/new metrics, no manifest writes,
  no index/cache/db.

Implementation chain:
**M5 comparison ownership → M6 authoritative decisions → persisted
`comparison_id` filter → gate engine → facade → API**.

## 2. Baseline

A **sandbox reset (#7)** was detected at turn start (HEAD reverted to
base `f86b670`, production root + venv wiped). The standard recovery
runbook was executed BEFORE any M34 work: git reconciled to the
authoritative remote (`reset --hard 404d120` — my own pushed M33
commits; no force-push), production restored from `code forge.zip`
(96 files / 4,002,745 B / 0 tmp), `sha256sum -c m33_pre.sha256` →
**96/96 OK**, venv rebuilt from `pyproject.toml`
(`pip install -e .[dev]` + pyflakes; torch 2.14.0+cu130, fastapi
0.141.1, pydantic 2.13.5), and the certified M33 baseline re-proven
end-to-end. Actual verified baseline before implementation:

* Commit: HEAD **`404d120`** == FETCH_HEAD (`arena/01a071e9-code-forge`),
  log `404d120` (M33 inventory) → `c53cc4f` (M33) → `9fb6630`;
  tree clean except `ai_model_forge.egg-info/`.
* Tests: **475 passed** (98.12 s on the fresh venv).
* OpenAPI: **65 paths**.
* Production: **96 files / 4,002,745 B / 0 tmp**, inventory/hash
  **96/96 OK** (0 drift vs certified M33).
* `m34_pre.sha256` written (96 entries); diff vs `m33_pre.sha256`:
  **EMPTY** (M31–M33 were strictly read-only, so pre == certified
  final).

## 3. Implementation

Exact files changed: `app/gates.py`, `app/engine.py`, `app/api.py`,
`README.md`, `tests/test_gates.py`, `tests/test_gates_api.py` (+6
test modules' OpenAPI-count comments/assertions),
`smoke_m34_live.py`, `M34_final_report.md`.

* **Engine** (`app/gates.py`):
  `list_decisions_for_comparison(model_id, comparison_id)` directly
  after the M23 method (mirroring its exact structure): the
  authoritative `list_decisions(model_id)` validates the model
  (unknown → `FileNotFoundError`), `self.comparison.get_comparison(
  model_id, comparison_id)` — the handle ALREADY composed in
  `GateEngine.__init__` (the M6 run uses the same handle; **zero new
  composition lines**) — validates M5 ownership (unknown or
  other-model → `FileNotFoundError`), then the listing is filtered by
  the persisted top-level `comparison_id` (`None` never matches).
  Full docstring documents grouping unit, VERBATIM matching,
  null-group exclusion, model-scoped ownership, ordering, read-only
  contract.
* **Facade** (`app/engine.py`):
  `list_gate_decisions_for_comparison` — thin delegation, no business
  logic.
* **API** (`app/api.py`): ONE GET route, tag `gates`,
  `response_model=list[GateDecision]`, 404 mapping identical to the
  M23 sibling; registered **after** M23 by-policy and **before** the
  generic `/gates/decisions/{decision_id}` getter (verified: M23 <
  by-comparison < generic). Landing-page M34 bullet + endpoint `<li>`
  + gate-routes header comment.
* **Documentation**: README M34 section (read-only behavior, model
  scoping, comparison-ownership validation, persisted-identity
  filtering, deterministic ordering, no writes) + counts 475→480 (2
  sites) + `gates.py` layout line `(M23/M34)`.

## 4. Tests

* Focused: engine **3/3** first run (`test_m34_engine_*`),
  API **2/2** first run (`test_m34_api_*`); gate modules together
  **44/44**.
* Full suite: **480 passed** (109.06 s) — exactly the predicted
  475 + 5.
* Repeated full suite (after stale-`/tmp` cleanup): **480 passed**
  (97.65 s).
* compileall: clean (app, tests, `smoke_m34_live.py`).
* pyflakes: **0 new findings** (pre-existing legacy: 2, confined to
  `smoke_m9_live.py`/`smoke_m11_live.py`). One honest fix during
  gates: an unused variable in the first smoke draft (`known_group`)
  was removed before any smoke run; engine tests needed one
  fixture-name correction (`_m23_env`, not `_m23_state`) caught by
  pyflakes before the first run.
* Coverage (engine): parity with `list_decisions()` filtered by
  persisted id; ordering `(created_at, decision_id)`; verbatim
  `get_gate_decision` parity; evidence-reuse (two identical gate runs
  → ONE comparison holding exactly both decisions); discovery-based
  partition (groups over every non-null id cover non-null decisions
  exactly once); fresh zero-decision comparisons → `[]`; unknown
  model/comparison → `FileNotFoundError`; cross-model (other model's
  id 404, listings disjoint); null decisions in NO group; ×3
  identical repeats; gate-file set byte-identical (read-only).
* Coverage (API): grouping/determinism/empty (incl. fresh M5
  comparisons → `200 []`, listing length stable), 404s (unknown
  model, ghost + malformed comparison, combined), cross-model 404
  via a second real model, M6 listing/getter + M23 by-policy
  regression (registered policy group + ghost policy 404), M31
  comparisons-by-tokenizer parity regression, generic ghost decision
  id 404 (no route capture), OpenAPI 66 + route order + GET-only +
  schema.

## 5. API / OpenAPI

* Endpoint:
  `GET /api/v1/models/{model_id}/gates/decisions/by-comparison/{comparison_id}`
  — appears **exactly once**, GET-only, tag `gates`, items
  `$ref GateDecision`.
* Route ordering verified (standalone + live): M23
  `by-policy/{policy_id}` < **by-comparison** < generic
  `gates/decisions/{decision_id}` — `by-comparison` can never be
  captured as a decision id, and the generic getter still 404s ghost
  ids.
* OpenAPI **65 → 66**; all 18 genuine `len(spec["paths"])` assertions
  updated with comment ladders (`+ 1 (M34 gate decisions
  by-comparison)`); unrelated `== 65` occurrences (hash lengths etc.)
  inspected and untouched; grep-verified 0 stale `== 65` remain.
* Semantics: known comparison → `200` + exact records; valid
  zero-decision comparison → `200 []`; unknown model → `404`; unknown
  or malformed comparison → `404`; other-model comparison → `404`;
  null-comparison records never returned.

## 6. Live Smoke

`smoke_m34_live.py` on port **8755**: **33/33 PASS first run**;
repeated 3× (exit 0 each). Server started fresh and stopped after
(log clean).

* Distribution DISCOVERED live (not assumed): 11 decisions of
  `4a0a871886ef` → **fc379bfcb50f → 4, d62f89e97c85 → 2,
  786de08efe4c → 1, 5c5ff22151ed → 1, d9a62dde016b → 1, null → 2**;
  zero-decision comparisons discovered = {baa361012e00, d683f9b81195,
  729f9c55ea89}.
* Known comparison: exact 4 records, parity with the filtered M6
  listing, verbatim detail-getter payloads, ASC order, every decision
  exactly once.
* Repeatability: three GETs raw-byte-identical.
* Empty: all 3 zero-decision comparisons → `200 []`.
* Unknown ids: unknown model 404 (with real comparison id); ghost +
  malformed comparison 404.
* Cross-model: real comparison id under `b5bc905326b6` → 404 (it has
  0 decisions).
* Null IDs: null decisions in NO group; groups partition the 9
  non-null decisions exactly.
* M23 regression: `m9-live-policy` → the exact 1 decision, parity
  with the filtered listing.
* M24–M33 regressions: evaluations 3/3/3 + 16 by-dataset + 16
  by-tokenizer; suite runs 10 + summary; comparisons 6/5/1 + 8
  by-dataset + 8 by-tokenizer; samples 4/0/0 + 4 by-tokenizer;
  sample-quality 2 by-tokenizer; M6 listing + all 11 detail getters
  verbatim.
* Dashboard hash unchanged; registries (M2/M3/M9/M11/M12/M14)
  unchanged; OpenAPI exactly 66 with correct route order.

## 7. Storage / Integrity

* Final production state: **96 files / 4,002,745 bytes / 0 tmp** —
  identical before/after.
* Smoke-internal per-file SHA256: **0 changed / 0 missing / 0 new**.
* Final audit `sha256sum -c m34_pre.sha256` from the data root:
  **96/96 OK, 0 non-OK**. M34 wrote NOTHING to production (strictly
  read-only; all test artifacts went to pytest tmp dirs).

## 8. Git / Certification

* Implementation commit: **`M34: add gate decision history by
  comparison`**; follow-up inventory commit (convention M21–M33):
  `M34: add pre-milestone storage inventory` tracking
  `m34_pre.sha256`.
* Branch `arena/01a071e9-code-forge`; pushed; after both commits
  **local HEAD == remote HEAD (FETCH_HEAD)**; working tree clean
  except expected `ai_model_forge.egg-info/`. No force-push used.
* Certification verdict: **M34 is CERTIFIED COMPLETE** — every
  checklist item verified (baseline 475/OpenAPI 65 pre-change; engine
  method reusing M6 listing + M5 ownership; verbatim persisted-id
  filtering incl. null exclusion; thin facade; exactly one GET route
  in correct order; all semantics; 480 ×2 full suites; compileall;
  0 new static findings; OpenAPI 66; 33/33 live smoke ×3; M23–M33
  regressions; dashboard/registries unchanged; storage + SHA256
  audits clean; committed + pushed + local == remote).

## 9. NEXT MILESTONE — M35

Selection was made from live inspection of the completed M34
codebase: the by-tokenizer family is exhausted (M30–M33 covered
evaluations/comparisons/samples/sample-quality), gate decisions now
have both persisted groupings (policy M23, comparison M34), and suite
runs keep dataset/tokenizer identity only per-probe (no top-level
field). The genuine remaining persisted identity is the M11 workflow
record's top-level `recipe_id`: model `4a0a871886ef` owns 13
workflows in `(created_at, workflow_id)` ASC order — 8 ad-hoc runs
with `recipe_id=None`, `m12-live-suite` → 2, `m14-comp` → 2,
`m12-live-ghost` → 1 — while the M12/M14 recipe registry holds 7
recipes, 4 of which (`m14-base`, `m14-chain-a/b/c`) have ZERO runs
for the model (natural `200 []` cases) and `b5bc905326b6` has no
workflows at all. The global M12 surface
`GET /workflows/recipes/{recipe_id}/runs` already exists — M35 is the
MODEL-SCOPED projection of the same idea, following the exact M23–M34
pattern.

```
# MILESTONE 35 — WORKFLOW HISTORY BY RECIPE (MODEL-SCOPED)

Continue the existing **AI Model Forge** project.

M34 is the current certified milestone.

The goal of M35 is to add one small, read-only API capability for
inspecting the existing M11 workflow history of ONE model grouped by
the M12/M14 recipe it was executed from.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M34 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 480 tests passing
* OpenAPI: 66 paths
* M34 gate-decision by-comparison endpoint working
* M33/M32/M31/M30 by-tokenizer endpoints working
* M29/M28/M27/M26/M25/M24 history endpoints working
* M23 by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working

* ONE read-only endpoint; ONE engine method; ONE facade; ONE route;
  at most ONE new `__init__` composition line (only if the workflow
  engine lacks a recipe-registry handle — inspect first; M12 recipe
  execution already resolves recipes, so a handle likely exists).
* No new workflow/recipe execution, no aggregation/ranking/scoring,
  no training/rollback/optimization/HPO, no workers, databases,
  caches, indexes, no new persistence, no automatic decisions, no
  dashboard redesign, no Gemini.
* Do not modify production data to manufacture fixtures; the empty
  cases come from the REAL production state (verified at M34: the
  registered recipes m14-base, m14-chain-a, m14-chain-b, m14-chain-c
  have ZERO workflows for 4a0a871886ef, and b5bc905326b6 has no
  workflows at all).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (66 -> 67) and their adjacent stale comments; inspect
  every hard-coded `== 66` and leave unrelated ones (hash lengths
  etc.) untouched.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 2. M35 OBJECTIVE

Add exactly ONE read-only endpoint:

`GET /api/v1/models/{model_id}/workflows/by-recipe/{recipe_id}`

answering "which immutable M11 workflow runs of this model were
executed from this registered recipe?" and nothing else.

Membership rules (identical philosophy to M23/M34):

* The authoritative source is the model's OWN M11 workflow listing
  (`list_workflows(model_id)`, the same one `GET /models/{id}/
  workflows` exposes); never re-derive from filenames, paths, plan
  hashes, statuses or the recipe's current definition.
* Each persisted WorkflowRecord carries a top-level `recipe_id` plus
  its matching `recipe_hash` provenance (M12). Match the requested id
  against the persisted `recipe_id` VERBATIM; ad-hoc runs keep
  `recipe_id=None`, belong to NO by-recipe group, and must NOT be
  dropped from the generic listing. Preserve `recipe_hash` exactly;
  never re-derive it.
* Each workflow appears EXACTLY ONCE.
* Inherit the exact existing M11 ordering (verified at M34:
  deterministic (created_at, workflow_id) ASCENDING; no new sort
  rule).
* Return verbatim WorkflowRecord payloads (status, stages,
  transitions, result_hash included).
* Recipes are GLOBAL: validate the recipe through the existing M12/
  M14 recipe registry first (the same resolution
  `GET /workflows/recipes/{recipe_id}` uses; unknown recipe -> 404);
  model scoping comes from the model's own M11 listing (a model never
  sees another model's runs). Unknown model -> 404.
* A VALID registered recipe with zero runs for the model -> `200 []`
  (NEVER 404).
* Register the route BEFORE any generic route that could interpret
  "by-recipe" as a dynamic id (inspect where `GET /models/{id}/
  workflows/{workflow_id}` sits, following the M23/M34 pattern); the
  M11 generic listing/getter and the GLOBAL M12
  `/workflows/recipes/{recipe_id}/runs` surface must stay intact and
  unconfused.
* Strictly read-only: no workflow/recipe execution, no new metrics,
  no manifest writes, no index/cache/db/duplicated storage.
* Required chain: M12/M14 recipe registry validation -> M11
  authoritative workflow listing -> persisted recipe_id filter ->
  thin facade -> route.

## 3. PRODUCTION FACTS (verified at M34)

* Registry recipes (7): m12-live-suite, m12-live-ghost, m14-base,
  m14-comp, m14-chain-c, m14-chain-b, m14-chain-a.
* Model `4a0a871886ef` owns 13 workflows in ASCENDING (created_at,
  workflow_id) order; distribution by persisted recipe_id: None -> 8
  (ad-hoc), m12-live-suite -> 2, m14-comp -> 2, m12-live-ghost -> 1.
* The 4 registered recipes m14-base, m14-chain-a, m14-chain-b,
  m14-chain-c have ZERO workflows for the model (natural valid-id
  empty cases -> 200 []).
* Model `b5bc905326b6` exists with NO workflows (model-scoped empty
  case for any valid recipe).
* The global M12 surface GET /workflows/recipes/{recipe_id}/runs
  lists 2 runs for m12-live-suite across the registry; ghost recipe
  -> FileNotFoundError (the established 404 contract).
* Dashboard hash (unchanged since M17, re-verified at M34):
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838.
* Storage: 96 files / 4,002,745 B / 0 `.tmp`.

## 4. REQUIRED BEHAVIOR

1. `200` with the exact filtered array for the known pair
   (`4a0a871886ef` + `m12-live-suite` -> its 2 runs; discover the
   exact ids).
2. Exact parity with the authoritative M11 listing filtered locally
   by persisted `recipe_id` (no missing, no extra, no duplicate).
3. Persisted identity + `recipe_hash` provenance travel VERBATIM (no
   substitution/rewriting); `None`-recipe runs never appear.
4. Authoritative (created_at, workflow_id) ordering preserved.
5. Deterministic: repeated GETs byte-identical (x3).
6. Valid registered recipe + zero runs for the model -> `200 []`
   (m14-base and the chain recipes).
7. Unknown model -> 404; unknown recipe -> 404 (well-formed AND
   malformed ids).
8. Cross-model isolation: tokenizers/recipes are global, but each
   model's group draws only from its own listing (b5bc905326b6 ->
   200 [] for every valid recipe).
9. M11 run/listing/getter, the global M12 recipe-runs surface, M23/
   M34 gate groupings, and the M24-M33 history surfaces byte-identical
   before and after.
10. Zero production storage growth.

## 5. ENGINE IMPLEMENTATION

* Inspect the workflow engine FIRST (its `__init__` handles, its
  listing method, and how M12 recipe execution resolves recipes —
  `get_workflow_recipe`).
* Add ONE method `list_workflows_for_recipe(model_id, recipe_id)`
  that validates the recipe through the existing registry handle
  (compose ONE line in `__init__` only if missing) and filters the
  authoritative listing by the persisted top-level `recipe_id`; the
  facade and route are thin pass-throughs.
* Docstring must explain the grouping unit, VERBATIM persisted-identity
  matching, the ad-hoc (None) exclusion, global-recipe/model-scoped
  semantics, and the read-only contract.

## 6. TESTS

* Target: 480 -> 485 tests (~5 new), in the existing workflow test
  modules (engine + API), REUSING their fixtures/helpers (never
  modify production; never inflate the corpus; reuse the registered
  recipe fixtures).
* Cover: authoritative-filter parity; verbatim identity (incl.
  recipe_hash provenance); exclusion of unrelated recipes; ad-hoc
  (None) exclusion; empty `[]` for a valid registered recipe with no
  runs; 404s (unknown model / unknown + malformed recipe id);
  cross-model isolation; ordering; determinism; no storage writes;
  M11/M12 regressions (generic listing/getter + the global
  recipe-runs surface); OpenAPI 66 -> 67 + route order + ghost id
  404 on the generic getter.
* Search EVERY hard-coded OpenAPI count (grep `== 66` and
  `len(spec["paths"])`) and update only genuinely affected assertions
  + adjacent stale comments.
* Gates in order: baseline tests -> focused -> full suite -> full
  suite again after stale `/tmp/forge-tests-*` cleanup (verify no
  process first) -> compileall (incl. the new smoke) -> pyflakes
  (report pre-existing findings separately) -> OpenAPI verification
  -> live smoke -> final SHA256 audit; commit only after all gates
  pass.

## 7. LIVE TEST

Write `smoke_m35_live.py` (mirror `smoke_m34_live.py`), port
**8756**, baseline inventory
`/tmp/m35-smoke-baseline-inventory.json`; capture the baseline FIRST
(files/bytes/tmp/SHA256/OpenAPI/test count/registries/workflow
distribution by recipe), then checks: A known pair 200 + exact
records; B discover ids; C listing-parity + verbatim payloads; D
ordering; E >= 3 byte-identical repeats; F empty `200 []` for the 4
zero-run recipes + b5bc905326b6; G unknown model 404; H unknown +
malformed recipe 404; I ad-hoc (None) runs in NO group + groups
partition the recipe-attributed runs; J M12 global recipe-runs
surface + recipe registry unchanged; K M11 listing/getters unchanged;
L M23/M34 gate groupings unchanged; M M24-M33 history regressions; N
dashboard + policy/probe/suite/recipe registries; O OpenAPI 67 with
route order; P storage zero drift (96/4,002,745/0, SHA256 before/
after 0 changed/new/missing; investigate + report ANY change
honestly).

## 8. FINAL AUDIT + RECOVERY

* `sha256sum -c` the pre-milestone inventory from the data root:
  96/96 OK, 0 changed/new/missing.
* Recovery runbook (sandbox resets have occurred between turns):
  verify HEAD/FETCH_HEAD first; restore production from
  `code forge.zip` -> verify against the correct pre-milestone
  inventory -> rebuild venv (`python3 -m venv ~/.venv && pip install
  -e ai-model-forge[dev] pyflakes`) -> reconcile git with the
  authoritative remote (`git fetch` + `git reset --hard FETCH_HEAD`,
  never force-push) -> re-prove the M34 baseline -> only then begin
  M35.

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; no execution, no scoring, no rankings, no
  automatic decisions, no schema change, no duplication of the
  authoritative listing, no caches/indexes/databases/workers/queues,
  no training/rollback/optimization/HPO/Gemini, no unrelated
  refactoring.
* Report honestly; never certify from a partial run.

## 10. COMMIT / PUSH / REPORT

* Commit message: `M35: add workflow history by recipe`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree (inventory `m35_pre.sha256` tracked in a follow-up commit per
  the M21-M34 convention).
* Write `M35_final_report.md` with EXACTLY 9 sections (objective,
  baseline, implementation, tests, API/OpenAPI, live smoke, storage/
  integrity, git/certification, next milestone) with EXACT measured
  values (workflow ids, per-recipe counts, ordering, determinism,
  404s, cross-model/scoping, SHA256, dashboard hash, commit/branch/
  sync/tree).
* Section 9 must contain the COMPLETE copy-ready M36 prompt, grounded
  ONLY in facts discovered and verified during M35 (no invented ids,
  counts, or endpoints), defining the next small additive read-only
  capability, including automatic M37 prompt generation.
```
