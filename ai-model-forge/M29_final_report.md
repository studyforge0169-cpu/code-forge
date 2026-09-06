# MILESTONE 29 — COMPARISON HISTORY BY DATASET — FINAL REPORT

## 1. Objective

Add exactly ONE new read-only endpoint to the existing **AI Model Forge**:

`GET /api/v1/models/{model_id}/comparisons/by-dataset/{dataset_id}`

answering *"which immutable M5 comparisons of this model measured this
dataset?"* — the model's authoritative M5 listing filtered by the
**persisted shared-probe dataset identity**. Required behavior, all
delivered:

- Both-sides semantics resolved by INSPECTION (the prompt's key open
  question): a comparison is valid only when BOTH sides measure the
  SAME probe, so `ComparisonRecord` persists exactly ONE top-level
  `dataset_id: str` + `dataset_version: int` pair — the shared probe
  both sides measured by construction (M5 refuses cross-probe requests
  — different dataset/version/split/tokenizer/window/seed — with 422
  before any artifact exists; per-side dataset identities cannot
  occur; verified: `ComparisonSide` carries NO dataset fields).
- **Once per comparison**: every matching comparison — including
  same-checkpoint A=B records whose two sides both measure the
  requested dataset — appears EXACTLY ONCE (dedup by comparison
  identity; the listing holds each record once).
- **Version preservation**: every version of the dataset is returned,
  each with its persisted `dataset_version` VERBATIM (never collapsed,
  resolved to the latest, aliased or rewritten).
- Dataset validation through the M2 registry (`DatasetEngine.load_meta`
  — the same call M4's run preflight, M28 and this engine's own
  `__init__`-composed `self.datasets` use; unknown dataset → 404; no
  raw filesystem checks). Datasets are GLOBAL; model scoping comes
  from the model's own M5 listing.
- Reuse the authoritative M5 ordering `(created_at, comparison_id)`
  ASCENDING; verbatim `ComparisonRecord` payloads (verdict/per-side
  losses included); no aggregates/rankings/leaderboards.
- Valid dataset with no comparisons for the model → `200 + []`.
- Route registered AFTER the M26 by-checkpoint route and BEFORE the
  generic `/comparisons/{comparison_id}` getter.
- Read-only: no training/rollback/HPO/ranking/new comparison metrics/
  automatic decisions/workers/databases/caches/indexes/analytics
  storage/new persistence/Gemini/dashboard redesign. No second
  comparison or dataset registry; no dataset-comparison index.

## 2. Baseline

Verified BEFORE any code change (the sandbox reset between milestones
again — the documented recovery runbook was executed FIRST: production
data restored from the repo-root `code forge.zip` and verified against
`m28_pre.sha256` (96 OK / 0 failed); venv rebuilt; git fetched, tree
proven hash-identical, `reset --hard FETCH_HEAD` at `321ebf1`; never
force-pushed):

- HEAD = **`321ebf1`** (`M28: add evaluation history by dataset`),
  branch `arena/01a071e9-code-forge`, local HEAD == remote HEAD
  (FETCH_HEAD-verified), working tree clean except the untracked
  `ai-model-forge/ai_model_forge.egg-info/` build artifact (not
  committed).
- Full suite serially: **450 passed** (106.10 s); OpenAPI **60 paths**
  (new path absent); production **96 files / 4,002,745 bytes /
  0 `.tmp`**; `m29_pre.sha256` created; diff vs `m28_pre.sha256` →
  **empty**.
- Inspection confirmed: `ComparisonEngine.__init__` ALREADY composes
  `self.datasets = DatasetEngine(storage)` (M2) — no new wiring;
  `ComparisonRecord` persists `dataset_id`/`dataset_version` top-level
  ("Shared probe conditions (identical for A and B by construction)"
  per the schema comment); `ComparisonRequest` exposes exactly ONE
  dataset identity per comparison (`dataset_version: Optional[int]`,
  None → latest).

**Authoritative distribution (independently re-discovered from the
manifests, then re-derived live by the smoke):** exactly ONE dataset —
`ee1a716c4573` (v1). Model `4a0a871886ef` owns **8 comparisons, ALL
persisted with the shared probe `(ee1a716c4573, v1)`**, in ASCENDING
`(created_at, comparison_id)` order with exact side identities:

| comparison | state A | state B | verdict |
|---|---|---|---|
| fc379bfcb50f | checkpoint 30a8bc5b82ab | checkpoint 025e6d8d8f15 | improved |
| baa361012e00 | checkpoint 30a8bc5b82ab | checkpoint 025e6d8d8f15 | improved |
| d683f9b81195 | current | checkpoint 025e6d8d8f15 | improved |
| 786de08efe4c | checkpoint 025e6d8d8f15 | checkpoint 025e6d8d8f15 (A=B) | unchanged |
| 5c5ff22151ed | current | checkpoint 0511de4c7372 | unchanged |
| d9a62dde016b | checkpoint 025e6d8d8f15 | checkpoint 30a8bc5b82ab | regressed |
| 729f9c55ea89 | checkpoint 30a8bc5b82ab | checkpoint 30a8bc5b82ab (A=B) | unchanged |
| d62f89e97c85 | checkpoint 025e6d8d8f15 | checkpoint 30a8bc5b82ab | regressed |

Side totals: 14 checkpoint sides + 2 current sides; sides carry NO
dataset fields; NO comparison has different datasets across sides
(structurally impossible); the two same-checkpoint A=B records
(786de08efe4c, 729f9c55ea89) are also same-dataset A=B cases. Model
`b5bc905326b6` has NO comparisons (natural model-scoped empty case).
M30 grounding verified: `TokenizerEngine.load` raises
`FileNotFoundError` for unknown tokenizers (`GET /tokenizers/{id}` →
404); exactly ONE tokenizer exists in production (`99106e3255c5`) and
all 16 evaluations persist it top-level.

## 3. Implementation

Reuse, not a second engine — a filter over the authoritative listing:

- **`app/comparison.py`** — `ComparisonEngine.
  list_comparisons_for_dataset(model_id, dataset_id)` directly after
  `list_comparisons_for_checkpoint` (M26). It (1) validates the
  dataset via the already-composed M2 registry handle
  `self.datasets.load_meta(dataset_id)` (`FileNotFoundError` → 404),
  then (2) returns `[r for r in self.list_comparisons(model_id) if
  r.dataset_id == dataset_id]` — ordering inherited from the M5
  `(created_at, comparison_id)` sort, `[]` when none, never writes,
  no dataset-comparison index, no duplicate manifest parsing. The
  docstring documents the shared-probe rule (ONE dataset identity per
  comparison; per-side dataset identities cannot occur), the
  exactly-once dedup (by comparison identity), the verbatim-version
  rule and the global-dataset / model-scoped-history semantics.
- **`app/engine.py`** — facade `ModelForge.list_comparisons_for_dataset`
  directly after `list_comparisons_for_checkpoint` (delegation +
  docstring; `ComparisonRecord` already imported since M26).
- **`app/api.py`** — route `GET /models/{model_id}/comparisons/
  by-dataset/{dataset_id}` (`response_model=list[ComparisonRecord]`,
  tags `["comparison"]`, `FileNotFoundError → 404`) placed after the
  M26 by-checkpoint route and **before** the generic
  `GET /models/{model_id}/comparisons/{comparison_id}`. Comparison
  section header extended; landing-page M29 bullet + one endpoint-list
  `<li>` added.
- **`README.md`** — Milestone 29 section (shared-probe identity,
  once-per-comparison, versions verbatim, global datasets /
  model-scoped history, M2-registry-first validation, exact M5
  ordering, `200 []`, 404s, read-only, route placement); test-count
  450 → 455 in Quickstart + Layout; `comparison.py` layout line
  extended to "by-checkpoint/by-dataset grouping (M26/M29)".
- No new modules, schemas, storage structures, caches, indexes,
  workers, databases or background processes.

## 4. Tests

Five new tests (prompt §16 items 1–14 covered; production-facts items
1–2 executed live by the smoke in §5; item 4's "one-side matching /
side A → requested dataset, side B → another dataset" is structurally
impossible per the discovered shared-probe schema — M5 refuses
cross-probe comparisons with 422 (existing test
`test_cross_dataset_refused`), so the tests instead VERIFY the
structural fact and cover unrelated-dataset exclusion):

- `tests/test_comparison.py` (engine, 3) — a cached `_m29_env` on the
  shared module env (ds_b pinned to BOTH its versions v1 + v2 —
  reusing/creating the module's existing v2 — plus a third dataset
  with zero comparisons):
  `test_m29_engine_filters_by_persisted_dataset_identity` (parity
  with the M5 listing filtered by persisted `dataset_id` for ds_a /
  ds_b / the empty ds_c; deterministic order; unique ids; verbatim
  `get_comparison` parity; ds_b = exactly the two pinned comparisons
  with versions **[1, 2] verbatim** (both versions returned, never
  collapsed); every ds_a record exactly once incl. same-checkpoint
  A=B; no overlap between dataset groups; structural check that
  `ComparisonSide.model_fields` contains NO dataset keys),
  `test_m29_engine_empty_404s_isolation_read_only` (empty dataset `[]`;
  model-scoped empty via the M26 second model; `FileNotFoundError`
  unknown model/dataset; a's ids never under b; comparison-manifest
  file-sets unchanged by reads),
  `test_m29_engine_repeated_calls_identical` (4 identical calls).
- `tests/test_comparison_api.py` (API, 2):
  `test_m29_api_by_dataset_grouping_versions_determinism` (cross +
  same-checkpoint A=B on ds, one comparison on a second dataset;
  exact ids/order; verbatim equality with the POST responses; dataset
  identity + version verbatim; parity with the filtered
  `GET /comparisons` listing — no missing/extra/duplicate; second
  dataset holds exactly its own record; 3 GETs raw-byte identical;
  detail getter not shadowed) and
  `test_m29_api_404s_isolation_regressions_openapi` (unknown
  model/dataset (well-formed + malformed) 404s; datasets global →
  another real model + this dataset = model-scoped `200 []`; M5
  run/list/get regression + ghost 404; M26 by-checkpoint unchanged
  (the record under BOTH its checkpoints); M28
  evaluations-by-dataset parity unchanged; OpenAPI: **61 paths**,
  path once, GET-only, `comparison` tag, `array` of `$ref
  ComparisonRecord`, after M26 and before the generic route).
- OpenAPI count assertions 60 → 61 updated at the exact 13 sites only
  (`test_sample_quality.py` ×4, `test_suite_runs.py` ×3,
  `test_comparison_api.py` ×1, `test_evaluation_api.py` ×2,
  `test_dashboards.py` ×1, `test_gates_api.py` ×1,
  `test_sampling.py` ×1); grep-verified 0 `== 60` remain; adjacent
  prose ladders corrected to 61. No existing test weakened or removed.

Honest failure log during development (first failures reported, none
hidden):

1. My first append of the engine tests contained a doubled line-
   continuation (`== \\`) and one leftover nonsense assertion line —
   `pyflakes` caught the syntax error before any test run (rc=1);
   both fixed, then pyflakes clean.
2. No runtime test failure occurred at any point: focused comparison
   suites passed first try (37/37), then the full suite 455 first try.

Final results: focused comparison modules **37 passed** (32
pre-existing + 5 new); full suite **455 passed** serially (110.17 s
first run; 110.58 s after the /tmp cleanup rerun); `compileall` clean
(incl. smoke); `pyflakes` clean (0 new findings; only the
long-standing legacy `smoke_m9_live.py`/`smoke_m11_live.py` notes);
OpenAPI standalone post-check: **61 paths**, endpoint exactly once,
GET-only, after the M26 route and before the generic comparison route.

## 5. Live Smoke

`smoke_m29_live.py` (new, committed) against the production
`FORGE_ROOT` on **port 8750**: **41/41 PASS, exit 0, FIRST RUN**.
Baseline per-file SHA256 inventory saved to
`/tmp/m29-smoke-baseline-inventory.json` (96 files). LIVE A–L:

- **A baseline**: exact 96/4,002,745/0; both models resolve; 3
  checkpoints register (other model none); M11/M16/M18–M28 pre-state
  intact (2 sample records, 10 by-suite, summary 10, 3 by-checkpoint
  evals, 1 gate, 10 by-checkpoint suite runs, 4 by-checkpoint samples,
  16 by-dataset evaluations); dashboard hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
  OpenAPI 61 with the new path once (GET-only, `comparison` tag).
- **B authoritative comparison discovery** (from the live M2 registry
  + M5 listing, printed): 1 dataset `ee1a716c4573` (versions `[1]`);
  8 comparisons over `(ee1a716c4573 v1) → 8`; 14 checkpoint sides +
  2 current sides; sides carry dataset fields: **False**; the two
  same-checkpoint A=B records exactly as audited; other model `[]`.
- **C known dataset**: `4a0a871886ef` + `ee1a716c4573` → 200 with
  exactly the discovered **8** records — exact ids in ASCENDING M5
  order; persisted `model_id` + `dataset_id` on every record;
  verbatim detail-getter parity for all 8 (verdict/losses included).
- **D exact parity**: response == the live M5 listing filtered locally
  by persisted dataset identity — no missing / extra / duplicate
  (8 unique ids).
- **E same-dataset A=B**: the two same-checkpoint A=B records
  (786de08efe4c, 729f9c55ea89) appear EXACTLY ONCE each; every id in
  the response unique (dedup by comparison identity, not side
  combinations).
- **F version integrity**: every returned `dataset_version` equals
  the persisted M5 listing value verbatim (all v1; no rewriting).
- **G deterministic repeats**: three GETs raw-byte-identical.
- **H valid empty case**: registry-verified FIRST (dataset valid via
  `GET /datasets/{id}` + `b5bc905326b6`'s M5 listing empty), then
  `200 + []`.
- **I unknown model**: 404 (even with the real dataset id).
- **J unknown dataset**: well-formed 12-hex → 404; malformed
  (percent-encoded spaces/`!!`) → 404, no crash.
- **K cross-model isolation + no shadowing**: none of the 8
  comparison ids leaks under `b5bc905326b6` (history `[]`); the
  detail getter `fc379bfcb50f` resolves verbatim; ghost comparison id
  → 404; the M26 by-checkpoint route still returns 6/5/1.
- **L regression + final audit**: M5 listing + all 8 detail getters,
  M2 registry + `GET /datasets/{id}`, M4 evaluations + M28
  by-dataset (16, parity with the filtered listing), M24
  evaluations-by-checkpoint (3/3/3), M27 samples-by-checkpoint
  (4/0/0), M26 comparisons-by-checkpoint (6/5/1), M6 gates, M16/M18–
  M20 sample-quality surfaces, M21 by-suite (10), M22 summary (10),
  M23 by-policy (1), M25 suite-runs-by-checkpoint (10), M17 dashboard
  (full hash, empty diagnostics), M11 workflows, M3 checkpoint
  registry, M9 policy + probe-suite registries, OpenAPI still 61 with
  the new path once (after M26, before generic) — all unchanged;
  by-dataset deterministic at the end; final storage 96/4,002,745/0
  with 0 changed / 0 new / 0 missing.

The uvicorn server was stopped after the smoke (no server left
running). Housekeeping: 4 stale `/tmp/forge-tests-*` dirs from the
completed suite runs were found during the final audit; confirmed no
test/server process running, removed them, and **reran the full suite:
455 passed** (110.58 s) with the storage inventory still 96 OK / 0
failed (the single fresh `/tmp/forge-tests-*` base is the byproduct of
that rerun itself; no process left running).

## 6. Storage Integrity

- Pre-change inventory `m29_pre.sha256` (repo root, committed) — diff
  vs `m28_pre.sha256` empty at baseline.
- Post-everything audit: **96 files / 4,002,745 bytes / 0 `.tmp`**;
  `sha256sum -c m29_pre.sha256` → **96 OK / 0 failed** (checked twice:
  after the smoke and after the suite rerun); smoke inventory diff →
  **0 changed / 0 new / 0 missing / 0 bytes growth**.
- The endpoint performs zero writes (engine filter over parsed
  manifests only; verified live by the byte-identical inventory). No
  comparison manifests, dataset-comparison indexes, caches, databases
  or summary manifests created.

## 7. Regression / Compatibility

- Full suite: 455/455 with all 450 pre-existing tests intact (only the
  13 exact OpenAPI-count assertions updated 60 → 61 with their
  comments; grep-verified).
- OpenAPI 60 → 61: exactly the one new path, GET-only, `comparison`
  tag, `list[ComparisonRecord]` response, registered after the M26
  by-checkpoint route and before the generic comparison detail route
  (no shadowing in any direction — live-verified).
- Live byte-identical before/after: M2 dataset registry +
  `GET /datasets/{id}`, M4 evaluations (16) + M28 by-dataset, M5
  run/list/get semantics, M6 gate decisions, M20
  sample-quality-by-checkpoint (2 under 0511), M21 by-suite (10), M22
  summary (10), M23 by-policy (1), M24 evaluations-by-checkpoint
  (3/3/3), M25 suite-runs-by-checkpoint (10), M26
  comparisons-by-checkpoint (6/5/1), M27 samples-by-checkpoint
  (4/0/0), M16/M18/M19 sample-quality surfaces, M17 dashboard (hash
  preserved), M11 workflows, M3 checkpoint registry, M9 policy +
  probe-suite registries.
- M1–M28 behavior untouched (additive milestone; no unrelated
  refactoring).

## 8. Commit / Final Status

- Commit: **`M29: add comparison history by dataset`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge`; local HEAD == remote HEAD
  (FETCH_HEAD-verified), working tree clean after the push (commit
  hash recorded in the git log; no unrelated commits amended;
  `ai_model_forge.egg-info/` NOT committed).
- Files changed: `app/comparison.py`, `app/engine.py`, `app/api.py`,
  `tests/test_comparison.py`, `tests/test_comparison_api.py`,
  `tests/test_sample_quality.py`, `tests/test_suite_runs.py`,
  `tests/test_evaluation_api.py`, `tests/test_sampling.py`,
  `tests/test_dashboards.py`, `tests/test_gates_api.py`, `README.md`,
  `smoke_m29_live.py` (new), `m29_pre.sha256` (new, audit artifact),
  `M29_final_report.md` (new).
- Environment note (honesty): the sandbox reset between M28 and M29;
  the documented recovery runbook was executed FIRST (see §2) and the
  M28 certified state was re-proven (450 passed / 60 paths / 96-file
  byte-identical production) before any M29 work began.
- **Final status: M29 COMPLETE AND CERTIFIED.** All gates pass:
  455/455 tests serially (450 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 60 → **61 paths** with the
  endpoint documented exactly once; live smoke **41/41 PASS** on
  production (distribution discovered live: one dataset
  `ee1a716c4573` v1 with all 8 comparisons over it, exact ids + side
  identities + ASCENDING order, same-dataset A=B exactly once,
  versions verbatim (all v1), registry-verified model-scoped empty
  `200 []`, clean 404s, cross-model isolation, byte-identical ×3
  repeats, M2–M28 surfaces + dashboard hash + registries unchanged);
  storage audit **96 files / 4,002,745 bytes / 0 `.tmp` / 0 changed /
  0 new / 0 missing / 0 bytes growth**; dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 30 — EVALUATION HISTORY BY TOKENIZER

Continue the existing **AI Model Forge** project.

M29 is the current certified milestone.

The goal of M30 is to add one small, read-only API capability for
inspecting the existing M4 evaluation history grouped by the tokenizer
each evaluation measured with.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M29 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 455 tests passing
* OpenAPI: 61 paths
* M29 comparisons by-dataset endpoint working
* M28 evaluations by-dataset endpoint working
* M27 samples by-checkpoint endpoint working
* M26 comparisons by-checkpoint endpoint working
* M25 suite-runs by-checkpoint endpoint working
* M24 evaluations by-checkpoint endpoint working
* M23 gate decisions by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working

Also inspect the existing implementations for:

* the M4 evaluation engine: `list_evaluations` / `get_evaluation` —
  the authoritative listing and its deterministic `(created_at,
  eval_id)` ordering; the M24 `list_evaluations_for_checkpoint` and
  M28 `list_evaluations_for_dataset` read-only grouping patterns
* the persisted `EvaluationRecord` (`app/schemas.py`): every record
  persists a top-level `tokenizer_id: str` (plus `tokenizer_hash`
  where applicable) — the tokenizer identity of the probe that
  produced the measurement
* the tokenizer registry (`app/tokenizer.py`):
  `TokenizerEngine.load(tokenizer_id)` raises `FileNotFoundError` for
  unknown ids and is exposed as `GET /tokenizers/{tokenizer_id}`
  (404 unknown) — the M2-style GLOBAL registry validation surface;
  tokenizers are global, so model scoping comes from the model's own
  M4 listing, exactly like M28/M29
* the M20–M29 read-only API/engine patterns (registry validation +
  authoritative-listing filter + thin facade + thin route registered
  BEFORE any literal-shadowing path — here
  `/evaluations/by-checkpoint/...` (M24), `/evaluations/by-dataset/
  ...` (M28) and `/evaluations/{eval_id}`)

If the sandbox has been reset since M29, execute the documented
recovery runbook FIRST (restore production data from the repo-root
`code forge.zip`, verify against `m29_pre.sha256`, rebuild the venv,
git fetch + prove the tree hash-identical + `reset --hard FETCH_HEAD`;
never force-push) and re-prove the M29 baseline before starting.

Create a SHA256 inventory of the production files before making
changes (`m30_pre.sha256`; diff vs `m29_pre.sha256` must be empty).

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M30 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/evaluations/by-tokenizer/{tokenizer_id}`

It answers ONE question: "which immutable M4 evaluations of this model
measured with this tokenizer?" — the model's authoritative M4 listing
filtered by the persisted tokenizer identity recorded in each
`EvaluationRecord` (top-level tokenizer_id; membership never comes
from filenames, eval ids, checkpoint ids, dataset ids or hashes).

Do NOT add: new evaluations, scoring, quality measurement, training,
retraining, rollback, optimization, HPO, ranking, new metrics,
background workers, databases, caches, indexes, analytics storage,
new persistence structures, automatic decisions, workflow changes,
dashboard redesign, Gemini, model improvement logic. No future
milestones early (no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/Gemini/
auto-training/auto-rollback/databases/distributed/rankings).

---

## 3. PRODUCTION FACTS (verified at M29)

Authoritative facts, read from the persisted manifests — do NOT invent
ids or counts; if a stated count disagrees with the manifests or the
live listing, the data wins and the discrepancy must be reported.

* Exactly ONE tokenizer exists in the production registry:
  `99106e3255c5` (only entry under `<root>/tokenizers/`).
* Model `4a0a871886ef` owns exactly **16 evaluations**, ALL persisted
  with top-level `tokenizer_id=99106e3255c5` (14 validation + 2 train
  splits over dataset `ee1a716c4573` v1), in ASCENDING
  `(created_at, eval_id)` order: a439eb92f9cd, b0502d871114,
  7a16eaa12120, 0cc96a125976, 425003213a0b, a884bf729ff7,
  b89a94306ce8, 75835b64d6af, 90aa392b9a2b, 340f5adbf881,
  c739c66638e9, c35a1c9902fe, fe7b42cdb411, ebb9b7eccbe3,
  0704399fea7b, 75a23351e91c.
* Model `b5bc905326b6` exists with NO evaluations (natural
  model-scoped empty case: valid global tokenizer + model with empty
  history -> 200 + []).
* The 8 comparisons of `4a0a871886ef` also persist
  `tokenizer_id=99106e3255c5` (M29 facts).
* Storage at M29 certification: 96 files / 4,002,745 bytes / 0 `.tmp`.
* M17 dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.

---

## 4. REQUIRED BEHAVIOR

* Validate the model (unknown -> 404) through the M4 listing path, and
  the tokenizer through the registry `TokenizerEngine.load(
  tokenizer_id)` (unknown tokenizer -> 404; never a raw filesystem
  check; never infer from filenames).
* Return the model's authoritative M4 `list_evaluations` output
  filtered by the persisted tokenizer identity, in the exact M4
  `(created_at, eval_id)` ASCENDING order, as verbatim
  `EvaluationRecord` payloads (loss/perplexity/state/dataset identity
  included).
* A valid tokenizer with zero evaluations for the model -> `200 + []`
  (never 404 — "no evaluations" is not "does not exist").
* Tokenizers are global: cross-MODEL isolation comes from the model's
  own listing (a model never sees another model's evaluations); there
  is no cross-tokenizer leakage because the filter matches the exact
  persisted `tokenizer_id`.
* Read-only and deterministic: repeated GETs byte-identical; the
  endpoint never writes; no duplicate manifest parsing beyond the
  authoritative listing; no new storage.
* The route MUST be registered BEFORE
  `GET /models/{model_id}/evaluations/{eval_id}` (and keep the M24
  by-checkpoint and M28 by-dataset routes fully intact and
  unconfused with this one).

---

## 5. ENGINE IMPLEMENTATION

Follow the M24/M28/M29 pattern exactly:

* ONE method on the M4 evaluation engine
  (`list_evaluations_for_tokenizer(model_id, tokenizer_id)`), placed
  directly after `list_evaluations_for_dataset` (M28): validate the
  tokenizer via the already-composed tokenizer engine handle (M4's
  `__init__` already composes `self.tokenizers` — add no new wiring),
  then filter the authoritative `list_evaluations` listing by the
  persisted `tokenizer_id`; `[]` when none; never writes. Docstring
  documents semantics + 404 mapping + the global-registry /
  model-scoped-history rule.
* Thin facade in `app/engine.py` after
  `list_evaluations_for_dataset`.
* Thin route in `app/api.py` (tags `["evaluation"]`,
  `response_model=list[EvaluationRecord]`, `FileNotFoundError -> 404`),
  placed after the M28 by-dataset route and before the generic
  evaluation detail getter; extend the evaluation section-header
  comment, the landing-page milestone list (one bullet) and the
  endpoint list (one `<li>`).
* README: minimal Milestone 30 section (registry-first validation,
  persisted tokenizer identity authoritative, verbatim payloads, exact
  M4 ordering, `200 []`, 404s, read-only) + update the two test-count
  mentions (455 -> 460) + the evaluation layout line.
* OpenAPI 61 -> 62; update ONLY the affected exact-count assertions
  (grep for `== 61`, 13 sites expected) and their adjacent stale
  comments.

---

## 6. TESTS

Fixture tests (pytest, in the existing evaluation test modules,
reusing their helpers/fixtures; never rebuild shared state per test —
cache it on the module-scoped fixture env):

1. grouping parity: evaluations over >= 2 tokenizers (the module env
   already trains a second tokenizer in some suites — verify): the
   endpoint output equals the authoritative M4 listing filtered by the
   persisted tokenizer_id, for each tokenizer and for an empty one
2. verbatim payloads (equal to `get_evaluation` for each record)
3. deterministic ordering + repeated GETs raw-byte identical
4. valid tokenizer with no evaluations for the model -> `200 + []`
   (train a fresh tokenizer if needed; plus the unknown-tokenizer 404)
5. unknown model -> 404; unknown tokenizer -> 404 (well-formed +
   malformed)
6. model scoping: another real model's evaluations never appear
7. M4 run/list/get + validation unchanged (regression)
8. M24 evaluations-by-checkpoint + M28 evaluations-by-dataset
   unchanged (different groupings of the same listing)
9. M29 comparisons-by-dataset unchanged
10. OpenAPI: 62 paths, the new path exactly once, GET-only,
    `evaluation` tag, `array` of `$ref EvaluationRecord`, before the
    generic evaluation route
11. route order: the generic evaluation detail getter still resolves;
    ghost eval id still 404

Expected: 455 -> 460 tests passing.

---

## 7. LIVE TEST

Write `smoke_m30_live.py` (pattern: `smoke_m29_live.py`) and run it
against a live server on **port 8751** with
`FORGE_ROOT=/home/user/ai-model-forge-data`. LIVE A–K:

* A baseline: exact 96/4,002,745/0 audit + per-file SHA256 inventory
  saved to `/tmp/m30-smoke-baseline-inventory.json` + model/tokenizer/
  checkpoint registries + M4 listing pre-state + M16/M18-M29
  pre-state + dashboard hash + OpenAPI 62 pre-state
* B discover the authoritative evaluation distribution FROM THE LIVE
  M4 LISTING (per tokenizer_id: total count + ids); print it;
  cross-check 16 records all under `99106e3255c5`; the registry holds
  exactly one tokenizer
* C known tokenizer: `4a0a871886ef` + `99106e3255c5` -> 200 with the
  EXACT 16 records — parity with the filtered M4 listing, exact M4
  order, persisted tokenizer identity on every record, verbatim
  detail-getter parity for every record
* D exact parity: response == M4 listing filtered locally by
  persisted tokenizer identity (no missing / extra / duplicate)
* E natural empty model case: `b5bc905326b6` + `99106e3255c5` ->
  200 + [] (registry-verify first: tokenizer valid + other model's
  M4 listing empty)
* F deterministic repeats: >= 3 GETs raw-byte-identical
* G unknown model -> 404
* H unknown tokenizer (well-formed + malformed) -> 404
* I cross-model isolation: none of the 16 evaluation ids leaks into
  the other model's response; the generic evaluation detail getter
  still resolves (no shadowing); ghost eval id 404; the M24
  by-checkpoint (3/3/3) and M28 by-dataset (16) routes unchanged
* J regression: M2 dataset registry, M3 checkpoint registry, M4
  list/get, M5 comparisons, M6 gates, M16 sample quality, M17
  dashboard hash, M18-M20 sample-quality surfaces, M21/M22, M23 (1),
  M25 (10), M26 (6/5/1), M27 (4/0/0), M29 comparisons-by-dataset (8),
  OpenAPI 62 — all unchanged
* K final audit: every pre-existing file byte-identical, ZERO new
  files, zero `.tmp`, zero storage growth

Exit code 0 = pass. Stop the server afterwards.

---

## 8. FINAL AUDIT

Static sequence — run in this order and report the FIRST failure
honestly if anything fails:

1. focused tests (evaluation modules)
2. full suite serially (expect 460 passed)
3. live smoke (LIVE A–K, port 8751)
4. `compileall` (app, tests, smoke script)
5. `pyflakes` (app, tests, smoke script; 0 new findings)
6. OpenAPI post-check: 62 paths, endpoint exactly once, before the
   generic evaluation route
7. SHA256 audit: production inventory vs the pre-M30 inventory
   (0 changed / 0 new / 0 missing / 0 bytes growth)
8. tree audit: only the intended files changed; no build artifacts
   (`ai_model_forge.egg-info/`) committed
9. remote sync check: local HEAD == remote HEAD, clean tree

If stale `/tmp/forge-tests-*` directories appear: verify no test
process is running, remove only stale directories, rerun the full
suite, repeat the required audits.

---

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; ONE engine method; ONE facade; ONE route.
* No new evaluations, scoring, training/rollback/optimization/HPO/
  ranking, no new metrics, no workers, databases, caches, indexes, no
  new persistence, no automatic decisions, no workflow changes, no
  dashboard redesign, no Gemini, no model improvement logic.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the real `b5bc905326b6` (valid model + valid
  tokenizer, empty history).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (61 -> 62) and their adjacent stale comments.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 10. COMMIT / PUSH / REPORT

* Commit message: `M30: add evaluation history by tokenizer`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M30_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (eval ids, per-tokenizer counts,
  ordering, determinism, 404s, cross-model/scoping, SHA256, dashboard
  hash, commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M31 prompt, grounded
  ONLY in facts discovered and verified during M30 (no invented ids,
  counts or endpoints — verify a surface exists before promising it),
  small/additive/read-only, no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/
  Gemini/auto-training/auto-rollback/databases/distributed/rankings,
  including automatic M32 prompt generation in its own report
  section.
```
