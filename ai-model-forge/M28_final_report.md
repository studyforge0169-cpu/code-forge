# MILESTONE 28 — EVALUATION HISTORY BY DATASET — FINAL REPORT

## 1. Objective

Add exactly ONE new read-only endpoint to the existing **AI Model Forge**:

`GET /api/v1/models/{model_id}/evaluations/by-dataset/{dataset_id}`

answering *"which immutable M4 evaluations of this model measured this
dataset?"* — the model's authoritative M4 listing filtered by the
**persisted dataset identity**. Required behavior, all delivered:

- Persisted identity authoritative: every `EvaluationRecord` carries
  top-level `dataset_id: str` + `dataset_version: int`; membership
  matches the persisted `dataset_id` only — never filenames, dataset
  directory names, tokenizer ids, eval ids, checkpoint ids, hashes or
  timestamps.
- **Version preservation**: the persisted `dataset_version` travels
  VERBATIM inside every returned record — all versions of the dataset
  are returned, each exactly as persisted; versions are never
  collapsed, resolved to the latest, aliased or rewritten.
- Dataset validation through the M2 registry (`DatasetEngine.load_meta`
  — the exact registry call M4's own run preflight uses; unknown
  dataset → 404; no raw filesystem checks). Datasets are GLOBAL; model
  scoping comes from the model's own M4 listing (a model never sees
  another model's evaluations).
- Reuse the authoritative M4 ordering `(created_at, eval_id)`
  ASCENDING; verbatim `EvaluationRecord` payloads (loss/perplexity/
  state identity included); no aggregates/averages/rankings/leaderboards.
- Valid dataset with no evaluations for the model → `200 + []`.
- Route registered AFTER the M24 by-checkpoint route and BEFORE the
  generic `/evaluations/{eval_id}` getter.
- Read-only: no training/rollback/HPO/ranking/new metrics/automatic
  decisions/workers/databases/caches/indexes/analytics storage/new
  persistence/Gemini/dashboard redesign. No second dataset or
  evaluation registry.

## 2. Baseline

Verified BEFORE any code change (no sandbox reset occurred this
milestone — environment intact from M27):

- HEAD = **`2044f78`** (`M27: add sample history by checkpoint`),
  branch `arena/01a071e9-code-forge`, local HEAD == remote HEAD
  (FETCH_HEAD-verified), working tree clean except the untracked
  `ai-model-forge/ai_model_forge.egg-info/` build artifact (not
  committed).
- Full suite serially: **445 passed** (106.64 s); `compileall` +
  `pyflakes` clean; OpenAPI **59 paths** (new path absent).
- Production: **96 files / 4,002,745 bytes / 0 `.tmp`**;
  `m28_pre.sha256` created; diff vs `m27_pre.sha256` → **empty**.
- Inspection confirmed: `EvaluationEngine.__init__` already composes
  `self.datasets = DatasetEngine(storage)` (M2) — no new wiring;
  `DatasetEngine.load_meta` raises `FileNotFoundError` for unknown
  datasets (the registry contract M4's `run` itself uses at
  `app/evaluation.py:196`); `list_evaluations` sorts
  `(created_at, eval_id)` ASCENDING.

**Authoritative distribution (independently re-discovered from the
manifests, then re-derived live by the smoke):** exactly ONE dataset
exists — `ee1a716c4573` ("m3-live-ds", 200 records, versions `[1]`).
Model `4a0a871886ef` owns **16 evaluations, ALL persisted with
`dataset_id=ee1a716c4573` and `dataset_version=1`** (14 `validation` +
2 `train` splits), in ASCENDING `(created_at, eval_id)` order:
`a439eb92f9cd, b0502d871114, 7a16eaa12120, 0cc96a125976, 425003213a0b,
a884bf729ff7, b89a94306ce8, 75835b64d6af, 90aa392b9a2b, 340f5adbf881,
c739c66638e9, c35a1c9902fe, fe7b42cdb411, ebb9b7eccbe3, 0704399fea7b,
75a23351e91c`. Model `b5bc905326b6` has NO evaluations (natural
model-scoped empty case). Cross-checks for M29 grounding: all 8
comparison manifests also persist `(dataset_id, dataset_version)` =
`ee1a716c4573` v1; suite-run manifests do NOT persist dataset fields.

**Honest correction (discovered during this milestone):** the M28
prompt I generated in the M27 report contained a wrong regression hint
— it claimed M24 evaluations-by-checkpoint counts of "3/0/0" for
`0511/025e/30a8`. The authoritative values are **3/3/3** (9
checkpoint-state evaluations, 3 per checkpoint), verified live during
the M28 smoke (the first smoke run FAILED on exactly this check and
was corrected — see §5).

## 3. Implementation

Reuse, not a second engine — a filter over the authoritative listing:

- **`app/evaluation.py`** — `EvaluationEngine.
  list_evaluations_for_dataset(model_id, dataset_id)` in a new "M28"
  section directly after `list_evaluations_for_checkpoint` (M24). It
  (1) validates the dataset via the already-composed M2 registry
  handle `self.datasets.load_meta(dataset_id)` (`FileNotFoundError` →
  404; the same call M4's run preflight uses), then (2) returns
  `[r for r in self.list_evaluations(model_id) if r.dataset_id ==
  dataset_id]` — ordering inherited from the M4 `(created_at, eval_id)`
  sort, `[]` when none, never writes, no dataset-evaluation index, no
  duplicate manifest parsing. The docstring documents the global-vs-
  model-scoped semantics and the verbatim version rule.
- **`app/engine.py`** — facade `ModelForge.list_evaluations_for_dataset`
  directly after `list_evaluations_for_checkpoint` (delegation +
  docstring; `EvaluationRecord` added to the existing schemas import —
  caught by pyflakes, see §4).
- **`app/api.py`** — route `GET /models/{model_id}/evaluations/
  by-dataset/{dataset_id}` (`response_model=list[EvaluationRecord]`,
  tags `["evaluation"]`, `FileNotFoundError → 404`) placed after the
  M24 by-checkpoint route and **before** the generic
  `GET /models/{model_id}/evaluations/{eval_id}`. Evaluation section
  header extended; landing-page M28 bullet + one endpoint-list `<li>`
  added.
- **`README.md`** — Milestone 28 section (M2-registry-first validation,
  persisted dataset identity + verbatim versions, global datasets /
  model-scoped history, exact M4 ordering, `200 []`, 404s, read-only,
  route placement, M24 as a different grouping kept intact); test-count
  445 → 450 in Quickstart + Layout; `evaluation.py` layout line
  extended with "by-checkpoint/by-dataset grouping (M24/M28)".
- No new modules, schemas, storage structures, caches, indexes,
  workers, databases or background processes.

## 4. Tests

Five new tests (prompt §13 items 1–12 covered; production-facts items
1–3 executed live by the smoke in §5):

- `tests/test_evaluation.py` (engine, 3) — a cached `_m28_env` on the
  shared module env (a model with 2 evaluations on dataset "base" + 1
  on "probe", a second model with NO evaluations, and a third freshly
  uploaded tokenized dataset with zero evaluations):
  `test_m28_engine_filters_by_persisted_dataset_identity` (parity with
  the M4 listing filtered by the persisted `dataset_id` for all three
  datasets incl. the empty one; deterministic order; persisted identity
  + model on every record; `dataset_version` verbatim vs the original
  run result; `get_evaluation` verbatim parity; no cross-dataset
  leakage),
  `test_m28_engine_empty_404s_model_scoping_read_only` (model-scoped
  empty `[]` for both valid datasets; `FileNotFoundError` unknown
  model/dataset; a's eval ids never under b; evaluation-manifest
  file-set unchanged by reads — reusing the module's `_m24_eval_files`
  helper),
  `test_m28_engine_repeated_calls_identical` (4 identical calls).
- `tests/test_evaluation_api.py` (API, 2):
  `test_m28_api_by_dataset_grouping_versions_determinism` (evals on two
  datasets via POST; exact ids/order; verbatim equality with the run
  responses; `dataset_id`/`dataset_version` verbatim; parity with the
  filtered `GET /evaluations` listing — no missing/extra/duplicate;
  second dataset holds exactly its own eval; 3 GETs raw-byte
  identical) and `test_m28_api_404s_scoping_regressions_openapi`
  (unknown model/dataset (well-formed + malformed) 404s; datasets
  global → another real model + this dataset = model-scoped `200 []`;
  M4 run/list/get regression; ghost eval 404; by-dataset shadows
  neither the generic detail getter nor M24 by-checkpoint; M27
  samples-by-checkpoint + M26 comparisons-by-checkpoint intact;
  OpenAPI: **60 paths**, path once, GET-only, `evaluation` tag,
  `array` of `$ref EvaluationRecord`, after M24 and before the generic
  route).
- OpenAPI count assertions 59 → 60 updated at the exact 12 sites only
  (`test_sample_quality.py` ×4, `test_suite_runs.py` ×3,
  `test_sampling.py` ×1, `test_comparison_api.py` ×1,
  `test_dashboards.py` ×1, `test_gates_api.py` ×1,
  `test_evaluation_api.py` ×1); grep-verified 0 `== 59` remain;
  adjacent prose ladders corrected to 60 in the same comments. No
  existing test weakened or removed.

Honest failure log during development (first failures reported, none
hidden):

1. `pyflakes` (run before pytest) caught `EvaluationRecord` undefined
   in `app/engine.py` — my facade return annotation referenced it
   while the M24 facade (unannotated) never had; import added before
   any test run.
2. Smoke run 1 crashed at LIVE B (script bug): `KeyError: 'dataset_id'`
   — the M2 `GET /datasets` list items key the id as `id` and
   `GET /datasets/{id}` nests the dataset under `"dataset"`. Fixed the
   script against the live registry shapes; production was never
   wrong.
3. Smoke run 2 FAILED exactly one check (K4): my hardcoded M24
   by-checkpoint regression counts (3/0/0) were wrong — the
   authoritative values are **3/3/3**, verified live immediately after.
   The wrong hint originated in my own M27-report §9 prompt; corrected
   in the script (and recorded here). Run 3: **34/34 PASS**.

Final results: focused evaluation modules **28 passed** (23
pre-existing + 5 new); full suite **450 passed** serially (110.71 s
first run; 106.77 s after the /tmp cleanup rerun); `compileall` clean
(incl. smoke); `pyflakes` clean (0 new findings; only the long-standing
legacy `smoke_m9_live.py`/`smoke_m11_live.py` notes); OpenAPI standalone
post-check: **60 paths**, endpoint exactly once, GET-only, after the
M24 route and before the generic evaluation route.

## 5. Live Smoke

`smoke_m28_live.py` (new, committed) against the production
`FORGE_ROOT` on **port 8749**: **34/34 PASS, exit 0** (after the two
script fixes logged in §4; production behavior never failed). Baseline
per-file SHA256 inventory saved to
`/tmp/m28-smoke-baseline-inventory.json` (96 files). LIVE A–K:

- **A baseline**: exact 96/4,002,745/0; both models resolve; 3
  checkpoints register (other model none); M11/M16/M18–M27 pre-state
  intact (2 sample records, 10 by-suite, summary 10, 3 by-checkpoint
  evals under 0511, 1 gate, 10 by-checkpoint suite runs, 4
  by-checkpoint samples); dashboard hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
  OpenAPI 60 with the new path once (GET-only, `evaluation` tag).
- **B authoritative dataset discovery** (from the live M2 registry +
  M4 listing, printed): 1 dataset `ee1a716c4573` (versions `[1]`,
  latest 1); 16 evaluations of `4a0a871886ef` over
  `(ee1a716c4573 v1) → 16`; exact ids in ASCENDING M4 order; other
  model's evaluation listing `[]`; split mix 14 validation + 2 train.
- **C known dataset**: `4a0a871886ef` + `ee1a716c4573` → 200 with
  exactly the discovered **16** records — the exact ids in M4 order;
  persisted `model_id` + `dataset_id` on every record; verbatim
  detail-getter parity for all 16 (loss/perplexity/state identity).
- **D exact parity**: response == `GET /models/4a0a871886ef/
  evaluations` filtered locally by persisted dataset identity — no
  missing, no extra, no duplicate (16 unique ids).
- **E version integrity**: every returned `dataset_version` equals the
  persisted M4 listing value verbatim (all v1; no normalization).
- **F natural empty model case**: `b5bc905326b6` + registry-valid
  `ee1a716c4573` → `200 + []`.
- **G deterministic repeats**: three GETs raw-byte-identical.
- **H unknown model**: 404 (even with the real dataset id).
- **I unknown dataset**: well-formed 12-hex → 404; malformed
  (percent-encoded spaces/`!!`) → 404, no crash.
- **J cross-model isolation**: registry-verified FIRST (dataset valid
  via `GET /datasets/{id}` + other model's M4 listing empty), then the
  by-dataset answer for `b5bc905326b6` is `200 + []`; none of the 16
  evaluation ids leaks into it. Nothing hard-coded.
- **K regression + final audit**: M4 listing + all 16 detail getters,
  M2 registry + `GET /datasets/{id}`, M24 evaluations-by-checkpoint
  (**3/3/3** — corrected, see §2), M25 suite-runs-by-checkpoint (10),
  M26 comparisons-by-checkpoint (6/5/1), M27 samples-by-checkpoint
  (4/0/0), M16/M18–M20 sample-quality surfaces, M21 by-suite (10),
  M22 summary (10), M23 by-policy (1), M17 dashboard (full hash, empty
  diagnostics), M11 workflows, M3 checkpoint registry, M9 policy +
  probe-suite registries, OpenAPI still 60 with the new path once —
  all unchanged; by-dataset deterministic at the end; final storage
  96/4,002,745/0 with 0 changed / 0 new / 0 missing.

The uvicorn server was stopped after the smoke (no server left
running). Housekeeping: 4 stale `/tmp/forge-tests-*` dirs from the
completed suite runs were found during the final audit; confirmed no
test/server process running, removed them, and **reran the full suite:
450 passed** (106.77 s) with the storage inventory still 96 OK / 0
failed (one fresh `/tmp/forge-tests-*` base is the byproduct of that
rerun itself; no process was left running).

## 6. Storage Integrity

- Pre-change inventory `m28_pre.sha256` (repo root, committed) — diff
  vs `m27_pre.sha256` empty at baseline.
- Post-everything audit: **96 files / 4,002,745 bytes / 0 `.tmp`**;
  `sha256sum -c m28_pre.sha256` → **96 OK / 0 failed** (checked twice:
  after the smoke and after the suite rerun); smoke inventory diff →
  **0 changed / 0 new / 0 missing / 0 bytes growth**.
- The endpoint performs zero writes (engine filter over parsed
  manifests only; verified live by the byte-identical inventory). No
  evaluation manifests, dataset indexes, caches, databases or summary
  files created.

## 7. Regression / Compatibility

- Full suite: 450/450 with all 445 pre-existing tests intact (only the
  12 exact OpenAPI-count assertions updated 59 → 60 with their
  comments; grep-verified).
- OpenAPI 59 → 60: exactly the one new path, GET-only, `evaluation`
  tag, `list[EvaluationRecord]` response, registered after the M24
  by-checkpoint route and before the generic evaluation detail route
  (no shadowing in any direction — live-verified).
- Live byte-identical before/after: M2 dataset registry +
  `GET /datasets/{id}`, M4 run/list/get semantics, M20
  sample-quality-by-checkpoint (2 under 0511), M21 by-suite (10), M22
  summary (10), M23 by-policy (1), M24 evaluations-by-checkpoint
  (3/3/3 exact counts), M25 suite-runs-by-checkpoint (10), M26
  comparisons-by-checkpoint (6/5/1), M27 samples-by-checkpoint (4/0/0),
  M16/M18/M19 sample-quality surfaces, M17 dashboard (hash preserved),
  M11 workflows, M3 checkpoint registry, M9 policy + probe-suite
  registries.
- M1–M27 behavior untouched (additive milestone; no unrelated
  refactoring).

## 8. Commit / Final Status

- Commit: **`M28: add evaluation history by dataset`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge`; local HEAD == remote HEAD
  (FETCH_HEAD-verified), working tree clean after the push (commit
  hash recorded in the git log; no unrelated commits amended;
  `ai_model_forge.egg-info/` NOT committed).
- Files changed: `app/evaluation.py`, `app/engine.py`, `app/api.py`,
  `tests/test_evaluation.py`, `tests/test_evaluation_api.py`,
  `tests/test_sample_quality.py`, `tests/test_suite_runs.py`,
  `tests/test_sampling.py`, `tests/test_comparison_api.py`,
  `tests/test_dashboards.py`, `tests/test_gates_api.py`, `README.md`,
  `smoke_m28_live.py` (new), `m28_pre.sha256` (new, audit artifact),
  `M28_final_report.md` (new).
- **Final status: M28 COMPLETE AND CERTIFIED.** All gates pass:
  450/450 tests serially (445 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 59 → **60 paths** with the
  endpoint documented exactly once; live smoke **34/34 PASS** on
  production (distribution discovered live: one dataset
  `ee1a716c4573` v1 with all 16 evaluations over it, exact ids +
  ordering + verbatim v1 versions, natural model-scoped empty `200 []`,
  clean 404s, registry-verified cross-model isolation, byte-identical
  ×3 repeats, M24 counts corrected to 3/3/3 and all M2/M4/M20–M27
  surfaces + dashboard hash + registries unchanged); storage audit
  **96 files / 4,002,745 bytes / 0 `.tmp` / 0 changed / 0 new /
  0 missing / 0 bytes growth**; dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 29 — COMPARISON HISTORY BY DATASET

Continue the existing **AI Model Forge** project.

M28 is the current certified milestone.

The goal of M29 is to add one small, read-only API capability for
inspecting the existing M5 comparison history grouped by the dataset
the comparisons measured.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M28 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 450 tests passing
* OpenAPI: 60 paths
* M28 evaluations by-dataset endpoint working
* M27 samples by-checkpoint endpoint working
* M26 comparisons by-checkpoint endpoint working
* M25 suite-runs by-checkpoint endpoint working
* M24 evaluations by-checkpoint endpoint working
* M23 gate decisions by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working

Also inspect the existing implementations for:

* the M5 comparison engine: `list_comparisons` / `get_comparison` —
  the authoritative listing and its deterministic `(created_at,
  comparison_id)` ordering; the M26
  `list_comparisons_for_checkpoint` read-only grouping pattern
* the persisted `ComparisonRecord` (`app/schemas.py`): every record
  persists the ONE shared probe identity of the comparison —
  top-level `dataset_id: str` + `dataset_version: int` (a comparison
  is valid only when both sides measure the SAME probe, so there is
  exactly ONE dataset identity per comparison; the two-sided
  state_a/state_b semantics of M26 do NOT apply here)
* the M2 dataset registry: `DatasetEngine.load_meta(dataset_id)` —
  the same registry validation M4's run preflight and M28's
  by-dataset grouping use (unknown dataset -> FileNotFoundError ->
  404); datasets are GLOBAL, so model scoping comes from the model's
  own M5 listing
* the M20–M28 read-only API/engine patterns (registry validation +
  authoritative-listing filter + thin facade + thin route registered
  BEFORE any literal-shadowing path — here
  `/comparisons/by-checkpoint/...` (M26) and
  `/comparisons/{comparison_id}`)

Create a SHA256 inventory of the production files before making
changes (`m29_pre.sha256`; diff vs `m28_pre.sha256` must be empty).

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M29 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/comparisons/by-dataset/{dataset_id}`

It answers ONE question: "which immutable M5 comparisons of this model
measured this dataset?" — the model's authoritative M5 listing
filtered by the persisted dataset identity recorded in each
`ComparisonRecord` (the shared probe's dataset_id; the persisted
dataset_version travels VERBATIM inside every returned record).

Do NOT add: new comparisons, new comparison metrics, scoring,
training, retraining, rollback, optimization, HPO, ranking, automatic
decisions, background workers, databases, caches, indexes, analytics
storage, new persistence structures, Gemini, model improvement logic.
No future milestones early (no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/
Gemini/auto-training/auto-rollback/databases/distributed/rankings).

---

## 3. PRODUCTION FACTS (verified at M28)

Authoritative facts, read from the persisted manifests — do NOT invent
ids or counts; if a stated count disagrees with the manifests or the
live listing, the data wins and the discrepancy must be reported.

* Exactly ONE dataset exists in the M2 registry: `ee1a716c4573`
  ("m3-live-ds", versions `[1]`, 200 records).
* Model `4a0a871886ef` owns exactly **8 comparisons**, ALL persisted
  with the shared probe identity `(dataset_id=ee1a716c4573,
  dataset_version=1)`, in ASCENDING `(created_at, comparison_id)`
  order: fc379bfcb50f, baa361012e00, d683f9b81195, 786de08efe4c,
  5c5ff22151ed, d9a62dde016b, 729f9c55ea89, d62f89e97c85 (M26
  by-checkpoint membership 025e→6 / 30a8→5 / 0511→1 — two A=B records,
  two current-vs-checkpoint records).
* Model `b5bc905326b6` exists with NO comparisons (natural
  model-scoped empty case: valid global dataset + model with empty
  history -> 200 + []).
* Storage at M28 certification: 96 files / 4,002,745 bytes / 0 `.tmp`.
* M17 dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.
* M24 evaluations-by-checkpoint regression counts are **3/3/3**
  (025e/0511/30a8) — an earlier prompt hint of "3/0/0" was wrong and
  was corrected live during M28; always re-verify counts from the
  authoritative listings.

---

## 4. REQUIRED BEHAVIOR

* Validate the model (unknown -> 404) through the M5 listing path, and
  the dataset through the M2 registry `DatasetEngine.load_meta`
  (unknown dataset -> 404; never a raw filesystem check; never infer
  from filenames).
* Return the model's authoritative M5 `list_comparisons` output
  filtered by the persisted dataset identity (top-level dataset_id of
  the shared probe), in the exact M5 `(created_at, comparison_id)`
  ASCENDING order, as verbatim `ComparisonRecord` payloads
  (verdict/per-side losses included).
* The persisted `dataset_version` travels VERBATIM inside every
  returned record — never collapsed, resolved or rewritten.
* A valid dataset with zero comparisons for the model -> `200 + []`
  (never 404).
* Datasets are global: cross-MODEL isolation comes from the model's
  own listing (a model never sees another model's comparisons).
* Read-only and deterministic: repeated GETs byte-identical; the
  endpoint never writes; no duplicate manifest parsing beyond the
  authoritative listing; no new storage.
* The route MUST be registered BEFORE
  `GET /models/{model_id}/comparisons/{comparison_id}` (and keep the
  M26 by-checkpoint route fully intact and unconfused with this one).

---

## 5. ENGINE IMPLEMENTATION

Follow the M24/M25/M26/M27/M28 pattern exactly:

* ONE method on the M5 comparison engine
  (`list_comparisons_for_dataset(model_id, dataset_id)`), placed
  directly after `list_comparisons_for_checkpoint` (M26): validate the
  dataset via the already-composed M2 dataset handle (add no new
  wiring — if `ComparisonEngine` does not already compose a dataset
  engine, compose it exactly once in `__init__` following the
  repository's existing style), then filter the authoritative
  `list_comparisons` listing by the persisted `dataset_id`; `[]` when
  none; never writes. Docstring documents semantics + 404 mapping +
  the single-shared-probe (one dataset identity per comparison) rule.
* Thin facade in `app/engine.py` after
  `list_comparisons_for_checkpoint`.
* Thin route in `app/api.py` (tags `["comparison"]`,
  `response_model=list[ComparisonRecord]`, `FileNotFoundError -> 404`),
  placed after the M26 by-checkpoint route and before the generic
  comparison detail getter; extend the comparison section-header
  comment, the landing-page milestone list (one bullet) and the
  endpoint list (one `<li>`).
* README: minimal Milestone 29 section (M2-registry-first validation,
  persisted shared-probe dataset identity, verbatim versions, exact M5
  ordering, `200 []`, 404s, read-only) + update the two test-count
  mentions (450 -> 455) + the comparison layout line.
* OpenAPI 60 -> 61; update ONLY the affected exact-count assertions
  (grep for `== 60`, 12 sites expected) and their adjacent stale
  comments.

---

## 6. TESTS

Fixture tests (pytest, in the existing comparison test modules,
reusing their helpers/fixtures; never rebuild shared state per test —
cache it on the module-scoped fixture env):

1. grouping parity: comparisons over >= 1 dataset (and a second
   dataset with its own comparison if the fixture supports it): the
   endpoint output equals the authoritative M5 listing filtered by the
   persisted dataset_id
2. a fixture with a SECOND dataset having zero comparisons for the
   model -> `200 + []` (plus the unknown-dataset 404)
3. verbatim payloads (equal to `get_comparison` for each record)
4. deterministic ordering + repeated GETs raw-byte identical
5. unknown model -> 404; unknown dataset -> 404
6. model scoping: another real model's comparisons never appear
7. M5 run/list/get + validation unchanged (regression)
8. M26 comparisons-by-checkpoint unchanged (different grouping of the
   same listing)
9. M28 evaluations-by-dataset + M27 samples-by-checkpoint unchanged
10. OpenAPI: 61 paths, the new path exactly once, GET-only,
    `comparison` tag, `array` of `$ref ComparisonRecord`
11. route order: the generic comparison detail getter still resolves;
    ghost comparison id still 404

Expected: 450 -> 455 tests passing.

---

## 7. LIVE TEST

Write `smoke_m29_live.py` (pattern: `smoke_m28_live.py`) and run it
against a live server on **port 8750** with
`FORGE_ROOT=/home/user/ai-model-forge-data`. LIVE A–K:

* A baseline: exact 96/4,002,745/0 audit + per-file SHA256 inventory
  saved to `/tmp/m29-smoke-baseline-inventory.json` + model/dataset/
  checkpoint registries + M5 listing pre-state + M16/M18-M28
  pre-state + dashboard hash + OpenAPI 61 pre-state
* B discover the authoritative comparison distribution FROM THE LIVE
  M5 LISTING (per dataset_id/version: total count + ids); print it;
  cross-check 8 records all under `ee1a716c4573` v1
* C known dataset: `4a0a871886ef` + `ee1a716c4573` -> 200 with the
  EXACT 8 records — parity with the filtered M5 listing, exact M5
  order, persisted dataset identity on every record, verbatim
  detail-getter parity for every record (verdict/losses included)
* D exact parity: response == M5 listing filtered locally by
  persisted dataset identity (no missing / extra / duplicate)
* E version integrity: every returned dataset_version equals the
  persisted M5 listing value VERBATIM (all v1)
* F natural empty model case: `b5bc905326b6` + `ee1a716c4573` ->
  200 + []
* G deterministic repeats: >= 3 GETs raw-byte-identical
* H unknown model -> 404
* I unknown dataset (well-formed + malformed) -> 404
* J cross-model isolation: registry-verify FIRST (dataset valid via
  GET /datasets/{id} + other model's M5 listing empty), then
  `b5bc905326b6` + `ee1a716c4573` -> 200 + []; no comparison id
  leaks; the generic comparison detail getter still resolves (no
  shadowing) and the M26 by-checkpoint route still returns 6/5/1
* K regression: M2 dataset registry, M3 checkpoint registry, M5
  list/get, M16 sample quality, M17 dashboard hash, M18-M20
  sample-quality surfaces, M21/M22, M23 (1), M24 (3/3/3), M25 (10),
  M26 (6/5/1), M27 (4/0/0), M28 (16 evaluations under
  ee1a716c4573), OpenAPI 61 — all unchanged; final audit: every
  pre-existing file byte-identical, ZERO new files, zero `.tmp`, zero
  storage growth

Exit code 0 = pass. Stop the server afterwards.

---

## 8. FINAL AUDIT

Static sequence — run in this order and report the FIRST failure
honestly if anything fails:

1. focused tests (comparison modules)
2. full suite serially (expect 455 passed)
3. live smoke (LIVE A–K, port 8750)
4. `compileall` (app, tests, smoke script)
5. `pyflakes` (app, tests, smoke script; 0 new findings)
6. OpenAPI post-check: 61 paths, endpoint exactly once, before the
   generic comparison route
7. SHA256 audit: production inventory vs the pre-M29 inventory
   (0 changed / 0 new / 0 missing / 0 bytes growth)
8. tree audit: only the intended files changed; no build artifacts
   (`ai_model_forge.egg-info/`) committed
9. remote sync check: local HEAD == remote HEAD, clean tree

---

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; ONE engine method; ONE facade; ONE route.
* No new comparisons, metrics, scoring, training/rollback/
  optimization/HPO/ranking, no workers, databases, caches, indexes,
  no new persistence, no automatic decisions, no workflow changes, no
  dashboard redesign, no Gemini, no model improvement logic.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the real `b5bc905326b6` (valid model + valid
  dataset, empty history).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (60 -> 61) and their adjacent stale comments.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 10. COMMIT / PUSH / REPORT

* Commit message: `M29: add comparison history by dataset`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M29_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (comparison ids, per-dataset
  counts, ordering, determinism, 404s, cross-model/scoping, SHA256,
  dashboard hash, commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M30 prompt, grounded
  ONLY in facts discovered and verified during M29 (no invented ids,
  counts or endpoints — verify a surface exists before promising it),
  small/additive/read-only, no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/
  Gemini/auto-training/auto-rollback/databases/distributed/rankings,
  including automatic M31 prompt generation in its own report
  section.
```
