# MILESTONE 30 — EVALUATION HISTORY BY TOKENIZER — FINAL REPORT

## 1. Objective

Add exactly ONE new read-only endpoint to the existing **AI Model Forge**:

`GET /api/v1/models/{model_id}/evaluations/by-tokenizer/{tokenizer_id}`

answering *"which immutable M4 evaluations of this model measured with
this tokenizer?"* — the model's authoritative M4 listing filtered by
the **persisted tokenizer identity**. Delivered semantics:

- Filter on the persisted `EvaluationRecord.tokenizer_id` VERBATIM —
  never inferred from filenames, eval ids, checkpoint/dataset
  identities, the tokenizer currently registered, or a
  latest-tokenizer substitution; no tokenizer versioning introduced
  (M2/M4 have none).
- Tokenizer validation through the existing registry
  (`TokenizerEngine.load` — the same getter `GET /tokenizers/{id}`
  exposes; unknown tokenizer → 404; no raw filesystem checks).
  Tokenizers are GLOBAL; model scoping comes from the model's own M4
  listing (a model never sees another model's evaluations).
- Reuse the authoritative M4 ordering `(created_at, eval_id)`
  ASCENDING; complete verbatim `EvaluationRecord` payloads; no new
  metrics, no execution/creation of evaluations, no aggregation, no
  caches/indexes/databases, no duplicated listing logic.
- Valid tokenizer with no evaluations for the model → `200 + []`;
  unknown model/tokenizer → `404`.
- Route registered after the M28 by-dataset route and BEFORE the
  generic `/evaluations/{eval_id}` getter.

## 2. Baseline

The sandbox reset between M29 and M30 — the documented recovery
runbook was executed FIRST (production restored from the repo-root
`code forge.zip`, verified against `m29_pre.sha256`: 96 OK / 0
failed; venv rebuilt; git fetched, tree proven hash-identical,
`reset --hard FETCH_HEAD` at `6005139`; never force-pushed). Then the
baseline was independently re-proven BEFORE any code change:

- HEAD = **`6005139`** (`M29: add comparison history by dataset`),
  branch `arena/01a071e9-code-forge`, local == remote
  (FETCH_HEAD-verified), tree clean except the untracked
  `ai_model_forge.egg-info/` artifact (not committed).
- Full suite serially: **455 passed** (76.98 s); OpenAPI **61 paths**
  (new path absent); production **96 files / 4,002,745 bytes /
  0 `.tmp`**; `m30_pre.sha256` created; diff vs `m29_pre.sha256` →
  **empty** (zero drift).
- Inspection confirmed: `EvaluationEngine.__init__` already composes
  `self.tokenizers = TokenizerEngine(storage)` — no new wiring;
  `TokenizerEngine.load` raises `FileNotFoundError` for unknown ids;
  `EvaluationRecord` persists top-level `tokenizer_id: str`.

**Authoritative distribution (freshly discovered from the manifests,
re-derived live by the smoke):** exactly ONE tokenizer in the registry
— `99106e3255c5`. Model `4a0a871886ef` owns **16 evaluations, ALL
persisted with `tokenizer_id=99106e3255c5`** (14 validation + 2 train
over dataset `ee1a716c4573` v1), in ASCENDING `(created_at, eval_id)`
order: a439eb92f9cd, b0502d871114, 7a16eaa12120, 0cc96a125976,
425003213a0b, a884bf729ff7, b89a94306ce8, 75835b64d6af, 90aa392b9a2b,
340f5adbf881, c739c66638e9, c35a1c9902fe, fe7b42cdb411, ebb9b7eccbe3,
0704399fea7b, 75a23351e91c. Model `b5bc905326b6` has NO evaluations
(natural model-scoped empty case). M31 grounding also re-verified: the
8 comparisons all persist `tokenizer_id=99106e3255c5`.

## 3. Implementation

`M2 tokenizer registry → M4 evaluation listing → persisted
tokenizer_id filter → thin facade → API route` — no second system:

- **`app/evaluation.py`** — `EvaluationEngine.
  list_evaluations_for_tokenizer(model_id, tokenizer_id)` directly
  after `list_evaluations_for_dataset` (M28). It (1) validates the
  tokenizer via the already-composed registry handle
  `self.tokenizers.load(tokenizer_id)` (`FileNotFoundError` → 404),
  then (2) returns `[r for r in self.list_evaluations(model_id) if
  r.tokenizer_id == tokenizer_id]` — ordering inherited from the M4
  `(created_at, eval_id)` sort, `[]` when none, never writes, no
  duplicated manifest parsing. Docstring documents the verbatim
  identity rule, the no-versioning fact and the global-tokenizer /
  model-scoped-history semantics.
- **`app/engine.py`** — facade `ModelForge.
  list_evaluations_for_tokenizer` directly after
  `list_evaluations_for_dataset` (pure delegation + docstring;
  `EvaluationRecord` already imported since M28).
- **`app/api.py`** — route `GET /models/{model_id}/evaluations/
  by-tokenizer/{tokenizer_id}` (`response_model=list[EvaluationRecord]`,
  tags `["evaluation"]`, `FileNotFoundError → 404`) placed after the
  M28 by-dataset route and **before** the generic
  `GET /models/{model_id}/evaluations/{eval_id}`. Evaluation section
  header extended; landing-page M30 bullet + one endpoint-list `<li>`
  added.
- **`README.md`** — Milestone 30 section (persisted-identity filter,
  global tokenizers / model-scoped history, registry-first validation,
  exact M4 ordering, `200 []`, 404s, read-only, route placement, M24/
  M28/M30 as different groupings of the same listing); test-count
  455 → 460 in Quickstart + Layout; `evaluation.py` layout line
  extended to "by-checkpoint/by-dataset/by-tokenizer grouping
  (M24/M28/M30)".
- No new modules, schemas, storage structures, caches, indexes,
  workers, databases or background processes.

## 4. Tests

Five new focused tests (within the prompt's 5–10 target; §8 items
1–15 covered, production-facts items executed live by the smoke):

- `tests/test_evaluation.py` (engine, 3) — a cached `_m30_env` on the
  shared module env (a SECOND tokenizer tok2 vocab 320 with which the
  base dataset is tokenized and ONE evaluation of the M28 model runs;
  a THIRD tokenizer tok3 with zero evaluations):
  `test_m30_engine_filters_by_persisted_tokenizer_identity` (parity
  with the M4 listing filtered by persisted `tokenizer_id` for tok1 /
  tok2 / the empty tok3; deterministic order; verbatim
  `get_evaluation` parity; persisted identity verbatim; **explicit
  partition** — every eval id under exactly one populated tokenizer,
  `under1 ∩ under2 = ∅`, `under1 ∪ under2` = the full listing),
  `test_m30_engine_empty_404s_model_scoping_read_only` (model-scoped
  empty `[]`; unknown model/tokenizer `FileNotFoundError`; a's ids
  never under b; manifest file-set unchanged by reads),
  `test_m30_engine_repeated_calls_identical` (4 identical calls).
- `tests/test_evaluation_api.py` (API, 2):
  `test_m30_api_by_tokenizer_grouping_partition_determinism` (two
  evals with tok1 + one with tok2; exact ids/order; verbatim equality
  with the POST responses; persisted identity verbatim; parity with
  the filtered `GET /evaluations` listing — no missing/extra/
  duplicates; tok2 holds exactly its own record; 3 GETs raw-byte
  identical) and `test_m30_api_404s_scoping_regressions_openapi`
  (unknown model/tokenizer (well-formed + malformed) 404s; global
  tokenizers → second real model + this tokenizer = model-scoped
  `200 []`; M4 run/list/get regression + ghost 404; by-tokenizer
  shadows neither M24 by-checkpoint, M28 by-dataset nor the generic
  getter; M29 comparisons-by-dataset intact; **existing tokenizer
  registry behavior unchanged** — `GET /tokenizers/{id}` 200 + verbatim
  id, unknown id 404; OpenAPI: **62 paths**, path once, GET-only,
  `evaluation` tag, `array` of `$ref EvaluationRecord`, after M28 and
  before the generic route).
- OpenAPI count assertions 61 → 62 updated at the exact 14 sites only
  (`test_sample_quality.py` ×4, `test_suite_runs.py` ×3,
  `test_comparison_api.py` ×2, `test_evaluation_api.py` ×2,
  `test_dashboards.py` ×1, `test_gates_api.py` ×1,
  `test_sampling.py` ×1); grep-verified 0 `== 61` remain; adjacent
  prose ladders corrected to 62. No existing test weakened.

Honest failure log during development (first failures reported, none
hidden):

1. First engine-test run FAILED 1/3: my `_m30_env` assumed the M28
   fixture model had ONE tok1 evaluation, but it has THREE (`e_b1`,
   `e_b2` on base + `e_p1` on probe — all with tok1). The authoritative
   parity assertion held; the fixture's expected-list was corrected to
   the three real evals. Production code was never wrong.
2. First append of the API tests contained leftover editing
   scaffolding (dead `if False else` expressions) — pyflakes caught
   the syntax error before any run; the helper was rewritten cleanly.
3. No runtime failure afterwards: focused 33/33, full suite 460
   first try, smoke 35/35 first run.

Final results: focused evaluation modules **33 passed** (28
pre-existing + 5 new); full suite **460 passed** serially (77.48 s
first run; 77.18 s after the /tmp cleanup rerun); `compileall` clean
(incl. smoke); `pyflakes` clean (0 new findings; only the
long-standing legacy `smoke_m9_live.py`/`smoke_m11_live` notes);
OpenAPI standalone post-check: **62 paths**, endpoint exactly once,
GET-only, after the M28 route and before the generic evaluation
route.

## 5. Live Smoke

`smoke_m30_live.py` (new, committed) against the production
`FORGE_ROOT` on **port 8751**: **35/35 PASS, exit 0, FIRST RUN**.
Baseline per-file SHA256 inventory saved to
`/tmp/m30-smoke-baseline-inventory.json` (96 files). LIVE A–R:

- **A baseline**: exact 96/4,002,745/0; both models resolve; 3
  checkpoints (other model none); M11/M16/M18–M29 pre-state intact
  (2 sample records, 10 by-suite, summary 10, 3 by-checkpoint evals,
  1 gate, 10 by-checkpoint suite runs, 4 by-checkpoint samples, 16
  by-dataset evals, 8 by-dataset comparisons); dashboard hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
  OpenAPI 62 with the new path once (GET-only, `evaluation` tag).
- **B distribution discovery** (live, printed): 1 tokenizer in the
  registry (`99106e3255c5`, resolvable via `GET /tokenizers/{id}`);
  16 evaluations of `4a0a871886ef` per tokenizer:
  `99106e3255c5 → 16`; exact ids in ASCENDING M4 order; other model
  `[]`.
- **C known tokenizer**: `4a0a871886ef` + `99106e3255c5` → 200 with
  exactly the discovered **16** records; exact ids; persisted
  `model_id` + `tokenizer_id` VERBATIM on every record; verbatim
  detail-getter parity for all 16.
- **D exact parity**: response == the live M4 listing filtered locally
  by persisted tokenizer identity (no missing / extra / duplicate).
- **E ordering**: `(created_at, eval_id)` ASCENDING exactly as the M4
  listing.
- **F deterministic repeats**: three GETs raw-byte-identical.
- **G valid tokenizer + second model**: registry-verified first
  (tokenizer valid + `b5bc905326b6`'s M4 listing empty) → `200 + []`.
- **H unknown model**: 404 (even with the real tokenizer id).
- **I unknown tokenizer**: well-formed 12-hex → 404; malformed
  (percent-encoded spaces/`!!`) → 404, no crash.
- **J M2 tokenizer registry unchanged**: list + `GET /tokenizers/{id}`
  verbatim; unknown id still 404.
- **K M4 generic listing unchanged**: listing + all 16 detail getters
  byte-identical.
- **L M24 checkpoint history unchanged**: 3/3/3.
- **M M28 dataset history unchanged**: 16 under `ee1a716c4573`, parity
  with the filtered listing.
- **N M29 comparison history unchanged**: 8 under `ee1a716c4573`,
  parity with the filtered M5 listing; plus M6 gates, M16/M18–M20,
  M21/M22/M23 (1), M25 (10), M26 (6/5/1), M27 (4/0/0) all unchanged.
- **O dashboard**: result_hash + full output unchanged (empty
  diagnostics).
- **P registries**: M11 workflows, M3 checkpoint registry, M9 policy +
  probe-suite registries, M12/M14 recipe registry unchanged.
- **Q OpenAPI**: exactly **62 paths**, the new path exactly once,
  after the M28 by-dataset route and before the generic evaluation
  route.
- **R storage zero drift**: by-tokenizer still deterministic at the
  end; every pre-existing file byte-identical (0 changed / 0 missing),
  0 new files, totals 96/4,002,745/0.

The uvicorn server was stopped after the smoke. Housekeeping: 6 stale
`/tmp/forge-tests-*` dirs from the completed suite runs were found
during the final audit; confirmed no test/server process running,
removed them, and **reran the full suite: 460 passed** (77.18 s) with
the storage inventory still 96 OK / 0 failed.

## 6. Storage Integrity

- Pre-change inventory `m30_pre.sha256` (repo root, committed) — diff
  vs `m29_pre.sha256` empty at baseline.
- Post-everything audit: **96 files / 4,002,745 bytes / 0 `.tmp`**;
  `sha256sum -c m30_pre.sha256` → **96 OK / 0 failed** (checked twice:
  after the smoke and after the suite rerun); smoke inventory diff →
  **0 changed / 0 new / 0 missing / 0 bytes growth**.
- The endpoint performs zero writes. No evaluation manifests, tokenizer
  manifests, caches, indexes, databases or temporary production
  artifacts created.

## 7. Regression / Compatibility

- Full suite: 460/460 with all 455 pre-existing tests intact (only the
  14 exact OpenAPI-count assertions updated 61 → 62 with comments;
  grep-verified).
- OpenAPI 61 → 62: exactly the one new path, GET-only, `evaluation`
  tag, `list[EvaluationRecord]` response, after M28 by-dataset and
  before the generic evaluation detail route (no shadowing —
  live-verified).
- Live byte-identical before/after: M2 tokenizer registry
  (list/get/404), M4 evaluations (listing + 16 getters), M5
  comparisons, M6 gates, M24 evaluations-by-checkpoint (3/3/3), M28
  evaluations-by-dataset (16, parity), M29 comparisons-by-dataset (8,
  parity), M25 (10), M26 (6/5/1), M27 (4/0/0), M16/M18–M20
  sample-quality surfaces, M17 dashboard (hash preserved), M11
  workflows, M3 checkpoint registry, M9 policy/probe-suite registries,
  M12/M14 recipe registry.
- M1–M29 behavior untouched (additive milestone; no unrelated
  refactoring).

## 8. Commit / Final Status

- Commit: **`M30: add evaluation history by tokenizer`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge`; local HEAD == remote HEAD
  (FETCH_HEAD-verified), working tree clean after the push (commit
  hash recorded in the git log; no unrelated commits amended;
  `ai_model_forge.egg-info/` NOT committed).
- Files changed: `app/evaluation.py`, `app/engine.py`, `app/api.py`,
  `tests/test_evaluation.py`, `tests/test_evaluation_api.py`,
  `tests/test_sample_quality.py`, `tests/test_suite_runs.py`,
  `tests/test_comparison_api.py`, `tests/test_dashboards.py`,
  `tests/test_gates_api.py`, `tests/test_sampling.py`, `README.md`,
  `smoke_m30_live.py` (new), `m30_pre.sha256` (new, audit artifact),
  `M30_final_report.md` (new).
- Environment note (honesty): the sandbox reset between M29 and M30;
  the documented recovery runbook was executed FIRST (see §2) and the
  M29 baseline re-proven before any M30 work began.
- **Final status: M30 COMPLETE AND CERTIFIED.** All gates pass:
  460/460 tests serially (455 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 61 → **62 paths** with the
  endpoint documented exactly once; live smoke **35/35 PASS** on
  production (distribution discovered live: one tokenizer
  `99106e3255c5` with all 16 evaluations, exact ids + order + verbatim
  persisted identity, registry-verified model-scoped empty `200 []`,
  clean 404s, byte-identical ×3 repeats, M2–M29 surfaces + dashboard
  hash + registries unchanged); storage audit **96 files / 4,002,745
  bytes / 0 `.tmp` / 0 changed / 0 new / 0 missing / 0 bytes growth**;
  dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 31 — COMPARISON HISTORY BY TOKENIZER

Continue the existing **AI Model Forge** project.

M30 is the current certified milestone.

The goal of M31 is to add one small, read-only API capability for
inspecting the existing M5 comparison history grouped by the tokenizer
the comparisons measured with.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M30 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 460 tests passing
* OpenAPI: 62 paths
* M30 evaluations by-tokenizer endpoint working
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

* the M5 comparison engine: `list_comparisons` / `get_comparison` —
  the authoritative listing and its deterministic `(created_at,
  comparison_id)` ordering; the M26 `list_comparisons_for_checkpoint`
  and M29 `list_comparisons_for_dataset` read-only grouping patterns
* the persisted `ComparisonRecord` (`app/schemas.py`): every record
  persists the ONE shared probe identity top-level, including
  `tokenizer_id: str` (a comparison is valid only when both sides
  measure the SAME probe — one tokenizer identity per comparison;
  per-side tokenizer identities cannot occur by construction)
* the tokenizer registry (`app/tokenizer.py`):
  `TokenizerEngine.load(tokenizer_id)` raises `FileNotFoundError` for
  unknown ids and is exposed as `GET /tokenizers/{id}` (404 unknown) —
  the same GLOBAL registry validation M30 uses. NOTE:
  `ComparisonEngine.__init__` currently composes `datasets`,
  `training` and `evaluation` handles but NOT a tokenizer handle —
  compose `self.tokenizers = TokenizerEngine(storage)` exactly once
  in `__init__` following the repository's existing style (the same
  one-line addition pattern the M29 prompt sanctioned for datasets)
* the M20–M30 read-only API/engine patterns (registry validation +
  authoritative-listing filter + thin facade + thin route registered
  BEFORE any literal-shadowing path — here
  `/comparisons/by-checkpoint/...` (M26), `/comparisons/by-dataset/
  ...` (M29) and `/comparisons/{comparison_id}`)

If the sandbox has been reset since M30, execute the documented
recovery runbook FIRST (restore production data from the repo-root
`code forge.zip`, verify against `m30_pre.sha256`, rebuild the venv,
git fetch + prove the tree hash-identical + `reset --hard FETCH_HEAD`;
never force-push) and re-prove the M30 baseline before starting.

Create a SHA256 inventory of the production files before making
changes (`m31_pre.sha256`; diff vs `m30_pre.sha256` must be empty).

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M31 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/comparisons/by-tokenizer/{tokenizer_id}`

It answers ONE question: "which immutable M5 comparisons of this model
measured with this tokenizer?" — the model's authoritative M5 listing
filtered by the persisted tokenizer identity recorded in each
`ComparisonRecord` (the shared probe's tokenizer_id, matched
VERBATIM; membership never comes from filenames, checkpoint ids,
dataset ids, hashes or the tokenizer currently registered).

Do NOT add: new comparisons, new comparison metrics, scoring,
training, retraining, rollback, optimization, HPO, ranking, automatic
decisions, background workers, databases, caches, indexes, analytics
storage, new persistence structures, Gemini, model improvement logic.
No future milestones early (no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/
Gemini/auto-training/auto-rollback/databases/distributed/rankings).

---

## 3. PRODUCTION FACTS (verified at M30)

Authoritative facts, read from the persisted manifests — do NOT invent
ids or counts; if a stated count disagrees with the manifests or the
live listing, the data wins and the discrepancy must be reported.

* Exactly ONE tokenizer exists in the production registry:
  `99106e3255c5` (only entry under `<root>/tokenizers/`).
* Model `4a0a871886ef` owns exactly **8 comparisons**, ALL persisted
  with the shared probe `tokenizer_id=99106e3255c5` (over dataset
  `ee1a716c4573` v1, validation split), in ASCENDING
  `(created_at, comparison_id)` order: fc379bfcb50f, baa361012e00,
  d683f9b81195, 786de08efe4c, 5c5ff22151ed, d9a62dde016b,
  729f9c55ea89, d62f89e97c85 (M26 membership 025e→6 / 30a8→5 / 0511→1;
  two same-checkpoint A=B records; two current-vs-checkpoint records).
* Model `b5bc905326b6` exists with NO comparisons (natural
  model-scoped empty case: valid global tokenizer + model with empty
  history -> 200 + []).
* The 16 evaluations all persist `tokenizer_id=99106e3255c5` too
  (M30 facts).
* Storage at M30 certification: 96 files / 4,002,745 bytes / 0 `.tmp`.
* M17 dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.

---

## 4. REQUIRED BEHAVIOR

* Validate the model (unknown -> 404) through the M5 listing path, and
  the tokenizer through the registry `TokenizerEngine.load(
  tokenizer_id)` (unknown tokenizer -> 404; never a raw filesystem
  check; never infer from filenames).
* Return the model's authoritative M5 `list_comparisons` output
  filtered by the persisted tokenizer identity, in the exact M5
  `(created_at, comparison_id)` ASCENDING order, as verbatim
  `ComparisonRecord` payloads (verdict/per-side losses included).
* The persisted tokenizer identity travels VERBATIM inside every
  returned record (never rewritten, never substituted with the latest
  tokenizer; no tokenizer versioning exists and none is introduced).
* Each matching comparison appears EXACTLY ONCE (the comparison, not
  the side, is the unit of grouping — the shared probe means both
  sides always measured the requested tokenizer).
* A valid tokenizer with zero comparisons for the model -> `200 + []`
  (never 404).
* Tokenizers are global: cross-MODEL isolation comes from the model's
  own listing (a model never sees another model's comparisons).
* Read-only and deterministic: repeated GETs byte-identical; the
  endpoint never writes; no duplicate manifest parsing beyond the
  authoritative listing; no new storage.
* The route MUST be registered BEFORE
  `GET /models/{model_id}/comparisons/{comparison_id}` (and keep the
  M26 by-checkpoint and M29 by-dataset routes fully intact and
  unconfused with this one).

---

## 5. ENGINE IMPLEMENTATION

Follow the M26/M29/M30 pattern exactly:

* ONE method on the M5 comparison engine
  (`list_comparisons_for_tokenizer(model_id, tokenizer_id)`), placed
  directly after `list_comparisons_for_dataset` (M29): compose the
  tokenizer handle exactly once in `__init__` if not already present,
  validate via `TokenizerEngine.load`, then filter the authoritative
  `list_comparisons` listing by the persisted `tokenizer_id`; `[]`
  when none; never writes. Docstring documents semantics + 404
  mapping + the single-shared-probe (one tokenizer identity per
  comparison) rule.
* Thin facade in `app/engine.py` after
  `list_comparisons_for_dataset`.
* Thin route in `app/api.py` (tags `["comparison"]`,
  `response_model=list[ComparisonRecord]`, `FileNotFoundError -> 404`),
  placed after the M29 by-dataset route and before the generic
  comparison detail getter; extend the comparison section-header
  comment, the landing-page milestone list (one bullet) and the
  endpoint list (one `<li>`).
* README: minimal Milestone 31 section (registry-first validation,
  persisted shared-probe tokenizer identity, verbatim matching, exact
  M5 ordering, once-per-comparison, `200 []`, 404s, read-only) +
  update the two test-count mentions (460 -> 465) + the comparison
  layout line.
* OpenAPI 62 -> 63; update ONLY the affected exact-count assertions
  (grep for `== 62`, 14 sites expected) and their adjacent stale
  comments.

---

## 6. TESTS

Fixture tests (pytest, in the existing comparison test modules,
reusing their helpers/fixtures; never rebuild shared state per test —
cache it on the module-scoped fixture env; target ~5 new tests):

1. grouping parity + explicit partition: comparisons over >= 2
   tokenizers (the module env's `test_cross_tokenizer_refused` already
   trains a second tokenizer m5-tok2 — reuse that pattern: train tok2,
   tokenize a dataset with it, run ONE comparison with tok2): the
   endpoint output equals the authoritative M5 listing filtered by the
   persisted tokenizer_id, for each tokenizer; the two groups are
   disjoint and their union is the full listing
2. a tokenizer with zero comparisons -> `200 + []` (fresh trained
   tokenizer; plus the unknown-tokenizer 404)
3. verbatim payloads (equal to `get_comparison` for each record) +
   persisted tokenizer id VERBATIM
4. deterministic ordering + repeated GETs raw-byte identical
5. unknown model -> 404; unknown tokenizer -> 404 (well-formed +
   malformed)
6. model scoping: another real model's comparisons never appear
7. M5 run/list/get + validation unchanged (regression)
8. M26 comparisons-by-checkpoint + M29 comparisons-by-dataset
   unchanged (different groupings of the same listing)
9. M30 evaluations-by-tokenizer unchanged
10. OpenAPI: 63 paths, the new path exactly once, GET-only,
    `comparison` tag, `array` of `$ref ComparisonRecord`, before the
    generic comparison route
11. route order: the generic comparison detail getter still resolves;
    ghost comparison id still 404

Expected: 460 -> 465 tests passing.

---

## 7. LIVE TEST

Write `smoke_m31_live.py` (pattern: `smoke_m30_live.py`) and run it
against a live server on **port 8752** with
`FORGE_ROOT=/home/user/ai-model-forge-data`. LIVE A–R:

* A baseline: exact 96/4,002,745/0 audit + per-file SHA256 inventory
  saved to `/tmp/m31-smoke-baseline-inventory.json` + model/tokenizer/
  checkpoint registries + M5 listing + M4/M16/M18-M30 pre-state +
  dashboard hash + OpenAPI 63 pre-state
* B discover the authoritative comparison distribution FROM THE LIVE
  M5 LISTING (per tokenizer_id: total count + ids); print it;
  cross-check 8 records all under `99106e3255c5`; the registry holds
  exactly one tokenizer
* C known tokenizer: `4a0a871886ef` + `99106e3255c5` -> 200 with the
  EXACT 8 records — parity with the filtered M5 listing, exact M5
  order, persisted tokenizer identity VERBATIM on every record,
  verbatim detail-getter parity (verdict/losses included)
* D exact parity: response == M5 listing filtered locally by
  persisted tokenizer identity (no missing / extra / duplicate; the
  two A=B records exactly once)
* E authoritative `(created_at, comparison_id)` ASCENDING ordering
  preserved
* F deterministic repeats: >= 3 GETs raw-byte-identical
* G valid tokenizer + second model: `b5bc905326b6` +
  `99106e3255c5` -> 200 + [] (registry-verify first: tokenizer valid +
  other model's M5 listing empty)
* H unknown model -> 404
* I unknown tokenizer (well-formed + malformed) -> 404
* J M2 tokenizer registry unchanged (list + get verbatim; unknown 404)
* K M5 generic listing + all 8 detail getters unchanged
* L M26 checkpoint history unchanged (6/5/1)
* M M29 dataset history unchanged (8 under ee1a716c4573)
* N M30 evaluation history by-tokenizer unchanged (16 under
  99106e3255c5) + M24 (3/3/3) / M25 (10) / M27 (4/0/0) / M28 (16)
* O dashboard output/hash unchanged
* P policy/probe/suite/recipe registries unchanged
* Q OpenAPI exactly 63 paths, the new path once, before the generic
  comparison route
* R storage zero drift (0 changed / 0 new / 0 missing / 0 bytes
  growth)

Exit code 0 = pass. Stop the server afterwards. If stale
`/tmp/forge-tests-*` directories appear: verify no test process is
running, remove only stale directories, rerun the full suite, repeat
the required audits.

---

## 8. FINAL AUDIT

Static sequence — run in this order and report the FIRST failure
honestly if anything fails:

1. focused tests (comparison modules)
2. full suite serially (expect 465 passed)
3. live smoke (LIVE A–R, port 8752)
4. `compileall` (app, tests, smoke script)
5. `pyflakes` (app, tests, smoke script; 0 new findings)
6. OpenAPI post-check: 63 paths, endpoint exactly once, before the
   generic comparison route
7. SHA256 audit: production inventory vs the pre-M31 inventory
   (0 changed / 0 new / 0 missing / 0 bytes growth)
8. tree audit: only the intended files changed; no build artifacts
   (`ai_model_forge.egg-info/`) committed
9. remote sync check: local HEAD == remote HEAD, clean tree

---

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; ONE engine method; ONE facade; ONE route;
  at most ONE new `__init__` composition line in ComparisonEngine.
* No new comparisons, metrics, scoring, training/rollback/
  optimization/HPO/ranking, no workers, databases, caches, indexes,
  no new persistence, no automatic decisions, no workflow changes, no
  dashboard redesign, no Gemini, no model improvement logic.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the real `b5bc905326b6` (valid model + valid
  tokenizer, empty history).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (62 -> 63) and their adjacent stale comments.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 10. COMMIT / PUSH / REPORT

* Commit message: `M31: add comparison history by tokenizer`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M31_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (comparison ids, per-tokenizer
  counts, ordering, determinism, 404s, cross-model/scoping, SHA256,
  dashboard hash, commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M32 prompt, grounded
  ONLY in facts discovered and verified during M31 (no invented ids,
  counts or endpoints — verify a surface exists before promising it),
  small/additive/read-only, no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/
  Gemini/auto-training/auto-rollback/databases/distributed/rankings,
  including automatic M33 prompt generation in its own report
  section.
```
