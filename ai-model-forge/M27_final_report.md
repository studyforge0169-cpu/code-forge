# MILESTONE 27 — SAMPLE HISTORY BY CHECKPOINT — FINAL REPORT

## 1. Objective

Add exactly ONE new read-only endpoint to the existing **AI Model Forge**:

`GET /api/v1/models/{model_id}/samples/by-checkpoint/{checkpoint_id}`

answering *"which immutable M15 samples were generated from this
checkpoint state?"* — the model's authoritative M15 listing filtered by
the **persisted sample identity** (every `SampleRecord` carries a
required non-nullable `checkpoint_id`; M15 generation always binds ONE
explicit verified checkpoint; there is no state_kind enum and no
current-state sample). Required behavior, all delivered:

- Membership NEVER inferred from directory names, timestamps, hashes,
  prompt text or tokenizer identity — the persisted `checkpoint_id` is
  authoritative.
- Checkpoint ownership validated through the model-scoped M3 registry
  (`TrainingEngine.get_checkpoint`): unknown model, unknown checkpoint,
  OR another model's checkpoint id → 404 (no raw filesystem existence
  checks).
- Reuse the authoritative M15 `list_samples` ordering
  `(created_at, sample_id)` ASCENDING; verbatim `SampleRecord` payloads
  (prompt, token ids, output text, strategy, `result_hash` included);
  no counts/averages/quality scores/rankings/summaries.
- Valid checkpoint with no samples → `200 + []` (never 404).
- Route registered BEFORE `/models/{model_id}/samples/{sample_id}`.
- Read-only: no training/rollback/optimization/HPO/ranking/quality
  scoring/new metrics/automatic decisions/workers/databases/caches/
  indexes/analytics storage/new persistence structures/Gemini/model
  improvement loops/dashboard redesign. No second sample registry.

## 2. Baseline

Verified BEFORE any code change (after the documented sandbox-reset
recovery — see §8 notes):

- HEAD = **`75eb541`** (`M26: add comparison history by checkpoint`),
  branch `arena/01a071e9-code-forge`, remote HEAD identical
  (FETCH_HEAD-verified), working tree clean except the fresh untracked
  `ai-model-forge/ai_model_forge.egg-info/` build artifact (not
  committed, per instructions).
- Recovery re-verified from scratch before baseline: production data
  restored from the repo-root `code forge.zip` and checked against
  `m26_pre.sha256` (96 OK / 0 failed); venv rebuilt; full suite **440
  passed**; OpenAPI **58 paths**; production **96 files / 4,002,745
  bytes / 0 `.tmp`**.
- Pre-change inventory `m27_pre.sha256` created; `diff` vs
  `m26_pre.sha256` → **empty** (zero drift from M26 certification).
- Inspection confirmed: `SamplingEngine` already composes
  `self.training = TrainingEngine(storage)` (M3, "checkpoint
  verification") — no new wiring; `list_samples` sorts
  `(created_at, sample_id)` ASC and returns `[]` for a checkpoint-less
  history; `SampleRecord.checkpoint_id: str` is required/non-nullable;
  `GET /workflows/recipes/{recipe_id}/runs` already exists (workflow
  grouping taken — not an M28 candidate).

**Authoritative sample distribution (discovered fresh from the
manifests and re-derived live by the smoke):** model `4a0a871886ef`
owns exactly **4 samples, ALL from checkpoint `0511de4c7372`**, in
ASCENDING `(created_at, sample_id)` order: `f8e66f9c7b50` (greedy,
12:13:33.461341Z), `e2b5166fa549` (greedy, .479056Z), `05820e8bc68a`
(temperature, .494498Z), `4e8463e0df06` (temperature, .511219Z).
Checkpoints `025e6d8d8f15` and `30a8bc5b82ab` are valid with ZERO
samples (natural live empty cases). Model `b5bc905326b6` has NO
checkpoints and NO samples (model dir = manifest + weights only).

## 3. Implementation

Reuse, not a second engine — a filter over the authoritative listing:

- **`app/sampling.py`** — `SamplingEngine.list_samples_for_checkpoint(
  model_id, checkpoint_id)` in a new "M27: per-checkpoint access
  (read-only)" section directly after `get_sample`. It (1) validates
  existence + ownership via the already-composed
  `self.training.get_checkpoint` (M3 registry; a checkpoint id
  belonging to another model raises `FileNotFoundError` exactly like an
  unknown one), then (2) returns `[r for r in self.list_samples(
  model_id) if r.checkpoint_id == checkpoint_id]` — ordering inherited
  from the M15 `(created_at, sample_id)` sort, `[]` when none, never
  writes, no duplicate manifest parsing. No changes to `run`,
  `_persist`, the manifest format or the storage layout.
- **`app/engine.py`** — facade `ModelForge.list_samples_for_checkpoint`
  directly after `get_sample` (delegation + docstring; `SampleRecord`
  was already imported).
- **`app/api.py`** — route `GET /models/{model_id}/samples/
  by-checkpoint/{checkpoint_id}` (`response_model=list[SampleRecord]`,
  tags `["sampling"]`, `FileNotFoundError → 404`) placed after
  `list_samples` and **before** the generic
  `GET /models/{model_id}/samples/{sample_id}` (the literal
  `by-checkpoint` segment is not a sample id). Sampling section header
  extended; landing-page M27 bullet + one endpoint-list `<li>` added.
- **`README.md`** — Milestone 27 section (registry-first ownership,
  persisted-identity filter, verbatim payloads, exact M15 ordering,
  `200 []`, 404s, read-only, route-before-detail, M20
  sample-quality-by-checkpoint is a different surface and untouched);
  test-count 440 → 445 in Quickstart + Layout; `sampling.py` layout
  line extended with the M27 grouping.
- No new modules, schemas, storage structures, caches, indexes,
  workers, databases or background processes.

## 4. Tests

Five new tests (prompt §11 items 1–12 covered; production-facts items
1–2 are executed live by the smoke in §5):

- `tests/test_sampling.py` (engine, 3) — a cached `_m27_state` on the
  shared module env (samples under BOTH main-model checkpoints, one on
  the small model's checkpoint, and a fully trained fresh model whose
  checkpoint has zero samples):
  `test_m27_engine_grouping_parity_order_verbatim` (parity with the M15
  listing filtered by the persisted checkpoint identity for all three
  (model, checkpoint) pairs, deterministic order, persisted identity on
  every record, unique ids, verbatim `get_sample` parity, each new
  sample under exactly its own checkpoint and no other),
  `test_m27_engine_empty_404s_cross_model_read_only` (`[]` for the
  valid-but-empty checkpoint; `FileNotFoundError` cross-model both
  directions, unknown model, unknown checkpoint; samples directory
  file-set unchanged by reads),
  `test_m27_engine_repeated_calls_identical` (4 identical calls).
- `tests/test_sampling.py` (API, 2):
  `test_m27_api_by_checkpoint_grouping_determinism_and_empty` (greedy +
  seeded-temperature generations; exact ids/order; verbatim equality
  with the POST responses; parity with the filtered `GET /samples`
  listing; the model's OTHER checkpoint → `200 + []`; three GETs
  raw-byte-identical; detail getter not shadowed) and
  `test_m27_api_404s_isolation_regressions_openapi` (unknown model /
  unknown checkpoint (well-formed + malformed) 404s; cross-model
  isolation both directions with a second real trained model whose own
  history is `[]`; M15 generate/list/get regression; ghost sample 404;
  M20 sample-quality-by-checkpoint intact (different surface); M26
  comparisons-by-checkpoint intact; OpenAPI: **59 paths**, path exactly
  once, GET-only, `sampling` tag, `array` of `$ref SampleRecord`,
  registered before the generic sample route).
- OpenAPI count assertions 58 → 59 updated at the exact 11 sites only
  (`test_sample_quality.py` ×4, `test_suite_runs.py` ×3,
  `test_comparison_api.py` ×1, `test_dashboards.py` ×1,
  `test_gates_api.py` ×1, `test_evaluation_api.py` ×1); grep-verified
  0 `== 58` remain; the adjacent prose comments (ladders/totals) were
  corrected to 59 in the same assertions. No existing test weakened or
  removed.

Honest failure log during development (first failures reported, none
hidden): the first `pyflakes` pass over the OpenAPI-comment edits found
nothing wrong, but my own review caught **stale prose totals** left in
four assertion comments (one reading `= 55`, one `= 57`, two `58`)
after the mechanical `== 59` update — each corrected to the full
milestone ladder before any test run; no runtime failure ever occurred
(tests passed first try: focused 19/19, then full 445).

Final results: focused sampling module **19 passed** (14 pre-existing +
5 new); full suite **445 passed** serially (120.80 s first run; 113.09 s
after the /tmp cleanup rerun described in §5); `compileall` clean
(incl. smoke); `pyflakes` clean (0 new findings; only the long-standing
legacy `smoke_m9_live.py`/`smoke_m11_live.py` notes); OpenAPI standalone
post-check: **59 paths**, endpoint exactly once, GET-only, before the
generic sample route.

## 5. Live Smoke

`smoke_m27_live.py` (new, committed) against the production
`FORGE_ROOT` on **port 8748**: **31/31 PASS, exit 0, first run**.
Baseline per-file SHA256 inventory saved to
`/tmp/m27-smoke-baseline-inventory.json` (96 files). LIVE A–I:

- **A baseline**: exact 96/4,002,745/0; both models resolve; the 3
  known checkpoints register; the other model's checkpoint registry is
  authoritatively EMPTY; M11/M16/M18–M26 pre-state intact (2 sample
  records, 10 by-suite, summary 10, 3 checkpoint evals, 1 gate, 10
  by-checkpoint suite runs); dashboard hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
  OpenAPI 59 with the new path once (GET-only, `sampling` tag).
- **B discover distribution** (from the live M15 listing, printed):
  4 samples total, 1 checkpoint → `0511de4c7372` → 4
  [f8e66f9c7b50, e2b5166fa549, 05820e8bc68a, 4e8463e0df06]; ASCENDING
  (created_at, sample_id) order; 2 greedy + 2 temperature; other model
  `[]` — matches the audited values, nothing assumed.
- **C known checkpoint `0511de4c7372`**: 200, exactly **4** records —
  the exact discovered ids in M15 order; every record's persisted
  `model_id` + `checkpoint_id` verified; parity with the filtered live
  listing (equals the full listing here); verbatim detail-getter parity
  for all 4 (prompt/token ids/output text/result_hash included).
- **D valid empty checkpoints**: `025e6d8d8f15` → `200 + []` and
  `30a8bc5b82ab` → `200 + []`.
- **E deterministic repeats**: three GETs per checkpoint (all three
  checkpoints) raw-byte-identical.
- **F unknown model**: 404 (even with a real checkpoint id).
- **G unknown checkpoint**: well-formed 12-hex → 404; malformed
  (percent-encoded spaces/`!!`) → 404, no crash.
- **H cross-model + no shadowing**: `b5bc905326b6` registry confirmed
  empty first (authoritative; no guessed statuses), then
  `b5bc905326b6` + the real registry-valid `0511de4c7372` → 404; no
  sample id reachable under the other model (history `[]`); detail
  getter `f8e66f9c7b50` resolves verbatim; ghost sample id → 404.
- **I regression + final audit**: M15 listing + all 4 detail getters,
  M16 sample quality, M18 records, M19 by-sample, M20 by-checkpoint
  (2), M21 by-suite (10), M22 summary (10), M23 by-policy (1), M24
  evaluations by-checkpoint (3), M25 suite-runs by-checkpoint (10),
  **M26 comparisons by-checkpoint (6/5/1)**, M17 dashboard (full hash,
  empty diagnostics), M11 workflows, M3 checkpoint + M9 policy
  registries, OpenAPI still 59 — all byte-identical to pre-state;
  by-checkpoint deterministic at the end; final storage 96/4,002,745/0
  with 0 changed / 0 new / 0 missing.

The uvicorn server was stopped after the smoke (no server left
running). Housekeeping: 3 stale `/tmp/forge-tests-*` dirs from the
completed suite runs were found during the final audit; confirmed no
test/server process running, removed them, and **reran the full suite:
445 passed** (113.09 s) with the storage inventory still 96 OK / 0
failed.

## 6. Storage Integrity

- Pre-change inventory `m27_pre.sha256` (repo root, committed) — diff
  vs `m26_pre.sha256` empty at baseline.
- Post-everything audit: **96 files / 4,002,745 bytes / 0 `.tmp`**;
  `sha256sum -c m27_pre.sha256` → **96 OK / 0 failed** (checked twice:
  after the smoke and again after the suite rerun); smoke inventory
  diff → **0 changed / 0 new / 0 missing / 0 bytes growth**.
- The endpoint performs zero writes (engine filter over parsed
  manifests only; verified live by the byte-identical inventory). No
  new sample manifests, indexes, caches, summary files or databases.

## 7. Regression / Compatibility

- Full suite: 445/445 with all 440 pre-existing tests intact (only the
  11 exact OpenAPI-count assertions updated 58 → 59, comments
  corrected alongside; grep-verified).
- OpenAPI 58 → 59: exactly the one new path, GET-only, `sampling` tag,
  `list[SampleRecord]` response, registered before the generic
  `/samples/{sample_id}` route (no shadowing in either direction —
  live-verified).
- Live byte-identical before/after: M15 generate/list/get semantics,
  M16 sample quality, M18 records, M19 by-sample, M20
  sample-quality-by-checkpoint (a different surface — quality
  measurements, not samples), M21/M22 suite surfaces, M23 by-policy,
  M24 evaluations by-checkpoint, M25 suite-runs by-checkpoint, M26
  comparisons by-checkpoint (6/5/1 exact counts), M11 workflows, M17
  dashboard (hash preserved), M3 checkpoint registry, M9 policy/suite
  registries.
- M1–M26 behavior untouched (additive milestone; no unrelated
  refactoring).

## 8. Commit / Final Status

- Commit: **`M27: add sample history by checkpoint`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge`; local HEAD == remote HEAD
  (FETCH_HEAD-verified), working tree clean after the push (commit
  hash recorded in the git log; no unrelated commits amended;
  `ai_model_forge.egg-info/` NOT committed).
- Files changed: `app/sampling.py`, `app/engine.py`, `app/api.py`,
  `tests/test_sampling.py`, `tests/test_sample_quality.py`,
  `tests/test_suite_runs.py`, `tests/test_comparison_api.py`,
  `tests/test_dashboards.py`, `tests/test_gates_api.py`,
  `tests/test_evaluation_api.py`, `README.md`,
  `smoke_m27_live.py` (new), `m27_pre.sha256` (new, audit artifact),
  `M27_final_report.md` (new).
- Environment note (honesty): the sandbox reset between M26 and M27;
  the documented recovery runbook was executed FIRST (production data
  restored from `code forge.zip` + verified 96 OK / 0 failed vs
  `m26_pre.sha256`; venv rebuilt; git fetched + tree proven
  hash-identical + `reset --hard FETCH_HEAD` at `75eb541`; never
  force-pushed), and the M26 certified state was re-proven (440
  passed / 58 paths / clean tree) before any M27 work began.
- **Final status: M27 COMPLETE AND CERTIFIED.** All gates pass:
  445/445 tests serially (440 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 58 → **59 paths** with the
  endpoint documented exactly once; live smoke **31/31 PASS** on
  production (distribution discovered live: 4 samples all from
  `0511de4c7372` with the exact ids, `200 + []` for both valid empty
  checkpoints, partition 4+0+0=4, clean 404s, cross-model isolation,
  byte-identical ×3 repeats, M26 counts 6/5/1 preserved); storage
  audit **96 files / 4,002,745 bytes / 0 `.tmp` / 0 changed / 0 new /
  0 missing / 0 bytes growth**; dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 28 — EVALUATION HISTORY BY DATASET

Continue the existing **AI Model Forge** project.

M27 is the current certified milestone.

The goal of M28 is to add one small, read-only API capability for
inspecting the existing M4 evaluation history grouped by the dataset
(and version) each evaluation measured.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M27 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 445 tests passing
* OpenAPI: 59 paths
* M27 samples by-checkpoint endpoint working
* M26 comparisons by-checkpoint endpoint working
* M25 suite-runs by-checkpoint endpoint working
* M24 evaluations by-checkpoint endpoint working
* M23 gate decisions by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working
* M15 sample generation / listing / getter working

Also inspect the existing implementations for:

* the M4 evaluation engine: `list_evaluations` / `get_evaluation` —
  the authoritative listing and its deterministic `(created_at,
  eval_id)` ordering; the M24 `list_evaluations_for_checkpoint`
  read-only grouping pattern
* the persisted `EvaluationRecord` (`app/schemas.py`): every record
  carries `dataset_id: str` AND `dataset_version: int` at the TOP
  LEVEL (plus split/tokenizer_id/state identity) — the dataset
  identity is a (dataset_id, dataset_version) pair, and BOTH fields
  are persisted per record
* the M2 dataset registry: `DatasetEngine.get(dataset_id)`
  (`app/dataset.py`, exposed as `GET /datasets/{dataset_id}`) — the
  ownership/existence validation surface (unknown dataset ->
  FileNotFoundError -> 404). NOTE: datasets are GLOBAL, not
  model-scoped — any model may legitimately be evaluated on any
  dataset; the model scoping of this endpoint comes from the model's
  own evaluation listing, exactly like M24/M27
* the M20–M27 read-only API/engine patterns (registry validation +
  authoritative-listing filter + thin facade + thin route registered
  BEFORE any literal-shadowing path — here
  `/evaluations/by-checkpoint/...` and `/evaluations/{eval_id}`)

Create a SHA256 inventory of the production files before making
changes (`m28_pre.sha256`; diff vs `m27_pre.sha256` must be empty).

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M28 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/evaluations/by-dataset/{dataset_id}`

It answers ONE question: "which immutable M4 evaluations of this model
measured THIS dataset?" — the model's authoritative M4 listing filtered
by the persisted dataset identity recorded in each `EvaluationRecord`.
An optional boolean strictness decision (filter by `dataset_id` across
all its versions, since `dataset_version` is persisted per record and
is part of each returned payload) must be documented in the docstring
and README exactly as implemented.

Do NOT add: new evaluations, scoring, quality measurement, training,
retraining, rollback, optimization, HPO, ranking, new metrics,
background workers, databases, caches, indexes, analytics storage, new
persistence structures, automatic decisions, workflow changes,
dashboard redesign, Gemini, model improvement logic. No future
milestones early (no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/Gemini/
auto-training/auto-rollback/databases/distributed/rankings).

---

## 3. PRODUCTION FACTS (verified at M27)

Authoritative facts, read from the persisted manifests — do NOT invent
ids or counts; if a stated count disagrees with the manifests or the
live listing, the data wins and the discrepancy must be reported.

* Model `4a0a871886ef` owns exactly **16 evaluations**, ALL persisted
  with `dataset_id=ee1a716c4573` and `dataset_version=1` (9
  checkpoint-state evals grouped 3/3/3 under checkpoints
  `025e6d8d8f15` / `0511de4c7372` / `30a8bc5b82ab`, and 7
  current-state evals); M4 listing order `(created_at, eval_id)`
  ASCENDING.
* Exactly ONE dataset exists in production: `ee1a716c4573` (only
  entry under `<root>/datasets/`) — so the non-empty case is
  16 records and there is no second non-empty dataset.
* Model `b5bc905326b6` exists with NO evaluations at all (model dir =
  manifest + weights only) -> `b5bc905326b6` + `ee1a716c4573` is the
  natural live `200 + []` case (valid model, valid dataset, empty
  history).
* Storage at M27 certification: 96 files / 4,002,745 bytes / 0 `.tmp`.
* M17 dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.

---

## 4. REQUIRED BEHAVIOR

* Validate the model (unknown -> 404) through the M4 listing path, and
  the dataset through the M2 registry `DatasetEngine.get(dataset_id)`
  (unknown dataset -> 404; never a raw filesystem existence check;
  never infer from filenames).
* Return the model's authoritative M4 `list_evaluations` output
  filtered by the persisted dataset identity, in the exact M4
  `(created_at, eval_id)` ASCENDING order, as verbatim
  `EvaluationRecord` payloads (loss/perplexity/state identity/dataset
  version included).
* A valid dataset with zero evaluations for the model -> `200 + []`
  (never 404 — "no evaluations" is not "does not exist").
* Datasets are global: cross-MODEL isolation comes from the model's
  own listing (a model never sees another model's evaluations); there
  is no cross-DATASET leakage because the filter matches the exact
  persisted `dataset_id`.
* Read-only and deterministic: repeated GETs byte-identical; the
  endpoint never writes; no duplicate manifest parsing beyond the
  authoritative listing; no new storage.
* The route MUST be registered BEFORE
  `GET /models/{model_id}/evaluations/{eval_id}` (and keep the M24
  `by-checkpoint` route fully intact and unconfused with this one).

---

## 5. ENGINE IMPLEMENTATION

Follow the M24/M25/M26/M27 pattern exactly:

* ONE method on the M4 evaluation engine
  (`list_evaluations_for_dataset(model_id, dataset_id)`), placed
  directly after `list_evaluations_for_checkpoint`: validate the
  dataset via the already-composed M2 dataset engine handle (add no
  new wiring), then filter the authoritative `list_evaluations`
  listing by the persisted `dataset_id`; `[]` when none; never
  writes. Docstring documents semantics + 404 mapping + the
  version-handling decision.
* Thin facade in `app/engine.py` after
  `list_evaluations_for_checkpoint`.
* Thin route in `app/api.py` (tags `["evaluation"]`,
  `response_model=list[EvaluationRecord]`, `FileNotFoundError -> 404`),
  placed after the M24 by-checkpoint route and before the generic
  evaluation detail getter; extend the evaluation section-header
  comment, the landing-page milestone list (one bullet) and the
  endpoint list (one `<li>`).
* README: minimal Milestone 28 section (M2-registry-first validation,
  persisted dataset identity authoritative, verbatim payloads, exact
  M4 ordering, `200 []`, 404s, read-only) + update the two test-count
  mentions (445 -> 450) + the evaluation layout line.
* OpenAPI 59 -> 60; update ONLY the affected exact-count assertions
  (grep for `== 59`, 11 sites expected).

---

## 6. TESTS

Fixture tests (pytest, in the existing evaluation test modules,
reusing their helpers/fixtures; never rebuild shared state per test —
cache it on the module-scoped fixture env):

1. grouping parity: evaluations over >= 1 dataset (and >= 2 dataset
   VERSIONS if the fixture creates them): the endpoint output equals
   the authoritative M4 listing filtered by the persisted dataset_id
2. a fixture with a SECOND dataset having zero evaluations for the
   model -> `200 + []` (plus the unknown-dataset 404)
3. verbatim payloads (equal to `get_evaluation` for each record)
4. deterministic ordering + repeated GETs raw-byte identical
5. unknown model -> 404; unknown dataset -> 404
6. model scoping: another real model's evaluations never appear
7. M4 run/list/get + validation unchanged (regression)
8. M24 evaluations-by-checkpoint unchanged (different grouping of the
   same listing)
9. M27 samples-by-checkpoint + M26 comparisons-by-checkpoint
   unchanged
10. OpenAPI: 60 paths, the new path exactly once, GET-only,
    `evaluation` tag, `array` of `$ref EvaluationRecord`
11. route order: the generic evaluation detail getter still resolves;
    ghost eval id still 404

Expected: 445 -> 450 tests passing.

---

## 7. LIVE TEST

Write `smoke_m28_live.py` (pattern: `smoke_m27_live.py`) and run it
against a live server on **port 8749** with
`FORGE_ROOT=/home/user/ai-model-forge-data`. LIVE A–I:

* A baseline: exact 96/4,002,745/0 audit + per-file SHA256 inventory
  saved to `/tmp/m28-smoke-baseline-inventory.json` + model/dataset/
  checkpoint registries + M4 listing pre-state + M16/M18-M27
  pre-state + dashboard hash + OpenAPI 60 pre-state
* B discover the authoritative evaluation distribution FROM THE LIVE
  M4 LISTING (per dataset_id: total count, versions, state kinds);
  print it; cross-check 16 records all under `ee1a716c4573` v1
* C known dataset: `4a0a871886ef` + `ee1a716c4573` -> 200 with the
  EXACT 16 records — parity with the filtered M4 listing, exact M4
  order, persisted dataset identity on every record, verbatim
  detail-getter parity for every record
* D natural empty case: `b5bc905326b6` + `ee1a716c4573` -> 200 + []
* E deterministic repeats: >= 3 GETs raw-byte-identical
* F unknown model -> 404
* G unknown dataset (well-formed + malformed) -> 404
* H no shadowing: `by-dataset` does not shadow the M24 by-checkpoint
  route (still 3/0/0 for 0511/025e/30a8 — verify live) nor the
  generic evaluation detail getter (real id resolves, ghost id 404);
  model scoping: no evaluation id of `4a0a871886ef` is reachable
  under `b5bc905326b6`
* I regression: M2 dataset registry + `GET /datasets/{id}`, M3
  checkpoint registry, M4 list/get, M16 sample quality, M17 dashboard
  hash, M18-M20 sample-quality surfaces, M21/M22, M23, M24 (3 records
  under 0511de4c7372), M25 (10), M26 (6/5/1), M27 (4/0/0 samples),
  OpenAPI 60 — all unchanged; final audit: every pre-existing file
  byte-identical, ZERO new files, zero `.tmp`, zero storage growth

Exit code 0 = pass. Stop the server afterwards.

---

## 8. FINAL AUDIT

Static sequence — run in this order and report the FIRST failure
honestly if anything fails:

1. focused tests (evaluation modules)
2. full suite serially (expect 450 passed)
3. live smoke (LIVE A–I, port 8749)
4. `compileall` (app, tests, smoke script)
5. `pyflakes` (app, tests, smoke script; 0 new findings)
6. OpenAPI post-check: 60 paths, endpoint exactly once, before the
   generic evaluation route
7. SHA256 audit: production inventory vs the pre-M28 inventory
   (0 changed / 0 new / 0 missing / 0 bytes growth)
8. tree audit: only the intended files changed; no build artifacts
   (`ai_model_forge.egg-info/`) committed
9. remote sync check: local HEAD == remote HEAD, clean tree

---

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; ONE engine method; ONE facade; ONE route.
* No new evaluations, scoring, training/rollback/optimization/HPO/
  ranking, no new metrics, no workers, databases, caches, indexes, no
  new persistence, no automatic decisions, no workflow changes, no
  dashboard redesign, no Gemini, no model improvement logic.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the real `b5bc905326b6` (valid model + valid
  dataset, empty history).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (59 -> 60) and their adjacent stale comments.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 10. COMMIT / PUSH / REPORT

* Commit message: `M28: add evaluation history by dataset`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M28_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (eval ids, per-dataset counts,
  ordering, determinism, 404s, cross-model/scoping, SHA256, dashboard
  hash, commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M29 prompt, grounded
  ONLY in facts discovered and verified during M28 (no invented ids,
  counts or endpoints — verify a surface exists before promising it),
  small/additive/read-only, no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/
  Gemini/auto-training/auto-rollback/databases/distributed/rankings,
  including automatic M30 prompt generation in its own report
  section.
```
