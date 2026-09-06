# MILESTONE 25 — SUITE-RUN HISTORY BY CHECKPOINT — FINAL REPORT

## 1. Objective

Add exactly ONE deterministic, read-only endpoint to the AI Model Forge
REST API:

```
GET /api/v1/models/{model_id}/suite-runs/by-checkpoint/{checkpoint_id}
```

It answers one question: *which immutable M10 suite-run records of this
model executed against this checkpoint state?* The implementation is a
pure reuse of the existing authoritative registries: the checkpoint is
resolved through the model's M3 checkpoint registry
(`TrainingEngine.get_checkpoint` — the same model-scoped ownership path
M20/M24 use; an unknown checkpoint, or a checkpoint id belonging to
another model, is a 404), and the response is the model's authoritative
M10 suite-run listing (`SuiteRunEngine.list_suite_runs`) filtered by the
**persisted run state** recorded inside each `SuiteRunRecord` — the
nested `state: ComparisonState` representation (`state.state_kind ==
"checkpoint"` with the requested `state.checkpoint_id`), never inferred
from directory names, run ids, timestamps or filenames. Records are
returned verbatim in the exact M10 deterministic order
`(created_at, suite_run_id)` ASCENDING. Current-state runs persist
`state.checkpoint_id = null` and therefore never appear. A valid
checkpoint with no suite runs returns `[]` (never 404); an unknown model
or checkpoint returns 404. No training, rollback, scoring, ranking,
databases, caches, indexes, workers or new persistence structures —
strictly a narrow observability/history milestone.

Verified live on production (`FORGE_ROOT=/home/user/ai-model-forge-data`):
`4a0a871886ef / 0511de4c7372` → HTTP 200 with **exactly 10 records**
(every production suite run of the model), payload- and order-identical
to the existing M10 listing, byte-identical on repeated GETs; the other
two checkpoints are natural live `200 + []` cases; zero production
storage changes.

## 2. Baseline

Environment verified at the authoritative M24 state before any change:
git at **`2f878d3`** ("M24: add evaluation history by checkpoint"),
clean working tree on `arena/01a071e9-code-forge`; production root and
venv intact.

Actual pre-change measurements (all verified **before** touching code):

| Check | Expected | Actual |
|---|---|---|
| Production files | 96 | **96** |
| Production bytes | 4,002,745 | **4,002,745** |
| `.tmp` files | 0 | **0** |
| Full test suite (serial) | 430 passing | **430 passed / 0 failed / 0 errors**, exit 0 |
| OpenAPI paths | 56 | **56** (`suite-runs/by-checkpoint` absent pre-change, verified) |
| `compileall` | clean | **clean** |
| `pyflakes` | clean | **clean** (`app/*.py` + `tests/*.py`) |
| Dashboard result hash | `f48557fe...` | **`f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`** (full value used in the smoke; verified live) |

Pre-change SHA256 inventory of all 96 production files created
(`m25_pre.sha256`; verified identical to the certified M24 inventory).

Production facts resolved from the **authoritative M10 listing** (not
assumed from documentation) — exactly as expected:

- Model `4a0a871886ef` owns **10 suite runs**, **all 10** executed
  against checkpoint state **`0511de4c7372`** (`state.state_kind ==
  "checkpoint"`); ASCENDING ids: `aae8e8a8ba9c, bb114d2ce42e,
  8f8aee834c9f, d9742459017b, 5684649d0ced, 2af508035191, 4cb388a21ce2,
  2dc9f7dff400, 2d203398304b, e4b1c2a7fb2d`
- Checkpoints `025e6d8d8f15` and `30a8bc5b82ab` are valid with **zero**
  suite runs → the two natural live `200 + []` cases
- Second model `b5bc905326b6`: exists, owns **no** checkpoints and **no**
  suite runs (cross-model probe)

Architecture findings from inspection: `SuiteRunRecord.state` is the
nested `ComparisonState` (`state_kind: EvalStateKind` +
`checkpoint_id: Optional[str] = None`) — the exact nullable semantics
M10/M11 persist; `SuiteRunEngine.list_suite_runs` is the authoritative
listing (model check → scan `suite-runs/*/manifest.json` → parse → sort
`(created_at, suite_run_id)` ASCENDING → `[]` when none → never writes);
`SuiteRunEngine` held datasets/evaluation/comparison/policies engines
but **not** the M3 registry, so the milestone added one composed
`TrainingEngine(storage)` reference (the same composition style
`evaluation.py` and `sample_quality.py` already use); the facade
delegates M10/M21/M22 methods thinly; the API's generic route
`/models/{id}/suite-runs/{suite_run_id}` requires the new literal
`by-checkpoint` segment to be registered **before** it.

## 3. Implementation

Repository files changed only — **no production artifact was ever
touched**, no new schema (`SuiteRunRecord` is returned directly), no new
persistence structure:

1. **`app/suite_runs.py`** — new engine method
   `list_suite_runs_for_checkpoint(model_id, checkpoint_id)` placed
   after the M22 summary method:
   `self.training.get_checkpoint(model_id, checkpoint_id)` (existence +
   model-scoped ownership → `FileNotFoundError`), then
   `[r for r in self.list_suite_runs(model_id)
   if r.state.state_kind == EvalStateKind.CHECKPOINT
   and r.state.checkpoint_id == checkpoint_id]`. Supporting additive
   changes: `TrainingEngine` import + `self.training` engine reference
   in `__init__` (M3 registry reuse), `EvalStateKind` import, and a
   module-docstring note. Reuses the existing parsing, model validation,
   checkpoint ownership validation, deterministic ordering and
   corruption handling; no second listing or storage scan.
2. **`app/engine.py`** — thin facade `list_suite_runs_for_checkpoint`
   with a docstring, delegating 1:1 to the engine (same shape as the
   M21–M24 facades).
3. **`app/api.py`** — thin route
   `GET /models/{model_id}/suite-runs/by-checkpoint/{checkpoint_id}`
   (`response_model=list[SuiteRunRecord]`, tag `suite-runs`,
   `FileNotFoundError → 404`), registered **before**
   `/models/{model_id}/suite-runs/{suite_run_id}` so the literal
   `by-checkpoint` segment can never be mistaken for a run id;
   landing-page milestone list + REST-API route list + section-header
   comment updated.
4. **`tests/test_suite_runs.py`** — new "M25" section: 5 focused tests
   (27 → 32 functions) with a cached `_m25_env` builder on top of the
   shared module env (one more checkpoint-state run, one CURRENT-state
   run, and a fresh tiny trained model whose checkpoint has zero runs).
5. **`tests/test_sample_quality.py`** + **`tests/test_dashboards.py`** +
   **`tests/test_suite_runs.py`** + **`tests/test_gates_api.py`** +
   **`tests/test_evaluation_api.py`** — the **nine** factual OpenAPI
   surface-count assertions updated 56 → 57 with refreshed comments
   (documented surface change; **no existing assertion weakened or
   otherwise altered**; grep-verified: 0 remaining `== 56`, 9 new
   `== 57`).
6. **`README.md`** — "Milestone 25" section; test counts 430 → 435 (two
   sites); `suite_runs.py` layout line notes "(M21/M22/M25)".
7. **`smoke_m25_live.py`** — new live production smoke (34 checks, LIVE
   A–J), modeled on the M20–M24 smokes.
8. **`m25_pre.sha256`** — pre-change SHA256 inventory of all 96
   production files (audit artifact; identical to `m24_pre.sha256`).

## 4. Tests

**Focused M25 tests** (3 engine + 2 API, all green):

| Test | Covers |
|---|---|
| `test_m25_engine_filters_by_persisted_state_and_model` | parity with the authoritative M10 listing filtered by the persisted run state, for two checkpoints; exact `(created_at, suite_run_id)` order; every record belongs to the model AND the checkpoint state; verbatim equality with the M10 single-record getter; the new ck_b run grouped only under ck_b; checkpoint histories disjoint; the CURRENT-state run never appears |
| `test_m25_engine_empty_404s_cross_model_and_read_only` | valid checkpoint with zero runs → `[]`; cross-model **both directions** → `FileNotFoundError` (model-scoped M3 registry); unknown model / unknown checkpoint → `FileNotFoundError`; run-manifest file set unchanged (read-only) |
| `test_m25_engine_repeated_calls_identical` | 4 repeated engine calls produce identical serialized JSON |
| `test_m25_api_by_checkpoint_grouping_parity_and_determinism` | HTTP 200; exactly the two ck_a runs; M10 order; verbatim payloads equal to the POST responses and to the listing filtered by persisted state; current-state run and the other checkpoint's run excluded (with a disjoint second history); repeated GET identical JSON **and raw bytes**; valid checkpoint with no runs → 200 + `[]` (isolated second trained model) |
| `test_m25_api_404s_isolation_and_prior_surfaces` | unknown model → 404 (real checkpoint); unknown checkpoint → 404 (real model); cross-model: another real model + this real checkpoint id → 404; M10 listing/detail unchanged (detail getter not shadowed); **M21 by-suite + M22 summary unchanged**; **M24 evaluations by-checkpoint unchanged** (200, correct checkpoint identity); M23 ghost-policy 404; M18–M20 surface intact; OpenAPI documents the new route (GET only, tag `suite-runs`, `array` of `$ref SuiteRunRecord`) and surface count 57 |

The production-facts tests (prompt §14 Tests 1–4: exactly 10 runs under
`0511de4c7372`, the two empty checkpoints, partition 10+0+0=10) are
fulfilled by the **live smoke** (unit tests run on throwaway temp roots
and cannot see production): smoke checks B1–B6, C1/D1 and E1–E3 assert
them against the authoritative registry.

**Full regression (serial, after implementation): 435 collected, 435
passed, 0 failed, 0 errors, exit 0** (430 pre-existing + 5 new; first
run green — all nine `== 56` sites were updated proactively and
grep-verified before the run; no stale assertion surfaced).

**Static checks**: `compileall` clean (`app`, `tests`, the M25 smoke);
`pyflakes` clean across `app/*.py`, `tests/*.py`, `smoke_m25_live.py` —
**0 new findings** (one unused local in a new test helper was caught by
pyflakes and removed before any test run; the two pre-existing f-string
notes in untouched legacy `smoke_m9_live.py`/`smoke_m11_live.py` remain,
byte-identical to the original upload).

**OpenAPI validation**: `/openapi.json` serves 200 with **57 paths**
(before: 56, verified pre-change); the new path appears **exactly
once**, with a single `get` operation, tag `suite-runs`, and response
schema `array` of `$ref: #/components/schemas/SuiteRunRecord`; all 86
`$ref`s resolve.

## 5. Live Smoke

`smoke_m25_live.py` run against a real HTTP server
(`FORGE_ROOT=/home/user/ai-model-forge-data`, uvicorn on
`127.0.0.1:8746`); server stopped after testing. **34/34 checks PASS,
exit 0**, organized exactly as LIVE A–J:

- **A (baseline)**: 96 files / 4,002,745 B / 0 `.tmp`; SHA256 inventory
  saved; both models resolve; the 3 known checkpoints register (other
  model has none); M10 listing pre-state = exactly the 10 known runs,
  ALL against checkpoint state `0511de4c7372`; M11 workflow history,
  M16/M18–M24 pre-state intact; the **full** dashboard hash equals
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.
- **B (known checkpoint)**: `4a0a871886ef / 0511de4c7372` → **HTTP 200
  with exactly 10 records** — the exact known ids in ASCENDING
  `(created_at, suite_run_id)` order; every record's persisted
  `model_id` + `state.state_kind="checkpoint"` +
  `state.checkpoint_id` verified; **parity with the full M10 listing**
  (all runs of the model used this checkpoint); M10 getter parity for 4
  sampled records; verbatim field set.
- **C/D (empty checkpoints)**: `025e6d8d8f15` → **200 + `[]`**;
  `30a8bc5b82ab` → **200 + `[]`**.
- **E (partition)**: **10 + 0 + 0 = 10** across the three checkpoint
  histories; **no duplicate** suite-run ids; the union equals the full
  M10 listing.
- **F (deterministic repeat)**: repeated GETs **raw-byte-identical**
  (×3).
- **G (unknown model)**: → **404**.
- **H (unknown checkpoint)**: well-formed unknown id → **404**;
  malformed (percent-encoded) id → **404** without crashing.
- **I (cross-model isolation)**: `b5bc905326b6` + the real
  `0511de4c7372` → **404** (the authoritative M3 registry is
  model-scoped and the other model owns no checkpoints — the correct
  expected semantics, verified rather than guessed); none of the 10 run
  ids is reachable under the other model; by-checkpoint does not shadow
  the M10 detail getter (real run id → 200, ghost id → 404).
- **J (regression + final audit)**: M18 records / M19 by-sample / M20
  by-checkpoint / M21 by-suite / M22 summary / M24
  evaluations-by-checkpoint / M23 by-policy outputs unchanged; M17
  dashboard hash preserved; **M10 listing, M11 workflow history, M3
  checkpoint registry and M9 policy registry byte-identical to
  pre-state**; by-checkpoint still deterministic at the end; storage
  diff: **0 changed, 0 missing, 0 new files, 96 / 4,002,745 / 0
  `.tmp`**.

The smoke **fails loudly** (exit 1 + named failures) if any requirement
is wrong.

## 6. Storage Integrity

| Audit | Before | After |
|---|---|---|
| Production files | 96 | **96** (0 new, 0 missing) |
| Production bytes | 4,002,745 | **4,002,745** (0 bytes growth) |
| `.tmp` files | 0 | **0** |
| Changed files (SHA256) | — | **0** — every one of the 96 pre-existing hashes identical (`m25_pre.sha256` diff empty; re-verified after the /tmp-cleanup rerun) |

M25 created **zero production persistence artifacts** — no database,
index, cache, history file or new directory; the smoke performs the
before/after per-file SHA256 comparison (LIVE A inventory vs LIVE J
audit) in addition to the shell-level audit. Determinism: repeated live
GETs return **raw-byte-identical** bodies; the engine produces identical
serialized listings across repeated calls; ordering is the exact M10
`(created_at, suite_run_id)` convention; identity comes only from the
persisted nested `state` fields, never filename inference.

`/tmp` note: serial test runs create `forge-tests-*` session roots
(~145 MB each, from `tests/conftest.py`'s `tempfile.mkdtemp`). After
confirming no test/server process was running, **4 stale roots were
removed**, the full suite was **rerun (435 passed)** and the production
inventory re-verified byte-identical.

## 7. Regression / Compatibility

- **M1–M24 intact**: full regression 430/430 pre-existing tests pass
  (only the nine documented OpenAPI count assertions 56 → 57 changed);
  live-verified unchanged on production: M10 suite-run listing (10
  records, byte-identical) and detail getter, **M11 workflow history
  (byte-identical)**, M3 checkpoint registry, M9 policy registry, M13
  dashboard hash `f48557fe...`, M16/M17 sample-quality surface, M18
  records / M19 by-sample / M20 by-checkpoint, M21 by-suite (10
  records), M22 summary (count 10), M23 by-policy (1 decision), M24
  evaluations by-checkpoint (3 records) — all byte-identical
  before/after (smoke checks J1–J3).
- **M25 is additive only**: one engine method + one composed M3 engine
  reference + one facade method + one route + tests + smoke + docs. No
  existing route, payload, manifest, ordering, storage format or
  behavior changed. The literal `by-checkpoint` segment is registered
  before `/{suite_run_id}` so no route shadowing is possible (verified
  live).
- **Known edges**: current-state suite runs intentionally never appear
  (persisted `state.checkpoint_id` is `null`); a corrupt suite-run
  manifest is skipped by the existing M10 listing semantics (unchanged);
  production has no suite run outside `0511de4c7372`, so the `200 + []`
  live cases rest on the two genuinely empty checkpoints — which exist
  naturally and required no production mutation.

## 8. Commit / Final Status

- Commit: **`M25: add suite-run history by checkpoint`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge`; local HEAD == remote HEAD
  (diff-verified), working tree clean after the push (commit hash
  recorded in the git log; no unrelated commits amended).
- Files changed: `app/suite_runs.py`, `app/engine.py`, `app/api.py`,
  `tests/test_suite_runs.py`, `tests/test_sample_quality.py`,
  `tests/test_dashboards.py`, `tests/test_gates_api.py`,
  `tests/test_evaluation_api.py`, `README.md`, `smoke_m25_live.py`
  (new), `m25_pre.sha256` (new, audit artifact),
  `M25_final_report.md` (new).
- **Final status: M25 COMPLETE AND CERTIFIED.** All gates pass:
  435/435 tests serially (430 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 56 → **57 paths** with the
  endpoint documented exactly once; live smoke **34/34 PASS** on
  production (exactly 10 runs under `0511de4c7372`, `200 + []` for both
  empty checkpoints, partition 10+0+0=10 with no duplication, clean
  404s, cross-model isolation, byte-identical repeats); storage audit
  **96 files / 4,002,745 bytes / 0 `.tmp` / 0 changed / 0 new /
  0 missing / 0 bytes growth**; dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 26 — COMPARISON HISTORY BY CHECKPOINT

Continue the existing **AI Model Forge** project.

M25 is the current certified milestone.

The goal of M26 is to add one small, read-only API capability for
inspecting existing M5 comparison history grouped by the checkpoint
states it compared.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and verify
the actual M25 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 435 tests passing
* OpenAPI: 57 paths
* M25 suite-runs by-checkpoint endpoint working
* M24 evaluations by-checkpoint endpoint working
* M23 gate decisions by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working

Also inspect the existing implementations for:

* the M5 comparison engine (`ComparisonEngine`): `list_comparisons` /
  `get_comparison` — the authoritative listing and its deterministic
  `(created_at, comparison_id)` ordering
* the persisted `ComparisonRecord.state_a` / `state_b` fields (each a
  `ComparisonState`: `state_kind` + nullable `checkpoint_id`)
* the M3 checkpoint registry (`TrainingEngine.get_checkpoint`) — the
  same model-scoped ownership validation M20/M24/M25 use
* M18–M25 read-only API/engine patterns (registry validation +
  authoritative-listing filter + thin facade + thin route registered
  before any literal-shadowing path)

Create a SHA256 inventory of the production files before making changes.

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M26 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/comparisons/by-checkpoint/{checkpoint_id}`

The endpoint should answer:

> Which existing immutable M5 comparison records of this model involve
> this checkpoint state (as side A and/or side B)?

A comparison involves the checkpoint when EITHER persisted side records
`state_kind == "checkpoint"` with that `checkpoint_id`. Current-state
sides (`checkpoint_id` null) never match.

Reuse the existing M5 comparison records, the existing M3 checkpoint
registry and the existing listing logic.

Do not create a second comparison system.

---

## 3. PRODUCTION FACTS (verified at M25)

Model `4a0a871886ef` owns 8 comparisons. By involved checkpoint:

* `025e6d8d8f15` — 6 comparisons: `fc379bfcb50f`, `baa361012e00`
  (A=30a8bc5b82ab, B=025e6d8d8f15), `d683f9b81195` (A=current,
  B=025e6d8d8f15), `786de08efe4c` (A=B=025e6d8d8f15, unchanged),
  `d9a62dde016b` and `d62f89e97c85` (A=025e6d8d8f15,
  B=30a8bc5b82ab, regressed)
* `30a8bc5b82ab` — 5 comparisons: `fc379bfcb50f`, `baa361012e00`,
  `d9a62dde016b`, `d62f89e97c85`, `729f9c55ea89` (A=B=30a8bc5b82ab,
  unchanged)
* `0511de4c7372` — 1 comparison: `5c5ff22151ed` (A=current,
  B=0511de4c7372, unchanged)
* `d683f9b81195` and `5c5ff22151ed` each involve one current-state side
  — those sides never match any checkpoint id.

Resolve these facts from the authoritative listing at baseline time; do
not assume them from this prompt.

---

## 4. REQUIRED BEHAVIOR

The endpoint must:

* verify the model exists
* verify the checkpoint exists using the model's M3 checkpoint registry
  (an unknown checkpoint, or a checkpoint id belonging to another model,
  is 404 — nothing is inferred from filenames)
* return only comparison records of the requested model where the
  persisted `state_a` OR `state_b` is a checkpoint state with the
  requested `checkpoint_id`
* preserve the existing `ComparisonRecord` representation (verbatim
  payloads, verdict/losses included)
* use deterministic ordering consistent with the existing comparison
  listing behavior (`(created_at, comparison_id)` ASCENDING)
* return `[]` when a valid checkpoint is involved in no comparison
* return 404 for an unknown model
* return 404 for an unknown checkpoint
* prevent cross-model data leakage
* be completely read-only; never infer identity from filenames — use
  the persisted side states

---

## 5. ENGINE IMPLEMENTATION

Inspect the current comparison engine first.

If necessary, add one small method such as:

`list_comparisons_for_checkpoint(model_id, checkpoint_id)`

The method should reuse the existing `list_comparisons` implementation
and the M3 `get_checkpoint` registry getter (exactly the ownership
validation pattern M20/M24/M25 established).

Do not introduce:

* a new database
* an index
* a cache
* another manifest format
* another comparison engine
* new production storage

Keep the implementation minimal. The API route should be a thin wrapper
around the engine.

---

## 6. TESTS

Add focused tests for:

1. Production facts (live smoke): the counts above — 6 / 5 / 1
   per checkpoint (recompute from the authoritative listing).
2. Every returned record involves the requested checkpoint on at least
   one persisted side.
3. Returned payloads match the existing comparison records verbatim.
4. Ordering is deterministic (`(created_at, comparison_id)` ASCENDING).
5. A comparison with A=B=same checkpoint appears exactly ONCE (no
   double-counting).
6. Valid checkpoint with no comparisons returns `[]` (unit fixture).
7. Unknown model returns 404.
8. Unknown checkpoint returns 404.
9. Cross-model access cannot expose another model's comparisons
   (`b5bc905326b6` + a real checkpoint id of `4a0a871886ef` -> 404).
10. Repeated requests return identical JSON.
11. Existing comparison APIs (run/listing/get) remain unchanged.
12. M16–M25 history endpoints remain unchanged.
13. New OpenAPI route is documented correctly (58 paths).

Do not weaken existing tests.

---

## 7. LIVE TEST

Create:

`smoke_m26_live.py`

Run a real HTTP server using:

`FORGE_ROOT=/home/user/ai-model-forge-data`

Test all three real checkpoints of `4a0a871886ef`.

Expected:

* `025e6d8d8f15` -> 200 with the 6 known comparison ids
* `30a8bc5b82ab` -> 200 with the 5 known comparison ids
* `0511de4c7372` -> 200 with exactly 1 record (`5c5ff22151ed`)
* parity with the existing M5 listing filtered by the persisted sides
* `786de08efe4c` / `729f9c55ea89` appear exactly once each
* deterministic repeated response (raw bytes)

Also test:

* unknown model -> 404
* unknown checkpoint -> 404
* cross-model isolation (`b5bc905326b6` + `025e6d8d8f15` -> 404)
* existing M25/M24/M23/M22/M21/M20/M19/M18 endpoints still work
* dashboard result hash unchanged (`f48557fe...` full value)

The smoke test must fail if any requirement is incorrect.

Stop the server after testing.

---

## 8. FINAL AUDIT

After implementation:

Run the full test suite serially.

Run:

* compileall
* pyflakes
* OpenAPI validation
* live smoke

Then compare the production directory against the pre-change SHA256
inventory.

M26 is read-only, so the expected production state must remain:

* 96 files
* 4,002,745 bytes
* 0 `.tmp`
* 0 changed files
* 0 new production files
* 0 missing production files
* 0 bytes of production storage growth

All previous production hashes must remain identical.

If temporary test files consume `/tmp`, clean only stale test artifacts
after confirming no test process is running, then rerun the affected
test.

---

## 9. KEEP THE SCOPE SMALL

M26 must NOT add:

* model training
* retraining
* LoRA/QLoRA/PEFT
* DPO/RL/RLHF/RLAIF
* HPO
* automatic model improvement
* automatic checkpoint selection
* quality scoring
* rankings
* leaderboards
* Gemini/external AI
* frontend redesign
* database
* background workers
* scheduling
* distributed execution
* new comparison execution, checkpoint mutation or verdict computation
  of any kind

M26 is only:

**one read-only checkpoint-grouping endpoint over existing comparison
records.**

Do not implement future milestones early.

---

## 10. COMMIT / PUSH / REPORT

Create a focused commit:

`M26: add comparison history by checkpoint`

Push to `arena/01a071e9-code-forge`; verify local HEAD == remote HEAD
and a clean working tree.

Then write `M26_final_report.md` with exactly these nine sections:

1. Objective
2. Baseline
3. Implementation
4. Tests
5. Live Smoke
6. Storage & Determinism
7. Audit / Limitations
8. Final Result (commit hash / branch / remote sync / tree status)
9. Next Milestone (complete copy-ready prompt for M27)

The M27 prompt must also require:

* baseline verification first
* minimal implementation
* focused tests
* full regression
* static checks
* OpenAPI validation
* live HTTP smoke
* deterministic behavior
* production SHA256 audit
* zero unintended production storage changes
* exact nine-section final report
* automatic M28 prompt generation

Do not declare M26 complete unless all tests, static checks, smoke
tests, and final storage audits pass.
```
