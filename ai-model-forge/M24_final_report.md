# MILESTONE 24 — EVALUATION HISTORY BY CHECKPOINT — FINAL REPORT

## 1. Objective

Add exactly ONE deterministic, read-only endpoint to the AI Model Forge
REST API:

```
GET /api/v1/models/{model_id}/evaluations/by-checkpoint/{checkpoint_id}
```

It answers one question: *which immutable M4 evaluation records measured
this checkpoint of this model?* The implementation is a pure reuse of
the existing authoritative registries: the checkpoint is resolved through
the model's M3 checkpoint registry (`TrainingEngine.get_checkpoint` — the
same model-scoped ownership path M20 uses; an unknown checkpoint, or a
checkpoint id belonging to another model, is a 404), and the response is
the model's authoritative M4 listing (`EvaluationEngine.list_evaluations`)
filtered by the **persisted** `checkpoint_id` recorded inside each
`EvaluationRecord` — never inferred from evaluation directory names, ids,
timestamps or filenames. Records are returned verbatim in the exact M4
deterministic order `(created_at, eval_id)` ASCENDING. Current-state
evaluations persist `checkpoint_id = null` and therefore never appear. A
valid checkpoint with no evaluations returns `[]` (never 404); an unknown
model or checkpoint returns 404. No training, rollback, scoring changes,
rankings, aggregation, caches, indexes, databases, workers or new
persistence structures — strictly an evaluation-history observability
flow.

Verified live on production (`FORGE_ROOT=/home/user/ai-model-forge-data`):
`4a0a871886ef` returns exactly **3 records per checkpoint (3+3+3 = 9)**,
payload- and order-identical to the existing M4 listing, byte-identical
on repeated GETs, with zero production storage changes.

## 2. Baseline

Environment verified at the authoritative M23 state before any change:
git at **`f78da99`** ("M23: add gate decision history by policy") on a
clean working tree of `arena/01a071e9-code-forge`; production root and
venv intact (no sandbox reset this time).

Actual pre-change measurements (all verified **before** touching code):

| Check | Expected | Actual |
|---|---|---|
| Production files | 96 | **96** |
| Production bytes | 4,002,745 | **4,002,745** |
| `.tmp` files | 0 | **0** |
| Full test suite (serial) | 425 passing | **425 passed / 0 failed / 0 errors**, exit 0 |
| OpenAPI paths | 55 | **55** (`evaluations/by-checkpoint` absent pre-change, verified) |
| `compileall` | clean | **clean** |
| `pyflakes` | clean | **clean** (`app/*.py` + `tests/*.py`) |
| Dashboard result hash | `f48557fe...` | **`f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`** (verified live in the smoke's Phase A) |

Pre-change SHA256 inventory of all 96 production files created
(`m24_pre.sha256`; verified identical to the certified M23 inventory).

Evaluation facts resolved from the **authoritative production registry**
(not assumed from documentation) — exactly as expected:

- Model `4a0a871886ef`: **16 total evaluations** — **9
  checkpoint-grouped** under exactly **3 checkpoints** (3 each) plus
  **7 current-state** evaluations
- Exact checkpoints and evaluation ids (ASCENDING `(created_at,
  eval_id)`):
  - `025e6d8d8f15` → `7a16eaa12120, 90aa392b9a2b, ebb9b7eccbe3`
  - `0511de4c7372` → `c739c66638e9, 0704399fea7b, 75a23351e91c`
  - `30a8bc5b82ab` → `b0502d871114, 75835b64d6af, fe7b42cdb411`
- Current-state ids (must never appear): `a439eb92f9cd, 0cc96a125976,
  425003213a0b, a884bf729ff7, b89a94306ce8, 340f5adbf881, c35a1c9902fe`
- Second model `b5bc905326b6`: exists, owns **no** evaluations and **no**
  checkpoints (cross-model probe)

Architecture findings from inspection: `EvaluationRecord` persists
`state_kind: EvalStateKind` + `checkpoint_id: Optional[str] = None` (the
nullable identity respected verbatim); `EvaluationEngine.list_evaluations`
is the authoritative listing (model check → scan
`models/<id>/evaluations/eval-*/manifest.json` → parse → sort
`(created_at, eval_id)` ASCENDING → `[]` when none → never writes);
`EvaluationEngine.__init__` already holds
`self.training = TrainingEngine(storage)` (checkpoint verification
reuse), so the M3 ownership lookup needed no new wiring; the facade
delegates `list_evaluations`/`get_evaluation` thinly; the API exposes
`GET /models/{id}/evaluations` and
`GET /models/{id}/evaluations/{eval_id}` — the new literal
`by-checkpoint` segment is registered **before** the detail route.

## 3. Implementation

Repository files changed only — **no production artifact was ever
touched**, no new schema (`EvaluationRecord` is returned directly), no
new persistence structure:

1. **`app/evaluation.py`** — new engine method
   `list_evaluations_for_checkpoint(model_id, checkpoint_id)` placed
   after the M4 getters:
   `self.training.get_checkpoint(model_id, checkpoint_id)` (existence +
   model-scoped ownership → `FileNotFoundError`), then
   `[r for r in self.list_evaluations(model_id)
   if r.checkpoint_id == checkpoint_id]`. Reuses the existing parsing,
   model validation, checkpoint ownership validation, deterministic
   sorting and corruption handling. Net diff: exactly **30 insertions**.
   *Full disclosure:* one intermediate edit accidentally dropped the
   `if not path.exists():` line of the adjacent `get_evaluation`; it was
   restored in the immediately following edit and the final `git diff`
   confirms `get_evaluation` is byte-identical to M23 (the broken
   intermediate state never executed any test).
2. **`app/engine.py`** — thin facade `list_evaluations_for_checkpoint`
   with a docstring, delegating 1:1 to the engine (same shape as the
   M21–M23 facades).
3. **`app/api.py`** — thin route
   `GET /models/{model_id}/evaluations/by-checkpoint/{checkpoint_id}`
   (`response_model=list[EvaluationRecord]`, tag `evaluation`,
   `FileNotFoundError → 404`), registered **before**
   `/models/{model_id}/evaluations/{eval_id}` so the literal
   `by-checkpoint` segment can never be mistaken for an evaluation id;
   landing-page milestone list + REST-API route list updated.
4. **`tests/test_evaluation.py`** — new "M24" section: 3 focused engine
   tests (13 → 16 functions) with a cached `_m24_env` builder (model
   with three checkpoints — evals on two of them plus one current-state
   eval — and a second model with its own checkpoint) and a read-only
   file-snapshot helper.
5. **`tests/test_evaluation_api.py`** — new "M24" section: 2 focused
   API tests (5 → 7 functions): grouping/parity/determinism/empty-case
   and 404s/isolation/prior-surfaces/OpenAPI.
6. **`tests/test_sample_quality.py`** + **`tests/test_dashboards.py`** +
   **`tests/test_suite_runs.py`** + **`tests/test_gates_api.py`** — the
   **eight** factual OpenAPI surface-count assertions updated 55 → 56
   with refreshed comments (documented surface change; **no existing
   assertion weakened or otherwise altered**; grep-verified: 0 remaining
   `== 55`, 8 new `== 56` — learning from the M23 miss).
7. **`README.md`** — "Milestone 24" section; test counts 425 → 430 (two
   sites).
8. **`smoke_m24_live.py`** — new live production smoke (33 checks, LIVE
   TESTS A–I), modeled on the M20–M23 smokes.
9. **`m24_pre.sha256`** — pre-change SHA256 inventory of all 96
   production files (audit artifact; identical to `m23_pre.sha256`).

## 4. Tests

**Focused M24 tests** (3 engine + 2 API, all green):

| Test | Covers |
|---|---|
| `test_m24_engine_filters_by_persisted_checkpoint_and_model` | every record belongs to the requested model AND checkpoint with `state_kind=checkpoint`; exact M4 order; verbatim payload parity with the M4 listing **and** single getter; a second checkpoint returns only its own evaluations; the current-state evaluation (`checkpoint_id=None`) never appears |
| `test_m24_engine_empty_404s_cross_model_and_read_only` | valid checkpoint with no evaluations → `[]`; the other model's checkpoint history is separate and disjoint; **cross-model both directions** (model A cannot query B's checkpoint id and vice versa → `FileNotFoundError` via the model-scoped M3 registry); unknown model / unknown checkpoint → `FileNotFoundError`; evaluation-manifest file set unchanged (read-only) |
| `test_m24_engine_repeated_calls_identical` | 4 repeated engine calls produce identical serialized JSON |
| `test_m24_api_by_checkpoint_grouping_parity_and_determinism` | HTTP 200; exactly the 2 checkpoint evaluations of 3 total runs; M4 order; verbatim payloads equal to the run responses and to the listing filtered by the persisted checkpoint identity; the other checkpoint's evaluation and the current-state evaluation stay outside; repeated GET identical JSON **and raw bytes**; valid checkpoint with no evaluations → 200 + `[]` (isolated fixture) |
| `test_m24_api_404s_isolation_and_prior_surfaces` | unknown model → 404 (real checkpoint); unknown checkpoint → 404 (real model); **cross-model: another real model + this real checkpoint id → 404**; M4 listing/detail unchanged (detail getter not shadowed, ghost id 404); M3 checkpoint registry unchanged; M20/M23 regression (sample-quality by-checkpoint and gates by-policy semantics intact); OpenAPI documents the new route (GET only, tag `evaluation`, `array` of `$ref EvaluationRecord`) and surface count 56 |

The production-facts tests (Tests 1–3 of the prompt: exactly 3
evaluations per checkpoint, 3+3+3=9 across the three real checkpoints,
current-state exclusion) are fulfilled by the **live smoke** (unit tests
run on throwaway temp roots and cannot see production): smoke checks
B1–B6, C1–C4 and D1–D2 assert them against the authoritative registry.

**Full regression (serial, after implementation): 430 collected, 430
passed, 0 failed, 0 errors, exit 0** (425 pre-existing + 5 new; no stale
count assertion surfaced this time — all eight `== 55` sites were
updated proactively and grep-verified before the run).

**Static checks**: `compileall` clean (`app`, `tests`, the M24 smoke);
`pyflakes` clean across `app/*.py`, `tests/*.py`, `smoke_m24_live.py` —
**0 new findings** (the two pre-existing f-string notes in untouched
legacy `smoke_m9_live.py`/`smoke_m11_live.py` remain, byte-identical to
the original upload).

**OpenAPI validation**: `/openapi.json` serves 200 with **56 paths**
(before: 55); the new path appears **exactly once**, with a single `get`
operation, tag `evaluation`, and response schema `array` of
`$ref: #/components/schemas/EvaluationRecord`; all 86 `$ref`s resolve.

## 5. Live Smoke

`smoke_m24_live.py` run against a real HTTP server
(`FORGE_ROOT=/home/user/ai-model-forge-data`, uvicorn on
`127.0.0.1:8745`); server stopped after testing. **33/33 checks PASS,
exit 0**, organized exactly as LIVE TESTS A–I:

- **A (baseline)**: 96 files / 4,002,745 B / 0 `.tmp`; SHA256 inventory
  saved; both models resolve; the 3 known checkpoints register; M4
  listing pre-state = 16 evaluations (9 checkpoint-grouped under the 3
  checkpoints, 7 current-state) in `(created_at, eval_id)` order; other
  model has none; M16/M18/M19/M20/M21/M22/M23 pre-state intact;
  dashboard hash `f48557fe8ab1...`.
- **B (first known checkpoint `0511de4c7372`)**: **HTTP 200 with exactly
  3 records** — `c739c66638e9, 0704399fea7b, 75a23351e91c` in ASCENDING
  order; every record's persisted `model_id` / `state_kind=checkpoint` /
  `checkpoint_id` verified; **parity** with the M4 listing filtered by
  the persisted checkpoint identity and with each record's M4 single
  getter (3/3); verbatim field set with metrics present.
- **C (all three checkpoints)**: each of `025e6d8d8f15`,
  `0511de4c7372`, `30a8bc5b82ab` → 200 with **exactly its 3
  evaluations**; **3+3+3 = 9 total; no duplication** between checkpoint
  histories; the union equals the listing's checkpoint-grouped set.
- **D (current-state exclusion)**: the **7 current-state ids appear in
  NO checkpoint response**; 9 + 7 = 16 total.
- **E (deterministic repeat)**: repeated GETs **raw-byte-identical**
  (×3).
- **F (valid empty case)**: production has **no** unevaluated checkpoint
  (all 3 respond 200, none 404) — the `200 + []` case is proven by the
  isolated unit/API fixtures; production was **not** mutated to
  manufacture it.
- **G (errors)**: unknown model → **404**; unknown well-formed checkpoint
  id → **404**; malformed (percent-encoded) checkpoint id → **404**
  without crashing.
- **H (cross-model isolation)**: `b5bc905326b6` + `4a0a871886ef`'s real
  checkpoint id → **404** (model-scoped M3 registry); no evaluation of
  `4a0a871886ef` is reachable under the other model; by-checkpoint does
  not shadow the M4 detail getter (real eval id → 200, ghost id → 404).
- **I (regression + final audit)**: M18 records / M19 by-sample / M20
  by-checkpoint / M21 by-suite / M22 summary / M23 by-policy outputs
  unchanged; M17 dashboard hash preserved; **M4 listing, M3 checkpoint
  registry and M9 policy registry byte-identical to pre-state**;
  by-checkpoint still deterministic at the end; storage diff: **0
  changed, 0 missing, 0 new files, 96 / 4,002,745 / 0 `.tmp`**.

The smoke **fails loudly** (exit 1 + named failures) if any requirement
is wrong.

## 6. Storage Integrity

| Audit | Before | After |
|---|---|---|
| Production files | 96 | **96** (0 new, 0 missing) |
| Production bytes | 4,002,745 | **4,002,745** (0 bytes growth) |
| `.tmp` files | 0 | **0** |
| Changed files (SHA256) | — | **0** — every one of the 96 pre-existing hashes identical (`m24_pre.sha256` diff empty; re-verified after the /tmp-cleanup rerun) |

No cache files, temporary manifests, indexes, databases, generated
analytics or logs were written into production storage; the smoke
performs the before/after per-file SHA256 comparison (Phase A inventory
vs Phase I audit) in addition to the shell-level audit. Determinism:
repeated live GETs return **raw-byte-identical** bodies; the engine
produces identical serialized listings across repeated calls; ordering
is the exact M4 `(created_at, eval_id)` convention; identity comes only
from the persisted `checkpoint_id` field, never filename inference.

`/tmp` note: serial test runs create `forge-tests-*` session roots
(~145 MB each, from `tests/conftest.py`'s `tempfile.mkdtemp`). After
confirming no test/server process was running, **4 stale roots were
removed**, the full suite was **rerun (430 passed)** and the production
inventory re-verified byte-identical.

## 7. Regression / Compatibility

- **M1–M23 intact**: full regression 425/425 pre-existing tests pass
  (only the eight documented OpenAPI count assertions 55 → 56 changed);
  live-verified unchanged on production: M4 evaluation listing (16
  records, byte-identical) and detail getter, M3 checkpoint registry
  (byte-identical), M9 policy registry, M13 dashboard hash
  `f48557fe...`, M16/M17 sample-quality surface, M18 records / M19
  by-sample / M20 by-checkpoint (2 records, identical payloads), M21
  by-suite (10 records), M22 summary (count 10), M23 by-policy (1
  decision) — all byte-identical before/after (smoke checks I1–I3).
- **M24 is additive only**: one engine method + one facade method + one
  route + tests + smoke + docs. No existing route, payload, manifest,
  ordering, storage format or behavior changed. The literal
  `by-checkpoint` segment is registered before `/{eval_id}` so no route
  shadowing is possible (verified live).
- **Known edges**: current-state evaluations intentionally never appear
  (persisted `checkpoint_id` is `null`); a corrupt evaluation manifest
  is skipped by the existing M4 listing semantics (unchanged); every
  production checkpoint happens to have evaluations, so the `200 + []`
  empty case rests on the isolated fixtures by design — production was
  not mutated to manufacture it.

## 8. Commit / Final Status

- Commit: **`M24: add evaluation history by checkpoint`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge` (commit hash recorded in the git
  log; no unrelated commits amended). Working-tree status after push:
  **clean** (only this report and the audit inventory added).
- Files changed: `app/evaluation.py`, `app/engine.py`, `app/api.py`,
  `tests/test_evaluation.py`, `tests/test_evaluation_api.py`,
  `tests/test_sample_quality.py`, `tests/test_dashboards.py`,
  `tests/test_suite_runs.py`, `tests/test_gates_api.py`, `README.md`,
  `smoke_m24_live.py` (new), `m24_pre.sha256` (new, audit artifact),
  `M24_final_report.md` (new).
- **Final status: M24 COMPLETE AND CERTIFIED.** All gates pass:
  430/430 tests serially (425 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 55 → **56 paths** with the
  endpoint documented exactly once; live smoke **33/33 PASS** on
  production (exactly 3 evaluations per checkpoint, 3+3+3 = 9 with no
  duplication, current-state exclusion, `200 + []` semantics, clean
  404s, cross-model isolation, byte-identical repeats); storage audit
  **96 files / 4,002,745 bytes / 0 `.tmp` / 0 changed / 0 new /
  0 missing / 0 bytes growth**; dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 25 — SUITE-RUN HISTORY BY CHECKPOINT

Continue the existing **AI Model Forge** project.

M24 is the current certified milestone.

The goal of M25 is to add one small, read-only API capability for
inspecting existing suite-run history grouped by the model state
(checkpoint) each run executed against.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and verify
the actual M24 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 430 tests passing
* OpenAPI: 56 paths
* M24 evaluations by-checkpoint endpoint working
* M23 gate decisions by-policy endpoint working
* M22 suite-run summary endpoint working
* M21 suite-run by-suite endpoint working
* M20/M19/M18 sample-quality endpoints working

Also inspect the existing implementations for:

* the M10 suite-run engine, especially the M21
  `list_suite_runs_for_suite` filter and the authoritative
  `list_suite_runs` listing with its `(created_at, suite_run_id)`
  ordering
* the persisted `SuiteRunRecord.state` field
  (`ComparisonState`: `state_kind` + nullable `checkpoint_id`) —
  checkpoint-state runs carry the id; current-state runs carry null
* the M3 checkpoint registry (`TrainingEngine.get_checkpoint`) — the
  same model-scoped ownership validation M20 and M24 use
* M18–M24 read-only API/engine patterns (registry validation +
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

## 2. M25 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/suite-runs/by-checkpoint/{checkpoint_id}`

The endpoint should answer:

> Which existing immutable suite-run records of this model executed
> against this checkpoint state?

Reuse the existing M10 suite-run records, the existing M3 checkpoint
registry and the existing listing logic.

Do not create a second suite-run system.

---

## 3. REQUIRED BEHAVIOR

The endpoint must:

* verify the model exists
* verify the checkpoint exists using the model's M3 checkpoint registry
  (an unknown checkpoint, or a checkpoint id belonging to another model,
  is 404 — nothing is inferred from filenames)
* return only suite-run records of the requested model whose persisted
  `state.state_kind == "checkpoint"` and
  `state.checkpoint_id` equals the requested id
* preserve the existing `SuiteRunRecord` representation (verbatim
  payloads)
* use deterministic ordering consistent with the existing suite-run
  listing behavior (`(created_at, suite_run_id)` ASCENDING)
* never return current-state suite runs (their persisted
  `state.checkpoint_id` is null)
* return `[]` when a valid checkpoint has no suite runs
* return 404 for an unknown model
* return 404 for an unknown checkpoint
* prevent cross-model data leakage
* be completely read-only; never infer identity from filenames — use the
  persisted state recorded in each run

---

## 4. ENGINE IMPLEMENTATION

Inspect the current suite-run engine first.

If necessary, add one small method such as:

`list_suite_runs_for_checkpoint(model_id, checkpoint_id)`

The method should reuse the existing `list_suite_runs` implementation
and the M3 `get_checkpoint` registry getter (exactly the ownership
validation pattern M20/M24 established).

Do not introduce:

* a new database
* an index
* a cache
* another manifest format
* another suite-run engine
* new production storage

Keep the implementation minimal. The API route should be a thin wrapper
around the engine.

---

## 5. TESTS

Add focused tests for:

1. Production facts (live smoke): model `4a0a871886ef` owns 10 suite
   runs, ALL recorded under checkpoint state `0511de4c7372` (state_kind
   "checkpoint"); the other two checkpoints (`025e6d8d8f15`,
   `30a8bc5b82ab`) are valid-but-empty -> 200 + [].
2. Every returned record belongs to the requested model and checkpoint
   state.
3. Returned payloads match the existing suite-run records verbatim.
4. Ordering is deterministic (`(created_at, suite_run_id)` ASCENDING).
5. Valid checkpoint with no suite runs returns `[]` (both a unit
   fixture and the two live production empty checkpoints).
6. Current-state suite runs never appear (unit fixture).
7. Unknown model returns 404.
8. Unknown checkpoint returns 404.
9. Cross-model access cannot expose another model's runs
   (`b5bc905326b6` + `0511de4c7372` -> 404).
10. Repeated requests return identical JSON.
11. Existing suite-run APIs (run/listing/detail/by-suite/summary) remain
    unchanged.
12. M16–M24 sample-quality, suite-run and gate/evaluation grouping APIs
    remain unchanged.
13. New OpenAPI route is documented correctly (57 paths).

Do not weaken existing tests.

---

## 6. LIVE TEST

Create:

`smoke_m25_live.py`

Run a real HTTP server using:

`FORGE_ROOT=/home/user/ai-model-forge-data`

Test `4a0a871886ef / 0511de4c7372`.

Expected:

* HTTP 200
* exactly 10 records (every production suite run of the model)
* parity with the existing M21 by-suite / M10 listing filtered by the
  persisted checkpoint state
* deterministic repeated response (raw bytes)

Also test:

* `025e6d8d8f15` and `30a8bc5b82ab` -> 200 + [] (valid, no runs)
* unknown model -> 404
* unknown checkpoint -> 404
* cross-model isolation (`b5bc905326b6` + `0511de4c7372` -> 404)
* existing M24 by-checkpoint (evaluations) endpoint still works
* existing M23/M22/M21/M20/M19/M18 endpoints still work
* dashboard result hash unchanged (`f48557fe...`)

The smoke test must fail if any requirement is incorrect.

Stop the server after testing.

---

## 7. FINAL AUDIT

After implementation:

Run the full test suite serially.

Run:

* compileall
* pyflakes
* OpenAPI validation
* live smoke

Then compare the production directory against the pre-change SHA256
inventory.

M25 is read-only, so the expected production state must remain:

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

## 8. KEEP THE SCOPE SMALL

M25 must NOT add:

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
* new suite-run execution, checkpoint mutation or state comparison of
  any kind

M25 is only:

**one read-only checkpoint-grouping endpoint over existing suite-run
records.**

Do not implement future milestones early.

---

## 9. FINAL REPORT

At the end, write `M25_final_report.md` with exactly these nine
sections:

1. Objective
2. Baseline
3. Implementation
4. Tests
5. Live Smoke
6. Storage & Determinism
7. Audit / Limitations
8. Final Result
9. Next Milestone (complete copy-ready prompt for M26)

The M26 prompt must also require:

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

Do not declare M25 complete unless all tests, static checks, smoke
tests, and final storage audits pass.
```
