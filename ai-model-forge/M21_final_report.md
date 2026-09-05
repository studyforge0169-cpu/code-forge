# MILESTONE 21 — SUITE-RUN HISTORY BY SUITE — FINAL REPORT

## §1 Summary

M21 added exactly **one read-only endpoint** to the AI Model Forge REST API:

```
GET /api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}
```

It answers one question: *which existing immutable suite-run records belong
to this model and this named suite?* The implementation is a pure reuse of
the existing M10 suite-run history: the suite is resolved through the
existing M9 probe-suite registry (`PolicyEngine.get_suite`), the model
through the existing model registry, and the response is the model's
authoritative M10 listing (`SuiteRunEngine.list_suite_runs`) filtered by the
**persisted** `suite_id` recorded inside each `SuiteRunRecord` — never
inferred from run-directory names. Records are returned verbatim in the
exact M10 deterministic order `(created_at, suite_run_id)` ASCENDING. A
valid suite with no runs for the model returns `[]`; an unknown model or an
unknown suite returns 404; another model's runs are unreachable. No new
engine, store, index, cache, manifest format, worker, statistic, score or
ranking was added — the route is a thin wrapper around one small engine
method, and the whole operation never writes.

Verified live on production (`FORGE_ROOT=/home/user/ai-model-forge-data`):
`4a0a871886ef / m9-live-suite` → HTTP 200 with exactly 10 records, payload-
and order-identical to the existing M10 listing, byte-identical on repeated
GETs, with zero production storage changes.

## §2 Baseline

Actual pre-change state (all values verified **before** touching code):

| Check | Expected | Actual |
|---|---|---|
| Production root | `/home/user/ai-model-forge-data` | ✔ |
| Production files | 96 | **96** |
| Production bytes | 4,002,745 | **4,002,745** |
| `.tmp` files | 0 | **0** |
| Full test suite (serial) | 408 passing | **408 passed / 0 failed / 0 errors**, exit 0 |
| OpenAPI paths | 52 | **52** |
| `compileall` | clean | **clean** |
| `pyflakes` | clean | **clean on the M21-touched surface** (two pre-existing f-string notes in untouched legacy `smoke_m9_live.py` / `smoke_m11_live.py`, byte-identical findings in the pristine baseline) |
| M20 by-checkpoint endpoint | working | **live 200** (2 records, checkpoint `0511de4c7372`) |
| M19 by-sample endpoint | working | **live 200** (sample `f8e66f9c7b50`) |
| M18 sample-quality records | working | **live 200** (2 records) |

A **pre-change SHA256 inventory of all 96 production files** was created
(`m21_pre.sha256`, kept in the repo) before any modification.

Architecture findings from inspecting the existing implementation:

- **M9 suite definitions**: `PolicyEngine.get_suite(suite_id)` resolves a
  suite from `probe-suites/<suite_id>/manifest.json`, raising
  `FileNotFoundError` for unknown ids — the existing registry getter with
  exactly the 404 semantics M21 needs.
- **M10 suite-run engine**: `SuiteRunEngine.list_suite_runs(model_id)` is
  the authoritative listing — live scan of `suite-runs/*/manifest.json`,
  parse into `SuiteRunRecord`, filter by the **persisted** `model_id` field,
  sort `(created_at, suite_run_id)` ASCENDING, `[]` when empty, never
  writes. Every record carries the persisted `suite_id` it was executed
  under.
- **M11 workflow integration**: workflow `suite_run` stages persist ordinary
  M10 run manifests referenced from workflow records — the same store, so
  the by-suite view automatically covers workflow-created runs too.
- **M13 dashboard**: the dashboard's `suite_runs` section scans the same
  storage-root family — unaffected by a read-only filter method.
- **M20 API/engine pattern**: `list_sample_evaluations_for_checkpoint` =
  registry-validated identity + filter of the authoritative listing; facade
  method in `engine.py`; thin route registered **before** the
  `/{evaluation_id}` literal route. M21 mirrors this pattern exactly
  (registered before `/{suite_run_id}`).

Production facts: model `4a0a871886ef` owns **all 10** suite-run records
(`m9-live-suite`, status completed, probe_count 2, recorded under checkpoint
`0511de4c7372`); model `b5bc905326b6` exists with **no** suite-run records
(a natural "valid suite, no runs → []" case); `m9-live-suite` is the single
registered suite.

## §3 Implementation

Repository files changed only — **no production artifact was ever touched**:

1. **`app/suite_runs.py`** — new engine method
   `list_suite_runs_for_suite(model_id, suite_id)` placed after the M10
   getters, plus a module-docstring note. The method validates the model
   (`_require_model`), resolves the suite through the existing M9 registry
   (`self.policies.get_suite`), then returns
   `[r for r in self.list_suite_runs(model_id) if r.suite_id == suite_id]`.
   It reuses the existing listing/parse/ordering implementation; adds no
   database, index, cache, manifest format, storage or second engine.
2. **`app/engine.py`** — thin facade `list_suite_runs_for_suite(...)` with a
   docstring, delegating to the engine method (same shape as the M20
   facade).
3. **`app/api.py`** — thin route
   `GET /models/{model_id}/suite-runs/by-suite/{suite_id}`
   (`response_model=list[SuiteRunRecord]`, tag `suite-runs`, `FileNotFoundError →
   404`), registered **before** `/models/{model_id}/suite-runs/{suite_run_id}`
   so the literal `by-suite` segment can never be mistaken for a run id;
   landing-page milestone list + REST-API route list updated; section
   header comment updated.
4. **`tests/test_suite_runs.py`** — new "M21" section: 6 focused tests
   (15 → 21 functions) covering engine filtering/parity/ordering, valid
   suite without runs, 404s, read-only behavior, repeated-call determinism,
   API grouping/parity/raw-byte repeats, cross-model isolation, unchanged
   M10 APIs, M16–M20 sample-quality 404 semantics intact, and OpenAPI
   documentation of the new route.
5. **`tests/test_sample_quality.py`** + **`tests/test_dashboards.py`** — the
   five factual OpenAPI surface-count assertions updated 52 → 53 with
   comments (documented surface change; **no other assertion weakened or
   altered**).
6. **`README.md`** — "Milestone 21" section; test counts 408 → 414 (two
   sites); `suite_runs.py` layout line notes M21.
7. **`smoke_m21_live.py`** — new live production smoke (40 checks, phases
   A–D), modeled on `smoke_m20_live.py`.
8. **`m21_pre.sha256`** — pre-change SHA256 inventory of all 96 production
   files (audit artifact).

## §4 Tests

**Focused M21 tests** (all in `tests/test_suite_runs.py`, all green):

| Test | Covers |
|---|---|
| `test_m21_engine_filters_by_persisted_suite_and_model` | every record belongs to the requested model AND suite; exact M10 order; verbatim payload parity with the M10 listing **and** the M10 single-record getter; other models' runs excluded |
| `test_m21_engine_valid_suite_without_runs_and_404s` | valid registered suite with no runs → `[]`; other model's own run returned but never model A's; unknown model → `FileNotFoundError`; unknown suite → `FileNotFoundError`; read-only (run-manifest set unchanged) |
| `test_m21_engine_repeated_calls_identical` | 4 repeated engine calls produce identical serialized JSON |
| `test_m21_api_by_suite_grouping_and_parity` | HTTP lifecycle: 200, exactly the 2 suite-a runs of 3 total, M10 order, payloads identical to the POST responses and to the listing filtered by persisted `suite_id`, suite-b run excluded, repeated GET identical JSON **and raw bytes**, second model + valid suite → 200 + `[]` |
| `test_m21_api_404s_and_cross_model_isolation` | unknown model → 404 (real suite), unknown suite → 404 (real model), cross-model (real model + real suite) → 200 + `[]` with no leaked run ids, detail getter still 404 for another model's run, `by-suite` does not shadow the detail getter |
| `test_m21_api_existing_surfaces_and_openapi` | existing M10 listing/detail/404s unchanged; by-suite == listing for a single-run model; M16–M20 sample-quality endpoints (200/404 semantics) intact; OpenAPI documents the new route (GET only, tag `suite-runs`, `array` of `$ref SuiteRunRecord`) and surface count 53 |

**Full regression (serial, after implementation): 414 collected, 414
passed, 0 failed, 0 errors, exit 0** (408 pre-existing + 6 new; no existing
test weakened — the only edits to pre-existing tests are the five
documented OpenAPI count assertions 52 → 53).

**Static checks**: `compileall` clean (`app`, `tests`, all smokes);
`pyflakes` clean on every M21-touched file and on the whole `app/` +
`tests/` tree — the only two findings are pre-existing f-string notes in
untouched legacy `smoke_m9_live.py`/`smoke_m11_live.py`, byte-identical to
the pristine baseline (0 new findings).

**OpenAPI validation**: `/openapi.json` serves 200 with **53 paths**; the
new path `/api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}` is
documented with a single `get` operation, tag `suite-runs`, and response
schema `array` of `$ref: #/components/schemas/SuiteRunRecord`; all 85
`$ref`s in the document resolve.

## §5 Live Smoke

`smoke_m21_live.py` run against a real HTTP server
(`FORGE_ROOT=/home/user/ai-model-forge-data`, uvicorn on
`127.0.0.1:8742`); server stopped after testing. **40/40 checks PASS,
exit 0.** Highlights:

- **Phase A (baseline audit)**: exact 96 files / 4,002,745 B / 0 `.tmp`;
  per-file SHA256 inventory saved; both models + `m9-live-suite` resolve in
  the registries; M10 listing pre-state = the 10 known runs in ASCENDING
  order; M16/M18/M19/M20 pre-state intact (2 records, 200s); M17 dashboard
  hash `f48557fe8ab1...` unchanged.
- **Phase B (new endpoint)**: `4a0a871886ef / m9-live-suite` → **HTTP 200,
  exactly 10 records**; ids and `(created_at, suite_run_id)` order exactly
  `aae8e8a8ba9c, bb114d2ce42e, 8f8aee834c9f, d9742459017b, 5684649d0ced,
  2af508035191, 4cb388a21ce2, 2dc9f7dff400, 2d203398304b, e4b1c2a7fb2d`;
  every record carries the persisted `model_id` + `suite_id`; **parity**
  with the existing M10 listing filtered by persisted suite_id (= the full
  listing here) and with each record's M10 single getter (10/10); full
  verbatim `SuiteRunRecord` field sets, no derived statistics; all runs
  recorded under checkpoint `0511de4c7372`; repeated GETs **raw-byte
  identical** (×3); M10 listing unchanged by the reads.
- **Phase C (404 semantics + cross-model isolation)**: unknown model → 404;
  unknown well-formed suite id → 404; malformed suite id → 404 (no crash);
  **cross-model**: real model `b5bc905326b6` + the real suite → **200 +
  `[]`** (valid suite, no runs) and none of the 10 run ids leaks; unknown
  suite under the other model → 404; `by-suite` does not shadow the M10
  detail getter (real run id → 200, ghost run id → 404); M10 listing
  unknown model still 404.
- **Phase D (final audit)**: M18 records / M19 by-sample / M20
  by-checkpoint outputs unchanged; M17 dashboard hash preserved; M10
  listing byte-identical; by-suite still deterministic at the end; storage
  diff vs Phase A: **0 changed, 0 missing, 0 new files, 96 /
  4,002,745 / 0 `.tmp`**.

The smoke **fails loudly** (exit 1 + named failures) if any requirement is
wrong. One smoke-internal fix during development: the malformed-suite-id
probe originally contained a raw space, which `urllib` rejects client-side
before any request — replaced with a percent-encoded id so the server's 404
is actually exercised.

## §6 Storage & Determinism

| Audit | Pre-change | Post-change |
|---|---|---|
| Production files | 96 | **96** (0 new, 0 missing) |
| Production bytes | 4,002,745 | **4,002,745** (0 bytes growth) |
| `.tmp` files | 0 | **0** |
| Changed files (SHA256) | — | **0** — every one of the 96 pre-existing hashes identical (`m21_pre.sha256` vs post-change inventory, diff empty) |

The audit was run three times (after implementation, after the /tmp cleanup
+ serial rerun, and after the final live smoke) — identical every time.
Determinism: repeated live GETs return **raw-byte-identical** bodies; the
engine produces identical serialized JSON across repeated calls; ordering is
the exact M10 `(created_at, suite_run_id)` convention.

`/tmp` note: serial test runs create `forge-tests-*` session roots
(~145 MB each, from `tests/conftest.py`'s `tempfile.mkdtemp`). After
confirming no test/server process was running, **7 stale roots (~1 GB) were
removed**, the full suite was **rerun (414 passed)** and the production
inventory re-verified byte-identical.

## §7 Limitations

M21 deliberately does **not**:

- execute, schedule, retry or modify suite runs (strictly read-only; the
  only mutation endpoints remain the existing M10 `POST /suite-runs` and
  M11/M14 workflow/recipe runs);
- aggregate anything — no scores, averages, pass rates, rankings,
  leaderboards, quality judgments, or cross-suite/cross-model comparisons;
- select best runs/checkpoints, or feed any future automation;
- add a database, index, cache, background worker, scheduler, distributed
  execution, or a second suite-run system/manifest format;
- paginate, filter by state/status/date, sort differently, or accept query
  parameters (the deterministic M10 order is the only order);
- infer model/suite relationships from filenames, directory names or graph
  edges — identity comes from persisted manifest fields only;
- change any M1–M20 behavior, payload, route or storage format.

Known edges: a corrupt suite manifest in the registry path would surface as
a parse error (500) rather than a 404 — identical pre-existing semantics to
the M19/M20 registry getters, unchanged by M21. Suites are forge-wide ids:
any existing model may group its own runs under any registered suite; a
model with no runs of that suite simply gets `[]`.

## §8 Scope Compliance

- **M1–M20 intact**: full regression 408/408 pre-existing tests pass
  unmodified (only five documented OpenAPI count assertions updated 52 →
  53); M10 suite-run run/listing/detail APIs byte-identical live; M13
  dashboard hash `f48557fe...3838` unchanged; M16/M17/M18/M19/M20
  sample-quality surface unchanged live (2 records, same payload); M18
  records / M19 by-sample / M20 by-checkpoint all 200 with identical
  output before/after.
- **M21 scope**: exactly one read-only endpoint + one small reusing engine
  method + facade + tests + smoke + docs. No training, retraining,
  LoRA/QLoRA/PEFT, DPO/RL/RLHF/RLAIF, HPO, automatic improvement,
  checkpoint selection, quality scoring, rankings, leaderboards, external
  AI, frontend redesign, database, workers, scheduling or distributed
  execution was added. Nothing from future milestones was implemented
  early.
- **Audit verdict**: all tests, static checks, live smoke and both storage
  audits pass — **M21 complete**.

## §9 M22 Prompt

```
# MILESTONE 22 — SUITE-RUN SUMMARY STATISTICS (READ-ONLY, PER SUITE)

Continue the existing **AI Model Forge** project.

M21 is the current certified milestone.

The goal of M22 is to expose one small, read-only aggregate view over the
EXISTING suite-run history: per-suite execution bookkeeping totals for one
model. Do not redesign the architecture or start future
training/improvement features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and verify
the actual M21 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 414 tests passing
* OpenAPI: 53 paths
* M21 by-suite endpoint working
  (`GET /api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}`)
* M20 by-checkpoint endpoint working
* M19 by-sample endpoint working
* M18 sample-quality records endpoint working

Also inspect the existing implementations for:

* suite definitions from M9
* suite-run engine from M10 (including the M21
  `list_suite_runs_for_suite` filter)
* workflow suite-run integration from M11
* suite-run dashboard observability from M13
* M18–M21 read-only API/engine patterns

Create a SHA256 inventory of the production files before making changes.

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M22 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}/summary`

The endpoint should answer:

> How many suite-run records, probes, completed/reused/failed probe
> outcomes and completed/failed runs exist for this model and suite?

Reuse the existing M21 `list_suite_runs_for_suite` listing and the
existing `SuiteRunRecord` bookkeeping fields. The summary must be a pure
deterministic function of those immutable records.

Do not create a second suite-run system. Do not compute loss/perplexity
aggregates, scores, averages, rankings or quality judgments — ONLY the
execution bookkeeping counts that the records already persist.

---

## 3. REQUIRED BEHAVIOR

The endpoint must:

* verify the model exists
* verify the suite exists using the existing suite registry
* summarize only suite-run records belonging to the requested model and
  suite (reuse the M21 filter exactly)
* report at least: run_count, total probe_count, completed_count,
  reused_count, failed_count (probe-level, summed from the records), and
  runs_failed (number of records with status "failed")
* expose the suite_id, model_id and a deterministic result hash over the
  summarized record identities (ids + result_hashes), so identical
  histories yield identical summaries
* return zero counts (not 404) for a valid suite with no runs
* return 404 for an unknown model
* return 404 for an unknown suite
* prevent cross-model data leakage (another model's runs are never
  counted)
* return identical JSON for repeated requests
* be completely read-only

---

## 4. ENGINE IMPLEMENTATION

Inspect the current suite-run engine first.

If necessary, add one small method such as:

`summarize_suite_runs_for_suite(model_id, suite_id)`

The method should reuse `list_suite_runs_for_suite` (M21) and aggregate
only persisted bookkeeping fields. Do not introduce:

* a new database
* an index
* a cache
* another manifest format
* another suite-run engine
* new production storage
* any loss/perplexity/score math

Keep the implementation minimal. The API route should be a thin wrapper
around the engine.

---

## 5. TESTS

Add focused tests for:

1. Existing production suite (live smoke): model `4a0a871886ef`, suite
   `m9-live-suite`, exactly 10 runs, probe_count 20 (10 × 2),
   failed_count 0, runs_failed 0.
2. Counts equal the sums over the M21 by-suite listing for arbitrary
   fixtures (mixed completed/failed, reused>0).
3. A suite with runs of two different suites never mixes counts.
4. Valid suite with no runs returns all-zero counts.
5. Unknown model returns 404.
6. Unknown suite returns 404.
7. Cross-model access cannot count another model's runs.
8. Repeated requests return identical JSON.
9. The summary result hash is stable and changes when the underlying
   record set changes.
10. Existing suite-run APIs (M10 listing/detail, M21 by-suite) remain
    unchanged.
11. M16–M20 sample-quality APIs remain unchanged.
12. New OpenAPI route is documented correctly (54 paths).

Do not weaken existing tests.

---

## 6. LIVE TEST

Create:

`smoke_m22_live.py`

Run a real HTTP server using:

`FORGE_ROOT=/home/user/ai-model-forge-data`

Test `4a0a871886ef / m9-live-suite`.

Expected:

* HTTP 200
* run_count 10, probe_count 20, failed_count 0, runs_failed 0
* counts exactly equal manual sums over the M21 by-suite response
* deterministic repeated response (raw bytes)
* summary result hash stable across calls

Also test:

* unknown model → 404
* unknown suite → 404
* cross-model isolation (b5bc905326b6 + m9-live-suite → all-zero
  summary, no leaked counts)
* existing M21 by-suite endpoint still works
* existing M20 endpoint still works
* existing M19 endpoint still works
* existing M18 endpoint still works

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

M22 is read-only, so the expected production state must remain:

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

M22 must NOT add:

* model training
* retraining
* LoRA/QLoRA/PEFT
* DPO/RL/RLHF/RLAIF
* HPO
* automatic model improvement
* automatic checkpoint selection
* quality scoring (loss/perplexity aggregates, pass rates)
* rankings
* leaderboards
* Gemini/external AI
* frontend redesign
* database
* background workers
* scheduling
* distributed execution

M22 is only:

**one read-only per-suite bookkeeping summary endpoint.**

Do not implement future milestones early.

---

## 9. FINAL REPORT

At the end, produce exactly these nine sections:

### §1 Summary

What M22 implemented.

### §2 Baseline

Actual pre-change files, bytes, tests, OpenAPI, and relevant architecture
findings.

### §3 Implementation

Files changed and implementation details.

### §4 Tests

Focused and full regression results.

### §5 Live Smoke

Complete live HTTP test results.

### §6 Storage & Determinism

Before/after file counts, bytes, SHA256 comparison, `.tmp`, and
repeat-request determinism.

### §7 Limitations

What this milestone does not do.

### §8 Scope Compliance

Confirm M1–M21 remain intact.

### §9 M23 Prompt

Generate a complete copy-ready prompt for the next small milestone.

The M23 prompt must also require:

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

Do not declare M22 complete unless all tests, static checks, smoke tests,
and final storage audits pass.
```
