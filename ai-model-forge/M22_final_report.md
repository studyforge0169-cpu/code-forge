# MILESTONE 22 — PER-SUITE READ-ONLY BOOKKEEPING SUMMARY — FINAL REPORT

## 1. Objective

Add exactly ONE deterministic, read-only endpoint to the AI Model Forge
REST API:

```
GET /api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}/summary
```

It answers one question: *how many suite runs exist for this model and
this suite, which ones, and when did they run?* The summary is a pure
derived view over the existing immutable `SuiteRunRecord` manifests via
the existing M21 filtering logic — model and suite are validated with the
existing registries, and the response contains only simple bookkeeping
already supported by `SuiteRunRecord`:

- `model_id`, `suite_id` — identity of the summarized group
- `total_count` — number of suite-run records
- `run_ids` — the ordered run ids (the exact deterministic M21 order)
- `earliest_created_at` / `latest_created_at` — first/last recorded run
  timestamps (`null` exactly when there are no runs)

A valid suite with no runs returns a **zero summary** (count 0, empty
ids, null timestamps), never a 404. Unknown model or suite returns 404.
No quality scores, rankings, averages, trends, regression judgments, new
metrics or automatic decisions. No second suite-run system, no database,
cache, index, background worker or new storage format; nothing is ever
written. The existing M21 endpoint remains unchanged.

## 2. Baseline

**Environment note (full disclosure):** the sandbox had been rebuilt
since the certified M21 state — the working tree still held the M21
files, but the local git history and the production root were missing.
Before any M22 work: the production root `/home/user/ai-model-forge-data`
was restored from the archived project zip and **verified byte-identical
to the certified M21 inventory** (`m21_pre.sha256`: all 96 SHA256 hashes
match); the local branch was re-synced to the remote M21 commits
(`7ed90ef` / `a1e03e5`) only after `git diff` confirmed an empty tree
difference; the Python venv was rebuilt. No production byte was ever
modified by the restore (hash-verified), and M22's own audits below were
run against the re-verified state.

Actual pre-change state (all values verified **before** touching code):

| Check | Expected | Actual |
|---|---|---|
| Production root | `/home/user/ai-model-forge-data` | ✔ |
| Production files | 96 | **96** |
| Production bytes | 4,002,745 | **4,002,745** |
| `.tmp` files | 0 | **0** |
| Full test suite (serial) | 414 passing | **414 passed / 0 failed / 0 errors**, exit 0 |
| OpenAPI paths | 53 | **53** (M21 by-suite present; `/summary` absent pre-change) |
| `compileall` | clean | **clean** |
| `pyflakes` | clean | **clean** (`app/`, `tests/`, M21 smoke; the two pre-existing f-string notes in untouched legacy `smoke_m9_live.py`/`smoke_m11_live.py` remain, byte-identical to the original upload) |
| M21 by-suite endpoint | working | **live 200** (10 records for `4a0a871886ef`/`m9-live-suite`) |
| M20 by-checkpoint | working | **live 200** |
| M19 by-sample | working | **live 200** |
| M18 records | working | **live 200** |

A **pre-change SHA256 inventory of all 96 production files** was created
(`m22_pre.sha256`, identical to the certified M21 inventory) before any
modification.

Architecture findings from inspecting the existing implementation:

- **M9 suite registry**: `PolicyEngine.get_suite(suite_id)` resolves a
  suite from its persisted manifest, `FileNotFoundError` for unknown ids.
- **M10 suite-run engine**: `SuiteRunEngine.list_suite_runs(model_id)` is
  the authoritative listing (live scan, persisted-`model_id` filter,
  `(created_at, suite_run_id)` ASCENDING, never writes); every
  `SuiteRunRecord` carries the persisted `suite_id` and `created_at`.
- **M21 by-suite history**: `list_suite_runs_for_suite(model_id,
  suite_id)` = model check + M9 suite resolution + persisted-`suite_id`
  filter of that listing — the exact reuse point required by M22.
- **M13 dashboard**: scans the same `suite-runs/` family for its
  `suite_runs` section — a derived summary method changes nothing there.
- **M18–M21 API patterns**: registry-validated identity → authoritative
  listing filter/derivation → thin facade → thin route before any
  literal-shadowing path; M22 follows the same shape (the `/summary`
  sub-path cannot shadow the M21 records route or the M10 detail getter).

Production facts: model `4a0a871886ef` owns all 10 suite-run records
(`m9-live-suite`, completed, 2 probes, checkpoint `0511de4c7372`;
ASCENDING order `aae8e8a8ba9c, bb114d2ce42e, 8f8aee834c9f, d9742459017b,
5684649d0ced, 2af508035191, 4cb388a21ce2, 2dc9f7dff400, 2d203398304b,
e4b1c2a7fb2d`; earliest `2026-09-04T09:21:11.493507Z`, latest
`2026-09-04T10:46:30.123758Z`); model `b5bc905326b6` exists with **no**
suite-run records (natural zero-summary case); `m9-live-suite` is the
single registered suite.

## 3. Implementation

Repository files changed only — **no production artifact was ever
touched**:

1. **`app/schemas.py`** — new `SuiteRunSummary` model (after
   `SuiteRunRecord`): `model_id`, `suite_id`, `total_count`,
   `run_ids` (default `[]`), `earliest_created_at` /
   `latest_created_at` (`Optional[datetime]`, `None` exactly when no
   runs). Never persisted; recomputed deterministically per request.
2. **`app/suite_runs.py`** — new engine method
   `list_suite_run_summary_for_suite(model_id, suite_id)` placed after
   the M21 method, plus a module-docstring note. The method calls the
   M21 `list_suite_runs_for_suite` exactly (model + M9 suite-registry
   validation, persisted-`suite_id` filter, deterministic order) and
   derives only counting/identity bookkeeping:
   `total_count=len(records)`, `run_ids=[r.suite_run_id …]` (already
   ASCENDING), `earliest=records[0].created_at`,
   `latest=records[-1].created_at` (`None`/`None` when empty). No new
   database, cache, index, manifest format, storage or engine; never
   writes.
3. **`app/engine.py`** — thin facade `list_suite_run_summary_for_suite`
   (delegates 1:1 to the engine; `SuiteRunSummary` import added).
4. **`app/api.py`** — thin route
   `GET /models/{model_id}/suite-runs/by-suite/{suite_id}/summary`
   (`response_model=SuiteRunSummary`, tag `suite-runs`,
   `FileNotFoundError → 404`), registered directly after the unchanged
   M21 by-suite route (`/summary` is a longer literal path — it cannot
   shadow `/by-suite/{suite_id}` or `/{suite_run_id}`); landing-page
   milestone list + REST-API route list updated; `SuiteRunSummary`
   import added.
5. **`tests/test_suite_runs.py`** — new "M22" section: 6 focused tests
   (21 → 27 functions; details in §4).
6. **`tests/test_sample_quality.py`** + **`tests/test_dashboards.py`** +
   the M21 OpenAPI test in **`tests/test_suite_runs.py`** — the six
   factual OpenAPI surface-count assertions updated 53 → 54 with
   comments (documented surface change; **no existing assertion weakened
   or otherwise altered**).
7. **`README.md`** — "Milestone 22" section; test counts 414 → 420 (two
   sites); `suite_runs.py` layout line now reads "(M21/M22)".
8. **`smoke_m22_live.py`** — new live production smoke (31 checks,
   phases A–D).
9. **`m22_pre.sha256`** — pre-change SHA256 inventory of all 96
   production files (audit artifact).

## 4. Tests

**Focused M22 tests** (all in `tests/test_suite_runs.py`, all green):

| Test | Covers |
|---|---|
| `test_m22_engine_summary_fields_order_and_parity` | exact bookkeeping-only field set; `total_count`/`run_ids` equal the M21 listing (same deterministic order); `earliest`/`latest` = the listing's first/last `created_at` (= min/max); a second suite summarizes only its own run |
| `test_m22_engine_zero_summary_404s_and_read_only` | valid suite + model with no runs → **zero summary** (0 / `[]` / `None`/`None`); the other model's own history stays separate; unknown model → `FileNotFoundError`; unknown suite → `FileNotFoundError`; summaries never write run manifests |
| `test_m22_engine_repeated_calls_identical` | 4 repeated engine calls produce identical serialized JSON |
| `test_m22_api_summary_parity_and_determinism` | HTTP 200; parity with the live M21 by-suite response (count, ordered ids, min/max timestamps as serialized); the suite-b run stays out; repeated GET identical JSON **and raw bytes**; second model + valid suite → the exact zero-summary JSON object |
| `test_m22_api_404s_isolation_and_prior_surfaces` | unknown model → 404 (real suite); unknown suite → 404 (real model); cross-model → 200 **zero summary** with no leaked run ids; **M21 by-suite unchanged** (full record list, `[]` for the other model); M10 listing/detail unchanged; M16–M20 sample-quality endpoints intact (200/404 semantics) |
| `test_m22_api_openapi_documented` | the new path is documented (GET only, tag `suite-runs`, response `$ref SuiteRunSummary`, schema in components) and the surface count is 54 |

**Full regression (serial, after implementation): 420 collected, 420
passed, 0 failed, 0 errors, exit 0** (414 pre-existing + 6 new). No
existing test was weakened — the only edits to pre-existing tests are the
six documented OpenAPI count assertions 53 → 54.

**Static checks**: `compileall` clean (`app`, `tests`, the M22 smoke);
`pyflakes` clean across `app/*.py`, `tests/*.py` and `smoke_m22_live.py`
(0 new findings; the two legacy notes in untouched M9/M11 smokes predate
M21 and are part of the original upload).

**OpenAPI validation**: `/openapi.json` serves 200 with **54 paths**;
the new path `/api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}/summary`
is documented with a single `get` operation, tag `suite-runs`, and
response schema `$ref: #/components/schemas/SuiteRunSummary`; all 86
`$ref`s in the document resolve.

## 5. Live Smoke

`smoke_m22_live.py` run against a real HTTP server
(`FORGE_ROOT=/home/user/ai-model-forge-data`, uvicorn on
`127.0.0.1:8743`); server stopped after testing. **31/31 checks PASS,
exit 0.** Highlights:

- **Phase A (baseline audit)**: exact 96 files / 4,002,745 B / 0 `.tmp`;
  per-file SHA256 inventory saved; both models + `m9-live-suite` resolve;
  M10 listing and M21 by-suite pre-state = exactly the 10 known runs in
  ASCENDING order (identical payloads); M16/M18/M19/M20 pre-state intact
  (2 records); M17 dashboard hash `f48557fe8ab1...` unchanged.
- **Phase B (new endpoint)**: `4a0a871886ef / m9-live-suite / summary` →
  **HTTP 200** with exactly the bookkeeping field set; **`total_count`
  is exactly 10**; `run_ids` equal the M21 by-suite listing in the
  ASCENDING order (the 10 known ids); `earliest_created_at` /
  `latest_created_at` equal the listing's min/max `created_at` **and**
  the known production timestamps (`2026-09-04T09:21:11.493507Z` /
  `2026-09-04T10:46:30.123758Z`); identity fields correct; repeated GETs
  **raw-byte identical** (×3); M21 by-suite unchanged by the reads.
- **Phase C (404s + cross-model isolation)**: unknown model → 404;
  unknown well-formed suite id → 404; malformed (percent-encoded) suite
  id → 404 without crashing; **cross-model**: real model `b5bc905326b6` +
  the real suite → **200 + ZERO summary**
  (`{"total_count": 0, "run_ids": [], "earliest_created_at": null,
  "latest_created_at": null, …}`) and none of the 10 run ids is counted
  or leaked; unknown suite under the other model → 404; the summary
  route does not shadow the M21 records route (still 200 with the 10
  full records) or the M10 detail getter (real run id → 200, ghost run
  id → 404).
- **Phase D (final audit)**: M18 records / M19 by-sample / M20
  by-checkpoint outputs unchanged; M17 dashboard hash preserved; M10
  listing + M21 by-suite byte-identical to pre-state; summary still
  deterministic at the end; storage diff vs Phase A: **0 changed,
  0 missing, 0 new files, 96 / 4,002,745 / 0 `.tmp`**.

The smoke **fails loudly** (exit 1 + named failures) if any requirement
is wrong.

## 6. Storage & Determinism

| Audit | Pre-change | Post-change |
|---|---|---|
| Production files | 96 | **96** (0 new, 0 missing) |
| Production bytes | 4,002,745 | **4,002,745** (0 bytes growth) |
| `.tmp` files | 0 | **0** |
| Changed files (SHA256) | — | **0** — every one of the 96 pre-existing hashes identical (`m22_pre.sha256` vs post-change inventory, diff empty; re-verified after the /tmp cleanup rerun) |

Determinism: repeated live GETs return **raw-byte-identical** bodies
(×3); the engine produces identical serialized summaries across repeated
calls; `run_ids` order is the exact M21/M10 `(created_at, suite_run_id)`
convention; the summary is recomputed from immutable manifests only, so
identical histories yield identical summaries.

`/tmp` note: serial test runs create `forge-tests-*` session roots
(~145 MB each, from `tests/conftest.py`'s `tempfile.mkdtemp`). After
confirming no test/server process was running, **3 stale roots were
removed**, the full suite was **rerun (420 passed)** and the production
inventory re-verified byte-identical.

## 7. Audit / Limitations

**Audit verdict — all gates pass:**

- full test suite serially: **420/420**, exit 0 (414 pre-existing intact)
- `compileall`: clean; `pyflakes`: 0 new findings
- OpenAPI validation: **54 paths**, summary route documented, all refs
  resolve
- live smoke on production: **31/31 PASS**
- storage audit: **96 files / 4,002,745 bytes / 0 `.tmp`; 0 changed,
  0 new, 0 missing, 0 bytes growth** — every pre-existing production
  hash identical

M22 deliberately does **not**:

- create, execute, schedule or modify suite runs (strictly read-only;
  the only mutating endpoints remain the existing M10 `POST /suite-runs`
  and M11/M14 workflow/recipe runs);
- compute scores, rankings, averages, trends, pass rates, regression
  judgments, verdicts or any new evaluation metric — the summary counts
  records and reports ids/timestamps only;
- aggregate probe-level or status-level counters in the response (the
  full per-run `probe_count`/`status` bookkeeping stays available
  verbatim on each `SuiteRunRecord` via the unchanged M10/M21 endpoints);
- add a database, cache, index, background worker, scheduler, new
  storage format or second suite-run system; persist anything; or infer
  identity from filenames (model/suite identity is always persisted);
- paginate, accept query parameters, or change any M1–M21 behavior,
  payload, route or storage format.

Known edges: the summary reflects only records that parse as valid
`SuiteRunRecord` manifests (corrupt manifests are skipped by the
existing M10 listing semantics — unchanged); a suite definition
registered but never run by the model yields a zero summary, which is
the specified behavior, not an error.

## 8. Final Result

**M22 is complete and certified.** One narrow read-only bookkeeping
summary endpoint added on top of the unchanged M21 grouping; 420 tests
green serially (414 + 6 new, none weakened); compileall + pyflakes +
OpenAPI (54 paths) clean; live production smoke 31/31 with exact
M21-listing parity (10 runs, ordered ids, earliest/latest timestamps),
zero summary for a valid suite without runs, clean 404s, cross-model
isolation and byte-identical repeats; production storage byte-identical
before and after (96 files / 4,002,745 bytes / 0 `.tmp` / 0 growth).
M1–M21 remain intact (verified live: M10 listing/detail, M13 dashboard
hash `f48557fe...3838`, M16/M17/M18/M19/M20 sample-quality surface, M21
by-suite). All work is committed and pushed to
`arena/01a071e9-code-forge`.

## 9. Next Milestone

```
# MILESTONE 23 — GATE-DECISION HISTORY BY POLICY

Continue the existing **AI Model Forge** project.

M22 is the current certified milestone.

The goal of M23 is to add one small, read-only API capability for
inspecting existing gate-decision history grouped by the M9 policy that
produced it.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and verify
the actual M22 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 420 tests passing
* OpenAPI: 54 paths
* M22 by-suite summary endpoint working
* M21 by-suite endpoint working
* M20 by-checkpoint endpoint working
* M19 by-sample endpoint working
* M18 sample-quality records endpoint working

Also inspect the existing implementations for:

* policy registry from M9 (`PolicyEngine.get_policy`)
* gate engine from M6 (`GateEngine.list_decisions` /
  `get_decision`, the `(created_at, decision_id)` ordering)
* `GateDecision.policy_id` / `policy_config_hash` fields (persisted
  policy identity; `None` for inline policies)
* workflow gate integration from M7 (decisions recorded by workflow
  gate stages use the same store)
* dashboard observability from M8/M13
* M18–M22 read-only API/engine patterns (registry validation +
  authoritative-listing filter + thin facade + thin route)

Create a SHA256 inventory of the production files before making changes.

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M23 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/gates/decisions/by-policy/{policy_id}`

The endpoint should answer:

> Which existing gate-decision records of this model were produced under
> this registered policy?

Reuse the existing M6 gate-decision records, the existing M9 policy
registry and the existing listing logic.

Do not create a second gate system.

---

## 3. REQUIRED BEHAVIOR

The endpoint must:

* verify the model exists
* verify the policy exists using the existing M9 policy registry
* return only gate-decision records of the requested model whose
  persisted `policy_id` equals the requested policy id
* preserve the existing `GateDecision` representation (verbatim
  payloads)
* use deterministic ordering consistent with the existing gate-decision
  listing behavior (`(created_at, decision_id)` ASCENDING)
* return `[]` when a valid policy has no decisions for that model
* return 404 for an unknown model
* return 404 for an unknown policy
* never return decisions made with inline policies (their persisted
  `policy_id` is `None` — they belong to no policy id)
* prevent cross-model data leakage
* be completely read-only; never infer relationships from filenames —
  use the persisted `policy_id` recorded in each decision

---

## 4. ENGINE IMPLEMENTATION

Inspect the current gate engine first.

If necessary, add one small method such as:

`list_decisions_for_policy(model_id, policy_id)`

The method should reuse the existing `list_decisions` implementation and
`PolicyEngine.get_policy`.

Do not introduce:

* a new database
* an index
* a cache
* another manifest format
* another gate engine
* new production storage

Keep the implementation minimal. The API route should be a thin wrapper
around the engine.

---

## 5. TESTS

Add focused tests for:

1. Production facts (live smoke): model `4a0a871886ef` owns exactly 11
   gate decisions; exactly ONE carries the persisted
   `policy_id='m9-live-policy'` (decision `6921d3b29b9d`, verdict
   `passed`); the other 10 have `policy_id=None`.
2. Every returned record belongs to the requested model and policy.
3. Returned payloads match the existing gate-decision records verbatim.
4. Ordering is deterministic (`(created_at, decision_id)` ASCENDING).
5. Valid policy with no decisions for the model returns `[]`
   (e.g. the other model `b5bc905326b6` + `m9-live-policy`).
6. Unknown model returns 404.
7. Unknown policy returns 404.
8. Cross-model access cannot expose another model's decisions.
9. Repeated requests return identical JSON.
10. Existing gate APIs (listing/detail) remain unchanged.
11. M16–M22 sample-quality and suite-run APIs remain unchanged.
12. New OpenAPI route is documented correctly (55 paths).

Do not weaken existing tests.

---

## 6. LIVE TEST

Create:

`smoke_m23_live.py`

Run a real HTTP server using:

`FORGE_ROOT=/home/user/ai-model-forge-data`

Test:

`4a0a871886ef / m9-live-policy`

Expected:

* HTTP 200
* exactly 1 decision (`6921d3b29b9d`, passed)
* parity with the existing M6 listing filtered by persisted policy_id
* deterministic repeated response (raw bytes)
* `b5bc905326b6 / m9-live-policy` → 200 + `[]`

Also test:

* unknown model → 404
* unknown policy → 404
* cross-model isolation
* existing M22 summary endpoint still works
* existing M21 by-suite endpoint still works
* existing M20/M19/M18 endpoints still work
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

M23 is read-only, so the expected production state must remain:

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

M23 must NOT add:

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
* new gate evaluation, policy mutation or decision creation of any kind

M23 is only:

**one read-only policy-grouping endpoint over existing gate decisions.**

Do not implement future milestones early.

---

## 9. FINAL REPORT

At the end, write `M23_final_report.md` with exactly these nine
sections:

1. Objective
2. Baseline
3. Implementation
4. Tests
5. Live Smoke
6. Storage & Determinism
7. Audit / Limitations
8. Final Result
9. Next Milestone (complete copy-ready prompt for M24)

The M24 prompt must also require:

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

Do not declare M23 complete unless all tests, static checks, smoke
tests, and final storage audits pass.
```
