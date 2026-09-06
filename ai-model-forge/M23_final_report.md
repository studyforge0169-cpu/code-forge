# MILESTONE 23 — GATE-DECISION HISTORY BY POLICY — FINAL REPORT

## 1. Objective

Add exactly ONE deterministic, read-only endpoint to the AI Model Forge
REST API:

```
GET /api/v1/models/{model_id}/gates/decisions/by-policy/{policy_id}
```

It answers one question: *which existing gate-decision records of this
model were produced under this registered policy?* The implementation is
a pure reuse of the existing authoritative M6 gate-decision history: the
model is validated through the existing model registry, the policy is
resolved through the existing M9 policy registry
(`PolicyEngine.get_policy`), and the response is the model's
authoritative M6 listing (`GateEngine.list_decisions`) filtered by the
**persisted** `policy_id` recorded inside each `GateDecision` — never
inferred from gate-directory names. Records are returned verbatim in the
exact M6 deterministic order `(created_at, decision_id)` ASCENDING.
Inline-policy decisions keep `policy_id=null` and never appear. A valid
policy with no decisions for the model returns `[]`; unknown model or
unknown policy returns 404; another model's decisions are unreachable.
No training, no scoring, no rankings, no aggregation, no mutation, no
database/cache/index/worker, no second gate system — strictly an
observability/history read flow.

Verified live on production (`FORGE_ROOT=/home/user/ai-model-forge-data`):
`4a0a871886ef / m9-live-policy` → HTTP 200 with **exactly 1 record**
(decision `6921d3b29b9d`, passed), payload- and order-identical to the
existing M6 listing, byte-identical on repeated GETs, with zero
production storage changes.

## 2. Baseline

Actual pre-change state (all values verified **before** touching code;
the sandbox was NOT reset this time — git at the M22 commit `f30456f`,
clean tree, production root and venv intact):

| Check | Expected | Actual |
|---|---|---|
| Production root | `/home/user/ai-model-forge-data` | ✔ |
| Production files | 96 | **96** |
| Production bytes | 4,002,745 | **4,002,745** |
| `.tmp` files | 0 | **0** |
| Full test suite (serial) | 420 passing | **420 passed / 0 failed / 0 errors**, exit 0 |
| OpenAPI paths | 54 | **54** (by-policy absent pre-change, verified) |
| `compileall` | clean | **clean** |
| `pyflakes` | clean | **clean** (`app/*.py` + `tests/*.py`) |

Production SHA256 inventory of all 96 files created before any
modification (`m23_pre.sha256`; verified identical to the certified M22
inventory `m22_pre.sha256`).

Relevant gate-decision counts (recorded pre-change, read-only):

- Model `4a0a871886ef`: **11 total gate decisions** — exactly **1**
  carrying the persisted `policy_id='m9-live-policy'` (decision
  `6921d3b29b9d`, verdict `passed`, created
  `2026-09-04T09:04:08.780839Z`), the other **10 inline decisions**
  (`policy_id=null`); ASCENDING order `8931835d4af6, baf767bdcbe1,
  9d4facca5153, ef8ba75f9e43, 5b0493c0fbed, 8968151a08bd, a82c95374a5a,
  e33c99f2f8ca, 0dae7b2c456e, 5c86e4494d2f, 6921d3b29b9d`
- Policy `m9-live-policy`: the single registered M9 policy (registry
  manifest with deterministic `config_hash`)
- Second model `b5bc905326b6`: exists, owns **no** gate decisions → the
  natural valid-empty case (`200 + []`)

Architecture findings from inspecting the existing implementation:

- **M6 gate engine**: `GateEngine.list_decisions(model_id)` is the
  authoritative listing — model existence check, live scan of
  `models/<id>/gates/gate-*/manifest.json`, parse into `GateDecision`,
  corrupt manifests skipped with a warning, sort
  `(created_at, decision_id)` ASCENDING, `[]` when none, never writes.
  Every record persists `policy_id: Optional[str]` (`None` for inline
  policies, the registry id for M9-resolved gates) plus
  `policy_config_hash`.
- **M9 policy registry**: `PolicyEngine.get_policy(policy_id)` resolves
  from `policies/<policy_id>/manifest.json`, `FileNotFoundError` for
  unknown ids — exactly the authoritative ownership/validation lookup
  required (no filesystem existence shortcut).
- **Existing wiring**: `GateEngine.__init__` already holds
  `self.policies = PolicyEngine(storage)`; the facade
  (`engine.py`) delegates `run_gate` / `list_gate_decisions` /
  `get_gate_decision` thinly; the API exposes
  `GET /models/{id}/gates/decisions` (list) and
  `GET /models/{id}/gates/decisions/{decision_id}` (detail) — the new
  literal `by-policy` segment must be registered **before** the detail
  route.
- **M18–M22 patterns**: registry-validated identity → authoritative
  listing filter → thin facade → thin route; M23 mirrors this shape
  exactly.

## 3. Implementation

Repository files changed only — **no production artifact was ever
touched**, no new schema was needed (`GateDecision` is returned
directly), and no new persistence structure was created:

1. **`app/gates.py`** — new engine method
   `list_decisions_for_policy(model_id, policy_id)` placed after the M6
   getters:
   `decisions = self.list_decisions(model_id)` (authoritative listing;
   validates the model → `FileNotFoundError`), then
   `self.policies.get_policy(policy_id)` (M9 registry resolution →
   `FileNotFoundError`), then
   `[d for d in decisions if d.policy_id == policy_id]`. Reuses the
   existing parsing, model validation, policy validation, deterministic
   sorting and corruption handling; adds no database, cache, index,
   manifest format or second gate engine; never writes.
2. **`app/engine.py`** — thin facade `list_gate_decisions_for_policy`
   with a docstring, delegating 1:1 to the engine method (same shape as
   the M21/M22 facades).
3. **`app/api.py`** — thin route
   `GET /models/{model_id}/gates/decisions/by-policy/{policy_id}`
   (`response_model=list[GateDecision]`, tag `gates`,
   `FileNotFoundError → 404`), registered **before**
   `/models/{model_id}/gates/decisions/{decision_id}` so the literal
   `by-policy` segment can never be mistaken for a decision id;
   landing-page milestone list + REST-API route list updated; gate-route
   section header comment updated.
4. **`tests/test_gates.py`** — new "M23" section: 3 focused engine tests
   (27 → 30 functions) with a cached `_m23_env` fixture builder (two
   models, three registered policies, decisions spread across policies
   plus one inline decision) and a read-only file-snapshot helper;
   `PolicyCreateRequest` import added.
5. **`tests/test_gates_api.py`** — new "M23" section: 2 focused API
   tests (7 → 9 functions): grouping/parity/determinism and
   404s/isolation/prior-surfaces/OpenAPI.
6. **`tests/test_sample_quality.py`** + **`tests/test_dashboards.py`** +
   **`tests/test_suite_runs.py`** — the seven factual OpenAPI
   surface-count assertions updated 54 → 55 with comments (documented
   surface change; **no existing assertion weakened or otherwise
   altered**).
7. **`README.md`** — "Milestone 23" section; test counts 420 → 425 (two
   sites); `gates.py` layout line notes M23.
8. **`smoke_m23_live.py`** — new live production smoke (34 checks,
   phases A–D), modeled on `smoke_m20_live.py`–`smoke_m22_live.py`.
9. **`m23_pre.sha256`** — pre-change SHA256 inventory of all 96
   production files (audit artifact; identical to `m22_pre.sha256`).

## 4. Tests

**Focused M23 tests** (3 engine in `tests/test_gates.py` + 2 API in
`tests/test_gates_api.py`, all green):

| Test | Covers |
|---|---|
| `test_m23_engine_filters_by_persisted_policy_and_model` | every record belongs to the requested model AND policy; exact M6 order; verbatim payload parity with the M6 listing **and** single getter; a second policy returns only its own decision; inline decision (`policy_id=None`) never appears |
| `test_m23_engine_valid_policy_without_runs_404s_and_read_only` | valid registered policy + model with no gates under it → `[]`; the other model's own policy returns exactly its decision; cross-model disjointness; unknown model / unknown policy → `FileNotFoundError`; decision-manifest file set unchanged (read-only) |
| `test_m23_engine_repeated_calls_identical` | 4 repeated engine calls produce identical serialized JSON |
| `test_m23_api_by_policy_grouping_parity_and_determinism` | HTTP 200; exactly the 2 pol-a decisions of 4 total; M6 order; payloads identical to the evaluate responses and to the listing filtered by persisted `policy_id`; pol-b and inline decisions excluded; repeated GET identical JSON **and raw bytes** |
| `test_m23_api_404s_isolation_and_prior_surfaces` | unknown model → 404 (real policy); unknown policy → 404 (real model); cross-model (real model + real policy) → 200 + `[]` with no leaked ids; M6 listing/detail unchanged (incl. detail 404 for a ghost id); M9 policy registry unchanged; M18–M20 sample-quality endpoints intact; M21 by-suite + M22 summary intact (registered suite, zero summary); OpenAPI documents the new route (GET only, tag `gates`, `array` of `$ref GateDecision`) and surface count 55 |

The production-facts test of the prompt's §11 Test 1
(`4a0a871886ef / m9-live-policy` → exactly 1 decision) is fulfilled by
the **live smoke** (unit tests run on throwaway temp roots and cannot
see production): smoke checks B1–B7 assert HTTP 200, exactly 1 record,
decision `6921d3b29b9d` with `model_id == 4a0a871886ef` and
`policy_id == m9-live-policy`, verified against the authoritative M6
listing.

**Full regression (serial, after implementation): 425 collected, 425
passed, 0 failed, 0 errors, exit 0** (420 pre-existing + 5 new). One
transparent hiccup during development: the first post-change full run
had **1 failure — `test_m22_api_openapi_documented`** — because my
update script missed that test's own `== 54` path-count assertion (a
second count assertion in `tests/test_suite_runs.py` beyond the M21
one). Fixed by updating that documented factual assertion 54 → 55 (the
exact same class of edit as the other six); rerun: **425 passed**. No
existing test was weakened — the only edits to pre-existing tests are
the seven documented OpenAPI count assertions 54 → 55.

**Static checks**: `compileall` clean (`app`, `tests`, the M23 smoke);
`pyflakes` clean across `app/*.py`, `tests/*.py`, `smoke_m23_live.py` —
**0 new findings** (the two pre-existing f-string notes in untouched
legacy `smoke_m9_live.py`/`smoke_m11_live.py` remain, byte-identical to
the original upload; nothing fixed, nothing added, nothing hidden).

**OpenAPI validation**: `/openapi.json` serves 200 with **55 paths**
(before: 54); the new path appears **exactly once**, with a single `get`
operation, tag `gates`, and response schema `array` of
`$ref: #/components/schemas/GateDecision`; all 86 `$ref`s resolve.

## 5. Live Smoke

`smoke_m23_live.py` run against a real HTTP server
(`FORGE_ROOT=/home/user/ai-model-forge-data`, uvicorn on
`127.0.0.1:8744`); server stopped after testing. **34/34 checks PASS,
exit 0.** Highlights:

- **Phase A (baseline audit)**: exact 96 files / 4,002,745 B / 0 `.tmp`;
  per-file SHA256 inventory saved; both models + `m9-live-policy`
  resolve in the registries; M6 listing pre-state = exactly the 11 known
  decisions in ASCENDING order; exactly one carries the persisted
  `policy_id='m9-live-policy'` and 10 are inline; M16/M18/M19/M20/M21/
  M22 pre-state intact (2 sample records, 10 suite runs, summary count
  10); M17 dashboard hash `f48557fe8ab1...` unchanged.
- **Phase B (new endpoint)**:
  `4a0a871886ef / m9-live-policy / by-policy` → **HTTP 200 with exactly
  1 record**; the decision is `6921d3b29b9d` (passed, created
  `2026-09-04T09:04:08.780839Z`); persisted `model_id` + `policy_id` as
  requested; **parity** with the M6 listing filtered by the persisted
  `policy_id` and with the M6 single-record getter; verbatim
  `GateDecision` payload (same field set as the listing; recorded
  `policy_config_hash` equals the registry definition's `config_hash`);
  inline-policy decisions never appear; repeated GETs **raw-byte
  identical** (×3); M6 listing unchanged by the reads.
- **Phase C (404 semantics + cross-model isolation)**: unknown model →
  404; unknown well-formed policy id → 404; malformed (percent-encoded)
  policy id → 404 without crashing; **cross-model**: real model
  `b5bc905326b6` + the real policy → **200 + `[]`** (valid policy, no
  decisions) and the foreign decision id never leaks; unknown policy
  under the other model → 404; by-policy does not shadow the M6 detail
  getter (real decision id → 200, ghost id → 404); M6 listing unknown
  model still 404; the M9 policy registry still resolves the policy.
- **Phase D (final audit)**: M18 records / M19 by-sample / M20
  by-checkpoint / M21 by-suite / M22 summary outputs unchanged; M17
  dashboard hash preserved; M6 listing + M9 policy definition
  byte-identical to pre-state; by-policy still deterministic at the end;
  storage diff vs Phase A: **0 changed, 0 missing, 0 new files,
  96 / 4,002,745 / 0 `.tmp`**.

The smoke **fails loudly** (exit 1 + named failures) if any requirement
is wrong.

## 6. Storage Integrity

| Audit | Before | After |
|---|---|---|
| Production files | 96 | **96** (0 new, 0 missing) |
| Production bytes | 4,002,745 | **4,002,745** (0 bytes growth) |
| `.tmp` files | 0 | **0** |
| Changed files (SHA256) | — | **0** — every one of the 96 pre-existing hashes identical (`m23_pre.sha256` diff empty; re-verified after the /tmp-cleanup rerun) |

No generated caches, temporary manifests, indexes, databases or logs
were written into production storage; the smoke itself performs the
before/after per-file SHA256 comparison (Phase A inventory vs Phase D
audit) in addition to the shell-level audit. Determinism: repeated live
GETs return **raw-byte-identical** bodies; the engine produces identical
serialized listings across repeated calls; ordering is the exact M6
`(created_at, decision_id)` convention; identity comes only from
persisted manifest fields.

`/tmp` note: serial test runs create `forge-tests-*` session roots
(~145 MB each, from `tests/conftest.py`'s `tempfile.mkdtemp`). After
confirming no test/server process was running, **6 stale roots were
removed**, the full suite was **rerun (425 passed)** and the production
inventory re-verified byte-identical.

## 7. Regression / Compatibility

- **M1–M22 intact**: full regression 420/420 pre-existing tests pass
  (only the seven documented OpenAPI count assertions 54 → 55 changed);
  live-verified unchanged on production: M6 gate listing (11 decisions,
  byte-identical) and detail getter, M9 policy registry
  (`m9-live-policy` definition byte-identical), M13 dashboard hash
  `f48557fe...3838`, M16/M17 sample-quality surface, M18 records /
  M19 by-sample / M20 by-checkpoint (2 records, identical payloads),
  M21 by-suite (10 records) and M22 summary (count 10) — all
  byte-identical before/after (smoke checks D1–D3).
- **M23 is additive only**: one engine method + one facade method + one
  route + tests + smoke + docs. No existing route, payload, manifest,
  ordering, storage format or behavior changed. The literal `by-policy`
  segment is registered before `/{decision_id}` so no route shadowing is
  possible (verified live: detail getter still resolves real ids and
  still 404s ghost ids).
- **Known edges**: decisions made with inline policies intentionally
  never appear under any policy id (their persisted `policy_id` is
  `null`); a corrupt gate manifest is skipped by the existing M6 listing
  semantics (unchanged); policies are forge-wide definitions, so any
  existing model may query any registered policy — a model with no
  decisions under it simply gets `[]`, which is the specified
  isolation behavior (its own decisions are unreachable from other
  models, verified live).

## 8. Commit / Final Status

- Commit: **`M23: add gate decision history by policy`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge` (commit hash in the git log; no
  unrelated commits amended).
- Files changed: `app/gates.py`, `app/engine.py`, `app/api.py`,
  `tests/test_gates.py`, `tests/test_gates_api.py`,
  `tests/test_sample_quality.py`, `tests/test_dashboards.py`,
  `tests/test_suite_runs.py`, `README.md`, `smoke_m23_live.py` (new),
  `m23_pre.sha256` (new, audit artifact), `M23_final_report.md` (new).
- **Final status: M23 COMPLETE AND CERTIFIED.** All gates pass:
  425/425 tests serially (420 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 54 → **55 paths** with the
  endpoint documented exactly once; live smoke **34/34 PASS** on
  production (exactly 1 decision for `4a0a871886ef`/`m9-live-policy`,
  parity with the M6 listing, `200 + []` for `b5bc905326b6`, clean 404s,
  cross-model isolation, inline exclusion, byte-identical repeats);
  storage audit **96 files / 4,002,745 bytes / 0 `.tmp` / 0 changed /
  0 new / 0 missing / 0 bytes growth**; dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 24 — EVALUATION HISTORY BY CHECKPOINT

Continue the existing **AI Model Forge** project.

M23 is the current certified milestone.

The goal of M24 is to add one small, read-only API capability for
inspecting existing M4 evaluation history grouped by the checkpoint it
measured.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and verify
the actual M23 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 425 tests passing
* OpenAPI: 55 paths
* M23 by-policy endpoint working
* M22 by-suite summary endpoint working
* M21 by-suite endpoint working
* M20 by-checkpoint endpoint working
* M19 by-sample endpoint working
* M18 sample-quality records endpoint working

Also inspect the existing implementations for:

* the M4 evaluation engine (`EvaluationEngine.list_evaluations` /
  `get_evaluation` — the authoritative listing and its
  `(created_at, eval_id)` ordering)
* the persisted `EvaluationRecord` fields `state_kind` and
  `checkpoint_id` (checkpoint evals carry the id; current-state evals
  carry `checkpoint_id=null`)
* the M3 checkpoint registry (`TrainingEngine.get_checkpoint`) —
  checkpoint ids are model-scoped; this is the same ownership validation
  M20 uses
* M18–M23 read-only API/engine patterns (registry validation +
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

## 2. M24 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/evaluations/by-checkpoint/{checkpoint_id}`

The endpoint should answer:

> Which existing immutable M4 evaluation records measured this
> checkpoint of this model?

Reuse the existing M4 evaluation records, the existing M3 checkpoint
registry and the existing listing logic.

Do not create a second evaluation system.

---

## 3. REQUIRED BEHAVIOR

The endpoint must:

* verify the model exists
* verify the checkpoint exists using the model's M3 checkpoint registry
  (an unknown checkpoint, or a checkpoint id belonging to another model,
  is 404 — nothing is inferred from filenames)
* return only evaluation records of the requested model whose persisted
  `state_kind == "checkpoint"` and `checkpoint_id` equals the requested
  id
* preserve the existing `EvaluationRecord` representation (verbatim
  payloads, loss_nats/perplexity included)
* use deterministic ordering consistent with the existing evaluation
  listing behavior (`(created_at, eval_id)` ASCENDING)
* never return current-state evaluations (their persisted
  `checkpoint_id` is null)
* return `[]` when a valid checkpoint has no evaluations
* return 404 for an unknown model
* return 404 for an unknown checkpoint
* prevent cross-model data leakage
* be completely read-only; never infer identity from filenames — use the
  persisted `checkpoint_id` recorded in each evaluation

---

## 4. ENGINE IMPLEMENTATION

Inspect the current evaluation engine first.

If necessary, add one small method such as:

`list_evaluations_for_checkpoint(model_id, checkpoint_id)`

The method should reuse the existing `list_evaluations` implementation
and the M3 `get_checkpoint` registry getter (exactly the ownership
validation pattern M20 established for sample-quality).

Do not introduce:

* a new database
* an index
* a cache
* another manifest format
* another evaluation engine
* new production storage

Keep the implementation minimal. The API route should be a thin wrapper
around the engine.

---

## 5. TESTS

Add focused tests for:

1. Production facts (live smoke): model `4a0a871886ef` owns 16 M4
   evaluations; exactly 9 are checkpoint evaluations — 3 under each of
   `025e6d8d8f15` (`7a16eaa12120, 90aa392b9a2b, ebb9b7eccbe3`),
   `0511de4c7372` (`c739c66638e9, 0704399fea7b, 75a23351e91c`) and
   `30a8bc5b82ab` (`b0502d871114, 75835b64d6af, fe7b42cdb411`); the
   other 7 are current-state evaluations (checkpoint_id null) and must
   never appear.
2. Every returned record belongs to the requested model and checkpoint.
3. Returned payloads match the existing evaluation records verbatim.
4. Ordering is deterministic (`(created_at, eval_id)` ASCENDING).
5. Valid checkpoint with no evaluations returns `[]` (unit fixtures).
6. Unknown model returns 404.
7. Unknown checkpoint returns 404.
8. Cross-model access cannot expose another model's evaluations
   (`b5bc905326b6` + a real checkpoint id of `4a0a871886ef` -> 404).
9. Repeated requests return identical JSON.
10. Existing evaluation APIs (listing/detail/run) remain unchanged.
11. M16–M23 sample-quality, suite-run and gate by-policy APIs remain
    unchanged.
12. New OpenAPI route is documented correctly (56 paths).

Do not weaken existing tests.

---

## 6. LIVE TEST

Create:

`smoke_m24_live.py`

Run a real HTTP server using:

`FORGE_ROOT=/home/user/ai-model-forge-data`

Test `4a0a871886ef / 0511de4c7372`.

Expected:

* HTTP 200
* exactly 3 records (`c739c66638e9, 0704399fea7b, 75a23351e91c`)
* parity with the existing M4 listing filtered by the persisted
  checkpoint identity
* deterministic repeated response (raw bytes)
* the same 3-record parity for the other two checkpoints
  (`025e6d8d8f15`, `30a8bc5b82ab`)

Also test:

* unknown model -> 404
* unknown checkpoint -> 404
* cross-model isolation (`b5bc905326b6` + `0511de4c7372` -> 404)
* existing M23 by-policy endpoint still works (1 decision)
* existing M22 summary endpoint still works (count 10)
* existing M21/M20/M19/M18 endpoints still work
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

M24 is read-only, so the expected production state must remain:

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

M24 must NOT add:

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
* new evaluation execution, checkpoint mutation or comparison of any kind

M24 is only:

**one read-only checkpoint-grouping endpoint over existing evaluation
records.**

Do not implement future milestones early.

---

## 9. FINAL REPORT

At the end, write `M24_final_report.md` with exactly these nine
sections:

1. Objective
2. Baseline
3. Implementation
4. Tests
5. Live Smoke
6. Storage & Determinism
7. Audit / Limitations
8. Final Result
9. Next Milestone (complete copy-ready prompt for M25)

The M25 prompt must also require:

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

Do not declare M24 complete unless all tests, static checks, smoke
tests, and final storage audits pass.
```
