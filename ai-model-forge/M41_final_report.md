# M41 Final Report — Gate-Decision History by Decision

## Recovery & Baseline

No reset this session (reset runbook standby not needed): HEAD ==
FETCH_HEAD == `b3a4a12` (M40 implementation `2c56607` + inventory
`b3a4a12`), tree clean except the known egg-info. Pre-implementation
state verified and recorded BEFORE any change:

- full suite: **510 passed @ 143.35 s** (exact M40 ladder);
- OpenAPI: **72 paths**;
- production: **96 files / 4,002,745 bytes / 0 .tmp**;
- `m40_pre.sha256` check -> **0 non-OK** (zero drift);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes, venv
  intact;
- `m41_pre.sha256` captured (96 entries) as the pre-milestone
  inventory;
- production grounding re-verified live later in the smoke (11 gate
  decisions, passed -> 7 / failed -> 4; `b5bc905326b6` owns none).

One routing note resolved by inspection during Step 0/5: the milestone
description wrote the endpoint with a hyphenated
`/gate-decisions/...` prefix, but the ACTUAL existing family in
`app/api.py` is `/models/{id}/gates/decisions/...` (listing, M23
by-policy, M34 by-comparison, and the generic
`/gates/decisions/{decision_id}` detail getter). The milestone's own
route-ordering requirement — "after the existing more-specific
gate-decision history routes, before any generic
`/gates/decisions/{decision_id}` route, so `by-decision` cannot be
interpreted as a decision ID" — is only meaningful inside that real
prefix, so the implemented path is the family-coherent:

```
GET /api/v1/models/{model_id}/gates/decisions/by-decision/{decision}
```

## Implementation

Three production files + docs, mirroring the M23/M34/M40 pattern:

- `app/gates.py` — new engine method `list_decisions_for_decision(
  model_id, decision: GateDecisionResult)` placed directly after
  `list_decisions_for_comparison`. It reuses the model's authoritative
  M6 listing (exact engine parse + deterministic `(created_at,
  decision_id)` ASC order) and filters by the persisted `decision`
  VERBATIM. Docstring records the contract: the decision is the
  immutable policy verdict persisted by the M6 run — it may
  legitimately DIFFER from the loss-only comparison verdict (e.g. an
  improved candidate still fails a `minimum_loss` ceiling) — and is
  NEVER recalculated from loss deltas, tolerances, policy thresholds,
  gate configuration or comparison results; decision results have NO
  registry (unlike the M23 policy / M34 comparison axes) — the enum IS
  the contract, so unsupported values are rejected with 422 at the API
  boundary and never reach the engine, while an unknown model raises
  FileNotFoundError exactly like the siblings; a valid decision with
  zero decisions returns `[]`; read-only, never writes.
  `GateDecisionResult` was already imported here (used by the run
  path).
- `app/engine.py` — facade `list_gate_decisions_for_decision`
  delegating to the engine, placed after the by-comparison delegation
  (+`GateDecisionResult` import). The facade signature matches its
  siblings exactly (no return annotation — `GateDecision` is not
  imported in the facade and no new import chain was introduced).
- `app/api.py` — route `@api.get("/models/{model_id}/gates/decisions/
  by-decision/{decision}", response_model=list[GateDecision],
  tags=["gates"])` with `decision: GateDecisionResult` as the path
  parameter (native pre-handler 422 for unsupported values), placed
  after the M34 by-comparison route and before the generic
  `/gates/decisions/{decision_id}` getter; landing-page bullet;
  endpoint `<li>` in the documented layout; gate-routes section
  comment extended with the M41 line (+`GateDecisionResult` import;
  all-three-files import rule honored).
- `README.md` — M41 section, counts 510 -> 515 (two places), layout
  line `M23/M34/M41`.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere.

## Tests

Five new tests (510 -> 515, exactly as targeted):

- `tests/test_gates.py` — 3 engine tests via the cached `_m41_state(
  env)` fixture: two PASSED gates (an improvement candidate and an
  unchanged same-state candidate) and one FAILED gate built from the
  SAME improved candidate under an impossible `minimum_loss` ceiling —
  the policy verdict legitimately differs from the loss-only
  comparison verdict, which is exactly why the persisted field (never
  a recalculation) must be the membership authority. Assertions:
  explicit both-enum coverage guard; filtered-listing parity for BOTH
  values + verbatim detail-getter payloads + `(created_at,
  decision_id)` ordering; explicit disjoint partition whose union is
  the full listing; a fresh model with NO gate decisions -> `[]` for
  both values; unknown model -> `FileNotFoundError`; cross-model
  isolation against the M23 second model (both models hold decisions,
  listings disjoint, each group a subset of its own listing);
  read-only (gate manifest file-set snapshots for all three models
  unchanged); repeated-call determinism.
- `tests/test_gates_api.py` — 2 API tests via the existing
  `_make_model`/`_gate_body`/`_policy` helpers: 200 + exact records +
  listing parity + detail-getter parity + raw-byte-identical x3 for
  both values + partition (passed 2 / failed 1) + no-side-effects;
  404 unknown model + valid decision; 422 unsupported values (case
  variant, interior-space, numeric, `skipped`) and unknown-model +
  invalid-decision -> 422 (pre-handler precedence); cross-model
  isolation (second model with no gates -> both groups `[]`); M6
  listing/getter + ghost-id 404 (no route capture); M23 by-policy
  regression (registered policy; inline decisions never appear) + M34
  by-comparison regression (listing parity); M40 samples-by-strategy
  regression (this harness has no samples — the natural valid-empty
  form with listing parity, per the milestone instruction); OpenAPI 73
  assertions (route order by-comparison < by-decision < generic,
  GET-only, tag `gates`, items `$ref GateDecision`, `decision` param
  `$ref GateDecisionResult`).

One bring-up iteration (fixed before any full run): a variable typo
(`d` vs `r`) in the engine repeats test raised NameError; corrected,
after which engine 3/3 + full module 36/36 and API 2/2 + full module
13/13 passed.

Results (nothing certified from partial runs):

- full suite, twice consecutively: **515 passed @ 149.63 s**, then
  (after cleaning accumulated sandboxes, no live process)
  **515 passed @ 151.25 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**.

## OpenAPI

- OpenAPI 72 -> **73 paths** (exactly one new path).
- Standalone verification (no server): the new path registered exactly
  once, GET-only, tag `gates`; route order by-comparison < by-decision
  < generic `{decision_id}`; `decision` parameter `$ref`
  `GateDecisionResult`; response items `$ref` `GateDecision`; no
  duplicate routes, no unintended methods.
- Repository sweep: every hard-coded count assertion searched and
  updated — **25 assertions `== 72` -> `== 73` across 8 test modules
  (one more than M40's 24: M40's own new API test added an
  assertion), 0 `== 72` remaining**, plus 17 ladder lines extended
  with `+ 1 (M41 gate decisions by-decision)`; no unrelated numeric
  literals touched; no existing OpenAPI assertion weakened.

## Live Smoke

`smoke_m41_live.py` (committed) — 33 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
33/33 PASS each, exit 0, first attempt**, on port **8762** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m41-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + full M4/M5/M6/
  M11/M15/M16/M18–M40 pre-state (including the M36–M40 groupings) +
  dashboard hash + OpenAPI 73 pre-state + known pair 200.
- B (DISCOVERY, never assumed): from the live M6 listing + the OpenAPI
  schema — **11 gate decisions, decision distribution
  `passed -> 7, failed -> 4` (both groups non-empty)**, ASC
  `(created_at, decision_id)`; enum values `['passed', 'failed']` —
  matching the M40-end grounding exactly.
- C: BOTH decisions -> 200 + EXACT records == listing filtered locally
  (7/4); verbatim detail-getter parity for all 11 records
  (verdict/evidence chain/rollback suggestion included).
- D: three GETs raw-byte-identical.
- E: ALL enum decisions -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid decision -> 404; unsupported values
  (interior-space, `PASSED`, `1`) -> 422 with the enum detail;
  unsupported decision + UNKNOWN model -> 422 (pre-handler
  precedence).
- G: the two groups partition the full listing exactly (7+4 == 11,
  every record exactly once, no record in two groups, persisted
  decision verbatim).
- H: every group preserves the authoritative M6 order exactly.
- I: M23 by-policy (1) + M34 by-comparison (4) + the M6 listing
  unchanged, all with listing parity.
- J: M22 suite runs (10/10/summary), M24/M28/M30/M36/M38 evaluation
  histories (3/3/3, 16, 16, by-split 14/2/0, by-state-kind 9/7),
  M26/M29/M31/M37/M39 comparison histories (6/5/1, 8, 8, by-split
  8/0/0, by-verdict 3/3/2), M27/M32/M40 sample histories (4/0/0, 4,
  by-strategy 2/2), M33 sample-quality (2).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe
  registries unchanged.
- M: OpenAPI exactly 73 paths, new path once, order by-comparison <
  by-decision < generic, M40 samples-by-strategy path still present
  once; storage zero drift (below).

## Storage Integrity

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: **0 changed / 0 missing / 0 new** files across all
  three runs;
- final audit after the server stopped: `sha256sum -c m41_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- stale test sandboxes (`/tmp/forge-tests-*`) were cleaned (after
  verifying no live pytest/uvicorn process) before the certifying
  second full-suite run and before the smoke; they are test
  artifacts, never production writes;
- M41 wrote NOTHING to production: the endpoint is pure data access
  over existing M6 gate manifests. Production remains byte-identical
  through M41.

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5/M6/M11/M15 listings + detail getters, M22
  suite-run histories, M23 gate decisions by policy, M24/M28/M30/M36/
  M38 evaluation groupings, M26/M29/M31/M37/M39 comparison groupings,
  M27/M32/M40 sample groupings, M33 sample-quality, M34 gate decisions
  by comparison, M35 workflow history by recipe (model-scoped + global
  M12 lineage), M16 pairings, M17 dashboard (full hash), M2/M9/M12/
  M14 registries.
- Full suite green at 515 with no weakened tests; the M40-style
  natural-valid-empty discipline was applied where this harness's
  premises legitimately produce empty results (samples under the gate
  harness), never by manufacturing artifacts.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged.
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit State

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M41: add gate-decision history by decision` — `app/gates.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 11 touched test
   modules (2 with new tests, 9 with the OpenAPI 73 sweep + ladders),
   `smoke_m41_live.py`, `M41_final_report.md`.
2. follow-up inventory commit adding `m41_pre.sha256` at the repo root
   (M21–M41 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed; treated
consistently with every prior milestone).

## Next Milestone

End-of-M41 live inspection (production server, real payloads + schema
inspection): every `WorkflowRecord` persists a top-level `status:
WorkflowStatus` — a THREE-value schema enum (completed = plan executed
through its last stage; failed = a stage raised; stopped = a gate
decision failed and no on_fail branch; a mid-flight `running` state is
deliberately never observable) — with production distribution
**completed -> 9, failed -> 3, stopped -> 1 (ALL THREE groups
non-empty)** across the 13 workflows of `4a0a871886ef`; the M11
listing is deterministic in `(created_at, workflow_id)` ASC order
(verified live), and the family already groups by-recipe (M35) — the
status axis is the missing enum-contract twin. `b5bc905326b6` owns NO
workflows (natural valid-empty for all three values). Runner-up
inspected and set aside: `SuiteRunStatus` (completed/failed —
production holds only `completed`, a single-valued, weakly grounded
partition); `SampleEvaluationRecord.window_rule` (single-valued
`single_window`).

Copy-ready M42 prompt:

> M42 — WORKFLOW HISTORY BY STATUS
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped workflow records by the persisted `status` enum field.
> Do NOT jump ahead into aggregation, rankings, retries, re-execution,
> rollback, retraining, optimization, or autonomous improvement.
> Preserve the minimum-files/minimum-storage architecture.
>
> STEP 0 — VERIFY STATE AND BASELINE. Inspect the repository and git
> state. If the familiar reset signature appears (HEAD `f86b670` /
> "Add files via upload", files untracked, production data missing,
> venv missing), execute the established recovery runbook: `git
> fetch` + `git reset --hard FETCH_HEAD` (never force-push, never
> discard authoritative commits), restore `ai-model-forge-data/` from
> the workspace zip, verify against `m41_pre.sha256`, rebuild the venv
> (`python3 -m venv ~/.venv` + `pip install -e ai-model-forge[dev]
> pyflakes`), then re-prove the baseline. Expected baseline: M41
> certified; HEAD/FETCH_HEAD inventory commit = the M41 inventory
> hash; 515 tests passing; OpenAPI 73; production 96 files /
> 4,002,745 bytes / 0 tmp; production workflows of `4a0a871886ef`:
> completed -> 9 / failed -> 3 / stopped -> 1 (13 total);
> `b5bc905326b6` has zero workflows. Capture `m42_pre.sha256` and
> record the exact baseline before changing anything. Inspect
> `app/workflows.py`, M11 listing behavior, the M35
> `list_workflows_for_recipe()` implementation, the workflow route
> family in `app/api.py`, and `tests/test_workflows.py` /
> `tests/test_workflow_recipes.py` conventions; run the complete
> suite before editing.
>
> GROUNDED SELECTION (live-verified at M41 end): `WorkflowRecord.
> status: WorkflowStatus` — the persisted top-level schema enum
> completed/failed/stopped; all three groups non-empty in production;
> listing order `(created_at, workflow_id)` ASC; no by-status surface
> exists in M1–M41.
>
> IMPLEMENTATION: In `app/workflows.py` add
> `list_workflows_for_status(model_id, status)` following the
> established M35/M41 thin pattern: validate the model via the
> authoritative M11 listing mechanism; obtain records from the
> existing `list_workflows(model_id)`; filter ONLY the persisted
> `record.status == status` (never inferred from stages, transitions,
> failed_stage_id, terminal_reason or result hashes; never resolved or
> rewritten; never re-executed); preserve the M11 `(created_at,
> workflow_id)` ASC ordering; `[]` for a valid model with no matching
> records; unknown model -> existing FileNotFoundError/404; invalid
> enum value -> native FastAPI/Pydantic 422 via a `WorkflowStatus`
> path parameter; no cross-model leakage; no writes; no cache/index/
> DB/secondary registry. In `app/engine.py` expose it through the
> facade (no duplicated filtering logic). In `app/api.py` add exactly:
> `GET /api/v1/models/{model_id}/workflows/by-status/{status}` with
> `response_model=list[WorkflowRecord]`, tags matching the workflow
> family, registered after the M35 by-recipe route and BEFORE the
> generic `/workflows/{workflow_id}` detail route (so "by-status" can
> never be interpreted as a workflow id). Documentation: landing
> bullet + endpoint `<li>` + section comment (verify M40/M41 anchors
> before editing); README M42 section + endpoint count 515 -> ~520 +
> layout line update.
>
> TESTS: ~3 engine tests + 2 API tests following the M41 conventions:
> both/all-three enum groups with listing parity (derive membership
> from the authoritative M11 listing, never hard-coded ids);
> deterministic ordering; persisted status verbatim; partition
> (completed + failed + stopped == full listing, disjoint);
> valid-empty (a model with no workflows -> 200 []); unknown model ->
> 404; invalid status -> 422 (case variant, interior-space, numeric)
> incl. unknown-model + invalid -> 422 pre-handler; cross-model
> isolation; byte-identical repeats; no writes; M35 by-recipe
> regression + M41 by-decision regression (natural-valid-empty or
> parity per the harness). IMPORTANT: do not manufacture workflow runs
> or other artifacts merely to make a regression assertion non-empty —
> use parity-derived/natural-valid-empty behavior exactly as M40/M41
> did. Target 515 -> ~520 (report the actual).
>
> OPENAPI: 73 -> 74 paths. Update every genuine path-count assertion
> (search the whole repo; do not blindly replace unrelated numbers).
> Verify: exactly one new path, GET only, workflow tag, items $ref
> WorkflowRecord, status parameter $ref WorkflowStatus, family
> ordering by-recipe < by-status < generic detail, no duplicate
> routes, no unintended methods. Do not weaken existing assertions.
>
> FULL VERIFICATION: focused workflow tests; API tests; complete
> suite; complete suite a second time; compileall; pyflakes; OpenAPI
> verification; production storage audit; SHA verification against
> `m42_pre.sha256`. Clean stale `/tmp/forge-tests-*` after verifying
> no live process. Expected: tests 515 -> ~520, OpenAPI 74, storage
> unchanged, zero drift.
>
> LIVE SMOKE: create `smoke_m42_live.py`, port **8763**, following the
> established A–M structure: baseline inventory capture; live
> DISCOVERY of the workflow status distribution (expected grounding:
> completed -> 9 / failed -> 3 / stopped -> 1; report the actual if
> legitimately different — never invent); all-three-group parity;
> partition 9+3+1 == 13; `(created_at, workflow_id)` ordering;
> byte-identical repeats; unknown model -> 404; invalid status -> 422;
> second model -> 200 []; cross-model isolation; M35 by-recipe
> unchanged; M41 gate by-decision unchanged; M18–M40 regressions
> unchanged; dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
> unchanged; policy/probe/recipe registries unchanged; storage
> remains 96 files / 4,002,745 bytes / 0 tmp; SHA256 0 changed / 0
> new / 0 missing; no new production artifacts.
>
> FINAL AUDIT + REPORT: inspect git diff/status; verify only intended
> files changed; verify production unchanged; no force-push; local
> HEAD == authoritative remote HEAD; only egg-info untracked. Commit
> the implementation (`M42: add workflow history by status`) and the
> inventory (`m42_pre.sha256`) separately per the M21–M41 convention;
> push normally. Final report with EXACTLY these 9 sections: 1.
> Recovery & Baseline, 2. Implementation, 3. Tests, 4. OpenAPI, 5.
> Live Smoke, 6. Storage Integrity, 7. Regression / Compatibility,
> 8. Git / Commit State, 9. Next Milestone. Section 9 must
> automatically select M43 from LIVE persisted production
> relationships discovered at the end of M42 (inspect every remaining
> persisted enum/identity axis across evaluations, comparisons,
> samples, sample-quality, suite-runs, gates, workflows, recipes;
> prefer an existing enum/registry with >= 2 non-empty live groups,
> thin filter, no schema redesign, no new storage; mention the
> runner-up) and include a COMPLETE copy-ready M43 prompt with the
> same structure as this one (Step 0 baseline/recovery, grounded
> selection, exact endpoint/engine method/field/enum, route ordering,
> tests, OpenAPI count, smoke port 8764, storage invariants, the
> exact 9-section report requirement, automatic M44 selection). Do
> not jump to advanced training, autonomous improvement, HPO, RL,
> Gemini integration, inference, deployment, or other future
> architecture unless the live evidence specifically makes that the
> next justified milestone.
