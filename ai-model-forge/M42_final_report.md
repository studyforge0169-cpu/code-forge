# M42 Final Report — Workflow History by Status

## Recovery & Baseline

No reset this session (recovery runbook standby not needed): HEAD ==
FETCH_HEAD == `e971c31` (M41 implementation `666c97a` + inventory
`e971c31`), tree clean except the known egg-info. Pre-implementation
state verified and recorded BEFORE any change:

- full suite: **515 passed @ 147.46 s** (exact M41 ladder);
- OpenAPI: **73 paths**;
- production: **96 files / 4,002,745 bytes / 0 .tmp**;
- `m41_pre.sha256` check -> **0 non-OK** (zero drift);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes, venv
  intact;
- `m42_pre.sha256` captured (96 entries) as the pre-milestone
  inventory;
- expected live grounding (workflows completed -> 9 / failed -> 3 /
  stopped -> 1) re-verified by discovery in the smoke.

## Implementation

Three production files + docs, mirroring the M35/M40/M41 pattern:

- `app/workflows.py` — new engine method `list_workflows_for_status(
  model_id, status: WorkflowStatus)` placed directly after
  `list_workflows_for_recipe`. It reuses the model's authoritative M11
  listing (exact engine parse + deterministic `(created_at,
  workflow_id)` ASC order) and filters by the persisted `status`
  VERBATIM. Docstring records the contract: the status is the
  THREE-value schema enum persisted at run end by the M7 orchestration
  (completed = plan executed through its last stage; failed = a stage
  raised; stopped = a gate decision failed with no on_fail branch; a
  mid-flight 'running' state is deliberately never modelled), NEVER
  inferred from stage results, failed stage ids, timestamps, artifact
  existence or recipe information, never recalculated, nothing
  re-executed; statuses have NO registry (unlike the M35 recipe axis)
  — the enum IS the contract, so unsupported values are rejected with
  422 at the API boundary and never reach the engine, while an
  unknown model raises FileNotFoundError exactly like the sibling; a
  valid status with zero matching runs returns `[]`; read-only, never
  writes. `WorkflowStatus` was already imported here.
- `app/engine.py` — facade `list_workflows_for_status` delegating to
  the engine, placed after the by-recipe delegation (+`WorkflowStatus`
  import; facade signature matches the unannotated sibling style).
- `app/api.py` — route `@api.get("/models/{model_id}/workflows/
  by-status/{status}", response_model=list[WorkflowRecord],
  tags=["workflows"])` with `status: WorkflowStatus` as the path
  parameter (native pre-handler 422 for unsupported values), placed
  after the M35 by-recipe route and before the generic
  `/workflows/{workflow_id}` getter; landing-page bullet; endpoint
  `<li>`; workflow-routes section comment extended with the M42 line
  (+`WorkflowStatus` import; all-three-files import rule honored).
- `README.md` — M42 section, counts 515 -> 520 (two places),
  workflows.py layout line extended with `M35/M42`.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere. (The milestone's endpoint path matched the
actual codebase family exactly — no hyphenation discrepancy this
time.)

## Tests

Five new tests (515 -> 520, exactly as targeted), all in
`tests/test_workflow_recipes.py` (the module holding the M35 sibling
engine+API block and its shared env/api harness):

- 3 engine tests via the cached `_m42_state(env)` fixture, which
  deterministically creates ALL THREE statuses: one COMPLETED
  checkpoint suite run, one FAILED run (a stage referencing an
  unknown suite — the failure IS persisted; the run call raises
  FileNotFoundError and the last listing record is the FAILED one),
  one STOPPED run (a checkpoint-baseline gate whose regressed
  candidate fails with no on_fail branch), plus a fresh model with
  ONE completed run (cross-model isolation partner) and a fresh model
  with NO workflows (natural valid-empty). Assertions: three-group
  filtered-listing parity + verbatim detail-getter payloads +
  `(created_at, workflow_id)` ordering; explicit pairwise-disjoint
  partition over ALL THREE enum values whose union is the full
  listing; empty-listing model -> `[]` for all three; unknown model
  -> `FileNotFoundError`; cross-model isolation (both models hold
  runs, listings disjoint, each group a subset of its own listing,
  fresh model's failed/stopped groups naturally empty); read-only
  (workflow manifest file-set snapshots for all three models
  unchanged); repeated-call determinism.
- 2 API tests: 200 + exact records for ALL THREE statuses + listing
  parity + detail-getter parity + raw-byte-identical x3 per status +
  three-way partition + no-side-effects; 404 unknown model + valid
  status; 422 unsupported values (case variant, interior-space,
  numeric, and the deliberately-unmodelled `running`) and
  unknown-model + invalid-status -> 422 (pre-handler precedence);
  cross-model isolation (second model with no runs -> all three
  groups `[]`); M11 listing/getter + ghost workflow id 404 (no route
  capture); M35 by-recipe regression (registered recipe run with
  listing parity; ad-hoc runs never appear) + by-status/by-recipe
  coherence on the same listing; M41 gate-decision by-decision
  regression + M40 samples-by-strategy regression (both
  natural-valid-empty with listing parity in this harness); OpenAPI
  74 assertions (route order by-recipe < by-status < generic,
  GET-only, tag `workflows`, items `$ref WorkflowRecord`, `status`
  param `$ref WorkflowStatus`).

Three bring-up iterations (all fixed before any full run): (1) a
literal `\\` line-continuation landed in a `cat`-appended assert
(SyntaxError) — repaired to a single continuation; (2) the gate
fixture helper read `h["ds"]/h["tok"]` which `_http_env` did not
return — the shared helper was extended ADDITIVELY to also return
`ds`/`tok` (no existing caller affected); (3) my minimum-loss gate
ceiling was backwards (`1e9` is a GENEROUS ceiling that passes; the
impossible one is `1e-9`) — the stopped run completed instead of
stopping until the ceiling was corrected.

Results (nothing certified from partial runs):

- full suite, twice consecutively: **520 passed @ 180.69 s**, then
  (after cleaning accumulated sandboxes, no live process)
  **520 passed @ 166.32 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**.

## OpenAPI

- OpenAPI 73 -> **74 paths** (exactly one new path).
- Standalone verification (no server): the new path registered exactly
  once, GET-only, tag `workflows`; route order by-recipe < by-status
  < generic `{workflow_id}`; `status` parameter `$ref
  WorkflowStatus`; response items `$ref WorkflowRecord`; no duplicate
  or conflicting workflow routes, no unintended methods.
- Repository sweep: every hard-coded count assertion searched and
  updated — **26 assertions `== 73` -> `== 74` across 8 test modules
  (one more than M41's 25: M41's own new API test added an
  assertion), 0 `== 73` remaining**, plus 18 ladder lines extended
  with `+ 1 (M42 workflows by-status)`; no existing OpenAPI assertion
  weakened.

## Live Smoke

`smoke_m42_live.py` (committed) — 34 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
34/34 PASS each, exit 0, first attempt**, on port **8763** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m42-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + full M4/M5/M6/
  M11/M15/M16/M18–M41 pre-state (including the M36–M41 groupings and
  the M41 by-decision counts) + dashboard hash + OpenAPI 74 pre-state
  + known pair 200.
- B (DISCOVERY, never assumed): from the live M11 listing + the OpenAPI
  schema — **13 workflow runs, status distribution
  `completed -> 9, failed -> 3, stopped -> 1` (all three groups
  non-empty)**, ASC `(created_at, workflow_id)`; enum values
  `['completed', 'failed', 'stopped']` — matching the M41-end
  grounding exactly.
- C: ALL THREE statuses -> 200 + EXACT records == listing filtered
  locally (9/3/1); verbatim detail-getter parity for all 13 records
  (stages, transitions, terminal_reason, result_hash and recipe
  provenance included).
- D: three GETs raw-byte-identical.
- E: ALL enum statuses -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid status -> 404; unsupported values (case
  variant, interior-space, numeric) -> 422 with the enum detail; the
  deliberately-unmodelled `running` + UNKNOWN model -> 422
  (pre-handler precedence).
- G: the three groups partition the full listing exactly (9+3+1 == 13,
  every record exactly once, no record in two groups, persisted status
  verbatim).
- H: every group preserves the authoritative M11 order exactly.
- I: M35 by-recipe (2, listing parity) + the M11 listing unchanged.
- J: M22 suite runs (10/10/summary), M24/M28/M30/M36/M38 evaluation
  histories (3/3/3, 16, 16, by-split 14/2/0, by-state-kind 9/7),
  M26/M29/M31/M37/M39 comparison histories (6/5/1, 8, 8, by-split
  8/0/0, by-verdict 3/3/2), M27/M32/M40 sample histories (4/0/0, 4,
  by-strategy 2/2), M33 sample-quality (2), M41/M34/M23
  gate-decision histories (by-decision passed 7 / failed 4 with
  listing parity, 4 by-comparison, 1 by-policy).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe
  registries unchanged.
- M: OpenAPI exactly 74 paths, new path once, order by-recipe <
  by-status < generic, M41 gate-decisions-by-decision path still
  present once; storage zero drift (below).

## Storage Integrity

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: **0 changed / 0 missing / 0 new** files across all
  three runs;
- final audit after the server stopped: `sha256sum -c m42_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- stale test sandboxes (`/tmp/forge-tests-*`) cleaned after verifying
  no live pytest/uvicorn process, before the certifying second
  full-suite run and the smoke;
- M42 wrote NOTHING to production: the endpoint is pure data access
  over existing M11 workflow manifests. Production remains
  byte-identical through M42.

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5/M6/M11/M15 listings + detail getters, M22
  suite-run histories, M23/M34/M41 gate-decision groupings, M24/M28/
  M30/M36/M38 evaluation groupings, M26/M29/M31/M37/M39 comparison
  groupings, M27/M32/M40 sample groupings, M33 sample-quality, M35
  workflow history by recipe (model-scoped + global M12 lineage),
  M16 pairings, M17 dashboard (full hash), M2/M9/M12/M14 registries.
- Full suite green at 520 with no weakened tests; natural-valid-empty
  discipline applied where this harness legitimately produces empty
  results (gate decisions and samples under the recipe harness), never
  by fabricating artifacts.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged.
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit State

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M42: add workflow history by status` — `app/workflows.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 9 touched test
   modules (1 with new tests, 8 with the OpenAPI 74 sweep + ladders),
   `smoke_m42_live.py`, `M42_final_report.md`.
2. follow-up inventory commit adding `m42_pre.sha256` at the repo root
   (M21–M42 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed; treated
consistently with every prior milestone).

## Next Milestone

End-of-M42 live inspection (production server, real payloads + schema
inspection): every `GateDecision` persists a top-level `verdict:
Optional[ComparisonVerdict]` — the loss-only comparison verdict
recorded by the M6 run (improved / regressed / unchanged), **None for
threshold-only gates that judged no comparison** (schema comment
verbatim) — with production distribution **improved -> 4, regressed ->
3, unchanged -> 2 (all three groups non-empty)** plus 2 null-verdict
threshold-only decisions across the 11 gate decisions of
`4a0a871886ef`; the M6 listing is deterministic in `(created_at,
decision_id)` ASC order (verified live through M41), and the
gate-decision family already groups by-policy (M23), by-comparison
(M34) and by-decision (M41) — by-verdict completes the family. The
null group mirrors M35's ad-hoc pattern: threshold-only decisions
(verdict None) belong to NO by-verdict group and stay in the generic
M6 listing. `b5bc905326b6` owns NO gate decisions (natural
valid-empty for all three values). Runners-up inspected and set
aside: `SuiteRunRecord.state` (all 10 production suite runs evaluate
a checkpoint — single-valued, weak), `SuiteRunStatus` (all
`completed`), `WorkflowRecord.composition` (a provenance list, not an
enum/registry — no clean contract), `SampleEvaluationRecord.
window_rule` (single-valued `single_window`) and `sample_id` (both
production records share ONE sample id).

Copy-ready M43 prompt:

> M43 — GATE-DECISION HISTORY BY VERDICT
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped gate-decision records by the persisted `verdict` enum
> field (the loss-only comparison verdict recorded by the M6 run).
> Do NOT jump ahead into aggregation, rankings, auto-gating,
> rollback, retraining, optimization, training-method expansion, or
> autonomous improvement. Preserve the minimum-files/minimum-storage
> architecture and the established authoritative-listing ->
> persisted-field-filter -> response pattern.
>
> STEP 0 — VERIFY STATE AND BASELINE. Inspect the repository and git
> state. If the familiar reset signature appears (HEAD `f86b670` /
> "Add files via upload", files untracked, production data missing,
> venv missing), execute the established recovery runbook: `git
> fetch` + `git reset --hard FETCH_HEAD` (never force-push, never
> discard authoritative commits), restore `ai-model-forge-data/` from
> the workspace zip, verify against `m42_pre.sha256`, rebuild the
> venv (`python3 -m venv ~/.venv` + `pip install -e
> ai-model-forge[dev] pyflakes`), then re-prove the baseline.
> Expected baseline: M42 certified; HEAD/FETCH_HEAD inventory commit
> = the M42 inventory hash; 520 tests passing; OpenAPI 74; production
> 96 files / 4,002,745 bytes / 0 tmp; zero SHA drift; production
> workflow distribution of `4a0a871886ef`: completed -> 9 / failed ->
> 3 / stopped -> 1; `b5bc905326b6` has zero workflows. Capture
> `m43_pre.sha256` and record the exact baseline before changing
> anything. Inspect `app/gates.py` (M41
> `list_decisions_for_decision()`), `app/engine.py`, `app/api.py`
> (the `/models/{id}/gates/decisions/...` family), `app/schemas.py`
> (`GateDecision.verdict: Optional[ComparisonVerdict]` — None for
> threshold-only gates), and the M41/M42 test blocks in
> `tests/test_gates.py` / `tests/test_gates_api.py`; run the complete
> suite before editing.
>
> GROUNDED SELECTION (live-verified at M42 end): `GateDecision.
> verdict` — the persisted top-level `ComparisonVerdict` enum
> (improved/regressed/unchanged); production distribution improved
> -> 4 / regressed -> 3 / unchanged -> 2 (all three non-empty) plus 2
> None threshold-only decisions that belong to NO group; M6 listing
> order `(created_at, decision_id)` ASC; no by-verdict gate surface
> exists in M1–M42. DISTINCT from M39 (comparisons by verdict — M5
> records) and from M41 (by-decision — the policy verdict): this
> groups DECISIONS by their recorded loss-only verdict, verbatim.
>
> IMPLEMENTATION: In `app/gates.py` add
> `list_decisions_for_verdict(model_id, verdict)` following the M41
> thin pattern: validate the model via the authoritative M6 listing
> mechanism (`list_decisions(model_id)`); filter ONLY the persisted
> `record.verdict == verdict` (never recalculated from loss deltas or
> comparisons; never resolved or rewritten; no gate re-evaluated);
> preserve the M6 `(created_at, decision_id)` ASC ordering; `[]` for
> a valid model with no matching records; unknown model -> existing
> FileNotFoundError/404; invalid verdict enum -> native FastAPI/
> Pydantic 422 via a `ComparisonVerdict` path parameter; threshold-
> only decisions (verdict None) belong to NO group but stay listed;
> no cross-model leakage; no writes; no cache/index/DB/secondary
> registry. In `app/engine.py` expose it through the facade
> (sibling-facade style, no duplicated filtering). In `app/api.py`
> add exactly: `GET /api/v1/models/{model_id}/gates/decisions/
> by-verdict/{verdict}` with `response_model=list[GateDecision]`,
> tags `["gates"]`, registered after the M41 by-decision route and
> BEFORE the generic `/gates/decisions/{decision_id}` route (so
> "by-verdict" can never be interpreted as a decision id).
> Documentation: landing bullet + endpoint `<li>` + section comment
> (verify M41/M42 anchors before editing); README M43 section +
> endpoint count 520 -> ~525 + layout line update.
>
> TESTS: ~3 engine tests + 2 API tests following the M41 conventions
> in `tests/test_gates.py` / `tests/test_gates_api.py`: all three
> verdict groups with listing parity (derive membership from the
> authoritative M6 listing, never hard-coded ids); deterministic
> ordering; persisted verdict verbatim; partition (improved +
> regressed + unchanged == the non-null decisions, disjoint;
> threshold-only None decisions in NO group but listed); valid-empty
> (a model with no gate decisions -> 200 []); unknown model -> 404;
> invalid verdict -> 422 (case variant, interior-space, numeric)
> incl. unknown-model + invalid -> 422 pre-handler; cross-model
> isolation; byte-identical repeats; no writes; M23 by-policy + M34
> by-comparison + M41 by-decision regressions (parity or
> natural-valid-empty per the harness — never manufacture artifacts).
> Target 520 -> ~525 (report the actual).
>
> OPENAPI: 74 -> 75 paths. Update every genuine path-count assertion
> (search the whole repo; do not blindly replace unrelated numbers).
> Verify: exactly one new path, GET only, tag gates, items $ref
> GateDecision, verdict parameter $ref ComparisonVerdict, family
> ordering by-decision < by-verdict < generic detail, no duplicate
> routes, no unintended methods. Do not weaken existing assertions.
>
> FULL VERIFICATION: focused gate tests; API tests; complete suite;
> complete suite a second time; compileall; pyflakes; OpenAPI
> verification; production storage audit; SHA verification against
> `m43_pre.sha256`. Clean stale `/tmp/forge-tests-*` after verifying
> no live process. Expected: tests 520 -> ~525, OpenAPI 75, storage
> unchanged 96/4,002,745/0, zero drift.
>
> LIVE SMOKE: create `smoke_m43_live.py`, port **8764**, following
> the established A–M structure: baseline inventory capture; live
> DISCOVERY of the verdict distribution (expected grounding:
> improved -> 4 / regressed -> 3 / unchanged -> 2 + 2 None
> threshold-only; report the actual if legitimately different —
> never invent); three-group parity; partition 4+3+2 == 9 non-null
> decisions (the 2 None decisions in no group but listed); ordering;
> byte-identical repeats; unknown model -> 404; invalid verdict ->
> 422; second model -> 200 []; cross-model isolation; M23/M34/M41
> gate surfaces unchanged; M42 by-status unchanged; M18–M41
> regressions unchanged; dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
> unchanged; policy/probe/recipe registries unchanged; storage
> remains 96 files / 4,002,745 bytes / 0 tmp; SHA256 0 changed / 0
> new / 0 missing; no new production artifacts.
>
> FINAL AUDIT + REPORT: inspect git diff/status; verify only
> intended files changed; verify production unchanged; no
> force-push; local HEAD == authoritative remote HEAD; only egg-info
> untracked. Commit the implementation (`M43: add gate-decision
> history by verdict`) and the inventory (`m43_pre.sha256`)
> separately per the M21–M42 convention; push normally. Final report
> with EXACTLY these 9 sections: 1. Recovery & Baseline, 2.
> Implementation, 3. Tests, 4. OpenAPI, 5. Live Smoke, 6. Storage
> Integrity, 7. Regression / Compatibility, 8. Git / Commit State,
> 9. Next Milestone. Section 9 must automatically select M44 from
> LIVE persisted production relationships discovered at the end of
> M43 (inspect every remaining persisted enum/identity axis across
> evaluations, comparisons, samples, sample-quality, suite-runs,
> gates, workflows, recipes; prefer an existing enum/registry with
> >= 2 non-empty live groups, thin filter, no schema redesign, no
> new storage; mention the runner-up) and include a COMPLETE
> copy-ready M44 prompt with the same structure as this one (Step 0
> baseline/recovery, grounded selection, exact endpoint/engine
> method/field/enum, route ordering, tests, OpenAPI count, smoke
> port 8765, storage invariants, the exact 9-section report
> requirement, automatic M45 selection). Do not jump to advanced
> training, autonomous improvement, HPO, RL, Gemini integration,
> inference, deployment, or other future architecture unless the
> live evidence specifically makes that the next justified
> milestone.
