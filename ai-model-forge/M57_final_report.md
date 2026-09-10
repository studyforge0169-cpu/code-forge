# M57 Final Report — Declarative Best Baseline for Gate Policies

## 1. Scope & Intent

M57 closes the last non-declarative slot of the canonical improvement
loop: a workflow/recipe GATE stage's INLINE policy may now declare
`baseline_from_best: true` — "judge the candidate against the best
checkpoint that existed when this workflow plan was resolved" — resolved
through the SAME M52 best-checkpoint selector and the SAME M53/M55/M56
workflow state-resolution architecture. The canonical loop

```text
train → evaluate(best) → gate(candidate vs best) → train(resume_from_best) → evaluate(best)
```

is now expressible end-to-end as ONE registered, immutable, re-runnable
recipe with zero hard-coded checkpoint ids. No automatic loops, no
rollback, no new gate algorithm — only the declarative baseline.

Environment note: the M56 push had been blocked by a stale sandbox token;
GitHub was reconnected this session and **M56 commits `8d91b95` + `255b592`
were pushed** before M57 work began (`3606de9..255b592`). Production is the
honestly-rebuilt root documented in the M56 report §7 (no further rebuild
was needed this milestone).

## 2. Architecture Inspection

Recorded before editing (all verified on HEAD `255b592`):

* **Tests**: 584 passed (exit 0, all dots); **OpenAPI 84**; compileall +
  pyflakes clean; worktree clean.
* **Storage**: 39 files / 7,644,998 B / 0 tmp — byte-exactly the
  certified post-M56 state (25 byte-identical + 3 modified + 11 new vs
  `m56_pre.sha256`); no servers running.
* **Gate baseline path** (traced): recipe → `WorkflowStage(gate)` →
  `WorkflowGateStage` (`policy` inline XOR `policy_id` registry, M9) →
  `GateRequest` → `GateEngine.run` → `gate_policy_for` (one policy
  resolution point) → `_resolve_baseline` per `baseline_type`
  (checkpoint / current / evaluation_result_hash / minimum_loss) →
  M5 `verified_state_hash` + `resolve_evaluation` (exact evidence reuse)
  → M5 comparison (reuse-or-create) → `_decide` (improved/unchanged →
  passed; regressed → passed only within `max_regression_delta`;
  optional `minimum_loss` ceiling; failed checkpoint-baseline gates
  carry a `suggested_checkpoint_id` — the gate NEVER rolls back).
* **Candidate vs baseline**: the gate CANDIDATE was already a
  `StageStateRef` (best/from_stage/checkpoint since M53); only the
  BASELINE lacked a best form — the exact gap M57 closes.
* **Existing baseline forms**: `GatePolicy.baseline_type` ∈
  {checkpoint (requires `baseline_checkpoint_id`), current,
  evaluation_result_hash, minimum_loss} with `_baseline_consistent`
  XOR-style validation; direct `POST /gates/evaluate` (ValueError → 422,
  FileNotFoundError → 404); M9 `register_policy` (immutable,
  idempotent-for-identical, 409 on content conflict).

## 3. Implementation

Six files touched, all minimal anchored edits in the established
M55/M56 style:

* `app/schemas.py` — `GatePolicy` gains exactly two fields:
  `baseline_from_best: bool = False` (declarative, workflow-only) and
  `resolved_baseline_checkpoint_id: Optional[str]` (the M53 pinned form,
  resolver-only, valid only with best). `_baseline_consistent` extended:
  best XOR `baseline_checkpoint_id`; best requires
  `baseline_type='checkpoint'` (the best IS a checkpoint baseline);
  a pin without best is rejected; the checkpoint branch's id requirement
  is relaxed only for best ("requires baseline_checkpoint_id (or the
  declarative baseline_from_best=True)"). Every existing baseline form
  and rule is unchanged; `minimum_loss`/`max_regression_delta`
  ride-alongs stay legal with best (it is a state baseline).
* `app/workflows.py` — `resolve_best_state_refs` (the ONE resolver)
  gains `_gate_baseline_unresolved()` in its needs-check and a GATE pin
  branch (alongside the existing candidate pin): the SAME single
  per-plan `select_best_checkpoint` call pins the concrete id onto the
  INLINE policy (`baseline_from_best=True` +
  `resolved_baseline_checkpoint_id=<id>`). `_execute_stage`'s GATE
  branch converts the pinned form to a **PURE M6 policy**
  (`baseline_checkpoint_id=pin`, best stripped) before `gates.run` —
  the gate engine never queries "best"; an unresolved best raises (no
  silent fallback). `policy_id` stages are untouched: registry
  definitions cannot declare best (below), so only inline policies
  participate.
* `app/gates.py` — direct-run guard at the top of `run()`: any DIRECT
  request whose policy declares `baseline_from_best` (pinned or not) →
  ValueError → 422 ("workflow gate-stage declaration — direct gate
  requests name the baseline explicitly"). Workflow executions convert
  first, so the guard never fires for them.
* `app/policies.py` — `register_policy` rejects best-baseline policies
  (ValueError → 422): registry manifests are immutable and shared
  across executions while the M52 selection is resolved and pinned PER
  PLAN — the pin lives in the workflow record, never in the registry.
  **Decision (documented per §9 of the M57 spec): best-baseline is
  restricted to INLINE policies.** This preserves registry immutability
  and record provenance without materializing registry policies into
  plans (which would rewrite stage provenance).
* `app/api.py` — documentation only (landing bullet, `gates_evaluate`
  and `create_policy` docstrings). **No new routes.**
* `README.md` — M57 section; test counts 584 → 590 (both occurrences).
* `tests/test_workflow_recipes.py` — 6 M57 tests; `smoke_m57_live.py` —
  the executed-once live certification smoke.

Untouched by design: M6 decision semantics, comparison/evaluation
engines, `app/training.py`, `app/recipes.py` (recipe stages embed
`WorkflowStage` payloads; M14 `_qualify_stage` rewrites stage-id
references only — policies reference checkpoint ids, nothing to
rewrite; M51 `_resolve` calls the same resolver), `app/suite_runs.py`.

One resolver, one selector: `select_best_checkpoint` remains defined
once (`app/training.py:214`) and invoked once per plan
(`app/workflows.py`); the gate system contains no best lookup of any
kind.

## 4. Tests & Verification

* **Baseline**: 584 passed, OpenAPI 84, statics clean (§2).
* **New M57 tests: 6** —
  `test_m57_schema_matrix_and_direct_rejection` (all existing baseline
  forms unchanged; best + pinned + ride-alongs valid; every
  contradiction rejected; direct run rejected with nothing persisted;
  registry registration rejected),
  `test_m57_resolution_immutability_and_provenance` (preflight pin ==
  M52 argmin; executed record byte-equal to the preflight plan; the
  persisted decision embeds the PURE M6 policy with the explicit id and
  its baseline side identifies the concrete checkpoint; re-run reuses
  evaluations+comparison, appends only one decision; determinism;
  best-change immutability: old record byte-stable, new preflight pins
  the new best, different `plan_hash`; pre-pinned ids never rewritten;
  recipe manifest byte-identical),
  `test_m57_composite_and_zero_checkpoint_semantics` (M14 expansion
  with qualified ids, ONE selection pinned on both gate policies, ONE
  workflow record, TWO decisions but ONE comparison — the second gate
  reuses the first's exact evidence; zero-checkpoint → established 404
  at preflight AND run with nothing persisted; unknown model 404),
  `test_m57_canonical_loop_plan_start_timing` (the 5-stage loop: ALL
  FOUR best declarations pin the SAME plan-start selection; the gate
  judged tr1's output against the PLAN-START best with pure-M6 decision
  provenance; ev2 reused ev1's evaluation; tr2 resumed from the same
  selection; §15: a better checkpoint appearing after the run leaves
  the old record byte-stable while a re-run re-resolves everywhere with
  a different `plan_hash`),
  `test_m57_discrimination_best_vs_latest` (§16: a non-latest
  checkpoint given the lowest persisted `validation_loss` — the gate
  baseline is that argmin, never the published latest; the M52 route
  agrees),
  `test_m57_api_registration_to_execution` (HTTP end-to-end:
  contradictions → 422 at registration; zero-checkpoint preflight →
  404; preflight pins the locally-computed argmin, ×2 byte-identical;
  execution → completed with the decision embedding the explicit
  baseline; direct gate request → 422; policy registration → 422;
  OpenAPI 84 with both fields in `GatePolicy`).
* **Final count: 590 passed × 2 consecutive full-suite runs** (exit 0,
  zero failures; 194.7s / 208.8s). No M1–M56 test regressed.
* **Statics**: compileall + pyflakes clean over `app`, `tests` and the
  smoke. **OpenAPI: 84 paths unchanged.**
* Error taxonomy verified: unknown model/recipe → 404; no selectable
  checkpoints → the established 404 ("no selectable checkpoints",
  M52's rule — ties/non-finite handling stays M52's, untouched);
  contradictory/invalid declarations → 422 at every boundary; direct
  best declaration → 422; corrupt states keep the existing
  409/verify semantics.

## 5. Live Smoke

Executed **exactly once** against production (port 8779):
**27/27 checks PASSED — zero corrections, zero post-hoc fixes needed**
(the first fully-clean smoke since M54).

* **Pre-state**: 39 files / 7,644,998 B / 0 tmp (`m57_pre.sha256`
  captured first, 39/39 verified); model `31db39e17a20` with 11
  checkpoints; live-computed argmin `0deba72e350b` @ 6.369091 vs
  published latest `86687445867e` — a persisted-loss TIE broken by the
  canonical (step, created_at) tie-break (best != latest as concrete
  ids); M52 endpoint agrees; listings 1 workflow / 1 evaluation / 0
  comparisons / 0 gate decisions / 0 suite-runs / 0 samples / 1 recipe.
* **Recipe**: `m57-live-best-gate` — the canonical 5-stage loop with
  `on_pass` branching and tolerance 1.0 (deterministic pass); registered
  once (+1 file, baseline byte-identical, fully declarative).
* **Preflight**: pinned the live argmin onto **all four** best
  declarations (ev1, gate baseline, tr2 resume, ev2); ×2 byte-identical;
  zero writes.
* **Execution** (one POST → completed): the record's plan is
  byte-identical to the preflight plan; the gate judged tr1's output
  (`1b6442a0fc58`) against the PLAN-START best (`0deba72e350b`) —
  **delta −0.037857 (improved) → passed**; the persisted decision
  `gate-722a06fe8116` embeds the PURE M6 policy (explicit
  `baseline_checkpoint_id=0deba72e350b`, best stripped) and its
  baseline side identifies the concrete checkpoint; **the gate's
  baseline evaluation REUSED ev1's M4 record** and ev2 reused it too
  (evaluations 1 → 3, not 4); tr2 received a PURE M54 resume
  (provenance initial/parent/config) with checkpoints
  `321c4a3ec9d2 ← 0deba72e350b` and `7a5e2e33f75e ← 321c4a3ec9d2`;
  recipe manifest and the selected checkpoint byte-identical; no
  rollback (latest = `7a5e2e33f75e`, a loop checkpoint); no
  best-pointer file.
* **Regression**: M52 resolves the live argmin over the grown registry —
  which MOVED to `1b6442a0fc58` @ **6.331234** (tr1's output — the loop
  genuinely improved again); the SAME recipe's preflight re-resolves
  the current argmin onto all four declarations (no cross-time lock);
  dashboard diff confined to checkpoints/training-runs/evaluations/
  comparisons/gate-decisions/workflows/artifact-graph/summary;
  suite-runs/samples unchanged; OpenAPI 84 with both fields.

## 6. Storage & Integrity

```
files before:  39         files after:  53         delta: +14 files
bytes before:  7,644,998  bytes after: 10,193,156   delta: +2,548,158 B
tmp files:     0 before / 0 after
```

vs `m57_pre.sha256`: **36 byte-identical, 3 modified, 14 new**. Every new
file justified (verified by the smoke's exact audit):

* `workflow-recipes/m57-live-best-gate/manifest.json` — the ONE new
  declarative recipe;
* `models/…/workflows/workflow-eb49104f5bfc/manifest.json` — the ONE
  workflow record;
* 8 checkpoint files — 2 checkpoints × (manifest + weights) per train
  stage (`0bae5e481ff6`, `1b6442a0fc58` from tr1; `321c4a3ec9d2`,
  `7a5e2e33f75e` from tr2);
* `evaluations/eval-b188b554e0cd` (best @ probe) and
  `evaluations/eval-ea4d70e46d31` (the gate's candidate) — exactly two,
  thanks to reuse;
* `comparisons/comp-07d354bb54d6/manifest.json` — the ONE comparison
  backing the gate;
* `gates/gate-722a06fe8116/manifest.json` — the ONE gate decision.

Modified (3, existing semantics): model manifest (provenance +
publication), `weights.pt` + `weights.sha256` (keep-best publication).
No weights copied; the selected checkpoint byte-identical; no
best-pointer file, no gate-baseline registry, no new record family, no
cached best results. Zero `.tmp` leftovers; server stopped after the
smoke (0 uvicorn verified).

## 7. API / OpenAPI

**OpenAPI path count unchanged: 84** — the capability is exposed purely
through schema fields on the existing workflow/recipe endpoints
(`WorkflowStage.gate.policy` inside recipe registration, M51 preflight
and recipe runs). `GatePolicy` gains `baseline_from_best` +
`resolved_baseline_checkpoint_id` in the OpenAPI schema (asserted in
tests and live). The DIRECT routes stay strict by design: a
`POST /gates/evaluate` body declaring `baseline_from_best` is rejected
422 (workflow-only declaration), and `POST /policies` rejects
best-baseline policies 422 (registry immutability). No new status codes.

## 8. Limitations

* **Resolution timing**: best resolves ONCE per plan at plan start (the
  unchanged M53 architecture). A gate with `baseline_from_best` judges
  against the best that existed when the plan resolved — a checkpoint
  created later in the SAME plan is reachable only through the
  candidate's explicit `from_stage` (as the canonical loop does).
  Multiple `best` declarations in one plan (evaluations, gate baseline,
  resumes, refs) intentionally share the SAME plan-start selection.
  Re-resolution happens on the NEXT execution — proven live
  (improved=True, preflight re-pins `1b6442a0fc58`).
* **Inline-only**: `baseline_from_best` lives on the gate stage's INLINE
  policy. Registry (`policy_id`) policies cannot declare best — the
  concrete pin belongs to the per-plan workflow record, never to an
  immutable shared manifest. Direct gate requests cannot declare best
  (422; `GET /models/{id}/checkpoints/best` answers the id).
* **Evidence reuse** is exact-identity (concrete checkpoints + identical
  probe): declaring best never duplicates evidence, but a moved best
  legitimately produces new evaluations on the next execution.
* **What M57 does NOT do**: no automatic loops/repetition/retraining, no
  auto-rollback (failed gates still only SUGGEST a rollback target), no
  accept/publish automation beyond the existing keep-best publication
  semantics, no mutable best pointer, no second selector. The
  improvement loop remains finite, explicit and user-driven — M57 gives
  it a fully declarative vocabulary, not autonomy.

## 9. Next Milestone

Inspection of the completed architecture: every step of
`train → evaluate → compare/gate → select → resume → evaluate →
accept/reject` is now declarative and live-proven — the only missing
word is **repeat**. Today, iterating the loop means re-issuing the same
recipe run N times manually (each execution re-resolves best at its own
plan start, which is exactly why the loop advances — verified live in
M56 and M57). The smallest high-value next milestone is therefore
**M58 — bounded finite recipe repetitions**: an explicit, user-declared
`repetitions: N` on the recipe-run request, executed as N sequential
resolve+execute cycles (N immutable workflow records, each with its own
plan-start resolution), aborting remaining repetitions if an iteration
ends `failed` or `stopped` (conservative, deterministic). No
while-improving, no auto-stop heuristics, no background execution — the
honest "repeat" primitive. Per-stage re-resolution inside one plan was
considered and rejected as the next step: it would silently change the
documented M53 once-per-plan timing.

---

### M58 — BOUNDED FINITE RECIPE REPETITIONS (ready-to-paste prompt)

You are continuing development of **AI Model Forge**.

Implement this milestone incrementally on top of the certified **M57**
state.

## Core objective

Add the explicit, bounded "repeat" primitive to the improvement-control
loop. A recipe run may declare a finite repetition count:

```text
POST /workflows/recipes/{recipe_id}/runs
    { "model_id": "<id>", "repetitions": 3 }
```

Semantics: **N sequential executions of the SAME registered recipe**,
each one a FULL existing resolution+execution cycle — each iteration
re-resolves every `best` declaration (M53 refs, M55 resume, M56
evaluate, M57 gate baseline) against the checkpoints existing at ITS
plan start, so the improvement loop genuinely advances across
iterations. N immutable workflow records are produced; no wrapper
record, no nested records.

This is NOT an automatic improvement loop: N is fixed, explicit and
user-declared; no while-improving, no auto-stop heuristics, no
background execution.

## 1. INSPECT FIRST — DO NOT EDIT YET

Run `git status`, `git log --oneline -10`, `pytest -q`. Expected
baseline: **590 tests, OpenAPI 84, worktree clean, M57 implementation
present** (if the environment was reset, follow the proven recovery
procedure: fetch the branch from origin, verify the worktree
byte-identical to the pushed HEAD, rebuild the venv — torch from plain
PyPI — re-certify the baseline, and rebuild production honestly through
the public API if needed, documenting it).

Inspect: the recipe-run request schema (`RecipeRunRequest` or the actual
inline body model), `app/recipes.py` (`run`, `_resolve`, M14 expansion),
`app/workflows.py` (`run`, `resolve_best_state_refs`, record
persistence, statuses), `app/api.py` (the recipe-runs route and its
response model), M35 workflow-history-by-recipe, the M51 preflight
route, and all M53–M57 tests. Do not assume field names.

## 2. DESIGN DECISIONS (choose, test, document)

* **Where the count lives**: prefer a REQUEST-level
  `repetitions: int = Field(1, ge=1, le=<sane bound>)` on the recipe-run
  request (recipes stay immutable single-iteration DEFINITIONS; the
  count is an execution parameter — a recipe-level field would bake a
  loop count into a reusable definition and complicate M12
  immutability). Default 1 must reproduce today's behavior byte-for-byte
  (response shape included).
* **Response shape**: with repetitions > 1 the natural response is the
  LIST of executed `WorkflowRecord`s (in execution order). Decide
  between always-list (a breaking change for repetitions=1) vs
  list-only-for-N>1 vs a dedicated response model — prefer the SMALLEST
  honest change and justify it; OpenAPI path count must stay 84 unless
  inspection proves otherwise.
* **Stop semantics**: if iteration k ends with status `failed` or
  `stopped` (e.g. a gate's stop path), the remaining repetitions do NOT
  run (conservative and deterministic). All executed records are
  returned; nothing is retried. Document that a `stopped` iteration is
  a NORMAL gate outcome, not an error.
* **Preflight (M51)** stays a SINGLE-iteration resolution preview —
  it must not grow repetition semantics (document why: a preflight
  cannot predict later iterations' resolutions because each iteration
  re-resolves against the state its predecessors created).

## 3. IMPLEMENTATION BOUNDARIES

* ONE executor: each iteration goes through the EXACT existing
  `RecipeEngine.run` → `WorkflowEngine.run` path — no second execution
  system, no batching inside one plan (that would wrongly share ONE
  per-plan best selection across iterations).
* No new best selector, no new resolver, no new record family, no
  wrapper/supervisor record, no background workers, no queues.
* Evidence reuse continues automatically (each iteration's
  evaluate/gate/comparison reuses exact existing records where identity
  matches — but a moved best legitimately creates new evaluations).
* Storage discipline: only the normal per-iteration artifacts; no
  copies, no caches; every new file justified in the audit.
* Inline `POST /workflows/run` plans get NO repetition parameter
  (recipes have registry identity + M35 lineage; inline plans do not) —
  document this boundary.

## 4. TEST MATRIX

Focused M58 tests: repetitions=1 is byte-identical to today's run
(response shape, record content, provenance); repetitions=3 produces 3
records whose best-pins may legitimately DIFFER (prove the loop
advances: iteration 2's gate/resume resolve the argmin iteration 1
created); stop-on-failure (a gate whose stop path triggers at iteration
  k → exactly k records, statuses honest, no retry); repetitions=0 /
negative / non-integer → 422; unknown recipe/model → 404 with zero
records; M35 history-by-recipe lists all N records; M51 preflight
unchanged (single-iteration); determinism where the registry state is
unchanged; all M1–M57 tests green. Report exact counts (baseline 590).

## 5. LIVE SMOKE (exactly once)

After the full suite passes twice, inspect production (read the M57
report §6 facts first). Execute ONE minimal state-mutating smoke: a
small recipe (the canonical loop or a minimal train→evaluate(best)→
gate(best) variant) with repetitions=2, verifying: two immutable
records; iteration 2's best-pins differ from iteration 1's where the
state improved (or honestly document a tie); evidence reuse across
iterations where identity matches; M35 listing shows both records;
storage delta exactly the per-iteration artifacts; recipe manifest
byte-identical; no wrapper records; OpenAPI 84. Capture
`m58_pre.sha256` BEFORE the smoke. If an assertion is wrong but the
implementation is correct, follow the M55/M56 discipline: diagnose,
correct the expectation, read-only post-hoc verification, document —
never re-run a state-mutating smoke.

## 6. FINAL QUALITY GATE + REPORT

`pytest` ×2 on the final state, statics, OpenAPI count, storage audit
(files/bytes before→after, every new file justified, untouched files
byte-identical), `git diff`/`git status`. Provide the final report with
exactly 9 sections (Scope & Intent; Architecture Inspection;
Implementation; Tests & Verification; Live Smoke; Storage & Integrity;
API / OpenAPI; Limitations; Next Milestone + a complete ready-to-paste
prompt for the next milestone — candidates to evaluate against the
ACTUAL remaining gaps: publish-on-pass accept semantics; per-iteration
stop conditions (gate-driven early stop across repetitions);
suite-run/sampling integration of the loop). Exact numbers only; do not
fabricate. Philosophy: minimum files + minimum storage + maximum
correctness + deterministic behavior + immutable provenance +
same-model iterative improvement.
