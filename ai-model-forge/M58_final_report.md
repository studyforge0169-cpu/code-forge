# M58 Final Report — Bounded Finite Recipe Repetitions

## 1. Scope & Intent

M58 adds the bounded, explicit "repeat" primitive to the improvement
loop: the recipe-run request may declare `repetitions: N` (default 1,
bounded **1..16** — a deliberately conservative finite maximum chosen
per project convention since no prior bound existed for this concept),
executing the recipe **N times sequentially**, each iteration a **FULL
independent resolution + execution cycle** through the existing
single-run path. This is the entire point: every `best` declaration
(M53 refs, M55 resume, M56 evaluate, M57 gate baseline) re-resolves at
THAT iteration's plan start, so the loop advances across iterations
instead of replaying one frozen resolution. NOT an automatic
improvement loop: N is fixed, explicit, user-declared — no
while-improving, no convergence detection, no retry, no rollback, no
background execution.

Baseline recorded before editing: HEAD `1bd7f14` (pushed, worktree
clean), **590 tests passed**, OpenAPI 84, statics clean, production
53 files / 10,193,156 B / 0 tmp (exactly the certified post-M57 state),
no servers running.

## 2. Architecture Inspection

Traced before editing: the recipe-run route
(`POST /workflows/recipes/{recipe_id}/runs`, body
`WorkflowRecipeRunRequest{model_id}`, response `WorkflowRecord`) →
facade `run_workflow_recipe` → `RecipeEngine.run` (lookup → validate →
M14 expansion → `WorkflowPlan` → M53 resolution → ONE
`WorkflowEngine.run` → ONE normal immutable record; a failing stage
persists a `failed` record and RE-RAISES; a gate stop RETURNS a
`stopped` record — `WorkflowStatus` ∈ {completed, failed, stopped});
M51 preflight (`resolve`) shares `_resolve` but never executes; M35
by-recipe history; existing bounded-int conventions
(`max_new_tokens le=512`, `vocab_size le=65536`) informed the MAX=16
choice; engine `run()` returns the record, so an exception is the only
failure signal (the failed record stays persisted and discoverable).

## 3. Implementation

Five files touched, minimal additive edits:

* `app/schemas.py` — `WorkflowRecipeRunRequest` gains
  `repetitions: int = Field(1, ge=1, le=16)` with a `field_validator`
  mapping explicit `null` → 1 (never "forever"); plus ONE new
  response-only model `WorkflowRecipeRepetitionRun`
  (recipe_id, model_id, repetitions, executed, stopped_early,
  ordered `workflow_ids`, full `records`) — an ordered VIEW, never a
  wrapper record, never persisted. **`WorkflowRecord` itself is
  untouched** (result_hash, storage and record shape unchanged).
* `app/recipes.py` — `run_repeated(recipe_id, model_id, repetitions)`:
  a thin sequential loop over the EXISTING `run()`; breaks when an
  iteration ends `stopped`; a `failed` iteration keeps the existing
  semantics (the exception propagates after the failed record
  persists). No second executor, no resolution caching, no skipping.
* `app/engine.py` — facade `run_workflow_recipe_repeated` (delegation
  only).
* `app/api.py` — the SAME route (no new paths): `repetitions == 1`
  takes today's exact code path and returns the single record;
  `repetitions > 1` returns the ordered batch. Response model becomes
  `WorkflowRecord | WorkflowRecipeRepetitionRun`. Landing-page bullet
  added.
* `README.md` — M58 section; test counts 590 → 596 (both occurrences).
* `tests/test_workflow_recipes.py` — 6 M58 tests;
  `smoke_m58_live.py` — the executed-once live certification smoke.

Untouched by design: `app/workflows.py` (the executor and the M53
resolver are already per-plan — repetition simply calls them N times),
`app/training.py`, `app/evaluation.py`, `app/gates.py`,
`app/suite_runs.py`. The registered recipe definition carries NO
repetition field (the count is a request parameter — verified live).

## 4. Tests & Verification

* **New M58 tests: 6** —
  `test_m58_request_validation_and_backward_compat` (default/null→1;
  1, 16 accepted; 0/-1/17/1.5/"two" rejected; single-run path and the
  recipe manifest byte-identical; a later single run pins the current
  argmin captured BEFORE the run),
  `test_m58_best_advances_across_iterations` (§15, DETERMINISTIC: a
  fresh model whose pre-existing checkpoints are all pinned at loss
  9.0 guarantees iteration 1's output becomes the new best —
  iteration 2 pins it, hashes differ, provenance concrete, 2 normal
  records),
  `test_m58_canonical_loop_repetitions` (§16: the 5-stage loop ×2 —
  two records; iteration 2's pins == the argmin over the registry
  EXCLUDING its own outputs, recomputed read-only; per-iteration gates
  pass with pure-M6 provenance; hash-inequality exactly tracks
  resolution differences — including the honest full-replay case where
  an iteration that beats nothing republishes nothing and iteration 2
  deterministically replays with FULL evidence reuse; M35 lists both),
  `test_m58_abort_semantics` (a `stopped` gate outcome ends the
  sequence with the ONE stopped record; a raising stage persists one
  `failed` record, propagates, and later iterations never run),
  `test_m58_composite_repetitions` (M14 parent→child ×2, per-iteration
  expansion, no nested records, bounded evidence growth),
  `test_m58_api_batch_response_and_history` (HTTP: invalid
  repetitions → 422; unknown recipe → 404; legacy body and null →
  today's exact single-record shape; repetitions=2 → ordered batch
  with counts/ids/records; M35 lists both; preflight stays a
  single-iteration preview; OpenAPI: 84 paths, bounded field with
  default 1 / maximum 16, batch component present).
* **Final count: 596 passed × 2 consecutive full-suite runs** (exit 0,
  zero failures). No M1–M57 test regressed. Two test-authoring
  corrections during development (hash-inequality made conditional on
  the actual resolution difference; evidence deltas widened to the
  honest reuse-dependent ranges) — both were over-strong expectations
  about the shared fixture state, not implementation issues.
* **Statics**: compileall + pyflakes clean over `app`, `tests`, the
  smoke. **OpenAPI: 84 paths unchanged** (schema-only change: the
  request field + the batch response component).

## 5. Live Smoke

Executed **exactly once** against production (port 8780): **27 checks,
26 passed in the run; 1 corrected with read-only post-hoc
verification** (B3 searched for the substring "repetitions" in the
recipe manifest JSON — the recipe's own human description text contains
"repetitions=2"; the implementation was correct: NO `repetitions` KEY
exists anywhere in the manifest structure, verified by a recursive key
walk; no re-execution, M55/M56/M57 discipline). Effective
certification 27/27.

* **Pre-state**: 53 files / 10,193,156 B / 0 tmp (`m58_pre.sha256`
  captured first, 53/53 verified); argmin `1b6442a0fc58` @ 6.331234 vs
  latest `7a5e2e33f75e` (persisted-loss tie, canonical tie-break);
  listings 2 workflows / 3 evaluations / 1 comparison / 1 gate / 0
  suite-runs / 0 samples / 2 recipes.
* **Recipe**: `m58-live-loop-x2` — the canonical 5-stage loop,
  registered once (+1 file, baseline byte-identical, fully declarative,
  no repetition field).
* **Preflight** (unchanged, single-iteration): pinned `1b6442a0fc58`
  onto all four declarations; ×2 byte-identical; zero writes.
* **Repeated execution** (ONE POST, `repetitions: 2` → 200 batch):
  `repetitions=2, executed=2, stopped_early=false`, records
  `workflow-a85965bed306` (iteration 1) and `workflow-6a63ebc817ce`
  (iteration 2), both `completed` with recipe provenance, no wrapper.
  * **Iteration 1** pinned `1b6442a0fc58` (== the preflight); its tr1
    produced `85c0ea2b6d52` @ **6.2908**; the gate judged tr1's output
    vs the plan-start best — **delta −0.040391 → passed**; tr2 resumed
    from `1b6442a0fc58` → `74b0d072b604`.
  * **Iteration 2 RE-RESOLVED**: pins advanced to `85c0ea2b6d52` —
    exactly the M52 argmin over the registry excluding iteration 2's
    own outputs (recomputed live from persisted manifests); its tr1
    produced `41870af53ab7` @ **6.2424**; gate **delta −0.048442 →
    passed**; tr2 → `eaf0984fbc2b`. Plan hashes differ
    (`001352269842…` vs `ed6fc7685fb7…`) — advanced=True.
  * Evidence reuse: ev2 reused ev1 in both iterations; evaluations
    3 → 5 (+2), comparisons 1 → 3 (+2), gate decisions 1 → 3 (exactly
    +2, append-only); every gate decision embeds the pure M6 policy.
  * Recipe manifest byte-identical after execution; no rollback
    (latest `eaf0984fbc2b`, a loop checkpoint); no best-pointer or
    repetition storage.
* **Regression**: M52 resolves the new argmin `41870af53ab7` @
  **6.242401** (from 6.331234 — the loop improved twice, ~0.089 nats);
  M35 lists both records; the same recipe's single-iteration preflight
  re-resolves the current argmin; dashboard diff confined to the
  expected sections; suite-runs/samples unchanged; OpenAPI 84 with the
  bounded field (default 1, maximum 16) and the batch component.

## 6. Storage & Integrity

```
files before:  53          files after:  78          delta: +25 files
bytes before:  10,193,156  bytes after: 15,281,640    delta: +5,088,484 B
tmp files:     0 before / 0 after
```

vs `m58_pre.sha256`: **50 byte-identical, 3 modified, 25 new**. Every
new file justified (verified by the smoke's exact structural audit):
1 recipe manifest, 2 workflow records, 16 checkpoint files (2
iterations × 2 train stages × 2 checkpoints × manifest+weights), 2
evaluations, 2 comparisons, 2 gate decisions — `other 0`. Modified (3,
existing semantics): model manifest, `weights.pt`, `weights.sha256`
(normal keep-best publication). No weights duplicated, no
`recipe-runs/`/`repetitions/`/`loop-state/`/`best-history/` trees, no
wrapper records, zero `.tmp`. Server stopped after the smoke (0
uvicorn verified).

## 7. API / OpenAPI

**No new routes** — the capability lives on the existing
`POST /workflows/recipes/{recipe_id}/runs`. **OpenAPI path count
unchanged: 84.** The schema-only changes (reported per §32): the
request model gains `repetitions` (integer, default 1, minimum 1,
maximum 16 — asserted in tests and live), and the route's response
becomes `anyOf[WorkflowRecord, WorkflowRecipeRepetitionRun]` with the
new `WorkflowRecipeRepetitionRun` component (the ordered batch view).
`repetitions=1`/omitted/null returns exactly today's single-record
response (backward compatible); `repetitions>1` returns the batch.
M51 preflight is deliberately untouched (a single-iteration preview —
later iterations cannot be predicted because each re-resolves against
the state its predecessors created). Inline `POST /workflows/run`
plans get no repetition parameter (recipes have registry identity +
M35 lineage; inline plans do not).

## 8. Limitations

* **Bounded and explicit only**: 1..16; no unbounded mode, no null
  meaning forever, no while-improving, no convergence/quality-based
  automatic stopping, no retry, no skip, no automatic rollback (a
  failed gate or stage simply ends the sequence; publishing a chosen
  checkpoint stays the existing explicit M3 rollback call).
* **Failure visibility**: an iteration whose stage raises maps to the
  existing 404/409/422 error semantics — the failed record (and all
  earlier records) are persisted and discoverable via M35, but the
  HTTP error response itself does not carry the batch view. A
  `stopped` iteration (normal gate outcome) DOES return the batch
  with `stopped_early=true`.
* **Repetition count is request-level**: not recorded on the workflow
  records themselves (records stay 100% M7/M12/M14-shaped); the batch
  response and the M35 by-recipe ordering are the caller's views.
* **Determinism nuance** (documented in tests): when an iteration's
  runs beat the model baseline, published weights advance and the next
  iteration explores new states; when they do not (M3 acceptance),
  published weights stay and the next iteration deterministically
  replays the same resolved plan with full evidence reuse — both
  behaviors are correct, and plan-hash equality/difference tracks the
  resolution equality/difference exactly.
* **Not autonomy**: M58 gives the loop a bounded repeat primitive, not
  decision-making. The user still decides N, still reads the results,
  still accepts (rollback-to-best) manually.

## 9. Next Milestone

Inspection of the completed architecture: the control loop
`train → evaluate → compare/gate → select → resume → evaluate →
accept/reject → repeat` is now fully expressible and live-proven —
including bounded repetition with per-iteration re-resolution. The
smallest remaining gap is **observability of the improvement
trajectory**: after N repetitions the user can list checkpoints (M46
by-run, M24 evaluations-by-checkpoint) and ask M52 for the current
best, but NO existing surface shows HOW the best selection evolved —
the sequence of argmin movements with losses and deltas that tells the
user whether another repetition is worth running (the exact decision
M58 leaves to the user). **M59 = read-only best-checkpoint history**
(`GET /models/{model_id}/checkpoints/best/history`): the ordered
timeline of M52-argmin movements over the persisted checkpoint
manifests (checkpoint id, validation_loss, created_at, producing run,
delta vs the previous best), computed live, zero storage, consistent
by construction with the M52 selection (the final entry IS the M52
answer). One new read-only route in the established M18–M50
history-surface family — a genuine gap, not a redundant API.

---

### M59 — READ-ONLY BEST-CHECKPOINT IMPROVEMENT HISTORY (ready-to-paste prompt)

You are continuing development of **AI Model Forge**.

Implement this milestone incrementally on top of the certified **M58**
state.

## Core objective

Make the improvement trajectory observable: a read-only,
model-scoped view of HOW the best-checkpoint selection evolved over
the model's persisted checkpoint history — the timeline of argmin
movements that tells the user whether the improvement loop is working
and whether another repetition is worth running.

```text
GET /api/v1/models/{model_id}/checkpoints/best/history
```

Each entry: the checkpoint that BECAME the best at that point
(checkpoint id, step, persisted validation_loss, perplexity, created_at,
producing run_id, parent lineage), plus the improvement delta vs the
previous best. The FINAL entry must be exactly the current M52 answer
(`GET /models/{id}/checkpoints/best`). Zero persistence — a computed
view like every M18–M50 history surface.

## 1. INSPECT FIRST — DO NOT EDIT YET

Run `git status`, `git log --oneline -10`, `pytest -q`. Expected
baseline: **596 tests, OpenAPI 84 (this milestone WILL add one path:
85), worktree clean, M58 implementation present** (if the environment
was reset, follow the proven recovery procedure: fetch the branch from
origin, verify the worktree byte-identical to the pushed HEAD, rebuild
the venv — torch from plain PyPI — re-certify the baseline, and rebuild
production honestly through the public API if needed, documenting it).

Inspect: `app/training.py` (`select_best_checkpoint`, the M52
ordering/tie-break rules, `CheckpointSelection`), `app/api.py` (the
M52 best route and the M18–M50 history-route conventions: response
models, 404 semantics, ordering documentation), `app/schemas.py`
(`CheckpointRecord`, history-view models), the M24/M46 history tests
for style. Do not assume field names.

## 2. ONE SELECTION SEMANTICS (mandatory)

The timeline's ordering and argmin rule must be EXACTLY M52's:
minimum persisted `validation_loss`, canonical `(step, created_at)`
ASCENDING tie-break (first among equals), non-finite values never
candidates. Reuse the same manifest-reading path the selector uses —
do NOT reimplement a second comparator; if sharing requires extracting
a tiny helper next to `select_best_checkpoint`, do that (one
definition). The final timeline entry MUST equal the live
`select_best_checkpoint` answer (assert it in tests). A model with no
selectable checkpoints follows the established M52 error semantics
(404) — decide consistently with the existing best route and document.

## 3. VIEW DESIGN (minimal)

A new read-only response model (e.g. `BestCheckpointHistory` /
entries) in the established history-surface style: model_id, entry
count, the ordered entries, and (honest computed fields only) each
entry's delta vs the previous best (None for the first). Ordering:
chronological by when each checkpoint BECAME best (i.e. by
`(created_at, checkpoint_id)` of the winning checkpoints — the natural
timeline). No pagination (match the family's conventions). Never
persist; never mutate; no new storage.

## 4. TEST MATRIX

Focused M59 tests: the timeline over a synthetic checkpoint set with
known losses/ties (fixture edits in throwaway storage are allowed for
ENGINE tests) — exact expected movement sequence including a
tie resolved by the canonical tie-break and a later checkpoint that
does NOT beat the best (no new entry); the FINAL entry equals
`select_best_checkpoint`; deltas correct (first entry None);
determinism (repeat call byte-identical); unknown model → 404; the
no-checkpoint case follows the documented M52-consistent choice;
API-level test (status codes, shape, OpenAPI now 85 with the new
path); regression of all M1–M58 tests. Report exact counts (baseline
596).

## 5. LIVE SMOKE (exactly once — READ-ONLY preferred)

After the full suite passes twice, inspect production (read the M58
report §6 facts first: 78 files / 15,281,640 B, 23 checkpoints, argmin
`41870af53ab7` @ 6.242401). The ideal M59 smoke is READ-ONLY (a pure
computed view): capture `m59_pre.sha256`, call the new history
endpoint ONCE, verify the timeline against independently recomputed
expectations from the raw manifests (movements, deltas, final entry ==
the M52 route answer, byte-identical repeat call), verify ZERO storage
change (files/bytes/inventory identical before/after), and stop. If —
and only if — a read-only smoke cannot prove something essential,
consider one minimal state-mutating execution; never repeat it. If an
assertion is wrong but the implementation is correct, follow the
M55–M58 discipline: diagnose read-only, correct the expectation,
post-hoc verify, document honestly.

## 6. FINAL QUALITY GATE + REPORT

`pytest` ×2 on the final state, statics (compileall, pyflakes),
OpenAPI count (85 — exactly one new path, report it), storage audit
(READ-ONLY smoke: byte-identical before/after), `git diff`/`git
status`/push/verify remote HEAD. Provide the final report with exactly
9 sections (Scope & Intent; Architecture Inspection; Implementation;
Tests & Verification; Live Smoke; Storage & Integrity; API / OpenAPI;
Limitations; Next Milestone + a complete ready-to-paste prompt —
candidates to evaluate against the ACTUAL remaining gaps: an
improvement-summary field on the M58 batch response; declarative
accept/publish semantics; suite-run/sampling integration of the loop).
Exact numbers only; do not fabricate. Philosophy: minimum files +
minimum storage + maximum correctness + deterministic behavior +
immutable provenance + same-model iterative improvement + no premature
autonomous behavior.
