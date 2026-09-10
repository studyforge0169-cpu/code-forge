# M56 Final Report — Best State References for Evaluation Stages

Milestone implemented on the certified M55 state. Environment note first,
because it shaped this milestone's live verification: the sandbox was
rebuilt between M55 and M56 (fresh clone at the base commit; the entire
`ai-model-forge-data` production root, the venv and the `/tmp` backup
were lost). The git history was recovered from origin (`git fetch origin
arena/01a071e9-code-forge`; 62 commits, HEAD `3606de9`), the persisted
working tree was verified **byte-identical** to `3606de9` (all 167 tracked
files; only the 5 known `egg-info` extras), the venv was rebuilt (torch
2.14.0 from plain PyPI), and the baseline was re-certified BEFORE any
editing: **579 tests passed, OpenAPI 84, statics clean, worktree clean**.
Production was then REBUILT from scratch through the public API (see §7) —
every artifact a real API call, ids new, nothing fabricated — and the M56
live smoke was executed **exactly once** against that rebuilt production.

## §1 — M56 implementation summary

M56 closes the last state-reference gap of M53/M55: a workflow/recipe
EVALUATE stage can declaratively consume the M52 best checkpoint.
Changes (6 files touched, all in the established minimal-anchored style):

* `app/schemas.py` — `WorkflowEvaluationStage` gains exactly two fields:
  `checkpoint_from_best: bool = False` (the declarative selector,
  workflow-only) and `resolved_checkpoint_id: Optional[str]` (the M53
  pinned form, set only by the resolver, valid only with
  `checkpoint_from_best=True`). The `_not_ambiguous` validator extends to
  a full three-way XOR: `checkpoint_from_best` conflicts with
  `config.checkpoint_id` AND with `checkpoint_from_stage`; a pin without
  best is rejected. No new model, no `EvaluationBestState`, no
  `BestEvaluationConfig` — the stage keeps its existing flat
  representation and the M55 pattern is mirrored field-for-field.
* `app/workflows.py` — `resolve_best_state_refs` (the ONE M53 resolver)
  gains `_evaluate_unresolved()` in its needs-check and an EVALUATE pin
  branch: the SAME single per-plan `select_best_checkpoint` call pins the
  concrete id onto every unresolved best-evaluation stage
  (`checkpoint_from_best=True` + `resolved_checkpoint_id=<id>`), exactly
  like M53 state refs and M55 train stages. `_execute_stage`'s EVALUATE
  branch converts the pinned form to a **PURE explicit-checkpoint
  `EvaluationConfig`** (`checkpoint_id=pin`) before the M4 path runs —
  the evaluation layer never queries "best"; an unresolved best raises
  (no silent fallback to current weights).
* `app/api.py` — documentation only: one landing-page feature bullet and
  an `evaluation_run` docstring note (the DIRECT M4 route has no such
  field). **No new routes** — OpenAPI paths stay 84.
* `README.md` — M56 section; test counts 579 → 584 (both occurrences).
* `tests/test_workflow_recipes.py` — 5 focused M56 tests (§6).
* `smoke_m56_live.py` — the executed-once live certification smoke.

Untouched by design: `app/evaluation.py` (M4 stays a deterministic
primitive against a concrete state — no `find_best`, no min-loss logic),
`app/recipes.py` (recipe stages embed `WorkflowStage` payloads; the new
fields flow through registration, M14 `_qualify_stage` (best references
no stage id, nothing to rewrite) and the M51 `_resolve` preflight
automatically), `app/suite_runs.py`, `app/training.py`.

## §2 — Evaluation state semantics

Exact mechanisms actually present (verified by inspection, §1/§2 of the
prompt):

| selector | where | meaning |
|---|---|---|
| current | `config.checkpoint_id = None` | the model's live published weights |
| checkpoint | `config.checkpoint_id = "<id>"` | that exact immutable checkpoint |
| from_stage | `checkpoint_from_stage = "<train-stage>"` | FINAL checkpoint of an earlier train stage |
| **best (M56)** | `checkpoint_from_best = true` | the M52 selection, resolved by the workflow resolver |

Conflict/XOR rules (schema-level, 422 at every boundary):

* `checkpoint_from_stage` XOR `config.checkpoint_id` (pre-existing M7);
* `checkpoint_from_best` XOR `config.checkpoint_id` — "the M52 selection
  decides the checkpoint OR it is named explicitly, never both";
* `checkpoint_from_best` XOR `checkpoint_from_stage` — exactly one state
  selector;
* `resolved_checkpoint_id` requires `checkpoint_from_best=True` (a pin
  without best is a contradiction); a pre-pinned id is respected
  (idempotent resolution, verified by test).
* Direct M4 requests (`POST /evaluations/run`, `EvaluationConfig` with
  `extra="forbid"`) have no such field at all: 'best' cannot even be
  expressed — structurally rejected (422, unknown field), verified live
  and by test. This is stronger than M55 (which needed a preflight
  rejection): the workflow-only selector lives on the stage, not the
  config.

## §3 — M52/M53 integration

```
evaluate(best)  (WorkflowEvaluationStage.checkpoint_from_best)
   → WorkflowEngine.resolve_best_state_refs   (the ONE M53 resolver)
   → TrainingEngine.select_best_checkpoint    (the ONE M52 selector:
     minimum persisted validation_loss over the authoritative M3 listing,
     canonical (step, created_at) tie-break)
   → concrete checkpoint B pinned (resolved_checkpoint_id)
   → _execute_stage converts to a PURE M4 config (checkpoint_id=B)
   → verified_state_hash + resolve_evaluation (M5/M6 evidence path)
   → M4 EvaluationEngine (deterministic, concrete state only)
```

There is exactly ONE best-selection implementation:
`select_best_checkpoint` is defined once (`app/training.py:214`), invoked
by the resolver once per plan (`app/workflows.py:303`) and exposed by the
M52 read-only route — nothing else. `grep` confirms no `find_best` and no
min-loss selection in `evaluation.py`/`workflows.py`. The same single
per-plan selection is shared by M53 state refs, M55 best-resume train
stages and M56 best-evaluation stages in one plan (asserted by the
canonical-loop test: all three declarations pin the identical id).

## §4 — Evaluation/evidence behavior

* **Concrete identity**: the resolved plan's execution path is
  byte-identical in shape to an explicit-checkpoint evaluation — the M4
  record carries `state_kind="checkpoint"`, `checkpoint_id=B`, and
  `state_hash == B`'s persisted `weights_sha256` (verified live in the
  smoke C7). Evaluations of B and C remain distinguishable exactly as
  before (different `state_hash`, different `result_hash`); no second
  evaluation hash was introduced.
* **Evidence reuse**: identity stays (concrete checkpoint, probe) — the
  existing `resolve_evaluation` exact-reuse lookup applies unchanged.
  Verified three ways: (1) re-running the same recipe reuses the same
  `eval_id` (no duplicate); (2) an explicit-checkpoint stage with the
  same probe reuses the best-declared stage's record (cross-form reuse —
  declaring 'best' never duplicates evidence); (3) live: the loop's
  stage 4 `evaluate(best)` REUSED stage 2's record (evaluations +1 for
  the whole 4-stage run, and the server log shows exactly ONE
  `forge.evaluation` line).
* **Immutable records**: the declarative recipe manifest is never
  rewritten (byte-identical after execution, live-verified); the
  workflow record's plan pins the concrete id; `plan_hash` covers the
  full plan dump, so different resolutions produce different execution
  identities while old records stay byte-stable after the best changes
  (test-verified with a fixture edit in throwaway storage).

## §5 — M51/M55/M14 integration

* **M51 preflight parity**: the recipe plan endpoint resolves through the
  SAME `_resolve` → `resolve_best_state_refs` path. Live: preflight
  pinned the live argmin onto all three best declarations; preflight ×2
  byte-identical with ZERO writes; the executed record's plan is
  byte-identical to the preflight plan (same `plan_hash`, same
  `recipe_hash`). No cross-time lock: the post-run preflight re-resolved
  the NEW argmin (`0deba72e350b`) — a later execution legitimately sees
  a different best; the executed record stays authoritative.
* **M55 composition**: the canonical finite improvement loop —
  `train → evaluate(best) → train(resume_from_best) → evaluate(best)` —
  is now expressible end-to-end without hard-coded checkpoint ids, as
  ONE registered immutable recipe. Live execution completed: stage 3
  trained FROM the pinned best (provenance `initial_checkpoint_id` ==
  the selection, pure M54 config, checkpoints descending from it) and
  produced `0deba72e350b @ 6.369091` — a genuinely better checkpoint
  than the selected `ed8dbd1b3578 @ 6.400210`; the argmin legitimately
  moved and the next execution's preflight picks it up.
* **M14 composite recipes**: parent → child(best-eval) expands
  deterministically with qualified stage ids (`leg.child_ev`), ONE
  selection pins both stages, ONE workflow record (no nested records),
  ONE evaluation (the second stage reuses the first's evidence), recipe
  manifests never mutated (test-verified).
* **Stage-order semantics (§11, documented)**: the WorkflowEngine
  resolves the ENTIRE plan ONCE, BEFORE execution — the M53 architecture,
  unchanged by M56. All `best` references in one plan (evaluate, train,
  compare/gate/suite refs) share the SAME plan-start selection.
  Therefore stage 4's `evaluate(best)` measures the plan-start best B,
  NOT stage 3's in-run output (in-run outputs are reachable only via the
  explicit `checkpoint_from_stage`, preserved unchanged). Each NEW
  execution re-resolves — the improvement loop advances across
  executions, never by hidden intra-run state discovery.

## §6 — Tests and verification

* Baseline (post-recovery, pre-edit): **579 passed** (all-dots, exit 0),
  OpenAPI 84, compileall + pyflakes clean, worktree clean at `3606de9`.
* New M56 tests: **5** (`test_m56_schema_matrix_and_direct_api_rejection`,
  `..._resolution_evidence_immutability`, `..._composite_and_zero_checkpoint_semantics`,
  `..._canonical_loop_resolution_timing`, `test_m56_api_registration_to_execution`)
  covering the full §16 matrix: schema (current/checkpoint/best/from_stage
  validity + all contradictory-selector rejections), resolver (M53 path,
  exact M52 match, one selection per plan, zero-checkpoint → established
  404 with nothing persisted), evaluation (concrete checkpoint consumed,
  `state_hash == weights_sha256`), evidence (reuse across re-runs,
  cross-form, within-plan; no duplicates), immutability (declarative
  manifest, pinned record, byte-stable old record after a best change,
  pre-pinned ids respected), hashing (different resolutions → different
  `plan_hash`; repeated resolution deterministic), M51 (preflight ==
  executed plan), M14 (composite), M55 integration (the canonical loop,
  plan-start timing, re-resolution across executions), OpenAPI (84,
  both fields in `WorkflowEvaluationStage`).
* Final tests: **584 passed × 3 consecutive full-suite runs** (exit 0,
  zero failures; e.g. 209.6s on the final worktree state). No M1–M55
  test regressed (579 pre-existing all green).
* Static checks: `compileall` and `pyflakes` clean over `app`, `tests`
  and the smoke.
* OpenAPI: **84 paths unchanged** (verified standalone, in-test and
  live); both fields present in the `WorkflowEvaluationStage` schema.
* Live smoke: executed **exactly once** — 27 checks, 25 passed in the
  run; 2 were over-tight FACT expectations (not implementation
  properties) and were corrected with read-only post-hoc verification
  (M55 precedent, no re-execution): A2 assumed best≠latest at plan start
  (the rebuilt bootstrap coincidentally made the last run's final
  checkpoint also the argmin — recorded honestly; the post-run state
  DOES separate them and the re-resolution follows the argmin
  `0deba72e350b`, never the latest `86687445867e`), and C7 demanded
  perplexity == exp(loss) within 1e-6 ABSOLUTE (the M4 record rounds
  `loss_nats` while `perplexity` is computed from the unrounded loss — a
  3.8e-7 relative difference; corrected to a relative tolerance). All
  post-hoc checks PASS; effective certification 27/27.
* Important invariants held: no new routes; one best resolver; recipe
  manifests declarative forever; records immutable and pinned; no
  weights copies; no mutable best pointer; no rollback; no automatic
  loops; `resume_from_best`/`checkpoint_from_*` defaults preserve every
  existing request.

## §7 — Storage/artifact audit

**Production provenance (honest)**: the historical production root was
lost with the sandbox; it was REBUILT from scratch through the public API
before the smoke — dataset `58e10a1d3c9b` (300 unique records), tokenizer
`63dd8dcd3215` (vocab 600), model `31db39e17a20` (155,968 params), two
bootstrap training runs (seeds 11/12; 7 checkpoints). `m56_pre.sha256`
(28 files) was captured AFTER the rebuild and BEFORE the smoke.

```
files before:  28        files after:  39        delta: +11 files
bytes before:  5,105,269 bytes after:  7,644,998 delta: +2,539,729 B
tmp files:     0 before / 0 after
```

Every new file justified (verified by the smoke's exact audit):

* `workflow-recipes/m56-live-best-eval/manifest.json` — the ONE new
  declarative recipe (registered once);
* `models/31db39e17a20/workflows/workflow-06bd7f819080/manifest.json` —
  the ONE workflow record of the single execution;
* 8 checkpoint files — 2 checkpoints × (manifest + weights) per train
  stage (`0e2c32af3ed3`, `0deba72e350b` from tr1; `80597a13c3b2`,
  `86687445867e` from tr2);
* `models/31db39e17a20/evaluations/eval-ec6e8eacccb7/manifest.json` —
  the ONE evaluation (stage 4 reused it; no duplicate).

Modified (3, existing semantics): the model manifest (provenance +
latest publication), `weights.pt` + `weights.sha256` (normal keep-best
publication). The other 25 pre-existing files are byte-identical. No
weights were copied; the selected checkpoint `ed8dbd1b3578` is
byte-identical (manifest + weights) — referenced, never duplicated; no
`best-evaluation-state.pt` or any equivalent exists; no best-pointer
file exists. Zero `.tmp` leftovers. Recipe manifest byte-identical after
execution.

## §8 — Limitations

* **Resolution timing**: best resolves ONCE per plan at plan start (the
  unchanged M53 architecture). Multiple `best` references within one
  plan intentionally share the same selection — the canonical loop's
  stage 4 measures the plan-start best, not stage 3's in-run output
  (in-run outputs require the explicit `checkpoint_from_stage`).
  Re-resolution happens on the NEXT execution; there is no cross-time
  lock, and no intra-run re-selection. M56 did not change this timing.
* **Workflow-only vs direct API**: `checkpoint_from_best` exists only on
  the workflow evaluation stage; the direct M4 route structurally cannot
  express 'best' (unknown field → 422). Direct runs name the checkpoint
  explicitly (`GET /checkpoints/best` answers it).
* **Evidence reuse**: reuse is exact-identity (concrete checkpoint +
  identical probe); a different probe on the same checkpoint creates a
  new evaluation by design. Declaring 'best' never duplicates evidence,
  but it also never merges different probes.
* **Live-verification provenance**: production was rebuilt (bootstrap
  coincidence best==latest at plan start, recorded honestly); the
  drift-discrimination is certified post-run, where the re-resolution
  pin follows the argmin and not the published latest. Two smoke
  expectations were corrected post-hoc (read-only) as detailed in §6.
* **What M56 does NOT automate**: no automatic loops, no retries, no
  rollback, no autonomous improvement, no accept/reject automation, no
  mutable best pointer, no new training methods. The improvement loop
  remains FINITE, EXPLICIT and user-driven: a declarative recipe the
  user re-runs. The system does NOT now have an autonomous improvement
  loop — it has the declarative vocabulary to express one iteration.

## §9 — NEXT MILESTONE

Inspection of the completed architecture: the canonical control loop is
now `train → evaluate(best) → train(resume_from_best) → evaluate(best)`
— fully declarative — but its ACCEPT/REJECT step, the gate, cannot yet
declare its BASELINE as best. `WorkflowGateStage.candidate` is a
`StageStateRef` (best/from_stage/checkpoint since M53), but the gate
`GatePolicy` baseline supports only `checkpoint` (explicit id) /
`current` / `evaluation_result_hash` / `minimum_loss`. Judging "did the
resume-from-best run improve over the best?" therefore still requires a
hard-coded baseline checkpoint id — the exact manual drift M55/M56
eliminated everywhere else. **M57 = declarative best BASELINE for gate
policies** is the smallest high-value step toward
`train → evaluate → compare/gate → select → resume → evaluate →
accept/reject → repeat`, completing the loop's declarativeness before
any repetition/looping milestone is considered.

---

### M57 — BEST BASELINE FOR GATE POLICIES (copy-ready prompt)

You are continuing development of **AI Model Forge**.

Implement this milestone incrementally on top of the certified **M56**
state.

## Core objective

Close the accept/reject gap in the improvement-control loop. M53 gave
gate stages a `candidate: StageStateRef` with `best` support; M55 gave
train stages `resume_from_best`; M56 gave evaluation stages
`checkpoint_from_best`. The gate policy's BASELINE, however, still
requires an explicit checkpoint id (`baseline_type="checkpoint"` +
`baseline_checkpoint_id`), `current`, an evaluation result hash, or an
absolute `minimum_loss`. M57 should allow a workflow gate stage to
declare "judge the candidate against the M52 best checkpoint":

```text
gate(candidate=from_stage, baseline=best)
```

The intended canonical recipe becomes fully declarative end-to-end:

```text
TRAIN
  ↓
TRAIN FROM BEST            (M55)
  ↓
GATE(candidate=from_stage, baseline=best)   (M57)
  ↓ on_pass / on_fail (existing M6/M7 branching)
```

Do not create a new gate engine. Do not duplicate M52 best-selection
logic. Do not modify M6 decision semantics. Do not introduce automatic
loops or automatic rollback.

## 1. INSPECT BEFORE EDITING

First run `git status`, `git log --oneline -10`, `pytest -q`.

Expected baseline: **584 tests, OpenAPI 84, worktree clean, M56
implementation present** (unless an environment-recovery note in the
latest final report says otherwise — then re-verify the baseline and
rebuild the venv per the established runbook before editing).

Inspect: `app/schemas.py` (`GatePolicy`, `GateBaselineType`,
`WorkflowGateStage`), `app/gates.py` (M6 engine: baseline resolution,
decision semantics, direct-run preflight), `app/workflows.py`
(`resolve_best_state_refs`, the GATE branch of `_execute_stage`,
`_state_of`), `app/recipes.py` (M51 `_resolve`, M14 `_qualify_stage`),
`app/api.py` (gates routes, landing), `tests/test_gates*.py`,
`tests/test_workflows.py`, `tests/test_workflow_recipes.py` (M53/M55/M56
tests and helpers). Do not assume field names.

## 2. TRACE THE GATE BASELINE PATH

Trace: recipe → `WorkflowStage(gate)` → `WorkflowGateStage`
(`policy` inline XOR `policy_id` registry) → `GateRequest` →
`GateEngine.run` → baseline resolution → (evaluation reuse / M5
comparison) → `GateDecision`. Document exactly how each baseline type
resolves today, how `policy_id` stages materialize their policy at
execution, and where a declarative best baseline could be pinned without
rewriting any immutable registry policy.

## 3. ONE BEST RESOLVER (mandatory)

The gate baseline must resolve through the SAME M53
`resolve_best_state_refs` → `select_best_checkpoint` path, ONCE per
plan, sharing the single per-plan selection with any M53 refs, M55
train-best and M56 evaluate-best declarations in the same plan. The gate
engine must receive a PURE M6 policy (explicit
`baseline_checkpoint_id`); it must never query "best" itself. There must
remain exactly one definition of "best checkpoint".

## 4. SCHEMA DESIGN (follow the M55/M56 pattern)

Prefer `GatePolicy.baseline_from_best: bool = False` +
`resolved_baseline_checkpoint_id: Optional[str]` (resolver pin, only
valid with best), XOR with `baseline_checkpoint_id` and contradictory
with the other baseline forms (`baseline_type` stays `"checkpoint"`
when best is declared — the best IS a checkpoint baseline; decide and
test the exact validator rules in the existing style). Default false
preserves every existing request. DIRECT gate requests
(`POST /gates/evaluate`) declaring best must be rejected (422, M55
pattern) — direct runs name the baseline explicitly.

**Registry policies (`policy_id`)**: a registered policy is immutable
and model-agnostic, and the plan carries only the id — decide
explicitly whether (a) best-baseline is restricted to INLINE policies
(recommended minimal step: registration of a best-baseline policy into
the registry → 422, documented), or (b) registry policies may declare
best and the resolver materializes them inline into the resolved plan
(bigger change: plan content, `plan_hash`, provenance). Choose one,
test it, and document why.

## 5. IMMUTABILITY + EVIDENCE

The workflow record's plan pins the concrete baseline id
(`resolved_baseline_checkpoint_id`); different resolutions → different
`plan_hash`; old records byte-stable after the best changes; recipe
manifests never rewritten. Gate evidence (M4 evaluations of candidate
and baseline, M5 comparison when a state baseline exists) reuses exact
existing records through the existing lookups — declaring best must not
duplicate evidence. Decision semantics (`passed`/`failed`, tolerance,
`max_regression_delta`, `minimum_loss`) are M6's, unchanged.

## 6. TEST MATRIX

Focused M57 tests for: schema (all baseline forms still valid; best
valid; every contradictory combination rejected at registration and at
the API); resolver (one selection per plan shared with M55/M56
declarations; exact M52 match; zero-checkpoint → established 404,
nothing persisted); execution (the gate engine received a PURE explicit
checkpoint baseline — prove via the persisted decision's embedded
policy); direct-run rejection; registry-policy rule; evidence reuse;
immutability/hashing; M51 preflight parity; M14 composite; the full
canonical recipe
`train → train(resume_from_best) → gate(candidate=from_stage,
baseline=best)` with both `on_pass` completion and a `failed`-gate stop
path (existing M7 semantics); regression of all M1–M56 tests.

## 7. LIVE SMOKE (exactly once)

After the full suite passes, inspect production (read the latest final
report's §7 facts first). Prefer ONE minimal state-mutating execution:
register the canonical gate recipe (small steps, existing model) and
execute it exactly once, verifying: the pin equals the live-computed
argmin (M52 endpoint agreement); the decision's embedded policy carries
the concrete baseline; evidence reuse; the record pins; the recipe
manifest and the selected checkpoint byte-identical; no rollback; no
best pointer; no duplicate gate engine. Capture the pre-inventory
(`m57_pre.sha256`) BEFORE the smoke and audit every new file after. If
production must be rebuilt (environment loss), rebuild honestly through
the public API first, document it, and treat the rebuilt state as the
baseline. Execute the smoke ONCE; correct over-tight FACT expectations
only with read-only post-hoc verification (M55/M56 precedent).

## 8. DISCIPLINE

Inspect first; test first; minimal implementation; minimum files;
minimum storage (no weights copies — references only); deterministic
behavior; immutable history; reuse existing infrastructure; one
executor; one best selector; no new routes unless inspection proves
necessity (M55/M56 added none); no redundant APIs; no scope expansion
(no automatic loops/retries/rollback/HPO/LoRA/DPO/deployment/DB
migration/frontend). Static checks (compileall, pyflakes) and the
OpenAPI count (expected 84) must be reported. Do not plan byte-exact
`weights.pt` restores around training. Count `== 1` splice anchors
before editing. Unique recipe/test ids. Run the full suite to green
twice before the smoke.

## 9. REQUIRED FINAL REPORT

Return exactly 9 sections: (1) implementation summary; (2) gate baseline
semantics (all forms + XOR rules + the registry-policy decision);
(3) M52/M53 integration (one resolver, pure-M6 conversion);
(4) decision/evidence behavior; (5) M51/M55/M56/M14 integration incl.
the canonical gate recipe; (6) tests and verification (baseline 584,
final counts ×2, OpenAPI, statics, smoke); (7) storage/artifact audit
(files/bytes before→after, every new file justified, untouched files
byte-identical); (8) limitations (resolution timing, workflow-only vs
direct, what M57 does not automate — no autonomous loop claims);
(9) NEXT MILESTONE: inspect the actual architecture and select the
smallest high-value step (candidates to evaluate: declarative best
baseline for M5 comparison REQUEST baselines if a gap remains; bounded
finite repetition (`repeat: N` unrolling) ONLY if the core loop is
robust and it can stay deterministic/immutable; accept-path semantics
such as publish-on-pass vs stop-on-fail recipes). Provide a complete
ready-to-copy prompt preserving: inspect first; test first; minimal
implementation; minimum files; minimum storage; deterministic behavior;
immutable history; reuse existing infrastructure; one executor; one
best selector; no redundant APIs; no premature scope expansion.
