# M55 Final Report — declarative best-resume for workflow train stages

Milestone date: 2026-09-09. Branch `arena/01a071e9-code-forge`.

## §1 — M55 implementation summary

Baseline verified before any edit: `HEAD 4e11b39` == remote (certified
M54), worktree clean, production **113 files / 5,289,253 B / 0 `.tmp`**
(the 3 `m54_pre` deltas = exactly M54's justified modifications), full
suite **574 passed @ 155.78s**, OpenAPI 84.

Changes (additive; 4 source files + tests + README + smoke):

- `app/schemas.py` — `TrainingConfig` gains `resume_from_best: bool =
  False` (the DECLARATIVE workflow option) and
  `resolved_resume_checkpoint_id: Optional[str]` (the resolver's pin,
  valid only with `resume_from_best`), with an XOR validator extending
  the existing `_epochs_xor_steps` model validator: best + explicit id
  = contradiction (rejected, never a silent preference); a resolved pin
  without best = rejected.
- `app/workflows.py` — the M53 resolver
  (`resolve_best_state_refs`) now also resolves declarative best-resume
  TRAIN stages: the SAME single per-plan M52 selection is pinned onto
  the stage's config (`resolved_resume_checkpoint_id`), flowing into
  the resolved plan → immutable record → `plan_hash` exactly like M53
  state refs. The TRAIN execution branch converts the pinned form into
  a **PURE M54 config** (`resume_from_checkpoint_id = <concrete id>`,
  best flag and pin stripped) before invoking the training engine.
- `app/training.py` — `preflight()` rejects a DIRECT run declaring
  `resume_from_best` (ValueError → 422): best is a workflow-stage
  declaration resolved by the workflow engine; direct runs name the
  checkpoint explicitly. The M54 resume path itself is untouched.
- `app/api.py` — documentation only (landing bullet, section comment,
  route docstring). **NO new route.**

## §2 — Declarative semantics

- `resume_from_best = false` (default) — existing behavior, exactly
  (all 574 pre-M55 tests green on the final code).
- `resume_from_best = true` — the workflow resolver resolves the M52
  best checkpoint (minimum persisted `validation_loss`, canonical
  `(step, created_at)` tie-break, non-finite values excluded) ONCE per
  plan at execution/M51-preflight time and pins the concrete id; the
  training engine then runs a pure M54 explicit resume from it.
- `resume_from_checkpoint_id = "ABC"` — M54 semantics unchanged:
  initialize from that concrete checkpoint (direct API or workflow).
- **Both supplied** — rejected at schema level (422): the selection
  decides OR the id is named, never both, never a silent preference.
  `resolved_resume_checkpoint_id` without `resume_from_best` is also
  rejected (it is the resolver's pin, not a user input).
- A DIRECT `POST /training/run` with `resume_from_best` → 422 (name
  the checkpoint explicitly; `GET /checkpoints/best` answers it).

## §3 — M53/M54 integration

The chain, with ONE selection implementation and ONE executor:

```
recipe: train(resume_from_best=true)
    ↓ RecipeEngine._resolve (the M51 path) → WorkflowPlan
    ↓ WorkflowEngine.resolve_best_state_refs (the M53 resolver)
    ↓ TrainingEngine.select_best_checkpoint (the M52 selector behind
      GET /checkpoints/best — no second definition of "best")
    ↓ concrete checkpoint id pinned: resolved_resume_checkpoint_id
    ↓ TRAIN stage execution: PURE M54 config
      (resume_from_checkpoint_id = <id>, best stripped)
    ↓ TrainingEngine.run — the untouched M54 verified-weight resume
```

The training layer never queries "best" — it receives an explicit
checkpoint id (separation of concerns, exactly as specified). Proven
live: the training provenance of the executed run carries
`resume_from_checkpoint_id = 0511de4c7372` with `resume_from_best:
false` (the pure M54 form), while the workflow record's plan carries
the declarative + pinned form.

## §4 — Workflow/recipe immutability

The registered recipe manifest stays DECLARATIVE forever
(`resume_from_best: true`, nothing pinned — asserted byte-identical
after execution). The immutable workflow record's plan pins the
concrete id (`resume_from_best: true` +
`resolved_resume_checkpoint_id: <id>`), so the record unambiguously
shows `best → concrete id`, and `plan_hash` (over the resolved plan)
differs between executions that resolved different checkpoints —
proven in tests (run 1 pins A; after a better checkpoint exists, run 2
pins B with a different `plan_hash`; the old record is byte-stable and
still pins A; a third run re-pins B with `plan_hash` equal to run 2's).
No mutable `current_best` pointer exists anywhere; recipe identity
(`config_hash`) remains the hash of the declarative stages. The
training provenance independently records the concrete start through
the EXISTING M54 fields (`initial_checkpoint_id`,
`parent_checkpoint_id`, full config JSON) and the run's checkpoints
descend from it — no new provenance fields were needed.

## §5 — M51/composite workflow behavior

- **Preflight parity**: `GET .../recipes/{r}/plan` resolves
  best-resume through the SAME `_resolve → resolve_best_state_refs`
  path executions use; the preflight plan is byte-identical to the
  executed record's plan (proven in tests and live, including ×2
  byte-identical preflight repeats with zero writes). No cross-time
  lock is claimed: same registry state + same resolver + same
  checkpoints = same resolution; the executed record is authoritative.
- **Composite (M14)**: a parent recipe calling a child best-resume
  recipe expands deterministically (qualified ids like
  `leg.child_tr`), resolves AFTER expansion through the ONE resolver,
  and both stages pin the SAME single per-plan selection into the ONE
  final record (no nested records, no second executor); tested with
  preflight == execution parity.
- **Zero checkpoints**: a best-resume recipe bound to a model with no
  selectable checkpoints hits the established 404
  ("no selectable checkpoints") at BOTH preflight and run, with
  nothing persisted; unknown model → the established 404. Note: train
  stages are model-pinned (M12 binding semantics), so a best-resume
  recipe resolves only against its bound model.
- **Resolution timing (documented rule)**: best resolves ONCE at PLAN
  START — a best-resume stage sees the checkpoints existing when the
  workflow starts, NOT earlier stages' in-run outputs (those are
  referenced explicitly via `from_stage`, as the multi-stage test's
  evaluate stage does). No hidden state discovery.

## §6 — Tests and verification

- **Before**: 574 passed @ 155.78s. **After**: **579 passed ×2**
  (151.61s / 151.47s) on final code; `compileall` OK; `pyflakes` clean
  on every touched file.
- **New tests (5)**: (1) schema matrix (default false; declarative
  true; pinned form; both-set rejected; pin-without-best rejected) +
  direct-run rejection with nothing persisted; (2) resolution →
  provenance → immutability (preflight pins the M52 argmin; execution
  record pins it; provenance carries the pure M54 form; lineage
  descends; recipe manifest + old record byte-stable across a changed
  best; different resolutions → different `plan_hash`; determinism);
  (3) composite + zero-checkpoint + unknown-model semantics; (4) the
  §13 multi-stage recipe (train → evaluate(from_stage) →
  train(resume_from_best)) proving plan-start resolution timing and the
  explicit `from_stage` mechanism for in-run outputs; (5) API
  end-to-end (XOR 422 at registration; preflight 404 on a
  zero-checkpoint model; preflight pins the locally computed argmin ×2
  byte-identical; execution parity; provenance; direct-run 422;
  OpenAPI 84 unchanged with both fields).
- **OpenAPI**: **84 → 84** — no new path; schema fields only.
- **Live smoke** (`smoke_m55_live.py`, port 8776, production FORGE_ROOT,
  executed EXACTLY ONCE): **23/24 checks PASS in the executed run; the
  single failure was an over-narrow allowlist in the smoke script
  itself** — D3 omitted the dashboard `workflows` section from the
  allowed diff, but this milestone's execution creates a WORKFLOW
  record (the recipe run), so `workflows.counts.completed (15 → 16)`
  and `workflows.records (19 → 20)` are exactly the expected change
  (D4, asserting 20 workflows, passed). The allowlist was corrected and
  the fact verified post-hoc against the persisted state: dashboard
  workflow counts = {completed 16, failed 3, stopped 1}, records ==
  the live 20-record listing, the m55 run present and completed,
  suite-runs/evaluations/comparisons/gates unchanged. The certified
  execution is workflow **`b7dfc1a51635`** / training run
  **`e7ebff459386`** (single execution; engine log: "baseline val
  6.2106, best val 6.1466, accepted=True, rollback=None").
- **The headline live result**: the production best-resume recipe
  resolved best → `0511de4c7372` @ 6.210553 (≠ latest `b985e7c679ca` —
  exactly the manual drift M55 removes), trained FROM it, and produced
  `3f174923fae3` @ **6.146612 — a genuinely better checkpoint**; the
  M52 selection and the M53 best-ref preflight both discovered the
  moved argmin live (D1/D2). The improvement loop demonstrably improved
  the same model's weights on production, fully declaratively.

## §7 — Storage/artifact audit

```
files before : 113          bytes before : 5,289,253
files after  : 119          bytes after  : 6,556,715
delta        : +6 files / +1,267,462 B / 0 .tmp  (+3 justified modifications)
```

New files (the ONE recipe registration + the ONE execution):

- `workflow-recipes/m55-live-best-resume/manifest.json` — the minimal
  declarative recipe (single best-resume train stage);
- `models/4a0a871886ef/workflows/workflow-b7dfc1a51635/manifest.json` —
  the execution's immutable workflow record (pins best → `0511de4c7372`);
- `models/4a0a871886ef/checkpoints/039ce4322d2e/{manifest,weights}.pt` —
  the run's first checkpoint (step 2, val 6.1558, accept), parent
  `0511de4c7372`;
- `models/4a0a871886ef/checkpoints/3f174923fae3/{manifest,weights}.pt` —
  the run's second checkpoint (step 4, val 6.1466, accept), the new
  best, parent `039ce4322d2e`.

Modified (each the EXISTING training-completion semantics):
`models/4a0a871886ef/manifest.json` (provenance append,
`latest_checkpoint`/`best_checkpoint` → `3f174923fae3`),
`weights.pt` + `weights.sha256` (keep-best published the improving
state). The selected source checkpoint `0511de4c7372` is byte-identical
(manifest + weights — referenced, never copied); every other
pre-existing file is byte-identical to `m55_pre.sha256`. No
best-pointer file, no weights copies, no caches.

## §8 — Limitations

- **Finite declarative recipes only.** M55 makes the loop EXPRESSIBLE
  — `train → evaluate → train from best → evaluate → gate` is one
  registered, immutable, re-runnable recipe — but there is NO
  automatic repetition: no `while improvement`, no `repeat_until_best`,
  no `auto_improve()`. Each execution is an explicit request;
  "repeat" = executing the recipe again (each run re-resolving the then-
  current best, as the live run demonstrated).
- **Plan-start resolution.** A best-resume stage resolves the best
  among checkpoints existing when the workflow STARTS; it cannot see
  earlier stages' in-run outputs (use `from_stage` / explicit ids for
  those — no hidden state discovery). A preflight does not lock a
  later execution across intervening training.
- **M54's limitation stands**: resume restores MODEL WEIGHTS only —
  no optimizer, scheduler, gradient-scaler or data-cursor state exists
  in M3 and none is restored; a best-resume run starts a fresh AdamW
  (documented on the field, README, API docstring).
- `resume_from_best` is workflow-only: direct training requests must
  name the checkpoint (422 otherwise). "Best" remains the single M52
  criterion — no claim of overall model quality.
- Evaluation stages still express state through
  `EvaluationConfig.checkpoint_id` / `checkpoint_from_stage` (no `best`
  there — see §9).

## §9 — NEXT MILESTONE

The loop is now declaratively complete for suite/compare/gate/train
stages. Evidence gap found by inspecting the loop's "measure" step:
the M53 `best` reference covers suite-run states, comparison sides and
gate candidates — but NOT evaluation stages, whose state is the
config-level `EvaluationConfig.checkpoint_id` / `checkpoint_from_stage`
(documented as a limitation in M53 §8). Consequence: a recipe can
train from best and gate best, but cannot declaratively say
"evaluate the current best" — after a non-improving best-resume run,
best is still the OLD checkpoint, and measuring "what is now best"
(the honest accept/reject evidence) requires hand-pinning the id.
Closing this one gap makes the canonical improvement recipe fully
declarative in every stage kind — the smallest high-value step
(rejected alternatives: automatic repetition — forbidden by philosophy
and by every milestone's scope; SFT differentiation — heavy
dataset-format semantics, deferred again; optimizer-state resume — a
real training-semantics change, not the smallest step).

Copy-ready M56 prompt:

```
# M56 — BEST STATE REFERENCES FOR EVALUATION STAGES

Extend AI Model Forge from the certified M55 state with exactly ONE
additive capability: a workflow EVALUATION stage may declaratively
evaluate the model's best checkpoint (the M52 selection), completing
the M53 state-reference coverage (suite-run/compare/gate already
support "best"; evaluation stages currently only support a literal
checkpoint id or checkpoint_from_stage).

1. Inspect first: app/schemas.py (WorkflowEvaluationStage,
   EvaluationConfig, StageStateRef's M53 "best" + resolved pin,
   validate_plan_stages), app/workflows.py
   (resolve_best_state_refs — the ONE resolver incl. the M55 train
   branch; _execute_stage EVALUATE branch), app/recipes.py (_resolve),
   app/api.py, tests/test_workflow_recipes.py (M53/M55 tests).
   Run the FULL suite BEFORE any edit (579 expected) and confirm
   OpenAPI 84. Capture m56_pre.sha256 (119 files) BEFORE any live
   testing.
2. Add WorkflowEvaluationStage.checkpoint_from_best: bool = False,
   validated XOR with config.checkpoint_id and
   checkpoint_from_stage (exactly one state source; contradictions
   422). Default false preserves every existing request exactly.
3. ONE resolver, reused: extend resolve_best_state_refs so an
   unresolved checkpoint_from_best stage is satisfied by the SAME
   single per-plan M52 selection, pinned into the RESOLVED plan by
   setting the stage's config.checkpoint_id to the concrete id (the
   record then shows checkpoint_from_best=true + the concrete id —
   unambiguous, plan_hash-differentiated across resolutions, exactly
   the M53/M55 pattern). Execution proceeds through the UNCHANGED
   evaluate path (the pinned config is a normal literal-id config);
   preflight parity is automatic (same resolver). A zero-checkpoint
   model -> the established 404 BEFORE anything executes or persists.
4. Tests (~4-5): schema XOR matrix; resolver pins the M52 argmin and
   the evaluation record carries the concrete checkpoint; preflight ==
   execution plan parity; immutability (a later best change leaves the
   old record pinned, new runs resolve the new best with a different
   plan_hash); composite recipe with a nested evaluate(best) stage;
   the canonical loop recipe train -> evaluate(from_stage) ->
   train(resume_from_best) -> evaluate(best) -> gate(candidate=best)
   executing end-to-end with every stage's concrete state visible in
   the record. Update README counts.
5. Gates in order: focused tests -> FULL suite x2 -> compileall ->
   pyflakes -> OpenAPI check (path count UNCHANGED, 84) -> live smoke
   (read-only verification preferred: production best is
   DISCOVERED live; a best-evaluation recipe may be registered and
   EXECUTED ONCE if a real evaluation artifact is needed — justify
   every new file; prefer evidence-reuse surfaces like an existing
   suite) -> final SHA audit. Never loop a state-mutating smoke; never
   certify intermediate code.
6. Final report with EXACTLY 9 sections; §9 selects the next
   milestone from evidence (candidates to weigh: SFT method
   differentiation, optimizer-state resume, dataset-version tooling —
   or honest exhaustion if the declarative loop is complete). Include
   a complete ready-to-copy prompt if justified. Completion standard:
   never a second selection implementation, never a moving reference
   in a persisted record, never automatic repetition, never certify
   from partial runs; report actual counts honestly.
```
