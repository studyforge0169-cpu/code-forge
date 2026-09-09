# M54 Final Report — explicit training resume point

Milestone date: 2026-09-09. Branch `arena/01a071e9-code-forge`.

## §1 — M54 implementation summary

Baseline verified before any edit: `HEAD 2c53bca` == remote (certified M53),
worktree clean, production **109 files / 4,026,085 B / 0 `.tmp`**
(`m53_pre.sha256` cross-checked: the 7 M53 artifacts present, the rest
intact), OpenAPI 84, full suite **570 passed @ 159.17s**. No recovery was
needed this time — the environment was intact.

Changes (additive; 4 source files + tests + README + smoke):

- `app/schemas.py` — `TrainingConfig.resume_from_checkpoint_id:
  Optional[str]` (default `None` = exactly today's behavior), placed next
  to the other artifact references, with the semantics and the
  model-weight-only limitation documented on the field.
- `app/training.py` — (1) `preflight()` validates an explicit resume
  point BEFORE anything is written, through the EXISTING
  `verify_checkpoint` (model-scoped lookup → unknown/foreign checkpoint
  = `FileNotFoundError`; unreadable or content-hash-mismatched weights =
  `RuntimeError`) — no second loader; (2) `run()` initializes the model
  from the resume checkpoint's VERIFIED weights instead of the published
  current weights when the field is set — a pure per-run initialization
  choice (no `write_weights`, no `latest_checkpoint` change to prepare
  the run); (3) the run's lineage start becomes the resume checkpoint:
  `initial_checkpoint_id = parent_checkpoint_id =
  resume_from_checkpoint_id or model_record.latest_checkpoint`.
- `app/api.py` — documentation only (route docstring, endpoint `<li>`,
  section comment). **NO new route.**
- Tests: 3 engine tests + 1 API test (details §6). README section +
  counts.

No new schema for provenance was needed: `RunProvenance` already carries
`initial_checkpoint_id`/`parent_checkpoint_id` and the full config JSON.
No second executor, no checkpoint copying, no optimizer-state work.

## §2 — Resume semantics

- **`resume_from_checkpoint_id = <id>`** — initialize THIS ONE run from
  that immutable checkpoint's verified model weights. The checkpoint is
  referenced, never copied; the model's published state (`weights.pt`,
  `latest_checkpoint`) is NOT touched to prepare the run; publication
  happens only through the normal training completion semantics
  (keep-best / final adoption) exactly as for any other run.
- **`rollback(checkpoint)`** — UNCHANGED (M3): verify, then PUBLISH the
  checkpoint as the model's current weights and set `latest_checkpoint`.
  Rollback = publish; resume = initialize one run. Distinct concepts,
  distinct code paths, tested separately.
- **current state** — `resume_from_checkpoint_id = None` (the default):
  the run starts from the model's current published weights, exactly as
  before M54 (all 570 pre-M54 tests green proves behavioral identity).
- **best state** — `GET /checkpoints/best` (M52) answers WHICH checkpoint
  is best; a workflow STATE reference `state_kind: "best"` (M53) resolves
  and pins it for evaluation-style stages. M54 does NOT auto-select: the
  caller names the concrete checkpoint (possibly obtained from M52) —
  composing best → resume is two explicit requests, no automation.

## §3 — Training provenance

The immutable run record answers "which checkpoint did this run start
from?" through EXISTING fields — verified live: `RunProvenance.
initial_checkpoint_id == parent_checkpoint_id == 30a8bc5b82ab` (the
resume checkpoint), `config.resume_from_checkpoint_id == 30a8bc5b82ab`
(the full TrainingConfig JSON is embedded per run), and
`TrainingReport.initial_model_version == 30a8bc5b82ab`. The run's first
new checkpoint descends from the resume point through the existing
lineage chain (`parent_checkpoint_id`), so the checkpoint graph itself
carries the trace — no parallel lineage, no new record type, no mutable
pointer.

## §4 — Baseline / acceptance behavior

M3 evaluates a baseline BEFORE the first update on the LOADED state and
anchors keep-best acceptance to it (`improved = val_loss < best_val`).
Because M54 initializes the model from the resume checkpoint, the
baseline is measured ON THE RESUMED STATE by construction — acceptance
decisions stay mathematically consistent with the actual starting
point, with NO new policy. Live proof: the resume run cloned the probe
configuration of the run that created the resume checkpoint
(dataset/tokenizer/`max_seq_len`/`batch_size`) and its
`baseline_validation_loss` (6.3972859382629395) reproduced the
checkpoint's persisted `validation_loss` (**6.397286**) exactly — the
run demonstrably started from that checkpoint's weights through the
same deterministic evaluation used at its creation (also the §16
equivalence proof: explicit resume loads the same state the existing
restore mechanism provides). No M4 evaluation records are created
(the baseline is M3's internal streaming evaluation, as always).

## §5 — Workflow compatibility

The field flows everywhere `TrainingConfig` flows — including M7/M12/
M14 workflow and recipe `train` stages — with zero schema work beyond
the config field itself. Proven by a dedicated test: an inline workflow
`train` stage with `resume_from_checkpoint_id` executes through the
sole `WorkflowEngine`, and the resulting run's provenance pins the
resume checkpoint. What is deliberately NOT done: a declarative
"resume from best" inside a train stage (the M53 `best` reference is a
STATE reference for evaluation-style stages; wiring it into the
training config would need resolver changes) — deferred as the natural
next milestone (§9), not smuggled into M54.

## §6 — Tests and verification

- **Before**: 570 passed @ 159.17s. **After**: **574 passed ×2**
  (157.11s / 157.45s) on final code; `compileall` OK; `pyflakes` clean
  on every touched file.
- **New tests (4)**: (1) engine — non-destructive resume with full
  provenance (initial/parent/config), baseline parity with the resumed
  checkpoint's persisted loss, existing publication semantics
  (`latest_checkpoint` = the run's last created, never the resume
  point), resume-checkpoint manifest+weights byte-identity, no
  duplication; (2) engine — determinism (two identical resume runs →
  identical per-step losses and baselines) + the error taxonomy
  (unknown 404, foreign-model checkpoint 404, corrupted weights
  `RuntimeError` with the checkpoint restored and resumable after);
  (3) engine — workflow train-stage resume; (4) API — 200 + provenance
  through the existing route, unknown/foreign 404, silently-perturbed
  weights 409 (integrity), unreadable weights 422 (established split,
  both asserted with their detail messages), empty id 422 (schema),
  OpenAPI 84 UNCHANGED with the field in `TrainingConfig`.
- **OpenAPI**: **84 → 84** — no new path (the config schema gained the
  optional field only).
- **Live smoke** (`smoke_m54_live.py`, port 8775, production FORGE_ROOT,
  executed EXACTLY ONCE — §18's no-repeated-mutation rule): **19/21
  checks PASS in the executed run; the 2 failures were over-narrow
  audit expectations in the smoke script itself, not behavior** — (a)
  the dashboard allowlist omitted the `summary` section (persisted
  model-manifest facts: `best_checkpoint`, `latest_checkpoint`,
  `training_run_count`, `updated_at` — a training run legitimately
  updates the model manifest, and the failure output shows the summary
  diff was EXACTLY those four manifest echoes, nothing else); (b) the
  modified-files allowlist omitted `weights.sha256` (the published-
  weights hash sidecar updated by the existing `write_weights`
  semantics). Both expectations were corrected in the script and the
  corrected facts were re-verified post-hoc against the persisted
  state (`m54_pre.sha256` comparison: exactly 4 new checkpoint files;
  modified = exactly {model manifest, weights.pt, weights.sha256};
  dashboard summary == current manifest facts; M52 best and the M53
  best-ref preflight resolving the live argmin over the grown
  5-checkpoint registry). The certified execution is production run
  **`fcdf1c5ac7d7`** (single execution; engine log: "baseline val
  6.3973, best val 6.3610, accepted=True, rollback=None").
- Important invariants verified live: the run started from the
  historical checkpoint `30a8bc5b82ab` (≠ `latest` `0511de4c7372`); the
  baseline reproduced `6.397286` EXACTLY; both new checkpoints descend
  from it; `latest_checkpoint` became the run's last created
  checkpoint (never the resume point — no rollback happened); the
  resume checkpoint's manifest and weights are byte-identical;
  training-only footprint (workflows/suite-runs/evaluations/
  comparisons/gates/samples/recipes all unchanged at 19/16/16/8/11/4/8).

## §7 — Storage/artifact audit

```
files before : 109           bytes before : 4,026,085
files after  : 113           bytes after  : 5,289,253
delta        : +4 files / +1,263,168 B / 0 .tmp  (+3 justified modifications)
```

New files (the ONE real training run `fcdf1c5ac7d7`, its own artifacts):

- `models/4a0a871886ef/checkpoints/2741cd7a8d72/manifest.json` +
  `weights.pt` — the run's first checkpoint (step 2, val 6.3662,
  accept), parent `30a8bc5b82ab`;
- `models/4a0a871886ef/checkpoints/b985e7c679ca/manifest.json` +
  `weights.pt` — the run's second checkpoint (step 4, val 6.3610,
  accept), parent `2741cd7a8d72`, the run's best.

Modified files (each the EXISTING training completion semantics, none a
preparation mutation):

- `models/4a0a871886ef/manifest.json` — provenance append (+1 run),
  `latest_checkpoint` → `b985e7c679ca`, `best_checkpoint` →
  `b985e7c679ca`, `updated_at`;
- `models/4a0a871886ef/weights.pt` — keep-best published the improving
  final state (accepted=True; `final_model_version b985e7c679ca`);
- `models/4a0a871886ef/weights.sha256` — the published-weights hash
  sidecar written together with `weights.pt`.

Source-checkpoint integrity: `30a8bc5b82ab`'s manifest and weights are
byte-identical (asserted before/after). No checkpoint copies, no
temporary published checkpoints, no resume pointers, no caches. Every
other pre-existing file is byte-identical to `m54_pre.sha256`.

## §8 — Limitations

- **Model-WEIGHT resume only.** M3 persists no optimizer state,
  LR-scheduler state, gradient scaler, or data-cursor; M54 restores
  NONE of them. A resumed run starts a FRESH AdamW at the configured
  learning rate and re-walks the deterministic data cycle from the
  beginning — this is checkpoint-initialized continued training, NOT
  exact training-process continuation. No claim of full
  training-state resume is made anywhere (field docstring, README,
  API docstring).
- `"best"` is not automatic: the caller names the concrete checkpoint
  (composing M52's answer is two explicit requests); no auto-selection,
  no mutable best pointer, no improvement loop automation.
- The resume checkpoint must belong to the SAME model (model-scoped
  lookup); foreign checkpoints are 404. Architecture compatibility is
  guaranteed by construction (same model config) and enforced by the
  existing `restore_state` check.
- Baseline/acceptance semantics are UNCHANGED M3 policy — M54 only
  changes WHICH STATE the baseline measures (the actual starting
  state), which is the mathematically consistent interpretation.
- Exact baseline-loss reproduction additionally requires the same
  probe configuration as the checkpoint's creating run (the live smoke
  cloned it); with a different probe the baseline is still measured on
  the resumed state (consistent), just not numerically equal to the
  stored loss.

## §9 — NEXT MILESTONE

The improvement loop is now fully expressible with explicit requests:
train → evaluate → gate/compare → `GET /checkpoints/best` →
`resume_from_checkpoint_id = best` → repeat. Every arrow is manual
today. Candidates weighed: SFT differentiation (heavy dataset-format
semantics — deferred again); dataset-version tooling (low evidence of
need); automatic anything (rejected, standing philosophy); a
rollback-to-best sugar route (rejected in M53 §9 — composition already
works). **The smallest high-value step is the remaining ergonomic gap
inside the loop: a workflow `train` stage cannot DECLARATIVELY say
"resume from best"** — `resume_from_checkpoint_id` needs a concrete id,
and the M53 `best` reference exists only for evaluation-style state
refs. Extending the SAME M53 resolver pattern to the training resume
point (a declarative `resume_from_best: bool`, XOR with the explicit
id, resolved ONCE at execution/preflight time through the SAME M52
selection and pinned as the concrete id in the resolved plan + run
provenance) makes the whole loop expressible in ONE registered,
immutable, re-runnable recipe — still fully explicit (the recipe
declares it), still no auto-improvement (a run happens only when the
recipe is executed by request).

Copy-ready M55 prompt:

```
# M55 — DECLARATIVE BEST-RESUME FOR WORKFLOW TRAINING STAGES

Extend AI Model Forge from the certified M54 state with exactly ONE
additive workflow capability: a workflow/recipe TRAIN stage may declare
that its training run initializes from the model's best checkpoint
(M52 selection) instead of a hand-pinned id.

1. Inspect first: app/schemas.py (TrainingConfig incl. M54
   resume_from_checkpoint_id, StageStateRef, WorkflowStage,
   validate_plan_stages), app/workflows.py
   (resolve_best_state_refs — the M53 resolver; run(); _execute_stage
   TRAIN branch), app/recipes.py (_resolve — the M51 path),
   app/training.py (select_best_checkpoint, run's resume handling),
   app/api.py, tests/test_workflow_recipes.py (M53 tests),
   tests/test_training.py (M54 tests). Run the FULL suite BEFORE any
   edit (574 expected) and confirm OpenAPI 84. Capture m55_pre.sha256
   (113 files) BEFORE any live testing.
2. Add TrainingConfig.resume_from_best: bool = False, validated XOR
   with resume_from_checkpoint_id (both set -> 422; neither -> today's
   behavior). Default False preserves every existing request exactly.
3. ONE resolver, reused: extend WorkflowEngine.resolve_best_state_refs
   (or factor its shared pinning helper) so a train stage with
   resume_from_best is resolved through the SAME
   TrainingEngine.select_best_checkpoint the M52 route and M53 refs
   use — resolved ONCE per plan at execution/M51-preflight time and
   PINNED: the resolved plan's train config carries the CONCRETE
   resume_from_checkpoint_id (resume_from_best stays declaratively
   true; the recipe manifest is never rewritten). No second selection
   implementation, no second executor, no new state kind.
4. Immutability: the resulting run's provenance must record the
   concrete starting checkpoint (M54 already does: initial/
   parent_checkpoint_id + config) — the record must make
   best -> concrete id unambiguous, exactly like M53's
   resolved_checkpoint_id. A zero-checkpoint model at resolution time
   -> the established 404 BEFORE anything executes or persists.
5. Recipe parity: the M51 preflight of a recipe containing a
   best-resume train stage must show the resolved concrete id (same
   resolver as execution; no cross-time lock claimed); composite
   recipes (M14) resolve after deterministic expansion; recipe
   identity (config_hash) stays the DECLARATIVE content
   (resume_from_best=true, no pinned id).
6. Tests (~4-5): schema XOR validation; engine resolution pins the
   M52 argmin and the run's provenance records it (parent chain
   descends from it); a LATER better checkpoint changes the next
   execution's resolution while the old record keeps its pinned id
   (historical immutability, M53-style); M51 preflight ==
   execution plan; zero-checkpoint 404; default-False byte-identical
   behavior. Update README counts.
7. Gates in order: focused tests -> FULL suite x2 -> compileall ->
   pyflakes -> OpenAPI check (path count UNCHANGED, 84) -> live smoke
   (ONE state-mutating execution maximum; read-only preflight
   assertions otherwise) -> final SHA audit. Never loop a
   state-mutating smoke; never certify intermediate code.
8. Live smoke (production FORGE_ROOT): model 4a0a871886ef now owns 5
   checkpoints (argmin DISCOVERED live — it may have moved after
   M54's run); register ONE minimal recipe with a best-resume train
   stage (smallest footprint config), verify the M51 preflight pins
   the live argmin, execute ONCE, verify the run's provenance +
   lineage descend from that concrete checkpoint, the recipe manifest
   stays declarative, and every new/modified storage file is
   justified (expected: the run's checkpoints + model manifest +
   weights publication + one recipe manifest + one workflow record).
9. Final report with EXACTLY 9 sections; §9 selects the next
   milestone from evidence (candidates to weigh: closing the loop
   further — e.g. evaluation stages following the resume stage inside
   one recipe; SFT differentiation; dataset-version tooling — or
   honest exhaustion). Include a complete ready-to-copy prompt if
   justified. Completion standard: never a second selection
   implementation, never auto-execution, never a moving reference in
   a persisted record, never certify from partial runs; report actual
   counts honestly.
```
