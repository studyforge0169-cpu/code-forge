# M53 Final Report — best-checkpoint state references

Milestone date: 2026-09-09. Branch `arena/01a071e9-code-forge`.

## §1 — M53 implementation summary

**Environment recovery first (honest disclosure).** The session environment
was re-provisioned again before M53 began (local HEAD at the branch base
`f86b670`, all M1–M52 work untracked, production data empty, venv gone).
The proven runbook was applied before any M53 work: branch-specific fetch
→ `git reset --hard FETCH_HEAD` (`4e0d3d4`, the M52 tip; branch confirmed
`arena/01a071e9-code-forge`) → 96-file production baseline re-extracted
from the workspace zip (`m51_pre.sha256` 96/96 OK) → venv rebuilt (torch
2.14.0, pydantic 2.13.5, fastapi 0.141.1) → **baseline full suite 566
passed @ 164.77s** → M51 re-certified live against the restored baseline
(58 checks, 57 PASS — the only failure D7 asserts the M51-era OpenAPI
count 83, stale by design against the certified M52 code with 84; all
storage/execution/regression checks green), restoring the 102-file
certified-equivalent state (96 originals byte-identical + 6 new justified
M51 execution artifacts).

**M53 changes (all additive, 4 source files + tests + README + smoke):**

- `app/schemas.py` — `EvalStateKind` gains `BEST` (a WORKFLOW state
  REFERENCE, never persisted on measurement records); `StageStateRef`
  gains `resolved_checkpoint_id: Optional[str]` (the pinned concrete id,
  set only by the resolver, only for `best`) with an extended validator;
  `ComparisonState` explicitly REJECTS `best` (direct comparison/gate/
  suite-run API requests keep `current`/`checkpoint`).
- `app/workflows.py` — `WorkflowEngine.resolve_best_state_refs(plan)`,
  the ONE resolver: pins the M52 selection's concrete id onto every
  unresolved `best` ref (suite-run states, comparison sides, gate
  candidates) and returns a fully re-validated `WorkflowPlan`;
  idempotent; returns plans without `best` refs verbatim.
  `run()` resolves FIRST — the resolved plan is what executes, persists
  and hashes. `_state_of()` maps a pinned `best` ref to a NORMAL literal
  `ComparisonState` (downstream engines never see the moving label).
- `app/recipes.py` — `_resolve()` (the shared M51 path of `run()` and
  `resolve()`) resolves `best` through `self.workflows.
  resolve_best_state_refs` — the same resolver executions use.
- `app/api.py` — documentation only (landing bullet, section comment);
  **NO new route**.

**Incident log (both recovered, fully disclosed).** (1) After the first
clean 49/49 smoke run, a scripting mistake re-invoked the state-mutating
smoke twice more in a loop (runs 2–3 correctly failed their own baseline
guards and each added 6 more execution artifacts). (2) The cleanup of
those extra artifacts used a keep-set that omitted the 16 pre-existing
workflow and 13 pre-existing suite-run directories, deleting 29
historical manifests. Recovery: all this-session M53 artifacts removed →
96-file baseline re-extracted from the zip and verified (`m51_pre`
96/96) → M51 re-certified once more (56/58: the stale D7 OpenAPI count
plus A6, see below) → `m53_pre.sha256` re-captured on the restored
102-file state → the M53 smoke re-run ONCE: **49/49 PASS** (the
certified run). No pre-M53 production byte was lost: the 96 originals
are bit-identical to their committed SHA inventory; the M51 execution
artifacts are a third generation of the same justified kind (new ids,
disclosed here: workflows `df5c358c38c5`/`103068008ae7`/`835c251fc707`,
suite-runs `05e4978a4fe3`/`eb0436254282`/`b5c78d106852`).

**Dashboard hash evolution (verified mechanism).** The second M51
re-certification failed A6 (`dashboard pre-hash`) where the first had
passed — with byte-identical storage, the only variable was the code.
Cause: `DashboardWorkflowSummary.records` embeds full `WorkflowRecord`s
(including plans → stage states), so the additive
`resolved_checkpoint_id: null` field changes the dashboard's canonical
JSON and its `result_hash` (96-file state: `f48557fe…` under M52 code →
`669d375b…` under M53 code). This is the established consequence of
additive embedded-schema evolution — the same happened when M14 added
the `recipe` stage field — not a storage or determinism regression: no
persisted file changed, and the dashboard remains deterministic per
code version (identical storage → identical output, verified live).

## §2 — State-reference semantics

- `current` — unchanged: the plan model's live published `weights.pt`.
- `checkpoint` — unchanged: a literal immutable M3 checkpoint id, or
  `from_stage` resolving to an earlier `train` stage's FINAL checkpoint
  (exactly one of the two, as before).
- `best` (M53, additive) — a DECLARATIVE request resolved at execution
  or M51-preflight time: the workflow engine's single resolver invokes
  the M52 selection (minimum PERSISTED `validation_loss` over the
  authoritative M3 listing, canonical `(step, created_at)` ASCENDING
  tie-break — first among equals — non-finite values never candidates),
  obtains the concrete checkpoint id, and pins it on the ref
  (`resolved_checkpoint_id`); execution then proceeds through the normal
  checkpoint path. `best` must not set `checkpoint_id`/`from_stage`
  (a contradiction → 422 at registration/plan validation);
  `resolved_checkpoint_id` is valid ONLY on `best` refs (rejected on
  `current`/`checkpoint` refs). Direct (non-workflow) state requests —
  `ComparisonRequest`, `GateRequest`, `SuiteRunRequest` — reject `best`
  explicitly: it is a workflow-stage reference, resolved by the workflow
  engine's resolver, never a direct engine input. Evaluation stages
  express state through `EvaluationConfig.checkpoint_id` /
  `checkpoint_from_stage` (a different, config-level mechanism) and are
  unchanged — see §8.

## §3 — M52 integration

There is exactly ONE selection implementation:
`TrainingEngine.select_best_checkpoint(model_id)` (M52), behind the
`ModelForge.select_best_checkpoint` facade used by
`GET /models/{id}/checkpoints/best`. `WorkflowEngine` already owns a
`TrainingEngine` over the same storage, and
`resolve_best_state_refs` calls `self.training.
select_best_checkpoint(plan.model_id)` — the same callable, the same
registry, the same tie-break. No selection code was copied into
`workflows.py` (verified behaviorally in tests: the pinned id equals
the facade's selection on the same registry). One selection runs per
plan (all `best` refs in a plan pin the same id — the registry cannot
change mid-request). A model with no selectable checkpoints fails
resolution with the established `FileNotFoundError` (404) BEFORE
anything executes or persists — no invented state, no fallback.

## §4 — Immutability and reproducibility

The RESOLVED plan (with `resolved_checkpoint_id` pinned) is the object
that executes, persists in the immutable run record, and hashes:

- the record's `plan` shows `state_kind: "best"` AND
  `resolved_checkpoint_id: <concrete id>` — unambiguous
  `best → concrete`;
- `plan_hash` (canonical JSON over the persisted plan) therefore
  differs between executions that resolved different checkpoints —
  proven in tests (run 1 pins A; after a better checkpoint exists, run
  2 pins B and `plan_hash` changes; run 3 re-pins B with the identical
  `plan_hash`);
- the DECLARATIVE identity is untouched: the registered recipe manifest
  keeps `state_kind: "best"` with nothing pinned (byte-identical after
  every execution — asserted live), and `recipe_hash` remains the
  canonical hash of the declarative stages (no resolution baked in);
- historical records never change meaning: the run-1 manifest re-read
  from disk after later runs is byte-stable and still pins A;
- `result_hash` covers the semantic execution (the resolved plan), so
  the concrete resolution participates in the record's identity;
- downstream records (suite runs, evaluations, comparisons, gates)
  receive NORMAL literal checkpoint states — byte-identical in shape to
  explicit-id runs (verified live: the suite-run record shows
  `state_kind: "checkpoint"` + the concrete id).

## §5 — M51 / composite recipe parity

`RecipeEngine._resolve` — the ONE resolution path shared by
`run()` and the M51 `resolve()` preflight — now resolves `best` via
`self.workflows.resolve_best_state_refs` (the sole executor's
resolver). There is no separate `preflight_best_resolution` /
`execution_best_resolution`: the invariant
`same repository state + same resolver + same available checkpoints =
same concrete resolution` is proven in tests and live (the preflight
plan is byte-identical to the executed record's plan; re-preflight
after executions resolves the same id). No cross-time lock is claimed:
a preflight at time T cannot bind a later execution if training happens
in between — the immutable run record is authoritative for what ran.
Composite recipes (M14): `best` refs expand with the deterministic
qualified-id expansion (`leg.child_suite`) and are resolved AFTER
expansion, on the final flat stage list, into the ONE record (single
executor, single selection, composition trace intact — tested with
parent → child(best) and verified live for the direct case). The M51
preflight of a recipe WITHOUT `best` refs is byte-identical to before
(the resolver returns such plans verbatim — asserted live for
`m12-live-suite`).

## §6 — Tests and verification

- **Before**: 566 passed @ 164.77s (post-recovery baseline), OpenAPI 84.
- **New focused tests**: 4 in `tests/test_workflow_recipes.py` —
  (1) schema matrix (best valid; best+checkpoint_id / best+from_stage /
  resolved-on-non-best / direct-ComparisonState-best all rejected;
  current/checkpoint unchanged) + engine resolution parity with the M52
  facade + downstream concrete records + declarative manifest +
  recipe-identity vs a literal-id twin (different plan_hash);
  (2) historical immutability + hashing (fixture edit makes a different
  checkpoint best; old record byte-stable from disk and still pins A;
  new runs pin B with a different plan_hash; determinism re-run);
  (3) M51 preflight/composite parity (preflight plan ==
  execution record plan, equal plan_hash; parent→child(best) qualified
  ids; zero-checkpoint model → FileNotFoundError at BOTH preflight and
  run with nothing persisted; unknown model 404; inline plans resolve
  through the same executor path; resolver idempotence);
  (4) API end-to-end (contradictory registration 422; preflight 404 on
  a zero-checkpoint model; after training, preflight resolves the
  locally computed argmin, ×3 byte-identical; execution pins the same
  id with plan == preflight; inline run resolves; OpenAPI 84 UNCHANGED
  with `best` in the `EvalStateKind` enum and
  `resolved_checkpoint_id` in `StageStateRef`).
- **After**: **570 passed @ 158.40s and @ 161.98s** (full suite ×2 on
  final code), `compileall` OK, `pyflakes` clean on every touched file.
- **OpenAPI**: **84 → 84 (unchanged)** — M53 adds NO route (§12); only
  the schema evolved (enum member + pinned-id field). No sweep needed;
  the existing `== 84` assertions (M52's and M53's own) verify it.
- **Live smoke**: `smoke_m53_live.py` (port 8774, production FORGE_ROOT)
  — **the certified run: 49/49 PASS**. Independently computed argmin
  from the persisted manifests: `0511de4c7372` @ 6.210553 (agreeing
  with the M52 endpoint); a real minimal recipe `m53-live-best`
  (single read-only `suite_run` stage on `m9-live-suite`, state `best`)
  registered (one manifest, declarative); M51 preflight pins
  best → `0511de4c7372` (×3 byte-identical, zero writes); **3 real
  synchronous executions** — each completed with the record's plan
  byte-identical to the preflight, all three sharing one resolved plan
  identity, artifacts + downstream suite-run records carrying the
  CONCRETE id with full evidence reuse (2/2 probes, "0 created, 2
  reused"), exactly 2 new manifests per run and nothing else; the
  recipe manifest byte-identical after all executions; no
  training/rollback/checkpoint mutation (checkpoint listing 3,
  training provenance 2, model manifest byte-identical); dashboard
  changed only in the workflow/suite-run/artifact-graph sections;
  M3/M35/M51/M52 surfaces + OpenAPI 84 unchanged; final audit 109 files
  with every new file justified.

## §7 — Storage and artifact audit

```
files before : 102          bytes before : 4,014,007
files after  : 109          bytes after  : 4,026,085
delta        : +7 files / +12,078 B / 0 .tmp
```

Every `m53_pre.sha256` entry (102 files, re-captured on the
re-certified baseline after the incident recovery; the 96 original
production files byte-identical to the committed M1–M50 inventories)
remains byte-identical. The 7 new files, each justified:

- `workflow-recipes/m53-live-best/manifest.json` — the ONE new minimal
  immutable recipe required for live verification (all 7 existing
  production recipes pin literal ids and may not be modified); it is
  DECLARATIVE (`state_kind: "best"`, nothing pinned).
- `models/4a0a871886ef/workflows/workflow-bf0112f97217/manifest.json`,
  `…workflow-4e3ba5524f10/manifest.json`,
  `…workflow-0b7c21aa6470/manifest.json` — the three REAL executions of
  the best recipe (completed; each pins best → `0511de4c7372` in its
  immutable plan).
- `suite-runs/4203b674c8f0/manifest.json`, `…455cb6b0d528/manifest.json`,
  `…549cfc7af1f7/manifest.json` — the suite-run records those
  executions created (read-only probe-suite runs at the resolved
  checkpoint; 2/2 probes satisfied by pre-existing evidence — execution
  bookkeeping only).

No weights copied, no best-pointer file, no cache/index, no snapshots.

## §8 — Limitations

- `"best"` is resolved at preflight/execution time against the
  checkpoints that exist at that moment. It is NOT a future lock and
  NOT a mutable global pointer: nothing like
  `models/{id}/best_checkpoint.json` exists; two executions of the same
  declarative recipe may resolve different checkpoints (each record
  pins its own resolution). A preflight cannot bind a later execution
  across intervening training — the run record is authoritative.
- "Best" means exactly the persisted validation-loss criterion (M52) —
  no claim about semantic quality, factuality, safety, instruction
  following, generalization, or overall model quality.
- `best` is a WORKFLOW state reference (`StageStateRef`): suite-run
  states, comparison sides, gate candidates. It is deliberately not
  accepted by direct comparison/gate/suite-run API requests
  (`ComparisonState` rejects it), and evaluation stages keep their
  existing config-level state mechanism
  (`EvaluationConfig.checkpoint_id` / `checkpoint_from_stage`) —
  extending `best` there would add a second representation; a possible
  follow-up, not M53.
- Nothing automatic: no rollback, no retraining, no promotion, no
  persisted selection state, no second executor, no new endpoint.

## §9 — NEXT MILESTONE

The by-* ladder stays closed. Evidence from the actual system after
M53:

- **gates consuming the selected checkpoint** — already possible since
  M53: a gate stage's `candidate` may be `state_kind: "best"` (resolved
  + pinned like any other ref). No work remains.
- **rollback-to-best convenience route** — rejected: `GET
  /checkpoints/best` + `POST /models/{id}/rollback` already compose
  into two EXPLICIT requests; a sugar route would couple selection to
  weight mutation and add a mutating endpoint with no new capability.
- **automatic anything** — rejected (standing philosophy).
- **SFT differentiation / data-quality / deployment** — still heavy or
  evidence-less (M52 §9 reasoning stands).

**SELECTED: explicit training resume point.** The evidence: training
resumes from `model_record.latest_checkpoint` (`training.py`:
`initial_checkpoint_id = parent_checkpoint_id =
model_record.latest_checkpoint`) — the run's starting state is never
requestable. Today the ONLY way to continue training from the best
checkpoint is `rollback(best)` first, which MUTATES the model's
published current state and `latest_checkpoint` as a side effect of
preparing a run. The iterative-improvement loop the project is building
toward — train → measure → select best → continue from best — cannot be
expressed non-destructively. A `TrainingConfig` field
(`resume_from_checkpoint_id: Optional[str]`, default None = today's
behavior) that initializes a run from a VERIFIED immutable checkpoint
(content-hash checked) with correct lineage
(`parent_checkpoint_id`) is the smallest step that closes this gap:
deterministic, explicit, no auto-selection (the caller names the
checkpoint — or composes it with M52's `GET /checkpoints/best`), and it
keeps the improvement loop fully non-destructive to published state.

Copy-ready M54 prompt:

```
# M54 — EXPLICIT TRAINING RESUME POINT (continue from a verified checkpoint)

Extend AI Model Forge from the certified M53 state with exactly ONE
additive training capability: a run may name the immutable checkpoint
it initializes from.

1. Inspect first: app/training.py (run(): how weights are loaded —
   storage.load_weights over the model's CURRENT published weights;
   initial/parent checkpoint lineage; keep_best), app/schemas.py
   (TrainingConfig, RunProvenance, CheckpointRecord.weights_sha256,
   verify_checkpoint), app/engine.py facades, app/api.py training
   routes, tests/test_training*.py. Run the FULL suite BEFORE any edit
   (570 expected) and confirm OpenAPI 84. Capture m54_pre.sha256
   (109 files) BEFORE any live testing.
2. Add TrainingConfig.resume_from_checkpoint_id: Optional[str] = None
   (model-scoped like every checkpoint reference; min/max length;
   default None = EXACTLY today's behavior — resume from the model's
   current published weights/latest lineage, byte-identical runs for
   existing configs). Do NOT change any existing field's meaning.
3. Semantics: when set, the run verifies the checkpoint
   (verify_checkpoint semantics: content hash over weights, 404/409
   taxonomy preserved) and initializes the model from ITS weights
   instead of the published current weights — a pure per-run
   initialization choice: the model's published weights.pt,
   latest_checkpoint, and every manifest stay UNTOUCHED (no rollback
   side effect). Lineage: the run's checkpoints record
   parent_checkpoint_id = the resumed checkpoint (the existing
   lineage field); RunProvenance records the resume point so the
   immutable report shows where the run started. Deterministic for a
   fixed seed + fixed resume checkpoint.
4. Validation: unknown checkpoint -> existing 404 taxonomy; a
   checkpoint belonging to ANOTHER model -> 404 (scoped like every
   other checkpoint reference); contradictory combinations rejected
   (e.g. resume_from_checkpoint_id must not collide with any existing
   field's meaning — inspect first, add only what the schema needs).
5. Reuse only: NO new executor, NO new registry, NO copy of weights
   (load from the existing checkpoint store), NO new endpoint (the
   existing POST /training/run accepts the extended config
   automatically). Workflow train stages gain the field via the
   existing TrainingConfig embedding (verify plan validation still
   holds; a train stage may name a resume point explicitly — from_stage
   semantics for STATE references stay unchanged).
6. Tests (~5): resume-from-checkpoint produces different (and
   deterministic) weights/losses than resume-from-current on the same
   seed; lineage + provenance record the resume point; published state
   untouched (weights.pt bytes + latest_checkpoint unchanged — the
   non-destructive guarantee); unknown/foreign checkpoint 404s;
   default-None byte-identical behavior for an existing config;
   determinism (same seed + same resume point -> identical run twice).
   Update README counts.
7. Gates in order: focused tests -> FULL suite x2 -> compileall ->
   pyflakes -> OpenAPI check (path count UNCHANGED, 84) -> live smoke
   x3 -> final SHA audit. Rerun the x2 gate if anything changes after
   it. Never certify intermediate code; never loop a state-mutating
   smoke — invoke it exactly once per certification.
8. Live smoke (production FORGE_ROOT; training IS weight-work, so this
   is the first milestone since M12 whose live verification legitimately
   creates checkpoints): model 4a0a871886ef owns 3 checkpoints with
   best 0511de4c7372 @ 6.210553 (DISCOVER live from manifests, never
   hardcode). Run ONE short training with resume_from_checkpoint_id =
   the best checkpoint: verify the run report records the resume point,
   the new checkpoints descend from it (parent_checkpoint_id), the
   model's published weights + latest_checkpoint + every pre-existing
   file are byte-identical (non-destructive), and the storage delta is
   exactly the new run's checkpoints + weights + updated model manifest
   entries — each justified. Then verify the M52 selection and M53 best
   references still resolve (the registry grew: assert the new argmin
   is discovered live). Also run ONE default-None training on the
   SAME seed pre-resume-test and post... only if needed to prove
   byte-identical default behavior — prefer the test suite for that
   proof and keep the live footprint minimal (each additional run
   creates real artifacts).
9. Final report with EXACTLY 9 sections; §9 selects the next milestone
   from evidence (candidates to weigh: composing the loop — a workflow
   train stage that resumes from the M52 best selection (state-ref
   style); SFT method differentiation; dataset-version tooling — or
   honest exhaustion). Include a complete ready-to-copy prompt if
   justified. Completion standard: never mutate published state as a
   side effect, never auto-select, never a second executor, never
   certify from partial runs; report actual counts honestly.
```
