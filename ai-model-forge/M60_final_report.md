# Milestone 60 — Declarative Best-Publication Stage: Final Report

**Date:** 2026-09-10 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

The canonical improvement loop now ends declaratively:
`TRAIN → EVALUATE(best) → GATE(best) → PUBLISH(best)` — and the M60
live smoke closed the production drift with exactly one call:
published/latest `eaf0984fbc2b` → **`41870af53ab7`** (the M52 best,
6.242401 nats).

---

## §1 — Baseline & inspection

**Baseline (before any edit):** full suite **599 passed** (exit 0);
HEAD `bb2c409` (M59, pushed); worktree clean (only the untracked
`ai_model_forge.egg-info/`); OpenAPI **85 paths**; production
inventory **78 files / 15,281,640 B / 0 tmp**, sha256-verified
byte-identical to `m59_pre.sha256` (M59 was read-only, as certified).
Production drift confirmed live and read-only:
M52 best `41870af53ab7` @ **6.242401** ≠ published
`latest_checkpoint` `eaf0984fbc2b` — the exact case M60 closes; no
test state needed to be constructed.

**Authoritative machinery identified and REUSED (nothing duplicated):**

| Concern | Existing mechanism (inspected, reused as-is) |
|---|---|
| Publication/rollback (M3) | `TrainingEngine.rollback(model_id, ckpt_id)` behind `POST /models/{id}/rollback`: `verify_checkpoint` (weights content-hash vs manifest) → `storage.write_weights` (atomic temp+fsync+rename + sidecar sha256) → model manifest update (`latest_checkpoint`, `updated_at`) |
| Best selection (M52) | `TrainingEngine.select_best_checkpoint` — the ONE selector (`_displaces_best`, shared with M59) |
| Best-state resolution (M53) | `WorkflowEngine.resolve_best_state_refs` — the ONE resolver (already pinning M53 refs, M55 resume, M56 evaluate, M57 baselines) |
| Workflow execution (M7) | `WorkflowEngine.run` / `_execute_stage` dispatch + failure path (`failed` record + re-raise) |
| Recipes / composition (M14) | `_expand_definition` / `_qualify_stage` — publish stages carry no `from_stage` refs, so they splice through untouched |
| Repetitions (M58) | `run_repeated` — every iteration is a full independent resolution, so publish re-resolves per iteration for free |
| Preflight (M51) | `RecipeEngine._resolve` — pins through the same resolver; parity is inherited |
| Record conventions | `_record`/`_persist` (immutable one-file-per-run), `result_hash` (semantic execution only, ids excluded) |

**Storage rules confirmed in force:** checkpoints are content-addressed
and immutable; atomic writes everywhere; no second pointer system
exists to duplicate.

## §2 — Implementation

Exactly five files touched (+725 / −8 lines), mirroring the M55–M57
adapter pattern:

- **`app/schemas.py`** — `StageType.PUBLISH = "publish"`;
  `ArtifactKind.PUBLICATION = "publication"`; new
  `WorkflowPublishStage` (`checkpoint_id` XOR `publish_from_best`
  XOR rules; `resolved_checkpoint_id` only valid with the declaration;
  neither source rejected; `extra="forbid"`); `WorkflowStage.publish`
  payload field wired into `_payload_matches_type` (payload list +
  expected map + error message).
- **`app/workflows.py`** — (1) the M53 resolver gained
  `_publish_unresolved` + a fast-path term + a pin branch: the SAME
  single per-plan M52 selection is pinned on
  `publish.resolved_checkpoint_id`, flowing into `plan_hash`;
  (2) `_execute_stage` gained the PUBLISH branch — a guard refusing an
  unpinned `publish_from_best` (never asks "what is best?"), then
  `self.training.rollback(model_id, <concrete id>)` (THE M3
  mechanism), then a **reference-only** `WorkflowArtifact`:
  kind `publication`, `artifact_id`/`checkpoint_id` = the published
  checkpoint, `state_hash` = its persisted `weights_sha256`,
  `final_validation_loss` = its persisted `validation_loss`.
- **`app/api.py`** — one landing-page feature bullet (M60), placed
  after the M59 bullet. No routes, no request/response models.
- **`README.md`** — Milestone 60 section; counts 599 → 606 tests;
  the honest-limitations bullet about manual rollback refreshed (the
  gate still executes nothing; an explicit user-authored PUBLISH stage
  is now possible).
- **`tests/test_workflow_recipes.py`** — the M60 section: 7 tests
  (schema matrix, drift acceptance, explicit/idempotence/determinism,
  resolution errors/ties/corruption, canonical loop + gate stop,
  composition + repetitions, API + OpenAPI).
- **`smoke_m60_live.py`** — the executed-once live certification
  script; **`m60_pre.sha256`** — the pre-smoke production inventory.

No changes to `app/engine.py`, `app/training.py`, `app/recipes.py`,
`app/storage.py`, `app/dashboards.py`, or any route.

## §3 — Best resolution

`publish_from_best` rides the **existing M53 resolver** — the same
method, the same single `select_best_checkpoint` call per plan, the
same idempotent pinned form (`publish_from_best=True` +
`resolved_checkpoint_id=<concrete id>`), the same
`plan_hash` identity semantics (a different resolution ⇒ a different
plan ⇒ a different run identity). M52 tie semantics are inherited
verbatim (tested with an exact tie: the canonical first-among-equals
wins the pin). A model with no selectable checkpoints raises
`FileNotFoundError` **before** anything executes or persists (the
established 404). The M51 preflight pins the identical id (tested:
`resolve().plan == run().plan`). M58 repetitions re-resolve per
iteration (tested: iteration 2 pinned and published an output of
iteration 1's train stage). The executor receives only a concrete id;
the un-pinned guard raises `ValueError` (tested directly on the stage
adapter — unreachable through public paths because `run()` resolves
first).

## §4 — Publication

The PUBLISH branch is a thin adapter over **the existing M3 verified
rollback/publication machinery** — the exact `TrainingEngine.rollback`
behind `POST /models/{id}/rollback`:

1. `verify_checkpoint` loads the checkpoint's weights and confirms the
   content hash against the manifest (`RuntimeError` on corruption —
   the existing integrity refusal; a corrupt resolved checkpoint fails
   the stage, persists a normal `failed` record, and publishes
   nothing);
2. `storage.write_weights` atomically restores the verified state as
   the model's current published weights (temp + fsync + rename, plus
   the sidecar content hash);
3. the model manifest's `latest_checkpoint` is updated (+
   `updated_at`) through `save_record`.

No second publication path, no weights copy, no new pointer system, no
mutation of the immutable source checkpoint. The direct M3 rollback
route keeps naming the concrete checkpoint (`RollbackRequest` still
has exactly one field — tested); 'best' never leaks into M3. Failure
safety is M3's own atomicity: verification happens BEFORE any write,
so a failed publication never partially replaces the live state
(tested with corrupted weights).

## §5 — Tests

**Targeted (7 new, all in the M60 section of
`tests/test_workflow_recipes.py`):**

| Test | Covers |
|---|---|
| `test_m60_schema_matrix_and_m3_surface_unchanged` | default `publish_from_best=False`; declarative/pinned forms; both/neither/pin-without-declaration rejected; payload/type wiring; gate-only branches; `RollbackRequest` unchanged; resolver passthrough for declarative-free plans (identical `plan_hash`) |
| `test_m60_publish_best_closes_drift` | **the primary acceptance test**: best A ≠ latest B (fixture edit); preflight parity; pin before execution; artifact identifies A (kind/id/content-hash/loss, no payload); published == A (pointer + tensor equality); A and B byte-immutable; no new checkpoint; M52 + M59 answers unchanged |
| `test_m60_explicit_idempotent_deterministic` | explicit checkpoint publication; republishing an already-published best (safe, weights identical, source untouched); equivalent runs → identical `result_hash`/`plan_hash`, distinct ids; the unpinned-executor guard |
| `test_m60_resolution_errors_ties_and_corruption` | exact-tie pin through the resolver (`tied=True`); unknown model / no selectable checkpoints → `FileNotFoundError` with nothing persisted; corrupt resolved checkpoint → `RuntimeError`, previous published state intact, `failed` record persisted (fixture restored in `finally`) |
| `test_m60_canonical_loop_and_gate_stop` | TRAIN → EVALUATE(best) → GATE(best) → PUBLISH(best) completes; ONE per-plan selection pins all four declarations; gate failure without `on_fail` → `stopped`, publish skipped, published state kept |
| `test_m60_composition_and_repetitions` | M14 nested recipe (composition trace + publish through expansion); M58 repetitions=2 — iteration 1 pins/publishes the plan-start best, iteration 2 re-resolves to iteration 1's output and publishes it; M55 resume + M60 publish pin the SAME id per plan |
| `test_m60_api_surface_openapi` | HTTP registration matrix (422s); 404 on a model with no checkpoints (nothing persisted); full drift → `PUBLISH(best)` → published == best over public APIs only (train, rollback-to-other, inline workflow run); M52/M59 endpoints unchanged; repetitions=2 over HTTP; OpenAPI 85 + schema exposure |

**Full suite:** **606 passed × 2 consecutive runs** (exit 0 both;
599 + 7). One intermediate failure during development (the tie fixture
used an absolute loss not guaranteed minimal under the full-suite
ordering — fixed to half the current global minimum by construction;
also `list_workflows` is oldest-first, so the newest failed record is
`[-1]`, not `[0]`).

**Statics:** `pyflakes` clean on `app/` + `tests/` +
`smoke_m60_live.py`; `compileall` clean.
**OpenAPI:** path count stays **85** (no new route — the stage rides
`POST /workflows/run`, recipe runs and the M51 preflight);
`WorkflowPublishStage` exposed with `publish_from_best.default=false`;
`WorkflowStage.publish` present; `StageType` and `ArtifactKind` each
gained exactly one value.

## §6 — Live smoke

Executed **exactly once** (uvicorn on 127.0.0.1:8782, production
root, stopped immediately after): **24/24 PASSED on the first
execution — no corrected assertions, no post-hoc fixes.**

- **Pre-smoke (read-only):** drift verified live — best
  `41870af53ab7` @ 6.242401 vs published `eaf0984fbc2b`; M59 history
  intact (15 entries, 23 candidates, the last four movements =
  M58's); source checkpoint hashes captured.
- **THE one state-mutating call:** `POST /api/v1/workflows/run` with
  the single-stage inline plan
  `{"type": "publish", "publish": {"publish_from_best": true}}`
  (workflow `228ef18f836d`, status `completed`).
- **Post-verification (read-only, against the live service):** plan
  pinned `41870af53ab7` before execution; artifact is
  reference-only (`state_hash` = the persisted weights content hash,
  `final_validation_loss` = 6.242401); **the invariant: published ==
  best** (`latest_checkpoint` = `41870af53ab7`); M52 best endpoint
  unchanged; M59 history byte-identical; 23 checkpoints (no new
  one); both the best and the former-latest checkpoint files
  byte-identical; published `weights.pt` tensor-equal to the best
  checkpoint's verified state; sidecar hash consistent; exactly one
  new workflow record, listed with the pinned lineage.
- **Audit:** 0 tmp entries; exactly ONE new file; only the three
  expected M3 publication files modified.

## §7 — Storage & lineage

| | before | after | delta |
|---|---|---|---|
| files (excl. tmp) | 78 | **79** | +1 |
| bytes | 15,281,640 | **15,283,812** | **+2,172** |
| tmp entries | 0 | 0 | 0 |
| checkpoints | 23 | 23 | 0 |

Every byte explained: **+2,172 B** = the ONE new immutable file
`workflows/workflow-228ef18f836d/manifest.json` (the run record —
M60's only new artifact). Three files legitimately **modified** by the
M3 publication path: `weights.pt` (630,164 B — the verified best
state atomically restored as current), `weights.sha256` (83 B sidecar)
and the model `manifest.json` (24,655 B — `latest_checkpoint` +
`updated_at`). The other **75 of 78** pre-existing files — including
all 46 checkpoint files — are sha256-identical. No second weights
copy, no second best pointer, no best-history materialization, no new
checkpoint.

**Lineage:** the run record's plan carries the declarative request
(`publish_from_best: true`) plus the pinned concrete id; the stage
artifact references the published checkpoint by id + persisted content
hash + persisted validation loss (no payload duplication); the record
chains into `result_hash` semantics (equivalent publications reproduce
it). The post-smoke live state: **published == best ==
`41870af53ab7` @ 6.242401**, with `eaf0984fbc2b` retained immutably.

## §8 — Git

- `M60: declarative best-publication stage` — implementation, tests,
  README/landing docs, smoke script, this report.
- `M60: pre-milestone production inventory (m60_pre.sha256)` — the
  pre-smoke inventory marker.
- Both pushed to `origin/arena/01a071e9-code-forge`;
  `HEAD == FETCH_HEAD`, worktree clean (only the untracked
  `ai_model_forge.egg-info/`).

## §9 — Next milestone

Inspection of the remaining architecture: with M60 the improvement
loop is **complete end-to-end** — declarative (M53–M57, M60),
composable (M14), bounded-repeating (M58), observable (M52/M59), and
now terminating in an explicit ACCEPT that actually publishes the
improved state. The roadmap's own "remaining natural steps" are either
autonomy (scheduled/automatic loops — forbidden by the standing
boundary) or new subsystems (served inference, LoRA — later roadmap
phases). The one operational gap the completed loop now exposes is
**storage**: every iteration appends immutable checkpoints
(content-addressed, so identical states dedupe — but distinct states
accumulate), and the only deletion primitive today is
whole-model delete. The smallest high-value next milestone is
**M61 — explicit verified checkpoint retention**: a request-only,
reference-safe, dry-run-capable pruning primitive over the checkpoint
registry that can NEVER remove the M52 best, the published
`latest_checkpoint`, or any checkpoint referenced by any immutable
record family (evaluations, comparisons, gate decisions/baselines/
suggestions, workflow artifacts incl. M60 publications, suite runs,
sample-quality measurements, M55 resume provenance) — bringing the
"minimum storage" pillar the same explicit, deterministic, no-autonomy
control every other capability already has.

---

### M61 — EXPLICIT VERIFIED CHECKPOINT RETENTION (ready-to-paste prompt)

```
Implement **Milestone 61 — Explicit Verified Checkpoint Retention** for
AI Model Forge.

The improvement loop is complete (train → evaluate(best) → gate(best) →
publish(best), composable and bounded-repeating), and every iteration
appends immutable content-addressed checkpoints. The only deletion
primitive today is whole-model delete. M61 adds the missing explicit,
request-only storage control: verified checkpoint retention (pruning).

### 1. INSPECT FIRST — DO NOT IMPLEMENT YET

Before changing anything, inspect and identify the authoritative
implementations for:

* M3 checkpoint registry/listing/storage layout (content-addressed
  weights, hard-link dedup, manifests).
* Every record family that can REFERENCE a checkpoint id: M4 evaluation
  records, M5 comparison records (state_a/state_b), M6 gate decisions
  (candidate/baseline/suggested_checkpoint_id), M7 workflow records
  (stage artifacts: training final checkpoints, evaluation
  checkpoint_id, publication artifacts, suite-run states,
  from_stage resolutions), M16 sample-quality records, training
  provenance (run manifests, parent/initial checkpoint ids), and the
  model manifest's latest_checkpoint / best_checkpoint fields.
* The existing whole-model delete and tokenizer/dataset delete
  conventions (request shape, confirmation, 404s, idempotence).
* Existing atomic-delete / rmtree conventions and startup tmp cleanup.
* The M52 selector and M59 history (they must keep working on the
  post-retention registry).
* Existing API/OpenAPI test conventions and fixture-edit patterns.

Run the current full test suite before implementation and record the
baseline (expect 606 passed, OpenAPI 85 paths).

### 2. M61 OBJECTIVE

A request-only primitive that removes SOME immutable checkpoints of ONE
model while guaranteeing, by construction:

* the M52 best checkpoint is NEVER removable;
* the published `latest_checkpoint` is NEVER removable;
* any checkpoint referenced by ANY immutable record family is NEVER
  removable (references must remain resolvable forever);
* retention is explicitly requested, dry-run capable, and fully
  auditable (what was removed, when, and what protected what).

No automatic pruning, no retention policies that run on their own, no
scheduled cleanup, no background jobs.

### 3. DESIGN CONSTRAINTS

* Follow the repository's conventions for delete-style requests; if the
  established style requires an explicit confirmation token or an
  idempotent response shape, reuse it.
* A DRY-RUN mode must return the exact would-be-removed /
  would-be-kept partition with the protecting references, writing
  nothing.
* The reference-safety analysis must be computed LIVE from the
  authoritative listings (never from a stored index that can drift).
* Deletion must be atomic-per-checkpoint (a mid-failure leaves the
  registry consistent; already-removed checkpoints stay removed and are
  reported).
* Unknown model → the family's 404; unknown checkpoint ids in an
  explicit-selection mode → the established 404/422 semantics.
* After retention: M52 selection, M59 history, M46 by-run listings and
  the published state must be unaffected (they operate on the
  surviving registry; the M59 history must degrade honestly if
  non-best movements are removed — inspect what the established
  semantics imply and preserve them exactly).
* Do not invent a second registry, a tombstone system, or new
  per-checkpoint metadata beyond what the audit response requires.

### 4. SELECTION MODES

Support at minimum:

* **auto (safe) mode**: remove every checkpoint that is neither the
  best, nor the published, nor referenced by anything — the mode dry
  runs first by design if requested so;
* **explicit mode**: the request names exact checkpoint ids; each must
  pass the same reference-safety checks individually (a protected id is
  refused with the reason, never silently skipped — inspect whether
  the repo style prefers all-or-nothing or per-item results and follow
  it).

### 5. AUDIT

The response (and only the response — no new persisted records unless
inspection proves the repo convention demands one) must report:

* removed checkpoint ids (with their persisted step/loss/size);
* kept ids with the reason (best / published / referenced-by, naming
  the referencing artifact families and ids);
* bytes and file counts reclaimed.

### 6. TEST MATRIX

Focused tests following existing style, at minimum:

* best/published/referenced checkpoints are never removable in either
  mode (one test per protecting family: evaluation, comparison, gate
  candidate/baseline/suggestion, workflow artifact incl. an M60
  publication artifact, suite run, sample-quality, training
  provenance parent/initial);
* auto mode on a fixture with removable checkpoints removes exactly
  them (registry count/listing verified, storage actually reclaimed);
* explicit mode: valid removal, protected refusal with reason,
  unknown id;
* dry run writes NOTHING (byte-level inventory comparison);
* atomicity: a mid-deletion interruption path leaves a consistent
  registry (simulate per inspection of what is reachable);
* M52/M59/M46/published-state behavior after retention;
* OpenAPI: exactly the new routes, count updated 85 → 85+N with the
  reason reported; README/landing updated;
* full regression: all existing tests green.

### 7. LIVE CERTIFICATION

After the full suite passes twice and statics are clean, run exactly
one live smoke against production:

* capture the pre-smoke inventory (m61_pre.sha256) and the reference
  graph;
* dry-run FIRST (must report the exact partition and write nothing —
  verify byte-identical inventory);
* then execute the smallest legitimate retention that actually removes
  at least one unprotected checkpoint (construct it through existing
  public APIs if production has none — e.g. train a throwaway run on
  the production model if — and only if — that matches the
  established production conventions; otherwise document honestly and
  use the safest available state);
* verify read-only afterwards: best/published/referenced untouched,
  reclaimed bytes match, M52/M59 answers consistent with the surviving
  registry, no tmp residue;
* never "repair" a failed smoke by mutating production; correct only
  over-tight FACT expectations and re-verify read-only.

### 8. REPORT

Report exactly these 9 sections: Baseline & inspection; Implementation;
Reference-safety analysis; Tests; Live smoke; Storage & lineage;
API/OpenAPI; Git; Next milestone — then provide the next ready-to-paste
implementation prompt.

### HARD CONSTRAINTS

* Do not skip inspection.
* Do not weaken existing tests.
* Do not remove or endanger the best/published/referenced checkpoints.
* No automatic/background/scheduled pruning — explicit requests only.
* No second registry, no tombstones, no stored reference index.
* Keep storage minimal; explain every reclaimed and new byte.
* Preserve backward compatibility.
* The authoritative invariant is: **after any retention request, every
  reference in every immutable record still resolves, the M52 best and
  the published checkpoint still exist, and nothing was removed that
  was not explicitly allowed.**
```
