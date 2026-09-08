# M52 Final Report — best-checkpoint selection (`checkpoints/best`)

Milestone date: 2026-09-08. Branch `arena/01a071e9-code-forge`.

## 1. Baseline

The session environment had been re-provisioned since the M51
certification (local HEAD reset to the branch base `f86b670`, all
M1–M51 work present only as untracked files, `/home/user/
ai-model-forge-data` empty, venv gone). The proven recovery runbook
was applied BEFORE any M52 work, step by step:

1. Branch-specific `git fetch origin arena/01a071e9-code-forge` →
   `FETCH_HEAD a383206`; confirmed the checked-out branch was
   `arena/01a071e9-code-forge` (never `main`); `git reset --hard
   FETCH_HEAD` → full M1–M51 history restored, worktree clean except
   the known untracked `ai_model_forge.egg-info/`.
2. Production storage extracted from the workspace backup zip
   (`code forge.zip`, 96 pre-M51 entries): **96 files /
   4,002,745 B / 0 `.tmp`**, `m51_pre.sha256` verified **96/96 OK**.
3. Venv rebuilt (`torch 2.14.0+cu130`, pydantic 2.13.5, fastapi
   0.141.1); `from app.api import app` OK, OpenAPI 83.
4. Baseline full suite re-proven: **564 passed @ 207.86s** (slower
   than M51's 135s — the rebuilt venv's CUDA-bundled torch wheel runs
   CPU-only; green is what matters, timing is an environment fact).
5. M51 re-certified live: `smoke_m51_live.py` run once against the
   restored 96-file baseline → **58/58 PASS**, re-executing the
   production recipe `m12-live-suite` ×3 (full evidence reuse each
   time) and leaving exactly 6 justified new manifests — workflows
   `884d2dc725a5` / `b092cd0b69f7` / `bc8c7ef3081f` + suite-runs
   `0214fa94a965` / `aadcf1f3c893` / `e34a273aff6a` (new ids —
   disclosed here; the ids in the M51 report refer to the original,
   lost execution artifacts). Certified state restored: **102 files /
   4,013,875 B / 0 `.tmp`**, original 96 byte-identical.

M52 baseline then verified: `HEAD a383206` == remote, tests 564,
OpenAPI **83**, production **102/0 tmp**, 3 checkpoints all persisting
`validation_loss`/`perplexity`/`decision` with `decision: accept`, and
the persisted argmin **`0511de4c7372` @ 6.210553** — the checkpoint the
production recipe pins. `m52_pre.sha256` (102 lines) captured and
verified before any live testing.

## 2. Grounding

- `CheckpointRecord` (`app/schemas.py`): `validation_loss: float` —
  **REQUIRED** on every checkpoint manifest, no finiteness constraint
  (python's json can round-trip non-standard `NaN`/`Infinity`
  literals, so a non-finite value is theoretically persistable —
  M52 therefore never treats one as a candidate).
- Authoritative source: `TrainingEngine.list_checkpoints(model_id)`
  (M3) — unknown model → `FileNotFoundError` (404); unreadable/
  malformed manifests are **skipped** with a warning (established
  semantics, M46-verified); zero checkpoints → `[]`; canonical order
  **`(step, created_at)` ASCENDING**.
- `decision` semantics (`app/training.py`): `ACCEPT` = "validation
  loss beat the best so far", `NOT_BEST` otherwise — adjacent evidence
  but NEVER a selection input.
- Production candidates (read directly from the manifests): `025e6d8d8f15`
  step 20 loss 6.360189; `0511de4c7372` step 16 loss **6.210553**;
  `30a8bc5b82ab` step 10 loss 6.397286 — all `accept`, argmin
  `0511de4c7372`. The selection uses the persisted
  `validation_loss` **verbatim** — never derived from evaluations,
  perplexity, gate decisions, comparisons, training configuration,
  ids, timestamps, filenames or model state.

## 3. Implementation

- `app/schemas.py` — `CheckpointSelection`: a minimal computed-view
  wrapper (`model_id`, `criterion: Literal["minimum_persisted_
  validation_loss"]`, `candidate_count` (ge 1), `tied: bool`,
  `checkpoint: CheckpointRecord` — the complete verbatim record;
  `extra="forbid"`). Never persisted; no rankings, trends, scores or
  recommendations. `CheckpointRecord` itself is UNCHANGED.
- `app/training.py` — `TrainingEngine.select_best_checkpoint(model_id)`:
  candidates = the authoritative M3 listing minus non-finite
  `validation_loss` values; selection = `min()` by persisted
  `validation_loss` over the listing's canonical `(step, created_at)`
  ASC order — an exact tie resolves to the **first checkpoint among
  equals in that established canonical order** (inspection-proven the
  repo's canonical ordering, not a silently invented timestamp rule)
  and is **disclosed** via `tied`. Valid model + no selectable
  checkpoints → `FileNotFoundError` (the family's 404; nothing
  manufactured). Read-only: no writes, no evaluation, no training, no
  weights, no `decision` change, no selection pointer.
- `app/engine.py` — thin facade `select_best_checkpoint` beside the
  other checkpoint facades (API → facade → selection in the training
  engine → authoritative M3 registry; no selection logic in `api.py`,
  no second registry).
- `app/api.py` — `GET /models/{model_id}/checkpoints/best`
  (`response_model=CheckpointSelection`, `tags=["training"]`, 404
  mapping) declared **before** the generic `{checkpoint_id}` detail
  route (FastAPI matches in declaration order — same slot as the M46
  `by-run` literal route), so `best` can never be captured as an id;
  verified live. Docstring/README state the criterion explicitly and
  that "best" is NOT a claim of overall model quality (semantic
  quality, factuality, safety, generalization are not established).
- **Dashboard untouched (§13)**: it aggregates persisted facts and its
  `result_hash` covers that aggregate; a computed selection view is
  not a persisted fact, and adding one would change the dashboard
  hash without any new storage fact — an unjustified mutation for a
  selection primitive. Left unchanged, verified byte-identical in the
  smoke.

## 4. Tests

Two focused tests appended to `tests/test_training_api.py` (564 →
**566**), following the M46 conventions:

1. `test_m52_best_checkpoint_selection_parity_determinism_errors` —
   multi-checkpoint model; expected argmin computed INDEPENDENTLY from
   the listing (canonical order tie-break); returned record verbatim-
   equal to the listing entry AND the M3 detail payload; criterion/
   candidate_count/tied correct; ×3 byte-identical; route collision
   (the response carries the criterion — the detail route would have
   404'd an id "best"); zero writes under the model dir; unknown
   model 404; a created-but-never-trained model → 404
   "no selectable checkpoints" with listing `[]`.
2. `test_m52_best_checkpoint_tie_corruption_openapi` — an EXACT tie
   crafted by editing one test-fixture manifest's `validation_loss`:
   selection returns the FIRST among equals in canonical order and
   `tied: true`; all manifests corrupted (`{ not json`, M46 style) →
   listing `[]` → 404; OpenAPI 84, path once, GET-only, training tag,
   `model_id` param, `$ref CheckpointSelection` → `$ref
   CheckpointRecord`, display order listing < by-run < best < generic
   detail.

**Gate (final code)**: full suite ×2 → **566 passed @ 210.91s** and
**566 passed @ 204.60s**; `compileall` OK; `pyflakes` clean on every
touched file. No implementation change after the gate.

## 5. OpenAPI

**83 → 84 — a real increase**: no existing route has selection
semantics (listing/detail/by-run are raw access; nothing anywhere
computes an argmin; `keep_best`/`best_validation_loss` are internal
training facts, not access paths). Contract: `GET
/api/v1/models/{model_id}/checkpoints/best` exactly once, GET-only,
`tags=["training"]`, one `model_id` string param, response
`$ref #/components/schemas/CheckpointSelection` whose `checkpoint`
property `$ref`s the existing `CheckpointRecord` (which enters
components via the wrapper — no duplicated schema). Sweep: **36
assertions** updated `== 83` → `== 84` (35 `len(spec["paths"])` across
9 modules + the M51 test's `len(keys)`) — none weakened, 0 remaining.

## 6. Live Selection

`smoke_m52_live.py` (port 8773, production FORGE_ROOT), run **three
times: 28/28 PASS each** (read-only — identical results every time):

- **Independent expectation**: all 3 production checkpoint manifests
  read directly from disk; argmin computed independently →
  `0511de4c7372` @ **6.210553** (candidates 6.360189 / 6.210553 /
  6.397286 printed live; the value was NOT hardcoded).
- **Selection correctness**: API returned exactly that checkpoint;
  returned `validation_loss` == the persisted value; the full record
  verbatim-equal to the authoritative manifest, the M3 listing entry
  and the M3 detail payload; `criterion
  == "minimum_persisted_validation_loss"`, `candidate_count == 3`,
  `tied == false`.
- **Determinism**: ×3 byte-identical bodies.
- **Empty model**: `b5bc905326b6` (valid, never trained, zero
  checkpoints) → 404 "no selectable checkpoints", listing `200 []` —
  nothing manufactured. **Unknown model** → 404.
- **Route collision**: `/checkpoints/best` resolves to the SELECTION
  route (criterion key present); a real checkpoint id still resolves
  through the detail route with the verbatim record; OpenAPI display
  order listing < by-run < best < generic detail.
- **Regressions**: M3 listing/detail, M46 by-run (2/1), evaluations
  16, comparisons 8, gate decisions 11, workflows 16 (12/3/1),
  suite-runs 13, recipes 7, the **M51 recipe plan preflight**
  (`m12-live-suite` resolves to the same plan its persisted run
  executed — and that plan pins the selected checkpoint), and the
  **dashboard byte-identical** (hash unchanged — read-only milestone).

## 7. Storage Integrity

Before: **102 files / 4,013,875 B / 0 `.tmp`**. After all smoke runs:
**102 / 4,013,875 / 0** — `sha256sum -c m52_pre.sha256` **102/102 OK**,
every pre-existing SHA byte-identical, ZERO new files, no cache, no
index, and **no persisted "best checkpoint" pointer** (the selection
is computed per request; no mutable derived state exists). Final
process/tmp audit clean.

## 8. Git

Two commits on `arena/01a071e9-code-forge` following the convention:
`M52: add best-checkpoint selection by persisted validation loss`
(implementation: schemas/training/engine/api, tests, README, smoke,
this report) and `M52: pre-milestone production inventory
(m52_pre.sha256)` (102-line SHA256 manifest captured before live
testing). Complete diff reviewed: only the intended files; no
production manifest touched; no temp files; no unrelated refactors.
Pushed; `HEAD == FETCH_HEAD ==` remote tip verified after push; worktree
clean except the pre-existing untracked `ai_model_forge.egg-info/`.
Never force-pushed.

## 9. M53 Selection

The by-* grouping ladder stays closed. Candidates weighed against the
actual architecture and live data:

- **automatic rollback to the selected checkpoint** — rejected: it is
  automatic improvement (mutates published weights without an explicit
  per-request instruction), which M51/M52 explicitly do not build.
- **checkpoint promotion/pinning as a persisted pointer** — rejected:
  M52's own constraint (mutable derived state); the pointer would go
  stale the moment a better checkpoint appears.
- **training continuation from an arbitrary checkpoint** — real
  training-semantics change (lineage, `latest_checkpoint`,
  `parent_checkpoint_id` interactions); heavier than the evidence
  justifies; deferred.
- **richer evaluation criteria** — rejected: every persisted number is
  a measurement actually taken; new criteria would invent measurements.
- **training-method compatibility (SFT differentiation)** — deferred
  again: needs dataset-format semantics (chat templates/role masking);
  not the smallest step.
- **inference/deployment** — M15 sampling already provides
  deterministic checkpoint-scoped generation; deployment infra is out
  of scope. **data-quality integration** — no data-quality evidence
  system exists to integrate; would invent scores.

**SELECTED: `best` state references — making the M52 selection
consumable by the explicit state-reference architecture.** Evidence:
every checkpoint consumer (M4 evaluations, M10 suite runs, M7/M12
recipe stages) references model state through `StageStateRef`
(`current` | `checkpoint id` | `from_stage`) — and the production
recipe `m12-live-suite` pins `0511de4c7372` **by hand**, which M52
proved is the persisted argmin today but will silently be stale after
any future training run. A new explicit `state_kind: "best"` —
resolved through the SAME M52 selection at execution time and **pinned
to the resolved concrete checkpoint id in the immutable record** —
removes hand-pinning drift without auto-anything: the request is still
explicit, the criterion is deterministic and disclosed, nothing
mutates. It is the smallest meaningful step connecting the M52
primitive to the state-reference architecture.

Copy-ready M53 prompt:

```
# M53 — "best" STATE REFERENCES (consume the M52 selection explicitly)

Extend AI Model Forge from the certified M52 state with exactly ONE
new explicit model-state reference kind — state_kind "best" — resolved
at execution time through the SAME M52 selection (minimum persisted
validation_loss over the authoritative M3 listing, canonical
(step, created_at) ASC tie-break, first among equals, non-finite
values never candidates). Non-negotiables:

1. Inspect first, edit second: app/schemas.py (StageStateRef, the
   state_kind enum, every consumer), app/evaluation.py,
   app/suite_runs.py, app/workflows.py, app/recipes.py (M51 _resolve),
   app/training.py (select_best_checkpoint), app/api.py, and the
   evaluation/suite-run/workflow tests. Run the FULL suite BEFORE any
   edit (566 expected) and confirm OpenAPI 84. Capture m53_pre.sha256
   from the FORGE_ROOT (102 files) BEFORE any live testing.
2. Resolution semantics: "best" is resolved ONCE per stage execution,
   synchronously, by calling the existing selection; the consuming
   record (evaluation / suite run / workflow stage artifact) must
   persist the RESOLVED CONCRETE checkpoint id (plus the original
   "best" request semantics where the record schema already carries
   state) — a record must NEVER reference the moving label. If the
   model has no selectable checkpoints at execution time, the existing
   M52 404 semantic applies (nothing manufactured, no fallback).
3. Scope: allow "best" exactly where "current"/"checkpoint" state refs
   are allowed today (evaluation state, suite-run state, workflow
   suite_run/evaluate stage state refs) — ONE mechanism, no new
   endpoint families, no mutation of any kind. The model stays the
   only runtime binding; the workflow never guesses — "best" is an
   EXPLICIT deterministic criterion in the request, never a default.
4. No second selection implementation: the state-ref resolution calls
   the same TrainingEngine.select_best_checkpoint (or its facade) the
   M52 route uses. No new schema explosion: extend the existing
   state_kind enum; do not fork StageStateRef.
5. OpenAPI: state kinds appear as enum values (verify how the existing
   enum renders); path count MUST NOT change (84) unless inspection
   proves a real new path is genuinely necessary — it should not be.
   Sweep all == 84 assertions only if something actually changes.
6. Tests (~4-6, minimal but proving): evaluation of state "best"
   records the resolved concrete checkpoint id and equals an
   evaluation of that same checkpoint id run separately (same
   result_hash — evidence-reuse or byte-equality as the engine
   behaves); suite_run "best" pins the resolved id; a recipe stage
   with "best" executes through the unchanged sole WorkflowEngine;
   zero-checkpoint model -> the established 404 at execution;
   determinism (same model state -> same resolution); regression that
   literal checkpoint refs and "current" are byte-identical in
   behavior. Update README counts.
7. Statics/gates in order: focused tests -> FULL suite x2 ->
   compileall -> pyflakes on every touched file -> OpenAPI check ->
   live smoke x3 -> final SHA audit. Rerun the full-suite x2 gate if
   anything changes after it. Never certify intermediate code.
8. Live smoke (read-only where possible; real production recipe
   execution is allowed ONLY with per-artifact justification): verify
   "best" resolves live to 0511de4c7372 (DISCOVERED from the
   persisted manifests, not hardcoded); the M51 preflight of a recipe
   with a "best" ref shows the resolved plan; every new storage
   artifact enumerated and justified; M3-M52 surfaces + dashboard
   sections + OpenAPI unchanged where read-only. If a real execution
   is used, capture the pre-inventory first and justify each new
   manifest.
9. Final report with EXACTLY 9 sections, the last one selecting M54
   from architectural evidence (by-* groupings remain closed; weigh:
   training continuation from an explicit checkpoint, SFT method
   differentiation, dataset-version tooling — or honest exhaustion),
   with a complete ready-to-copy M54 prompt if justified. Completion
   standard: never a second selection implementation, never a moving
   reference in a persisted record, never auto-mutation, never
   certify from partial runs; report actual counts honestly.
```
