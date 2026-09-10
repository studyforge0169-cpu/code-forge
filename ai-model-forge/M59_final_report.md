# M59 Final Report — Read-Only Best-Checkpoint Improvement History

## 1. Scope & Intent

M59 adds the observability layer the improvement loop was missing:
`GET /api/v1/models/{model_id}/checkpoints/best/history` — the
chronological sequence of checkpoints that **BECAME** the M52-selected
best as the model's checkpoint registry grew (only winners; non-winning
checkpoints excluded by construction). Each entry carries the
authoritative persisted fields (checkpoint id, producing `run_id`,
`step`, `created_at`, `validation_loss`, `perplexity`) plus
`delta_loss_nats` — the improvement vs the previous best under the
established sign convention (current − previous; negative = improvement;
None on the first entry; 0.0 only when the canonical tie-break actually
moved the selection). Read-only, model-scoped, deterministic, computed
live from persisted manifests, **zero storage**. It decides nothing —
no automatic stopping, no convergence detection, no repetition
selection.

Baseline recorded before editing: HEAD `3e1cc8a` (pushed, worktree
clean), **596 tests passed**, OpenAPI 84, statics clean, production
78 files / 15,281,640 B / 0 tmp (exactly the certified post-M58
state), no servers running, live M52 argmin `41870af53ab7` @ 6.242401.

## 2. Architecture Inspection

Traced before editing: `select_best_checkpoint` (M52: the authoritative
M3 listing — model-scoped, unreadable manifests skipped with a logged
warning, canonical `(step, created_at)` ASC sort — minus non-finite
losses, then min-by-loss = first-among-equals; `CheckpointSelection`
with explicit criterion/candidate_count/tied); `list_checkpoints`
(corruption-skip semantics; `[]` for a checkpointless valid model;
FileNotFoundError for unknown models); the M52 route (declared BEFORE
the generic `{checkpoint_id}` detail route — the collision discipline);
the M18–M50 history-surface conventions; the M52/M54 test styles in
`tests/test_training_api.py` (`_prepare`/`_run_cfg` helpers,
fixture-edit tie/corruption tests, OpenAPI display-order assertions);
M58's per-iteration resolution provenance (the trajectory M59 must
expose).

## 3. Implementation

Five files touched, minimal additive edits:

* `app/schemas.py` — two new read-only view models:
  `BestCheckpointHistoryEntry` (authoritative fields + nullable
  `delta_loss_nats`) and `BestCheckpointHistory` (model_id, criterion,
  candidate_count, entries).
* `app/training.py` — the ONE winner rule extracted into a shared
  static helper `_displaces_best(candidate, best)` (strictly lower
  persisted loss, or equal loss with a strictly earlier canonical
  `(step, created_at)` key); `select_best_checkpoint` refactored to use
  it (behavior identical — the M52 tests pass unchanged); new
  `best_checkpoint_history(model_id)`: replays that ONE rule over the
  authoritative listing in chronological `(created_at, checkpoint_id)`
  order, emitting an entry exactly when a checkpoint displaced the
  running best. Non-finite losses never candidates (M52's exclusion);
  unreadable manifests skipped by the listing (established semantics);
  unknown model → FileNotFoundError; empty → empty history.
* `app/engine.py` — facade `best_checkpoint_history` (delegation only).
* `app/api.py` — the ONE new route, placed after `/checkpoints/best`
  and before the generic `{checkpoint_id}` detail (display order
  asserted); landing-page route-list line + feature bullet. **No other
  routes touched.**
* `README.md` — M59 section; test counts 596 → 599 (both occurrences).
* `tests/test_training_api.py` — 3 M59 tests + the cumulative OpenAPI
  count assertions bumped 84 → 85 across the suite (41+2 assertions in
  9 files — the established global-bump practice; one enumerating
  comment updated to name M59).
* `smoke_m59_live.py` — the executed-once READ-ONLY live smoke.

Untouched by design: `app/workflows.py`, `app/recipes.py`,
`app/evaluation.py`, `app/gates.py`, `app/dashboards.py` (endpoint-only
per §23 — no dashboard integration; the history is a checkpoint-family
surface). One selector, one comparator: `_displaces_best` is the only
best-selection comparison in the codebase, shared by M52 and M59.

## 4. Tests & Verification

* **New M59 tests: 3** —
  `test_m59_history_parity_determinism_errors` (shape + EXACT parity
  with an independent local replay `_m59_expected_history`; non-winner
  exclusion via a crafted worse checkpoint; the FINAL entry equals the
  M52 route answer verbatim; monotonic losses; delta correctness incl.
  the None first entry and ≤ 0 after; verbatim provenance fields;
  byte-identical repeated GETs; zero writes; unknown model 404; empty
  model → 200 `[]` + candidate_count 0 while the M52 selection stays
  404; second-model scoping isolation),
  `test_m59_history_tie_nonfinite_live_computation` (a crafted
  8-checkpoint landscape across three runs: BOTH tie directions — an
  exact tie with a LATER canonical key creates NO movement (M52 keeps
  the first among equals) while an exact tie with an EARLIER canonical
  key (smaller step from a later run) SWITCHES the selection (movement,
  delta 0.0); `inf` and `NaN` excluded per M52; live recomputation: a
  better checkpoint immediately becomes the final entry with the old
  best retained; corruption: unreadable manifests skipped → empty
  history / selection 404),
  `test_m59_history_openapi` (85 paths; the path exactly once; GET-only;
  training tag; model_id param; typed `$ref` response; display order
  listing < by-run < best < history < generic detail; components
  present).
* **Final count: 599 passed × 2 consecutive full-suite runs** (exit 0,
  zero failures). No M1–M58 test regressed (the only pre-existing-test
  edits were the cumulative OpenAPI count bumps 84 → 85).
* **Statics**: compileall + pyflakes clean over `app`, `tests`, the
  smoke. **OpenAPI: 84 → 85** (exactly the one new read-only path —
  reported per §30).

## 5. Live Smoke

Executed **exactly once** against production (port 8781) — **READ-ONLY:
zero state-mutating calls**. 13 checks: 12 passed in the run; 1 (B7)
was an over-specific FACT slice in my own expectation — corrected with
read-only post-hoc verification (no re-execution; M55–M58 discipline).
Effective certification 13/13.

* **B1–B6 (all PASS)**: `GET .../checkpoints/best/history` → 200;
  shape (model_id, criterion `minimum_persisted_validation_loss`,
  candidate_count 23 of 23 checkpoints); **EXACT parity with an
  independent local replay recomputed directly from the raw persisted
  manifests** (15 entries, every field); only winners (15 movements ≤
  23 checkpoints), losses monotonically non-increasing; deltas correct
  (first None, all ≤ 0, exactly current − previous); the FINAL entry
  EQUALS the live M52 route answer — `41870af53ab7` @ **6.242401**
  (id and persisted loss, exact).
* **B7 (corrected, PASS)**: my original slice expected the last THREE
  movements to be [M57's final, iteration-1 final, iteration-2 final] —
  forgetting that each M58 iteration's first train stage produced TWO
  winning checkpoints (its step-2 AND step-4 outputs both beat the
  prior best). The certified M58 trajectory occupies the last FOUR
  movements: `958bf4b220ed` @ **6.297584** → `85c0ea2b6d52` @
  **6.290843** → `976df555e307` @ **6.24882** → `41870af53ab7` @
  **6.242401** (verified read-only post-hoc; the first value is the
  persisted 6.297584 — the run log displays it rounded to 6.2976).
* **The full production trajectory** (15 movements, total **−0.210304
  nats**): 6.452705 → 6.444046 → 6.439390 → 6.438945 (bootstrap run 1)
  → 6.415687 → 6.401624 → 6.400210 (bootstrap run 2) → 6.373884 →
  6.369091 (M56 loop) → 6.336222 → 6.331234 (M57 loop) → 6.297584 →
  6.290843 (M58 iteration 1) → 6.248820 → 6.242401 (M58 iteration 2).
  Every milestone's advance is legible — exactly the §24 goal.
* **C1–C2, D1–D3 (all PASS)**: repeated GETs byte-identical; unknown
  model → 404; OpenAPI 85 with the path exactly once (GET-only,
  training tag, typed `$ref`, display order best < history < generic
  detail); **storage BYTE-IDENTICAL before/after** (78 files /
  15,281,640 B / 0 tmp; full SHA-256 inventory equal — zero writes,
  zero growth).

## 6. Storage & Integrity

```
files before:  78          files after:  78          delta: 0
bytes before:  15,281,640  bytes after: 15,281,640    delta: 0 B
tmp files:     0 before / 0 after
sha256 inventory: IDENTICAL (78/78, verified twice: by the smoke and
                  post-smoke with the server stopped)
```

M59 is a pure computed view: **0 files, 0 bytes, 0 persistent state
added on every read** — no `best-history.json`, no history database, no
cache, no pointer. `m59_pre.sha256` (78 files) was captured before the
smoke and the final audit is byte-identical. No weights loaded, no
evaluation, no tokenization — metadata-only. Server stopped after the
smoke (0 uvicorn verified).

## 7. API / OpenAPI

**One new route** (the milestone's purpose):
`GET /api/v1/models/{model_id}/checkpoints/best/history` →
`BestCheckpointHistory`. **OpenAPI: 84 → 85** — the only change (no
other routes, no unrelated schema edits). Route declared between
`/checkpoints/best` and the generic `/checkpoints/{checkpoint_id}`
detail so neither "best" nor "best/history" can be captured as a
checkpoint id (display order asserted in tests). Response components
`BestCheckpointHistory` / `BestCheckpointHistoryEntry` added. Error
taxonomy preserved: unknown model → 404; a valid model with no
checkpoints → 200 with an EMPTY history (consistent with the collection
endpoints; the M52 selection route stays 404 for the same state).
Read-only: no PUT/POST/DELETE anywhere in the family.

## 8. Limitations

* **Observation only**: the endpoint exposes the trajectory; it never
  decides. No automatic stopping, no stagnation detection, no
  repetition-count selection, no convergence heuristics — those remain
  future milestones' explicit decisions.
* **Replay semantics**: the timeline is the running argmin under the
  ONE M52 rule over the registry in chronological `(created_at,
  checkpoint_id)` order — the honest history of selection movements. A
  tie moves the selection only when the canonical `(step, created_at)`
  tie-break actually switches it (delta 0.0); equal-loss non-winners
  are invisible by design (they never held the selection).
* **Manifest-authoritative**: losses are read verbatim from persisted
  manifests (never recomputed); a manifest edited after the fact
  changes the computed history retroactively — the same authority model
  as every M18–M50 history surface (the manifests are the immutable
  record; the view is derived). Corrupt (unreadable) manifests are
  skipped by the listing — the established semantics, not a new policy.
* **No dashboard integration** (endpoint-only, per the minimal-change
  rule); the dashboard's existing sections are unchanged.
* **Scale**: one deterministic read pass over the model's checkpoint
  manifests per call (O(n log n) sort); no pagination — consistent with
  the family (23 checkpoints in production today).

## 9. Next Milestone

Inspection of the remaining architecture: the loop is fully declarative
(M53–M57), bounded-repeating (M58) and now observable (M59) — but the
**accept** step of `train → evaluate → compare/gate → select → resume →
evaluate → accept/reject → repeat` is still manual: after a repeated
run the user reads the history and, if satisfied, calls the existing
M3 `POST /models/{id}/rollback` with the best checkpoint id to PUBLISH
it as the model's current state (today the published state is whatever
the last accepted training run left). The smallest high-value next
milestone is **M60 — declarative best-publication stage**: a workflow
PUBLISH stage (or a publish-on-pass gate branch consummating the
canonical loop) that publishes the M52-selected best checkpoint through
the EXISTING M3 rollback machinery, resolved and pinned by the same M53
resolver (`publish_from_best`), so the canonical loop can end with the
improved state actually live. It reuses the one selector, one executor,
existing publication semantics, adds no new storage, and keeps the
accept decision explicit (a stage in the recipe the user wrote — not
automation).

---

### M60 — DECLARATIVE BEST-PUBLICATION STAGE (ready-to-paste prompt)

You are continuing development of **AI Model Forge**.

Implement this milestone incrementally on top of the certified **M59**
state.

## Core objective

Close the ACCEPT gap: after the improvement loop trains better
checkpoints, publishing the best one as the model's current state still
requires a manual `POST /models/{id}/rollback` with a hard-coded
checkpoint id. M60 makes the canonical loop end declaratively:

```text
TRAIN → EVALUATE(best) → GATE(candidate vs best) → TRAIN(resume_from_best)
→ EVALUATE(best) → PUBLISH(best)
```

A workflow PUBLISH stage declares "publish the M52 best checkpoint as
the model's current state", resolved through the SAME M53 resolver and
published through the EXISTING M3 rollback machinery (no second
publication path, no new storage).

## 1. INSPECT FIRST — DO NOT EDIT YET

Run `git status`, `git log --oneline -10`, `pytest -q`. Expected
baseline: **599 tests, OpenAPI 85, worktree clean, M59 implementation
present** (if the environment was reset, follow the proven recovery
procedure: fetch the branch from origin, verify the worktree
byte-identical to the pushed HEAD, rebuild the venv — torch from plain
PyPI — re-certify the baseline, and rebuild production honestly through
the public API if needed, documenting it).

Inspect: `app/training.py` (`rollback` / the M3 publication machinery —
exactly what it writes: model manifest, weights.pt, weights.sha256,
latest_checkpoint), `app/schemas.py` (`WorkflowStage`, the M55/M56/M57
declarative-field patterns, `StageType`), `app/workflows.py`
(`resolve_best_state_refs`, `_execute_stage`, stage-result artifacts,
`ArtifactKind`), `app/recipes.py` (M12 registration, M14 expansion,
M51 preflight), `app/api.py` (the rollback route, the workflow/recipe
routes), the M53–M59 tests and smokes. Do not assume field names.

## 2. DESIGN (follow the M55–M57 pattern exactly)

Likely shape: a new `StageType.PUBLISH` with a payload carrying ONE
declarative selector `publish_from_best: bool` (+ the resolver pin
`resolved_checkpoint_id`, valid only with best) — XOR with an explicit
`checkpoint_id` and with `from_stage` (the final checkpoint of an
earlier train stage), mirroring `WorkflowEvaluationStage`. The resolver
pins the SAME single per-plan M52 selection onto unresolved
best-publications (extend `resolve_best_state_refs` with a
`_publish_unresolved` branch); `_execute_stage` converts the pinned
form to a PURE explicit publication through the EXISTING M3 rollback
machinery (the publish layer never queries "best"). Direct
`POST /models/{id}/rollback` gains NO new field (it already names the
checkpoint explicitly). Validation: exactly one selector; contradictions
→ 422; a publish stage produces a normal stage artifact (decide the
`ArtifactKind` — reuse `TRAINING_REPORT`-style provenance or add the
minimal honest kind) recording the published checkpoint id; plan_hash
pins the resolution exactly like M53–M57. Zero new routes; OpenAPI
stays 85.

## 3. SEMANTICS & SAFETY

Publication is the M3 rollback semantics UNCHANGED (the model's
published weights become the selected checkpoint's; the checkpoint
itself is referenced, never copied; the model manifest/weights/
weights.sha256 are the only writes — the existing-semantics set).
No automatic publication: the stage exists only because the user wrote
it into a recipe/plan. A publish of a checkpoint that is ALREADY the
published state should be a verified no-op or a clean idempotent
re-publication — inspect what M3 rollback does today and preserve it
exactly (test it). Gate interplay: publishing after a passed gate is
just stage ordering (the user's recipe); no new gate semantics.

## 4. TEST MATRIX

Focused M60 tests: schema (selector XOR matrix, pinned form, 422s);
resolver (one selection per plan shared with M55/M56/M57 declarations;
exact M52 match; zero-checkpoint → the established 404, nothing
persisted); execution (the model's published state IS the selected
checkpoint afterwards — weights content hash equals the checkpoint's
`weights_sha256`, `latest_checkpoint` set per M3 semantics; the stage
artifact records the concrete id; provenance pure); idempotence/
already-published behavior (whatever M3 does today, preserved);
immutability (recipe manifest never rewritten; record pins; old records
byte-stable after the best changes; different resolutions → different
plan_hash); M51 preflight parity; M14 composite; the full canonical
recipe ending in PUBLISH(best) executed with M58 repetitions=2 (each
iteration publishes ITS plan-start best — document the timing honestly:
iteration 2's publish may publish a checkpoint iteration 2 itself
created); direct-route unchanged; regression of all M1–M59 tests.
Report exact counts (baseline 599).

## 5. LIVE SMOKE (exactly once)

After the full suite passes twice, inspect production (read the M59
report §6 facts first: 78 files / 15,281,640 B, argmin `41870af53ab7`
@ 6.242401 — note it is NOT the published latest `eaf0984fbc2b`, the
exact drift M60's publish stage closes). Register ONE minimal recipe
ending in `evaluate(best) → publish(best)` (or the full canonical loop
with repetitions=1), execute ONCE, verify: the pin equals the
live-computed argmin; afterwards the model's published state IS that
checkpoint (weights content hash == the checkpoint's
`weights_sha256`, `latest_checkpoint` per M3 semantics); the stage
artifact + record pin; the recipe manifest byte-identical; the
selected checkpoint byte-identical (referenced, never copied); storage
delta exactly the M3 existing-semantics modifications (model manifest,
weights.pt, weights.sha256) + the new workflow/recipe/evaluation
manifests — every file justified; no second publication path. Capture
`m60_pre.sha256` BEFORE the smoke. If an assertion is wrong but the
implementation is correct, follow the M55–M59 discipline: diagnose
read-only, correct the expectation, post-hoc verify, document — never
re-run a state-mutating smoke.

## 6. FINAL QUALITY GATE + REPORT

`pytest` ×2 on the final state, statics (compileall, pyflakes),
OpenAPI count (85 — unchanged), storage audit (files/bytes
before→after, every change justified, untouched files byte-identical),
`git diff`/`git status`/push/verify remote HEAD. Provide the final
report with exactly 9 sections (Scope & Intent; Architecture
Inspection; Implementation; Tests & Verification; Live Smoke; Storage
& Integrity; API / OpenAPI; Limitations; Next Milestone + a complete
ready-to-paste prompt — candidates to evaluate against the ACTUAL
remaining gaps: an improvement-summary field on the M58 batch response
consuming M59; declarative early-stop conditions across repetitions;
suite-run/sampling integration of the loop). Exact numbers only; do not
fabricate. Philosophy: minimum files + minimum storage + maximum
correctness + deterministic behavior + immutable provenance +
same-model iterative improvement + no premature autonomous behavior.
