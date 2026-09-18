# Milestone 70 — Model-Owned Record Retention: Final Report

**Date:** 2026-09-18 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

The four directory-backed model-owned EVIDENCE families — workflow
records, evaluations, comparisons, gate decisions — get explicit
VERIFIED per-record deletion plus read-only retention views, with the
M69 usage overview as the ONE canonical blocker source (no second
scanner anywhere). Guard order, blocker parity, integrity-first
refusals, atomic typed removal and the full dependency-unblock chain
are all proven live. **Live-certified 31/31 on the single execution;
production byte-identical (52/52); every success deletion ran on a
discarded copy.**

---

## 1. Scope & Inspection

**Baseline:** HEAD `6f470cc` (the M69 delta, pushed), worktree clean
(only the untracked `egg-info`); **649 tests** (the M69 delta's two
consecutive full runs); **OpenAPI 97 paths / 7 delete operations**;
production **52 files / 8,926,403 B**, byte-identical to
`m69_pre.sha256` (`m70_pre.sha256` was captured from the same
unchanged state and verified identical per-file). **Inspection (from
the actual code, not names):** the four family engines each carry a
`result_hash` staticmethod (`workflows.py`, `evaluation.py`,
`comparison.py`, `gates.py`) and a directory helper
(`_workflow_dir` / `_eval_dir` / **`_comp_dir`** / `_gate_dir` — the
comparison helper's name differs from the family); the
`atomic_delete_dir` + measure pattern (`DatasetEngine.delete`, reused
by M61/M65/M68) is the removal primitive; the M69 analysis
(`model_records_usage_overview` + its four class constants) is the
ONE blocker source; `_scope_data_artifact` (M65/M67) is the scope
resolver template; every family getter raises `FileNotFoundError`
for unknown/registry-invisible records. **Verified M69 reference
topology reused verbatim** (M70 adds no scanner): comparison sides →
evaluations (persisted `evaluation_id` per side) and checkpoints;
gate sides → evaluations + checkpoints and `comparison_id` →
comparisons; workflow stage artifacts → their run / evaluation /
comparison / gate targets (+ train-stage final checkpoint +
`suggested_checkpoint_id`); suite runs (EXTERNAL) → state checkpoint
+ probe evaluation ids; the M61 lineage edges (checkpoint parents,
run provenance) target checkpoints only and never the four families;
workflows and gates are referenced by NOTHING except workflow stage
artifacts (a gate can be a workflow artifact target — proven live on
production, where both gates are blocked by workflow artifacts). The
M52 best selection targets checkpoints only — no M70 guard needs a
live equivalent (documented). **Hazard found during insertion:** a
method spliced in before an `@staticmethod def result_hash` anchor
lands BETWEEN decorator and def (the decorator then binds to the new
method) — caught by `result_hash() takes 1 positional argument` at
the first engine `run()`, repaired in all four files by re-anchoring
on the decorator+def pair.

## 2. Implementation

**Engine primitives:** each of the four family engines gains
`delete(self, model_id, <record_id>) -> tuple[int, int]` — files +
bytes measured from the directory immediately before ONE
`atomic_delete_dir` (hidden `.tmp-delete-*` sibling rename; registry
scans skip it) — placed before `result_hash`, with `atomic_delete_dir`
imported. **Schemas (4):** `ModelRecordDeletionBlocker` (category,
reference_id, detail), `ModelRecordDeletionResult` (model_id,
category, record_id, files_removed, bytes_reclaimed),
`ModelRecordDeletionBlocked` (message, model_id, category, record_id,
protected, blockers) and `ModelRecordRetentionOverview` (model_id,
category, record_id, created_at, files, size_bytes,
integrity_verified, deletable, blockers) — inserted before
`ModelDeletionBlocker`. **Guard layer (`app/engine.py`, after
`model_records_usage_overview`):** `MODEL_RECORD_DELETABLE_CATEGORIES
= ("workflow", "evaluation", "comparison", "gate")`;
`_deletable_record_scope` (category → engine + record + dir + the
family's `result_hash` check, via `_scope_data_artifact` —
out-of-scope category → `ValueError`, unknown/registry-invisible →
`FileNotFoundError`);
`model_record_deletion_blockers` (EXACTLY the M69 overview's
references for the record filtered to non-lineage categories — same
categories, ids, canonical order);
`_record_blocker_detail` (typed one-line identifying detail per
referencing family, the M67 pattern);
`model_record_retention_overview` (identity, ordered artifact files
+ bytes via the M63-boundary walk `_artifact_files`, the integrity
outcome, `deletable` = integrity AND no blockers);
`delete_model_record` (scope → integrity `RuntimeError` → blockers
`ValueError` → the engine's atomic `delete`). **API:** four DELETE
routes on the EXISTING family resource paths
(`/models/{model_id}/workflows/{workflow_id}`,
`.../evaluations/{eval_id}` (path param is `eval_id`),
`.../comparisons/{comparison_id}`,
`.../gates/decisions/{decision_id}`) + four GET-only retention
routes (`.../retention`), each pair inserted before its family's
GET-one route; blocked deletion → structured 409 detail dict
{message, model_id, category, record_id, protected: true, blockers}
(re-derived through the same blocker function — view == guard),
integrity refusal → plain-string 409; landing bullet + four
route-list lines. **OpenAPI 97 → 101 paths; delete operations 7 →
11.** Count assertions bumped (97 → 101 at 56 sites across 17 test
files) and the four hardcoded delete-path lists regenerated
(`test_model_retention.py`, `test_model_usage.py`,
`test_suite_sample_lifecycle.py` — lists regenerated from the live
spec; `test_model_record_usage.py` — 7 → 11 with the four new paths
in sorted position). Zero new persistent storage; M61/M67/M68 code
paths untouched.

## 3. Tests

**3 new tests (`tests/test_model_record_retention.py`, 695 lines) →
649 → 652 total.** A module fixture builds the full evidence graph
(dataset/tokenizer/model/training/checkpoints, a standalone
evaluation + comparison, a gate with ITS OWN comparison, a workflow
with train+evaluate stages, a suite run with a probe evaluation).
The engine test proves the HEADLINE invariant through a real unblock
chain: for every record, retention blockers == DELETE blockers ==
the M69 non-lineage references == an INDEPENDENT raw-manifest oracle
(ids, categories, order — structurally separate derivation), both
directions; deleting the gate unblocks its comparison, deleting the
comparison unblocks its side evaluations (which the comparison
dedupe-reused), the M68 suite-run deletion unblocks the probe
evaluation externally, the workflow (leaf) deletes freely and
unblocks its stage-artifact targets; the M62 checkpoint retention
loses EXACTLY the deleted evaluation's blocker entry (upstream
shrink); M69 usage agrees at every step; determinism; out-of-scope
categories raise `ValueError`. The edge test proves: unknown
model/record → 404s; a tampered `result_hash` → integrity refusal
with ZERO mutation and full restore; a corrupt manifest →
registry-invisible 404; exact delete stats == an independent walk;
isolation (inventory diff == exactly the record's own files); repeat
deletion → 404; a blocked deletion never mutates (no cascade). The
HTTP test proves the API lifecycle end-to-end: typed 200 results,
the structured 409 shape (all six keys), view == guard, the
gate → comparison → evaluation chain over HTTP (including the
dedupe case: the gate's sides reused the standalone comparison's
evaluations, so that comparison must go first), M69 agreement,
byte-identical determinism, and OpenAPI 101 paths / 11 deletes /
GET+DELETE on the four family paths. **Full suite: 652 passed ×2
consecutive on the final state (exit 0 both; a third confirmation
run also green); pyflakes + compileall clean.** Test-side
corrections during development (all script-side, none touching
implementation semantics): the gates route is `/gates/evaluate`; the
workflow evaluate stage takes `checkpoint_from_best` (no stage-level
`checkpoint_id`); the upstream M62 assert must run AFTER the
evaluation deletion; the HTTP chain needed the standalone comparison
deleted before its shared side evaluation; the four pre-existing
API tests hardcode the delete-path list (7 → 11).

## 4. Live Certification

`smoke_m70_live.py` (port 8796). **Protocol:** the script was
REHEARSED once on a discarded byte-identical COPY of production
(validating the script itself — the rehearsal caught one pure script
bug, a missing `/api/v1` prefix in the OpenAPI path assertions,
fixed; rehearsal 31/31 on the copy, discarded), then executed
against production **EXACTLY ONCE: 31/31 PASSED.** Production phase
(read-only + refusal-only): project/registry coherence; disk ==
`m70_pre.sha256` 52/52; M69 certified counts (30 records, 57
references, 57 internal, 0 external); all 11 retention views —
identity, integrity, files/bytes == the real directories, and the
HEADLINE: blockers == the M69 non-lineage references == the
certified blocker graph (8 blocked: all 4 evaluations — 2 by
comparison+gate sides, 2 by workflow artifacts; both comparisons —
by their gates; both gates — by workflow gate artifacts; 3 leaf
workflows), deletable == blockers empty, byte-identical determinism;
all 8 blocked records DELETE → structured 409 with blockers ==
retention blockers (zero mutation after); the 3 leaf workflows
certified deletable VIEW-ONLY (never deleted on production — the
success path is definitionally certified on the copy); unknown
record/model → 404 for GET and DELETE across all four families;
OpenAPI 101 / 11 deletes / 4 GET-only retention routes / 4 new
DELETE ops / 4 schemas / `ModelRecordDeletionBlocked` 409 $ref;
production byte-identical (52/52). Copy phase (all mutations on ONE
disposable copy, discarded): the copy reproduces every production
retention view exactly; a tampered gate `result_hash` →
integrity_verified False, deletable False, DELETE refused with
`RuntimeError` BEFORE the blocker analysis (integrity first),
restored → True again with zero net mutation; out-of-scope
categories (`checkpoint`, `training_run`, `dataset`) → `ValueError`,
unknown record → `FileNotFoundError`; **THE FULL UNBLOCK CHAIN in
rounds — every round's deletable set == the certified graph
(3 workflows → 2 gates + 2 evaluations → 2 comparisons → the last 2
evaluations), blocked records keep refusing with EXACTLY the view
blockers, 11 atomic deletions with typed results and bytes == the
retention views (11 files / 30,778 B reclaimed)**; the post-chain
M69 view == the pre view minus the 11 records minus every reference
FROM a deleted record (19 records; the four families empty;
training_run 6 / checkpoint 13 untouched); the M62 upstream shrink
live (checkpoint blockers now model-pointer only: 12 deletable /
1 protected, deletable == no non-lineage refs per checkpoint); the
exact storage effect (ONLY the 11 record manifests removed, every
other byte identical); deleted records → retention 404 + repeat
DELETE 404; copy discarded; final production audit byte-identical
(52/52) — verified again externally after the server stopped.
**Production mutation: ZERO.**

## 5. Production Storage

**Before == after, exactly: 52 files / 8,926,403 B, every SHA-256
unchanged against `m70_pre.sha256`** (== `m69_pre.sha256` —
production has not changed since the M68 delta), 0 tmp entries, 0
manifest modifications, 0 new persistent files, 0 servers left
running, 0 leftover copies (rehearsal and certification copies both
discarded and removed). **M70 adds zero persistent storage** — the
retention views and blocker analyses are computed live from the
authoritative records; no dependency indexes, lifecycle journals or
caches exist. The certification's only writes (the 11 deletions, the
integrity tamper + restore) lived and died on the discarded copy.

## 6. Git

- `M70: model-owned record retention & deletion` — the four engine
  `delete` primitives, the four schemas, the engine guard layer
  (scope → integrity → M69 blockers → atomic delete + the retention
  overview), the four DELETE + four retention routes + landing, the
  new test file, the OpenAPI count bumps (97 → 101, 56 sites + the
  four delete-path lists), the README section, the smoke, this
  report.
- `M70: pre-smoke production inventory (m70_pre.sha256)` — the
  production root's certification baseline (52 files;
  byte-identical to the M69 delta inventory).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch
  tip == local HEAD verified; worktree clean (only the untracked
  `egg-info`, never committed). **Delivery gap found and fixed by
  the post-spec audit:** at audit time the two M70 commits existed
  only locally (the remote tip was still the M69 inventory commit —
  the original push evidently never landed); the audit pushed them
  and verified remote == HEAD. The audit also found the README
  test-count lines still at 649 (the suite is 652 across 28 suites)
  and bumped them — the audit delta commit (`e593a22`).

## 7. Limitations

- No bulk deletion, no force flag, no cascade (by design): a blocked
  record is unblocked only by explicitly deleting (or, for suite-run
  probe results, M68-deleting) its referencing records first.
- Training runs remain manifest entries and checkpoints keep their
  M61 guard — both are OUT OF SCOPE by the M69/M70 contract
  (`delete_model_record` raises `ValueError` for any other
  category).
- Production holds no suite runs/samples/measurements, so the
  EXTERNAL blocker directions of M70 (suite-run probe results →
  evaluations) are certified in tests and on the discarded copy
  only; production's certified blocker graph is entirely internal.
- The M52 best selection is computed, not persisted, and therefore
  invisible to M69/M70 — it targets checkpoints only (M61's guard
  keeps its own live check), so no M70 surface is affected.
- The root-level DEFINITION families (workflow recipes, gate
  policies, probe suites) still have no lifecycle of their own —
  they block model deletion (M67) but cannot be deleted
  individually; that is M71.

## 8. M70 Result

**M70 is complete against the spec.** Numbers: tests **649 → 652**
(full suite ×2 consecutive on the final state, exit 0 both; a third
run confirmed); **OpenAPI 97 → 101 paths; delete operations 7 → 11**
(the four M70 DELETEs on existing family resource paths + four
GET-only retention routes); guard order per record: **scope (404) →
integrity-first (409, never deletable) → M69 non-lineage blockers
(structured 409) → ONE atomic removal** with the typed
{model_id, category, record_id, files_removed, bytes_reclaimed}
result; headline invariant proven live both directions:
**retention.blockers == DELETE blockers == M69 non-lineage
references (same categories, ids, order); retention.deletable ==
DELETE succeeds**; production M70 records: **11 (3 workflows, 4
evaluations, 2 comparisons, 2 gates) — 8 blocked / 3 leaves**;
disposable-copy chain: **4 rounds (3 → 4 → 2 → 2), 11 deletions,
11 files / 30,778 B reclaimed, post-chain M69 == 19 records with the
four families empty, M62 shrink 10/3 → 12/1**; live smoke
**31/31 on the single execution** (rehearsal 31/31 on a discarded
copy beforehand); production **52 files / 8,926,403 B unchanged**,
SHA-256 inventory identical (52/52, verified in-smoke and
externally); git: commits `6fc990c` (implementation) + `22f145b`
(inventory) + the audit delta `e593a22` (README counts; the delivery
push gap disclosed in §6) pushed, remote == HEAD verified, worktree
clean. Independently re-verified by the post-spec audit: full suite
652 ×2 (exit 0 both), pyflakes/compileall clean, OpenAPI 101 paths /
11 deletes with the four route pairs, production byte-identical.

## 9. Next Milestone

**M71 — explicit verified retention for the root-level DEFINITION
families** (workflow recipes, gate policies, probe suites): the last
lifecycle gap. Every directory-backed record family now has verified
deletion (datasets/tokenizers M65, models M67, suite runs/samples/
measurements M68, checkpoints M61, the four evidence families M70) —
but the three definition families that BLOCK model deletion (M67)
have no deletion, retention view or usage analysis of their own.
Certified production facts for M71: ONE recipe (`m62-live-loop`,
integrity surface `config_hash`, model binding via stage configs)
referenced by TWO of the three workflows (`recipe_id` provenance,
M12); ZERO policies; ZERO probe suites. The ready-to-paste prompt
follows.

### M71 — EXPLICIT DEFINITION RETENTION (ready-to-paste prompt)

```
Implement **Milestone 71 — Explicit Definition Retention** for
AI Model Forge.

M67 made model deletion refuse while model-bound workflow recipes
or gate policies exist; M68 and M70 gave every directory-backed
RECORD family verified deletion. The three root-level DEFINITION
families — workflow recipes, gate policies, probe suites — are the
last lifecycle gap: they block model deletion, yet cannot be
deleted, inspected for readiness, or pruned individually. M71 gives
them the M70 pattern: explicit verified deletion + read-only
retention views, with per-family blocker analyses that REUSE the
ONE existing listings (no second scanner).

### 1. INSPECT FIRST — DO NOT MODIFY YET

Before changing anything, inspect and identify:

* the M70 guard layer (`MODEL_RECORD_DELETABLE_CATEGORIES`,
  `_deletable_record_scope`, `model_record_deletion_blockers`,
  `model_record_retention_overview`, `delete_model_record`) — the
  template to mirror; the M67/M68/M70 architecture (scope ->
  integrity-first -> blockers -> atomic_delete_dir -> typed
  result/409);
* the three definition engines and their storage layouts:
  `app/recipes.py` (dir prefix, `config_hash` — the M12 content
  hash over the canonical stage JSON — `created_at`,
  `list_workflows_for_recipe`: the recipe-scoped reverse scan over
  workflow manifests), `app/policies.py` (`PolicyDefinition` with
  the persisted `model_id` binding, `list_policies`;
  `list_gate_decisions_for_policy` on the gates engine) and the
  probe-suite side of `app/policies.py` (`ProbeSuite`,
  `list_suites`; suite runs persist `suite_id` — the M9/M10
  by-suite listings are MODEL-scoped, but a suite may serve MANY
  models, so a suite's dependents must be scanned registry-wide
  through the ONE suite-run listing);
* the persisted DEPENDENT edges (the blocker directions):
  `WorkflowRecord.recipe_id` (+ `recipe_hash`, M12 provenance) ->
  recipes; gate decisions' policy reference -> policies; suite runs'
  `suite_id` -> probe suites; and the M67 model-guard's binding
  detections (`_recipe_model_ids`, the policy `model_id`) —
  definitions REFERENCE models (and recipes/policies bind them), so
  definition deletion must NOT require the bound model's existence;
  deleting a definition is exactly what can UNBLOCK an M67 model
  deletion (its live blocker input shrinks — the M68 pattern one
  level up; the M67 guard code stays untouched);
* the integrity surface per family (recipes: `config_hash`;
  policies/suites: inspect what persisted hash/check exists and
  decide the integrity verification per the M61/M65 pattern — if a
  family has NO semantic hash, follow the manifest/sidecar
  convention actually in place and document the decision);
* the existing routes (GET-one at `/workflows/recipes/{recipe_id}`,
  `/policies/{policy_id}`, `/probe-suites/{suite_id}`) and the
  OpenAPI/README/landing conventions (expect 101 paths, 11 deletes,
  652 tests).

Record the pre-implementation baseline (tests, OpenAPI, production
inventory — production currently holds 1 recipe, 0 policies, 0
probe suites).

### 2. M71 OBJECTIVE

Three new explicit, verified, atomic deletions on the EXISTING
definition resource paths:

DELETE /workflows/recipes/{recipe_id}
DELETE /policies/{policy_id}
DELETE /probe-suites/{suite_id}

* guard order per definition: scope (unknown definition -> 404,
  nothing deleted; decide from inspection whether a registry-wide
  dependent scan replaces the model-scoped scope) -> INTEGRITY
  FIRST (the family's persisted hash must reproduce; tampered ->
  409, never deletable, no force) -> blockers -> ONE atomic removal
  (atomic_delete_dir) with the typed
  {<definition id>, category, files_removed, bytes_reclaimed}
  result (decide the exact result schema shape from the M68
  root-level precedent and document it);
* blockers = the persisted DEPENDENTS that would be orphaned:
  workflow records whose `recipe_id` matches (recipes), gate
  decisions referencing the policy (policies), suite runs whose
  `suite_id` matches ANY model (probe suites) — blocker
  ids/categories from the ONE existing listings, canonical order,
  the same nothing-protected-that-is-not-shown invariant;
* blocked deletion -> structured typed 409 with the ordered blocker
  list; zero filesystem mutation on refusal (byte-verified);
* no cascade: deleting a definition never rewrites or removes its
  referencing records; M67 model deletion, M68 suite-run/sample
  deletion and every M70 surface remain byte-for-byte unchanged
  (only their live blocker inputs can shrink);
* read-only retention views per definition
  (GET .../retention x3): identity, artifact files/bytes,
  integrity, created_at, deletable, ordered blockers — the SAME
  list the guard refuses on; retention.deletable == DELETE would
  succeed, both directions.

### 3. TEST MATRIX

Focused tests following the established style, at minimum:

* per-family deletion: success with exact stats == an independent
  walk; only the definition's own directory removed; repeat ->
  404; unknown -> 404; cross-model probe-suite blockers (suite runs
  of TWO different models);
* blocker parity: blocker ids/categories == the ONE listings
  (workflows-for-recipe, by-policy, registry-wide by-suite), both
  directions;
* integrity: tampered hash -> 409 + zero mutation; corrupt
  manifest -> registry-invisible 404; restore-then-delete
  round-trips;
* the unblock chains: deleting the last referencing workflow
  unblocks a recipe; deleting the last recipe/policy can unblock an
  M67 model deletion (its retention view flips deletable — assert
  live); M67/M68/M70 views agree at every step;
* retention/delete invariant both directions; determinism;
* HTTP: typed 200s, typed 409s, 404s, OpenAPI (three DELETE
  operations on existing paths + three retention GET paths:
  101 -> 104; deletes 11 -> 14), zero mutation on refusal;
* full regression: the complete suite must stay green with
  M61-M70 semantics untouched.

### 4. LIVE CERTIFICATION

Production holds exactly ONE definition (the `m62-live-loop`
recipe) and it IS referenced (two workflows carry its `recipe_id`)
— so production certifies: the retention view with the exact
workflow blockers, the DELETE refusal (structured 409, zero
mutation), 404s, OpenAPI. The success paths + the M67 unblock chain
run on a DISPOSABLE COPY (copy production, delete the two
referencing workflows (M70) -> the recipe unblocks -> delete it ->
the M67 model retention view flips accordingly -> exact accounting
-> discard). Run the smoke exactly once (a rehearsal on a discarded
copy first is allowed); on a script assertion bug, stop, verify
read-only, fix, re-execute. Production must remain byte-identical.

### 5. REPORT

Report exactly these 9 sections: Scope & Inspection; Implementation;
Tests; Live Certification; Production Storage; Git; Limitations;
M71 Result; Next Milestone — ending with the next ready-to-paste
implementation prompt.

### HARD CONSTRAINTS

* Inspect first; record the baseline.
* REUSE the ONE listings for every blocker decision; no second
  scanner; no dependency indexes.
* The headline invariant: blockers == the dependents the ONE
  listings report; retention.deletable == DELETE would succeed.
* Integrity before mutation; atomic removal; deterministic typed
  results; unknown -> 404; no force, no bulk, no cascade, no
  automation, no retention policies.
* Definitions REFERENCE models; deleting a definition never
  requires or touches the bound model; M67 guard code unchanged.
* M61-M70 semantics byte-for-byte unchanged; production
  byte-identical; destructive certification only on discarded
  copies.
* Preserve every previous milestone; full suite x2; statics clean;
  git clean and pushed.
```
