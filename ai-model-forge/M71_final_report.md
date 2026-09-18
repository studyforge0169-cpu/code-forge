# Milestone 71 — Definition Retention: Final Report

## 1. Scope & Inspection

M71 closes the LAST lifecycle gap: explicit verified retention +
deletion for the three ROOT-LEVEL DEFINITION families — workflow
recipes (`workflow-recipes/<id>/`), gate policies (`policies/<id>/`)
and probe suites (`probe-suites/<id>/`) — plus a read-only
deletion-readiness view per definition. Inspection (read-only,
before any edit) proved the topology:

* **Storage shape**: ONE `manifest.json` per definition directory;
  registries are live directory scans with NO global index — an
  atomic self-contained deletion fits the existing architecture
  exactly (the §6 pattern).
* **Dependent directions (all proven from persisted fields)**:
  `WorkflowRecord.recipe_id` (+`composition`) → recipes (2 of the 3
  production workflow runs carry the recipe's provenance;
  `8f022ff69681` is ad-hoc); `GateDecision.policy_id` (None for
  inline policies) → policies; `SuiteRunRecord.suite_id` → suites;
  composite recipes' `composition` → recipes. **The two open
  inspection items from the plan resolved**: (a) recipes DO
  structurally persist `policy_id` — a `WorkflowGateStage` carries
  exactly one policy source, inline `policy` XOR registry
  `policy_id` — and `WorkflowSuiteRunStage` persists `suite_id`, so
  RECIPES are dependents of policies and suites (deleting such a
  definition would break every future run of the recipe at
  resolution); (b) the M66 external category vocabulary is
  `workflow_recipe` + `policy` (probe suites bind models only at
  RUN time — `model_ids` is always empty), and the M64/M65 dataset
  and tokenizer usage surfaces count suite RUNS, never suite
  DEFINITIONS — a suite deletion changes no dataset blocker.
* **Executed-run coverage is transitive**: every registry-policy
  gate execution persists a `GateDecision.policy_id`, and every
  suite-run stage execution persists a `SuiteRunRecord.suite_id`,
  so workflow runs need no separate scan — the canonical
  `list_workflows_for_recipe` (M35), `list_gate_decisions_for_policy`
  (M23) and `list_suite_runs_for_suite` (M21) filters, iterated
  across every registered model (they are model-scoped), ARE the
  blocker source. No second scanner anywhere.
* **Baseline before editing**: HEAD `93258c1` == remote tip (§13
  clean, no unpushed commits); 652 tests ×2 exit 0; pyflakes/
  compileall 0; OpenAPI 101 paths / 11 deletes; production 52 files
  / 8,926,403 B byte-identical to `m70_pre.sha256` (== `m69_pre`).

**Pre-existing bug found and fixed** (exposed by M71's fixture, not
introduced by it): `_recipe_model_ids` — the ONE M66 analysis —
crashed with `AttributeError` on any recipe whose gate stage uses a
REGISTRY `policy_id` (`stage.gate.policy` is None in that form).
Fixed by resolving the registry policy's target model through the
ONE policy registry (staticmethod → instance method; all call sites
already bound); a legacy dangling id is skipped defensively.

## 2. Implementation

The M70 guard pattern lifted to definitions — `app/schemas.py`
(4 new models), `app/recipes.py` + `app/policies.py` (3 delete
primitives), `app/engine.py` (guard layer), `app/api.py` (6 routes
+ landing):

* **Schemas**: `DefinitionDeletionBlocker` (`category` ∈ workflow /
  workflow_recipe / gate / suite_run, `reference_id`, `detail`),
  `DefinitionDeletionResult` (`family`, `definition_id`,
  `files_removed`, `bytes_reclaimed`), `DefinitionDeletionBlocked`
  (structured 409: `message`, `family`, `definition_id`,
  `protected: true`, `blockers`), `DefinitionRetentionOverview`
  (`family`, `definition_id`, `created_at`, `model_ids`, `files`,
  `size_bytes`, `integrity_verified`, `deletable`, `blockers`).
* **Engine primitives** (the `DatasetEngine.delete` pattern): 
  `RecipeEngine.delete`, `PolicyEngine.delete_policy`,
  `PolicyEngine.delete_suite` — measure files/bytes, then
  `atomic_delete_dir` (ONE rename to a hidden `.tmp-delete-*`
  sibling the registry scans skip); the facade owns the safety
  decision.
* **Engine guard layer**: `_definition_scope` (family getter
  through the `_scope_data_artifact` convention — unknown OR
  registry-invisible/unparseable manifest → FileNotFoundError/404,
  nothing deleted — plus the family's OWN content-hash check:
  recipes `config_hash` (composites over stages + their recorded
  composition references), policies `policy_config_hash`, suites
  `probes_hash`); `definition_model_ids` (the ONE M66 analysis
  reused); `definition_deletion_blockers` (the ONE canonical
  filters across all models + the recipe-registry structural
  scan; ordered by category then reference id);
  `definition_retention_overview` (read-only, live, zero storage);
  `delete_definition` — guard order identical to
  M61/M65/M67/M68/M70: **scope → INTEGRITY FIRST (tampered →
  RuntimeError → 409, never deletable, no force flag) → blockers
  (any reference → ValueError with the ordered typed list) → ONE
  atomic removal** with exact files/bytes. No cascade, no force,
  no bulk, no automatic cleanup.
* **API**: `DELETE /api/v1/workflows/recipes/{recipe_id}`,
  `DELETE /api/v1/policies/{policy_id}`,
  `DELETE /api/v1/probe-suites/{suite_id}` — new DELETE OPERATIONS
  on the EXISTING GET-one paths — plus the GET-only
  `.../retention` views; 404 scope / 409 structured (integrity
  refusals keep the plain-string detail); landing page bullet +
  route-list lines updated.

## 3. Tests

`tests/test_definition_retention.py` (NEW, 4 tests, 872 lines):

* **One blocker source**: guard blockers == expected canonical
  lists == an independent RAW-MANIFEST oracle (parses workflow
  manifests' `recipe_id`, recipes' `composition` + gate
  `policy_id` + suite-run `suite_id` stage configs, gate decisions'
  `policy_id`, suite runs' `suite_id`); retention view shows the
  SAME list; `deletable == not blockers`; files/bytes == an
  independent walk; blocked engine DELETEs refuse AND the whole
  read-only pass leaves the root byte-identical.
* **§7 model-unblock + §8 chains**: the M66 `workflow_recipe` /
  `policy` categories shrink by EXACTLY the deleted definition (a
  pre/post snapshot proves no other reference vanishes); the
  final-ref model (untrained, ONE policy as its only external
  reference) flips the UNTOUCHED M67 retention view from blocked
  to `deletable: true`; recipe chain (composite leaf → workflow
  run → base recipe), policy chain (gate decision → structural
  recipe → policy), suite chain (cross-model suite runs on TWO
  models → M68 deletions → structural recipe → suite); dataset
  blockers unchanged by the suite deletion; leaf definitions
  delete cleanly; model/dataset/tokenizer survive (no cascade).
* **Integrity/scope/atomicity** (disposable root): tampered recipe
  stage seed / policy tolerance / probe set → `integrity_verified:
  false`, `deletable: false` even with zero dependents, DELETE →
  RuntimeError, disk unchanged; unparseable manifest → 404
  (registry-invisible); a leaf deletion removes EXACTLY its own
  files (inventory delta proof, every survivor byte-identical).
* **API lifecycle**: register → run → retention (blocked, exact
  shape) → DELETE 409 structured (same blockers) → M70 record
  delete → DELETE 200 typed → 404 after; policy + suite leaf
  lifecycles; API-level integrity refusal (tamper via the bound
  forge root, then RESTORED and deleted — zero residue in the
  shared session root); OpenAPI: **104 paths / 14 deletes**, the
  three new GET-only retention routes, `{get, delete}` on the
  three family paths, the four Definition schemas, the 409 `$ref`.

**Pre-existing expectations updated to the M71 surface** (the spec
puts DELETE on these existing routes): 56 `== 101` path-count
sites → `== 104` across 17 files; explicit expected-DELETE lists in
`test_model_usage.py`, `test_model_retention.py`,
`test_suite_sample_lifecycle.py` and count asserts in
`test_model_record_usage.py` / `test_model_record_retention.py`
(11 → 14); `test_policies.py` (DELETE-405 → exercised M71 leaf
deletion 200 + 404 after) and `test_workflow_recipes.py`
(DELETE-405 → scope-404 before the runs, blocked-409 with the two
run blockers after them). Full suite: **652 → 656 tests, 28 → 29
suites, 656 ×2 green on the final state** (the second full-suite
runs initially exposed three cross-test interactions — two more
hardcoded delete lists and the tampered-recipe residue — all
diagnosed read-only and fixed; final state green twice).

## 4. Live Certification

`smoke_m71_live.py` (562 lines, 28 checks) — **rehearsal 28/28 on
a discarded byte-identical copy FIRST** (catching four pure script
bugs: blocker `reference_id`s are bare workflow ids, not directory
names — twice; the OpenAPI path parameters are `{recipe_id}`/
`{policy_id}`/`{suite_id}`, not `{definition_id}`; `pairs()` must
accept engine pydantic blockers; `atomic_delete_dir` leaves the
empty family dir, so "empty" not "absent"). Then the **live run
EXACTLY ONCE against production: 28/28**.

* **Production (read-only + ONE guaranteed-409 DELETE)**: project/
  registry facts; disk == `m71_pre.sha256` (52/52); 1 recipe / 0
  policies / 0 suites; the recipe's retention view — identity,
  integrity, 1 file / 3,685 B, `model_ids == [a266c8480cb7]`,
  blockers == the two certified workflow runs, `deletable: false`;
  deterministic byte-identical repeat; M66 coupling
  (`workflow_recipe == [m62-live-loop]`, `policy == []`) and M67
  blocked by exactly the recipe; **`DELETE /workflows/recipes/
  m62-live-loop` → 409** with the structured body whose blockers ==
  the view's; zero mutation after the refusal; unknown ids → 404
  (all three families, GET + DELETE); OpenAPI 104/14 + routes +
  schemas + `$ref`; final byte-identity.
* **Disposable copy (all success paths)**: the copy reproduces the
  production view exactly; integrity tamper → refused → restored;
  the recipe chain (2 M70 record deletions → atomic recipe DELETE,
  1 file / 3,685 B) with the **§7 proof LIVE** — the M66
  `workflow_recipe` category empties, nothing else vanishes, and
  the model's M67 retention flips to `deletable: true` with ZERO
  M67 code changes; the policy chain (registered + gate run →
  blocked by the decision → M70 deletion → atomic DELETE; the M66
  `policy` category grows and shrinks live); the suite chain
  (registered + run → blocked by the suite run → M68 deletion →
  atomic DELETE; NO dataset blocker change from the suite
  deletion); the STRUCTURAL directions (recipes' gate/suite-run
  stages block with no evidence records; recipe deletion
  unblocks); the composite direction; final copy delta exactly the
  intended removals (recipe + 2 workflow records) plus the
  exercise artifacts (gate/suite run/evals), definition families
  empty at the end.
* **Production after the smoke: 52/52 byte-identical, `tmp/`
  empty — ZERO production mutation.**

## 5. Production Storage

Unchanged: **52 files / 8,926,403 B**, byte-identical to
`m71_pre.sha256` == `m70_pre.sha256` (captured at the repo root
before the smoke, committed). Production definitions: 1 recipe
(`m62-live-loop`), 0 policies, 0 suites — the policy/suite
families were exercised ONLY on the disposable copy. The one
production DELETE was the guaranteed-409 protected refusal.

## 6. Git

Local commits on `arena/01a071e9-code-forge` (work tree was clean
at the turn start; §13 check: HEAD `93258c1` == remote tip before
editing — no pre-existing unpushed commits):

* implementation commit — schemas/engine/recipes/policies/api,
  `tests/test_definition_retention.py`, the 24 updated test files,
  README (656 tests / 29 suites; the M71 section),
  `smoke_m71_live.py`, this report;
* inventory commit — `m71_pre.sha256` (52/52, == `m70_pre`).

**DISCLOSURE — the push did NOT land.** At push time the sandbox's
GitHub authentication is broken: `gh auth status` reports "The
github.com token in GH_TOKEN is no longer valid" and
`git ls-remote origin` fails with "could not read Username for
'https://github.com': terminal prompts disabled". Both commits are
committed locally and verified (`git log` below); `git push origin
arena/01a071e9-code-forge` MUST be re-run once GitHub is
reconnected in Arena, and remote HEAD == local HEAD verified
(§13). This is reported per the standing rule: never claim pushed
when auth fails.

## 7. Limitations

* Suite definitions are NOT dataset/tokenizer usage categories
  (M64 counts suite RUNS only) — a suite whose probes name a
  dataset does not block that dataset's deletion, by pre-existing
  M64 design; M71 documents this asymmetry rather than changing
  M64 (no scope creep). The M71 suite guard fully protects the
  SUITE side (suite runs block suite deletion).
* A tampered/corrupt definition is permanently undeletable
  (integrity-first, no force flag) — recovery is a manual
  out-of-band filesystem operation, unchanged from M61 onward.
* Deleting a recipe/policy does not (and must not) touch the
  workflow runs / gate decisions that carry its provenance —
  those records keep their historical `recipe_id`/`policy_id`
  pointing at a now-unknown id; all readers already treat unknown
  definition ids as clean 404s at resolution time (the M35/M23
  semantics). This is the documented evidence-vs-definition split.
* No cascade, no force, no bulk, no background cleanup, no
  retention policies/TTLs/quotas (scope bans all honored).

## 8. M71 Result

**Delivered and certified**: explicit verified definition retention
for the three root-level families, completing the deletion
lifecycle matrix — every persistable family in the system now has
either a verified deletion (models M67, datasets/tokenizers M65,
training runs/checkpoints M68, suite runs/samples/sample-quality
M68, records M70, definitions M71) or is intentionally
undeletable-by-design (M61 lineage). Certified numbers: tests
**652 → 656** (4 new, `test_definition_retention.py`), suites
**28 → 29**, OpenAPI **101 → 104 paths / 11 → 14 deletes**; full
suite **656 ×2 green**; live smoke **28/28 once** (rehearsal
28/28 first, 4 script bugs fixed read-only); production **52/52
byte-identical**, zero mutation, one guaranteed-409 protected
DELETE exercised; disposable-copy chains proved §7 (M66 shrink +
M67 flip with zero M67 changes) and §8 (records, suite runs,
structural recipe references and composites as real blockers, all
unblockable through their own guards). One pre-existing M66 bug
(`_recipe_model_ids` on registry-policy gate stages) found and
fixed. The push did not land (auth failure — see §6); both commits
await re-push.

## 9. Next Milestone

**M72 — read-only PROJECT retention inventory**
(`GET /project/retention`): ONE live-computed, zero-storage,
zero-mutation overview of the WHOLE deletion surface — per family
(models, datasets, tokenizers, workflow recipes, gate policies,
probe suites, and the model-owned record families via the ONE M69
overview): counts, total bytes, blocked vs deletable counts, and
the aggregate `files`/`bytes_reclaimable` if every currently
deletable artifact were deleted — every number derived from the
EXISTING M62/M65/M66/M67/M68/M69/M70/M71 analyses (no second
scanner), with the canonical family order and the established
scope conventions (unknown/registry-invisible project state →
404). Read-only by design (the M64→M65 / M66→M67 / M69→M70
rhythm: the capstone view after the mutation milestone), full
suite ×2, live smoke once (production is read-only for this), 9
sections, and the inventory `m72_pre.sha256` == `m71_pre.sha256`
(52/52) proving zero mutation.
