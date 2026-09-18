# Milestone 68 — Explicit Verified Suite-Run & Sample Lifecycle: Final Report

**Date:** 2026-09-18 (delta) · initial delivery 2026-09-16 · **Branch:**
`arena/01a071e9-code-forge` · **Status:** COMPLETE (initial delivery +
spec-compliance delta)

M68 gives the root-level runtime records that externally reference
models their own explicit, verified, atomic deletions — which for the
first time lets M67's model blocker surface legitimately shrink when
those records are intentionally removed. Delivered in two phases: the
initial implementation (from the M67 report's ready-to-paste prompt),
then this spec-compliance delta against the authoritative detailed
spec — whose audit found ONE gap (§5 retention views) and closed it.

---

## 1. Scope & Inspection

**Selected runtime families (from repository facts, §2's conditions
A+B+C):** suite runs (`suite-runs/<suite_run_id>/`, record id
`suite_run_id`, model ref `model_id`), samples
(`samples/<model_id>/sample-<id>/`, `sample_id`, `model_id` +
`checkpoint_id`), sample-quality measurements
(`sample-evaluations/<model_id>/evaluation-<id>/`, `evaluation_id`,
`sample_id` + `model_id`). All three persist independently of
`models/<id>/`, are M67 external blockers, have stable manifests and
public GET-one routes. **Excluded by inspection:** workflow recipes
and gate policies (registry-level, immutable by design — never folded
into runtime-record deletion); probe-suite definitions (no model
reference). **Inspected topology (the §3 dependency directions):**
NOTHING persists a `suite_run_id` (leaf) and NOTHING persists a
measurement `evaluation_id` (leaf), while a measurement persists
`sample_id` OUTSIDE the sample's directory — so sample deletion is
blocked by its measurements, and suite runs / measurements are leaves.
Each family has a persisted `result_hash` (sha256 over the semantic
payload) — a real integrity tier for these weights-less records.
**Baselines:** initial delivery — HEAD `26eda92` (M67), 641 tests,
OpenAPI 93, production 52 files / 8,926,410 B (post-4th-re-provision
rebuild, `m68_pre.sha256`); delta — HEAD `4d763fa` (M68), 646 tests
(after initial delivery), OpenAPI 93, production 52 files /
8,926,403 B (post-5th-re-provision rebuild,
`m68_retention_pre.sha256`; see §7). **M67's external blocker
families that can block model deletion (§1.12):** suite_run, sample,
sample_quality, workflow_recipe, policy — M68 makes the first three
deletable; recipes/policies stay immutable by design.

## 2. Implementation

**Initial delivery (commits `c900c12` + `4d763fa`):** three family
low-level `delete()` primitives (measure + `atomic_delete_dir`, the
`DatasetEngine.delete` pattern — no raw rmtree anywhere); facade
guards `delete_suite_run`, `sample_deletion_blockers` +
`delete_sample`, `delete_sample_evaluation` — each in the
M61/M65/M67 order: scope (family getter through
`_scope_data_artifact`: unknown model/record or registry-invisible →
404, nothing deleted) → INTEGRITY FIRST (persisted `result_hash`
must reproduce; tampered → RuntimeError → 409, never deletable, no
force) → blockers (samples only: the ONE M19
`list_sample_evaluations_for_sample` listing → typed 409
`{message, model_id, sample_id, protected, blockers}`) → atomic
removal with typed `{model_id, <record id>, files_removed,
bytes_reclaimed}` results. Three DELETE routes ON EXISTING resource
paths (the repo's model-scoped convention): `DELETE
/models/{id}/suite-runs/{run}`, `DELETE /models/{id}/samples/{sample}`,
`DELETE /models/{id}/sample-quality/{eval}` — zero duplicate paths,
delete-operation set 4 → 7. **No second scanner anywhere:** the
sample blocker list is the ONE M19 listing; the model-side effect
needs NO new analysis (the M67 guard reads the ONE M66 usage view,
which reads the ONE family listings — deletions shrink those
listings live). **Spec-compliance delta (this delivery):** the §5
gap closed — three read-only retention views following the
M65/M67 pattern: `GET /models/{id}/suite-runs/{run}/retention`, `GET
/models/{id}/samples/{sample}/retention`, `GET
/models/{id}/sample-quality/{eval}/retention` (schemas
`SuiteRunRetentionOverview` / `SampleRetentionOverview` /
`SampleEvaluationRetentionOverview`: identity, ordered artifact
files, size_bytes, integrity_verified, deletable, ordered blockers —
computed live, zero storage; engine methods
`suite_run_retention_overview` / `sample_retention_overview` /
`sample_evaluation_retention_overview` reuse the same scope
resolution, `_artifact_files` walk, result-hash check and the ONE M19
listing the guards use). OpenAPI 93 → 96 (three GET-only paths; zero
new DELETE routes in the delta). The M67 guard code is untouched
throughout — only its live input set can shrink.

## 3. Tests

`tests/test_suite_sample_lifecycle.py` (initial: 4 tests → 645 total;
delta: +1 test and HTTP extensions → **646 total**). Initial
coverage: suite-run deletion (stats == independent walk, only its
directory removed, probe evaluations preserved, repeat/unknown →
FileNotFoundError, tampered-hash → RuntimeError, corrupt manifest →
registry-invisible 404, both with pre/post SHA-256 equality and
restore-then-delete round-trips); sample + measurement (typed
blocker == the independent manifest-parse oracle, zero mutation on
refusal, exact stats and isolation, unknowns, tamper/corrupt
refusals); the unblock chains (both chains step-by-step with the
M66/M67 views and the model DELETE agreeing at every step, ending in
exact model-deletion stats); the full HTTP lifecycle (fixture through
public routes, typed sample 409, model 409 listing all three blockers
in canonical order, the three deletions, the model then deleting,
404s + deterministic repeats, OpenAPI). Delta coverage (§8.D
retention/delete invariant): the new engine test proves shapes,
leaves-deletable, blocked-sample blockers == the guard's,
artifact accounting == independent walks, the §5 invariant in BOTH
directions (blocked → DELETE 409s with identical blockers and zero
mutation; deletable → DELETE succeeds with stats == the view),
tampered-sample integrity (integrity_verified False → deletable False
→ RuntimeError; restored → blocked again), live flips (measurement
deleted → sample view flips → deletes with stats == the flipped
view), determinism, zero mutation, unknown 404s, and the full chain
to a drained model (usage external 0 → retention deletable → DELETE
stats == view); the HTTP test now exercises the three retention
routes (shapes, 404s, byte-identical repeats, zero mutation, 409
blockers == the view's blockers, the live flip before the sample
DELETE, DELETE stats == the view's accounting, OpenAPI 96 with three
GET-only retention routes + schemas). Updated tests (intentional
contract changes, documented in place): M66/M67 delete-set assertions
4 → 7 paths; the two M15/M16-era "DELETE → 405" assertions now assert
the verified-deletion contract; OpenAPI count assertions 93 → 96
(54 sites). **Full suite: 645 ×2 (initial) and 646 ×2 (delta), exit 0
all four runs**; pyflakes + compileall clean; OpenAPI
standalone-verified (96 paths, 7 delete operations, three GET-only
retention routes). One pre-existing flake was fixed WITH cause
analysis during the initial delivery (a sampling test's tolerance
compared a 6-decimal-rounded persisted value against full-precision
float32 at 1e-6 — made like-for-like; the semantic claim unchanged).

## 4. Live Certification

**Initial delivery — `smoke_m68_live.py` (port 8791):** first
execution **21/24** — C4–C6 failed on a pure script assertion bug
(the independent `walk_stats(dir)` was evaluated AFTER each deletion
had removed the directory, yielding (0,0)); protocol followed:
stopped, verified read-only that production was byte-identical
(52/52, zero mutations — the only production DELETE was the M67
refusal 409), confirmed the copy was discarded, fixed the evaluation
order, re-executed: **24/24 PASSED** (production read-only phase:
coherence + M66/M67 surfaces + unknown-id 404s + the M67 model
DELETE still 409 on the recipe + determinism + zero mutation; copy
phase: a fresh model blocked by all three families in canonical
order, the blocked sample 409, the three deletions with exact stats,
the probe evaluation PRESERVED — no cascade —, the model then
unblocked and deleted with stats == its retention view, full-circle
copy inventory proof, copy discarded). **Delta —
`smoke_m68_retention_live.py` (port 8792):** **21/21 PASSED on the
single first execution, zero corrections** — production read-only
(disk == `m68_retention_pre.sha256` 52/52; M66 usage == certified
counts == the independent manifest oracle; M67 retention blocked by
the recipe only; the three retention routes' unknown-id 404s —
production holds 0 runtime records; deterministic byte-identical
repeats; OpenAPI 96 / 7 deletes / three GET-only retention routes;
zero mutation), then the disposable-copy §5 invariant (fresh model
blocked by all three records; leaves deletable + stats ==
independent walks; sample blocked with view blockers == guard
blockers; DELETE refuses with zero mutation; measurement → sample
view flips live → sample → suite run each deleting with stats == its
view; the model then unblocked and deleted with stats == its
retention view; the production model on the copy untouched and still
blocked; copy discarded). Production DELETEs attempted: two (initial,
both refusals) + zero (delta — the delta smoke issues no production
DELETE at all); every mutating deletion ran on discarded copies.

## 5. Production Storage

**Production before == after, exactly, in BOTH phases.** Initial:
52 files / 8,926,410 B, every SHA-256 unchanged against
`m68_pre.sha256` across both smoke executions. Delta: 52 files /
8,926,403 B, every SHA-256 unchanged against
`m68_retention_pre.sha256` (verified before the smoke, after all
read-only checks, and in the final independent audit; 0 tmp entries;
0 manifest modifications; 0 servers left running; 0 leftover copies).
**Environment events (disclosed):** the sandbox re-provisioned a
FOURTH time during the initial delivery and a FIFTH time before this
delta; each recovery followed the proven protocol (repo reset to the
remote tip, venv rebuilt, production root rebuilt via
`m62_production_rebuild.py` — SAME certified structure, NEW
content-derived ids; the prior root's exact bytes are unrecoverable
by design, exactly as in every previous re-provision). **The
disposable-copy accounting (§19):** every deletion's `files_removed`
== its independent pre-walk == its retention view's files, and
`bytes_reclaimed` == the byte sum of exactly those files — each
runtime record is a single-manifest directory (1 file each:
measurement, sample, suite run), and the model deletions' stats
equaled their retention views' files/bytes exactly (the delta
additionally proves view == walk == DELETE for all three families and
the model). No unrelated artifact changed on any copy (full-inventory
diffs); no persistent retention/dependency record was created
anywhere (all analysis is live-computed — §12's banned list honored).

## 6. Git

- Initial: `c900c12` "M68: explicit suite-run & sample lifecycle" (14
  files: engine/schemas/api/suite_runs/sampling/sample_quality, the
  new test file, M66/M67 test updates, README, smoke, report) +
  `4d763fa` "M68: pre-smoke production inventory (m68_pre.sha256)";
  both pushed and verified (remote tip == HEAD at the time).
- Delta: `M68: retention views (spec compliance)` (schemas/engine/api
  retention views + landing/README, the delta test + OpenAPI count
  bumps 93 → 96, the delta smoke, this rewritten report) +
  `M68: delta pre-smoke production inventory (m68_retention_pre.sha256)`.
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch
  tip == local HEAD verified after push; worktree clean (only the
  untracked `egg-info`, never committed).

## 7. Limitations

- Production contains NO suite runs / samples / measurements (M66
  certified counts 0/0/0), so their production-exercisable surface is
  the unknown-id 404 path; the success paths are certified on
  discarded copies (fixtures cover the rest).
- Recipes, gate policies and probe-suite definitions remain
  undeletable BY DESIGN (immutable registries): a model blocked ONLY
  by a recipe or policy can never become deletable through existing
  operations.
- Suite-run deletion leaves its probe evaluations in place
  intentionally (model-owned history — pruning those individually is
  M69/M70 territory).
- Sample-quality measurement retention views carry no `created_at`
  (the record schema has none) — identity is model/sample/evaluation
  ids.
- No cascade, no force, no bulk, no policies, no automation — by
  scope.

## 8. M68 Result

**M68 is complete against the authoritative spec.** Numbers: tests
641 (M67 baseline) → **645** (initial) → **646** (delta), full suite
×2 green in each phase (exit 0); OpenAPI **93 → 96** paths (initial:
3 new DELETE operations on existing resource paths, 0 new paths;
delta: 3 new GET-only retention paths); **7 delete operations**
total (4 pre-existing + M68's three); **0 duplicate DELETE paths**;
selected families: **suite runs, samples, sample-quality
measurements**; blocker categories: **sample_quality** (samples
only; suite runs and measurements are leaves with empty blocker
lists); new facade/engine operations: 3 deletes + 3 blocker/retention
analyses + 3 retention overviews; the §7 M67 cross-milestone
invariant proven live end-to-end (before: model external references
contain the runtime record; after explicit deletion: they do not,
and every remaining reference still blocks; all external references
removed → the model deletes, with the M67 guard code untouched);
independent oracles agree (stats, blocker ids, usage, retention
views); production SHA-256 inventories identical before/after in
both phases; all destructive certification on discarded copies; git
clean and pushed. Transient events, handled per protocol: the
initial smoke's first execution failed 3 checks on a script
assertion bug (post-deletion stat evaluation) — stopped, verified
zero mutation read-only, fixed, re-executed 24/24; one pre-existing
test flake fixed with cause analysis; the delta smoke passed 21/21
first-run with zero corrections. Five sandbox re-provisions total
across the session, each recovered with the established protocol and
disclosed.

## 9. Next Milestone

**M69 — model-owned record usage overview** (read-only): the
M66-pattern dependency overview one level down — what references a
model's OWNED internal records (evaluations, comparisons, gate
decisions, workflow records), which today can only be deleted with
the whole model. M69 is the discovery surface; M70 would then add
explicit per-record deletion with internal reference safety. The
ready-to-paste prompt follows.

### M69 — MODEL-OWNED RECORD USAGE OVERVIEW (ready-to-paste prompt)

```
Implement **Milestone 69 — Model-Owned Record Usage Overview** for AI
Model Forge.

M61-M68 completed explicit verified deletion for datasets, tokenizers,
checkpoints, models and every root-level runtime record. The remaining
undeletable records are the model-OWNED internal families —
evaluations, comparisons, gate decisions, workflow records — which are
removed atomically WITH the model (M67 ownership) but can never be
pruned individually. M69 is the read-only discovery step for that
(the M66 pattern one level down); deletion itself is M70.

### 1. INSPECT FIRST — DO NOT MODIFY YET

Before changing anything, inspect and identify:

* the four record families' persistence: EvaluationRecord /
  ComparisonRecord / GateDecision / WorkflowRecord — storage layout
  (models/<id>/{evaluations,comparisons,gates,workflows}/), fields,
  loaders, listings (including every by-* filter);
* EVERY persisted cross-reference INTO and OUT OF these records:
  - gate decisions referencing comparisons (by-comparison gates) and
    embedding policies;
  - suite-run records persisting probe evaluation_ids (root-level
    references INTO model-owned evaluations);
  - workflow records embedding stage configs, checkpoint refs and
    stage outcome records (training runs, evaluations, gates created
    BY the workflow);
  - the model manifest's latest_checkpoint / best_checkpoint pointers
    and training_provenance (what the MODEL itself references);
  - checkpoint references from comparisons/gates/samples/suite-runs
    (already covered by M61/M62 — do not duplicate, just understand);
* the M64/M66 usage-analysis architecture and the M62 checkpoint
  retention analysis (the reuse precedents);
* the OpenAPI conventions (expect 93 paths, 645 tests) and the
  README/landing conventions.

Record the pre-implementation baseline. Enumerate the ACTUAL reference
topology before designing anything; do not invent categories.

### 2. M69 OBJECTIVE

Read-only usage/dependency overviews for ONE model's internal records,
computed live from the ONE authoritative listings (never a second
scanner):

GET /models/{model_id}/evaluations/{eval_id}/usage
GET /models/{model_id}/comparisons/{comparison_id}/usage
GET /models/{model_id}/gates/{decision_id}/usage
GET /models/{model_id}/workflows/{workflow_id}/usage

Each answers: "what persisted records reference THIS record?" — with
ordered categories from the ACTUAL inspected topology, sorted
reference ids, deterministic counts, and a `referenced` /
total summary. Unknown or registry-invisible record -> the family's
404. A fresh record with no references -> valid empty usage. GET never
mutates. Lightweight reference summaries only — never duplicate full
records.

### 3. DESIGN CONSTRAINTS

* REUSE the ONE listings and existing by-* filters wherever they
  exist; derive cross-references from the AUTHORITATIVE records (e.g.
  a gate's persisted comparison_id), not from new scans;
* the facade is the public entry point; the API stays a thin adapter;
* zero new persistent files, zero mutation, byte-identical repeats;
* do NOT implement any deletion in M69 (M70), and do NOT touch any
  existing guard (M61/M65/M67/M68 semantics byte-identical);
* document the category order deterministically per family.

### 4. TEST MATRIX

* per family: known record returns usage; unknown -> 404; empty case;
  deterministic repeats; zero mutation (SHA-256 inventories);
* category coverage per the INSPECTED topology (e.g. a by-comparison
  gate appears in its comparison's usage; a suite run's probe
  evaluation appears in the evaluation's usage; a workflow's created
  records appear in the workflow's usage — whatever inspection proves
  real);
* INDEPENDENT manifest-parsing oracle agreeing on every category;
* isolation: other models'/records' references never leak;
* HTTP: shapes, 404s, determinism, OpenAPI (93 -> 97 if all four
  routes are new), README + landing updates; full regression green.

### 5. LIVE CERTIFICATION

Production read-only: the existing model's 4 evaluations / 2
comparisons / 2 gates / 3 workflows have REAL references (suite-run
probe ids are absent — suite_run count is 0 — but the inspected
internal topology applies); verify usage views against the independent
oracle and the listings; determinism; zero mutation; production
byte-identical. No destructive paths exist in M69. Run the smoke
exactly once; fix script bugs read-only and rerun only after verifying
zero mutation.

### 6. REPORT

Report exactly these 9 sections: Inspection & Baseline; Record
Reference Topology; API; Determinism & Reuse; Verification; Live
Certification; Storage Proof; Git; Final Status & Next Milestone —
then provide the next ready-to-paste implementation prompt.

### HARD CONSTRAINTS

* Inspect first; record the baseline; no invented categories.
* Reuse the ONE listings; no second scanner; no dependency indexes.
* Read-only: zero production writes; GET only.
* No deletion, no guards changed, no cascade, no automation.
* Preserve every previous milestone; full suite ×2; statics clean;
  production byte-identical; git clean and pushed.
```
