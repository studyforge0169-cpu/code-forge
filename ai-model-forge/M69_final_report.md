# Milestone 69 — Model-Owned Record Usage Overview: Final Report

**Date:** 2026-09-18 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

`GET /models/{model_id}/records/usage` answers, read-only and
live-computed: "what persisted records currently reference each
model-OWNED record?" — the complete reference topology one level below
the M66 model usage, with the internal/external split and the M61
lineage classification that together make the future per-record
blocker surface obvious. **Live-certified 20/20; production
byte-identical; M69 is strictly read-only — no deletion logic was
added or changed.**

---

## 1. Scope & Inspection

**Baseline:** HEAD `d226fc4` (the M68 delta, pushed), worktree clean
(only the untracked `egg-info`); **646 tests** (the delta's two
consecutive full runs); **OpenAPI 96 paths**; production **52 files /
8,926,403 B**, byte-identical to `m68_retention_pre.sha256`
(`m69_pre.sha256` was captured from the same unchanged state).
**Topology inspection (from the actual persisted schemas, not
names):** the six model-owned families and every persisted reference
edge TO them — checkpoints persist `run_id` (→ training runs) and
`parent_checkpoint_id` (checkpoint→checkpoint lineage); evaluations
persist `checkpoint_id`; comparisons persist BOTH sides'
`checkpoint_id` AND `evaluation_id` (a `ComparisonSide` always carries
its backing evaluation id); gate decisions persist candidate/baseline
sides (checkpoint + evaluation ids) and `comparison_id`; workflow
records persist per-stage `WorkflowArtifact`s (`kind` ∈
training_report/evaluation/comparison/gate_decision → the underlying
record id, plus train-stage final `checkpoint_id`) and
`suggested_checkpoint_id`; the model manifest persists
`latest_checkpoint`/`best_checkpoint` and run-provenance pointers
(`parent/initial/final/rolled_back_to`); ROOT-LEVEL external
referencers: suite runs (`state.checkpoint_id` + per-probe
`evaluation_id`s), samples (`checkpoint_id`) and sample-quality
measurements (`checkpoint_id` — persisted directly, not just via the
sample). **The M61 classification preserved:** checkpoint-parent and
run-provenance edges are informational lineage that M61 deletion
deliberately never blocks on (nothing loads state through them); the
one M61 blocker that is computed rather than persisted (the live M52
best selection) is documented as invisible to M69. `parent_model_id`
remains a dead field (no setter — re-verified). Existing canonical
filters inspected (`list_checkpoints_for_run`,
`list_evaluations_for_checkpoint`, `list_comparisons_for_checkpoint`,
`list_gate_decisions_for_comparison`,
`list_suite_runs_for_checkpoint`, `list_samples_for_checkpoint`) —
M69's ONE analysis iterates the same ONE listings; no second scanner.

## 2. Implementation

**ONE canonical analysis** — `ModelForge.model_records_usage_overview`
(+ the class constants `MODEL_RECORD_CATEGORIES` — the six families in
the M66 internal order — `RECORD_REFERENCE_CATEGORIES` — the ten
referencing families in canonical order: model, checkpoint,
run_provenance, workflow, evaluation, comparison, gate, suite_run,
sample, sample_quality — `RECORD_REFERENCE_EXTERNAL` — the three
root-level families — and `RECORD_REFERENCE_LINEAGE` — the two M61
lineage edge kinds). The analysis iterates the ONE authoritative
listings once, builds the reverse-reference map (unique (category,
reference-id) pairs per record — a record referencing the same target
through several fields counts once), and assembles per-family
categories with references sorted by canonical category order then id,
records sorted by id, and exact per-record + aggregate
internal/external splits. **Schemas:** `RecordReference` (category,
reference_id, external), `ModelOwnedRecordUsage` (record_id,
references, total/internal/external counts), `ModelOwnedRecordCategory`
(category, records), `ModelRecordsUsageOverview` (model_id, name,
total_records, total_references, internal_references,
external_references, categories). **API:** ONE new read-only route
`GET /models/{model_id}/records/usage` (thin adapter; unknown or
registry-invisible model → 404) + landing bullet + route-list line.
**OpenAPI 96 → 97** (54 count assertions bumped); the delete-operation
set is unchanged (7). Zero new persistent storage — the overview is
computed live (no snapshots, no indexes, no caches).

## 3. Tests

**3 new tests (`tests/test_model_record_usage.py`) → 646 → 649
total.** The engine test proves: canonical category order; record sets
== the ONE public listings (the §6 invariant); EVERY record's
references == an INDEPENDENT raw-manifest oracle (ids, categories,
order, counts — structurally separate derivation); exact counts as
sums; external flags correct; the fixture's exact topology spot-checks
(suite-run state checkpoint + probe evaluation, sample +
sample-quality on the best checkpoint, model manifest pointers,
training runs ← checkpoints + workflow artifacts, comparisons ← gates,
evalutions ← comparison/gate sides + workflow, workflows/gates as
leaves); the M61 consistency both ways (checkpoints whose references
are ALL lineage edges have EMPTY `checkpoint_blockers`; an
evaluation-referenced checkpoint IS blocked); the M66 cross-check
(internal category record sets identical; M66 external record ids ⊆
M69 external referencing ids; the fixture's exact 4 external entries);
determinism and zero mutation. The edge test covers: internal-only
model (references > 0, external == 0, oracle parity); fresh model
(all-empty categories, zero totals); unknown model → 404; malformed
manifest → registry-invisible 404 (restored); a dangling reference (a
tampered evaluation's bogus checkpoint id) producing NO phantom
record, deterministic, read-only (restored byte-exact); a
corrupt-but-parseable record (tampered result_hash) leaving the usage
view deterministic and read-only; duplicate reference ids (a
comparison with both sides on the same checkpoint → ONE entry);
oracle agreement after a newly created record. The HTTP test covers
the route end-to-end: shape, aggregate == sums, oracle parity over
the session root, the external set == the M66 external ids, the
suite-run targets discovered, byte-identical determinism, zero
mutation, unknown 404, OpenAPI 97 with the GET-only route and the
unchanged 7-delete set. **Full suite: 649 passed ×2 consecutive (exit
0 both)**; pyflakes + compileall clean. Test-side corrections during
development (all script-side, none touching implementation
semantics): a dead variable, an oracle dict initialization typo, the
external-entry-vs-distinct-id count distinction (the suite run
externally references TWO records — its state checkpoint AND its
probe evaluation), and a directory-path inventory comparison.

## 4. Live Certification

`smoke_m69_live.py` (port 8793), executed against the production
root. **First execution: 19/20** — C1 failed on a pure script
assertion bug in the disposable-copy phase: my expected totals ignored
that the suite run's probe EXECUTES a new evaluation record (+1
internal reference, +1 record); the engine data was correct
throughout (the same run's C3 — full oracle parity including the
external families — PASSED). Protocol followed: stopped, verified
read-only that production was byte-identical (52/52; the production
phase of that execution was entirely read-only and passed 14/14),
diagnosed the actual numbers on a fresh throwaway copy (discarded),
fixed the four expected values, re-executed. **Second execution:
20/20 PASSED** — production read-only phase (project/registry
coherence; disk == `m69_pre.sha256` 52/52; M66 usage certified counts;
M62 retention 13/10/3; the M69 route: canonical order, record counts
6/13/3/4/2/2 == the listings == the oracle, EVERY record's references
== the independent manifest oracle, aggregates == exact sums ==
certified totals 30 records / 57 references / 57 internal / 0
external, M66 agreement, the M61 lineage classification live — 10
lineage-only checkpoints with EMPTY M62 blockers —, unknown-model 404,
byte-identical determinism, OpenAPI 97 / GET-only / 7 deletes, zero
mutation), then the disposable-copy phase (external discovery live: a
suite run + sample + measurement created on the copy → exactly 4
external entries in the canonical order [suite-run state checkpoint,
sample, measurement, suite-run probe evaluation], internal 57→58 and
total 57→62 accounting for the probe's new evaluation record,
full oracle parity including external families, deterministic repeat,
copy discarded). Production mutation: ZERO across both executions
(the only writes ran on discarded copies).

## 5. Production Storage

**Before == after, exactly: 52 files / 8,926,403 B, every SHA-256
unchanged against `m69_pre.sha256`** (== `m68_retention_pre.sha256`
— production has not changed since the M68 delta), 0 tmp entries, 0
manifest modifications, 0 new persistent files, 0 servers left
running, 0 leftover copies. **M69 is completely read-only: 0 files
added, 0 removed, 0 modified** — the overview is computed live from
the authoritative records; no usage snapshots, dependency indexes,
lifecycle journals or caches exist (§11 honored). The disposable-copy
phase's writes (one probe suite definition, one suite run, one sample,
one measurement, one probe evaluation) lived and died on the discarded
copy.

## 6. Git

- `M69: model-owned record usage overview` — the ONE canonical
  analysis (engine constants + method), the four schemas, the read-only
  route + landing, the new test file, the OpenAPI count bumps
  (96 → 97, 54 sites), the README section + counts, the smoke, this
  report.
- `M69: pre-smoke production inventory (m69_pre.sha256)` — the
  production root's certification baseline (52 files; byte-identical
  to the M68 delta inventory).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch
  tip == local HEAD verified after the push; worktree clean (only the
  untracked `egg-info`, never committed).

## 7. Limitations

- The overview reports REFERENCES, not integrity: a corrupt record
  still appears with its references (deterministic, read-only, never
  repaired) — integrity views remain the M62/M68 retention surfaces.
- The one M61 blocker not visible in M69 is the live M52 best
  selection (computed, not persisted) — documented in the schema and
  README; a future per-record guard must add it the way M61 does.
- Training runs are manifest entries (not individually deletable
  records); M69 reports them and their references, but a future
  per-record deletion milestone would scope itself to the four
  directory-backed evidence families (workflows, evaluations,
  comparisons, gates) — checkpoints already have M61.
- Production holds no suite runs/samples/measurements, so the
  production-exercisable external surface is the certified-zero
  external count; external discovery was certified live on the
  discarded copy and in tests.
- No per-record drill-down route (the surface is the aggregate
  model-scoped overview, per the M64/M66 convention); per-record
  detail remains the existing family GET-one routes.

## 8. M69 Result

**M69 is complete against the spec, and it is strictly read-only —
no deletion logic was added or changed anywhere.** Numbers: tests
**646 → 649** (full suite ×2, exit 0 both); **OpenAPI 96 → 97**
(exactly one new GET-only route; delete set unchanged at 7);
model-owned categories: **training_run, checkpoint, workflow,
evaluation, comparison, gate** (6); reference categories: **10**
(7 internal incl. the 2 lineage kinds, 3 external); production
topology: **30 records, 57 references — 57 internal, 0 external**
(the fixture/discovery topology: 4 external entries from one suite
run + one sample + one measurement, with the probe-evaluation
side-effect accounted); independent-oracle agreement: **100%** —
every record's references, in order, in tests and live (both
executions); production file/byte counts **52 / 8,926,403 B
unchanged**, SHA-256 inventory identical (52/52); git: two commits
pushed, remote == HEAD, worktree clean. Transient event, handled per
protocol: the first smoke execution failed ONE check on the script's
expected totals (the probe-evaluation side effect) — stopped,
production verified read-only, diagnosed on a discarded copy, fixed,
re-executed 20/20.

## 9. Next Milestone

**M70 — explicit verified per-record retention for the model-owned
evidence families** (workflows, evaluations, comparisons, gate
decisions): the M67/M68 pattern applied one level down, with the M69
analysis as the ONE blocker source — exactly what M69 was built to
enable. The ready-to-paste prompt follows.

### M70 — EXPLICIT MODEL-OWNED RECORD RETENTION (ready-to-paste prompt)

```
Implement **Milestone 70 — Explicit Model-Owned Record Retention** for
AI Model Forge.

M69 exposed the complete persisted reference topology of every
model-OWNED record. The four directory-backed evidence families —
workflow records, evaluations, comparisons, gate decisions — still
cannot be pruned individually (they only go with the whole model,
M67). M70 gives them explicit verified deletion, the M67/M68 pattern,
with the M69 analysis as the ONE blocker source.

### 1. INSPECT FIRST — DO NOT MODIFY YET

Before changing anything, inspect and identify:

* the M67/M68 deletion architecture (scope -> integrity-first ->
  blockers -> atomic_delete_dir -> typed result/409) and the M68
  retention-view pattern — the templates to mirror;
* the M69 analysis (`model_records_usage_overview`,
  `MODEL_RECORD_CATEGORIES`, `RECORD_REFERENCE_CATEGORIES`,
  `RECORD_REFERENCE_EXTERNAL`, `RECORD_REFERENCE_LINEAGE`) — the ONE
  blocker source this milestone must REUSE (never a second scanner);
* the four family engines, their storage layouts, loaders, persisted
  `result_hash` staticmethods and listing conventions;
* the M61 lineage decision and the live M52 best selection (the one
  M61 blocker M69 cannot see) — decide from inspection whether any
  M70 guard needs an equivalent live check, and document the decision;
* what deleting a record means for referencing records (evaluations
  are referenced by comparison/gate sides and suite-run probe results;
  comparisons by gates and workflow artifacts; gates and workflows by
  nothing) — the blocker directions come from M69 verbatim;
* the OpenAPI conventions (expect 97 paths, 649 tests) and the
  README/landing conventions.

Record the pre-implementation baseline (tests, OpenAPI, production
inventory).

### 2. M70 OBJECTIVE

Four new explicit, verified, atomic deletions on the EXISTING
resource paths (their GET-one routes already live there):

DELETE /models/{model_id}/workflows/{workflow_id}
DELETE /models/{model_id}/evaluations/{evaluation_id}
DELETE /models/{model_id}/comparisons/{comparison_id}
DELETE /models/{model_id}/gates/decisions/{decision_id}

* guard order per record: scope (unknown model/record or
  registry-invisible -> 404, nothing deleted) -> INTEGRITY FIRST (the
  persisted result_hash must reproduce; tampered -> 409, never
  deletable, no force) -> blockers -> ONE atomic removal
  (atomic_delete_dir) with the typed
  {model_id, <record id>, files_removed, bytes_reclaimed} result;
* blockers come from the ONE M69 analysis: a record is blocked by
  every NON-LINEAGE reference M69 reports for it — internal
  referencing records (comparison/gate sides and their persisted ids,
  workflow stage artifacts, the model manifest's own pointers where
  applicable) and EXTERNAL root-level references (suite-run probe
  results naming an evaluation) — the SAME categories and reference
  ids M69 reports, in the SAME canonical order (the M67/M68 headline
  invariant, one level down: nothing protected that is not shown,
  nothing shown that is not protected). The M61 LINEAGE edges
  (checkpoint parents, run provenance) never block;
* blocked deletion -> structured typed 409 with the ordered blocker
  list; zero filesystem mutation on refusal (byte-verified);
* no cascade: deleting a record never rewrites or removes its
  referencing records — they are protected BY the guard;
* read-only retention views per record
  (GET .../workflows/{id}/retention, .../evaluations/{id}/retention,
  .../comparisons/{id}/retention,
  .../gates/decisions/{id}/retention) following the M68 delta
  pattern: identity, artifact files/bytes, integrity, deletable,
  ordered blockers — the SAME list the guard refuses on;
  retention.deletable == DELETE would succeed, both directions;
* checkpoints (M61) and training runs (manifest entries) are OUT OF
  SCOPE; model deletion (M67) and every M68 surface must remain
  byte-for-byte unchanged (only their live inputs can shrink).

### 3. TEST MATRIX

Focused tests following the established style, at minimum:

* per-family deletion: success with exact stats == an independent
  walk; only the record's own directory removed; registries shrink;
  repeat -> 404; unknown model/record -> 404;
* blockers per real referencing family (comparison sides block
  evaluation deletion; gate comparison_id blocks comparison deletion;
  workflow artifacts block their targets; suite-run probe results
  block evaluation deletion) — blocker ids/categories == the M69
  overview verbatim, both directions (the headline invariant);
* integrity: tampered result_hash -> 409 + zero mutation; corrupt
  manifest -> registry-invisible 404; restore-then-delete round-trips;
* the unblock chain: deleting a gate unblocks its comparison (and the
  comparison's side evaluations); M69 usage, the retention views and
  the DELETEs agree at every step; the model's own views stay
  coherent (its internal history shrinks live);
* retention/delete invariant both directions; determinism;
* HTTP: typed 200 results, typed 409s, 404s, OpenAPI (four DELETE
  operations on existing paths + four retention GET paths:
  97 -> 105), zero mutation on refusal;
* full regression: the complete suite must stay green with M61-M69
  semantics untouched.

### 4. LIVE CERTIFICATION

Production's four families are all REFERENCED (M69: evaluations
referenced by comparisons/gates; comparisons by gates; workflows and
gates are leaves — inspect which production records are actually
unreferenced before certifying). Production protocol: read-only
inventory + views + protected-DELETE refusals with the exact M69
blockers; the success paths on a DISPOSABLE COPY (copy production,
delete a leaf record there with exact accounting, verify the M69 view
shrinks live, discard the copy). Run the smoke exactly once; on a
script assertion bug, stop, verify read-only, fix, re-execute.
Production must remain byte-identical.

### 5. REPORT

Report exactly these 9 sections: Scope & Inspection; Implementation;
Tests; Live Certification; Production Storage; Git; Limitations;
M70 Result; Next Milestone — ending with the next ready-to-paste
implementation prompt.

### HARD CONSTRAINTS

* Inspect first; record the baseline.
* REUSE the ONE M69 analysis for every blocker decision; no second
  scanner; no dependency indexes.
* The headline invariant: M70 blockers == the M69-visible NON-LINEAGE
  references, same categories, same ids, same order.
* Integrity before mutation; atomic removal; deterministic typed
  results; unknown -> 404; no force, no bulk, no cascade, no
  automation, no policies.
* M61/M67/M68 semantics byte-for-byte unchanged; production
  byte-identical; destructive certification only on discarded copies.
* Preserve every previous milestone; full suite x2; statics clean;
  git clean and pushed.
```
