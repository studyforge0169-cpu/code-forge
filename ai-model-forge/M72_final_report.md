# Milestone 72 — Project Retention Inventory: Final Report

## 1. Scope & Inspection

M72 is the capstone read-only view after the M71 mutation milestone:
`GET /api/v1/project/retention` — the WHOLE deletion surface in ONE
inventory. Inspection (read-only, before any edit) mapped the
aggregation sources: every family already has its ONE per-artifact
retention view (M62 `checkpoint_retention_overview` — a per-model
registry overview with aggregates + per-entry sums; M65
dataset/tokenizer views; M67 model views; M68 suite-run/sample/
sample-quality views; M70 `model_record_retention_overview`; M71
definition views) and the ONE M63 `project_storage_overview` walk
carries the TRUE physical totals. Record enumeration comes from the
ONE M69 usage overview (per model, per category). No second scanner
is needed anywhere — M72 is pure aggregation.

Two semantic decisions, proven against the existing surfaces:
(a) the model family's size INCLUDES its owned records (M67
ownership semantics) — so family sizes overlap and must never sum
into the total; (b) `training_run` is the ONE record family without
a deletion lifecycle (the model manifest's own run provenance —
ownership, it goes WITH the model): it contributes counts only
(`deletion_supported: false`). Baseline before editing: HEAD
`a448bbe`; the pending M71 push landed at turn start (§6); 656 ×2
green at that tree; OpenAPI 104/14; production 52 files /
8,926,403 B byte-identical to `m71_pre.sha256`.

## 2. Implementation

* **Schemas** (2): `ProjectFamilyRetention` (`family`,
  `deletion_supported`, `count`, `files`, `size_bytes`,
  `deletable_count`, `blocked_count`, `reclaimable_files`,
  `reclaimable_bytes`) and `ProjectRetentionOverview` (`families`,
  `total_count`, `total_files`, `total_size_bytes`,
  `total_deletable`, `total_blocked`, `reclaimable_files`,
  `reclaimable_bytes`).
* **Engine**: `PROJECT_RETENTION_FAMILIES` (the canonical 15-family
  order: model, dataset, tokenizer, workflow_recipe, gate_policy,
  probe_suite, training_run, checkpoint, workflow, evaluation,
  comparison, gate, suite_run, sample, sample_quality) +
  `project_retention_overview()` — one pass over the registries and
  models; per family the SUM of the existing per-artifact views
  (checkpoints via the M62 aggregates verbatim); totals from the
  ONE M63 walk; the EXACT top-level reclaimable implements the
  model/record overlap rule (a deletable model contributes its
  WHOLE directory, subsuming its records; a blocked model
  contributes only its own deletable records — isolated per model
  via before/after snapshots of the cumulative record-family sums);
  root-level M68 records always count (they live outside
  `models/`).
* **API**: `GET /api/v1/project/retention` (tags `meta`), the
  defensive 404 scope convention, landing-page bullet + route-list
  line. Zero storage, zero mutation, deterministic.

## 3. Tests

`tests/test_project_retention.py` (NEW, 4 tests, 566 lines):

* **Oracles**: per-family numbers == the SUM of the per-item views
  (the aggregation source) AND raw filesystem walks for the
  simple-path families (models, datasets, tokenizers, definitions,
  suite runs, samples, checkpoints); totals == the M63 walk; the
  family sizes OVERLAP (asserted strictly greater than the true
  total — they must never sum into it); the EXACT reclaimable ==
  the overlap rule computed independently per model; the fixture's
  model C (deletable WITH a deletable record) proves the per-family
  sums double-count while the project total does not.
* **Live deltas**: deleting ONE leaf record moves the inventory by
  EXACTLY its numbers; the reclaimable for a deletable model tracks
  its CURRENT directory; deleting the model reclaims exactly the
  predicted bytes; the blocked dataset/tokenizer survive.
* **API + OpenAPI**: the route == the engine view on the same root,
  byte-identical repeat, 105 paths, GET-only, both schemas.
* **Empty project**: zeroed families; the single `project.json` is
  the whole storage.

The independent oracles caught ONE implementation bug before any
certification: the engine's initialization added the definitions'
reclaimable FILES but not their BYTES (an 800-byte hole in the
fixture) — fixed, then everything green. Pre-existing expectations
updated: 57 `== 104` sites → `== 105` across 18 test files (the
historical `smoke_m71_live.py` stays as-certified). Full suite:
**656 → 660 tests, 29 → 30 suites, 660 ×2 green** on the final
state.

## 4. Live Certification

`smoke_m72_live.py` (14 checks, read-only — the milestone IS a
read-only inventory; nothing deleted/created/mutated anywhere).
**Rehearsal 14/14 on a discarded byte-identical copy first**,
catching ONE pure FACT bug: the M70 report's "M62 13/10/3" notation
is total/DELETABLE/blocked — I had read it as blocked/deletable;
the live M62 overview shows 13 checkpoints / 10 deletable / 3
protected (2 held by workflow stage artifacts, 1 by
best+published+manifest pointers). FACT corrected read-only; then
the **live run EXACTLY ONCE against production: 14/14** — project
facts; disk == `m72_pre.sha256` (52/52); the exact response shape +
canonical 15-family order; per-family certified counts (34
artifacts); `training_run` the one lifecycle-less family; the
certified deletable/blocked split (ckpt 10/3, workflows 3/0, evals
0/4, comparisons 0/2, gates 0/2, model/dataset/tokenizer/recipe all
blocked; totals 13 deletable / 15 blocked); **the aggregation
oracle LIVE** (family files/bytes == the per-item views fetched
individually); TRUE totals == the M63 walk == the raw disk
(52 / 8,926,403 B); **EXACT reclaimable = 3 leaf workflow records +
the 10 deletable checkpoint artifact sets = 23 files / 6,323,505 B**
(the model is blocked by the recipe → records only); deterministic
byte-identical repeat; OpenAPI 105/14 + GET-only + schemas; final
byte-identity. **Production after the smoke: 52/52 byte-identical,
`tmp/` empty — ZERO mutation.**

## 5. Production Storage

Unchanged: **52 files / 8,926,403 B**, byte-identical to
`m72_pre.sha256` == `m71_pre.sha256` (committed at the repo root).
The inventory reports it exactly: 34 artifacts across 15 families,
13 currently deletable (3 leaf workflow records + 10 checkpoint
artifact sets), 15 blocked, 6,323,505 B reclaimable through the
existing verified deletion routes — reported, never executed.

## 6. Git

§13 first: the M71 turn ended with both commits UNPUSHED (sandbox
GitHub auth failure, disclosed in the M71 report §6). At THIS
turn's start the auth was repaired and the push landed:
`93258c1..a448bbe` — remote HEAD == local HEAD verified before any
editing. M72 commits on `arena/01a071e9-code-forge`:
the implementation commit (schemas/engine/api/landing, the 18
updated test files, `tests/test_project_retention.py`, README
counts + the M72 section, `smoke_m72_live.py`, this report) and the
inventory commit (`m72_pre.sha256`, 52/52 == `m71_pre`). Both
pushed; remote HEAD == local HEAD verified after the push.

## 7. Limitations

* The per-family `blocked_count` conflates "held by references" and
  "failed integrity" (both mean `deletable is False` in the family
  view) — the split is available per artifact in the underlying
  views, not aggregated.
* `training_run` contributes counts but no storage numbers (its
  provenance lives in the model manifest — ownership); a user
  wanting "how big are my runs" reads the M63/M62 surfaces.
* The inventory is a point-in-time live computation; it is not
  persisted, not diffable over time, and runs no cleanup (all scope
  bans honored — no policies, TTLs, quotas, background jobs).
* The top-level reclaimable is exact for "delete every
  currently-deletable artifact NOW"; deletions change downstream
  deletability (that is the M70/M71 unblock behavior), so the
  number is recomputed live on every call.

## 8. M72 Result

**Delivered and certified.** The retention story is complete end to
end: per-artifact views (M62/M65/M67/M68/M70/M71), per-family
deletion guards (M61/M65/M67/M68/M70/M71), and now the ONE project
inventory that rolls the whole surface up with TRUE totals and an
EXACT reclaimable prediction. Certified numbers: tests
**656 → 660** (4 new), suites **29 → 30**, OpenAPI
**104 → 105 paths / 14 deletes**; full suite **660 ×2 green**; live
smoke **14/14 once** (rehearsal 14/14 first, one inverted FACT
fixed read-only); one implementation bug (missing defs-bytes init)
caught by the independent oracles and fixed before certification;
production **52/52 byte-identical, zero mutation**; the pending M71
push landed at turn start (§13 clean). Pushed; remote == HEAD.

## 9. Next Milestone

**M73 — read-only DELETION IMPACT PREVIEW**
(`GET .../retention/impact` on ONE artifact): the forward
complement of the retention view — for ONE deletable-or-blocked
artifact, what its VERIFIED deletion would unblock: the immediate
dependents whose blocker lists lose this artifact, and (bounded,
deterministic) the artifacts that BECOME deletable as a result —
every step derived from the EXISTING M61–M72 analyses recomputed on
a throwaway in-memory shadow (never mutating storage), with the
exact files/bytes that would become reclaimable (the M72 overlap
rule). Read-only, zero mutation, full suite ×2, live smoke once,
9 sections, inventory `m73_pre.sha256` == `m72_pre.sha256`.
