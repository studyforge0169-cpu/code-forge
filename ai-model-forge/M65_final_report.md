# Milestone 65 — Explicit Verified Dataset & Tokenizer Retention: Final Report

**Date:** 2026-09-11 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

`DELETE /datasets/{id}` and `DELETE /tokenizers/{id}` are explicit,
verified and reference-safe — dataset/tokenizer deletion is refused on
EXACTLY the references the M64 usage overview reports, with integrity
verified before any reference analysis (the M61 ordering) and ONE
atomic removal. Read-only retention views expose the same decision.
Both live certifications passed on their single first executions
(18/18 initial + 21/21 spec-compliance delta), with production
byte-identical.

> **ENVIRONMENT EVENT (disclosed):** the sandbox was re-provisioned a
> THIRD time between the initial M65 delivery and this spec-compliance
> delta. The repository was fully recovered from the remote (`git
> reset --mixed FETCH_HEAD` onto `ff741ce`; worktree byte-identical,
> all 65 milestones safe), the Python environment rebuilt (same
> versions; baseline re-verified **630 passed**), and the production
> root reconstructed a third time via the committed
> `m62_production_rebuild.py` — the certified structure reproduced
> exactly (dataset `8a2af1e3d1fa` with 16 references 6/2/4/2/0/1/1,
> tokenizer `02673690c5ff` with 15 references 6/2/4/2/0/0/0/1, model
> `5939483e70ac`, M62 retention 13/10/3, 52 files / 8,926,407 B / 0
> tmp; new content-derived ids). This delta's pre-inventory is
> `m65_retention_pre.sha256` (52/52).

---

## 1. Inspection & Baseline

**Baseline (before the delta):** HEAD `ff741ce` (the initial M65
delivery: `8c3d555` impl + `ff741ce` inventory), worktree clean, full
suite **630 passed** (exit 0), OpenAPI **89 paths**, production root
rebuilt as above. **Relevant M61 findings (reused, not reinvented):**
the guard order scope → INTEGRITY VERIFICATION → reference analysis →
atomic removal; unreadable manifests are listing/registry-invisible →
404 (nothing deleted); corrupt-but-parseable artifacts → RuntimeError
→ 409 (deletion never bypasses integrity validation); the API's
structured 409 detail `{message, ids, protected, blockers}`;
`remove_checkpoint`'s measure-then-`atomic_delete_dir` split.
**Relevant M64 findings:** the ONE canonical usage analysis
(`dataset_usage_overview` / `tokenizer_usage_overview` — 7 dataset /
8 tokenizer categories, deterministic sorted reference ids); the
existing verifiers (`DatasetEngine.verify` — records, splits,
tokenized artifacts; NO tokenizer verifier existed); the registries
skip unparseable manifests (registry-invisible). **Gap analysis
against this spec:** the initial delivery lacked integrity
verification before deletion, malformed/corrupt refusals, and a
retention view exposing `deletable` — all delivered in this delta.

## 2. What Changed

**Initial delivery (commits `8c3d555` + `ff741ce`):** `ArtifactDeletionBlocker`,
`DatasetDeletionResult`/`TokenizerDeletionResult`, the typed 409
details; the low-level `DatasetEngine.delete`/`TokenizerEngine.delete`
rewritten to the measure-then-`atomic_delete_dir` pattern;
`dataset_deletion_blockers`/`tokenizer_deletion_blockers` (the guard
analysis from the ONE M64 overview); the guarded facade deletes; both
DELETE routes rewritten M61-style; README/landing; 4 new tests + 3
honest updates to existing tests; `smoke_m65_live.py` (18/18).

**This spec-compliance delta:** `app/tokenizer.py` — the new
`verify()` (manifest resolution + tokenizer.json sha256 vs the
persisted `tokenizer_hash` → ok/failed report, read-only);
`app/schemas.py` — `DatasetRetentionOverview` /
`TokenizerRetentionOverview` (identity + ordered artifact files +
total bytes + integrity_verified + deletable + ordered blockers);
`app/engine.py` — `_scope_data_artifact` (the M61
registry-invisible-404 scope convention), `_artifact_files` (the
M63-boundary ordered walk), `dataset_retention_overview` /
`tokenizer_retention_overview`, and BOTH deletes reordered to the
M61-precise sequence (scope → integrity → blockers → atomic removal);
`app/api.py` — the two new GET retention routes, `RuntimeError → 409`
on both DELETEs, malformed-manifest → 404 hardening on the usage
routes, landing updates; `README.md`; 2 new tests (retention views +
integrity-first, engine and HTTP); OpenAPI bumps 89 → 91 (50
assertions); evidence: `smoke_m65_retention_live.py` +
`m65_retention_pre.sha256`.

## 3. Dataset Retention

**Blocker model:** `ArtifactDeletionBlocker{reason, detail}` — reason
is an M64 dataset category (training_run, workflow, evaluation,
comparison, suite_run, tokenizer_training, tokenized_version), detail
is the sorted reference ids — IDENTICAL to what
`GET /datasets/{id}/usage` reports. **Verification:** the existing M2
`DatasetEngine.verify` (manifest validity, record count, per-record
hash, split recomputation, tokenized size/hash/counts) runs BEFORE the
reference analysis; `status != ok` → RuntimeError → 409, nothing
deleted. An unparseable `dataset.json` is registry-invisible → 404
(the M61 unreadable-manifest convention; the registries skip it).
**Deletion semantics:** scope → integrity → blockers (ValueError →
409 with the re-derived ordered list) → ONE `atomic_delete_dir` of the
dataset's OWN directory (meta + versions + records + tokenized
artifacts) → the deterministic `{dataset_id, files_removed,
bytes_reclaimed}`. **Retention view:** `GET /datasets/{id}/retention`
— identity, the ordered artifact file list (sorted relative paths,
tokenized artifacts included) + total bytes, the integrity outcome,
`deletable` (integrity AND zero blockers) and the ordered blockers. A
corrupt dataset is never deletable; refusals write nothing (proven
byte-level).

## 4. Tokenizer Retention

**Blocker model:** the same `ArtifactDeletionBlocker` with the M64
tokenizer categories (training_run, workflow, evaluation, comparison,
suite_run, sample, sample_quality, tokenized_dataset). **Verification
(new, the gap the spec exposed):** `TokenizerEngine.verify` — the
manifest must resolve AND the tokenizer.json content hash must equal
the persisted `tokenizer_hash`; mismatch → `status: failed` → the
guard refuses with 409. An unparseable manifest is registry-invisible
→ 404. **Deletion semantics:** the identical M61-precise sequence
(the tokenizer family's pre-M65 deletion was an UNGUARDED rmtree —
now scope → integrity → blockers → atomic removal). **Retention
view:** `GET /tokenizers/{id}/retention` — identity (incl.
`trained_on_dataset_id` provenance), the ordered artifact files
(manifest.json + tokenizer.json) + bytes, integrity, `deletable`,
ordered blockers. The pre-M65 mutual-protection consequence is
preserved and documented: a tokenized pair blocks in both directions
(`tokenized_dataset` ↔ `tokenizer_training` + `tokenized_version`);
no cascade, no force.

## 5. Canonical Dependency Invariant

**One analysis, structurally:** `dataset_deletion_blockers` and
`tokenizer_deletion_blockers` literally call the ONE M64 usage
overview and convert every non-empty category to a blocker verbatim —
there is no second scanner anywhere in the codebase; the retention
views and the DELETE guards consume the same objects. **Bidirectional
invariant, tested per category:** `usage count > 0 ⟺ category in
blockers ⟺ deletable == False` — asserted as an explicit per-category
loop over the full-coverage fixtures (every category non-empty), as
list-equality between usage/retention/guard at engine level, and as
HTTP parity (the 409 `detail.blockers` == the usage route's non-empty
categories rebuilt as blockers == the retention view's blockers).
Conversely `usage count == 0 ⟹ category absent from blockers`.
Integrity never bypasses the invariant: a corrupt artifact is
not-deletable regardless of references (the M61 ordering — proven by
a corrupt+referenced dataset refusing with the INTEGRITY error, not
the blocker error). Cross-milestone: M62 (13/10/3) and M63/M64 views
re-verified green in the suites and the live smokes.

## 6. Verification

**Focused M65 tests (6 total in `tests/test_data_retention.py`):**
the two guard-parity tests (both directions, full coverage, refusals
write nothing, unknown → 404); unreferenced-deletion + live-recompute
(exact stats vs independent walks, only-own-files-disappear
inventories, the unblock chain); the new retention-views test
(unreferenced: deletable + exact file lists/bytes; referenced: triple
parity + per-category invariant; corrupt dataset/tokenizer: never
deletable, deletion refused RuntimeError, restore flips integrity
back; malformed manifests: registry-invisible 404 for view AND
delete, then restored); the new HTTP test (shapes, corrupt → 409 over
HTTP, malformed → 404, determinism, zero-mutation inventory, OpenAPI
91). Plus 3 honestly-updated existing tests (typed deletion results,
the guard message, the mutual-protection cleanup tail). **Full suite:
630 → 632, ×2 consecutive runs on the final state** (exit 0 both).
**Statics:** pyflakes + compileall clean. **OpenAPI:** **89 → 91**
(exactly the two retention GETs; DELETEs unchanged on their existing
paths; 50 count assertions bumped; standalone-verified). **Live
smokes (each exactly once, zero corrections):** the initial
`smoke_m65_live.py` — **18/18** (production read-only; guards + safe
dataset deletion verified on a discarded copy); this delta's
`smoke_m65_retention_live.py` — **21/21** (production read-only:
retention views with independent artifact walks, triple parity with
usage + the protected 409s, determinism, zero mutation; on the copy:
the views equal production, a fresh tokenizer's retention view
proves it deletable, the ONE safe deletion with exact stats
2 files / 4,682 B, live recompute, copy discarded).

## 7. Storage Proof

**Production before: 52 files / 8,926,407 B. After: 52 files /
8,926,407 B — byte-identical** (sha256-verified 52/52 against
`m65_retention_pre.sha256`; also verified after the initial M65 smoke
against its own inventory), 0 tmp, 0 residue, 0 servers. Protected
DELETE calls produced zero filesystem mutation (verified in tests
byte-level and in both smokes). **The one safe deletion certified**
(§11) executed exactly once, on the discarded COPY: the fresh
tokenizer `manifest.json + tokenizer.json` — 2 files / 4,682 B,
removed atomically (no `.tmp-delete-*` residue), with the
before-measurement equal to the reported result; the previous
certification's safe deletion (a 4-file/1,936 B dataset, also on a
copy) was likewise one-shot. Production contains no safe
dataset/tokenizer (both referenced — 16 and 15 references), so per
§12 no production deletion was manufactured; the safe paths are
certified by the deterministic fixtures + the copy protocol. M65 adds
zero persistent retention records — both analyses are live-computed.

## 8. Git

- `8c3d555` — M65: explicit verified dataset & tokenizer retention
  (initial delivery, 11 files) · `ff741ce` — M65 pre-smoke inventory
  (both pushed previously).
- This delta: `M65: integrity-first guards + retention views (spec
  compliance)` — engine/schemas/api/tokenizer, README, tests, the
  smoke, this report; then `M65: retention-smoke production inventory
  (m65_retention_pre.sha256)` — the rebuilt root's pre-smoke
  inventory.
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch
  tip == HEAD; worktree clean (only the untracked `egg-info`).

## 9. Final Status & Next Milestone

**M65 is complete** against the full quality bar: both deletions
protected by the canonical M64 analysis (no second scanner — the
guards call the overview directly), deterministic ordered blockers,
integrity verified before mutation, atomic deletion, bidirectional
usage/blocker agreement, zero mutation on refusals, safe deletion
removing only the intended directory, unrelated artifacts
byte-identical, 632 ×2 green, statics clean, OpenAPI 91 correct, both
live certifications first-execution clean, storage minimal (zero
persistent records), git pushed. **Limitations (documented, not
gaps):** a tokenized dataset/tokenizer pair is mutually protected —
removing derived tokenized artifacts requires a future EXPLICIT
operation (never a cascade); a corrupt unreferenced artifact cannot
be deleted through the API until manually repaired (the deliberate
M61 semantics); the tokenizer content-hash verifier is new (there was
none) and minimal by design.

**Next milestone — M66, read-only model usage overview.** The model
itself is the last unguarded deletion: `DELETE /models/{id}` is still
a plain M1-era rmtree. Three ROOT-LEVEL families persist model
references outside the model directory — M10 suite-run records,
M15 samples, M16 sample-quality measurements — so deleting a model
today silently orphans them. M66 applies the proven M64 pattern:
one live-computed view per model listing every external reference
(through the ONE existing listings) plus the internal ownership
summary (the M62/M63 numbers), before any future guard. The
ready-to-paste prompt follows.

---

### M66 — READ-ONLY MODEL USAGE OVERVIEW (ready-to-paste prompt)

```
Implement **Milestone 66 — Read-Only Model Usage Overview** for AI
Model Forge.

M65 completed the retention matrix for checkpoints, datasets and
tokenizers. The MODEL itself is the last unguarded deletion:
`DELETE /models/{id}` is a plain rmtree of the model directory
(M1-era `Storage.remove`). Most of a model's evidence lives INSIDE
its directory and dies coherently with it — but three ROOT-LEVEL
families persist model references outside it: M10 suite-run records,
M15 samples and M16 sample-quality measurements. This milestone is
visibility only (the M64 pattern): before any future model-retention
decision, the operator must see what a deletion would orphan.

### 1. INSPECT FIRST — DO NOT IMPLEMENT YET

Before changing anything, inspect and identify the authoritative
implementations for:

* model deletion today (Storage.remove, engine.delete_model, the
  DELETE /models/{id} route — current semantics, 404s, response);
* every ROOT-LEVEL family that persists a model_id outside the model
  directory: suite-runs (SuiteRunRecord.model_id), samples
  (samples/<model_id>/), sample-evaluations
  (sample-evaluations/<model_id>/) — and their ONE listing engines;
* whether anything else references a model from outside its
  directory (workflow RECIPES embed model ids in stage configs —
  decide honestly whether a recipe is a per-model reference or a
  reusable definition, following the M64 suite-registry precedent);
  datasets/tokenizers do NOT reference models;
* the model's internal families (checkpoints, evaluations,
  comparisons, gates, workflows) — what lives inside models/<id>/
  and therefore goes with the model (the M63 per-model row and its
  category attribution are the ONE storage accounting);
* the M64 usage-overview conventions (category constants, reference
  id formats, referenced flag, determinism, zero-writes tests);
* the OpenAPI count conventions and the README/landing conventions.

Run the current full test suite before implementation and record the
baseline (expect 632 passed, OpenAPI 91 paths).

### 2. M66 OBJECTIVE

ONE read-only, live-computed route:

GET /models/{model_id}/usage

reporting, WITHOUT any mutation:

* the model's identity (id, name, created_at, architecture,
  parameter_count);
* EXTERNAL references (what a model deletion would orphan), by
  category in a canonical order: suite_run (root-level M10 records),
  sample (root-level M15 records), sample_quality (root-level M16
  records) — each as deterministic sorted record ids, computed
  through the ONE existing listings;
* an INTERNAL ownership summary (what lives inside the model's own
  directory and goes with it): checkpoint count + bytes, evaluation /
  comparison / gate / workflow record counts — the SAME numbers the
  M62 retention overview and the M63 per-model row report (never a
  second accounting);
* aggregates: external reference counts per category,
  total_external_references, and a boolean `externally_referenced`.

### 3. DESIGN CONSTRAINTS

* REUSE: the ONE listing engines for the root-level families; the
  ONE M62 retention overview and the ONE M63 per-model accounting
  for the internal summary. No second scanner, no stored index.
* Read-only, zero storage, zero mutation, deterministic
  (byte-identical over unchanged state); no cache, no pointer.
* Unknown model -> the family's 404; a model with no external
  references -> 200 with empty external categories (the collection
  convention).
* Do NOT change delete_model semantics (M66 is visibility only — a
  future milestone may add the guard).
* Do NOT add: deletion guards, policies, automation, bulk
  operations, or any write path.

### 4. TEST MATRIX

Focused tests following existing style, at minimum:

* a model referenced by suite runs, samples and sample-quality
  measurements (external) with internal history (checkpoints,
  evaluations, comparisons, gates, workflows);
* every external category appears with correct ids/counts (parity
  with the ONE listings);
* internal summary parity: counts == the M62 overview / M63 row /
  family listings;
* deterministic ordering and byte-identical repeat calls;
* zero writes (byte-level inventory comparison);
* unknown model -> 404; a fresh model -> zeroed external categories;
* a new external reference (suite run) -> the overview recomputes;
* OpenAPI: exactly one new path (91 -> 92, reason reported); README
  + landing updates; full regression suite green.

### 5. LIVE CERTIFICATION

After the full suite passes twice and statics are clean, run exactly
one READ-ONLY live smoke against production: capture the inventory,
query usage for every production model, verify the external
categories against the public listing routes (independent
cross-check), verify the internal summary against the M62/M63
routes, repeat for byte equality, and verify the inventory is
byte-identical afterwards. Zero state-mutating calls of any kind.

### 6. REPORT

Report exactly these 9 sections: Baseline & inspection;
Implementation; Usage view; Consistency; Tests; Live smoke; Storage;
Git; Next milestone — then provide the next ready-to-paste
implementation prompt.

### HARD CONSTRAINTS

* Inspect first.
* Read-only only; zero storage; zero mutation.
* One new route; no new scanners or registries.
* Reuse the ONE listings, the ONE M62 retention analysis and the ONE
  M63 accounting.
* Do not change any deletion semantics.
* No guards, no policies, no automation, no bulk operations.
* Preserve every previous milestone.
* The authoritative invariant is: **every external reference M66
  reports is a real persisted root-level record found through the
  authoritative listings, and every such record appears in M66's
  overview — nothing invented, nothing hidden; the internal summary
  is exactly what the M62/M63 views report.**
```
