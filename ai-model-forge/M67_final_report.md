# Milestone 67 — Explicit Verified Model Retention: Final Report

**Date:** 2026-09-11 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

The M1-era `DELETE /models/{model_id}` — a bare `shutil.rmtree(...,
ignore_errors=True)` with no scope, no verification and no guards — is
replaced by the proven M61/M65 verified-retention pattern: scope →
integrity-first verification → the ONE M66 dependency analysis as the
blocker source → ONE atomic removal. **Live-certified 26/26; production
byte-identical; the success path proven on a discarded copy.** Zero new
DELETE paths; the route, not duplicated, was extended.

---

## 1. Inspection & Baseline

**Baseline:** HEAD `a064c08` (M66, pushed), worktree clean (only the
untracked `egg-info`); full suite **636 passed** (exit 0); OpenAPI **92
paths**; production **52 files / 8,926,407 B / 0 tmp** — byte-identical
to the M66 certification. **The old M1 deletion path, inspected in
detail:** route `DELETE /models/{model_id}` checked `model_ids()`
membership (→404), called `engine.delete_model` → `Storage.remove` →
`shutil.rmtree(model_dir, ignore_errors=True)`, and returned
`{"deleted": id}` — no integrity check, no reference analysis, not even
an error on a half-corrupt model. Also inspected: the M61 checkpoint
guard, the M65 dataset/tokenizer guards + retention views + 409
marshalling (the pattern mirrored here), the M66 usage analysis
(`MODEL_USAGE_CATEGORIES`, `model_usage_overview`,
`_recipe_model_ids`), the M2 verifier (`Storage.verify_integrity` =
manifest parse → `load_weights` with sidecar-hash cross-check →
empty-state check; **`load_weights` raises `FileNotFoundError` for
missing weights** — decisive for the 404/409 split), the model artifact
layout (manifest, weights.pt, weights.sha256, checkpoints/, workflows/,
evaluations/, comparisons/, gates/ — all inside `models/<id>/`), every
M66 reference category, and the existing deletion tests
(`test_api.py:61` — the old response shape; `test_model_engine.py:243`
— a fresh-model delete). **Two topology findings from fresh
inspection:** (1) `policies/<policy_id>/manifest.json` persists
`policy.model_id` and `resolve_policy` refuses a mismatched model — a
real root-level external reference M66 had missed (the recipe pattern);
(2) `ModelRecord.parent_model_id` has **no setter anywhere** in the
codebase — a dead field, not a persistable reference. One transient
note: the very first post-change full-suite run reported `1 failed, 640
passed`; the failed test passed on its immediate isolated rerun and
never recurred (see §5).

## 2. Model Retention Architecture

**M66 amended (spec §4's inspection escape hatch):** `policy` was added
as the 11th canonical category (`MODEL_USAGE_CATEGORIES` external
tail) — inspection proved the M66 surface incomplete: registered gate
policies persist `policy.model_id` at the root and are
resolution-checked against it, exactly the inert-definition-binds-a-model
pattern that made recipes a reference. The category is computed through
the ONE policy listing (`PolicyEngine.list_policies`), the M66 tests
were extended to cover it (fixture item 9, oracle scan, external count
4→5, `/api/v1/policies` parity), and the README M66 section now
documents five external families. `parent_model_id` was inspected and
deliberately NOT added (no operation persists it). **New schemas:**
`ModelDeletionBlocker` (category, reference_id, detail — per-reference,
typed), `ModelDeletionResult` (model_id, files_removed,
bytes_reclaimed), `ModelDeletionBlocked` (the structured 409 detail),
`ModelRetentionOverview` (identity, architecture, parameter_count,
ordered files, size_bytes, integrity_verified, deletable, blockers).
**Engine:** `model_deletion_blockers()` derives the blocker list from
**the ONE M66 overview** — no second scanner: external categories only,
canonical category order, sorted reference ids, with short
authoritative `detail` strings from the SAME listings; and
`model_retention_overview()` (scope → artifact walk → verify outcome →
blockers → deletable). `delete_model()` was **replaced in place**
(same facade method, M66/M65 conventions), now returning
`ModelDeletionResult`. **API:** the SAME `DELETE /models/{model_id}`
route extended (response model `ModelDeletionResult`, 409 documented
via `ModelDeletionBlocked`), plus ONE new read-only route
`GET /models/{model_id}/retention`. OpenAPI **92 → 93** (52 count
assertions bumped); zero new DELETE paths (the delete-operation set is
asserted to be exactly the four pre-existing ones).

## 3. Deletion Safety

The guard, in order: **(1) scope** — `storage.load_record` through the
M61 `_scope_data_artifact` convention: unknown OR registry-invisible
(unparseable manifest) model → `FileNotFoundError` → 404, nothing
deleted. **(2) INTEGRITY FIRST** — the existing M2 verifier must pass;
because scope already proved the manifest parses, ANY verifier failure
(missing weights — notably a `FileNotFoundError` that must NOT become a
404 — corrupt weights, sidecar hash mismatch, empty state) is refused
as `RuntimeError` → 409. A corrupt model is never deletable; no force
flag, no filesystem fallback. **(3) blockers** — the ONE M66 analysis:
ANY external root-level reference → `ValueError` → **409 with the
typed ordered blocker list** `{"message", "model_id", "protected":
true, "blockers": [{category, reference_id, detail}]}` (the route
re-derives the list for the detail, the M65 pattern). **(4) atomic
removal** — `atomic_delete_dir` (the M61 primitive: rename to hidden
sibling, then rmtree), preceded by an `_artifact_files` measure; the
result reports exact `files_removed`/`bytes_reclaimed`. **No
cascade:** a protected DELETE changes nothing on disk (byte-verified in
tests and live); a successful DELETE removes ONLY the model's own
directory; root-level families are protected BY the guard and never
rewritten or removed. The blocker policy is explicit and testable:
**external references block (they would be orphaned); internal
model-scoped families are ownership — they live inside `models/<id>/`
and are removed atomically WITH the model, so they never orphan a
record and never block.**

## 4. M66/M67 Invariant

**The deletion blocker surface exactly matches the M66 visible
EXTERNAL dependency surface — nothing protected that is not shown,
nothing shown that is not protected.** Mechanically: the guard calls
the ONE `model_usage_overview` and maps its external categories
(suite_run, sample, sample_quality, workflow_recipe, policy — the same
canonical order, the same persisted reference ids the usage route
reports) to per-reference blockers; internal categories are skipped by
construction. Proven in three independent ways: **(1) tests** —
`test_m67_headline_invariant` verifies both directions on a model with
ALL ELEVEN categories non-empty (every blocker ↔ every M66-visible
external reference, no internal id ever in the blocker set) against an
**independent manifest-parsing oracle** that derives the external
subset from raw JSON; per-category tests then prove each external
family blocks alone (suite-run-only, recipe-only, sample+quality,
policy-only) while an internal-only model with rich history (runs,
checkpoints, evaluations, comparison, gate, workflow) is deletable;
**(2) HTTP** — the 409 blocker list is asserted equal to the usage
overview's external references and to the retention view's blockers;
**(3) live** — the production model's protected DELETE returned
blockers exactly equal to its M66 usage external references (the one
recipe), verified against the smoke's own independent oracle.

## 5. Verification

**Focused tests (5 new, `tests/test_model_retention.py`):** the
headline invariant (bidirectional, oracle-checked, determinism,
zero-mutation); per-external-category guards + live recompute (a fresh
model flips deletable→blocked when a new policy references it, with no
cross-talk); the internal-only ownership proof (delete succeeds with
exact stats vs an independent walk, only its directory removed,
byte-identical remainder, no `.tmp-delete-*` residue, registries
shrink, repeat → `FileNotFoundError`); six corruption cases (malformed
manifest → 404; missing manifest → 404; missing weights → 409;
corrupted weights → 409; sidecar hash mismatch → 409; schema-invalid
metadata → 404 — each with pre/post SHA-256 equality and restoration);
and the full HTTP surface (retention shape/parity, structured 409 ==
M66 external refs, zero-mutation on refusal, fresh-model 200 result,
404s, byte-identical repeats, OpenAPI 93 with the delete set unchanged
and the 409 response model documented). **Updated tests:** the M66
suite for the `policy` category (justified: M66 was incomplete by
inspection); `test_api.py`'s delete assertion to the new typed result
shape (its old `{"deleted": id}` assumption intentionally conflicts
with the newly required semantics). **Full suite: 636 → 641.** Honest
account: the first run reported `1 failed, 640 passed`; the failed
test's name was lost to an output-capture mistake, it **passed on its
immediate isolated rerun** (`--last-failed`), and **three consecutive
subsequent full runs were 641 passed, exit 0** — recorded as an
unidentified transient, not silently ignored. **Statics:** pyflakes +
compileall clean. **OpenAPI:** 93 paths, standalone-verified.

## 6. Live Certification

`smoke_m67_live.py` (port 8790). **First execution hit a pure script
assertion bug** at the protected-DELETE check: the script parsed the
409 body without unwrapping FastAPI's `{"detail": ...}` envelope. The
protocol was followed exactly: **stopped before any further
operation**, verified read-only that the DELETE had been **refused**
(server log: `409 Conflict`; production byte-identical 52/52; model
intact; tmp clean; the copy phase had not started), fixed the one-line
script bug, and re-executed. **Result: 26/26 PASSED** — production
read-only checks (project/registry coherence, M62 13/10/3, M63
storage, M64 usage 16/15, M66 usage 11 categories with certified
counts 6/13/3/4/2/2/0/0/0/1/0 = 30 internal + 1 external, usage ==
independent manifest oracle, retention 40 files/8,874,610 B ==
independent walk with integrity True/deletable False, the protected
DELETE → structured 409 whose blockers equal the M66 external
references exactly — the headline invariant live —, byte-identical
repeats, listing parity, unknown-model 404s on all three endpoints,
OpenAPI 93 with no new DELETE paths, zero mutation, tmp clean), then
the **disposable-copy success path** (copy of production; guard parity
— the production model still refused on the copy with zero mutation; a
fresh model created and TRAINED on the copy — internal references do
not block; retention safe state confirmed; DELETE with exact stats
**7 files / 1,897,111 B** == the independent pre-walk == the retention
view; only that directory removed, atomically, no residue; registries
shrink; everything else on the copy byte-identical; copy discarded).
One production DELETE was attempted in each execution — both refused
(non-mutating); the only mutating deletion in this milestone ran
exactly once, on the discarded copy.

## 7. Storage Proof

**Production: before == after, exactly** — 52 files / 8,926,407 B,
every SHA-256 unchanged against `m67_pre.sha256` (which is itself
byte-identical to `m66_pre.sha256` — production has not changed since
the M66 certification), zero tmp entries, zero manifest modifications,
zero new persistent files (the blocker analysis is live-computed; no
dependency index, no retention record, no reference count — §22's full
banned list honored). **The disposable-copy deletion accounts
exactly:** `files_removed = 7 = |victim model artifact files|` and
`bytes_reclaimed = 1,897,111 = Σ(victim model artifact file sizes)`,
both equal to the pre-deletion independent walk and the retention
view; no unrelated artifact changed on the copy (full inventory
diff: exactly the victim's files); the copy was discarded. Final
independent audit after everything: production 52/52 byte-identical, 0
servers running, 0 leftover copies.

## 8. Git

- `M67: explicit verified model retention` — the engine guard +
  retention view + replaced deletion, the M66 `policy`-category
  amendment, schemas, API (route replaced in place + retention route),
  landing/README, the M66-test extension, the new M67 tests, the
  OpenAPI bumps, the smoke, this report.
- `M67: pre-smoke production inventory (m67_pre.sha256)` — the
  production root's pre-certification inventory (52 files;
  byte-identical to the M66 inventory, evidencing the unchanged
  state).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch tip
  == HEAD; worktree clean (only the untracked `egg-info`).

## 9. Final Status & Next Milestone

**M67 is complete** against the full quality bar: the existing M1
route retained and extended in place (zero new DELETE paths — the
spec's delete-operation set asserted unchanged), the unsafe rmtree
replaced by verified deletion, M66 still the single canonical analysis
(now including the `policy` category its inspection gap hid), blocker
ordering deterministic with ids matching M66, integrity verified
before mutation (corrupt/malformed models refused, never deleted),
structured typed 409s, protected DELETE proven non-mutating,
`atomic_delete_dir` reused, no cascade, the independent oracle in
agreement (tests + live), 641 passed across three consecutive full
runs, statics clean, OpenAPI correct, production byte-identical, the
success path certified only on a discarded copy, no persistent
index. **Limitations (documented):** an externally-referenced model
cannot currently become deletable through existing operations — the
five external families (suite runs, samples, sample-quality,
recipes, policies) have no deletion endpoints of their own, so
unblocking requires future explicit dependent-artifact operations (a
deliberate M67 boundary, per the no-cascade rule); recipes and
policies are immutable registries by design. **The lifecycle-safety
lattice is now complete:** datasets (M65), tokenizers (M65),
checkpoints (M61), models (M67) — every deletion in the system is
explicit, verified, dependency-safe and atomic.

**Next milestone — M68, explicit suite-run & sample lifecycle.** The
natural next rung (the M67 spec itself anticipated "future explicit
dependent-artifact operations"): give the root-level families that
guard models their own verified deletion — `DELETE
/suite-runs/{suite_run_id}` and `DELETE
/models/{model_id}/samples/{sample_id}` — following the identical
pattern (scope → integrity → atomic removal, with sample-quality
measurements treated per their real persistence semantics), which for
the first time creates a legitimate path to unblock a referenced
model. The ready-to-paste prompt follows.

---

### M68 — EXPLICIT SUITE-RUN & SAMPLE LIFECYCLE (ready-to-paste prompt)

```
Implement **Milestone 68 — Explicit Suite-Run & Sample Lifecycle** for
AI Model Forge.

M67 completed the lifecycle-safety lattice for models, datasets,
tokenizers and checkpoints. But an externally-referenced model can
currently NEVER become deletable: its blockers (suite runs, samples,
sample-quality measurements, recipes, policies) are root-level or
registry families with no deletion operations of their own. M68 gives
the two runtime-record families — suite runs and samples — their own
explicit verified deletion, following the established M61/M65/M67
pattern. Recipes and policies are immutable registries BY DESIGN and
stay undeletable.

### 1. INSPECT FIRST — DO NOT MODIFY YET

Before changing anything, inspect and identify the authoritative
implementations for:

* the M67 model deletion guard (scope → integrity-first → the ONE M66
  analysis → atomic_delete_dir → typed result/409) — the pattern to
  replicate;
* the M65 dataset/tokenizer guards and retention views;
* the suite-run family: storage layout (suite-runs/<suite_run_id>/),
  SuiteRunRecord fields, what a suite run OWNS (its record; its probe
  EVALUATIONS live in models/<id>/evaluations/ — model-owned, not
  suite-owned), creation/listing/get routes, and every record that
  references a suite_run_id;
* the sample family: storage layout (samples/<model_id>/sample-<id>/),
  SampleRecord fields, the sample-quality records that reference a
  sample (sample-evaluations/<model_id>/), and how samples reference
  models+checkpoints;
* what integrity verification exists or can be honestly defined for
  these record families (manifest parse at minimum; be explicit about
  what "integrity" means for a record with no weights);
* the OpenAPI count conventions (expect 93 paths, 641 tests) and the
  README/landing conventions.

Record the pre-implementation baseline.

### 2. M68 OBJECTIVE

Two new explicit, verified, atomic deletions:

DELETE /suite-runs/{suite_run_id}
DELETE /models/{model_id}/samples/{sample_id}

* suite-run deletion: scope (unknown/unparseable -> 404, nothing
  deleted) -> integrity (the record must load and be self-consistent)
  -> atomic removal of the suite run's OWN directory. The probe
  evaluations it triggered are MODEL-OWNED (inside models/<id>/) and
  are NOT deleted, NOT rewritten — no cascade; they merely lose
  nothing (they never referenced the suite run id unless inspection
  proves otherwise — inspect first).
* sample deletion: scope -> integrity -> blockers per ACTUAL
  persistence semantics (a sample-quality measurement referencing the
  sample is the candidate blocker — decide from inspection whether it
  blocks, or whether sample deletion must handle it explicitly; do
  not cascade silently) -> atomic removal of the sample's own
  directory.
* both: typed results ({id, files_removed, bytes_reclaimed}), typed
  structured 409 blocker details where blockers exist, zero
  mutation on refusal, no force/bulk/policies.
* after deleting the referencing records, a previously blocked model
  MUST become deletable through live recompute (the M67 guard reads
  the ONE M66 analysis live) — this is M68's payoff and must be
  tested end-to-end: suite run deleted -> model's usage/retention
  shrink -> model deletable.

### 3. DESIGN CONSTRAINTS

* REUSE the established patterns and primitives: _scope-style
  resolution, verify-before-mutate ordering, _artifact_files for
  stats, atomic_delete_dir for removal, the typed 409 convention.
* Do NOT introduce a second scanner for any family; if a usage view is
  needed, derive it from the ONE listings.
* Do NOT touch recipes/policies/probe-suite registries (immutable by
  design), datasets, tokenizers, checkpoints, or model deletion
  semantics (M67's guard must remain byte-for-byte its current
  behavior — only its INPUT set can shrink as references disappear).
* No cascade, no force, no bulk, no policies, no automation.

### 4. TEST MATRIX

Focused tests following existing style, at minimum:

* suite-run deletion: success with exact stats vs an independent
  walk; only its directory removed; probe evaluations of the model
  remain (count unchanged); unknown -> 404; repeat -> 404; refusal
  cases if inspection finds real blockers;
* sample deletion: success with exact stats; sample-quality blocker
  semantics per the inspected decision (both directions tested);
  unknown sample / unknown model -> 404; zero mutation on refusal;
* the unblock chain end-to-end: a model blocked ONLY by a suite run
  becomes deletable after the suite run's deletion (M66 usage, M67
  retention and the DELETE all agree, live); same for a model blocked
  only by sample/sample-quality;
* HTTP: 200 results, typed 409s, 404s, determinism, zero mutation,
  OpenAPI (two new DELETE paths: 93 -> 95), README + landing updates;
* full regression: the complete suite must stay green (M61-M67
  semantics untouched).

### 5. LIVE CERTIFICATION

Production currently has NO suite runs or samples (M66/M67 certified
counts 0) — so the production phase is read-only coherence as usual
(inventory, usage/retention unchanged, unknown-id 404s), and the
success paths are certified on a DISPOSABLE COPY: copy the production
root, create a suite run + sample + quality measurement there (the
copy has a model, dataset and tokenizer to drive), verify the model is
blocked, delete the referencing records through the new routes,
verify the model becomes deletable, delete it, verify exact
accounting, discard the copy. Production must remain byte-identical.
Run the certification smoke exactly once; if the script has an
assertion bug, stop, verify read-only, fix, rerun.

### 6. REPORT

Report exactly these 9 sections: Inspection & Baseline; Suite-Run &
Sample Retention Architecture; Deletion Safety; Unblock-Chain
Coherence (M66/M67 agreement); Verification; Live Certification;
Storage Proof; Git; Final Status & Next Milestone — then provide the
next ready-to-paste implementation prompt.

### HARD CONSTRAINTS

* Inspect first; record the baseline.
* Reuse the established guard/verify/atomic pattern — no new
  deletion primitives, no second scanners.
* No cascade: model-owned probe evaluations are never touched by
  suite-run deletion; the sample-quality decision must be explicit,
  inspected and tested, not silent.
* M67's model guard code must not change; only its live inputs may
  shrink.
* Recipes, policies and probe-suite definitions remain immutable.
* Integrity before mutation; atomic removal; deterministic typed
  results; unknown -> 404.
* No force, no bulk, no policies, no automation, no GC.
* Production byte-identical; destructive certification only on a
  discarded copy.
* Preserve every previous milestone.
```
