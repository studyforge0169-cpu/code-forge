# Milestone 61 — Explicit Verified Checkpoint Retention: Final Report

**Date:** 2026-09-10 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

One explicit, verified, reference-aware deletion primitive now exists:
`DELETE /models/{model_id}/checkpoints/{checkpoint_id}` — and the live
smoke reclaimed the first-ever production checkpoint (**`4772821d72c9`**,
the bootstrap best, 630,554 B) with a byte-perfect audit: exactly its 2
files removed, **nothing else touched**.

---

## §1 — Baseline & inspection

**Baseline (before any edit):** full suite **606 passed** (exit 0);
HEAD `683c8bb` — M60's two commits, **pushed at the start of this
milestone** (the M59-era token failure had cleared); worktree clean;
OpenAPI **85 paths**; production **79 files / 15,283,812 B / 0 tmp**,
**23 checkpoints**, best == published == `41870af53ab7` @ 6.242401
(post-M60), legacy manifest `best_checkpoint` = `eaf0984fbc2b`.

**Reference/dependency architecture discovered (the load-bearing
inspection):**

| Family | Persisted checkpoint references |
|---|---|
| Model manifest (authoritative state) | `latest_checkpoint` (published), `best_checkpoint` (stored best-known weights ref) |
| Checkpoint manifests (M3) | `parent_checkpoint_id` — and M3 **chains checkpoints within every run** (each save's id becomes the next save's parent; the run's initial is the first parent) |
| Run provenance (model manifest) | `initial`/`final`/`rolled_back_to` + the full `TrainingConfig` (incl. `resume_from_checkpoint_id`) |
| Workflow records (M7) | plan pins (M53 refs, M55 resume, M56 evaluate, M57 baseline, M60 publish — explicit + resolved) and stage artifacts (`checkpoint_id`: training finals, evaluated states, publications) |
| M4/M5/M6/M10/M15/M16 records | `EvaluationRecord.checkpoint_id`; `ComparisonSide.checkpoint_id` (both sides); `GateDecision` candidate/baseline sides + `suggested_checkpoint_id`; `SuiteRunRecord.state`; `SampleRecord.checkpoint_id` (required); `SampleEvaluationRecord.state` |
| Computed views (no storage) | M52 selection, M59 history, M8 dashboard — the dashboard **explicitly tolerates missing references** (a `DashboardDiagnostic` per dangling edge; valid artifacts stay visible) |

The facade already exposes authoritative per-family checkpoint filters
(`list_evaluations_for_checkpoint` M24, `list_comparisons_for_checkpoint`
M26, `list_suite_runs_for_checkpoint` M25, `list_samples_for_checkpoint`
M27, `list_sample_evaluations_for_checkpoint` M20) — reused verbatim as
the reference scanners. `_ckpt_dir` joins ids without traversal
sanitization (consistent across every existing checkpoint route);
deletion scoping therefore goes through the listing (ids only ever come
from real registry entries). All engines are synchronous/process-local;
`Storage.remove` (whole model) is a plain rmtree — no atomic
directory-delete primitive existed.

## §2 — Implementation

Seven files (+~1,050 lines incl. tests), no new modules:

- **`app/storage.py`** — `atomic_delete_dir(path)`: the smallest
  reusable delete primitive (one `os.rename` to a hidden
  `.tmp-delete-<uuid>` sibling the listings already skip, then rmtree) —
  the mirror of the atomic writers, independently tested.
- **`app/schemas.py`** — `CheckpointDeletionResult`
  (model_id/checkpoint_id/files_removed/bytes_reclaimed) and
  `CheckpointDeletionBlocker` (reason/detail).
- **`app/training.py`** — `remove_checkpoint(model_id, ckpt_id)`: the
  low-level M3 atomic removal (sizes → `atomic_delete_dir`), documented
  as deliberately NOT deciding deletability.
- **`app/engine.py`** — the facade (the one integration point where
  every family's engine is visible): `CHECKPOINT_BLOCKER_ORDER`
  (canonical deterministic order), `_workflow_checkpoint_refs`
  (plans + artifacts of one record), `checkpoint_blockers(model_id,
  ckpt_id)` (the LIVE analysis) and `delete_checkpoint(model_id,
  ckpt_id)` (the fixed-order guard: listing scope → M3 integrity
  verification → blockers → atomic removal).
- **`app/api.py`** — one route:
  `DELETE /models/{model_id}/checkpoints/{checkpoint_id}` (declared
  next to the existing GET detail on the same path); 404 unknown /
  listing-invisible; 409 integrity (RuntimeError from the verifier);
  409 protection with a structured ordered blocker detail
  (`{message, model_id, checkpoint_id, protected, blockers[]}`);
  landing-page feature bullet + route-list line.
- **`README.md`** — Milestone 61 section, counts 606 → 613, the
  storage-efficiency rule amended (checkpoints removable ONLY through
  the explicit verified route).
- **`tests/test_training.py` + `tests/test_training_api.py`** — 7 new
  tests; **`smoke_m61_live.py`** + **`m61_pre.sha256`** (evidence).

## §3 — Reference safety

**Blocking (authoritative dependencies), in canonical order:**

1. `best` — the live M52 selection (hard constraint; computed live via
   the ONE selector).
2. `published` — the manifest's `latest_checkpoint` (the live state).
3. `manifest_reference` — the manifest's `best_checkpoint` (a stored
   authoritative weights pointer; nothing resolves it mechanically, but
   dangling authoritative state is never created).
4. `workflow_reference` — every pin/explicit id in immutable workflow
   records: plan `StageStateRef`s (explicit + M53-resolved), M55/M56/M57
   resolved ids, M60 publish ids, and stage artifacts' `checkpoint_id`
   (training finals, evaluated states, publications) — the replay/audit
   trail.
5–10. `evaluation_reference`, `comparison_reference` (either side),
`gate_reference` (candidate/baseline/suggestion),
`suite_run_reference`, `sample_reference`,
`sample_quality_reference` — the persisted checkpoint ids of the
immutable M4/M5/M6/M10/M15/M16 EVIDENCE records, scanned via the
existing read-only `..._for_checkpoint` filters (one source of truth,
no stored index, no second registry).

**Deliberately NOT blocking (inspected + documented):** pure lineage
metadata — a surviving checkpoint's `parent_checkpoint_id` and run
provenance (`initial`/`final`/`rolled_back_to`) — and the computed
views (M59 history, M8 dashboard). Rationale: (a) lineage is
informational history — nothing resolves it to state, and the
dashboard's diagnostic design explicitly tolerates a missing target;
(b) **blocking it would make retention dead code**: M3 chains
checkpoints within every run (each checkpoint is its successor's
parent; each run's final is the provenance final), so lineage-blocking
protects every checkpoint ever created — contradicting the required
"ordinary checkpoint must succeed" case. Appearing in M59 history
protects nothing; the history recomputes over survivors. Multiple
blockers are returned together, deterministically ordered (canonical
category order, then detail).

## §4 — Deletion & atomicity

The guard runs in a fixed order (route → facade → engines):
**(1)** scope via the authoritative listing — unknown model/checkpoint,
or an unreadable manifest (the listing skips it) → 404, nothing
deleted; **(2)** integrity via the EXISTING M3 verifier
(`verify_checkpoint`: weights load + content-hash check) — corrupt or
missing weights → 409, refused (deletion never bypasses integrity
validation; a malformed-manifest checkpoint is listing-invisible →
404); **(3)** the LIVE blocker analysis → 409 with the ordered
structured blockers; **(4)** removal: `TrainingEngine.remove_checkpoint`
→ `atomic_delete_dir` — ONE `os.rename` of the checkpoint directory to
a hidden sibling, then rmtree. No observer can ever see a half-deleted
checkpoint (the registry scan skips dot-dirs by convention); a crash
can only leave hidden residue, never a valid-looking partial one —
exactly the established crash-window semantics of the atomic writers.
The registry IS the directory listing, so removal needs no
manifest/pointer updates: the model manifest, `weights.pt`, every
record family and every surviving checkpoint are untouched (proven
byte-level in tests and the smoke). Repeated deletion → 404 (the
established not-found). Concurrency (§12): all engines are synchronous
and process-local; the atomic rename guarantees per-operation
consistency; concurrent conflicting requests fail cleanly with
deterministic errors rather than corrupting state — documented as a
process-local constraint, no locking subsystem added.

## §5 — Tests

**Targeted (7 new):**

| Test | Covers |
|---|---|
| `test_m61_atomic_delete_dir_primitive` | the primitive directly: gone, no residue, raises on already-gone, listings skip hidden siblings |
| `test_m61_ordinary_deletion_succeeds` | **the primary acceptance**: superseded mid-run checkpoint (a former M59 winner) deleted; exact (2 files, exact bytes); registry/listing; GET 404; best/published/weights/model-manifest byte-identical; siblings verbatim; M59 recomputed (ids filter, deltas monotone, first None); repeated delete 404 |
| `test_m61_protection_best_published_manifest` | best / published / best==published (ordered `[best, published]`) / the manifest's best-known reference (forced distinct via fixture edit); every rejection deleted nothing |
| `test_m61_reference_families_block` | all seven evidence families on ONE checkpoint (workflow plan+artifact via an EVALUATE stage, M4, M5 side, M6 baseline, M10, M15, M16) → exact ordered 8-entry blocker list; deterministic across calls; every referencing record still resolves |
| `test_m61_lineage_and_history_non_blocking` | the documented policy: a parent (likely former winner) with zero blockers deletes; the surviving child's manifest + model manifest stay byte-identical; M46/M52/M59 coherent |
| `test_m61_corruption_malformed_missing` | corrupt weights → RuntimeError; missing weights → RuntimeError; malformed manifest → 404 (untouched); restored → deletable; unknown model/checkpoint; cross-model scoping; never-trained model |
| `test_m61_api_explicit_checkpoint_retention` | the HTTP surface end-to-end: 404s, 409 structured blockers (best first), rejection mutates nothing, 200 deterministic result, verbatim listing equality, M52/M59/published coherence, weights hash unchanged, OpenAPI |

**Full suite: 606 → 613, ×2 consecutive runs on the final state**
(exit 0 both). One mid-development correction: the tie-fixture-style
lesson repeated — a workflow fixture using a PUBLISH stage actually
PUBLISHED the victim (switched to an EVALUATE-stage reference);
and the OpenAPI count needed NO bump (the DELETE rides the existing
checkpoint detail path — 45 assertions verified unchanged at 85).
**Statics:** pyflakes clean (app/, tests/, smoke), compileall clean.
**OpenAPI:** **85 paths unchanged**; the `delete` operation joins the
existing `get` on `/models/{model_id}/checkpoints/{checkpoint_id}`;
`CheckpointDeletionResult` exposed with exactly the four result fields.

## §6 — Live smoke

Executed **exactly once** (uvicorn, port 8783, production root; server
stopped immediately after). Target chosen from the inspection's
deletable set (14 candidates) and certified safe THREE independent
ways before mutation: not best/published/manifest-best, zero
references in an independent scan of every record family, and
`checkpoint_blockers() == []`.

**Target:** `4772821d72c9` (step 2, run `15fa1b096305`, loss 6.452705 —
**the FIRST M59 winner**, superseded since the second checkpoint ever
saved; the §9 historical-best case by construction).

**In-run (authoritative server log):** prelude GETs 200 (model,
best, history — 15 entries / 23 candidates, registry 23, victim
detail); the two protected DELETEs → **409** (best==published with
ordered blockers `[best, published, …]`; the legacy manifest reference
with `manifest_reference`); rejections deleted nothing; then the ONE
mutation — `DELETE .../checkpoints/4772821d72c9` → **200**, engine
logging exactly *"deleted checkpoint 4772821d72c9 (2 files, 630554
bytes)"*; victim GET → **404**; repeated DELETE → **404**; post GETs
200.

**Honest correction:** the smoke script crashed at its V12 residue
assertion on a bug in the CHECK code (`os.listdir` returns `str`, not
`Path`) — after the mutation was already complete and correct. The
check is fixed in the committed script (with an execution-record note)
and **every remaining fact was re-verified READ-ONLY post-hoc:
17/17** — registry 22, best `41870af53ab7` @ 6.242401 unchanged,
published unchanged, M59 history recomputed to exactly the 14 expected
survivor winners (first delta None, M58 tail intact, final entry ==
live best), blocker sets re-confirmed, and the byte-level inventory
diff. No production state was mutated after the single deletion.

## §7 — Storage

| | before | after | delta |
|---|---|---|---|
| files (excl. tmp) | 79 | **77** | **−2** |
| bytes | 15,283,812 | **14,653,258** | **−630,554** |
| tmp entries | 0 | 0 | 0 |
| checkpoints | 23 | 22 | −1 |

Every byte accounted: the removed files are exactly
`checkpoints/4772821d72c9/manifest.json` (558 B) and
`checkpoints/4772821d72c9/weights.pt` (629,996 B) — matching the API's
`files_removed: 2, bytes_reclaimed: 630554`. **Zero new files, zero
modified files** (sha256-verified against `m61_pre.sha256`): the model
manifest, `weights.pt`, its sidecar, all 44 surviving checkpoint files
and every record family are byte-identical. No orphan temps, no
residue. The reclaimed 630,554 B is real production reclamation of a
superseded state — the first in the project's history.

## §8 — Git

- `M61: explicit verified checkpoint retention` — implementation,
  tests, README/landing docs, smoke script (with execution record),
  this report.
- `M61: pre-milestone production inventory (m61_pre.sha256)` — the
  pre-smoke inventory marker (79 files, pre-deletion state).
- Both pushed to `origin/arena/01a071e9-code-forge` (the M60 commits
  `6623559`/`683c8bb` were pushed at the start of this milestone after
  the GitHub connection was restored); `HEAD == FETCH_HEAD`, worktree
  clean (only the untracked `ai_model_forge.egg-info/`).

## §9 — Next milestone

Inspection of the remaining architecture: the loop is complete
(M53–M60), observable (M52/M59/M8), bounded-repeating (M58) and now
storage-controlled (M61) — but retention is **blind**: the only way to
learn what is reclaimable, or what protects a given checkpoint, is to
attempt a deletion and read the 409. The M61 engine analysis
(`checkpoint_blockers` + the per-family reference scans) already
computes the complete answer; nothing exposes it read-only. The
smallest high-value next milestone is **M62 — read-only retention
overview**: one computed-live, model-scoped route
(`GET /models/{id}/checkpoints/retention`) that reports, for every
checkpoint in the authoritative listing, its persisted identity, its
on-disk size, whether it is deletable, and its ordered blockers (the
SAME analysis, no second scanner) plus model-level totals (reclaimable
files/bytes, protected counts by category) — the M59-pattern
observability counterpart of the M61 mutation, with zero storage, no
bulk operations, and no policy automation.

---

### M62 — READ-ONLY RETENTION OVERVIEW (ready-to-paste prompt)

```
Implement **Milestone 62 — Read-Only Retention Overview** for AI Model
Forge.

M61 added explicit verified checkpoint retention
(DELETE /models/{id}/checkpoints/{checkpoint_id}) with a live
reference-safety analysis. But retention is currently blind: the only
way to learn what is reclaimable — or what protects a specific
checkpoint — is to attempt a deletion and read the 409.

### 1. INSPECT FIRST — DO NOT IMPLEMENT YET

Before changing anything, inspect and identify the authoritative
implementations for:

* the M61 blocker analysis (`ModelForge.checkpoint_blockers`,
  `CHECKPOINT_BLOCKER_ORDER`, `_workflow_checkpoint_refs`) and the
  delete guard;
* the M3 checkpoint listing and per-checkpoint on-disk artifact layout
  (files + sizes);
* the M52 selection and the M52/M59 response conventions
  (criterion/candidate_count fields);
* the read-only route conventions (M24/M26/M25/M27/M20
  ..._for_checkpoint routes; M59 history route) — declaration order,
  404/empty semantics, deterministic ordering;
* the README/landing documentation conventions;
* the OpenAPI count test conventions.

Run the current full test suite before implementation and record the
baseline (expect 613 passed, OpenAPI 85 paths, HEAD = the M61 commits).

### 2. M62 OBJECTIVE

One read-only, model-scoped, computed-live view answering "what can I
reclaim, and what protects what":

GET /models/{model_id}/checkpoints/retention

with:

* per checkpoint (in the authoritative M3 listing order): checkpoint
  id, step, run id, persisted validation_loss, on-disk file count and
  bytes, whether it is deletable (the M61 guard's answer), and its
  ordered blockers (same categories, same details as the 409);
* model-level summary: total checkpoints, deletable count, reclaimable
  files/bytes, protected counts by blocker category.

Do NOT add: any delete/bulk-delete operation, retention policies,
keep-N rules, age rules, automation, background jobs, or a second
reference scanner. The overview must reuse the ONE M61 analysis.

### 3. DESIGN CONSTRAINTS

* The analysis must be the SAME one the delete guard uses (one source
  of truth — factor shared internals if needed WITHOUT changing the
  guard's semantics; the M61 tests must pass unchanged).
* Deletability in the overview must EQUAL deletability in the guard
  (parity is a required test: every checkpoint reported deletable
  deletes successfully in a throwaway copy; every protected one is
  refused with the same ordered blockers).
* Zero storage: computed live on every call; no cache, no pointer, no
  record, no second registry.
* Deterministic: byte-identical responses over unchanged storage; the
  listing's canonical (step, created_at) order; blockers in the
  canonical M61 order.
* Semantics: unknown model -> 404; a valid model with no checkpoints
  -> the established empty-collection behavior (200 with an empty
  per-checkpoint list and zeroed totals); read-only — never writes.
* The route must be declared so no other checkpoint path can capture
  it (inspect the existing declaration order conventions).
* Keep the response model minimal and typed; sizes are the actual
  on-disk file sizes (the same measurement the M61 result reports).

### 4. TEST MATRIX

Focused tests following existing style, at minimum:

* overview parity with the M52/M59 views (same registry, same order);
* deletable/protected classification matches checkpoint_blockers and
  actual delete outcomes exactly (both directions, in throwaway
  storage);
* blocker categories/details identical to the 409 detail for the same
  checkpoint;
* totals arithmetic (sums over the per-checkpoint rows);
* determinism (repeated calls byte-identical);
* unknown model 404; empty model 200 + zeroed totals;
* read-only: byte-level inventory unchanged by any number of calls;
* OpenAPI: exactly one new path (85 -> 86, with the reason reported);
  README + landing updates; full regression suite green.

### 5. LIVE CERTIFICATION

After the full suite passes twice and statics are clean, run exactly
one READ-ONLY live smoke against production: capture the inventory,
query the overview over HTTP, verify against independently computed
facts (the current 22-checkpoint registry, the M61 smoke's known
deletable set and known blockers for the protected ids, byte totals),
verify the inventory is byte-identical afterwards, and verify
determinism with a second read-only call. No production mutation of
any kind is permitted in this smoke.

### 6. REPORT

Report exactly these 9 sections: Baseline & inspection; Implementation;
Analysis reuse (how the ONE M61 analysis backs the view); Tests; Live
smoke; Storage & integrity; API/OpenAPI; Git; Next milestone — then
provide the next ready-to-paste implementation prompt.

### HARD CONSTRAINTS

* Inspect first.
* Do not weaken or modify M61's guard semantics.
* One reference-safety analysis, reused — never a second scanner.
* Read-only, zero storage, deterministic.
* No deletion, no bulk operations, no policies, no automation.
* Preserve every previous milestone.
* The authoritative invariant is: **for every checkpoint, the
  overview's deletable flag and ordered blockers are EXACTLY what the
  M61 delete guard would decide.**
```
