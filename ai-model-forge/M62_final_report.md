# Milestone 62 — Read-Only Retention Overview: Final Report

**Date:** 2026-09-10 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

`GET /models/{model_id}/checkpoints/retention` makes the M61 retention
state visible BEFORE any deletion attempt — per-checkpoint
deletability + ordered blockers from the ONE M61 analysis, plus
reclaimable totals. The live smoke certified it read-only against the
rebuilt production root: **16/16, zero mutations, byte-identical
storage**.

> **ENVIRONMENT EVENT (disclosed):** between M61 and M62 the sandbox
> was re-provisioned. The Git repository — all 61 milestones — was
> fully recovered from the remote (`git reset --mixed FETCH_HEAD` onto
> `229f860`; the working tree was byte-identical, only the known
> untracked `egg-info` remained). The Python environment was rebuilt
> (torch 2.14.0+cu130, same as before; full suite re-verified). The
> **production data directory was lost** — it deliberately lived
> outside the repository and was never committed. A NEW production
> root was reconstructed through the public engine facade
> (`m62_production_rebuild.py`, committed as evidence), mirroring the
> documented shape of the original trajectory; its facts are certified
> below. No number from the lost root is claimed as current.

---

## §1 — Baseline & inspection

**Baseline:** full suite **613 passed** (exit 0) on the restored
repository + rebuilt environment; HEAD `229f860` (M61, pushed); OpenAPI
**85 paths**; production root **rebuilt** (see above): **52 files /
8,926,415 B / 0 tmp**, **13 checkpoints**, best == published ==
`650965025ae1` @ 5.996844, 10 deletable / 3 protected, 6 runs, 3
workflows, 4 evaluations, 2 comparisons, 2 gates, 1 recipe, 13 M59
history entries.

**M61 reference architecture reused (one of each, no second
implementation):** `ModelForge.checkpoint_blockers` +
`CHECKPOINT_BLOCKER_ORDER` (the ONE blocker analysis and its canonical
ordering), `TrainingEngine.verify_checkpoint` (the ONE integrity
verifier), `TrainingEngine.list_checkpoints` (the authoritative listing
and its canonical (step, created_at) order), M61's artifact-set
measurement inside `remove_checkpoint` (extracted into the ONE
`checkpoint_artifact_stats`, now shared by removal and overview), and
the M3/M52/M60 state pointers. Route conventions: the M59
`best/history` pattern (read-only, model-scoped, computed live,
declared BEFORE the generic `{checkpoint_id}` capture, empty-collection
semantics).

## §2 — Implementation

Five files changed (+~1,050 lines incl. tests), no new modules:

- **`app/schemas.py`** — `CheckpointRetentionEntry`
  (checkpoint_id/run_id/step/created_at/validation_loss/files/
  size_bytes/integrity_verified/deletable/blockers — the blockers reuse
  the M61 `CheckpointDeletionBlocker` model verbatim) and
  `CheckpointRetentionOverview` (model_id + 5 aggregates +
  checkpoints[]).
- **`app/training.py`** — `checkpoint_artifact_stats(model_id,
  ckpt_id)`: the ONE (files, bytes) measurement, extracted from
  `remove_checkpoint` (which now calls it — M61's numbers unchanged,
  its tests green).
- **`app/engine.py`** — `checkpoint_retention_overview(model_id)` on
  the facade: iterates the authoritative listing in canonical order;
  per checkpoint calls the ONE `checkpoint_blockers`, the ONE
  `verify_checkpoint`, and the ONE stats helper; `deletable =
  integrity_verified and not blockers` — exactly the M61 guard's
  decision; aggregates are deterministic sums.
- **`app/api.py`** — ONE route
  `GET /models/{model_id}/checkpoints/retention` (404 unknown model;
  empty+zeroed for a valid model without checkpoints), declared before
  the generic `{checkpoint_id}` detail; landing-page feature bullet +
  route-list line.
- **`README.md`** — Milestone 62 section; counts 613 → 617.
- Evidence: **`smoke_m62_live.py`**, **`m62_production_rebuild.py`**,
  **`m62_pre.sha256`**.

No new engines, no stored index, no cache, no policy, no bulk
operations. The blocker logic stayed in the facade (never the API
layer); the API layer contains no filesystem traversal.

## §3 — Retention view

Per checkpoint (canonical M3 (step, created_at) listing order): the
persisted identity (checkpoint_id, run_id, step, created_at,
validation_loss — verbatim `CheckpointRecord` fields), the artifact-set
file count and byte size (the same measurement a deletion reports),
the M3 integrity-verification outcome, `deletable` (exactly what an
immediate M61 DELETE would decide), and the ordered blockers (the same
categories/details/ordering as the M61 409). Aggregates:
`total_checkpoints`, `deletable_checkpoints`, `protected_checkpoints`
(= total − deletable; referenced OR integrity-failed),
`total_checkpoint_bytes`, `reclaimable_checkpoint_bytes` (ONLY
currently-deletable checkpoint artifact sets — never model weights,
tokenizer, dataset or record storage). Deterministic: byte-identical
JSON over unchanged state. Unreadable-manifest checkpoints are
listing-invisible (the established M61 corruption semantics); lineage
metadata and the M59/M8 computed views remain non-blocking (the M61
contract, unchanged).

## §4 — Consistency

M52: exactly ONE entry carries the `best` blocker, and it equals
`GET .../checkpoints/best` (tested engine-level and over HTTP).
M60: the `published` entry equals the manifest's `latest_checkpoint`;
a rollback moves it and the overview recomputes (tested); drift — when
best ≠ published both appear protected for their respective reasons
(tested; in the rebuilt production root the standalone M60-style
publish closed the loop's drift, so best == published there, protected
as exactly `[best, published]` — certified in the smoke). M61: the
headline invariant is tested in BOTH directions — every overview
`blockers` list equals the M61 DELETE 409 `detail.blockers` exactly
(HTTP test), every deletable entry deletes with exactly the reported
files/bytes, every protected entry is refused, and a corrupt
checkpoint flips to `integrity_verified=false / deletable=false` while
M61 refuses it with the integrity error (never a silent 200).

## §5 — Tests

**Targeted (4 new):** `test_m62_overview_basics_order_totals_
determinism` (ordering/identity parity, aggregate arithmetic,
determinism, ZERO writes proven by a byte-level root inventory,
empty/unknown models, M52/M60 single-best/published consistency);
`test_m62_deletability_and_blockers_match_m61` (the full reference
landscape — evaluation/comparison/gate/suite/sample/sample-quality/
workflow references + manifest best-known + best/published — corrupt
-flip-restore on a deletable historical M59 winner, then both parity
directions with exact files/bytes and refusal reasons);
`test_m62_overview_recomputes_after_state_changes` (rollback moves the
published blocker; a new reference protects a deletable checkpoint; a
deletion shrinks the view — live recompute, no cache);
`test_m62_api_retention_overview` (HTTP: shape/order/totals, M52/M60
agreement over the public routes, **the 409-vs-overview blocker
exact-equality**, byte-equal repeat, real deletion matching the entry,
unknown/empty models, OpenAPI).

**Full suite: 613 → 617, ×2 consecutive runs on the final state**
(exit 0 both). **Statics:** pyflakes clean (app/, tests/, both
scripts), compileall clean. **OpenAPI:** **85 → 86** (exactly the one
new GET; 46 existing count assertions bumped; response + entry schemas
exposed; route declared before the generic capture — verified in the
OpenAPI JSON ordering).

## §6 — Live smoke

Executed **exactly once** (uvicorn, port 8784, rebuilt production
root; stopped immediately after): **16/16 PASSED on the first
execution — zero corrections, ZERO state-mutating calls** (the server
log shows only GETs). Certified against an INDEPENDENT oracle computed
in the smoke from the filesystem + record manifests (not via the
app's analysis): totals (13 / 10 / 3 / 8,197,262 / 6,305,615),
canonical ordering + persisted-identity parity, the deletable set
(exact equality with the oracle), reclaimable bytes, best == published
`650965025ae1` protected as exactly `[best, published]`, one safe
checkpoint (deletable, zero blockers), one artifact-referenced
checkpoint (protected with its record reasons), a deterministic
byte-identical repeat, and storage byte-identical against
`m62_pre.sha256`.

## §7 — Storage

**Before: 52 files / 8,926,415 B. After: 52 files / 8,926,415 B —
byte-identical (sha256-verified, 52/52), 0 tmp, 0 residue, 0 servers
left running.** M62 added zero persistent storage and performed zero
mutations: the overview is a computed view. (The production root
itself is the documented rebuild — see the disclosure above; its
construction script and pre-inventory are committed as evidence.)

## §8 — Git

- `M62: read-only retention overview` — implementation, tests,
  README/landing docs, the smoke, the production-rebuild evidence
  script, this report.
- `M62: pre-smoke production inventory (m62_pre.sha256)` — the
  rebuilt root's pre-smoke inventory (52 files).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch tip
  == HEAD; worktree clean (only the untracked `egg-info`).

## §9 — Next milestone

Inspection of the remaining architecture: the retention arc is now
complete end-to-end — safe deletion (M61), visible state (M62), the
improvement loop (M53–M60), observability (M52/M59/M8) and bounded
repetition (M58). Everything the operator can see is MODEL-scoped; the
one blind spot left is the PROJECT scope: with multiple models the
operator cannot answer "what does the whole project store, per model
and per family, and how much is reclaimable" without walking every
model. The smallest architecturally-consistent next milestone is
**M63 — read-only project storage overview**: one computed-live,
zero-storage route summarizing every model in the registry —
per-model checkpoint/record counts, weights size, and the M62
retention aggregates (via the SAME analyses, never a second scanner) —
plus project-level totals. It completes the observability ladder
(checkpoint → model → project) and informs future explicit retention
decisions, with no autonomy, no policies and no bulk operations.

---

### M63 — READ-ONLY PROJECT STORAGE OVERVIEW (ready-to-paste prompt)

```
Implement **Milestone 63 — Read-Only Project Storage Overview** for AI
Model Forge.

M62 made per-model retention visible. The project scope is still
blind: with multiple models registered, there is no single view of
what the whole project stores — per model and per family — or how much
is reclaimable in total.

### 1. INSPECT FIRST — DO NOT IMPLEMENT YET

Before changing anything, inspect and identify the authoritative
implementations for:

* the model registry (storage.model_ids / list_models) and the model
  listing route's conventions (ordering, fields, 404s);
* the M62 retention overview (checkpoint_retention_overview — the ONE
  per-model analysis this milestone must REUSE, never re-implement);
* the M59/M52/M46 listing semantics (candidate counts, empty
  collections);
* the M8 dashboard's family scans (evaluations/comparisons/gates/
  workflows/suite runs/samples per model) — reuse the same listing
  engines, not new scans;
* the storage layer's on-disk layout for weights/tokenizers/datasets
  (what a "family byte size" can honestly mean — prefer authoritative
  record metadata over filesystem traversal where the architecture
  already defines it);
* the OpenAPI count conventions and the README/landing conventions.

Run the current full test suite before implementation and record the
baseline (expect 617 passed, OpenAPI 86 paths).

### 2. M63 OBJECTIVE

One read-only, project-scoped, live-computed route:

GET /storage/overview (or the repository's exact established prefix/
naming convention)

reporting, WITHOUT any mutation:

* project identity (root path is NOT exposed unless existing
  conventions already do so);
* per model (deterministic order — the registry's canonical order):
  model id, name, config hash, created_at, checkpoint count, record
  counts per family (evaluations, comparisons, gates, workflows,
  suite runs, samples, sample evaluations), the published/latest and
  M52 best checkpoint ids where they exist, weights size, and the M62
  retention aggregates (deletable/protected counts, total and
  reclaimable checkpoint bytes) — via the ONE existing analyses;
* project-level totals (models, checkpoints, deletable/protected,
  total and reclaimable checkpoint bytes).

### 3. DESIGN CONSTRAINTS

* REUSE: the M62 overview per model (the ONE retention analysis), the
  existing listing engines for counts, the storage layer for sizes.
  No second scanner, no second retention analysis, no new registry.
* Read-only, zero storage, zero mutation, deterministic
  (byte-identical over unchanged state); no cache, no pointer.
* A project with no models -> the established empty-collection
  behavior. Unknown-model 404s do not apply (project scope).
* Keep the response compact and typed; per-model entries must not
  duplicate entire manifests — identity + counts + aggregates only.
* Sizes must be honest: checkpoint bytes come from the M62 measurement;
  weights bytes from the storage layer's own file; do NOT invent
  derived statistics beyond deterministic sums.
* Do NOT add: deletion, policies, automation, cross-model operations,
  or any write path.

### 4. TEST MATRIX

Focused tests following existing style, at minimum:

* multi-model project: per-model rows match each model's OWN M62
  overview and listing counts exactly (parity test);
* totals are exact sums over the rows;
* deterministic ordering and byte-identical repeat calls;
* zero writes (byte-level inventory comparison);
* empty project (fresh root) -> empty collection semantics;
* a model with no checkpoints -> zeroed retention aggregates;
* consistency: each row's best/published match the per-model routes;
* OpenAPI: exactly one new path (86 -> 87, reason reported); README +
  landing updates; full regression suite green.

### 5. LIVE CERTIFICATION

After the full suite passes twice and statics are clean, run exactly
one READ-ONLY live smoke against production: capture the inventory,
query the project overview over HTTP, verify every per-model row
against that model's own M62 route and listing routes (independent
cross-check), verify the totals, repeat the call for byte equality,
and verify the inventory is byte-identical afterwards. Zero
state-mutating calls of any kind.

### 6. REPORT

Report exactly these 9 sections: Baseline & inspection; Implementation;
Analysis reuse; Tests; Live smoke; Storage & integrity; API/OpenAPI;
Git; Next milestone — then provide the next ready-to-paste
implementation prompt.

### HARD CONSTRAINTS

* Inspect first.
* Read-only only; zero storage; zero mutation.
* One new route; no new scanners or registries.
* Reuse the ONE M62 retention analysis and the existing listing
  engines.
* No deletion, no policies, no automation, no cross-model writes.
* Preserve every previous milestone.
* The authoritative invariant is: **every per-model row is exactly
  what that model's own authoritative routes (M62 retention, M52 best,
  listings) report, and the project totals are exact deterministic
  sums over the rows.**
```
