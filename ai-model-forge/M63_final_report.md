# Milestone 63 — Read-Only Project Storage Overview: Final Report

**Date:** 2026-09-11 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

`GET /project/storage` lifts storage visibility one level — checkpoint
(M62) → model → **project**: total physical files/bytes, a complete
category partition with **no double counting**, one compact row per
model (its M62 retention aggregates VERBATIM), and project-level
reclaimable totals. Live-certified read-only against production:
**17/17 checks, zero mutations, byte-identical storage**.

> **ENVIRONMENT EVENT (disclosed):** the sandbox was re-provisioned
> AGAIN between M62 and M63 — the second time. The Git repository was
> fully recovered from the remote (`git reset --mixed FETCH_HEAD` onto
> `e6bb63c`; worktree byte-identical to the commit, all 62 milestones
> safe). The Python environment was rebuilt (same versions; baseline
> re-verified 617 passed). The production data directory was lost
> AGAIN (never in git) and was reconstructed a second time through the
> public facade by re-executing the committed
> `m62_production_rebuild.py` (logic-driven, fixed seeds): the
> certified STRUCTURE reproduced exactly — 13 checkpoints, 10
> deletable / 3 protected, best == published @ 5.996844, checkpoint
> bytes 8,197,262, reclaimable 6,305,615, M61 deletion 2 files /
> 630,555 B — with new content-derived ids (model `70aa05e0bee6`) and
> a 52-file / 8,926,421 B root (6 bytes more than the M62 root;
> timestamp precision). Every number below is a fact about THIS root,
> captured in `m63_pre.sha256` before the smoke.

---

## §1 — Baseline & inspection

**Baseline:** full suite **617 passed** (exit 0) on the recovered
repository + rebuilt environment; HEAD `e6bb63c` (M62, pushed); OpenAPI
**86 paths**; production root rebuilt as above: **52 files /
8,926,415→8,926,421 B / 0 tmp**, 1 model, 13 checkpoints.

**Storage ownership model (inspected before any code):** the root is
`project.json` + `tmp/` (atomic-write scratch, startup-cleaned) +
`models/<id>/` (manifest.json + weights.pt + weights.sha256 directly;
`checkpoints/`, `evaluations/`, `comparisons/`, `gates/`,
`workflows/` inside) + the root-level families `datasets/`,
`tokenizers/`, `suite-runs/`, `samples/`, `sample-evaluations/`,
`policies/`, `probe-suites/`, `workflow-recipes/`. Each family's
directory constant lives in exactly ONE module
(`CHECKPOINTS_DIR`, `EVALUATIONS_DIR`, … — imported, never
redefined). Model enumeration: `Storage.model_ids()` (physical,
sorted) vs `list_models()` (record-valid, created_at order, corrupt
manifests skipped — the registry convention). No
content-addressed dedup exists; a checkpoint artifact set is exactly
its directory (manifest + weights). Existing summaries: the coarse
`Storage.usage()` (total bytes + counts) and `GET /project` — M63
adds the category partition and the retention aggregation without
touching either.

## §2 — Implementation

Five files changed + two new (+~1,060 lines incl. tests):

- **`app/schemas.py`** — `ProjectStorageCategory` (name/files/bytes),
  `ProjectModelStorageSummary` (identity + model_bytes +
  records_bytes + the six M62 aggregates + total_model_bytes),
  `ProjectStorageOverview` (totals + retention aggregates +
  categories + models).
- **`app/engine.py`** — `PROJECT_STORAGE_CATEGORIES` (the canonical
  fixed category order, a class constant mirroring
  `CHECKPOINT_BLOCKER_ORDER`) and `project_storage_overview()`:
  ONE physical walk classifies every file (tmp/ and hidden
  crash-residue entries outside the boundary) into EXACTLY ONE
  category via the imported family constants; per-model rows reuse
  `list_models()` order and call the ONE
  `checkpoint_retention_overview(model_id)` per model (never a
  second retention analysis); project aggregates are deterministic
  sums over the rows.
- **`app/api.py`** — ONE route `GET /project/storage` (tags `meta`,
  beside `/project`), landing-page feature bullet + route-list line.
- **`README.md`** — Milestone 63 section; counts 617 → 623, 20 → 21
  suites.
- **`tests/test_project_storage.py`** (new) + OpenAPI count bumps
  86 → 87 across the 9 existing files (47 assertions).
- Evidence: **`smoke_m63_live.py`**, **`m63_pre.sha256`**.

## §3 — Storage accounting

**Boundary:** every physical file under the storage root, excluding
`tmp/` scratch and hidden entries (`.tmp-delete-*` crash residue).
**Categories (canonical order):** models / checkpoints / model_records
/ datasets / tokenizers / suite_runs / samples / sample_evaluations /
policies / probe_suites / workflow_recipes / project / unclassified —
every file in EXACTLY ONE category (first-match on the imported
family layout constants), so
`sum(category files/bytes) == total_files/total_bytes` — the
no-double-counting invariant, proven against an independent oracle
that re-walks and re-classifies the filesystem. Unknown layouts land
in the explicit `unclassified` category (never silently discarded).
**Per-model attribution:** a model's own state bytes
(`models/<id>/*` files), its evidence-record bytes (its
evaluations/comparisons/gates/workflows) and its checkpoint bytes are
disjoint by construction; shared project storage (datasets,
tokenizers, recipes, project.json) is counted once at the project
level. `§13` ownership identity tested:
`total_bytes == row_A.total_model_bytes + row_B.total_model_bytes +
shared categories`.

## §4 — Retention aggregation

Each model row carries that model's M62
`checkpoint_retention_overview` aggregates **verbatim** — the ONE M61
blocker analysis, the ONE M3 verifier and the ONE
`checkpoint_artifact_stats` measurement, never recomputed.
Project-level `checkpoint_count`, `deletable/protected_checkpoints`,
`total/reclaimable/protected_checkpoint_bytes` are deterministic sums
over the rows. `reclaimable_checkpoint_bytes` counts ONLY
currently-deletable checkpoint artifact sets (6,305,615 of 8,197,262
checkpoint bytes; never model weights, tokenizer, dataset or record
storage); **M63 reports storage; M61 performs deletion.** The chain:
`M61 blocker analysis → M62 model retention overview → M63 project
aggregation` — one source of truth end to end, tested by querying the
M62 route per model and comparing every row.

## §5 — Tests

**Targeted (6 new, `tests/test_project_storage.py`):**
`test_m63_reconciles_with_oracle_and_m62` (independent physical-oracle
reconciliation of totals AND every category, partition exactness,
per-model M62 parity, per-model physical attribution, project
aggregates as row sums, healthy-storage category == registry,
determinism, zero-writes sha inventory);
`test_m63_multi_model_aggregation_and_shared_storage` (two models
with different architectures/retention states — evaluation vs sample
references; project totals == A + B M62 views; the §13 ownership
identity; shared dataset/tokenizer counted exactly once; M52 best +
M60 published + sampled checkpoint protected);
`test_m63_empty_project` (zeroed totals, empty collection, only
`project` category, nothing manufactured);
`test_m63_unclassified_and_boundary` (stray root file + unknown
nested dir → explicit `unclassified`; tmp/ + hidden files outside the
boundary; unregistered model-looking dir counted but no row
manufactured);
`test_m63_corruption_accounting` (same-length weights corruption →
integrity-failed checkpoint never reclaimable, reclaimable drops by
exactly its size, physical totals unchanged; removed manifest →
listing-invisible checkpoint, registry bytes drop by the full artifact
set while the physical category keeps the orphan weights — the honest
divergence; unparseable model manifest → row disappears, bytes stay
counted; full restore → byte-identical recompute);
`test_m63_api_project_storage` (HTTP shape, live-session-root oracle,
rows == `GET /models` listing, per-row M62 parity over HTTP, M52/M60
consistency, determinism, zero writes, OpenAPI 87 + schemas +
ordering).

**Full suite: 617 → 623, ×2 consecutive runs on the final state**
(exit 0 both). **Statics:** pyflakes clean, compileall clean.
**OpenAPI:** **86 → 87** (exactly the one new GET; 47 existing count
assertions bumped; `ProjectStorageOverview` imported as the response
model, the nested row/category schemas exposed automatically;
standalone-verified).

## §6 — Live smoke

Executed against production (uvicorn, port 8785, all-GET — the
server log confirms zero mutating calls). **Full disclosure:** the
first execution **crashed at check V6** — a pure script bug
(attribute access on JSON dicts), not a fact failure; the 12 checks
that had run all PASSED, and a read-only inventory comparison proved
the crashed run mutated nothing. The bug was fixed and the complete
certification ran: **17/17 PASSED** — prelude (project/listing/M62 +
M52 + M60 coherence per model; disk == `m63_pre.sha256` 52/52),
totals vs the independent physical oracle (52 / 8,926,421), canonical
category order + exact partition + independent classifier parity,
rows == registry listing and == each model's own M62 route (verbatim
aggregates + physical attribution), project aggregates == row sums,
certified retention facts (13 / 10 / 3 / 8,197,262 / 6,305,615;
reclaimable < checkpoint bytes < total bytes), healthy-storage
checkpoints category == registry total (26 files), M52 best + M60
published protected, deterministic byte-identical repeat, storage
byte-identical before/after, tmp clean.

## §7 — Storage

**Before: 52 files / 8,926,421 B. After: 52 files / 8,926,421 B —
byte-identical (sha256-verified 52/52 against `m63_pre.sha256`), 0
tmp, 0 residue, 0 servers left running.** M63 added zero persistent
storage and performed zero mutations: the overview is a computed
view. (The production root itself is the documented second rebuild —
see the disclosure above; its pre-inventory is committed as evidence.)

## §8 — Git

- `M63: read-only project storage overview` — implementation, tests,
  README/landing docs, the smoke, this report (18 files).
- `M63: pre-smoke production inventory (m63_pre.sha256)` — the
  rebuilt root's pre-smoke inventory (52 files).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch tip
  == HEAD; worktree clean (only the untracked `egg-info`).

## §9 — Next milestone

The observability ladder is now complete — checkpoint (M61/M62) →
model (M62) → project (M63) — and it exposed the next blind spot one
family over: **datasets and tokenizers have deletion
(`delete_dataset` / `delete_tokenizer`, thin delegations with NO
reference-safety analysis) but no visibility into what still
references them.** Deleting a dataset that a training run's persisted
provenance, an evaluation, a comparison or a tokenized-version
manifest points at would silently orphan immutable evidence — exactly
the pre-M61 situation, one level up. The architecturally-consistent
next milestone is **M64 — read-only dataset & tokenizer usage
overview**: one live-computed view per dataset/tokenizer listing every
authoritative reference (training runs' persisted provenance, M4
evaluations, M5 comparisons, M15 samples, M16 sample-quality
measurements, tokenized dataset versions) — reusing the ONE existing
`..._for_dataset` / `..._for_tokenizer` filters, never a second
scanner. Pure visibility before any future explicit retention
decision; no deletion semantics change, no policies, no automation.

---

### M64 — READ-ONLY DATASET & TOKENIZER USAGE OVERVIEW (ready-to-paste prompt)

```
Implement **Milestone 64 — Read-Only Dataset & Tokenizer Usage
Overview** for AI Model Forge.

M63 completed the storage observability ladder (checkpoint -> model ->
project) and exposed the next blind spot: datasets and tokenizers can
be DELETED (`DELETE /datasets/{id}`, `DELETE /tokenizers/{id}` — thin
delegations with no reference-safety analysis) but nothing shows what
still REFERENCES them. This milestone is visibility only, the M62
pattern applied one family over: before any future explicit retention
decision, the operator must be able to ask "what would I orphan if I
deleted this dataset/tokenizer?"

### 1. INSPECT FIRST — DO NOT IMPLEMENT YET

Before changing anything, inspect and identify the authoritative
implementations for:

* dataset records and the dataset listing/delete routes (versions,
  tokenized-version manifests, verify semantics);
* tokenizer records and the tokenizer listing/delete routes;
* EVERY persisted record family that can reference a dataset_id or
  tokenizer_id: training runs' persisted provenance in model
  manifests, M4 evaluations, M5 comparisons, M15 samples, M16
  sample-quality measurements, M10 suite runs (via their embedded
  records), workflow/recipe records (via stage artifacts) — and the
  existing `..._for_dataset` / `..._for_tokenizer` listing filters
  (the ONE cross-reference sources this milestone must REUSE);
* the M61/M62/M63 retention-view conventions (blocker categories,
  ordering, computed-live, zero storage);
* the OpenAPI count conventions and the README/landing conventions.

Run the current full test suite before implementation and record the
baseline (expect 623 passed, OpenAPI 87 paths).

### 2. M64 OBJECTIVE

ONE read-only, live-computed route pair:

GET /datasets/{dataset_id}/usage
GET /tokenizers/{tokenizer_id}/usage

reporting, WITHOUT any mutation, per artifact:

* the artifact's identity (id, name, created_at, plus
  version count for datasets / vocab size for tokenizers);
* an ordered, categorized reference list (mirroring the M61 blocker
  style): training-run provenance references (model id + run id),
  evaluation references, comparison references, sample references,
  sample-quality references, and — datasets only — tokenized-version
  references (which tokenizer versions were derived from this
  dataset); tokenizers additionally get the tokenized-version
  back-references (which datasets they have tokenized);
* aggregates: reference counts per category, total references, and a
  boolean `referenced` (True iff any category is non-empty).

### 3. DESIGN CONSTRAINTS

* REUSE: consume the ONE existing `..._for_dataset` /
  `..._for_tokenizer` filters and the ONE persisted record listings —
  never a second scanner, never a stored index. Where a reference
  source has no existing filter (e.g. training-run provenance in
  model manifests), traverse the authoritative listings read-only the
  same way the M61 blocker analysis does.
* Read-only, zero storage, zero mutation, deterministic
  (byte-identical over unchanged state); no cache, no pointer.
* Unknown dataset/tokenizer -> the family's 404; an unreferenced
  artifact -> 200 with an empty reference list and `referenced:
  false` (the collection convention).
* Do NOT change `delete_dataset` / `delete_tokenizer` semantics (M64
  is visibility only — a future milestone may add the guard).
* Do NOT add: deletion guards, policies, automation, bulk operations,
  or any write path.

### 4. TEST MATRIX

Focused tests following existing style, at minimum:

* multi-model fixture where one dataset/tokenizer is referenced by
  training runs, evaluations, comparisons, samples and
  sample-quality records, and another is unreferenced;
* every reference category appears with correct ids/counts;
* parity: each category count equals the corresponding existing
  `..._for_dataset` / `..._for_tokenizer` route's listing length;
* deterministic ordering and byte-identical repeat calls;
* zero writes (byte-level inventory comparison);
* unknown ids -> 404; unreferenced -> 200 zeroed;
* a reference added (new evaluation) then the overview recomputes;
* OpenAPI: exactly two new paths (87 -> 89, reason reported); README
  + landing updates; full regression suite green.

### 5. LIVE CERTIFICATION

After the full suite passes twice and statics are clean, run exactly
one READ-ONLY live smoke against production: capture the inventory,
query usage for every production dataset and tokenizer, verify each
category against the corresponding public listing routes (independent
cross-check), verify the training-provenance references against the
model manifests, repeat for byte equality, and verify the inventory
is byte-identical afterwards. Zero state-mutating calls of any kind.

### 6. REPORT

Report exactly these 9 sections: Baseline & inspection;
Implementation; Usage view; Consistency; Tests; Live smoke; Storage;
Git; Next milestone — then provide the next ready-to-paste
implementation prompt.

### HARD CONSTRAINTS

* Inspect first.
* Read-only only; zero storage; zero mutation.
* Exactly two new routes; no new scanners or registries.
* Reuse the ONE existing cross-reference filters and authoritative
  listings.
* Do not change any deletion semantics.
* No guards, no policies, no automation, no bulk operations.
* Preserve every previous milestone.
* The authoritative invariant is: **every reference M64 reports is a
  real persisted record found through the authoritative listings, and
  every reference the authoritative listings contain appears in
  M64's overview — nothing invented, nothing hidden.**
```
