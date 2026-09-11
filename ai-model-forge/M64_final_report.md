# Milestone 64 — Read-Only Dataset & Tokenizer Usage Overview: Final Report

**Date:** 2026-09-11 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

`GET /datasets/{dataset_id}/usage` and `GET /tokenizers/{tokenizer_id}/usage`
answer "what still references this dataset/tokenizer?" BEFORE any deletion
decision — every persisted referencing record by category, computed live
through the ONE existing cross-reference filters. Live-certified read-only
against production: **25/25 checks, zero mutations, byte-identical
storage**. No deletion semantics were added or changed.

> Environment: stable since the M63 certification — no re-provision this
> milestone (verified: venv, production root and git all intact at
> `542dfd3` before work began; the production root is byte-identical to
> the M63 state, `m64_pre.sha256 == m63_pre.sha256` 52/52).

---

## §1 — Baseline & inspection

**Baseline:** full suite **623 passed** (exit 0); HEAD `542dfd3` (M63,
pushed); OpenAPI **87 paths**; production root **52 files / 8,926,421 B /
0 tmp** (1 model, 13 checkpoints, 1 dataset, 1 tokenizer).

**Inspected reference topology (before any code):** datasets are
referenced by model manifests' `RunProvenance` (dataset_id per run),
workflow records' embedded plans (train/evaluate/compare stage configs),
M4 evaluation records, M5 comparison records, M10 suite-run probe
results, tokenizers' `trained_on_dataset_id` (the ONE category the
existing `delete_dataset` guard refuses on — it raises ValueError → 409
listing exactly those tokenizers) and the M2 tokenized artifacts under
`datasets/<id>/v<N>/tokenized/<tok>/`. Tokenizers are referenced by the
same record families plus M15 samples and M16 sample-quality
measurements, plus the reverse tokenized layout. **`delete_tokenizer`
has NO guard at all** (plain rmtree) — the exact pre-M61 situation, one
family over. Existing ONE cross-reference sources: the
`..._for_dataset` / `..._for_tokenizer` filters (evaluations M28/M30,
comparisons M29/M31, samples M32, sample-quality M33), all
persisted-identity-only membership, all model-scoped (project scope =
iterate the registry). Suite-run stages reference a suite id (the
shared M9 registry), never a dataset/tokenizer directly — that
indirection is deliberately not followed (definitions are not per-model
evidence; documented).

## §2 — Implementation

Five files changed + one new test file (+~700 lines incl. tests):

- **`app/schemas.py`** — `ArtifactUsageCategory` (category + sorted
  reference ids), `DatasetUsageOverview` (identity + version_count +
  latest_version + referenced + total_references + categories),
  `TokenizerUsageOverview` (identity incl. `trained_on_dataset_id` as
  provenance + the same aggregates).
- **`app/engine.py`** — `DATASET_USAGE_CATEGORIES` (7) and
  `TOKENIZER_USAGE_CATEGORIES` (8), canonical fixed orders (provenance
  first, evidence families in the M61 blocker order, artifact-specific
  last); `_workflow_data_refs(plan)` (the read-only stage-config walk:
  train/evaluate/compare configs only); `_tokenized_tokenizers()`
  (the ONE M2 layout walked read-only); `dataset_usage_overview()` and
  `tokenizer_usage_overview()` — both validate through the registry
  getters (unknown → FileNotFoundError → 404), iterate `list_models()`
  (registry convention: corrupt manifests skipped), call the ONE
  filters per model, reuse `_referencing_tokenizers` (the guard's own
  scan) and count per-record references (a workflow/suite run/training
  run counts ONCE however many of its stages/probes name the
  artifact).
- **`app/api.py`** — TWO routes beside their family's detail routes,
  with the family's 404 mapping; landing-page feature bullet + two
  route-list lines.
- **`README.md`** — Milestone 64 section; counts 623 → 626, 21 → 22
  suites.
- **`tests/test_data_usage.py`** (new) + OpenAPI count bumps 87 → 89
  across the 10 existing files (48 assertions).
- Evidence: **`smoke_m64_live.py`**, **`m64_pre.sha256`**.

## §3 — Usage view

Per artifact: identity (id/name/created_at; datasets: version_count +
latest_version; tokenizers: requested/actual vocab +
trained_on_dataset_id), then the canonical category list — datasets:
`training_run` (model/run ids), `workflow`, `evaluation`,
`comparison`, `suite_run` (model/record ids), `tokenizer_training`
(the tokenizer ids the deletion guard refuses on), `tokenized_version`
(`v<N>/<tokenizer_id>`); tokenizers: the same shared five plus
`sample`, `sample_quality` and `tokenized_dataset`
(`<dataset_id>/v<N>`). Every category always appears (empty lists
allowed — self-describing and deterministic); references are sorted
id strings; `referenced` = any category non-empty;
`total_references` = the exact sum. Unreferenced artifact → 200 with
empty categories; unknown → the family's 404.

## §4 — Consistency

Every category is computed through the ONE existing analysis for that
family — never a second scanner: evaluations/comparisons/samples/
sample-quality through the `..._for_dataset`/`..._for_tokenizer`
filters (the same ones the public routes expose), training runs
through the authoritative model manifests, workflows through a
read-only plan walk (the M61 blocker-analysis style), tokenizer
training through `_referencing_tokenizers` — the scan the
`delete_dataset` guard itself uses, so **the usage view explains the
existing guard exactly** (tested: the deletion refusal names exactly
the `tokenizer_training` references, and writes nothing). The
production smoke verified the shared categories agree between the two
views and that both M62 (retention 13/10/3) and M63 (52-file project
storage) remain coherent — cross-milestone consistency.

## §5 — Tests

**Targeted (3 new, `tests/test_data_usage.py`):**
`test_m64_dataset_usage_overview` (a multi-reference fixture — direct
run/eval/comparison, a TRAIN+EVALUATE workflow, a suite run, a sample
+ sample-quality measurement, two tokenizers trained on the dataset;
per-category parity against INDEPENDENT recomputation — manifests,
plan scans, filter calls, the guard scan, filesystem walks; totals;
determinism; zero-writes sha inventory; **the guard invariant**
(delete refused with exactly the tokenizer_training ids, nothing
written); unknown → FileNotFoundError; unreferenced dataset → empty
zeroed view);
`test_m64_tokenizer_usage_overview` (identity incl.
trained_on_dataset_id; all eight categories' parity; the reverse
tokenized walk; live recompute — a new evaluation grows the count in
BOTH views; unknown → FileNotFoundError; unreferenced tokenizer);
`test_m64_api_usage_overviews` (HTTP shapes; parity with the public
`by-dataset`/`by-tokenizer` routes; 404s; an unreferenced dataset over
HTTP; deterministic byte-equal repeats; zero mutation over the session
root; OpenAPI 89 + both paths GET-only + schemas).

Honest note on test development: two of my initial expectations were
wrong and were corrected against the implementation's actual (correct)
behavior — a comparison evaluates BOTH sides (so the fixture produces
5 evaluation references, not 3) and workflow record ids are bare
hashes (no directory prefix). One API-test ordering bug (the
zero-writes inventory captured before a fixture upload) was
restructured. The engine needed no changes.

**Full suite: 623 → 626, ×2 consecutive runs on the final state**
(exit 0 both). **Statics:** pyflakes clean, compileall clean.
**OpenAPI:** **87 → 89** (exactly the two new GETs; 48 existing
assertions bumped; standalone-verified).

## §6 — Live smoke

Executed **exactly once** (uvicorn, port 8786, all-GET — the server
log confirms zero mutating calls): **25/25 PASSED on the first
execution — zero corrections.** Certified against an INDEPENDENT
oracle that parses the raw manifests/filesystem directly (no app
code): prelude (project/dataset/tokenizer registries; disk ==
`m64_pre.sha256` 52/52; M62 retention 13/10/3 and M63 project storage
coherent), dataset usage (identity; canonical order; every category ==
oracle + certified counts — training_run 6, workflow 2, evaluation 4,
comparison 2, suite_run 0, tokenizer_training 1, tokenized_version 1;
evaluation/comparison == the public cross-reference routes;
tokenizer_training == the guard's list; total 16), tokenizer usage
(identity incl. trained_on; canonical order; every category == oracle
— 6/2/4/2/0/0/0/1; public-route parity; total 15), the shared
categories agreeing across both views, unknown → 404s, deterministic
byte-identical repeats, storage byte-identical before/after, tmp
clean.

## §7 — Storage

**Before: 52 files / 8,926,421 B. After: 52 files / 8,926,421 B —
byte-identical (sha256-verified 52/52; identical to the M63 state), 0
tmp, 0 residue, 0 servers left running.** M64 added zero persistent
storage and performed zero mutations.

## §8 — Git

- `M64: read-only dataset & tokenizer usage overview` —
  implementation, tests, README/landing docs, the smoke, this report.
- `M64: pre-smoke production inventory (m64_pre.sha256)` — the
  production root's pre-smoke inventory (52 files; byte-identical to
  the M63 inventory, evidencing the unchanged state).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch tip
  == HEAD; worktree clean (only the untracked `egg-info`).

## §9 — Next milestone

The visibility arc is complete and symmetric: checkpoints (M61/M62),
project storage (M63), datasets/tokenizers (M64). The remaining
asymmetry is on the WRITE side: `delete_dataset` guards only ONE
category (tokenizers trained on it) while `delete_tokenizer` has NO
guard at all — deleting a tokenizer that tokenized dataset versions,
that evaluations/comparisons/samples measured with, or that training
runs used silently orphans immutable evidence, exactly the situation
M61 eliminated for checkpoints. Now that M64 makes every reference
visible, the architecturally-consistent next milestone is **M65 —
explicit verified dataset & tokenizer retention**: extend the
dataset deletion guard to refuse on ANY M64-visible reference and add
the missing tokenizer guard, both with the M61 pattern (fixed blocker
order, live analysis reusing the ONE M64 machinery, atomic removal,
409 with the ordered reference list, no bulk, no policies, no
cascade).

---

### M65 — EXPLICIT VERIFIED DATASET & TOKENIZER RETENTION (ready-to-paste prompt)

```
Implement **Milestone 65 — Explicit Verified Dataset & Tokenizer
Retention** for AI Model Forge.

M64 made every dataset/tokenizer reference visible.
`delete_dataset` still guards only ONE category (tokenizers trained
on it) and `delete_tokenizer` has NO guard at all — deleting a
referenced data artifact silently orphans immutable evidence. M65
applies the M61 pattern to the data families: deletion becomes
explicit, verified and reference-safe.

### 1. INSPECT FIRST — DO NOT IMPLEMENT YET

Before changing anything, inspect and identify the authoritative
implementations for:

* the M61 checkpoint deletion guard (checkpoint_blockers,
  CHECKPOINT_BLOCKER_ORDER, the 409 shape, the atomic removal
  pattern in remove_checkpoint, its exact API error mapping);
* the M64 usage analyses (dataset_usage_overview /
  tokenizer_usage_overview, the category constants, the reference id
  formats) — the ONE reference analysis this milestone must REUSE;
* the current delete_dataset guard (_referencing_tokenizers +
  ValueError -> 409) and delete_tokenizer (unguarded rmtree);
* the dataset/tokenizer family layouts (what a deletion physically
  removes: versions, records, tokenized artifacts, manifests) and
  what must NEVER be touched (models/, evaluations/, ...);
* the M61 final report's atomicity + test conventions;
* the OpenAPI count conventions and the README/landing conventions.

Run the current full test suite before implementation and record the
baseline (expect 626 passed, OpenAPI 89 paths).

### 2. M65 OBJECTIVE

Make BOTH data-artifact deletions explicit, verified and
reference-safe, exactly the M61 semantics one family over:

DELETE /datasets/{dataset_id}
DELETE /tokenizers/{tokenizer_id}

* refuse with 409 + an ORDERED blocker list while ANY M64-visible
  reference exists (datasets: training_run, workflow, evaluation,
  comparison, suite_run, tokenizer_training, tokenized_version;
  tokenizers: training_run, workflow, evaluation, comparison,
  suite_run, sample, sample_quality, tokenized_dataset) — the blocker
  categories and details in the SAME canonical order and id formats
  the M64 overview reports (the user never sees "referenced" in M64
  and then a successful deletion in M65, or vice versa);
* on success: remove the artifact's OWN directory atomically (the M61
  rename-to-hidden-then-delete pattern; a crash can never leave a
  valid-looking partial artifact), return a deterministic result
  (what was removed: files + bytes, measured before removal);
* unknown id -> the family's 404; nothing deleted.

### 3. DESIGN CONSTRAINTS

* REUSE the ONE M64 analysis for the guard (never a second scanner,
  never a second reference enumeration — if the smallest extraction
  helps, extract, do not duplicate);
* the guard analysis is LIVE (no stored index, no cache) and
  deterministic;
* deletion removes ONLY the artifact's own directory — never model
  state, checkpoints, or other families' records (they are protected
  BY the guard);
* the existing tokenizer_training refusal becomes one blocker
  category in the ordered list (the 409 detail stays informative);
* no cascade deletion, no force flag, no bulk mode, no policies, no
  keep-N, no age rules, no background cleanup;
* preserve every previous milestone's semantics (M61-M64 untouched:
  the M64 usage view must show exactly what M65's guard refuses on).

### 4. TEST MATRIX

Focused tests following existing style, at minimum:

* referenced dataset: every blocker category refuses with the
  ordered list == the M64 overview's categories (both directions:
  every M65 blocker appears in M64, every M64 reference produces a
  refusal);
* referenced tokenizer: the same, for all eight categories;
* unreferenced dataset/tokenizer: deletion succeeds, removes exactly
  the artifact's own files (file/byte counts verified), atomically
  (no residue, no tmp), and the registries/listings shrink;
* after a referencing record family becomes empty (e.g. delete the
  referencing evidence via its own M61 route), the artifact becomes
  deletable (live recompute);
* the tokenizer_training category still refuses exactly as before
  (backwards-compatible 409 semantics, now ordered);
* unknown ids -> 404, nothing deleted;
* zero writes on refusal (byte-level inventory);
* OpenAPI: NO new paths (both DELETEs ride the existing routes);
  response/blocker schemas exposed; README + landing updates; full
  regression suite green.

### 5. LIVE CERTIFICATION

Because the production artifacts are all referenced, the live smoke
must be READ-ONLY: capture the inventory, query M64 usage for every
production dataset/tokenizer, attempt NO deletion over HTTP — verify
instead through the engine in a THROWAWAY COPY of the production
root (copy the root, delete there, verify the guard + a successful
deletion of an unreferenced artifact, discard the copy). The
production root itself must remain byte-for-byte identical.

### 6. REPORT

Report exactly these 9 sections: Baseline & inspection;
Implementation; Guard & blocker order; Consistency; Tests; Live
smoke; Storage; Git; Next milestone — then provide the next
ready-to-paste implementation prompt.

### HARD CONSTRAINTS

* Inspect first.
* Reuse the ONE M64 reference analysis; never a second scanner.
* The headline invariant: **M65's guard refuses on EXACTLY what
  M64's usage overview reports, in the same canonical order —
  nothing protected that is not shown, nothing shown that is not
  protected.**
* Atomic removal; deterministic results; unknown -> 404.
* No cascade, no force, no bulk, no policies, no automation.
* Production stays byte-identical (certification on a copy).
* Preserve every previous milestone.
```
