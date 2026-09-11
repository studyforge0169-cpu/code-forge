# Milestone 65 — Explicit Verified Dataset & Tokenizer Retention: Final Report

**Date:** 2026-09-11 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

`DELETE /datasets/{id}` and `DELETE /tokenizers/{id}` become explicit,
verified and reference-safe — the M61 pattern one family over. The
guard refuses on EXACTLY what the M64 usage overview reports (ordered
blockers, same categories, same ids); a successful deletion removes the
artifact's own directory atomically and returns the deterministic
files/bytes result. The tokenizer family's pre-M65 deletion was an
UNGUARDED rmtree — now fully guarded. Live-certified: **18/18 checks,
production byte-identical (deletions verified on a discarded copy)**.

> Environment: stable since the M63 certification (verified before
> work began: venv, production root, git at `4b6d9c4`; the production
> root is byte-identical to the M64 state, `m65_pre.sha256 ==
> m64_pre.sha256` 52/52).

---

## §1 — Baseline & inspection

**Baseline:** full suite **626 passed** (exit 0); HEAD `4b6d9c4` (M64,
pushed); OpenAPI **89 paths**; production root **52 files / 8,926,421 B
/ 0 tmp** (1 model, 1 dataset with 16 references, 1 tokenizer with 15).

**Inspected before any code:** the M61 guard pattern
(`checkpoint_blockers` + `CHECKPOINT_BLOCKER_ORDER` → ValueError → the
API's structured 409 detail `{message, ids, protected, blockers}`; the
`remove_checkpoint` measure-then-`atomic_delete_dir` split; the
rename-to-hidden-sibling primitive); the M64 analyses (the ONE usage
overview per artifact + canonical category orders); the current
`delete_dataset` (single-category guard: `_referencing_tokenizers` →
ValueError → 409 string detail) and `delete_tokenizer` (**no guard at
all** — plain `shutil.rmtree`); the family layouts (a dataset's own
tree = meta + versions + records + tokenized artifacts; a tokenizer's =
manifest + tokenizer.json); and every existing test touching the two
deletions (response shapes `{"deleted": id}`, the "training source"
match, the pipeline test's cleanup order).

## §2 — Implementation

Six files changed + one new test file (+~700 lines incl. tests):

- **`app/schemas.py`** — `ArtifactDeletionBlocker` (reason = an M64
  category, detail = the sorted reference ids), `DatasetDeletionResult`
  / `TokenizerDeletionResult` (id + files_removed + bytes_reclaimed),
  and the typed 409 details `DatasetDeletionBlocked` /
  `TokenizerDeletionBlocked` (message + id + protected + blockers).
- **`app/dataset.py` / `app/tokenizer.py`** — the low-level `delete()`
  primitives rewritten to the `remove_checkpoint` pattern: validate
  existence, measure (files, bytes), ONE `atomic_delete_dir`, return
  the measurement. The CALLER owns the safety decision.
- **`app/engine.py`** — `dataset_deletion_blockers()` /
  `tokenizer_deletion_blockers()`: the guard analysis computed LIVE
  from the ONE M64 overview (never a second scanner) — every non-empty
  category becomes one blocker in the SAME canonical order with the
  SAME ids; and the guarded facade `delete_dataset()` /
  `delete_tokenizer()`: scope (registry 404) → blockers (ValueError
  with the ordered summary) → atomic removal → the deterministic
  result.
- **`app/api.py`** — both DELETE routes rewritten to the M61 shape:
  result response models, 404 mapping, structured 409 with re-derived
  blockers, and declared `responses={409: ...}` so the blocker schemas
  are exposed in OpenAPI; landing-page feature bullet + two route-list
  lines.
- **`README.md`** — Milestone 65 section; counts 626 → 630, 22 → 23
  suites.
- **`tests/test_data_retention.py`** (new) + three honest updates to
  existing tests (below).
- Evidence: **`smoke_m65_live.py`**, **`m65_pre.sha256`**.

## §3 — Guard & blocker order

Dataset blockers (canonical M64 order): training_run → workflow →
evaluation → comparison → suite_run → tokenizer_training →
tokenized_version. Tokenizer blockers: training_run → workflow →
evaluation → comparison → suite_run → sample → sample_quality →
tokenized_dataset. Guard sequence per deletion: (1) scope through the
family registry (unknown → 404, nothing deleted); (2) the LIVE
reference-safety analysis — the ONE M64 overview; ANY visible
reference → 409 with the ordered list (the API re-derives it for the
structured detail, the M61 pattern); (3) ONE atomic removal of the
artifact's OWN directory only. The old single-category dataset
refusal (`tokenizer_training`) is preserved as one blocker category —
same refusal, now structured and ordered. No cascade, no force, no
bulk, no policies, no keep-N, no age rules, no background cleanup.

**Honest consequence (documented in README + tests):** a tokenized
dataset/tokenizer pair is MUTUALLY protected — the tokenizer's derived
artifacts live under the dataset (`tokenized_dataset` blocks the
tokenizer; `tokenizer_training` + `tokenized_version` block the
dataset). Removing derived tokenized artifacts would require a future
EXPLICIT operation; M65 never cascades. The pre-M65 "delete the
tokenizer first" cleanup order is therefore no longer possible once a
dataset has been tokenized — the pre-existing pipeline test was
updated to assert the new mutual protection (both 409s with their
exact blocker categories).

## §4 — Consistency

The headline invariant is structural: the guard calls the ONE M64
analysis and converts its non-empty categories to blockers verbatim —
**nothing protected that is not shown, nothing shown that is not
protected**, in the same canonical order with the same reference ids.
Tested in both directions at engine level (blockers == the overview's
non-empty categories, full-coverage fixtures where EVERY category is
non-empty) and over HTTP (the 409 `detail.blockers` == the usage
route's categories rebuilt as blockers). Cross-milestone: M61
checkpoint deletion, M62 retention, M63 project storage and M64 usage
are untouched (their suites run green verbatim; the production smoke
re-verified M62 13/10/3 and the M64 totals 16/15). The M64 test's
guard-interop assertion still passes unchanged (the refusal still
names the tokenizer_training tokenizers and writes nothing).

## §5 — Tests

**Targeted (4 new, `tests/test_data_retention.py`):**
`test_m65_dataset_guard_is_m64` / `test_m65_tokenizer_guard_is_m64`
(both-directions parity on full-coverage fixtures — all 7 / all 8
categories non-empty; refusals write nothing — byte-level sha
inventories; no `.tmp-delete-*` residue; unknown → the family 404);
`test_m65_unreferenced_deletion_and_live_recompute` (exact
files/bytes results verified against independent walks; ONLY the
artifact's own files disappear from the inventory; registries shrink;
repeated deletion → 404; live recompute — deleting the unreferenced
tokenizer shrinks the dataset's tokenizer_training but the dataset
stays protected by its other references; the full unblock chain:
fresh dataset blocked ONLY by tokenizer_training → delete the
tokenizer → the dataset becomes deletable);
`test_m65_api_retention` (HTTP: 409 details with the exact overview
blockers both directions; refusals write nothing; unknown → 404;
unreferenced artifacts delete with exact stats; the guarded artifacts
survive; OpenAPI: 89 paths UNCHANGED, DELETE rides the existing detail
routes, all five M65 schemas exposed).

**Updated existing tests (3, honest semantic changes):**
`test_data_api.py` — the two deletion response-shape assertions
(`{"deleted": id}` → the typed files/bytes results) and the pipeline
test's cleanup tail (now asserts the mutual protection 409s with
their exact blocker categories); `test_data_engine.py` — the guard
message match ("training source" → "tokenizer_training").

**Full suite: 626 → 630, ×2 consecutive runs on the final state**
(exit 0 both). **Statics:** pyflakes clean, compileall clean.
**OpenAPI:** **89 paths, UNCHANGED** (no new routes — both DELETEs
ride the existing detail paths; the result + typed-409 + blocker
schemas are newly exposed; standalone-verified).

## §6 — Live smoke

Executed **exactly once** (uvicorn, port 8787): **18/18 PASSED on the
first execution — zero corrections.** Protocol per the spec: the
production artifacts are both referenced, so **no deletion was
attempted over HTTP** (the server log shows GETs only). Production
(read-only): project/registries; disk == `m65_pre.sha256` 52/52; the
M64 usage views coherent (16 / 15 references); M62 retention 13/10/3.
Throwaway COPY (engine-level, discarded afterwards): the copy's usage
views equal the production HTTP answers; both guards refuse with
EXACTLY the M64 blocker lists (6 and 5 non-empty categories); the
refusals are read-only (copy inventory stable at 52 files); unknown
ids → FileNotFoundError; live recompute on the copy (train a new
tokenizer → tokenizer_training 1→2; delete it → back to 1, with the
exact 2-file result); an unreferenced dataset deletes with exact
stats (4 files / 1,936 B, no residue); the protected artifacts survive
everything; the copy is discarded. Production byte-identical
before/after.

## §7 — Storage

**Before: 52 files / 8,926,421 B. After: 52 files / 8,926,421 B —
byte-identical (sha256-verified 52/52; identical to the M64 state), 0
tmp, 0 copy residue, 0 servers left running.** M65 added zero
persistent storage; production was never mutated.

## §8 — Git

- `M65: explicit verified dataset & tokenizer retention` —
  implementation, tests (new + updated), README/landing docs, the
  smoke, this report.
- `M65: pre-smoke production inventory (m65_pre.sha256)` — the
  production root's pre-smoke inventory (52 files; byte-identical to
  the M64 inventory, evidencing the unchanged state).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch tip
  == HEAD; worktree clean (only the untracked `egg-info`).

## §9 — Next milestone

The retention matrix is now complete for checkpoints (M61/M62),
datasets and tokenizers (M64/M65) — but the MODEL itself is the
remaining unguarded deletion: `DELETE /models/{id}` is still a plain
`rmtree` of the model directory (M1-era `Storage.remove`). Most of a
model's evidence lives INSIDE its directory and dies coherently with
it — but three ROOT-LEVEL families persist model references outside
it: M10 suite-run records (`suite-runs/<id>` carrying `model_id`),
M15 samples (`samples/<model_id>/…`) and M16 sample-quality
measurements (`sample-evaluations/<model_id>/…`). Deleting a model
today silently orphans those records — exactly the pre-M61 situation,
now the LAST family with the gap. The architecturally-consistent next
milestone is **M66 — read-only model usage overview** (the M64
pattern): one live-computed view per model listing every external
reference (suite runs, samples, sample-quality measurements — via the
ONE existing listings, never a second scanner) plus the model's
internal-family ownership summary (what lives inside its directory
and therefore goes with it). Pure visibility; the guard (M67) follows
once the reference surface is shown.

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
baseline (expect 630 passed, OpenAPI 89 paths).

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
* OpenAPI: exactly one new path (89 -> 90, reason reported); README
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
