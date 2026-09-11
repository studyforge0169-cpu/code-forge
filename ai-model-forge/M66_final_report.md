# Milestone 66 — Read-Only Model Usage Overview: Final Report

**Date:** 2026-09-11 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

`GET /models/{model_id}/usage` makes every persisted reference to a
model visible BEFORE any future model-retention guard: ten canonical
categories (six internal model-scoped families + four external
root-level families that persist the model id outside the model
directory), deterministic sorted record ids, and the
internal/external split that separates ownership from what a deletion
would orphan. Live-certified read-only: **17/17 checks, zero
mutations, byte-identical storage**. No model deletion or guard was
implemented or changed.

---

## 1. Inspection & Baseline

**Baseline:** HEAD `6697d04` (M65 spec-compliance delta, pushed),
worktree clean (only the untracked `egg-info`); full suite **632
passed** (exit 0); OpenAPI **91 paths**; production root **52 files /
8,926,407 B / 0 tmp** — byte-identical to the M65 certification state
(`m66_pre.sha256 == m65_retention_pre.sha256`). Environment stable
(fourth consecutive milestone on this sandbox).

**Model topology (inspected before any code):** the model registry is
`Storage.model_ids()` / `list_models()` (record-valid, created_at
order; unparseable manifests skipped — registry-invisible). `DELETE
/models/{id}` already exists as an M1-era unguarded rmtree
(`Storage.remove` → `engine.delete_model` → route) — the M67 subject,
untouched here. **Every persisted model reference, enumerated from the
actual repository:** inside `models/<id>/` — the manifest's own
`RunProvenance` list (run ids), checkpoints (`CheckpointRecord` no
explicit model_id — the directory IS the scope), workflows
(`WorkflowRecord.model_id`), evaluations
(`EvaluationRecord.model_id`), comparisons
(`ComparisonRecord.model_id`), gate decisions
(`GateDecision.model_id`); OUTSIDE the model directory — suite-run
records (`SuiteRunRecord.model_id`, root `suite-runs/<id>/`), samples
(`SampleRecord.model_id`, root `samples/<model_id>/`),
sample-quality measurements (`SampleEvaluationRecord`, root
`sample-evaluations/<model_id>/`), and **workflow recipes** (verified
against the production manifest: train/evaluate/gate stage configs
persist `model_id` — a recipe is inert data but its definition is
bound to the models it names; deleting such a model would leave the
recipe unresolvable). Datasets/tokenizers/policies/probe-suites
persist NO model references (verified by schema inspection). The ONE
existing model-scoped listings for every family already exist on the
facade (`list_checkpoints`, `list_workflows`, `list_evaluations`,
`list_comparisons`, `list_gate_decisions`, `list_suite_runs`,
`list_samples`, `list_sample_evaluations`) — all reused, no second
scanner.

## 2. Canonical Model Reference Analysis

`ModelForge.MODEL_USAGE_CATEGORIES` (canonical, fixed): **training_run
→ checkpoint → workflow → evaluation → comparison → gate** (INTERNAL —
persisted inside `models/<id>/`, ownership) then **suite_run → sample
→ sample_quality → workflow_recipe** (EXTERNAL — root-level, the
future-guard surface). `model_usage_overview(model_id)` scopes through
`_scope_data_artifact` (unknown OR registry-invisible model →
FileNotFoundError → 404), then builds every category through the ONE
authoritative listings; the recipe category uses
`_recipe_model_ids(recipe)` (the M64 `_workflow_data_refs` pattern:
train/evaluate/gate stage configs only — publish/suite-run/recipe
stages never name a model). References are the listings' record ids,
sorted and unique (reference ids, never duplicated records or bytes);
aggregates: per-category counts,
`internal_references`/`external_references`/`total_references`,
`referenced`, `externally_referenced`. Deterministic: canonical
category order + sorted ids + listing-derived (never filesystem
enumeration order). Zero storage, zero mutation, byte-identical over
unchanged state, recomputed live.

## 3. API

ONE route: `GET /models/{model_id}/usage` (tags `models`, beside
`get_model`), thin adapter over the facade (no manifest scanning or
directory walking in the route — the established separation). Schemas:
`ModelUsageOverview` (model_id, name, created_at, architecture,
parameter_count, referenced, externally_referenced,
total_references, internal_references, external_references,
categories) and `ModelUsageCategory` (category, references) — both
exposed in OpenAPI. Known model → 200; unknown → the family's 404;
malformed manifest → registry-invisible 404; a fresh model → 200 with
all-empty categories; GET never mutates. **No DELETE routes added; no
deletion semantics changed.** OpenAPI: **91 → 92** (exactly this one
new GET; 51 existing count assertions bumped; standalone-verified:
the spec's delete operations are exactly the four pre-existing ones).

## 4. Root-Level References

The three root-level families M65 identified are included with their
real persisted `model_id` fields, discovered through their ONE
model-scoped listings: `suite_run` (`suite-runs/<id>/manifest.json`),
`sample` (`samples/<model_id>/sample-<id>/`), `sample_quality`
(`sample-evaluations/<model_id>/`). Inspection found ONE additional
external family beyond M65's list: **workflow recipes** — verified
against the production manifest (`workflow-recipes/m62-live-loop`
persists the model id in its train/evaluate/gate stage configs), a
real persisted dependency (the recipe cannot resolve without the
model), included as the `workflow_recipe` category via a read-only
stage-config scan. The M64 suite-registry precedent was considered
and distinguished: a suite/policy definition does NOT name a model
(states are bound at run time), while a recipe's stage configs DO —
hence recipes are references and suites are not.

## 5. Verification

**Focused tests (4 new, `tests/test_model_usage.py`):**
`test_m66_model_usage_overview` (a full-coverage fixture — all ten
categories non-empty via training, evaluation, comparison, gate,
workflow, suite run, sample, sample-quality and a model-bound recipe;
identity; canonical order; per-category parity with the ONE listings;
an INDEPENDENT manifest-parsing oracle (raw JSON, never the app's
analysis) agreeing on every category; no duplicates; the 30/4→30/4
internal/external split as exact sums; physical-layout spot checks;
determinism; zero-writes sha inventory);
`test_m66_scope_isolation_and_empty` (a second model sees only its
OWN references — no cross-leak; a fresh model → all-empty categories
and zero aggregates; unknown → the family 404; malformed manifest →
registry-invisible 404, restored);
`test_m66_live_recompute` (a new suite run grows external_references
1 AND the probe's M4 evaluation internal_references — live recompute,
no cache, no cross-talk with the rich model's unchanged view);
`test_m66_api_model_usage` (HTTP shape; parity with every public
listing route incl. the independent recipe scan; the oracle over the
session root; 404s; determinism; zero mutation; OpenAPI 92 + the
delete-set unchanged). Test-side honest notes: three of my initial
expectations were corrected against the implementation's correct
behavior (a suite run also executes an M4 evaluation; the recompute
capture ordering; "no DELETE anywhere" → M66 adds zero deletes while
the pre-existing M1 model DELETE stays). **Full suite: 632 → 636, ×2
consecutive runs on the final state** (exit 0 both). **Statics:**
pyflakes + compileall clean. **OpenAPI:** 91 → 92, standalone-verified.
**Live smoke:** executed exactly once — **17/17 PASSED, zero
corrections** (see §7).

## 6. Cross-Milestone Safety

The full regression suite (636 ×2) covers every prior milestone's
behavior — M64 dataset/tokenizer usage (its suite untouched, green),
M65 dataset/tokenizer guards + retention views (green), M61/M62
checkpoint retention (green), M63 project storage (green), and all
family behavior. The M66 engine method is purely additive (one new
method + one class constant + one static helper; no existing code
path modified — the imports are the only edits to shared files). The
live smoke additionally re-verified the production M62 retention
(13/10/3), M63 storage (52 files / 1 model) and M64 usage (16 / 15
references) — all coherent, and the OpenAPI delete-operation set is
provably unchanged (exactly the four pre-existing DELETEs).

## 7. Storage Proof

**Before: 52 files / 8,926,407 B. After: 52 files / 8,926,407 B —
byte-identical (sha256-verified 52/52 against `m66_pre.sha256`, which
is itself identical to the M65 inventory), 0 tmp, 0 manifest
modifications, 0 servers left running.** The live smoke
(`smoke_m66_live.py`, executed exactly once, zero corrections) ran
GET-only (the server log confirms): prelude coherence (project,
registry, M62/M63/M64); the usage view's identity, canonical order,
every category == the independent manifest-parsing oracle AND the
certified counts (6/13/3/4/2/2/0/0/0/1), the 30/1/31 splits as exact
category sums, parity with every public listing route, unknown-model
404, a deterministic byte-identical repeat, OpenAPI 92 with the
unchanged delete set, and the byte-identical before/after inventory.
Production contains exactly one model (referenced — 31 references),
so no second "few-references model" exists in production; that
coverage comes from the deterministic fixtures (fresh model →
all-empty categories).

## 8. Git

- `M66: read-only model usage overview` — implementation
  (schemas/engine/api), README, the new test file, the OpenAPI bumps,
  the smoke, this report.
- `M66: pre-smoke production inventory (m66_pre.sha256)` — the
  production root's pre-smoke inventory (52 files; byte-identical to
  the M65 inventory, evidencing the unchanged state).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch tip
  == HEAD; worktree clean (only the untracked `egg-info`).

## 9. Final Status & Next Milestone

**M66 is complete** against the full quality bar: every persisted
model reference inspected and enumerated (including the recipe
category M65's list did not name), the canonical surface
deterministic (fixed category order, sorted unique ids,
byte-identical repeats), existing listings reused (no second
scanner), root-level references included, the API read-only with
correct unknown/malformed semantics, the independent oracle agreeing
exactly, M64/M65 and all prior behavior unchanged (636 ×2 green),
OpenAPI correct (92, no new deletes), the live smoke first-execution
clean (17/17), production byte-identical, zero new persistent files,
git pushed. **Limitations (documented):** no production model with
zero references exists (covered by fixtures); the overview reports
references, not storage bytes (M63 remains the storage view); no
deletion semantics exist for the guard — deliberately, per scope.

**Next milestone — M67, explicit verified model retention.** The
progression completes exactly as the M66 spec drew it: with every
external model reference now visible, the M1-era unguarded
`DELETE /models/{id}` (a plain rmtree) becomes the last unsafe
deletion in the system — deleting a model today silently orphans
root-level suite runs, samples, sample-quality measurements and
model-bound recipes. M67 applies the proven M61/M65 pattern: scope →
integrity verification (the existing M2 model verifier) → the ONE
M66 usage analysis as the blocker source (ordered blockers on the
EXTERNAL categories; the internal families are ownership and go with
the model) → atomic removal of the model's own directory with a
deterministic files/bytes result. The ready-to-paste prompt follows.

---

### M67 — EXPLICIT VERIFIED MODEL RETENTION (ready-to-paste prompt)

```
Implement **Milestone 67 — Explicit Verified Model Retention** for AI
Model Forge.

M66 made every persisted reference to a model visible. The model
deletion itself is still the M1-era unguarded rmtree
(`DELETE /models/{id}`): deleting a model today silently orphans the
root-level suite-run records, samples, sample-quality measurements
and model-bound workflow recipes that reference it. M67 applies the
proven M61/M65 pattern to the last unprotected family.

### 1. INSPECT FIRST — DO NOT IMPLEMENT YET

Before changing anything, inspect and identify the authoritative
implementations for:

* the M61 checkpoint deletion guard and the M65 dataset/tokenizer
  guards (blocker derivation from the ONE usage analysis, the
  integrity-first ordering, the structured 409 detail, the typed
  result, the measure-then-`atomic_delete_dir` split);
* the M66 model usage analysis (`model_usage_overview`,
  `MODEL_USAGE_CATEGORIES`, the internal/external split,
  `_recipe_model_ids`) — the ONE reference analysis this milestone
  must REUSE;
* the current model deletion (`Storage.remove`,
  `engine.delete_model`, the `DELETE /models/{id}` route — current
  response, 404s);
* the M2 model verifier (`verify_model` / `Storage.verify_integrity`:
  manifest parse, state reload, weights hash sidecar) and the M63
  `_artifact_files` walk (for the deletion result's stats);
* what physically goes with a model (its whole directory: manifest,
  weights, checkpoints, evaluations, comparisons, gates, workflows)
  vs what stays (root-level families — protected BY the guard);
* the OpenAPI count conventions and the README/landing conventions.

Run the current full test suite before implementation and record the
baseline (expect 636 passed, OpenAPI 92 paths).

### 2. M67 OBJECTIVE

Make model deletion explicit, verified and reference-safe:

DELETE /models/{model_id}   (the EXISTING route — extend it)

* refuse with 409 + an ORDERED blocker list while ANY M66-visible
  EXTERNAL reference exists (suite_run, sample, sample_quality,
  workflow_recipe — the categories outside the model directory).
  The INTERNAL categories (training_run, checkpoint, workflow,
  evaluation, comparison, gate) are ownership: they live inside
  models/{id}/ and go with the model — they are NOT blockers;
* verify INTEGRITY first (the existing M2 verifier: manifest parse,
  state reload, weights sha256 sidecar): corrupt -> 409, never
  deletable; unparseable manifest -> registry-invisible 404 (the M61/
  M65 convention);
* on success: remove the model's OWN directory atomically
  (`atomic_delete_dir` — never a second primitive), returning the
  deterministic result {model_id, files_removed, bytes_reclaimed}
  measured before removal;
* unknown model -> the family's 404; nothing deleted.

### 3. DESIGN CONSTRAINTS

* REUSE the ONE M66 analysis for the guard (never a second scanner):
  external categories with references -> blockers, in the M66
  canonical order, with the SAME reference ids;
* optionally expose a read-only retention view
  (`GET /models/{id}/retention`) mirroring the M65 pattern
  (identity, artifact files/bytes, integrity, deletable, ordered
  blockers) — consistent with the existing dataset/tokenizer
  retention views;
* deletion removes ONLY the model's own directory; root-level
  families are protected BY the guard and never touched;
* no cascade, no force flag, no bulk mode, no policies, no keep-N,
  no age rules, no background cleanup;
* preserve every previous milestone's semantics (M61–M66 untouched:
  the M66 usage view must show exactly what M67's guard refuses on).

### 4. TEST MATRIX

Focused tests following existing style, at minimum:

* a model with external references: every external category refuses
  with the ordered list == the M66 usage overview's non-empty
  external categories (both directions — the headline invariant);
* internal-only references (training/checkpoints/evaluations/...) do
  NOT block: a model with rich internal history but zero external
  references IS deletable;
* integrity failure -> 409 before the reference analysis (the M61
  ordering — a corrupt+referenced model refuses on integrity);
* malformed manifest -> registry-invisible 404;
* successful deletion: exact files/bytes vs an independent walk,
  ONLY the model's own directory disappears, root-level families
  byte-identical, registries shrink, atomic (no .tmp-delete-*
  residue), repeated deletion -> 404;
* refusals write nothing (byte-level inventory);
* after the referencing root-level records are deleted through their
  own paths (or a recipe is the only blocker), the model becomes
  deletable (live recompute);
* HTTP: 409 details with the exact overview blockers, 200 results,
  404s, determinism, zero mutation on refusal, OpenAPI (a retention
  route adds 92 -> 93 if exposed; the DELETE rides the existing
  path); README + landing updates; full regression suite green.

### 5. LIVE CERTIFICATION

The production model is referenced (M66: 31 references incl. the
external recipe) — the production protocol is READ-ONLY: capture the
inventory, query M66 usage + the retention view, verify the
protected DELETE refuses with the exact ordered blockers, verify
determinism and zero mutation. The success path (an internal-only
model deleting with exact stats) is verified on a THROWAWAY COPY of
the production root (copy, train a fresh unreferenced model there,
verify its deletion, discard the copy). Production must remain
byte-for-byte identical.

### 6. REPORT

Report exactly these 9 sections: Baseline & inspection;
Implementation; Guard & blocker order; Consistency; Tests; Live
smoke; Storage; Git; Next milestone — then provide the next
ready-to-paste implementation prompt.

### HARD CONSTRAINTS

* Inspect first.
* Reuse the ONE M66 reference analysis; never a second scanner.
* The headline invariant: **M67's guard refuses on EXACTLY the
  M66-visible EXTERNAL references, in the same canonical order —
  nothing protected that is not shown, nothing shown that is not
  protected.**
* Integrity verified before mutation; atomic removal; deterministic
  results; unknown -> 404.
* No cascade, no force, no bulk, no policies, no automation.
* Production stays byte-identical (success-path certification on a
  copy).
* Preserve every previous milestone.
```
