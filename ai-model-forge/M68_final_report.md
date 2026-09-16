# Milestone 68 — Explicit Suite-Run & Sample Lifecycle: Final Report

**Date:** 2026-09-16 · **Branch:** `arena/01a071e9-code-forge` · **Status:** COMPLETE

The root-level runtime records — suite runs, samples, sample-quality
measurements — get their own explicit, verified, atomic deletions (the
M61/M65/M67 pattern), which for the first time creates a legitimate
path to **unblock a referenced model**: delete the referencing records
and the M67 guard's live analysis lets the model go. **Live-certified
24/24; production byte-identical; all destructive certification on a
discarded copy.** Zero new OpenAPI paths (the three DELETEs are new
operations on existing resource paths); the M67 guard code is
untouched — only its live input set can shrink.

---

## 1. Inspection & Baseline

**Environment event (disclosed):** the sandbox re-provisioned a FOURTH
time mid-session — git sat at the initial commit `f86b670` with all
milestone work untracked (the platform reset `.git` to the session's
starting state while the working tree kept the latest snapshot), and
`/home/user/venv` plus the production root were gone. Recovery (the
proven protocol, ~4 min): `git fetch origin arena/01a071e9-code-forge
&& git reset --mixed FETCH_HEAD` (HEAD `26eda92` == remote tip,
worktree clean); venv rebuilt (torch 2.14.0+cu130, fastapi 0.141.1,
pydantic 2.13.5); production root rebuilt via
`m62_production_rebuild.py` — SAME certified structure but NEW
content-derived ids (timestamps in the rebuild inputs): dataset
`e7ee867f06df` (16 refs: 6/2/4/2/0/1/1), tokenizer `dbb34eaa8804`
(15 refs: 6/2/4/2/0/0/0/1), model `3770ca1bfa23` (M62 13/10/3);
`m68_pre.sha256` captured (52 files / 8,926,410 B / 0 tmp). **Baseline
verified after recovery:** 641 passed (exit 0), OpenAPI 93 paths,
production facts re-derived through the facade (M66 usage
6/13/3/4/2/2/0/0/0/1/0 = 30 internal + 1 external; M67 retention
integrity True / deletable False / blockers exactly
`[workflow_recipe: m62-live-loop]` / 40 files / 8,874,613 B).

**Inspected before any code:** the three record families' engines and
layouts (all **manifest-only** single-file records: `suite-runs/<id>/`,
`samples/<model_id>/sample-<id>/`,
`sample-evaluations/<model_id>/evaluation-<id>/`); their loaders
(`get_suite_run` / `get_sample` / `get_sample_evaluation` —
FileNotFoundError for unknown model/record, pydantic validation on
load); their **persisted `result_hash`** fields — each family has a
`@staticmethod result_hash(record)` computing a sha256 over the
record's semantic payload (excluding ids/timestamps/durations), which
gives these weights-less families a REAL integrity tier (the M65
tokenizer content-hash pattern); the reference topology — **nothing
persists a `suite_run_id`** (leaf record), **nothing persists a
sample-quality `evaluation_id`** (leaf record), while
`SampleEvaluationRecord` persists `sample_id` OUTSIDE the sample's
directory (a real reference — the candidate blocker); the M65
`DatasetEngine.delete` low-level pattern, `_scope_data_artifact`
(catches ValueError subclasses — pydantic ValidationError and
json.JSONDecodeError both map to registry-invisible 404),
`atomic_delete_dir`, and the ONE M19
`list_sample_evaluations_for_sample` filter; the API route conventions
(each family's GET-one route is model-scoped:
`/models/{model_id}/suite-runs/{suite_run_id}` etc.); and the M15/M16
tests' "no edit/delete endpoints (405)" contract assertions.

## 2. Suite-Run & Sample Retention Architecture

**The inspection-driven decision (the one my M68 prompt left open):**
sample-quality measurements **block** sample deletion — a measurement
persists `sample_id` outside the sample's own directory, exactly the
reference-safety rule every other family follows (the same reasoning
that made samples block model deletion in M66/M67). Consequently the
measurement needs its own deletion route for the unblock chain to
complete — **three new deletions, not two**. And following the repo's
actual route conventions (model-scoped, mirroring each family's
GET-one route — the prompt's root-level `/suite-runs/{id}` path was
adjusted to the repository's real convention, avoiding a second
resolution mechanism), all three DELETEs landed on **existing resource
paths**: `DELETE /models/{id}/suite-runs/{run}`,
`DELETE /models/{id}/samples/{sample}`,
`DELETE /models/{id}/sample-quality/{eval}`. **Schemas:**
`SuiteRunDeletionResult`, `SampleDeletionResult`,
`SampleEvaluationDeletionResult` (typed `{model_id, <record id>,
[sample_id,] files_removed, bytes_reclaimed}`) and
`SampleDeletionBlocked` (the structured 409); `ArtifactDeletionBlocker`
reused for the sample blockers (docstring extended to cover samples).
**Engine:** one low-level `delete()` per family engine
(`SuiteRunEngine` / `SamplingEngine` / `SampleQualityEngine` — measure
+ `atomic_delete_dir`, the `DatasetEngine.delete` pattern, caller owns
the safety decision). **Facade:** `delete_suite_run`,
`sample_deletion_blockers` + `delete_sample`, `delete_sample_evaluation`
— full guards, no second scanner (the blocker list comes from the ONE
M19 listing; the model-side effect needs no new analysis at all: the
M67 guard reads the ONE M66 usage view, which reads the ONE family
listings — deletions simply shrink those listings live).

## 3. Deletion Safety

The guard, per family, in the M61/M65/M67 order: **(1) scope** — the
family getter through the `_scope_data_artifact` convention: unknown
model, unknown record, or registry-invisible (unparseable manifest)
→ FileNotFoundError → 404, nothing deleted. **(2) INTEGRITY FIRST** —
the record's persisted `result_hash` must reproduce from its semantic
payload (each family's own `result_hash` staticmethod); a tampered or
corrupt record → RuntimeError → 409, never deletable, no force flag,
no filesystem fallback. **(3) blockers** — samples only: ANY
sample-quality measurement referencing the sample (the ONE M19
listing) → ValueError → **409 with the typed ordered blocker list**
(`{message, model_id, sample_id, protected: true, blockers:
[{reason: "sample_quality", detail: measurement ids}]}` — the SAME ids
the by-sample listing reports); suite runs and measurements are LEAF
records with no reference guard. **(4) atomic removal** —
`atomic_delete_dir` on the record's own directory only, with the
deterministic files/bytes result measured immediately before.
**No cascade, ever:** a suite run's probe EVALUATIONS are model-owned
(inside `models/<id>/evaluations/`) and are never touched by suite-run
deletion (proven live); a blocked sample deletion changes nothing on
disk (byte-verified); recipes / policies / probe-suite definitions
remain immutable by design; the M67 model guard is byte-for-byte
unchanged.

## 4. Unblock-Chain Coherence

**Deleting the referencing runtime records makes a blocked model
deletable, live — with M66 usage, M67 retention and the model DELETE
agreeing at every step.** The chain (proven in tests and live on the
copy): a model referenced by a suite run + a sample + a quality
measurement shows exactly those three EXTERNAL categories in its M66
usage and exactly those three blockers (canonical order
`suite_run < sample < sample_quality`) in its M67 retention view, and
its DELETE refuses on them. Deleting the measurement unblocks nothing
by itself (the sample still blocks — shown step-by-step); deleting the
sample leaves the suite run; deleting the suite run drains
`external_references` to 0 — the retention view flips to deletable
with empty blockers, and the model DELETE succeeds with stats exactly
equal to the retention view's files/bytes. No cache exists anywhere:
every view recomputes from the ONE listings on each call, and the
guard code that refuses is the same code that earlier refused — only
its live input set shrank. The reverse direction holds by
construction: creating a new reference (a suite run against a fresh
model) flips the view back to blocked immediately (tested live in the
M67 suite's recompute test, re-verified here through C2's blocked
state arising from fresh creations).

## 5. Verification

**Focused tests (4 new, `tests/test_suite_sample_lifecycle.py` →
641 + 4 = 645):** suite-run deletion (stats == independent walk, only
its directory removed, probe evaluations preserved — count equality,
M66 usage recomputes live, repeat/unknown → FileNotFoundError,
tampered `result_hash` → RuntimeError and corrupt manifest →
registry-invisible 404, both with pre/post SHA-256 equality and a
restore-then-delete round-trip proving integrity — not scope —
refused); sample + measurement (blocked deletion with the typed
blocker == the independent manifest-parse oracle, zero mutation;
measurement deletion with exact stats and isolation; sample deletion
afterwards; unknowns; tampered-hash and corrupt-manifest refusals for
both records with restoration and clean deletion after); the unblock
chains (self-contained: both chains step-by-step with the M66/M67
views and the model DELETE agreeing at every step, ending in exact
model-deletion stats == the retention view); and the full HTTP
lifecycle (fixture through public routes; the typed sample 409; the
model 409 listing all three blockers in canonical order; the three
deletions over HTTP; the model then deleting; 404s and deterministic
repeats; OpenAPI: 93 paths, the three paths now `{get, delete}`, the
delete-operation set == 7, `SampleDeletionBlocked` documented on the
409). **Updated tests (each an intentional contract change, documented
in place):** the M66/M67 delete-set assertions 4 → 7 paths; the two
M15/M16-era "DELETE → 405" assertions now assert the new
verified-deletion contract end-to-end (PUT remains 405 — no edit
endpoints exist). **One pre-existing flake fixed with cause analysis:**
`test_first_generated_token_conditioned_on_prompt` failed ONCE in
eight runs — its tolerance compared the record's 6-decimal-ROUNDED
`loss_nats` against a full-precision float32 recomputation at 1e-6,
numerically invalid at rounding boundaries; made like-for-like
(round the recomputation, 2e-6 for float32 summation-order noise; the
semantic claim — the measurement IS the mean NLL over the generated
positions — is unchanged). **Full suite: 645 passed ×2 consecutive
(exit 0 both)**, plus the 641 baseline and diagnostic runs; pyflakes +
compileall clean; OpenAPI standalone-verified.

## 6. Live Certification

`smoke_m68_live.py` (port 8791). **First execution: 21/24** — C4–C6
failed on a **pure script assertion bug** (the independent
`walk_stats(dir)` was evaluated inside each check AFTER the deletion
had removed the directory, yielding (0,0)); the protocol was followed:
stopped, verified read-only that production was byte-identical
(52/52, zero mutations — the server log confirms the only production
DELETE was the M67 refusal 409), confirmed the copy was discarded,
fixed the evaluation order (stats captured before each deletion), and
re-executed. **Second execution: 24/24 PASSED** — production
read-only phase (project/registry; disk == `m68_pre.sha256` 52/52;
M62 13/10/3; M63 52 files; M64 usage 16/15; M66 usage == certified
counts == the independent manifest oracle; M67 retention blocked by
the recipe only; the three new DELETE routes' unknown-id 404s; the M67
model DELETE unchanged — 409 with the recipe blocker; deterministic
byte-identical repeats; OpenAPI 93 paths / 7 delete operations with
the 409 model documented; zero mutation), then the **disposable-copy
phase** (guard parity for the production model; a fresh trained model
blocked by ALL THREE new families in canonical order; its sample
deletion refused with the typed blocker and zero mutation; the
measurement, sample and suite run deleted with exact stats vs
independent pre-walks; the suite run's probe evaluation PRESERVED —
no cascade; the model then unblocked and deleted with stats == the
retention view; the copy returned **byte-for-byte to its post-copy
state plus exactly the one immutable suite definition registered for
the fixture**; copy discarded). Production DELETEs attempted: two
(one per execution), both refusals; all mutating deletions ran on
discarded copies.

## 7. Storage Proof

**Production: before == after, exactly** — 52 files / 8,926,410 B,
every SHA-256 unchanged against `m68_pre.sha256` (captured after the
post-re-provisioning rebuild and verified unchanged across both smoke
executions and all test runs), 0 tmp entries, 0 manifest modifications,
0 new persistent files (no dependency index, no retention record, no
reference count — §22's banned list honored; blocker analysis is
live-computed). **The disposable-copy accounting:** each deletion's
`files_removed`/`bytes_reclaimed` equaled its independent pre-walk
(measurement 1 file, sample 1 file, suite run 1 file, model ==
retention files/bytes); the full-circle check proved the copy's final
inventory equals its post-copy inventory plus exactly the one
registered suite manifest — every created artifact was either deleted
through the new verified routes or is an immutable definition; no
`.tmp-delete-*` residue; 0 leftover copies after disposal.

## 8. Git

- `M68: explicit suite-run & sample lifecycle` — the three
  family-engine delete primitives, the facade guards + blocker helper,
  schemas, the three DELETE routes (operations on existing paths),
  landing/README, the new test file, the delete-set/contract/flake
  test updates, the smoke, this report.
- `M68: pre-smoke production inventory (m68_pre.sha256)` — the
  rebuilt production root's certification baseline (52 files).
- Both pushed to `origin/arena/01a071e9-code-forge`; remote branch tip
  == HEAD; worktree clean (only the untracked `egg-info`).

## 9. Final Status & Next Milestone

**M68 is complete** against the quality bar: the established
guard/verify/atomic pattern reused with zero new deletion primitives
and zero second scanners; the sample-quality blocker decision made
explicitly from inspection and tested in both directions; no cascade
(probe evaluations preserved, blocked deletions byte-inert); the
M67 guard untouched with the unblock chain proven live end-to-end;
independent oracles agreeing (stats, blocker ids, usage); 645 ×2
green; statics clean; OpenAPI correct (93 paths, 7 delete operations);
production byte-identical through both smoke executions; destructive
certification only on discarded copies. **Limitations (documented):**
production contains no suite runs/samples/measurements (counts 0), so
their production-exercisable surface is the unknown-id 404 path — the
success paths were certified on the copy; recipes, policies and
probe-suite definitions remain undeletable BY DESIGN (immutable
registries — a model blocked ONLY by a recipe or policy can never
become deletable through existing operations); suite-run deletion
leaves its probe evaluations in place intentionally (model-owned
history — deleting those is future work).

**The lifecycle lattice after M68:** datasets/tokenizers (M65),
checkpoints (M61), models (M67), and all root-level runtime records
(M68) have explicit verified deletion; the remaining undeletable
records are the model-OWNED internal families — evaluations,
comparisons, gate decisions, workflow records — which go with the
model atomically but cannot be pruned individually. **Next milestone —
M69, model-owned record retention (usage view first):** a read-only
dependency overview for ONE model's internal records (what references
an evaluation/comparison/gate/workflow record — e.g. gate decisions
reference comparisons by-comparison; suite-run probe results name
evaluation ids inside their records), the M66 pattern one level down,
followed in M70 by explicit per-record deletion with internal
reference safety. The ready-to-paste prompt follows.

---

### M69 — MODEL-OWNED RECORD USAGE OVERVIEW (ready-to-paste prompt)

```
Implement **Milestone 69 — Model-Owned Record Usage Overview** for AI
Model Forge.

M61-M68 completed explicit verified deletion for datasets, tokenizers,
checkpoints, models and every root-level runtime record. The remaining
undeletable records are the model-OWNED internal families —
evaluations, comparisons, gate decisions, workflow records — which are
removed atomically WITH the model (M67 ownership) but can never be
pruned individually. M69 is the read-only discovery step for that
(the M66 pattern one level down); deletion itself is M70.

### 1. INSPECT FIRST — DO NOT MODIFY YET

Before changing anything, inspect and identify:

* the four record families' persistence: EvaluationRecord /
  ComparisonRecord / GateDecision / WorkflowRecord — storage layout
  (models/<id>/{evaluations,comparisons,gates,workflows}/), fields,
  loaders, listings (including every by-* filter);
* EVERY persisted cross-reference INTO and OUT OF these records:
  - gate decisions referencing comparisons (by-comparison gates) and
    embedding policies;
  - suite-run records persisting probe evaluation_ids (root-level
    references INTO model-owned evaluations);
  - workflow records embedding stage configs, checkpoint refs and
    stage outcome records (training runs, evaluations, gates created
    BY the workflow);
  - the model manifest's latest_checkpoint / best_checkpoint pointers
    and training_provenance (what the MODEL itself references);
  - checkpoint references from comparisons/gates/samples/suite-runs
    (already covered by M61/M62 — do not duplicate, just understand);
* the M64/M66 usage-analysis architecture and the M62 checkpoint
  retention analysis (the reuse precedents);
* the OpenAPI conventions (expect 93 paths, 645 tests) and the
  README/landing conventions.

Record the pre-implementation baseline. Enumerate the ACTUAL reference
topology before designing anything; do not invent categories.

### 2. M69 OBJECTIVE

Read-only usage/dependency overviews for ONE model's internal records,
computed live from the ONE authoritative listings (never a second
scanner):

GET /models/{model_id}/evaluations/{eval_id}/usage
GET /models/{model_id}/comparisons/{comparison_id}/usage
GET /models/{model_id}/gates/{decision_id}/usage
GET /models/{model_id}/workflows/{workflow_id}/usage

Each answers: "what persisted records reference THIS record?" — with
ordered categories from the ACTUAL inspected topology, sorted
reference ids, deterministic counts, and a `referenced` /
total summary. Unknown or registry-invisible record -> the family's
404. A fresh record with no references -> valid empty usage. GET never
mutates. Lightweight reference summaries only — never duplicate full
records.

### 3. DESIGN CONSTRAINTS

* REUSE the ONE listings and existing by-* filters wherever they
  exist; derive cross-references from the AUTHORITATIVE records (e.g.
  a gate's persisted comparison_id), not from new scans;
* the facade is the public entry point; the API stays a thin adapter;
* zero new persistent files, zero mutation, byte-identical repeats;
* do NOT implement any deletion in M69 (M70), and do NOT touch any
  existing guard (M61/M65/M67/M68 semantics byte-identical);
* document the category order deterministically per family.

### 4. TEST MATRIX

* per family: known record returns usage; unknown -> 404; empty case;
  deterministic repeats; zero mutation (SHA-256 inventories);
* category coverage per the INSPECTED topology (e.g. a by-comparison
  gate appears in its comparison's usage; a suite run's probe
  evaluation appears in the evaluation's usage; a workflow's created
  records appear in the workflow's usage — whatever inspection proves
  real);
* INDEPENDENT manifest-parsing oracle agreeing on every category;
* isolation: other models'/records' references never leak;
* HTTP: shapes, 404s, determinism, OpenAPI (93 -> 97 if all four
  routes are new), README + landing updates; full regression green.

### 5. LIVE CERTIFICATION

Production read-only: the existing model's 4 evaluations / 2
comparisons / 2 gates / 3 workflows have REAL references (suite-run
probe ids are absent — suite_run count is 0 — but the inspected
internal topology applies); verify usage views against the independent
oracle and the listings; determinism; zero mutation; production
byte-identical. No destructive paths exist in M69. Run the smoke
exactly once; fix script bugs read-only and rerun only after verifying
zero mutation.

### 6. REPORT

Report exactly these 9 sections: Inspection & Baseline; Record
Reference Topology; API; Determinism & Reuse; Verification; Live
Certification; Storage Proof; Git; Final Status & Next Milestone —
then provide the next ready-to-paste implementation prompt.

### HARD CONSTRAINTS

* Inspect first; record the baseline; no invented categories.
* Reuse the ONE listings; no second scanner; no dependency indexes.
* Read-only: zero production writes; GET only.
* No deletion, no guards changed, no cascade, no automation.
* Preserve every previous milestone; full suite ×2; statics clean;
  production byte-identical; git clean and pushed.
```
