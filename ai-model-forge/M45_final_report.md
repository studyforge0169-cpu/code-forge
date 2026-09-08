# M45 Final Report — Gate-Decision History by Baseline Type

## Recovery & Baseline

No reset was needed at M45 start: `git fetch` confirmed `HEAD 8d9ec36`
(the M44 pre-milestone inventory commit) == `FETCH_HEAD`, the worktree
held only the M44-certified tree, the venv was intact, and production
`/home/user/ai-model-forge-data/` audited at exactly **96 files /
4,002,745 bytes / 0 `.tmp`**. Baseline re-proven before any change:
full suite **530 tests passing in 185.71s**, OpenAPI **76 paths**,
`m45_pre.sha256` captured (96 lines, paths relative to
`/home/user/ai-model-forge-data/`). No live server processes were
running.

## Implementation

Three narrow edits, all inside the established
authoritative-listing → persisted-field-filter → response pattern:

1. **`app/gates.py`** — `GateEngine.list_decisions_for_baseline_type(
   model_id, baseline_type)` placed after
   `list_decisions_for_verdict()`. It is exactly the authoritative M6
   `list_decisions(model_id)` (deterministic `(created_at,
   decision_id)` ASC) filtered by `d.policy.baseline_type ==
   baseline_type` — the persisted NESTED field of the policy embedded
   verbatim in each decision. The filter is VERBATIM: never derived
   from checkpoint-id presence, comparison/evaluation references,
   policy contents outside `baseline_type`, the gate result, the
   verdict, loss deltas or timestamps. Unknown model →
   `FileNotFoundError` (404 at the API). Read-only, never writes.

2. **`app/engine.py`** — `GateBaselineType` added to the schemas
   import block and thin facade
   `ForgeEngine.list_gate_decisions_for_baseline_type()` after
   `list_gate_decisions_for_verdict()`.

3. **`app/api.py`** — `GET /models/{model_id}/gates/decisions/
   by-baseline-type/{baseline_type}` registered in the
   `/gates/decisions/...` family **after** by-verdict (M43) and
   **before** the generic `{decision_id}` route (literal path segments
   before the parameterized catch-all). `baseline_type` is typed
   `GateBaselineType` — the schema enum IS the contract (all four
   values incl. `evaluation_result_hash`; unsupported values are
   rejected 422 by FastAPI pre-handler, even for an unknown model).
   Landing-page bullet, endpoint `<li>` and section comment added.
   No new persistence, cache, index or registry.

`README.md` gained the `### Milestone 45` section (semantics, route
ordering, enum contract, natural valid-empty) and the test counts were
updated to 535 in both places.

## Tests

5 new tests, all first-try pass, reusing M6/M9/M43 fixtures (no
fabricated artifacts):

- **`tests/test_gates.py`** (+3, module 42/42): `_m45_state` cached
  fixture reuses `_m43_state` and adds a current-baseline gate (seed
  4501 via `run_gate(baseline_type="current", ...)`); tests cover
  exact listing parity per baseline type, TRUE disjoint partition
  (every decision exactly once, no None case — the field is
  required), and unknown-model 404 + natural valid-empty.
- **`tests/test_gates_api.py`** (+2, module 17/17): test 1 seeds
  three gates (one per populated kind) and asserts parity, partition,
  determinism; test 2 asserts 404 (unknown model + valid type), 422
  for all four unsupported-value probes (case variant, interior
  space, numeric, non-enum word) including unknown-model+invalid →
  422, model isolation, the M6 ghost-decision 404, and the
  M23/M34/M41/M43/M44/M42 regression surfaces + full OpenAPI 77
  verification.

Full suite ×2 with tmp cleanup between runs (0 live processes
verified first): **535 @ 190.72s** and **535 @ 195.46s**.
`compileall` OK; `pyflakes` clean (0 findings).

## OpenAPI

The sweep updated **29 genuine path-count assertions** `== 76` →
`== 77` across the test modules (0 stale `== 76` remaining) and added
**21 M45 ladder lines**. Verified: **77 paths**; the new path appears
exactly once with GET only, `tags=["gates"]`, `baseline_type`
parameter `$ref` `GateBaselineType`, items `$ref` `GateDecision`;
route order by-verdict < by-baseline-type < generic `{decision_id}`;
the M44/M43/M42/M41 routes still present exactly once.

## Live Smoke

`smoke_m45_live.py` (committed) against the production `FORGE_ROOT`
on port **8766**, server on `0.0.0.0`, run **three times: 34/34 PASS,
exit 0 each, first attempt**. Structure LIVE A–M:

- **A** baseline audit 96/4,002,745/0 + per-file SHA inventory +
  registries + M4–M44 pre-state (11 gate decisions, 7
  checkpoint-baseline, 4 by-verdict improved, 7 by-decision passed, 13
  workflows, 16 evaluations, 8 comparisons, 10 suite-runs, 4 samples,
  7 recipes, dashboard hash) + OpenAPI 77 + known pair 200.
- **B** distribution DISCOVERED live from the M6 listing + the
  `GateBaselineType` schema enum (not assumed):
  **checkpoint → 7, current → 2, minimum_loss → 2,
  evaluation_result_hash → 0** among 11 decisions; enum holds exactly
  the four values.
- **C** all four groups return the EXACT locally-filtered M6 records
  (no missing/extra/duplicate; persisted nested
  `policy.baseline_type` VERBATIM; verbatim detail-getter parity for
  all 11 records).
- **D** three GETs raw-byte-identical.
- **E** all four enum values under `b5bc905326b6` → 200 `[]` (valid
  empty; `evaluation_result_hash` is a VALID value that returns `[]`
  even on the known model — no route omitted for empty categories).
- **F** unknown model + valid type → 404; unsupported values → 422 on
  the known model (case variant, interior space, numeric, non-enum
  word); unsupported + unknown model → 422 (schema validation fires
  pre-handler).
- **G** TRUE disjoint partition: 7 + 2 + 2 + 0 == 11, pairwise
  disjoint, union == full listing, no None case.
- **H** every group preserves the authoritative `(created_at,
  decision_id)` ASC order.
- **I** M23/M34/M41/M43 + M6 regressions with listing parity
  (by-comparison 4, by-policy 1, by-decision passed 7, by-verdict
  improved 4 / regressed 3 / unchanged 2 with the 2 null-verdict
  records in NO group).
- **J** M22–M44 regressions: suite-runs 10/10/summary-10; evaluations
  3/3/3 by-checkpoint, 16 by-dataset, 16 by-tokenizer, by-split
  14/2/0, by-state-kind 9/7; comparisons 6/5/1 by-checkpoint, 8/8
  by-dataset/tokenizer, by-split 8/0/0, by-verdict 3/3/2, by-state-kind
  8/2 either-side; samples 4/0/0 by-checkpoint, 4 by-tokenizer,
  by-strategy 2/2; sample-quality 2 by-tokenizer; workflows
  9/3/1 by-status, 2 by-recipe.
- **K** M12/M16/M35: recipe registry 7 verbatim, global
  m12-live-suite runs 2, zero-run recipes 200 `[]`, unknown recipe
  404, M16 evaluation ids still paired.
- **L** dashboard result_hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite registries
  unchanged.
- **M** OpenAPI 77 with correct route placement; by-baseline-type
  still deterministic at the end; every pre-existing file
  byte-identical; zero new files; totals unchanged.

## Storage Integrity

Final audit after the third smoke run and server shutdown (0 live
processes verified, `/tmp/forge-tests-*` cleaned):
`sha256sum -c m45_pre.sha256` run from
`/home/user/ai-model-forge-data/` (manifest paths are FORGE_ROOT
relative) → **96/96 OK, 0 non-OK**; storage exactly **96 files /
4,002,745 bytes / 0 `.tmp`**. M45 is read-only — zero production
writes. (Process note: running the check from the wrong directory
yields a silent all-fail — the 96 OK lines must be counted from the
FORGE_ROOT directory.)

## Regression / Compatibility

No pre-existing route, schema, engine method, storage layout or
registry changed. All 530 pre-existing tests pass unmodified except
the 29 genuine OpenAPI count assertions (76 → 77) and 21 ladder-line
additions. Production evidence re-verified live: M23 by-policy 1,
M34 by-comparison 4, M41 by-decision 7/4, M43 by-verdict 4/3/2
(null excluded), M44 comparisons by-state-kind 8/2, M42 workflows
9/3/1, M38 evaluations by-state-kind 9/7, dashboard hash, all
registries — byte-identical behavior. `b5bc905326b6` still owns no
gate decisions (200 `[]` for every baseline type).

## Git / Commit State

Two commits on `arena/01a071e9-code-forge`, both pushed, ending with
`HEAD == FETCH_HEAD`:

1. `M45: add gate-decision history by baseline type` — README.md,
   app/api.py, app/engine.py, app/gates.py, tests/test_gates.py,
   tests/test_gates_api.py, the eight swept test modules,
   smoke_m45_live.py and this report.
2. `M45: pre-milestone production inventory (m45_pre.sha256)` — the
   untracked baseline manifest (96 lines).

`ai_model_forge.egg-info/` remains untracked and uncommitted.

## Next Milestone

End-of-M45 live inspection (production server on port 8766, real
payloads + schema + engine code): every remaining simple enum/identity
axis is single-valued or already implemented — gate `policy.split`
(all 11 validation), `dataset_version` (all 1), suite-run
`state.state_kind` (all 10 checkpoint) and `status` (all completed),
sample `checkpoint_id` (all 4 = 0511), workflow `terminal_reason`
(nullable free-form), checkpoints `decision` (all accept; enum
accept/not_best) and `method` (all continued_pretraining). The M44
runner-up — M16 sample-quality by `sample_id` — turned out to be
ALREADY IMPLEMENTED as M19 (`GET .../sample-quality/by-sample/
{sample_id}`; the live grep caught what a stale plan would have
missed), so that candidate is retired. The one remaining axis with a
persisted registry relationship AND two non-empty live groups is the
**training-run lineage of checkpoints**: every `CheckpointRecord`
carries a REQUIRED `run_id` (plain free-form uuid hex, "lineage
within/across runs" per the schema comment), and every run is
registered in the model's OWN manifest `training_provenance`
(`RunProvenance.run_id`, REQUIRED, persisted in `manifest.json`).
Production `4a0a871886ef`: 2 provenance entries — run `291a16d755fc`
→ 2 checkpoints (30a8 step 10, 025e step 20), run `85438934f86a` →
1 (0511 step 16) — a TRUE disjoint partition of the 3 checkpoints
(`run_id` is required, no None case). The M4 listing is deterministic
in `(step, created_at)` ASC order (verified live in
`TrainingEngine.list_checkpoints`). `b5bc905326b6` has 0 provenance
entries and 0 checkpoints. A run registered with zero checkpoints IS
constructible via the real engine (steps < `eval_every_steps`
appends the provenance entry without saving a checkpoint), giving a
natural valid-empty without fabrication. Runner-up inspected and set
aside: checkpoints by `parent_checkpoint_id` (2 live groups: root →
1, 30a8 → 2 — but the field is `Optional[str]`; the root group is
unrepresentable verbatim in a path parameter and would need a
sentinel, violating the verbatim-filter discipline).

Copy-ready M46 prompt:

> M46 — CHECKPOINT HISTORY BY TRAINING RUN
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped checkpoint records by the training run that produced
> them. Do NOT jump ahead into aggregation, rankings, lineage graphs,
> auto-gating, rollback, retraining, optimization, HPO, RL, Gemini
> integration, inference, deployment, or autonomous improvement.
> Preserve the minimum-files/minimum-storage architecture and the
> established authoritative-listing → persisted-field-filter →
> response pattern.
>
> STEP 0 — VERIFY STATE AND BASELINE. Inspect the repository and git
> state. If the familiar reset signature appears (HEAD `f86b670` /
> "Add files via upload", files untracked, production data missing,
> venv missing), execute the established recovery runbook: `git
> fetch` + `git reset --hard FETCH_HEAD` (never force-push, never
> discard authoritative commits), restore `ai-model-forge-data/` from
> the workspace zip by extracting ONLY its `ai-model-forge-data/`
> prefix, verify against `m45_pre.sha256` (run `sha256sum -c` from
> `/home/user/ai-model-forge-data/` — manifest paths are FORGE_ROOT
> relative; count the OK lines), rebuild the venv (`python3 -m venv
> ~/.venv` + `pip install -e ai-model-forge[dev] pyflakes`), then
> re-prove the baseline. Expected baseline: M45 certified; HEAD ==
> FETCH_HEAD == the M45 inventory commit; 535 tests passing; OpenAPI
> 77; production 96 files / 4,002,745 bytes / 0 tmp; zero SHA drift;
> production checkpoint run distribution of `4a0a871886ef`:
> `291a16d755fc` -> 2 (30a8 step 10, 025e step 20) /
> `85438934f86a` -> 1 (0511 step 16); manifest training_provenance
> holds exactly those two run ids; `b5bc905326b6` has 0 provenance
> entries and 0 checkpoints. Capture `m46_pre.sha256` and record the
> exact baseline before changing anything. Inspect `app/training.py`
> (`list_checkpoints()` sorts `(step, created_at)` ASC;
> `get_checkpoint()`), `app/engine.py` (facades at ~ln 269), and
> `app/api.py` (the `/models/{id}/checkpoints` family — currently
> ONLY the listing and the generic `{checkpoint_id}` route,
> `response_model=list[dict]`, `tags=["training"]`).
>
> STEP 1 — GROUNDED SELECTION. The selection is already grounded in
> live post-M45 evidence (recorded in `M45_final_report.md` §9):
> `CheckpointRecord.run_id` is a REQUIRED persisted identity field
> (free-form uuid hex, NOT an enum) with TWO non-empty live groups
> (2/1) forming a TRUE disjoint partition of the 3 production
> checkpoints; the owning model's manifest `training_provenance`
> (RunProvenance.run_id, REQUIRED) is the persisted registry that
> validates it. If the live state contradicts any of this, re-run the
> end-of-milestone inspection and re-ground before implementing.
>
> STEP 2 — IMPLEMENT THE THIN FILTER.
> - Engine (`app/training.py`): `list_checkpoints_for_run(model_id,
>   run_id)` — resolve the model (`FileNotFoundError` → 404),
>   validate the run id against the model's OWN
>   `training_provenance` run ids (unknown run or a run id belonging
>   to another model → `FileNotFoundError` → 404; model-scoped
>   ownership, exactly the M19 sample-ownership precedent), then
>   return `[c for c in self.list_checkpoints(model_id) if
>   c.run_id == run_id]` — the authoritative M4 listing `(step,
>   created_at)` ASC filtered VERBATIM by the checkpoint's own
>   persisted `run_id`. NEVER derive run membership from steps,
>   epochs, timestamps, losses, parent ids, or any provenance field
>   other than the checkpoint's own `run_id`; the provenance list is
>   used ONLY for existence validation (the 404), never for building
>   the response. A registered run with no checkpoints returns [].
>   Read-only, never writes.
> - Facade (`app/engine.py`): thin
>   `list_checkpoints_for_run()` delegation, placed after
>   `list_checkpoints`.
> - API (`app/api.py`): `GET /models/{model_id}/checkpoints/by-run/
>   {run_id}` registered AFTER the `/checkpoints` listing and
>   BEFORE the generic `/checkpoints/{checkpoint_id}` route (literal
>   path segments before the parameterized catch-all — the gates
>   family is the precedent). `response_model=list[dict]`,
>   `tags=["training"]`, verbatim `model_dump(mode="json")` payloads,
>   `FileNotFoundError` → 404. `run_id` is free-form (no enum → no
>   422 path; an unknown model 404s first). Add the landing-page
>   bullet, endpoint `<li>` and section comment following the M45
>   wording. No new persistence, cache, index, registry or schema.
> - README: add `### Milestone 46` before `## Quickstart` (semantics,
>   route ordering, provenance-validated 404, natural valid-empty)
>   and update both test-count mentions.
>
> STEP 3 — TESTS. ~3 engine tests (reuse the M4/M5/M7 cached
>   training fixtures; cover exact listing parity per run, TRUE
>   disjoint partition 2+1 == all, and the zero-checkpoint run —
>   constructible through the REAL engine with steps <
>   eval_every_steps, never by fabricating files) + ~2 API tests
>   (parity/partition/determinism; 404 unknown model, 404 unknown
>   run, 404 for a run id owned by another model, `b5bc905326b6`
>   model isolation, the M4/M5 listing + detail regressions, and a
>   full OpenAPI verification). No fabricated artifacts. Run the
>   gate ladder: focused modules first, then the FULL suite twice
>   (expect 535 -> ~540), tmp cleanup between runs with 0 live
>   processes first, `compileall`, `pyflakes` clean. Sweep ONLY the
>   genuine OpenAPI path-count assertions `== 77` → `== 78` (verify
>   0 stale remain) and add the M46 ladder lines.
>
> STEP 4 — LIVE SMOKE (port 8767). Write `smoke_m46_live.py` with
>   the A–M structure of `smoke_m45_live.py`: baseline audit
>   96/4,002,745/0 + SHA inventory; DISCOVER the run distribution
>   live from the M4 listing + manifest training_provenance
>   (expect 291a16d755fc -> 2 / 85438934f86a -> 1); exact listing
>   parity per run with detail-getter parity for all 3 checkpoints;
>   byte-identical x3 repeats; TRUE partition 2 + 1 == 3 with the
>   `(step, created_at)` order preserved; 404 unknown model / unknown
>   run / other-model run (note `b5bc905326b6` has no provenance —
>   every run id 404s under it, mirroring M19 ownership); the
>   M4/M5/M22-M45 regression surfaces (checkpoints 3 total, by-run
>   groups, evaluations 3/3/3 by-checkpoint, comparisons 6/5/1,
>   gates 11 with by-baseline-type 7/2/2/0, workflows 13 with 9/3/1,
>   suite-runs 10, dashboard hash
>   `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`,
>   registries); OpenAPI 78 with the new path before the generic
>   `{checkpoint_id}` route; final storage audit 96/4,002,745/0 with
>   zero drift vs `m46_pre.sha256`. Start the server with
>   `FORGE_ROOT=/home/user/ai-model-forge-data` on 0.0.0.0:8767, run
>   the smoke three times (exit 0 each), then stop it, verify 0 live
>   processes, clean `/tmp/forge-tests-*`, and run the final
>   `sha256sum -c m46_pre.sha256` FROM THE FORGE_ROOT (96/96 OK).
>
> STEP 5 — REPORT + COMMITS. Write `M46_final_report.md` with
>   EXACTLY 9 sections (Recovery & Baseline / Implementation / Tests
>   / OpenAPI / Live Smoke / Storage Integrity / Regression &
>   Compatibility / Git / Commit State / Next Milestone). §9 must
>   re-ground the M47 selection from LIVE post-M46 inspection
>   (persisted field/registry relationship preferred, at least two
>   meaningful live groups where practical, thin filter, no schema
>   redesign, no new storage, no aggregation/recalculation; careful
>   with nested/nullable/free-form/provenance fields; two-sided
>   candidates reuse prior either-side semantics; mention the
>   runner-up) and contain the complete copy-ready M47 prompt
>   (title, recovery runbook, grounded selection, endpoint, engine
>   method, field + semantics, route ordering, tests, OpenAPI count,
>   smoke port 8768, storage invariants, the 9-section requirement,
>   automatic M47). Commit the implementation (README, app/, tests/,
>   smoke, report) as `M46: add checkpoint history by training run`,
>   then `m46_pre.sha256` as a separate inventory commit, push both
>   to `arena/01a071e9-code-forge`, verify `HEAD == FETCH_HEAD`, and
>   present the report. Do NOT commit `ai_model_forge.egg-info/`.
