# M46 Final Report — Checkpoint History by Training Run

## 1. Recovery & Baseline

The sandbox HAD been reset (HEAD `f86b670` "Add files via upload", all
files untracked, production data missing, venv missing). The
established recovery runbook was executed: `git fetch` +
`git reset --hard FETCH_HEAD` on `arena/01a071e9-code-forge` to
`6c883a7` (the M45 inventory commit; never force-pushed), extraction
of ONLY the 96 `ai-model-forge-data/` members from
`code forge.zip`, verification against `m45_pre.sha256` run from
`/home/user/ai-model-forge-data/` (96/96 OK, 0 non-OK), venv rebuild
(`python3 -m venv ~/.venv` + `pip install -e ai-model-forge[dev]
pyflakes` — torch 2.14.0). Baseline re-proven before any change: full
suite **535 passed @ 127.41s**, OpenAPI **77 paths**, production
exactly **96 files / 4,002,745 bytes / 0 `.tmp`**, 0 live processes.
`m46_pre.sha256` captured (96 lines, FORGE_ROOT-relative).

## 2. Implementation

Four narrow edits inside the established authoritative-listing →
persisted-field-filter → response pattern; no new storage, registry,
index or cache:

1. **`app/training.py`** — `TrainingEngine.list_checkpoints_for_run(
   model_id, run_id)` placed after `get_checkpoint()`. It loads the
   model record (`storage.load_record`, FileNotFoundError → 404 for
   unknown models), validates run OWNERSHIP against the model's OWN
   manifest `training_provenance` (`run_id in {p.run_id for p in
   record.training_provenance}`; unknown run or a run belonging to
   another model → FileNotFoundError → 404 — exactly the M19
   sample-ownership precedent), then returns the authoritative M3
   `list_checkpoints(model_id)` (deterministic `(step, created_at)`
   ASC) filtered VERBATIM by `c.run_id == run_id`. The provenance
   list is used ONLY for existence validation — never for building
   the response; membership is never derived from checkpoint
   directories, steps, epochs, timestamps, losses or parent ids.
   Read-only, never writes.
2. **`app/engine.py`** — thin facade
   `ForgeEngine.list_checkpoints_for_run()` after `get_checkpoint`.
3. **`app/api.py`** — `GET /models/{model_id}/checkpoints/by-run/
   {run_id}` registered AFTER the `/checkpoints` listing and BEFORE
   the generic `/checkpoints/{checkpoint_id}` route (literal path
   segments before the parameterized catch-all), `response_model=
   list[dict]`, `tags=["training"]`, verbatim `model_dump(mode=
   "json")` payloads, FileNotFoundError → 404. `run_id` is a
   persisted identifier, not an enum — no artificial validation, no
   422 path. Landing bullet, endpoint `<li>` and a Training section
   milestone comment added.
4. **`README.md`** — `### Milestone 46` section before
   `## Quickstart` (concept, verbatim filtering, provenance-validated
   boundaries, honest statement that no separate run registry is
   introduced — the model manifest IS the registry) + the training.py
   layout note + test counts 535 → 540 in both places.

## 3. Tests

5 new tests (3 engine + 2 API), reusing the M3 fixtures and the
module's own tampering conventions — no fabricated run ids:

- **`tests/test_training.py`** (+3, module 23/23): parity + TRUE
  disjoint partition (2+1 == 3) + per-group `(step, created_at)`
  order for two real runs; the zero-checkpoint path reached through
  the M3 listing's corruption resilience (honest engine reality —
  see below); unknown model / unknown run / cross-model run ids all
  FileNotFoundError.
- **`tests/test_training_api.py`** (+2, module 8/8): HTTP parity /
  partition / determinism / detail-getter parity; 404 × 4 (unknown
  model, unknown run, other model's run, reverse direction), 200 []
  for the registered-run-with-no-readable-checkpoints case, M3
  listing/detail/provenance regressions, full OpenAPI 78 verification
  (new path once, GET-only, tag training, `model_id`+`run_id`
  parameters, array-of-object response, route order listing < by-run
  < generic detail, M45 path still present).

One honest fix was required: the first draft of the engine
zero-checkpoint test assumed a run with `steps < eval_every_steps`
produces no checkpoints. The engine actually guarantees the opposite
— `eval_points` is built as `range(eval_every, total+1, eval_every)
| {total_steps}`, so the FINAL step is always an eval point and a
healthy registered run ALWAYS owns ≥ 1 checkpoint (`steps >= 1`). The
test was rewritten to reach the `[]` path through the M3 listing's
documented corruption resilience (unreadable manifests are skipped),
mirroring the module's existing tampering tests. Full suite ×2 with
tmp cleanup between runs (0 live processes verified first):
**540 @ 125.78s** and **540 @ 125.01s**. `compileall` OK; `pyflakes`
0 findings.

## 4. OpenAPI

Before 77 → after **78**. The sweep updated **30 genuine
path-count assertions** `== 77` → `== 78` across 8 test modules (0
stale remaining) and added **21 M46 ladder lines** (20 ladder-form +
1 short-form site). Verified: the new path appears exactly once,
GET-only, `tags=["training"]`, parameters `model_id` + `run_id`,
response `{"type": "array", "items": {"type": "object",
"additionalProperties": true}}` (same shape as the M3 checkpoint
family), route order `/checkpoints` (idx 17) < `/checkpoints/by-run/
{run_id}` (18) < `/checkpoints/{checkpoint_id}` (19); M45/M44/M43/
M42/M41 routes still present exactly once.

## 5. Live Smoke

`smoke_m46_live.py` (committed) against the production `FORGE_ROOT`
on port **8767**, server on `0.0.0.0`, run **three times: 31/31
PASS, exit 0 each, first attempt**. Structure LIVE A–M:

- **A** baseline audit 96/4,002,745/0 + per-file SHA inventory +
  registries + M3–M45 pre-state + dashboard hash + OpenAPI 78 +
  known pair 200.
- **B** distribution DISCOVERED live from the M3 listing + the live
  manifest `training_provenance` (not assumed):
  **291a16d755fc → 2, 85438934f86a → 1** among 3 checkpoints;
  provenance registers exactly those two run ids; `b5bc905326b6`
  has 0 provenance + 0 checkpoints.
- **C** both runs return the EXACT locally-filtered M3 records;
  verbatim detail-getter parity for all 3 checkpoints.
- **D** three GETs raw-byte-identical.
- **E** empty boundaries without fabrication: `b5bc905326b6`
  listing 200 `[]` and both REAL run ids 404 under it (ownership);
  healthy-run invariant verified live — every registered production
  run owns ≥ 1 checkpoint, so no production run yields `[]` (the
  200 `[]` path is engine-real through listing resilience and is
  covered by the M46 tests; zero production writes performed).
- **F** unknown model + the REAL run id → 404; valid model +
  unknown run → 404; `run_id` is a persisted identifier, not an
  enum — no 422 path exists.
- **G** TRUE disjoint partition: 2 + 1 == 3, pairwise disjoint,
  union == full listing, no None case, and every checkpoint's
  `run_id` registered in provenance (total coverage).
- **H** every group preserves the authoritative M3 `(step,
  created_at)` ASC order.
- **I** M3/M4 regressions: listing identical to LIVE A, detail
  getters verbatim, provenance registry identical.
- **J** M24–M45 regressions all with listing parity: evaluations
  3/3/3 by-checkpoint, 16/16 by-dataset/tokenizer, by-split 14/2/0,
  by-state-kind 9/7; comparisons 6/5/1, 8/8, 8/0/0, by-verdict
  3/3/2, by-state-kind 8/2; samples 4/0/0, 4, by-strategy 2/2;
  sample-quality 2; suite-runs 10/10/summary-10; workflows 13,
  9/3/1 by-status, 2 by-recipe; gates 11 with by-decision 7/4,
  by-verdict 4/3/2, by-baseline-type 7/2/2/0, by-comparison 4,
  by-policy 1.
- **K** M12/M16/M35: recipe registry 7, global m12-live-suite runs
  2, zero-run recipes 200 `[]`, unknown recipe 404, M16 evaluation
  ids still paired.
- **L** dashboard result_hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  + full output unchanged; tokenizer/policy/probe-suite/recipe/model
  registries unchanged (exactly the two known models).
- **M** OpenAPI 78 with correct route placement + parameter names;
  by-run still deterministic at the end; every pre-existing file
  byte-identical; zero new files; totals unchanged.

## 6. Storage Integrity

Final audit after the third smoke run and server shutdown (0 live
processes verified, `/tmp/forge-tests-*` + smoke inventory cleaned):
`sha256sum -c m46_pre.sha256` run from
`/home/user/ai-model-forge-data/` → **96/96 OK, 0 non-OK**; storage
exactly **96 files / 4,002,745 bytes / 0 `.tmp`**. M46 is read-only —
zero production writes, zero new checkpoint manifests, no run
registry, no cache/index files, zero SHA drift, zero growth.

## 7. Regression / Compatibility

No pre-existing route, schema, engine method, storage layout or
registry changed. All 535 pre-existing tests pass unmodified except
the 30 genuine OpenAPI count assertions (77 → 78) and 21 ladder-line
additions. M3 checkpoint behavior (training run, listing, detail,
verify, rollback) is byte-identical — re-verified live (listing
unchanged across the smoke, detail getters verbatim, provenance
identical). M24–M45 history surfaces, the M17 dashboard hash and the
policy/probe-suite/recipe/tokenizer/model registries all re-verified
live and unchanged (smoke sections J/K/L). `b5bc905326b6` still owns
0 provenance entries and 0 checkpoints.

## 8. Git / Commit State

Two commits on `arena/01a071e9-code-forge`, both pushed:

1. `M46: add checkpoint history by training run` — README.md,
   app/api.py, app/engine.py, app/training.py,
   tests/test_training.py, tests/test_training_api.py, the eight
   swept test modules, smoke_m46_live.py and this report.
2. `M46: pre-milestone production inventory (m46_pre.sha256)` — the
   baseline manifest (96 lines).

HEAD == remote branch tip == branch-specific FETCH_HEAD;
`ai_model_forge.egg-info/` remains untracked and uncommitted; working
tree clean otherwise. No force-push.

## 9. Next Milestone

End-of-M46 live inspection (production server on port 8767, real
payloads + schemas + engine code, existing-implementation grep):
every remaining identity axis is consumed or unusable — workflow
`hint` / `failed_stage_id` (nullable free-form; None majorities),
`composition` (nullable nested recipe chain), sample `seed` /
`temperature` (nullable — None groups unrepresentable in a path),
model `parent_model_id` (no parent relationships in production),
checkpoint `decision` / `method` (single-valued: all accept /
continued_pretraining), sample-quality `window_rule` (constant
`single_window`), evaluation/comparison `dataset_version` (all 1),
suite-run `state.state_kind` / `status` (all checkpoint / completed).
The strongest remaining axis is a REQUIRED persisted classification
on the evaluation record: **`truncated: bool`** — the engine's
verbatim record of whether `max_eval_tokens` stopped the evaluation
before the split ended. Live distribution across the 16 evaluations
of `4a0a871886ef`: **False → 15, True → 1** (the truncated record is
`a884bf729ff7`, train split, current state, `records_covered` None,
`token_count` 500) — TWO populated groups forming a TRUE disjoint
partition (a boolean is required on every record — no None case).
The M4 listing is deterministic in `(created_at, eval_id)` ASC order
(verified live). `b5bc905326b6` owns 0 evaluations → natural `[]`
for BOTH values. No `by-truncated` exists anywhere (grep 0 hits).
The boolean gives a CLOSED two-value contract — exactly the shape of
the M41 by-decision grouping (passed/failed 7/4) — with no registry
and no arbitrary-value semantics. Runner-up inspected and set aside:
evaluations by `seed` (7 live groups: 11→4 / 1→3 / 7→2 / 3→2 /
7002→2 / 7101→2 / 12→1 — persisted required int, but a raw config
echo with an open-ended value space and no closed contract; grouping
by an arbitrary integer is weaker than grouping by a classification).

Copy-ready M47 prompt:

> M47 — EVALUATION HISTORY BY TRUNCATION
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped evaluation records by the persisted `truncated`
> flag. Do NOT jump ahead into aggregation, rankings, coverage
> statistics, re-evaluation, automatic re-runs with higher caps,
> training changes, rollback, HPO, RL, Gemini integration,
> inference, deployment, or autonomous improvement. Preserve the
> minimum-files/minimum-storage architecture and the established
> authoritative-listing → persisted-field-filter → response pattern.
>
> STEP 0 — VERIFY STATE AND BASELINE. Inspect the repository and git
> state. If the familiar reset signature appears (HEAD `f86b670` /
> "Add files via upload", files untracked, production data missing,
> venv missing), execute the established recovery runbook: `git
> fetch` + `git reset --hard FETCH_HEAD` on
> `arena/01a071e9-code-forge` (never force-push), restore
> `ai-model-forge-data/` from the workspace zip by extracting ONLY
> its `ai-model-forge-data/` prefix, verify against `m46_pre.sha256`
> (run `sha256sum -c` from `/home/user/ai-model-forge-data/` —
> manifest paths are FORGE_ROOT relative; count the OK lines),
> rebuild the venv (`python3 -m venv ~/.venv` + `pip install -e
> ai-model-forge[dev] pyflakes`), then re-prove the baseline.
> Expected baseline: M46 certified; HEAD == FETCH_HEAD == the M46
> inventory commit; 540 tests passing; OpenAPI 78; production 96
> files / 4,002,745 bytes / 0 tmp; zero SHA drift; production
> checkpoint run distribution of `4a0a871886ef`:
> `291a16d755fc` -> 2 / `85438934f86a` -> 1. Capture
> `m47_pre.sha256` and record the exact baseline before changing
> anything. Inspect `app/evaluation.py`
> (`list_evaluations()` ~ln 105; the by-checkpoint/by-dataset/
> by-tokenizer/by-split/by-state-kind filters; M4 order is
> `(created_at, eval_id)` ASC), `app/engine.py` (evaluation facades
> ~ln 346+), `app/api.py` (the `/models/{id}/evaluations/...`
> family: by-checkpoint ln ~1093, by-split ~1178, by-state-kind
> ~1207, generic `{eval_id}` ~1243 — all `response_model=
> list[EvaluationRecord]`, `tags=["evaluation"]`) and
> `app/schemas.py` (`EvaluationRecord.truncated: bool = False` —
> "max_eval_tokens stopped before the split ended", set by the
> engine on every record; `records_covered` is the paired nullable
> count but MUST NOT be used for filtering).
>
> STEP 1 — GROUNDED SELECTION. The selection is already grounded in
> live post-M46 evidence (recorded in `M46_final_report.md` §9):
> `truncated` is a persisted REQUIRED boolean on every
> EvaluationRecord with TWO populated live groups (False -> 15 /
> True -> 1 among the 16 evaluations of `4a0a871886ef`), forming a
> TRUE disjoint partition with no None case; `b5bc905326b6` owns 0
> evaluations (natural `[]` for both values); no by-truncated route
> exists anywhere. If the live state contradicts any of this, re-run
> the end-of-milestone inspection and re-ground before implementing.
>
> STEP 2 — IMPLEMENT THE THIN FILTER.
> - Engine (`app/evaluation.py`): `list_evaluations_for_truncated(
>   model_id, truncated)` placed after `list_evaluations_for_split`
>   (before/after the other by-* filters following file order) —
>   resolve the model (existing `_model_exists` FileNotFoundError →
>   404), then return `[e for e in self.list_evaluations(model_id)
>   if e.truncated == truncated]` — the authoritative M4 listing
>   `(created_at, eval_id)` ASC filtered VERBATIM by the persisted
>   boolean. NEVER derive truncation from `records_covered`,
>   `token_count`, splits, tokenizers, state kinds, losses,
>   perplexities, timestamps or config contents — the boolean IS the
>   record. A model with no truncated evaluations returns [] for
>   True (and vice versa). Read-only, never writes.
> - Facade (`app/engine.py`): thin
>   `list_evaluations_for_truncated()` delegation after
>   `list_evaluations_for_state_kind`.
> - API (`app/api.py`): `GET /models/{model_id}/evaluations/
>   by-truncated/{truncated}` registered in the evaluations family
>   AFTER by-state-kind (M38) and BEFORE the generic
>   `/evaluations/{eval_id}` route (literal path segments before the
>   parameterized catch-all). `truncated: bool` path parameter —
>   FastAPI parses true/false case-insensitively; anything else is
>   rejected 422 pre-handler (the boolean IS the contract — no
>   registry, no enum, no artificial validation). Unknown model +
>   valid boolean -> 404. `response_model=list[EvaluationRecord]`,
>   `tags=["evaluation"]`, verbatim payloads. Add the landing-page
>   bullet, endpoint `<li>` and the evaluations section milestone
>   comment following the M46 wording. No new persistence, cache,
>   index, registry or schema.
> - README: add `### Milestone 47` before `## Quickstart`
>   (semantics, route ordering, closed two-value contract, natural
>   valid-empty) and update both test-count mentions.
>
> STEP 3 — TESTS. ~3 engine tests (reuse the M4 evaluation fixtures;
> cover exact listing parity per value with a BOTH-groups seed —
> construct a truncated evaluation through the REAL engine with a
> small `max_eval_tokens` cap and a complete one without —, TRUE
> disjoint partition (False + True == all, no None case), and
> unknown-model FileNotFoundError) + ~2 API tests (parity/partition/
> determinism; 404 unknown model; 422 for non-boolean spellings on
> the known model incl. unknown-model+invalid -> 422; both values
> under an empty model -> 200 []; the M4 listing + detail + M24/
> M28/M30/M36/M38 by-* regressions; a full OpenAPI verification).
> No fabricated artifacts. Run the gate ladder: focused modules
> first, then the FULL suite twice (expect 540 -> ~545), tmp cleanup
> between runs with 0 live processes first, `compileall`, `pyflakes`
> clean. Sweep ONLY the genuine OpenAPI path-count assertions
> `== 78` -> `== 79` (verify 0 stale remain) and add the M47 ladder
> lines.
>
> STEP 4 — LIVE SMOKE (port 8768). Write `smoke_m47_live.py` with
> the A–M structure of `smoke_m46_live.py`: baseline audit
> 96/4,002,745/0 + SHA inventory; DISCOVER the truncation
> distribution live from the M4 listing (expect False -> 15 /
> True -> 1; the True record `a884bf729ff7`, train split, current
> state); exact listing parity per value with detail-getter parity
> for all 16 records; byte-identical x3 repeats; TRUE partition
> 15 + 1 == 16 with the `(created_at, eval_id)` order preserved;
> both values under `b5bc905326b6` -> 200 []; 404 unknown model +
> valid boolean; 422 non-boolean spellings (e.g. "maybe", "2") on
> the known model and on an unknown model; the M4/M24/M28/M30/M36/
> M38 evaluation + M26/M29/M31/M37/M39/M44 comparison + M27/M32/
> M40 sample + M22/M25 suite-run + M42/M35 workflow + M23/M34/M41/
> M43/M45 gate + M46 checkpoint-by-run (291a16d755fc -> 2 /
> 85438934f86a -> 1) regression surfaces; dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
> registries; OpenAPI 79 with the new path before the generic
> `{eval_id}` route; final storage audit 96/4,002,745/0 with zero
> drift vs `m47_pre.sha256`. Start the server with
> `FORGE_ROOT=/home/user/ai-model-forge-data` on 0.0.0.0:8768, run
> the smoke three times (exit 0 each), then stop it, verify 0 live
> processes, clean `/tmp/forge-tests-*`, and run the final
> `sha256sum -c m47_pre.sha256` FROM THE FORGE_ROOT (96/96 OK).
>
> STEP 5 — REPORT + COMMITS. Write `M47_final_report.md` with
> EXACTLY 9 sections (Recovery & Baseline / Implementation / Tests /
> OpenAPI / Live Smoke / Storage Integrity / Regression &
> Compatibility / Git / Commit State / Next Milestone). §9 must
> re-ground the M48 selection from LIVE post-M47 inspection
> (persisted field/registry relationship preferred, at least two
> meaningful live groups where practical, thin filter, no schema
> redesign, no new storage, no aggregation/recalculation; careful
> with nested/nullable/free-form/provenance fields; two-sided
> candidates reuse prior either-side semantics; mention the
> runner-up) and contain the complete copy-ready M48 prompt (title,
> recovery runbook, grounded selection, endpoint, engine method,
> field + semantics, route ordering, tests, OpenAPI count, smoke
> port 8769, storage invariants, the 9-section requirement,
> automatic M48). Commit the implementation (README, app/, tests/,
> smoke, report) as `M47: add evaluation history by truncation`,
> then `m47_pre.sha256` as a separate inventory commit, push both
> to `arena/01a071e9-code-forge`, verify `HEAD == FETCH_HEAD`
> (branch-specific fetch), and present the report. Do NOT commit
> `ai_model_forge.egg-info/`.
