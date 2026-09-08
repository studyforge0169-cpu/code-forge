# M47 Final Report — Evaluation History by Truncation

## 1. Recovery & Baseline

No reset was needed at M47 start: `git fetch` (branch-specific) confirmed
`HEAD d44d150` (the M46 inventory commit) == FETCH_HEAD == remote tip,
the worktree was clean (only `ai_model_forge.egg-info/` untracked), the
venv was intact, 0 live server processes, 0 stale tmp, and production
`/home/user/ai-model-forge-data/` audited at exactly **96 files /
4,002,745 bytes / 0 `.tmp`**. Baseline re-proven before any change:
full suite **540 passed @ 133.53s**, OpenAPI **78 paths**.
`m47_pre.sha256` captured (96 lines, FORGE_ROOT-relative, 96/96 OK).

## 2. Implementation

Four narrow edits inside the established authoritative-listing →
persisted-field-filter → response pattern; no new storage, registry,
index or cache:

1. **`app/evaluation.py`** — `EvaluationEngine.
   list_evaluations_for_truncated(model_id, truncated)` placed after
   `list_evaluations_for_state_kind()`. It is exactly the
   authoritative M4 `list_evaluations(model_id)` (deterministic
   `(created_at, eval_id)` ASC) filtered by `r.truncated ==
   truncated` — the persisted REQUIRED boolean the engine records on
   every `EvaluationRecord` at run time. Matched VERBATIM: never
   recalculated, never derived from `records_covered`, token counts,
   split length, the evaluation configuration, timestamps, durations,
   state kinds or any other field. Unknown model →
   `FileNotFoundError` (404 at the API). Read-only, never writes.
2. **`app/engine.py`** — thin facade
   `ForgeEngine.list_evaluations_for_truncated()` after
   `list_evaluations_for_state_kind`.
3. **`app/api.py`** — `GET /models/{model_id}/evaluations/
   by-truncated/{truncated}` registered AFTER by-state-kind (M38) and
   BEFORE the generic `/evaluations/{eval_id}` route.
   `truncated: bool` path parameter — non-boolean spellings are
   rejected 422 by FastAPI/Pydantic schema validation pre-handler
   (booleans are never silently reinterpreted, and the validation
   fires before the handler even for an unknown model); unknown model
   + valid boolean → 404. `response_model=list[EvaluationRecord]`,
   `tags=["evaluation"]`, verbatim payloads. The route docstring
   explicitly states the boolean carries NO quality judgment.
   Landing bullet, endpoint `<li>` and the evaluations section
   milestone comment added.
4. **`README.md`** — `### Milestone 47` section before
   `## Quickstart` (concept, verbatim filtering, closed two-value
   contract, no-quality-javadoc statement, boundaries) + the
   evaluation.py layout note + test counts 540 → 545 in both places.

## 3. Tests

5 new tests (3 engine + 2 API), reusing the M4/M28/M38 fixtures —
both statuses constructed through the REAL engine (a capped TRAIN-split
evaluation → `truncated=True`; full evaluations → `False`), no
fabricated records:

- **`tests/test_evaluation.py`** (+3, module 31/31): `_m47_env`
  cached fixture on top of `_m38_env` adding one capped evaluation;
  parity/order/verbatim detail parity for BOTH values; TRUE disjoint
  partition (False + True == all, no None case — every record's
  `truncated` is a bool); empty-history model → `[]` for both;
  unknown model → FileNotFoundError for both.
- **`tests/test_evaluation_api.py`** (+2, module 17/17): HTTP
  parity/partition/determinism/detail parity; 404 unknown model +
  valid boolean; 422 for three non-boolean spellings on the known
  model AND on an unknown model; valid-empty `200 []`; cross-model
  isolation; M4 listing/detail + M24/M36/M38 by-* + M46 by-run
  regressions; full OpenAPI 79 verification (new path once, GET-only,
  tag evaluation, boolean parameter schema, `EvaluationRecord` items,
  route order by-state-kind < by-truncated < generic detail, M46
  path still present).

One fix required: the OpenAPI parameter-schema assertion initially
compared the whole schema dict; FastAPI emits `{"type": "boolean",
"title": "Truncated"}` — corrected to assert the `"type"` key.
Full suite ×2 with tmp cleanup between runs (0 live processes
verified first): **545 @ 129.23s** and **545 @ 124.66s**.
`compileall` OK; `pyflakes` 0 findings.

## 4. OpenAPI

Before 78 → after **79**. The sweep updated **31 genuine path-count
assertions** `== 78` → `== 79` across the test modules (20
M46-ladder-form sites + 1 short-ladder site + 10 bare assertions; 0
stale remaining). Verified: the new path appears exactly once,
GET-only, `tags=["evaluation"]`, parameters `model_id` +
`truncated` with `{"type": "boolean"}` schema, response
`{"type": "array", "items": {"$ref": "#/components/schemas/
EvaluationRecord"}}`, route order by-state-kind < by-truncated <
generic `{eval_id}`; M46/M45/M44/M43/M42/M41 routes still present
exactly once.

## 5. Live Smoke

`smoke_m47_live.py` (committed) against the production `FORGE_ROOT`
on port **8768**, server on `0.0.0.0`, run **three times: 30/30
PASS, exit 0 each, first attempt**. Structure LIVE A–M:

- **A** baseline audit 96/4,002,745/0 + per-file SHA inventory +
  registries + M4–M46 pre-state + dashboard hash + OpenAPI 79 +
  known pair 200.
- **B** distribution DISCOVERED live from the M4 listing (not
  assumed): **false → 15, true → 1** among 16 evaluations; the
  single True record is `a884bf729ff7` (train split, current state);
  every record's `truncated` is a bool.
- **C** both groups return the EXACT locally-filtered M4 records;
  verbatim detail-getter parity for all 16 records.
- **D** three GETs raw-byte-identical.
- **E** both statuses under `b5bc905326b6` → 200 `[]`.
- **F** unknown model + valid boolean → 404; non-boolean spellings
  ("maybe", "2", "yes-no") → 422 on the known model AND on an
  unknown model (schema-level validation fires pre-handler).
- **G** TRUE disjoint partition: 15 + 1 == 16, pairwise disjoint,
  union == full listing, no None case.
- **H** every group preserves the authoritative `(created_at,
  eval_id)` ASC order.
- **I** M4/M24/M28/M30/M36/M38 regressions: listing identical to
  LIVE A, 3/3/3 by-checkpoint, 16/16 by-dataset/tokenizer, by-split
  14/2/0, by-state-kind 9/7 — all with listing parity.
- **J** M22–M46 regressions: comparisons 6/5/1, 8/8, 8/0/0,
  by-verdict 3/3/2, by-state-kind 8/2; samples 4/0/0, 4, by-strategy
  2/2; sample-quality 2; suite-runs 10/10/summary-10; workflows 13,
  9/3/1 by-status, 2 by-recipe; gates 11 with by-decision 7/4,
  by-verdict 4/3/2, by-baseline-type 7/2/2/0, by-comparison 4,
  by-policy 1; M46 checkpoints by-run 2/1.
- **K** M12/M16/M35: recipe registry 7, global m12-live-suite runs
  2, zero-run recipes 200 `[]`, unknown recipe 404, M16 evaluation
  ids still paired.
- **L** dashboard result_hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  + full output unchanged; tokenizer/policy/probe-suite/recipe/model
  registries unchanged.
- **M** OpenAPI 79 with boolean parameter schema + correct route
  placement; by-truncated still deterministic at the end; every
  pre-existing file byte-identical; zero new files; totals unchanged.

## 6. Storage Integrity

Final audit after the third smoke run and server shutdown (0 live
processes verified, `/tmp/forge-tests-*` + smoke inventory cleaned):
`sha256sum -c m47_pre.sha256` run from
`/home/user/ai-model-forge-data/` → **96/96 OK, 0 non-OK**; storage
exactly **96 files / 4,002,745 bytes / 0 `.tmp`**. M47 is read-only —
zero production writes, zero SHA drift, zero growth, no new
index/cache/registry files.

## 7. Regression / Compatibility

No pre-existing route, schema, engine method, storage layout or
registry changed. All 540 pre-existing tests pass unmodified except
the 31 genuine OpenAPI count assertions (78 → 79). M4 evaluation
behavior (run, listing, detail getter) is byte-identical — re-verified
live. M24–M46 history surfaces (evaluations by-checkpoint/dataset/
tokenizer/split/state-kind, comparisons by-checkpoint/dataset/
tokenizer/split/verdict/state-kind, samples by-checkpoint/tokenizer/
strategy, sample-quality by-tokenizer, suite-runs by-suite/
by-checkpoint, workflows by-recipe/by-status, gates by-policy/
by-comparison/by-decision/by-verdict/by-baseline-type, checkpoints
by-run), the M17 dashboard hash and all registries re-verified live
and unchanged (smoke sections I/J/K/L). `b5bc905326b6` still owns
no evaluations (200 `[]` for both statuses).

## 8. Git / Commit State

Two commits on `arena/01a071e9-code-forge`, both pushed:

1. `M47: add evaluation history by truncation` — README.md,
   app/api.py, app/engine.py, app/evaluation.py,
   tests/test_evaluation.py, tests/test_evaluation_api.py, the
   swept test modules, smoke_m47_live.py and this report.
2. `M47: pre-milestone production inventory (m47_pre.sha256)` — the
   baseline manifest (96 lines).

HEAD == remote branch tip == branch-specific FETCH_HEAD;
`ai_model_forge.egg-info/` remains untracked and uncommitted; working
tree clean otherwise. No force-push.

## 9. Next Milestone

End-of-M47 live inspection (production server on port 8768, real
payloads + schemas + repo grep). Remaining closed typed axes are
exhausted or single-valued: `CheckpointRecord.decision` (closed enum
accept/not_best but ALL 3 production checkpoints are accept — one
populated group), `SuiteRun.reused_count` (2→9 / 0→1 — a derived
count, not a classification), workflow stage `executed`/`skipped`
(nested inside stage payloads, not a listing axis),
`RunProvenance.accepted` (both production entries True),
`tokenized_bin_sha256` (14/2 — exactly mirrors the already-covered
split axis), sample `seed`/`temperature` (nullable — None groups
unrepresentable in a path parameter). The strongest remaining
multi-valued persisted typed axis is **`EvaluationRecord.seed`** —
the REQUIRED int the engine persists on every record ("effective
seed used (default derived from config)"), a reproducibility
identity, not a config echo lookup: the listing is filtered by the
record's own persisted value. Live distribution across the 16
evaluations of `4a0a871886ef`: **11 → 4, 1 → 3, 7 → 2, 3 → 2,
7002 → 2, 7101 → 2, 12 → 1** — SEVEN populated groups forming a
TRUE disjoint partition (a required int on every record — no None
case). The M4 listing is deterministic in `(created_at, eval_id)`
ASC. `b5bc905326b6` owns 0 evaluations → `[]` for any seed. No
by-seed route/engine method exists (repo grep: only an unrelated
local variable name in one suite-run test). Runner-up inspected and
set aside: comparisons by `seed` (1 → 5 / 7101 → 2 / 7002 → 1 —
the same semantics on the M5 family, but fewer live groups and the
evaluation family already owns the pattern); also set aside:
suite-runs by `reused_count` (derived count, open-ended, weak
grouping semantics).

Copy-ready M48 prompt:

> M48 — EVALUATION HISTORY BY SEED
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped evaluation records by the persisted effective seed.
> Do NOT jump ahead into aggregation, statistics, seed sweeps,
> variance analysis, re-evaluation, training changes, rollback, HPO,
> RL, Gemini integration, inference, deployment, or autonomous
> improvement. Preserve the minimum-files/minimum-storage
> architecture and the established authoritative-listing →
> persisted-field-filter → response pattern.
>
> STEP 0 — VERIFY STATE AND BASELINE. Inspect the repository and git
> state. If the familiar reset signature appears (HEAD `f86b670` /
> "Add files via upload", files untracked, production data missing,
> venv missing), execute the established recovery runbook: `git
> fetch` + `git reset --hard FETCH_HEAD` on
> `arena/01a071e9-code-forge` (never force-push), restore
> `ai-model-forge-data/` from the workspace zip by extracting ONLY
> its `ai-model-forge-data/` prefix, verify against `m47_pre.sha256`
> (run `sha256sum -c` from `/home/user/ai-model-forge-data/` —
> manifest paths are FORGE_ROOT relative; count the OK lines),
> rebuild the venv (`python3 -m venv ~/.venv` + `pip install -e
> ai-model-forge[dev] pyflakes`), then re-prove the baseline.
> Expected baseline: M47 certified; HEAD == FETCH_HEAD == the M47
> inventory commit; 545 tests passing; OpenAPI 79; production 96
> files / 4,002,745 bytes / 0 tmp; zero SHA drift; production
> evaluation truncation distribution of `4a0a871886ef`: false -> 15
> / true -> 1 (the True record a884bf729ff7). Capture
> `m48_pre.sha256` and record the exact baseline before changing
> anything. Inspect `app/evaluation.py`
> (`list_evaluations()` ~ln 105; the by-* filters; M4 order is
> `(created_at, eval_id)` ASC; the M47
> `list_evaluations_for_truncated` at the end of the filter block),
> `app/engine.py` (evaluation facades), `app/api.py` (the
> `/models/{id}/evaluations/...` family: by-state-kind, then the M47
> by-truncated, then the generic `{eval_id}` route) and
> `app/schemas.py` (`EvaluationRecord.seed: int` — REQUIRED,
> "effective seed used (default derived from config)").
>
> STEP 1 — GROUNDED SELECTION. The selection is already grounded in
> live post-M47 evidence (recorded in `M47_final_report.md` §9):
> `seed` is a persisted REQUIRED int on every EvaluationRecord with
> SEVEN populated live groups (11 -> 4 / 1 -> 3 / 7 -> 2 / 3 -> 2 /
> 7002 -> 2 / 7101 -> 2 / 12 -> 1 among the 16 evaluations of
> `4a0a871886ef`), forming a TRUE disjoint partition with no None
> case; `b5bc905326b6` owns 0 evaluations (natural `[]` for any
> seed); no by-seed route or engine method exists anywhere. If the
> live state contradicts any of this, re-run the end-of-milestone
> inspection and re-ground before implementing.
>
> STEP 2 — IMPLEMENT THE THIN FILTER.
> - Engine (`app/evaluation.py`): `list_evaluations_for_seed(
>   model_id, seed)` placed after `list_evaluations_for_truncated` —
>   resolve the model (existing `_model_exists` FileNotFoundError →
>   404), then return `[e for e in self.list_evaluations(model_id)
>   if e.seed == seed]` — the authoritative M4 listing `(created_at,
>   eval_id)` ASC filtered VERBATIM by the persisted effective seed.
>   NEVER derive the seed from the embedded config dict, dataset
>   identity, splits, tokenizers, state kinds, losses, timestamps or
>   any other field — the persisted record value is the only
>   membership authority. A model with no evaluations at that seed
>   returns []. Read-only, never writes.
> - Facade (`app/engine.py`): thin `list_evaluations_for_seed()`
>   delegation after `list_evaluations_for_truncated`.
> - API (`app/api.py`): `GET /models/{model_id}/evaluations/by-seed/
>   {seed}` registered AFTER by-truncated (M47) and BEFORE the
>   generic `/evaluations/{eval_id}` route. `seed: int` path
>   parameter — the framework parses integers natively: non-integer
>   spellings are rejected 422 pre-handler (schema-level validation,
>   never a silent reinterpretation, and it fires BEFORE the handler
>   even for an unknown model); any integer is type-valid (seed is a
>   VALUE axis with no registry — like split/state-kind/truncated),
>   so an unmatched seed on a valid model is 200 []; unknown model +
>   valid integer -> 404. `response_model=list[EvaluationRecord]`,
>   `tags=["evaluation"]`, verbatim payloads. Add the landing-page
>   bullet, endpoint `<li>` and the evaluations section milestone
>   comment following the M47 wording. No new persistence, cache,
>   index, registry or schema.
> - README: add `### Milestone 48` before `## Quickstart`
>   (semantics, route ordering, value-axis contract with no
>   registry, natural valid-empty for unmatched seeds) and update
>   both test-count mentions.
>
> STEP 3 — TESTS. ~3 engine tests (reuse the M4/M47 evaluation
>   fixtures; cover exact listing parity for multiple seeds, TRUE
>   disjoint partition across ALL seeds present in the listing (the
>   per-seed groups are disjoint and their union is the full
>   listing; no None case), unmatched-seed [] on a valid model, and
>   unknown-model FileNotFoundError) + ~2 API tests (parity/order/
>   determinism for at least two seeds; 404 unknown model + valid
>   int; 422 for non-integer spellings on the known model AND on an
>   unknown model; 200 [] for an unmatched seed; cross-model
>   isolation; the M4 listing/detail + M24/M28/M30/M36/M38/M47
>   by-* regressions; a full OpenAPI verification). No fabricated
>   artifacts. Run the gate ladder: focused modules first, then the
>   FULL suite twice (expect 545 -> ~550), tmp cleanup between runs
>   with 0 live processes first, `compileall`, `pyflakes` clean.
>   Sweep ONLY the genuine OpenAPI path-count assertions `== 79` ->
>   `== 80` (verify 0 stale remain) and add the M48 ladder lines.
>
> STEP 4 — LIVE SMOKE (port 8769). Write `smoke_m48_live.py` with
>   the A–M structure of `smoke_m47_live.py`: baseline audit
>   96/4,002,745/0 + SHA inventory; DISCOVER the seed distribution
>   live from the M4 listing (expect 11 -> 4 / 1 -> 3 / 7 -> 2 /
>   3 -> 2 / 7002 -> 2 / 7101 -> 2 / 12 -> 1; print it; verify the
>   per-seed groups form a TRUE disjoint partition of all 16);
>   exact listing parity per seed with detail-getter parity; an
>   unmatched seed -> 200 []; byte-identical x3 repeats; 404 unknown
>   model + valid int; 422 non-integer spellings on the known model
>   and on an unknown model; both statuses under `b5bc905326b6` ->
>   200 []; the M4/M24/M28/M30/M36/M38 evaluation + M47 by-truncated
>   (false -> 15 / true -> 1) + M26/M29/M31/M37/M39/M44 comparison +
>   M27/M32/M40 sample + M22/M25 suite-run + M42/M35 workflow +
>   M23/M34/M41/M43/M45 gate + M46 checkpoint-by-run (2/1) regression
>   surfaces; dashboard hash
>   `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
>   registries; OpenAPI 80 with the new path before the generic
>   `{eval_id}` route and an integer parameter schema; final storage
>   audit 96/4,002,745/0 with zero drift vs `m48_pre.sha256`. Start
>   the server with `FORGE_ROOT=/home/user/ai-model-forge-data` on
>   0.0.0.0:8769, run the smoke three times (exit 0 each), then stop
>   it, verify 0 live processes, clean `/tmp/forge-tests-*`, and run
>   the final `sha256sum -c m48_pre.sha256` FROM THE FORGE_ROOT
>   (96/96 OK).
>
> STEP 5 — REPORT + COMMITS. Write `M48_final_report.md` with
> EXACTLY 9 sections (Recovery & Baseline / Implementation / Tests /
> OpenAPI / Live Smoke / Storage Integrity / Regression &
> Compatibility / Git / Commit State / Next Milestone). §9 must
> re-ground the M49 selection from LIVE post-M48 inspection
> (persisted field/registry relationship preferred, at least two
> meaningful live groups where practical, thin filter, no schema
> redesign, no new storage, no aggregation/recalculation; careful
> with nested/nullable/free-form/provenance fields; two-sided
> candidates reuse prior either-side semantics; mention the
> runner-up — note that after the seed axis the remaining candidates
> are weak: comparison seed 3 groups, suite-run reused_count, nested
> workflow stage fields, config echoes) and contain the complete
> copy-ready M49 prompt (title, recovery runbook, grounded
> selection, endpoint, engine method, field + semantics, route
> ordering, tests, OpenAPI count, smoke port 8770, storage
> invariants, the 9-section requirement, automatic M49). Commit the
> implementation (README, app/, tests/, smoke, report) as
> `M48: add evaluation history by seed`, then `m48_pre.sha256` as a
> separate inventory commit, push both to
> `arena/01a071e9-code-forge`, verify `HEAD == FETCH_HEAD`
> (branch-specific fetch), and present the report. Do NOT commit
> `ai_model_forge.egg-info/`.
