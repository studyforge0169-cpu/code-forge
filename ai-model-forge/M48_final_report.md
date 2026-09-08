# M48 Final Report — Evaluation History by Seed

## 1. Recovery & Baseline

No reset was needed at M48 start: branch-specific `git fetch` confirmed
`HEAD 1ba9e24` (the M47 inventory commit) == FETCH_HEAD == remote tip,
worktree clean (only `ai_model_forge.egg-info/` untracked), venv
intact, 0 live processes, 0 stale tmp, production
`/home/user/ai-model-forge-data/` at exactly **96 files / 4,002,745
bytes / 0 `.tmp`**. Baseline re-proven before any change: full suite
**545 passed @ 130.60s**, OpenAPI **79 paths**. `m48_pre.sha256`
captured (96 lines, FORGE_ROOT-relative, 96/96 OK).

## 2. Implementation

Four narrow edits inside the established authoritative-listing →
persisted-field-filter → response pattern; no new storage, registry,
index or cache:

1. **`app/evaluation.py`** — `EvaluationEngine.
   list_evaluations_for_seed(model_id, seed)` placed after
   `list_evaluations_for_truncated()`. It is exactly the authoritative
   M4 `list_evaluations(model_id)` (deterministic `(created_at,
   eval_id)` ASC) filtered by `r.seed == seed` — the REQUIRED integer
   persisted on every record at run time (the effective seed used,
   default derived from the config). Matched VERBATIM: never
   recalculated, never normalized, never derived from the embedded
   config dict, request parameters, dataset identity, splits,
   tokenizers, state kinds, losses, ids, timestamps or any other
   field. Unknown model → `FileNotFoundError` (404 at the API).
   Read-only, never writes.
2. **`app/engine.py`** — thin facade
   `ForgeEngine.list_evaluations_for_seed()` after
   `list_evaluations_for_truncated`.
3. **`app/api.py`** — `GET /models/{model_id}/evaluations/by-seed/
   {seed}` registered AFTER by-truncated (M47) and BEFORE the generic
   `/evaluations/{eval_id}` route. `seed: int` path parameter —
   non-integer spellings are rejected 422 by FastAPI/Pydantic schema
   validation pre-handler (integers are never silently reinterpreted,
   and the validation fires before the handler even for an unknown
   model); unknown model + valid integer → 404; ANY integer is
   type-valid (the seed is an open value axis with no registry and no
   artificial range constraint), so an unmatched seed on a valid
   model → 200 `[]`. `response_model=list[EvaluationRecord]`,
   `tags=["evaluation"]`, verbatim payloads. The route docstring
   explicitly states the seed is bookkeeping identity, not a quality
   metric. Landing bullet, endpoint `<li>` and the evaluations
   section milestone comment added.
4. **`README.md`** — `### Milestone 48` section before
   `## Quickstart` (concept, verbatim filtering, open value axis,
   no-quality-metric statement, boundaries) + the evaluation.py layout
   note + test counts 545 → 550 in both places.

## 3. Tests

5 new tests (3 engine + 2 API), reusing the M4/M28/M38/M47 fixtures
(the seed groups are derived from the listing itself, not hard-coded):

- **`tests/test_evaluation.py`** (+3, module 34/34): parity/order/
  verbatim detail parity for every seed present in the listing (≥4
  groups with different cardinalities); TRUE disjoint partition
  (pairwise-disjoint per-seed groups whose union is the full listing;
  no None case — every record's `seed` is an int); unmatched integer
  → `[]`; empty-history model → `[]` for populated and unmatched
  seeds; unknown model → FileNotFoundError.
- **`tests/test_evaluation_api.py`** (+2, module 19/19): three seeds
  with cardinalities 2/2/1 through the REAL engine; exact returned
  IDs per seed; disjoint groups; union == listing; order; detail
  parity; byte-identical repeats; unmatched → 200 `[]`; 404 unknown
  model + valid integer; 422 for three non-integer spellings on the
  known model AND on an unknown model; cross-model isolation both
  directions; M4 listing/detail + M24/M36/M38/M47 by-* regressions;
  full OpenAPI 80 verification (new path once, GET-only, tag
  evaluation, integer parameter schema, `EvaluationRecord` items,
  route order by-truncated < by-seed < generic detail, generic detail
  + M47 routes still present).

No corrections were required — all first-try pass after the sweep.
Full suite ×2 with tmp cleanup between runs (0 live processes
verified first): **550 @ 127.73s** and **550 @ 127.07s**.
`compileall` OK; `pyflakes` 0 findings.

## 4. OpenAPI

Before 79 → after **80**. The sweep updated **32 genuine path-count
assertions** `== 79` → `== 80` across the test modules (20 M47-ladder
sites + 1 short-ladder site + 11 bare assertions; 0 stale remaining).
Verified: the new path appears exactly once, GET-only,
`tags=["evaluation"]`, parameters `model_id` + `seed` with
`{"type": "integer"}` schema, response `{"type": "array", "items":
{"$ref": "#/components/schemas/EvaluationRecord"}}`, route order
by-truncated < by-seed < generic `{eval_id}`; the M47/M46/M45/M44/
M43/M42/M41 routes and the generic detail route still present exactly
once.

## 5. Live Smoke

`smoke_m48_live.py` (committed) against the production `FORGE_ROOT`
on port **8769**, server on `0.0.0.0`, run **three times: 30/30
PASS, exit 0 each, first attempt**. Structure LIVE A–M:

- **A** baseline audit 96/4,002,745/0 + per-file SHA inventory +
  registries + M4–M47 pre-state (16 evaluations, truncation 15/1, 3
  checkpoints, 8 comparisons, 11 gate decisions, 13 workflows, 4
  samples, 10 suite-runs) + dashboard hash + OpenAPI 80 + known pair
  200.
- **B** distribution DISCOVERED live from the M4 listing (not
  assumed): **11 → 4, 1 → 3, 7 → 2, 3 → 2, 7002 → 2, 7101 → 2,
  12 → 1** — SEVEN populated groups, 4+3+2+2+2+2+1 == 16, every
  record's `seed` an int.
- **C** all seven groups return the EXACT locally-filtered M4
  records; verbatim detail-getter parity for all 16 records.
- **D** three GETs raw-byte-identical.
- **E** a populated seed AND the unmatched integer 987654 under
  `b5bc905326b6` → 200 `[]`; the unmatched seed on the KNOWN model →
  200 `[]` (open value axis).
- **F** unknown model + valid integer → 404; non-integer spellings
  ("abc", "1.5", "12x") → 422 on the known model AND on an unknown
  model (schema-level validation fires pre-handler).
- **G** TRUE disjoint partition: pairwise-disjoint groups, union ==
  all 16, every record exactly once, no None case.
- **H** every group preserves the authoritative `(created_at,
  eval_id)` ASC order.
- **I** M4/M24/M28/M30/M36/M38/M47 regressions: listing identical to
  LIVE A, 3/3/3 by-checkpoint, 16/16 by-dataset/tokenizer, by-split
  14/2/0, by-state-kind 9/7, by-truncated 15/1 — all with listing
  parity.
- **J** M22–M46 regressions: comparisons 6/5/1, 8/8, 8/0/0,
  by-verdict 3/3/2, by-state-kind 8/2; samples 4/0/0, 4, by-strategy
  2/2; sample-quality 2; suite-runs 10/10/summary-10; workflows 13,
  9/3/1 by-status, 2 by-recipe; gates 11 with by-decision 7/4,
  by-verdict 4/3/2, by-baseline-type 7/2/2/0, by-comparison 4,
  by-policy 1; M46 checkpoints by-run 2/1.
- **K** M12/M16/M35: recipe registry 7, global m12-live-suite runs 2,
  zero-run recipes 200 `[]`, unknown recipe 404, M16 evaluation ids
  still paired.
- **L** dashboard result_hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  + full output unchanged; tokenizer/policy/probe-suite/recipe/model
  registries unchanged.
- **M** OpenAPI 80 with integer parameter schema + correct route
  placement; by-seed still deterministic at the end; every
  pre-existing file byte-identical; zero new files; totals unchanged.

## 6. Storage Integrity

Final audit after the third smoke run and server shutdown (0 live
processes verified, `/tmp/forge-tests-*` + smoke inventory cleaned):
`sha256sum -c m48_pre.sha256` run from
`/home/user/ai-model-forge-data/` → **96/96 OK, 0 non-OK**; storage
exactly **96 files / 4,002,745 bytes / 0 `.tmp`**. M48 is read-only —
zero production writes, zero SHA drift, zero growth, no seed
registry/index/cache files.

## 7. Regression / Compatibility

No pre-existing route, schema, engine method, storage layout or
registry changed. All 545 pre-existing tests pass unmodified except
the 32 genuine OpenAPI count assertions (79 → 80). M4 evaluation
behavior (run, listing, detail getter) is byte-identical — re-verified
live. M24–M47 history surfaces (evaluations by-checkpoint/dataset/
tokenizer/split/state-kind/truncated, comparisons by-checkpoint/
dataset/tokenizer/split/verdict/state-kind, samples by-checkpoint/
tokenizer/strategy, sample-quality by-tokenizer, suite-runs
by-suite/by-checkpoint, workflows by-recipe/by-status, gates
by-policy/by-comparison/by-decision/by-verdict/by-baseline-type,
checkpoints by-run), the M17 dashboard hash and all registries
re-verified live and unchanged (smoke sections I/J/K/L).
`b5bc905326b6` still owns no evaluations (200 `[]` for populated and
unmatched seeds).

## 8. Git / Commit State

Two commits on `arena/01a071e9-code-forge`, both pushed:

1. `M48: add evaluation history by seed` — README.md, app/api.py,
   app/engine.py, app/evaluation.py, tests/test_evaluation.py,
   tests/test_evaluation_api.py, the swept test modules,
   smoke_m48_live.py and this report.
2. `M48: pre-milestone production inventory (m48_pre.sha256)` — the
   baseline manifest (96 lines).

HEAD == remote branch tip == branch-specific FETCH_HEAD;
`ai_model_forge.egg-info/` remains untracked and uncommitted; working
tree clean otherwise. No force-push.

## 9. Next Milestone

End-of-M48 live inspection (production server on port 8769, real
payloads + schemas + repo grep). Remaining axes are narrowing: sample
`max_new_tokens`/`prompt_token_count` (single-valued: all 8/24),
sample-quality window ints (single-valued), suite-run `reused_count`
(derived count 9/1, not a classification), workflow stage
`executed`/`skipped` (nested inside stage payloads),
`RunProvenance.accepted` (all True), nested `GatePolicy.tolerance`
(floats 0.0001 → 10 / 0.0 → 1 — two groups, but a nested config
echo with float path-parameter messiness). The strongest remaining
multi-valued persisted typed axis is the SAME seed semantics on the
M5 comparison family: **`ComparisonRecord.seed`** — a REQUIRED int
(schema ln 789, "seed" of the identical-probe A/B measurement),
persisted verbatim on every comparison. Live distribution across the
8 comparisons of `4a0a871886ef`: **1 → 5, 7101 → 2, 7002 → 1** —
THREE populated groups forming a TRUE disjoint partition (no None
case; verified live: 0 null seeds). The M5 listing is deterministic
in `(created_at, comparison_id)` ASC order (verified live). No
comparison by-seed exists (the 5 repo hits for "by-seed" are all the
M48 evaluation route just added). `b5bc905326b6` owns 0 comparisons
→ `[]` for any seed. This mirrors M48 exactly on the comparison
family, exactly as M31 followed M30. Runner-up inspected and set
aside: gate decisions by nested `policy.tolerance` (two live groups
10/1 — but nested float config echo; weaker than a top-level
required int, and float path parameters invite parsing ambiguity).

Copy-ready M49 prompt:

> M49 — COMPARISON HISTORY BY SEED
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped comparison records by the persisted effective seed.
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
> its `ai-model-forge-data/` prefix, verify against `m48_pre.sha256`
> (run `sha256sum -c` from `/home/user/ai-model-forge-data/` —
> manifest paths are FORGE_ROOT relative; count the OK lines),
> rebuild the venv (`python3 -m venv ~/.venv` + `pip install -e
> ai-model-forge[dev] pyflakes`), then re-prove the baseline.
> Expected baseline: M48 certified; HEAD == FETCH_HEAD == the M48
> inventory commit; 550 tests passing; OpenAPI 80; production 96
> files / 4,002,745 bytes / 0 tmp; zero SHA drift; production
> evaluation seed distribution of `4a0a871886ef`: 11 -> 4 / 1 -> 3 /
> 7 -> 2 / 3 -> 2 / 7002 -> 2 / 7101 -> 2 / 12 -> 1. Capture
> `m49_pre.sha256` and record the exact baseline before changing
> anything. Inspect `app/comparison.py` (the by-* filter methods;
> M5 order is `(created_at, comparison_id)` ASC), `app/engine.py`
> (comparison facades), `app/api.py` (the `/models/{id}/comparisons/
> ...` family: by-checkpoint, by-dataset, by-tokenizer, by-split,
> by-verdict, by-state-kind — the new route goes after by-state-kind
> and before the generic `{comparison_id}` route) and
> `app/schemas.py` (`ComparisonRecord.seed: int` — REQUIRED, the
> seed of the identical-probe A/B measurement).
>
> STEP 1 — GROUNDED SELECTION. The selection is already grounded in
> live post-M48 evidence (recorded in `M48_final_report.md` §9):
> `seed` is a persisted REQUIRED int on every ComparisonRecord with
> THREE populated live groups (1 -> 5 / 7101 -> 2 / 7002 -> 1 among
> the 8 comparisons of `4a0a871886ef`; 0 null seeds), forming a TRUE
> disjoint partition with no None case; `b5bc905326b6` owns 0
> comparisons (natural `[]` for any seed); no comparison by-seed
> route or engine method exists. If the live state contradicts any
> of this, re-run the end-of-milestone inspection and re-ground
> before implementing.
>
> STEP 2 — IMPLEMENT THE THIN FILTER.
> - Engine (`app/comparison.py`): `list_comparisons_for_seed(
>   model_id, seed)` placed after `list_comparisons_for_state_kind` —
>   resolve the model (existing FileNotFoundError → 404), then
>   return `[c for c in self.list_comparisons(model_id) if c.seed ==
>   seed]` — the authoritative M5 listing `(created_at,
>   comparison_id)` ASC filtered VERBATIM by the persisted record
>   value. NEVER derive the seed from the config dict, state
>   payloads, verdicts, loss deltas, ids, timestamps or any other
>   field. A model with no comparisons at that seed returns [].
>   Read-only, never writes.
> - Facade (`app/engine.py`): thin `list_comparisons_for_seed()`
>   delegation after `list_comparisons_for_state_kind`.
> - API (`app/api.py`): `GET /models/{model_id}/comparisons/by-seed/
>   {seed}` registered AFTER by-state-kind (M44) and BEFORE the
>   generic `/comparisons/{comparison_id}` route. `seed: int` path
>   parameter — non-integer spellings are rejected 422 pre-handler
>   (schema-level validation, never a silent reinterpretation, and
>   it fires BEFORE the handler even for an unknown model); any
>   integer is type-valid (an open value axis with no registry —
>   like the M48 evaluation by-seed), so an unmatched seed on a
>   valid model is 200 []; unknown model + valid integer -> 404.
>   `response_model=list[ComparisonRecord]`, `tags=["comparison"]`,
>   verbatim payloads. Add the landing-page bullet, endpoint `<li>`
>   and the comparisons section milestone comment following the M48
>   wording. No new persistence, cache, index, registry or schema.
> - README: add `### Milestone 49` before `## Quickstart`
>   (semantics, route ordering, value-axis contract with no
>   registry, natural valid-empty for unmatched seeds) and update
>   both test-count mentions.
>
> STEP 3 — TESTS. ~3 engine tests (reuse the M5 comparison fixtures;
>   cover exact listing parity for multiple seeds with different
>   cardinalities, TRUE disjoint partition across ALL seeds present
>   in the listing (no None case), unmatched-seed [] on a valid
>   model, empty-model [], and unknown-model FileNotFoundError) +
>   ~2 API tests (parity/order/determinism for at least two seeds;
>   404 unknown model + valid int; 422 for non-integer spellings on
>   the known model AND on an unknown model; 200 [] for an unmatched
>   seed; cross-model isolation; the M5 listing/detail + M26/M29/
>   M31/M37/M39/M44 by-* regressions; a full OpenAPI verification).
>   No fabricated artifacts. Run the gate ladder: focused modules
>   first, then the FULL suite twice (expect 550 -> ~555), tmp
>   cleanup between runs with 0 live processes first, `compileall`,
>   `pyflakes` clean. Sweep ONLY the genuine OpenAPI path-count
>   assertions `== 80` -> `== 81` (verify 0 stale remain) and add
>   the M49 ladder lines.
>
> STEP 4 — LIVE SMOKE (port 8770). Write `smoke_m49_live.py` with
>   the A–M structure of `smoke_m48_live.py`: baseline audit
>   96/4,002,745/0 + SHA inventory; DISCOVER the comparison seed
>   distribution live from the M5 listing (expect 1 -> 5 / 7101 ->
>   2 / 7002 -> 1; print it; verify the per-seed groups form a TRUE
>   disjoint partition of all 8); exact listing parity per seed with
>   detail-getter parity; an unmatched seed -> 200 []; byte-identical
>   x3 repeats; 404 unknown model + valid int; 422 non-integer
>   spellings on the known model and on an unknown model; any seed
>   under `b5bc905326b6` -> 200 []; the M5/M26/M29/M31/M37/M39/M44
>   comparison + M48 evaluation-by-seed (11 -> 4 / 1 -> 3 / 7 -> 2 /
>   3 -> 2 / 7002 -> 2 / 7101 -> 2 / 12 -> 1) + M47 by-truncated
>   (false -> 15 / true -> 1) + M24/M28/M30/M36/M38 evaluation +
>   M27/M32/M40 sample + M22/M25 suite-run + M42/M35 workflow +
>   M23/M34/M41/M43/M45 gate + M46 checkpoint-by-run (2/1) regression
>   surfaces; dashboard hash
>   `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
>   registries; OpenAPI 81 with the new path before the generic
>   `{comparison_id}` route and an integer parameter schema; final
>   storage audit 96/4,002,745/0 with zero drift vs `m49_pre.sha256`.
>   Start the server with `FORGE_ROOT=/home/user/ai-model-forge-data`
>   on 0.0.0.0:8770, run the smoke three times (exit 0 each), then
>   stop it, verify 0 live processes, clean `/tmp/forge-tests-*`, and
>   run the final `sha256sum -c m49_pre.sha256` FROM THE FORGE_ROOT
>   (96/96 OK).
>
> STEP 5 — REPORT + COMMITS. Write `M49_final_report.md` with
> EXACTLY 9 sections (Recovery & Baseline / Implementation / Tests /
> OpenAPI / Live Smoke / Storage Integrity / Regression &
> Compatibility / Git / Commit State / Next Milestone). §9 must
> re-ground the M50 selection from LIVE post-M49 inspection — after
> the comparison seed axis, explicitly assess whether simple history
> axes are exhausted (remaining candidates are weak: nested
> GatePolicy.tolerance floats 10/1, suite-run reused_count 9/1
> derived, workflow nested stage booleans, config echoes; state that
> fact honestly rather than manufacturing a weak milestone) and, if a
> defensible axis remains, contain the complete copy-ready M50 prompt
> (title, recovery runbook, grounded selection, endpoint, engine
> method, field + semantics, route ordering, tests, OpenAPI count,
> smoke port 8771, storage invariants, the 9-section requirement,
> automatic M50). Commit the implementation (README, app/, tests/,
> smoke, report) as `M49: add comparison history by seed`, then
> `m49_pre.sha256` as a separate inventory commit, push both to
> `arena/01a071e9-code-forge`, verify `HEAD == FETCH_HEAD`
> (branch-specific fetch), and present the report. Do NOT commit
> `ai_model_forge.egg-info/`.
