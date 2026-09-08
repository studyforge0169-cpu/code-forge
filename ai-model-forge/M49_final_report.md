# M49 Final Report — Comparison History by Seed

## 1. Recovery & Baseline

No reset was needed at M49 start: branch-specific `git fetch` confirmed
`HEAD 399f793` (the M48 inventory commit) == FETCH_HEAD == remote tip,
worktree clean (only `ai_model_forge.egg-info/` untracked), venv
intact, 0 live processes, 0 stale tmp, production
`/home/user/ai-model-forge-data/` at exactly **96 files / 4,002,745
bytes / 0 `.tmp`**. Baseline re-proven before any change: full suite
**550 passed @ 135.23s**, OpenAPI **80 paths**. `m49_pre.sha256`
captured (96 lines, FORGE_ROOT-relative, 96/96 OK).

## 2. Implementation

Four narrow edits inside the established authoritative-listing →
persisted-field-filter → response pattern; no new storage, registry,
index or cache:

1. **`app/comparison.py`** — `ComparisonEngine.
   list_comparisons_for_seed(model_id, seed)` placed after
   `list_comparisons_for_state_kind()`. It is exactly the
   authoritative M5 `list_comparisons(model_id)` (deterministic
   `(created_at, comparison_id)` ASC) filtered by `r.seed == seed` —
   the REQUIRED integer persisted on every record at run time (the
   seed of the identical-probe A/B measurement). Matched VERBATIM:
   never recalculated, never normalized, never derived from the
   comparison configuration, either side's evaluation, state payloads,
   verdicts, loss deltas, ids, timestamps or any other field. Unknown
   model → `FileNotFoundError` (404 at the API). Read-only, never
   writes.
2. **`app/engine.py`** — thin facade
   `ForgeEngine.list_comparisons_for_seed()` after
   `list_comparisons_for_state_kind`.
3. **`app/api.py`** — `GET /models/{model_id}/comparisons/by-seed/
   {seed}` registered AFTER by-state-kind (M44) and BEFORE the
   generic `/comparisons/{comparison_id}` route. `seed: int` path
   parameter — non-integer spellings are rejected 422 by
   FastAPI/Pydantic schema validation pre-handler (integers are never
   silently reinterpreted, and the validation fires before the
   handler even for an unknown model); unknown model + valid integer
   → 404; ANY integer is type-valid (an open value axis with no
   registry and no artificial range constraint), so an unmatched seed
   on a valid model → 200 `[]`. `response_model=list[ComparisonRecord]`,
   `tags=["comparison"]`, verbatim payloads. The route docstring
   explicitly states the seed is bookkeeping identity, not a quality
   metric. Landing bullet, endpoint `<li>` and the comparisons
   section milestone comment added.
4. **`README.md`** — `### Milestone 49` section before
   `## Quickstart` + the comparison.py layout note + test counts
   550 → 555 in both places.

One correction during docs: the endpoint-`<li>` anchor initially
failed because the actual M44 li reads "EITHER side" (the grep
context line had shown "EITHER SIDE"); the anchor was re-derived from
the raw line before splicing. No code changes were needed.

## 3. Tests

5 new tests (3 engine + 2 API), reusing the M26/M39 comparison
fixtures (seed groups derived from the listing, never hard-coded):

- **`tests/test_comparison.py`** (+3, module 43/43): `_m49_env`
  cached fixture on top of `_m39_env`/`_m26_env` adding a shared seed
  4900 ×2 (an A=B and a cross-checkpoint pair) + seed 4901 ×1 through
  the REAL engine; parity/order/verbatim detail parity for every seed
  in the listing (≥4 groups); TRUE disjoint partition (pairwise
  disjoint, union == full listing, no None case); unmatched integer →
  `[]`; zero-comparison model → `[]`; unknown model →
  FileNotFoundError.
- **`tests/test_comparison_api.py`** (+2, module 19/19): seeds
  4900 ×2 / 4901 ×1 through the REAL engine; exact returned IDs;
  disjoint groups; union == listing; order; detail parity;
  byte-identical repeats; unmatched → 200 `[]`; 404 unknown model +
  valid integer; 422 for three non-integer spellings on the known
  model AND on an unknown model; cross-model isolation both
  directions; M5 listing/detail + M26/M39/M44 by-* regressions; full
  OpenAPI 81 verification (new path once, GET-only, tag comparison,
  integer parameter schema, `ComparisonRecord` items, route order
  by-state-kind < by-seed < generic detail, generic + M48 routes
  still present).

All first-try pass after the sweep. Full suite ×2 with tmp cleanup
between runs (0 live processes verified first): **555 @ 141.36s** and
**555 @ 153.75s**. `compileall` OK; `pyflakes` 0 findings.

## 4. OpenAPI

Before 80 → after **81**. The sweep updated **33 genuine path-count
assertions** `== 80` → `== 81` across the test modules (20 M48-ladder
sites + 1 short-ladder site + 12 bare assertions; 0 stale remaining).
Verified: the new path appears exactly once, GET-only,
`tags=["comparison"]`, parameters `model_id` + `seed` with
`{"type": "integer"}` schema, response `{"type": "array", "items":
{"$ref": "#/components/schemas/ComparisonRecord"}}`, route order
by-state-kind < by-seed < generic `{comparison_id}`; the
M48/M47/M46/M45/M44/M43/M42/M41 routes and the generic detail route
still present exactly once.

## 5. Live Smoke

`smoke_m49_live.py` (committed) against the production `FORGE_ROOT`
on port **8770**, server on `0.0.0.0`, run **three times: 30/30
PASS, exit 0 each, first attempt**. Structure LIVE A–M:

- **A** baseline audit 96/4,002,745/0 + per-file SHA inventory +
  registries + M4–M48 pre-state (8 comparisons, 16 evaluations, 3
  checkpoints, 11 gate decisions, 13 workflows, 4 samples, 10
  suite-runs) + dashboard hash + OpenAPI 81 + known pair 200.
- **B** distribution DISCOVERED live from the M5 listing (not
  assumed): **1 → 5, 7101 → 2, 7002 → 1** — THREE populated groups,
  5+2+1 == 8, **0 null seeds**, every record's `seed` an int.
- **C** all three groups return the EXACT locally-filtered M5
  records; verbatim detail-getter parity for all 8 records.
- **D** three GETs raw-byte-identical.
- **E** a populated seed AND the unmatched integer 987654 under
  `b5bc905326b6` → 200 `[]`; the unmatched seed on the KNOWN model →
  200 `[]` (open value axis).
- **F** unknown model + valid integer → 404; non-integer spellings
  ("abc", "1.5", "12x") → 422 on the known model AND on an unknown
  model (schema-level validation fires pre-handler).
- **G** TRUE disjoint partition: 5 + 2 + 1 == 8, pairwise disjoint,
  union == full listing, every record exactly once, no None case.
- **H** every group preserves the authoritative `(created_at,
  comparison_id)` ASC order (no new sort key).
- **I** M5/M26/M29/M31/M37/M39/M44 regressions: listing identical to
  LIVE A, 6/5/1 by-checkpoint, 8/8 by-dataset/tokenizer, by-split
  8/0/0, by-verdict 3/3/2, by-state-kind 8/2 either-side — all with
  listing parity.
- **J** M22–M48 regressions: evaluations 3/3/3, 16/16, by-split
  14/2/0, by-state-kind 9/7, by-truncated 15/1, by-seed
  4/3/2/2/2/2/1; samples 4/0/0, 4, by-strategy 2/2; sample-quality 2;
  suite-runs 10/10/summary-10; workflows 13, 9/3/1 by-status, 2
  by-recipe; gates 11 with by-decision 7/4, by-verdict 4/3/2,
  by-baseline-type 7/2/2/0, by-comparison 4, by-policy 1; M46
  checkpoints by-run 2/1.
- **K** M12/M16/M35: recipe registry 7, global m12-live-suite runs 2,
  zero-run recipes 200 `[]`, unknown recipe 404, M16 evaluation ids
  still paired.
- **L** dashboard result_hash
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  + full output unchanged; tokenizer/policy/probe-suite/recipe/model
  registries unchanged.
- **M** OpenAPI 81 with integer parameter schema + correct route
  placement; by-seed still deterministic at the end; every
  pre-existing file byte-identical; zero new files; totals unchanged.

## 6. Storage Integrity

Final audit after the third smoke run and server shutdown (0 live
processes verified, `/tmp/forge-tests-*` + smoke inventory cleaned):
`sha256sum -c m49_pre.sha256` run from
`/home/user/ai-model-forge-data/` → **96/96 OK, 0 non-OK**; storage
exactly **96 files / 4,002,745 bytes / 0 `.tmp`**. M49 is read-only —
zero production writes, zero SHA drift, zero growth, no seed
registry/index/cache files.

## 7. Regression / Compatibility

No pre-existing route, schema, engine method, storage layout or
registry changed. All 550 pre-existing tests pass unmodified except
the 33 genuine OpenAPI count assertions (80 → 81). M5 comparison
behavior (run, listing, detail getter) is byte-identical — re-verified
live. M24–M48 history surfaces (evaluations by-checkpoint/dataset/
tokenizer/split/state-kind/truncated/seed, comparisons
by-checkpoint/dataset/tokenizer/split/verdict/state-kind, samples
by-checkpoint/tokenizer/strategy, sample-quality by-tokenizer,
suite-runs by-suite/by-checkpoint, workflows by-recipe/by-status,
gates by-policy/by-comparison/by-decision/by-verdict/
by-baseline-type, checkpoints by-run), the M17 dashboard hash and all
registries re-verified live and unchanged (smoke sections I/J/K/L).
`b5bc905326b6` still owns no comparisons (200 `[]` for populated and
unmatched seeds).

## 8. Git / Commit State

Two commits on `arena/01a071e9-code-forge`, both pushed:

1. `M49: add comparison history by seed` — README.md, app/api.py,
   app/engine.py, app/comparison.py, tests/test_comparison.py,
   tests/test_comparison_api.py, the swept test modules,
   smoke_m49_live.py and this report.
2. `M49: pre-milestone production inventory (m49_pre.sha256)` — the
   baseline manifest (96 lines).

HEAD == remote branch tip == branch-specific FETCH_HEAD;
`ai_model_forge.egg-info/` remains untracked and uncommitted; working
tree clean otherwise. No force-push.

## 9. Next Milestone

End-of-M49 live inspection (production server on port 8770, real
payloads + schemas + repo grep), with candidates explicitly compared
because the simple typed axes are narrowing. Rejected outright:
comparison `batch_size`/`max_seq_len` (single-valued: all 8/32),
suite-run `probe_count`/`suite_probes_hash` (single-valued: all 2 /
one hash), workflow `hint`/`failed_stage_id` (nullable free-form —
None majorities), workflow `composition` (nullable nested recipe
chain), sample-quality window ints (single-valued), evaluation
`tokenized_bin_sha256` (duplicates the covered split axis),
`RunProvenance.accepted` (all True, and nested in the model manifest
rather than a listing), evaluation `records_covered` (nullable).
Compared head-to-head, the two remaining multi-group candidates are:

- **`SuiteRunRecord.reused_count`** — REQUIRED int persisted on every
  suite-run record at run time ("probes satisfied by pre-existing
  evidence", schema comment: "execution bookkeeping only (never a
  score)"). Live distribution across the 10 production suite runs of
  `4a0a871886ef`: **2 → 9, 0 → 1** — TWO populated groups forming a
  TRUE disjoint partition (no None case). The M10 listing is
  deterministic in `(created_at, suite_run_id)` ASC (verified live).
  `b5bc905326b6` owns 0 suite-runs → `[]`. No by-reused route exists
  (grep 0 hits). Int axis (clean path contract), engine-recorded
  bookkeeping exactly like M47's `truncated`.
- **`ComparisonRecord.tolerance`** — REQUIRED float, live groups
  0.0001 → 7 / 0.0 → 1. Set aside: a float path axis invites spelling
  ambiguity (0.0001 vs 1e-4), and it is a direct config echo; the
  same objection applies to the nested `GatePolicy.tolerance`
  (10/1).

**Honest exhaustion statement:** after a by-reused-count milestone,
the simple persisted-identity history axes ARE exhausted — every
remaining field is single-valued, nullable free-form, a nested config
echo, a duplicated axis, or a derived metric that would turn
bookkeeping into quality judgment. M50 should therefore be treated as
the LAST simple grouping milestone; its §9 must evaluate an
architectural transition on live evidence rather than manufacture a
weak M51 grouping.

Copy-ready M50 prompt:

> M50 — SUITE-RUN HISTORY BY REUSED COUNT
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped suite-run records by the persisted reuse count. This
> is expected to be the LAST simple grouping milestone — treat scope
> discipline accordingly. Do NOT jump ahead into aggregation, cache
> statistics, coverage scoring, re-runs, training changes, rollback,
> HPO, RL, Gemini integration, inference, deployment, or autonomous
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
> its `ai-model-forge-data/` prefix, verify against `m49_pre.sha256`
> (run `sha256sum -c` from `/home/user/ai-model-forge-data/` —
> manifest paths are FORGE_ROOT relative; count the OK lines),
> rebuild the venv (`python3 -m venv ~/.venv` + `pip install -e
> ai-model-forge[dev] pyflakes`), then re-prove the baseline.
> Expected baseline: M49 certified; HEAD == FETCH_HEAD == the M49
> inventory commit; 555 tests passing; OpenAPI 81; production 96
> files / 4,002,745 bytes / 0 tmp; zero SHA drift; production
> comparison seed distribution of `4a0a871886ef`: 1 -> 5 / 7101 ->
> 2 / 7002 -> 1. Capture `m50_pre.sha256` and record the exact
> baseline before changing anything. Inspect `app/suite_runs.py`
> (the M21 `list_suite_runs_for_suite` / M25
> `list_suite_runs_for_checkpoint` filters; listing order is
> `(created_at, suite_run_id)` ASC), `app/engine.py` (suite-run
> facades), `app/api.py` (the `/models/{id}/suite-runs/...` family:
> by-suite ln ~2002, by-suite/summary ~2026, by-checkpoint ~2047,
> the generic `{suite_run_id}` detail ~2073 — the new route goes
> AFTER by-checkpoint and BEFORE the generic detail route) and
> `app/schemas.py` (`SuiteRunRecord.reused_count: int` — REQUIRED,
> "probes satisfied by pre-existing evidence ... execution
> bookkeeping only (never a score)").
>
> STEP 1 — GROUNDED SELECTION. The selection is already grounded in
> live post-M49 evidence (recorded in `M49_final_report.md` §9):
> `reused_count` is a persisted REQUIRED int on every SuiteRunRecord
> with TWO populated live groups (2 -> 9 / 0 -> 1 among the 10
> suite runs of `4a0a871886ef`), forming a TRUE disjoint partition
> with no None case; `b5bc905326b6` owns 0 suite-runs (natural `[]`
> for any value); no by-reused route or engine method exists. If the
> live state contradicts any of this, re-run the end-of-milestone
> inspection and re-ground before implementing.
>
> STEP 2 — IMPLEMENT THE THIN FILTER.
> - Engine (`app/suite_runs.py`): `list_suite_runs_for_reused_count(
>   model_id, reused_count)` placed after
>   `list_suite_runs_for_checkpoint` — resolve the model (existing
>   FileNotFoundError → 404), then return `[r for r in
>   self.list_suite_runs(model_id) if r.reused_count ==
>   reused_count]` — the authoritative M10/M21 listing (created_at,
>   suite_run_id) ASC filtered VERBATIM by the persisted record
>   value. NEVER derive the count from `results`, `completed_count`,
>   `failed_count`, `probe_count`, timestamps or any other field;
>   NEVER recompute cache hits. A model with no suite runs at that
>   count returns []. Read-only, never writes.
> - Facade (`app/engine.py`): thin
>   `list_suite_runs_for_reused_count()` delegation after
>   `list_suite_runs_for_checkpoint`.
> - API (`app/api.py`): `GET /models/{model_id}/suite-runs/by-reused/
>   {reused_count}` registered AFTER by-checkpoint (M25) and BEFORE
>   the generic `/suite-runs/{suite_run_id}` route.
>   `reused_count: int` path parameter — non-integer spellings are
>   rejected 422 pre-handler (schema-level validation, never a
>   silent reinterpretation, and it fires BEFORE the handler even
>   for an unknown model); any integer is type-valid (an open value
>   axis with no registry — like the M48/M49 by-seed routes), so an
>   unmatched count on a valid model is 200 []; unknown model +
>   valid integer -> 404. `response_model=list[SuiteRunRecord]`,
>   `tags=["suites"]` (match the actual family tag), verbatim
>   payloads. The docstring must state the count is execution
>   bookkeeping, NEVER a score or quality metric. Add the
>   landing-page bullet, endpoint `<li>` and the suite-runs section
>   milestone comment following the M49 wording. No new persistence,
>   cache, index, registry or schema.
> - README: add `### Milestone 50` before `## Quickstart`
>   (semantics, route ordering, value-axis contract with no
>   registry, natural valid-empty for unmatched counts, the
>   bookkeeping-not-a-score statement) and update both test-count
>   mentions.
>
> STEP 3 — TESTS. ~3 engine tests (reuse the M10/M21/M25 suite-run
>   fixtures in `tests/test_suite_runs.py`; cover exact listing
>   parity for the populated values (a fresh run -> 0, an identical
>   re-run -> probe_count reuses — check the existing reuse tests
>   for the real construction), TRUE disjoint partition across ALL
>   values present in the listing (no None case), unmatched integer
>   [] on a valid model, empty-model [], and unknown-model
>   FileNotFoundError) + ~2 API tests (parity/order/determinism;
>   404 unknown model + valid int; 422 for non-integer spellings on
>   the known model AND on an unknown model; 200 [] for an unmatched
>   count; cross-model isolation; the M10/M21/M22/M25 listing +
>   detail + summary regressions; a full OpenAPI verification). No
>   fabricated artifacts. Run the gate ladder: focused modules
>   first, then the FULL suite twice (expect 555 -> ~560), tmp
>   cleanup between runs with 0 live processes first, `compileall`,
>   `pyflakes` clean. Sweep ONLY the genuine OpenAPI path-count
>   assertions `== 81` -> `== 82` (verify 0 stale remain) and add
>   the M50 ladder lines.
>
> STEP 4 — LIVE SMOKE (port 8771). Write `smoke_m50_live.py` with
>   the A–M structure of `smoke_m49_live.py`: baseline audit
>   96/4,002,745/0 + SHA inventory; DISCOVER the reused_count
>   distribution live from the M10 listing (expect 2 -> 9 / 0 -> 1;
>   print it; verify the groups form a TRUE disjoint partition of
>   all 10); exact listing parity per value with detail-getter
>   parity; an unmatched count -> 200 []; byte-identical x3 repeats;
>   404 unknown model + valid int; 422 non-integer spellings on the
>   known model and on an unknown model; any count under
>   `b5bc905326b6` -> 200 []; the M10/M21/M22/M25 suite-run + M48
>   evaluation-by-seed + M49 comparison-by-seed (1 -> 5 / 7101 -> 2
>   / 7002 -> 1) + M47 by-truncated + M24/M28/M30/M36/M38 evaluation
>   + M26/M29/M31/M37/M39/M44 comparison + M27/M32/M40 sample +
>   M42/M35 workflow + M23/M34/M41/M43/M45 gate + M46
>   checkpoint-by-run regression surfaces; dashboard hash
>   `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
>   registries; OpenAPI 82 with the new path before the generic
>   `{suite_run_id}` route and an integer parameter schema; final
>   storage audit 96/4,002,745/0 with zero drift vs `m50_pre.sha256`.
>   Start the server with `FORGE_ROOT=/home/user/ai-model-forge-data`
>   on 0.0.0.0:8771, run the smoke three times (exit 0 each), then
>   stop it, verify 0 live processes, clean `/tmp/forge-tests-*`, and
>   run the final `sha256sum -c m50_pre.sha256` FROM THE FORGE_ROOT
>   (96/96 OK).
>
> STEP 5 — REPORT + COMMITS. Write `M50_final_report.md` with
> EXACTLY 9 sections (Recovery & Baseline / Implementation / Tests /
> OpenAPI / Live Smoke / Storage Integrity / Regression &
> Compatibility / Git / Commit State / Next Milestone). §9 is
> CRITICAL this time: the simple persisted-identity history axes are
> exhausted after M50 (per `M49_final_report.md` §9 — remaining
> fields are single-valued, nullable free-form, nested config echoes,
> duplicated axes or derived metrics). Do NOT manufacture a weak
> grouping endpoint. Instead, re-verify the exhaustion claim against
> LIVE post-M50 state and then evaluate, on live evidence alone, the
> strongest ARCHITECTURAL next step the codebase actually supports —
> candidates to inspect honestly include: consolidating the 28
> by-* history routes behind a documented query-parameter surface (a
> REFACTOR with byte-identical responses, zero new semantics), a
> read-only cross-model registry/index view (M1-M2/M9/M12/M14 data
> that already exists), or an explicitly-scoped operational feature
> from the project README's roadmap that the production artifacts
> justify — and if none is genuinely justified by evidence, say so
> and recommend consolidating documentation instead. Whatever is
> selected must be grounded in executed live inspection, not
> speculation, and §9 must contain the complete copy-ready M51 prompt
> (title, recovery runbook, grounded selection with evidence,
> endpoint/semantics or refactor contract, tests, OpenAPI count if
> applicable, smoke port 8772, storage invariants, the 9-section
> requirement, automatic M51). Commit the implementation (README,
> app/, tests/, smoke, report) as `M50: add suite-run history by
> reused count`, then `m50_pre.sha256` as a separate inventory
> commit, push both to `arena/01a071e9-code-forge`, verify
> `HEAD == FETCH_HEAD` (branch-specific fetch), and present the
> report. Do NOT commit `ai_model_forge.egg-info/`.
