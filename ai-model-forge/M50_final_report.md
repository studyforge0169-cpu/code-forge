# M50 Final Report — Suite-Run History by Reused Count

## 1. Baseline

No reset was needed at M50 start: branch-specific `git fetch` confirmed
`HEAD 845f795` (the M49 inventory commit) == FETCH_HEAD == remote tip,
worktree clean (only `ai_model_forge.egg-info/` untracked), venv
intact, 0 live processes, 0 stale tmp, production
`/home/user/ai-model-forge-data/` at exactly **96 files / 4,002,745
bytes / 0 `.tmp`**. Baseline re-proven before any change: full suite
**555 passed @ 130.59s**, OpenAPI **81 paths**. `m50_pre.sha256`
captured (96 lines, FORGE_ROOT-relative, 96/96 OK).

## 2. Grounding

`SuiteRunRecord.reused_count` inspected in `app/schemas.py` (ln 1890):
**REQUIRED `int`** (no default, no Optional) — "probes satisfied by
pre-existing evidence", sitting in the execution-bookkeeping block
(`probe_count` / `completed_count` / `reused_count` / `failed_count`)
whose own comments state "execution bookkeeping only (never a score)".
The authoritative listing is `SuiteRunEngine.list_suite_runs(model_id)`
(ln 198), deterministic `(created_at, suite_run_id)` ASC; the route
family is `/models/{id}/suite-runs/...` with tag `suite-runs`
(by-suite M21, by-suite/summary M22, by-checkpoint M25, generic
detail). **Independently verified against the persisted production
manifests BEFORE implementation** (reading the 10
`suite-runs/*/manifest.json` files directly, not via the API):
distribution **{2: 9, 0: 1}**, 10 total, 0 null values, all owned by
`4a0a871886ef`; `b5bc905326b6` owns none. The field shape matched the
expected contract exactly — no discrepancy.

## 3. Implementation

Four narrow edits inside the established authoritative-listing →
persisted-field-filter → response pattern; no new storage, registry,
index or cache:

1. **`app/suite_runs.py`** — `SuiteRunEngine.
   list_suite_runs_for_reused_count(model_id, reused_count)` placed
   after `list_suite_runs_for_checkpoint()`: exactly the authoritative
   M10 `list_suite_runs(model_id)` filtered by `r.reused_count ==
   reused_count` — matched VERBATIM, never recalculated, never derived
   from probe outcomes, completed_count, failed_count, probe_count,
   suite size, status, timestamps, artifact ids, evaluation or
   comparison records, configuration or any other field. Unknown model
   → `FileNotFoundError` (404 at the API). Read-only, never writes.
2. **`app/engine.py`** — thin facade
   `ForgeEngine.list_suite_runs_for_reused_count()` after
   `list_suite_runs_for_checkpoint`.
3. **`app/api.py`** — `GET /models/{model_id}/suite-runs/by-reused/
   {reused_count}` registered AFTER by-checkpoint (M25) and BEFORE the
   generic `/suite-runs/{suite_run_id}` route (verified route indices
   57 < 58 < 59). `reused_count: int` path parameter — non-integer
   spellings rejected 422 by FastAPI/Pydantic pre-handler (integers
   never silently reinterpreted; fires before the handler even for an
   unknown model); unknown model + valid integer → 404; ANY integer is
   type-valid (open value axis), so an unmatched count → 200 `[]`.
   `response_model=list[SuiteRunRecord]`, `tags=["suite-runs"]`
   (the actual family tag, confirmed by inspection), verbatim
   payloads. The route docstring states the count is EXECUTION
   BOOKKEEPING, never a score. Landing bullet, endpoint `<li>` and
   the suite-runs section milestone comment added.
4. **`README.md`** — `### Milestone 50` section before
   `## Quickstart` (noting this closes the simple grouping ladder) +
   the suite_runs.py layout note + test counts 555 → 560 in both
   places.

## 4. Tests

5 new tests (3 engine + 2 API) in `tests/test_suite_runs.py`,
reusing the module's `env`/`_http_env` fixtures; the reuse groups are
constructed through the REAL engine (first run creates evidence →
`reused_count` 0; identical second run reuses → `reused_count` 2):

- **Engine** (+3, module 37/37 after sweep): `_m50_env` cached
  fixture; parity/order/verbatim detail parity for every value in the
  listing; TRUE disjoint partition (pairwise disjoint, union == full
  listing, no None case); unmatched 987654 → `[]`; `fresh_model` with
  no suite runs → `[]` for populated and unmatched counts; unknown
  model → FileNotFoundError for both probes.
- **API** (+2): 0/2 groups with exact IDs, disjoint, union ==
  listing, order, detail parity, byte-identical repeats; unmatched →
  200 `[]`; 404 unknown model + valid int; 422 for "abc"/"1.5"/"12x"
  on the known model AND an unknown model; cross-model isolation both
  directions; no-suite-runs model → 200 `[]`; M10/M21/M22/M25
  regressions (listing, detail, by-suite, summary total, by-checkpoint
  — all with listing parity); full OpenAPI 82 verification (new path
  once, GET-only, tag suite-runs, integer parameter schema,
  `SuiteRunRecord` items, route order by-checkpoint < by-reused <
  generic detail, generic + M49 routes still present).

Full suite ×2 with tmp cleanup between runs (0 live processes
verified first): **560 @ 135.34s** and **560 @ 146.99s**.
`compileall` OK; `pyflakes` 0 findings.

## 5. OpenAPI

Before 81 → after **82**. The sweep updated **34 genuine path-count
assertions** `== 81` → `== 82` across the test modules (21 M49-ladder
sites + 1 short-ladder site + 12 bare assertions; 0 stale remaining).
Verified: the new path appears exactly once, GET-only,
`tags=["suite-runs"]`, parameters `model_id` + `reused_count` with
`{"type": "integer"}` schema, response `{"type": "array", "items":
{"$ref": "#/components/schemas/SuiteRunRecord"}}`, route order
by-checkpoint < by-reused < generic `{suite_run_id}`; the
M49/M48/M47/M46/M45/M44/M43/M42/M41 routes and the generic detail
route still present exactly once.

## 6. Live Smoke

`smoke_m50_live.py` (committed) against the production `FORGE_ROOT`
on port **8771**, server on `0.0.0.0`, run **three times: 30/30
PASS, exit 0 each** (after one correction, below). Structure LIVE
A–M: baseline audit 96/4,002,745/0 + SHA inventory + registries +
M3–M49 pre-state + dashboard hash + OpenAPI 82 + known pair 200;
distribution DISCOVERED live from BOTH the M10 listing AND the
persisted manifests (**2 → 9, 0 → 1**, two groups, 0 null values);
exact listing parity per group + verbatim detail parity for all 10
records; byte-identical ×3 repeats; unmatched 987654 → 200 `[]` on
both models; 404 unknown model + valid integer; 422 for
"abc"/"1.5"/"12x" on the known model AND an unknown model; TRUE
disjoint partition 9 + 1 == 10; `(created_at, suite_run_id)` order
preserved; M10/M21/M22/M25 suite-run regressions; the full M22–M49
regression battery (evaluations 3/3/3 + 16/16 + 14/2/0 + 9/7 + 15/1 +
seed 4/3/2/2/2/2/1; comparisons 6/5/1 + 8/8 + 8/0/0 + 3/3/2 + 8/2 +
seed 5/2/1; samples 4/0/0 + 4 + 2/2; sample-quality 2; workflows 13 +
9/3/1 + 2 by-recipe; gates 7/4 + 4/3/2 + 7/2/2/0 + 4 + 1; checkpoints
by-run 2/1); M12/M16/M35 surfaces; dashboard + registries; OpenAPI 82
placement; final storage audit zero drift.

**Correction (honest reporting):** the first smoke run failed B1
(29/30) — the manifest cross-check globbed
`models/{MODEL}/suite-runs/*/manifest.json`, but suite-run manifests
live at the STORAGE ROOT (`suite-runs/<suite_run_id>/manifest.json`,
model-scoped by the record's `model_id` field, per
`SuiteRunEngine._runs_root`). The glob was corrected to read
`suite-runs/*/manifest.json` filtered by `model_id`; after the fix
all three runs passed 30/30. No production data was touched; the
failure was in the smoke script's independent-source check only.

## 7. Storage Integrity

Final audit after the third smoke run and server shutdown (0 live
processes verified, `/tmp/forge-tests-*` + smoke inventory cleaned):
`sha256sum -c m50_pre.sha256` run from
`/home/user/ai-model-forge-data/` → **96/96 OK, 0 non-OK**; storage
exactly **96 files / 4,002,745 bytes / 0 `.tmp`**. M50 is read-only —
zero production writes, zero SHA drift, zero growth, no
registry/index/cache files.

## 8. Git

Two commits on `arena/01a071e9-code-forge`, both pushed:

1. `M50: add suite-run history by reused count` — README.md,
   app/api.py, app/engine.py, app/suite_runs.py,
   tests/test_suite_runs.py, the swept test modules,
   smoke_m50_live.py and this report.
2. `M50: pre-milestone production inventory (m50_pre.sha256)` — the
   baseline manifest (96 lines).

HEAD == remote branch tip == branch-specific FETCH_HEAD; worktree
clean except the known untracked `ai_model_forge.egg-info/`; diff
inspected before commit — only intended files changed, no production
data, no temp artifacts. No force-push.

## 9. M51 Selection / Architectural Transition

**Candidate axes actually inspected (live, port 8771, post-M50):**
comparison `tolerance` (0.0001→7 / 0.0→1 — REQUIRED float but a
direct config echo with float-path spelling ambiguity), nested
`GatePolicy.tolerance` (10/1 — same objection, plus nested),
`ComparisonRecord.batch_size`/`max_seq_len` (single-valued 8/32),
suite-run `probe_count` (single-valued 2), `suite_probes_hash` (one
value), `status` (all completed), `failed_count` (all 0),
`completed_count` (all 2), workflow `hint`/`failed_stage_id`
(nullable free-form, None majorities 12/13 and 10/13), `composition`
(nullable nested recipe chain, 11 None), sample `seed`/
`temperature` (nullable — None groups unrepresentable in a path
parameter), `max_new_tokens`/`prompt_token_count` (single-valued),
sample-quality window ints (single-valued), evaluation
`tokenized_bin_sha256` (14/2 — exactly duplicates the covered split
axis), `records_covered` (nullable), `dataset_version` (all 1),
checkpoint `decision` (all accept)/`method` (all CPT)/
`parent_checkpoint_id` (Optional, root unrepresentable),
`RunProvenance.accepted` (all True, nested in the model manifest),
`ModelRecord.parent_model_id` (all None — no lineage events), and
suite-run `results[].evaluation_id` (a nested-list membership axis —
yet another synthetic grouping).

**Verdict on every remaining candidate:** each is single-valued,
nullable with a None majority or None-only groups, a duplicated axis,
a config echo, or a nested-list membership that would be one more
synthetic `/by-{field}` endpoint. **The simple model-scoped
history-query surface is EXHAUSTED.** Milestones M18–M50 exposed 29
by-* groupings across 9 record families; every REQUIRED persisted
field with genuine multi-group identity now has a route.

**The four architectural options, inspected on evidence:**

- **A. Query consolidation refactor — REJECTED on measurement.** The
  codebase holds 32 `list_*_for_*` engine methods; 27 are single
  list-comprehension filters over authoritative listings. The actual
  duplicated LOGIC per method is one predicate line — the bulk is
  per-method docstrings that ARE the semantic contract (derivation
  prohibitions, ordering, boundary behavior). A shared abstraction
  would save ~one line per route while risking byte-identity across
  29 live surfaces and 560 tests. Inspection does NOT demonstrate
  duplication worth removing.
- **B. Cross-model registry view — REJECTED on live evidence.** The
  global registries that exist (models, datasets, tokenizers,
  policies, probe-suites, recipes) already have global endpoints.
  The only missing cross-model surface would be a cross-model
  history view — but production holds exactly ONE productive model
  (`b5bc905326b6` owns zero evaluations/comparisons/gates/samples/
  workflows/suite-runs). Zero live utility; premature.
- **C. Evidence-based roadmap feature — NO EVIDENCE BAR MET.** The
  README's own remaining steps are: scheduled/repeatable execution
  (a background-worker layer, deliberately out of scope until
  extended on purpose), suite aggregation/quality notions (forbidden
  by the standing no-judgment rule — adopting it would be a policy
  decision, not an engineering one), richer policy inputs (schema
  redesign), an automatic improvement loop, interactive inference,
  and training methods beyond CPT/SFT (LoRA). Each is a large
  architectural jump requiring a deliberate scope decision; none is
  justified by the current production artifacts (one productive
  model, 16 evaluations, 8 comparisons, 11 gate decisions, 13
  workflows, 10 suite runs).
- **D. Honest exhaustion — SELECTED.** Per this milestone's own stop
  condition: *"The simple model-scoped history-query surface is
  exhausted; the next milestone requires a deliberate architecture
  decision rather than another synthetic grouping endpoint."*

**Consequence — no copy-ready M51 prompt is provided, deliberately.**
Manufacturing one would violate §12 of the M50 contract ("Do NOT
automatically invent M51"). What the evidence DOES support is a
concrete, ranked decision framework for the user's next deliberate
choice, each option with its exact scope and its evidence bar:

1. **Repeatable/scheduled execution of the synchronous engines**
  (roadmap-designated next step): the smallest version is a
  deterministic "re-run" surface — e.g., re-executing a registered
  recipe or suite against a new explicit model binding, reusing the
  existing engines synchronously (NO background workers, NO cron —
  the M12 recipe lineage and the M10 evidence-reuse machinery already
  exist as foundations). Evidence bar: met partially (7 recipes, 2
  recipe runs, 10 suite runs with reuse live). Requires the user to
  confirm the no-background-worker boundary stays.
2. **Adopting a suite aggregation/quality notion** — a POLICY
  decision that reverses the standing no-judgment rule; engineering
  only after that decision is made explicitly.
3. **Interactive/served inference or LoRA** — large new layers the
  README explicitly defers; not the smallest next step.
4. **Documentation consolidation only** (the README status header
  still reads "Milestone 15 complete" — 35 milestones stale, ~2,950
  lines): a zero-code maintenance milestone, justifiable on its own
  but not an architectural transition.

My recommendation, stated plainly: **option 1 in its smallest
synchronous form** (a deterministic recipe/suite re-run surface over
the existing engines), as the next milestone — but it must be
commissioned deliberately, because it extends execution semantics
rather than read-only history. Until that decision is made, M50 is
the correct final stop of the grouping ladder.
