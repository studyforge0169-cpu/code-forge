# MILESTONE 31 — COMPARISON HISTORY BY TOKENIZER — FINAL REPORT

## 1. Objective

Add exactly ONE small, read-only API capability for inspecting the
existing M5 comparison history grouped by the tokenizer the
comparisons measured with:

`GET /api/v1/models/{model_id}/comparisons/by-tokenizer/{tokenizer_id}`

* Membership comes from the persisted top-level shared-probe
  `tokenizer_id` on the authoritative M5 listing ONLY. A comparison is
  valid only when both sides measure the SAME probe, so
  `ComparisonRecord` persists exactly ONE top-level `tokenizer_id` —
  the tokenizer identity of the shared probe both sides measured by
  construction (M5 refuses cross-probe requests with 422; per-side
  tokenizer identities cannot occur; `ComparisonSide` carries NO
  tokenizer fields and NONE were added).
* The comparison (not the side) is the unit of grouping: every
  matching record — including same-checkpoint A=B records — appears
  EXACTLY ONCE.
* Never inferred from filenames, paths, checkpoint ids, dataset
  identities, hashes, the currently registered tokenizer, or a
  latest-tokenizer substitution; opaque ids; no tokenizer versioning.
* Model validated through the M5 listing path, tokenizer through M2
  `TokenizerEngine.load` (unknown → 404); tokenizers are GLOBAL, model
  scoping comes from the model's own M5 listing.
* Exact M5 `(created_at, comparison_id)` ordering inherited; verbatim
  `ComparisonRecord` payloads (verdict + per-side losses included).
* Valid tokenizer + zero comparisons → `200 []`, never 404.
* Route registered after the M29 by-dataset route and BEFORE the
  generic `/comparisons/{comparison_id}` detail getter.
* Strictly read-only: no comparison/evaluation execution, no new
  metrics/manifest writes/index/cache/db/duplicated storage.
* Required chain: M2 tokenizer validation → M5 authoritative listing
  → persisted tokenizer_id filter → thin facade → route.

No architecture redesign, no training/rollback/optimization/HPO/
ranking, no workers/caches/databases, no schema change, no M5 listing
duplication, no unrelated refactoring, no production fixture
manipulation.

## 2. Baseline

Verified BEFORE any code change (this turn, sandbox NOT reset:
HEAD `5277aa9` == FETCH_HEAD, tree clean except egg-info, production
data 11 dirs + venv intact):

* Production root `/home/user/ai-model-forge-data`: **96 files /
  4,002,745 B / 0 `.tmp`**.
* Full suite: **460 passed** serially in **77.59 s** (0 failed,
  0 skipped).
* OpenAPI: **62 paths** (pre-M31 state).
* `m31_pre.sha256` written (96 entries, data-root-relative); diff vs
  `m30_pre.sha256`: **EMPTY** (identical pre-state to the certified
  M30 milestone).
* Discovery (live registry + M5 listing): tokenizer registry holds
  exactly ONE tokenizer `99106e3255c5`; model `4a0a871886ef` owns
  8 comparisons, ALL persisted with shared-probe
  `tokenizer_id=99106e3255c5` (dataset `ee1a716c4573` v1), ASC
  `(created_at, comparison_id)`: `fc379bfcb50f`, `baa361012e00`,
  `d683f9b81195`, `786de08efe4c`, `5c5ff22151ed`, `d9a62dde016b`,
  `729f9c55ea89`, `d62f89e97c85`. M26 membership 025e→6 / 30a8→5 /
  0511→1; two same-checkpoint A=B records; two current-vs-checkpoint
  records. `b5bc905326b6`: zero comparisons (natural model-scoped
  empty case).
* `ComparisonEngine.__init__` inspected first: composed
  datasets/training/evaluation engines only — no tokenizer handle
  existed (M31 adds exactly ONE composition line).

## 3. Implementation

`ai-model-forge/app/comparison.py`

* Import `TokenizerEngine` (after the `.storage` import,
  alphabetical).
* `__init__`: ONE new composition line
  `self.tokenizers = TokenizerEngine(storage)   # registry validation
  (M31)` following the existing style.
* `list_comparisons_for_tokenizer(model_id, tokenizer_id)` directly
  after `list_comparisons_for_dataset` (full shared-probe docstring):
  validates the model through `self.datasets`/listing path (unknown →
  `FileNotFoundError`), validates the tokenizer through
  `self.tokenizers.load` (unknown → `FileNotFoundError`, the M2
  registry contract), then returns the authoritative
  `list_comparisons()` filtered by the persisted top-level
  `tokenizer_id` VERBATIM — same objects, same order, no new sort
  rule, no side-derived inference, no dedup needed (the listing holds
  each record exactly once).

`ai-model-forge/app/engine.py` — thin facade passthrough
`list_comparisons_for_tokenizer` (delegates to
`self.comparisons.list_comparisons_for_tokenizer`).

`ai-model-forge/app/api.py` — ONE route at the by-tokenizer path:
summary "List comparisons by tokenizer", tag `comparison`, GET-only,
`response_model=list[ComparisonRecord]`, error mappings identical to
the M26/M29 siblings; registered after the M29 by-dataset route
(:950) and BEFORE the generic `/comparisons/{comparison_id}` detail
getter — by-tokenizer sits at :983, generic at :1016 (order verified
live: M29 by-dataset < by-tokenizer < generic). Landing-page M31
bullet + endpoint `<li>` + comparison-section header comment updated.

OpenAPI: **62 → 63 paths** (new path exactly once, GET-only).
README: new M31 section + test counts 460→465 (2 sites) + comparison
layout line extended to M26/M29/M31.

No schema change: `ComparisonSide` untouched (no tokenizer fields).
No new storage, cache, index, or worker anywhere.

## 4. Tests

**5 new tests** (465 total = 460 + 5), reusing existing fixtures and
the M5 `m5-tok2` second-tokenizer pattern; production never touched:

`tests/test_comparison.py` (+3, engine side, on the shared module env
via `_m31_env`: trains `m31-tok2` (vocab 320) + one comparison with
it, `m31-tok3` (vocab 300) with zero comparisons, and one fresh record
under the env's original tokenizer so partition assertions hold under
`-k m31` too):

* `test_m31_engine_filters_by_persisted_tokenizer_identity` — for each
  tokenizer: parity with the M5 listing filtered by persisted
  `tokenizer_id`; `(created_at, comparison_id)` order; unique ids;
  `model_id`/`tokenizer_id` on every record; verbatim `get_comparison`
  parity; tok2 group is exactly its one run; explicit disjoint
  partition whose union is the full listing.
* `test_m31_engine_empty_404s_model_scoping_read_only` — fresh
  tokenizer → `[]`; second model (M26 `b`) + original tokenizer → `[]`
  (global tokenizer, model-scoped history); unknown model/tokenizer →
  `FileNotFoundError`; cross-model isolation; comparison-file set
  byte-identical before/after (read-only).
* `test_m31_engine_repeated_calls_identical` — 3 repeats,
  `model_dump(mode="json")` equality.

`tests/test_comparison_api.py` (+2, API side):

* `test_m31_by_tokenizer_grouping_partition_determinism` — two tok1
  runs + two tok2 runs (one same-checkpoint A=B) + fresh tok3;
  by-tokenizer(tok1) == listing filtered, exact ids/order, verbatim
  detail-getter parity per record, 3 raw-byte-identical repeats;
  disjoint partition; tok2 == exactly its 2 runs (A=B once);
  tok3 → `200 []`; filter adds no records (listing still 4).
* `test_m31_by_tokenizer_404s_isolation_regressions_openapi` — unknown
  model 404 (alone and combined with unknown tokenizer), unknown
  well-formed + malformed tokenizer 404; second model → `200 []`;
  M5 listing/getter, M26 by-checkpoint, M29 by-dataset, M30
  evaluations-by-tokenizer parity, generic ghost id 404 (no route
  capture); OpenAPI 63, path once, GET-only, tag `comparison`,
  `ComparisonRecord` items, route order M29 by-dataset < by-tokenizer
  < generic.

All 15 prompt-listed areas covered. OpenAPI count assertions updated
**62 → 63 at all 15 hard-coded sites** across 7 test modules, each
with its adjacent comment ladder extended (`+ 1 (M31 comparisons
by-tokenizer)`); the dashboards comment (`M18–M30` → `M18–M31`,
62 → 63) updated too — verified by grep: 0 `== 62` remain, 15
`== 63` now.

**Measured:** focused 3/3 (engine) + 11/11 (API module); full suite
**465 passed** in **84.13 s**, and after stale-`/tmp` cleanup
**465 passed** again in **82.23 s**. pyflakes clean on all changed
files (legacy findings only: 2, confined to `smoke_m9_live.py` /
`smoke_m11_live.py`). compileall clean (app, tests,
`smoke_m31_live.py`).

## 5. Live Smoke

`smoke_m31_live.py` (new, port **8752**, baseline inventory
`/tmp/m31-smoke-baseline-inventory.json`), run against the production
`FORGE_ROOT`: **31/31 checks PASS first run** (repeated 3× total, all
green). The smoke DISCOVERS the distribution live, then checks
A–S:

* **A** baseline audit 96/4,002,745/0 + SHA256 inventory saved (96) +
  both models resolve + 3 known checkpoints + M11/M16/M18–M30
  pre-state + dashboard hash + OpenAPI 63 pre-state + known pair 200.
* **B** discovery: 1 tokenizer `99106e3255c5`; 8 comparisons of
  `4a0a871886ef`, ALL under it, ASC order exactly
  `fc379bfcb50f, baa361012e00, d683f9b81195, 786de08efe4c,
  5c5ff22151ed, d9a62dde016b, 729f9c55ea89, d62f89e97c85`;
  `b5bc905326b6` none.
* **C** exact parity: response == M5 listing filtered by persisted
  shared-probe `tokenizer_id` (no missing/extra/duplicate; every
  comparison EXACTLY ONCE).
* **D** persisted identity VERBATIM on all 8 records + verbatim
  detail-getter parity (verdict + per-side losses).
* **E** `(created_at, comparison_id)` ASCENDING exactly as the M5
  listing.
* **F** three GETs raw-byte-identical.
* **G** `b5bc905326b6` + `99106e3255c5` → `200 []`
  (registry-verified model-scoped empty).
* **H** unknown model → 404 (even with the real tokenizer id).
* **I** unknown well-formed + malformed tokenizer → 404.
* **J** M2 tokenizer registry unchanged (list + get verbatim; unknown
  id still 404).
* **K** M5 listing + all 8 detail getters unchanged.
* **L** M26 by-checkpoint unchanged (025e→6 / 30a8→5 / 0511→1).
* **M** M29 by-dataset unchanged (8, filtered-listing parity).
* **N** M30 evaluations by-tokenizer unchanged (16 under
  `99106e3255c5`).
* **O** M2/M4/M15–M25 surfaces unchanged (evaluations 16, M24 3/3/3,
  M28 16 by-dataset, M16 sample quality 2, M18 suite runs 10, M25
  by-checkpoint 10, M27 samples 4/0/0, M6 gates 1).
* **P** dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  + full output unchanged.
* **Q** policy/probe-suite/suite/recipe/workflow/checkpoint registries
  unchanged.
* **R** OpenAPI exactly 63, new path once, after M29 by-dataset,
  before the generic comparison route.
* **S** zero drift (final §6).

Server started fresh for the smoke and STOPPED after (uvicorn on
8752, log clean).

## 6. Storage Integrity

* Before: **96 files / 4,002,745 B / 0 `.tmp`**; after: **96 files /
  4,002,745 B / 0 `.tmp`**.
* Smoke internal per-file SHA256: 0 changed / 0 missing / 0 new.
* Final audit `sha256sum -c m31_pre.sha256` from the data root:
  **96/96 OK, 0 non-OK** — every pre-existing production file
  byte-identical. M31 wrote NOTHING to production storage (the
  endpoint is strictly read-only; all writes in tests went to pytest
  tmp dirs).

## 7. Regression / Compatibility

* Full suite green before (460) and after (465) — no existing test
  weakened; only genuinely affected OpenAPI-count assertions updated
  (62 → 63 ×15 sites) with their adjacent comments.
* M5 run/list/get, M26 by-checkpoint, M27 samples-by-checkpoint, M28
  by-dataset, M29 by-dataset, M30 evaluations-by-tokenizer all
  re-verified live unchanged; generic detail getter 404s ghost ids
  (no route capture — by-tokenizer registered before it).
* M2 tokenizer registry contract intact (unknown → 404, list/get
  verbatim); M3/M6/M9/M11/M12/M14/M15–M25 surfaces + dashboard hash
  unchanged (live checks O/P/Q).
* README: new M31 section + counts 465 (2 sites) + M26/M29/M31
  grouping line; landing page + endpoint list extended.
* No schema changes; `ComparisonSide` carries no tokenizer fields
  (asserted by existing M5 structure tests, still green).

## 8. Commit / Final Status

* Commit: **`M31: add comparison history by tokenizer`** on branch
  `arena/01a071e9-code-forge` (files: README.md, app/api.py,
  app/comparison.py, app/engine.py, tests ×7, smoke_m31_live.py,
  M31_final_report.md; egg-info NOT committed; `m31_pre.sha256`
  kept untracked per prior-milestone convention).
* Push verified: local == FETCH_HEAD, tree clean after commit.
* All gates passed in order: baseline (460 @ 77.59 s, OpenAPI 62,
  SHA256 pre-inventory) → focused (3/3 + 11/11) → full suite
  (465 @ 84.13 s) → stale-`/tmp` cleanup (18 dirs removed after
  verifying no live process) → full suite rerun (465 @ 82.23 s) →
  compileall → pyflakes → OpenAPI standalone (63, order verified) →
  live smoke (31/31, zero drift) → final SHA256 audit (96/96 OK).
* **M31 is certified complete.**

## 9. Next Milestone Prompt

```
# MILESTONE 32 — SAMPLE HISTORY BY TOKENIZER

Continue the existing **AI Model Forge** project.

M31 is the current certified milestone.

The goal of M32 is to add one small, read-only API capability for
inspecting the existing M15 sample store grouped by the tokenizer the
samples were generated with.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M31 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 465 tests passing
* OpenAPI: 63 paths
* M31 comparisons by-tokenizer endpoint working
* M30 evaluations by-tokenizer endpoint working
* M29 comparisons by-dataset endpoint working
* M28 evaluations by-dataset endpoint working
* M27 samples by-checkpoint endpoint working
* M26 comparisons by-checkpoint endpoint working
* M25 suite-runs by-checkpoint endpoint working
* M24 evaluations by-checkpoint endpoint working
* M23 gate decisions by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working

* ONE read-only endpoint; ONE engine method; ONE facade; ONE route;
  at most ONE new `__init__` composition line (only if the sample
  engine lacks a tokenizer handle — inspect first).
* No new samples/generation, metrics, scoring, training/rollback/
  optimization/HPO/ranking, no workers, databases, caches, indexes,
  no new persistence, no automatic decisions, no workflow changes, no
  dashboard redesign, no Gemini, no model improvement logic.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the real `b5bc905326b6` (valid model + valid
  tokenizer, empty sample history).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (63 -> 64) and their adjacent stale comments.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 2. M32 OBJECTIVE

Add exactly ONE read-only endpoint:

`GET /api/v1/models/{model_id}/samples/by-tokenizer/{tokenizer_id}`

answering "which persisted M15 samples of this model were generated
with this tokenizer?" and nothing else.

Membership rules (identical philosophy to M30/M31):

* The authoritative source is the model's OWN M15/M27 sample listing
  (the same one `GET /models/{id}/samples/by-checkpoint/...` groups);
  never re-derive from filenames, paths, checkpoint ids, dataset ids,
  hashes, the currently registered tokenizer, or a latest-tokenizer
  substitution.
* Each persisted sample record carries a top-level `tokenizer_id`;
  match it VERBATIM.
* Each sample appears EXACTLY ONCE.
* Inherit the exact existing listing order (no new sort rule).
* Return verbatim sample payloads.
* Tokenizers are GLOBAL: validate the tokenizer through the existing
  M2 registry first (unknown tokenizer -> 404); model scoping comes
  from the model's own sample listing (a model never sees another
  model's samples).
* Valid tokenizer + zero samples for the model -> `200 []` (NEVER
  404); unknown model or unknown tokenizer -> 404.
* Register the route so it does not capture or get captured by the
  existing sample routes (M27 by-checkpoint, the generic sample
  getter); M15/M27/M30/M31 routes must stay intact and unconfused.
* Strictly read-only: no sample generation, no new metrics, no
  manifest writes, no index/cache/db/duplicated storage.
* Required chain: M2 tokenizer validation -> authoritative sample
  listing -> persisted tokenizer_id filter -> thin facade -> route.

## 3. PRODUCTION FACTS (verified at M31)

* Exactly ONE tokenizer exists in the registry: `99106e3255c5`.
* Model `4a0a871886ef` owns 4 samples, ALL persisted with
  `tokenizer_id=99106e3255c5` (M27 groups the same 4 under checkpoint
  `0511de4c7372`; the other two known checkpoints have 0 each).
* Model `b5bc905326b6` exists with no samples (natural model-scoped
  empty case: valid global tokenizer + model with empty history ->
  `200 []`).
* Sample ids are opaque 12-hex (e.g. `f8e66f9c7b50`).
* Dashboard hash (unchanged since M17, re-verified at M31):
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.
* Storage: 96 files / 4,002,745 B / 0 `.tmp`.

## 4. REQUIRED BEHAVIOR

1. `200` with the exact filtered array for a known pair
   (`4a0a871886ef` + `99106e3255c5` -> the 4 samples).
2. Exact parity with the authoritative sample listing filtered locally
   by persisted `tokenizer_id` (no missing, no extra, no duplicate).
3. Persisted identity travels VERBATIM (no substitution/rewriting).
4. Authoritative listing order preserved exactly.
5. Deterministic: repeated GETs byte-identical (x3).
6. Valid tokenizer + zero samples for the model -> `200 []`.
7. Unknown model -> 404; unknown tokenizer -> 404 (well-formed AND
   malformed ids).
8. Cross-model isolation: tokenizers are global, but each model's
   group draws only from its own listing.
9. M15 sample run/list/get, M27 by-checkpoint, M30/M31 by-tokenizer
   surfaces byte-identical before and after.
10. Zero production storage growth.

## 5. ENGINE IMPLEMENTATION

* Inspect the sample engine FIRST (its `__init__`, its listing
  method, the M27 by-checkpoint method for the established pattern).
* Add ONE method `list_samples_for_tokenizer(model_id, tokenizer_id)`
  that validates through the tokenizer registry handle (compose ONE
  line in `__init__` only if missing) and filters the authoritative
  listing by the persisted top-level `tokenizer_id`; the facade and
  route are thin pass-throughs.
* Docstring must explain the grouping unit and that identity is
  matched VERBATIM from the persisted record.

## 6. TESTS

* Target: 465 -> 470 tests (~5 new), in the existing sample test
  modules, REUSING their fixtures/helpers (never modify production;
  never inflate the corpus; follow the established second-tokenizer
  fixture pattern if a second tokenizer is needed).
* Cover: authoritative-filter parity; verbatim identity; exclusion of
  unrelated tokenizers; empty `[]` for a valid tokenizer; 404s
  (unknown model/tokenizer); cross-model isolation; ordering;
  determinism; no storage writes; M15/M27 regressions; OpenAPI 63 ->
  64 + route order + ghost id 404 on the generic getter.
* Search EVERY hard-coded OpenAPI count (grep `== 63`) and update only
  genuinely affected assertions + adjacent stale comments.
* Gates in order: baseline tests -> focused -> full suite -> full
  suite again after stale `/tmp/forge-tests-*` cleanup (verify no
  process first) -> compileall (incl. the new smoke) -> pyflakes
  (report pre-existing findings separately) -> OpenAPI verification ->
  live smoke -> final SHA256 audit; commit only after all gates pass.

## 7. LIVE TEST

Write `smoke_m32_live.py` (mirror `smoke_m31_live.py`), port **8753**,
baseline inventory `/tmp/m32-smoke-baseline-inventory.json`; capture
the baseline FIRST (files/bytes/tmp/SHA256/OpenAPI/test count/
registries/sample distribution), then checks A-S: A known pair 200;
B discover ids; C listing-parity; D persisted id verbatim; E listing
order; F >= 3 byte-identical repeats; G empty `200 []` for
`b5bc905326b6`; H unknown model 404; I unknown tokenizer 404;
J tokenizer registry unchanged; K M15 listing+getters; L M27
by-checkpoint; M M30; N M31; O M2/M4/M16-M29 surfaces; P dashboard;
Q policy/probe/suite/recipe registries; R OpenAPI 64; S zero drift.
Storage must stay 96/4,002,745/0 with SHA256 before/after (0 changed/
new/missing; investigate + report ANY change honestly).

## 8. FINAL AUDIT

* `sha256sum -c` the pre-milestone inventory from the data root: 96/96
  OK, 0 changed/new/missing.
* Recovery runbook: restore zip -> verify pre-milestone SHA256 ->
  rebuild venv -> reconcile git with the authoritative remote ->
  re-prove baseline; NEVER force-push.

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; no sample generation, no scoring, no
  rankings, no automatic decisions, no schema change, no duplication
  of the authoritative listing, no caches/indexes/databases/workers/
  queues, no training/rollback/optimization/HPO/Gemini, no unrelated
  refactoring.
* Report honestly; never certify from a partial run.

## 10. COMMIT / PUSH / REPORT

* Commit message: `M32: add sample history by tokenizer`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M32_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (sample ids, per-tokenizer
  counts, ordering, determinism, 404s, cross-model/scoping, SHA256,
  dashboard hash, commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M33 prompt, grounded
  ONLY in facts discovered and verified during M32 (no invented ids,
  counts, or endpoints), defining the next small additive read-only
  capability, including automatic M34 prompt generation.
```
