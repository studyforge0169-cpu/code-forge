# MILESTONE 32 — SAMPLE HISTORY BY TOKENIZER — FINAL REPORT

## 1. Objective

Add exactly ONE small, read-only API capability for inspecting the
existing M15 sample history grouped by the tokenizer the samples were
generated with:

`GET /api/v1/models/{model_id}/samples/by-tokenizer/{tokenizer_id}`

* Membership comes from the persisted sample tokenizer identity on the
  authoritative M15 listing ONLY: every `SampleRecord` carries a
  required non-nullable top-level `tokenizer_id` (M15 generation
  always takes ONE explicit tokenizer from the request; the record
  also persists the matching `tokenizer_hash` audit field, preserved
  verbatim and never re-derived). NO M15 schema change was made.
* The persisted id is matched VERBATIM — never filenames, paths,
  checkpoint metadata, prompt text, generated tokens, hashes, the
  currently registered tokenizer, or a latest-tokenizer substitution;
  opaque ids; no tokenizer versioning introduced.
* Each matching sample appears EXACTLY ONCE; exact M15
  `(created_at, sample_id)` ordering inherited; verbatim
  `SampleRecord` payloads (prompt, token ids, output text, strategy,
  `result_hash` included).
* Tokenizer validated through the GLOBAL M2 registry
  (`TokenizerEngine.load`; unknown → 404); model scoping from the
  model's own M15 listing (a model never sees another model's
  samples); valid tokenizer + zero samples → `200 []`, never 404.
* Route registered after the M27 by-checkpoint route and BEFORE the
  generic `/samples/{sample_id}` detail getter.
* Strictly read-only: no sample generation, no evaluation, no new
  metrics/aggregation/ranking, no manifest writes, no
  index/cache/db/duplicated storage.

Required chain (final architectural principle):
**M2 tokenizer registry → M15 authoritative sample listing →
persisted `tokenizer_id` filter → thin facade → API route**.

## 2. Baseline

Verified BEFORE any code change (sandbox NOT reset this turn: HEAD
`0697f5e` == FETCH_HEAD — M31 implementation `1fbc8db` + inventory
`0697f5e` both present and pushed; tree clean except
`ai_model_forge.egg-info/`; production 11 dirs, venv OK):

* Full suite: **465 passed** serially in **83.39 s** (0 failed,
  0 skipped).
* OpenAPI: **63 paths** (pre-M32 state).
* Production root `/home/user/ai-model-forge-data`: **96 files /
  4,002,745 B / 0 `.tmp`**.
* `m32_pre.sha256` written (96 entries, data-root-relative); diff vs
  `m31_pre.sha256`: **EMPTY** (identical pre-state to certified M31).
* Discovery (live registry + M15 listing, not assumed): registry
  holds exactly ONE tokenizer `99106e3255c5`; model `4a0a871886ef`
  owns 4 samples, ALL persisted `tokenizer_id=99106e3255c5`, ALL
  from checkpoint `0511de4c7372`, ASC `(created_at, sample_id)`:
  `f8e66f9c7b50`, `e2b5166fa549`, `05820e8bc68a`, `4e8463e0df06`;
  `b5bc905326b6` zero samples (natural model-scoped empty case).
* `SamplingEngine` inspected FIRST: `__init__` ALREADY composes
  `self.tokenizers = TokenizerEngine(storage)` (M15 generation uses
  it) — so M32 needed **zero** new composition lines.

## 3. Implementation

`ai-model-forge/app/sampling.py`

* `list_samples_for_tokenizer(model_id, tokenizer_id)` directly after
  `list_samples_for_checkpoint` (full docstring documenting the
  grouping unit, VERBATIM persisted-identity matching, the paired
  `tokenizer_hash` audit field, global-tokenizer/model-scoping
  semantics, and read-only contract): validates the tokenizer through
  the EXISTING `self.tokenizers.load` handle (unknown →
  `FileNotFoundError`, the M2 registry contract), then returns the
  authoritative `list_samples()` filtered by the persisted top-level
  `tokenizer_id` — same objects, same `(created_at, sample_id)`
  order, no new sort rule, no filename/path/checkpoint inference, no
  substitution, no writes. No `__init__` change (handle already
  existed).

`ai-model-forge/app/engine.py` — thin facade passthrough
`list_samples_for_tokenizer` (delegates to
`self.samples.list_samples_for_tokenizer`).

`ai-model-forge/app/api.py` — ONE route
`GET /models/{model_id}/samples/by-tokenizer/{tokenizer_id}`:
summary "List samples by tokenizer", tag `sampling`, GET-only,
`response_model=list[SampleRecord]`, 404 mapping identical to the
M27 sibling; registered after the M27 by-checkpoint route and BEFORE
the generic `/samples/{sample_id}` detail getter (order verified:
M27 < by-tokenizer < generic). Landing-page M32 bullet + endpoint
`<li>` + sampling-section header comment updated.

OpenAPI: **63 → 64 paths** (new path exactly once, GET-only, tag
`sampling`). README: new M32 section + test counts 465→470 (2 sites)
+ layout line `(M27/M32)`.

No schema change (`SampleRecord` untouched); no new storage, cache,
index, worker, or second registry anywhere.

## 4. Tests

**5 new tests** (470 total = 465 + 5) in `tests/test_sampling.py`
(both sides of the existing module: engine env + HTTP client),
reusing existing fixtures/helpers (`_g`, `_http_env`, `_gen_body`,
`_corpus`); production never touched:

* `test_m32_engine_grouping_parity_order_verbatim` — for each
  (model, tokenizer) pair: parity with the M15 listing filtered by
  persisted `tokenizer_id`; `(created_at, sample_id)` order; unique
  ids; `model_id`/`tokenizer_id` on every record; verbatim
  `get_sample` parity; tok2 group exactly its one run; full
  partition over every tokenizer that actually has samples
  (disjoint groups whose union is the listing, each sample exactly
  once).
* `test_m32_engine_empty_404s_cross_model_read_only` — fresh
  tokenizer → `[]`; model-scoped empty (tok2 valid globally, small
  model has none with it); cross-model isolation (listings disjoint;
  no group ever contains the other model's ids); unknown
  model/tokenizer → `FileNotFoundError`; samples-root file set
  byte-identical before/after (read-only).
* `test_m32_engine_repeated_calls_identical` — 3 repeats,
  `model_dump(mode="json")` equality.
* `test_m32_api_by_tokenizer_grouping_partition_determinism` — 2
  samples with the original tokenizer + 2 with a fresh tok2 + fresh
  tok3; by-tokenizer == listing filtered, exact membership/order,
  verbatim detail-getter parity per record, 3 raw-byte-identical
  repeats; disjoint partition covering the listing; tok3 → `200 []`;
  filter adds no records (listing still 4).
* `test_m32_api_404s_isolation_regressions_openapi` — unknown model
  404 (alone and combined with unknown tokenizer), unknown
  well-formed + malformed tokenizer 404; second real model →
  `200 []` (cross-model isolation); M15 listing/getter, M27
  by-checkpoint, M30 evaluations-by-tokenizer parity, M31
  comparisons-by-tokenizer route intact, generic ghost sample id 404
  (no route capture); OpenAPI 64, path once, GET-only, tag
  `sampling`, `SampleRecord` items, route order M27 by-checkpoint <
  by-tokenizer < generic.

**Honest fixture notes (2 real mistakes found and fixed before
certification):** (1) the first draft assumed the big model had no
samples under the small tokenizer — an earlier M15 vocab-coverage
test generates exactly such a sample, so the full-module run broke
the fixed-pair partition/empty assertions while `-k m32` passed;
fixed by making the partition DISCOVERY-based (groups over every
tokenizer that actually has samples) and using the M32-created tok2
for the model-scoped empty case. (2) A pyflakes-unused `url` variable
and one line clipped by an over-eager edit were repaired; final files
pyflakes-clean.

All 15 prompt-listed areas covered. OpenAPI count assertions updated
**63 → 64 at all 16 hard-coded sites** across 7 test modules with
adjacent comment ladders extended (`+ 1 (M32 samples by-tokenizer)`);
grep-verified: 0 `== 63` remain (46 `== 64`, including pre-existing
unrelated 64s).

**Measured:** focused 5/5 (`-k m32`) and full module 24/24 in BOTH
select modes; full suite **470 passed** in **82.94 s**, and after
stale-`/tmp` cleanup **470 passed** again in **81.85 s**. compileall
clean (app, tests, `smoke_m32_live.py`). pyflakes clean on all
changed/new files (pre-existing legacy findings only: 2, confined to
`smoke_m9_live.py` / `smoke_m11_live.py`, reported separately).

## 5. Live Smoke

`smoke_m32_live.py` (new, port **8753**, baseline inventory
`/tmp/m32-smoke-baseline-inventory.json`), run against the production
`FORGE_ROOT`: **32/32 checks PASS first run** (repeated 3× total,
exit 0 each). The smoke DISCOVERS the distribution live, then checks
A–T:

* **A** baseline audit 96/4,002,745/0 + SHA256 inventory saved (96) +
  both models resolve + 3 known checkpoints + M11/M16/M18–M31
  pre-state + dashboard hash + OpenAPI 64 pre-state + known pair 200.
* **B** discovery: 1 tokenizer `99106e3255c5`; 4 samples of
  `4a0a871886ef`, ALL under it, ALL from checkpoint `0511de4c7372`,
  ASC order exactly `f8e66f9c7b50, e2b5166fa549, 05820e8bc68a,
  4e8463e0df06`; `b5bc905326b6` none.
* **C** exact parity: response == M15 listing filtered by persisted
  `tokenizer_id` (no missing/extra/duplicate; every sample EXACTLY
  ONCE).
* **D** persisted identity VERBATIM on all 4 records + verbatim
  detail-getter parity (prompt, token ids, output text, strategy,
  `result_hash`).
* **E** `(created_at, sample_id)` ASCENDING exactly as the M15
  listing.
* **F** three GETs raw-byte-identical.
* **G** `b5bc905326b6` + `99106e3255c5` → `200 []`
  (registry-verified model-scoped empty).
* **H** unknown model → 404 (even with the real tokenizer id).
* **I** unknown well-formed + malformed tokenizer → 404.
* **J** M2 tokenizer registry unchanged (list + get verbatim; unknown
  id still 404).
* **K** M15 listing + all 4 detail getters unchanged.
* **L** M27 by-checkpoint unchanged (0511→4 / 025e→0 / 30a8→0).
* **M** M30 evaluations by-tokenizer unchanged (16 under
  `99106e3255c5`).
* **N** M31 comparisons by-tokenizer unchanged (8 under
  `99106e3255c5`).
* **O** M24/M28 evaluation histories unchanged (3/3/3 by-checkpoint;
  16 by-dataset with filtered-listing parity).
* **P** M26/M29 comparison histories unchanged (025e→6 / 30a8→5 /
  0511→1; 8 by-dataset with filtered-listing parity).
* **Q** dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  + full output unchanged.
* **R** M11 workflows, M3 checkpoint registry, M9 policy/probe-suite
  registries, M12/M14 recipe registry unchanged.
* **S** OpenAPI exactly 64, new path once, after M27 by-checkpoint,
  before the generic sample route.
* **T** zero drift (final §6).

Server started fresh for the smoke and STOPPED after (uvicorn on
8753, log clean).

## 6. Storage Integrity

* Before: **96 files / 4,002,745 B / 0 `.tmp`**; after: **96 files /
  4,002,745 B / 0 `.tmp`**.
* Smoke internal per-file SHA256: 0 changed / 0 missing / 0 new.
* Final audit `sha256sum -c m32_pre.sha256` from the data root:
  **96/96 OK, 0 non-OK** — every pre-existing production file
  byte-identical. M32 wrote NOTHING to production storage (strictly
  read-only; all test writes went to pytest tmp dirs).

## 7. Regression / Compatibility

* Full suite green before (465) and after (470); no existing test
  weakened; only genuinely affected OpenAPI-count assertions updated
  (63 → 64 ×16 sites) with adjacent comments.
* M2 tokenizer registry contract intact (unknown → 404, list/get
  verbatim — live J).
* M15 sampling: generate/list/get unchanged (live K; module 24/24).
* M27 sample-by-checkpoint unchanged (live L: 4/0/0).
* M30 evaluation-by-tokenizer (16) and M31 comparison-by-tokenizer
  (8) unchanged (live M/N).
* M24/M28 evaluation histories (3/3/3; 16 by-dataset) and M26/M29
  comparison histories (6/5/1; 8 by-dataset) unchanged (live O/P).
* M6 gates, M16/M18–M20 sample-quality, M21/M22/M25 suite-runs,
  M17 dashboard hash/output, M11/M12/M14 registries unchanged (live
  A4/Q/R).
* Generic `/samples/{sample_id}` detail getter still 404s ghost ids
  (no route capture — by-tokenizer registered before it); OpenAPI 64
  with correct route order.
* README + landing page + endpoint list extended; no redundant
  documentation created.

## 8. Commit / Final Status

* Implementation commit: **`M32: add sample history by tokenizer`**
  on branch `arena/01a071e9-code-forge` (files: README.md, app/api.py,
  app/engine.py, app/sampling.py, tests/test_sampling.py + 6 further
  test modules' OpenAPI assertions, smoke_m32_live.py,
  M32_final_report.md; egg-info NOT committed).
* Follow-up inventory commit (repo convention M21–M31):
  `M32: add pre-milestone storage inventory` tracking
  `m32_pre.sha256`.
* Push verified: local == FETCH_HEAD after both commits; working
  tree clean except expected `ai_model_forge.egg-info/`.
* All gates passed in order: baseline (465 @ 83.39 s, OpenAPI 63,
  SHA256 pre-inventory) → focused (5/5; module 24/24 both modes) →
  full suite (470 @ 82.94 s) → stale-`/tmp` cleanup (10 dirs removed
  after confirming 0 active test processes) → full suite rerun
  (470 @ 81.85 s) → compileall → pyflakes → standalone OpenAPI
  verification (64, order+GET+schema) → live smoke (32/32, zero
  drift) → final SHA256 audit (96/96 OK).
* **M32 is certified complete.**

## 9. Next Milestone Prompt

```
# MILESTONE 33 — SAMPLE-QUALITY HISTORY BY TOKENIZER

Continue the existing **AI Model Forge** project.

M32 is the current certified milestone.

The goal of M33 is to add one small, read-only API capability for
inspecting the existing M16 sample-quality history grouped by the
tokenizer the measured samples were generated with.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M32 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 470 tests passing
* OpenAPI: 64 paths
* M32 samples by-tokenizer endpoint working
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
  at most ONE new `__init__` composition line (only if the
  sample-quality engine lacks a tokenizer handle — inspect first).
* No new sample-quality measurements, no metrics/scoring/ranking,
  no training/rollback/optimization/HPO, no workers, databases,
  caches, indexes, no new persistence, no automatic decisions, no
  workflow changes, no dashboard redesign, no Gemini, no model
  improvement logic.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the real `b5bc905326b6` (valid model + valid
  tokenizer, empty sample-quality history).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (64 -> 65) and their adjacent stale comments.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 2. M33 OBJECTIVE

Add exactly ONE read-only endpoint:

`GET /api/v1/models/{model_id}/sample-quality/by-tokenizer/{tokenizer_id}`

answering "which persisted M16 sample-quality records of this model
measured samples generated with this tokenizer?" and nothing else.

Membership rules (identical philosophy to M30/M31/M32):

* The authoritative source is the model's OWN M16 sample-quality
  listing (the same one the existing M19/M20 by-sample/by-checkpoint
  groupings use); never re-derive from filenames, paths, sample
  text, checkpoint ids, dataset ids, hashes, the currently
  registered tokenizer, or a latest-tokenizer substitution.
* Each persisted SampleEvaluationRecord carries a top-level
  `tokenizer_id` (verified live at M32: both production records
  persist `tokenizer_id=99106e3255c5`); match it VERBATIM.
* Each record appears EXACTLY ONCE.
* Inherit the exact existing M16 listing order (discover the exact
  deterministic sort key during inspection; do not invent a new
  one).
* Return verbatim SampleEvaluationRecord payloads.
* Tokenizers are GLOBAL: validate the tokenizer through the existing
  M2 registry first (unknown tokenizer -> 404); model scoping comes
  from the model's own sample-quality listing (a model never sees
  another model's records).
* Valid tokenizer + zero records for the model -> `200 []` (NEVER
  404); unknown model or unknown tokenizer -> 404.
* Register the route so it does not capture or get captured by the
  existing sample-quality routes (M19/M20 by-sample/by-checkpoint
  and any generic getter); M16/M19/M20/M30/M31/M32 routes must stay
  intact and unconfused.
* Strictly read-only: no measurements, no new metrics, no manifest
  writes, no index/cache/db/duplicated storage.
* Required chain: M2 tokenizer validation -> authoritative M16
  sample-quality listing -> persisted tokenizer_id filter -> thin
  facade -> route.

## 3. PRODUCTION FACTS (verified at M32)

* Exactly ONE tokenizer exists in the registry: `99106e3255c5`.
* Model `4a0a871886ef` owns 2 sample-quality records,
  `8ff910cf2a9e` and `31a283413c75`, BOTH persisted with top-level
  `tokenizer_id=99106e3255c5`, both measuring sample `f8e66f9c7b50`
  (generated from checkpoint `0511de4c7372`); M20 groups both under
  that checkpoint.
* Model `b5bc905326b6` exists with no sample-quality records
  (natural model-scoped empty case: valid global tokenizer + model
  with empty history -> `200 []`).
* Dashboard hash (unchanged since M17, re-verified at M32):
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.
* Storage: 96 files / 4,002,745 B / 0 `.tmp`.

## 4. REQUIRED BEHAVIOR

1. `200` with the exact filtered array for a known pair
   (`4a0a871886ef` + `99106e3255c5` -> the 2 records).
2. Exact parity with the authoritative M16 listing filtered locally
   by persisted `tokenizer_id` (no missing, no extra, no duplicate).
3. Persisted identity travels VERBATIM (no substitution/rewriting).
4. Authoritative listing order preserved exactly.
5. Deterministic: repeated GETs byte-identical (x3).
6. Valid tokenizer + zero records for the model -> `200 []`.
7. Unknown model -> 404; unknown tokenizer -> 404 (well-formed AND
   malformed ids).
8. Cross-model isolation: tokenizers are global, but each model's
   group draws only from its own listing.
9. M16/M19/M20 sample-quality surfaces, M30/M31/M32 by-tokenizer
   surfaces byte-identical before and after.
10. Zero production storage growth.

## 5. ENGINE IMPLEMENTATION

* Inspect the sample-quality engine FIRST (its `__init__`, its
  listing method, the M20 by-checkpoint method for the established
  pattern).
* Add ONE method `list_sample_evaluations_for_tokenizer(model_id,
  tokenizer_id)` (or the exact existing naming convention) that
  validates through the tokenizer registry handle (compose ONE line
  in `__init__` only if missing) and filters the authoritative
  listing by the persisted top-level `tokenizer_id`; the facade and
  route are thin pass-throughs.
* Docstring must explain the grouping unit and that identity is
  matched VERBATIM from the persisted record.

## 6. TESTS

* Target: 470 -> 475 tests (~5 new), in the existing sample-quality
  test module, REUSING its fixtures/helpers (never modify
  production; never inflate the corpus; follow the established
  second-tokenizer fixture pattern if a second tokenizer is needed).
* Cover: authoritative-filter parity; verbatim identity; exclusion
  of unrelated tokenizers; empty `[]` for a valid tokenizer; 404s
  (unknown model/tokenizer); cross-model isolation; ordering;
  determinism; no storage writes; M16/M19/M20 regressions; M32 (or
  the newest by-tokenizer surface) regression; OpenAPI 64 -> 65 +
  route order + ghost id 404 on the generic getter.
* Search EVERY hard-coded OpenAPI count (grep `== 64`) and update
  only genuinely affected assertions + adjacent stale comments.
* Gates in order: baseline tests -> focused -> full suite -> full
  suite again after stale `/tmp/forge-tests-*` cleanup (verify no
  process first) -> compileall (incl. the new smoke) -> pyflakes
  (report pre-existing findings separately) -> OpenAPI verification
  -> live smoke -> final SHA256 audit; commit only after all gates
  pass.

## 7. LIVE TEST

Write `smoke_m33_live.py` (mirror `smoke_m32_live.py`), port
**8754**, baseline inventory
`/tmp/m33-smoke-baseline-inventory.json`; capture the baseline FIRST
(files/bytes/tmp/SHA256/OpenAPI/test count/registries/
sample-quality distribution), then checks A-T: A known pair 200; B
discover ids; C listing-parity; D persisted id verbatim; E listing
order; F >= 3 byte-identical repeats; G empty `200 []` for
`b5bc905326b6`; H unknown model 404; I unknown tokenizer 404;
J tokenizer registry unchanged; K M16 listing+getters; L M20
by-checkpoint; M M30 evaluations-by-tokenizer; N M31
comparisons-by-tokenizer; O M32 samples-by-tokenizer + M24/M28
evaluation histories; P M26/M29 comparison histories; Q dashboard;
R policy/probe/suite/recipe registries; S OpenAPI 65; T zero drift.
Storage must stay 96/4,002,745/0 with SHA256 before/after (0
changed/new/missing; investigate + report ANY change honestly).

## 8. FINAL AUDIT

* `sha256sum -c` the pre-milestone inventory from the data root:
  96/96 OK, 0 changed/new/missing.
* Recovery runbook: restore zip -> verify pre-milestone SHA256 ->
  rebuild venv -> reconcile git with the authoritative remote ->
  re-prove baseline; NEVER force-push.

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; no measurements, no scoring, no rankings,
  no automatic decisions, no schema change, no duplication of the
  authoritative listing, no caches/indexes/databases/workers/queues,
  no training/rollback/optimization/HPO/Gemini, no unrelated
  refactoring.
* Report honestly; never certify from a partial run.

## 10. COMMIT / PUSH / REPORT

* Commit message: `M33: add sample-quality history by tokenizer`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M33_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (record ids, per-tokenizer
  counts, ordering, determinism, 404s, cross-model/scoping, SHA256,
  dashboard hash, commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M34 prompt,
  grounded ONLY in facts discovered and verified during M33 (no
  invented ids, counts, or endpoints), defining the next small
  additive read-only capability, including automatic M35 prompt
  generation.
```
