# MILESTONE 33 — SAMPLE-QUALITY HISTORY BY TOKENIZER — FINAL REPORT

## 1. Objective

Add exactly ONE small, read-only API capability for inspecting the
existing M16 sample-quality history grouped by the tokenizer of the
measured samples:

`GET /api/v1/models/{model_id}/sample-quality/by-tokenizer/{tokenizer_id}`

* Membership comes from the persisted measurement tokenizer identity
  on the authoritative M16 listing ONLY: every
  `SampleEvaluationRecord` carries a required non-nullable top-level
  `tokenizer_id` (M16 measures an immutable M15 sample under its own
  RECORDED state; the record persists that state's tokenizer
  identity). `SampleEvaluationRecord` was NOT altered for this
  endpoint.
* The persisted id is matched VERBATIM — never filenames, paths,
  sample ids, checkpoint ids, tokenizer contents or hashes, the
  currently registered tokenizer, or a latest-tokenizer substitution;
  opaque ids; no tokenizer versioning introduced.
* Each matching record appears EXACTLY ONCE; exact M16
  `(created_at, evaluation_id)` ordering inherited; verbatim
  `SampleEvaluationRecord` payloads (loss_nats/perplexity included).
* Tokenizer validated through the GLOBAL M2 registry
  (`TokenizerEngine.load`; unknown → 404); model scoping from the
  model's own M16 listing (a model never sees another model's
  measurements); valid tokenizer + zero records → `200 []`, never
  404.
* Route registered after the M20 by-checkpoint route and BEFORE the
  generic `/sample-quality/{evaluation_id}` detail getter.
* Strictly read-only: no measurements, no sample generation, no new
  metrics/scoring/aggregation/ranking, no manifest writes, no
  index/cache/db/duplicated storage.

Required chain (final architectural principle):
**M2 tokenizer registry → M16 authoritative sample-quality listing →
persisted `tokenizer_id` filter → thin facade → API route**.

## 2. Baseline

Verified BEFORE any code change (sandbox NOT reset this turn: HEAD
`9fb6630` == FETCH_HEAD — M32 implementation `5d8ef1a` + inventory
`9fb6630` both present and pushed; tree clean except
`ai_model_forge.egg-info/`; production 96/4,002,745/0, venv OK):

* Full suite: **470 passed** serially in **82.80 s** (0 failed,
  0 skipped).
* OpenAPI: **64 paths** (pre-M33 state).
* Production root `/home/user/ai-model-forge-data`: **96 files /
  4,002,745 B / 0 `.tmp`**.
* `m33_pre.sha256` written (96 entries, data-root-relative); diff vs
  `m32_pre.sha256`: **EMPTY** (identical pre-state to certified M32).
* Discovery (live registry + M16 listing, not assumed): registry
  holds exactly ONE tokenizer `99106e3255c5`; model `4a0a871886ef`
  owns 2 measurements, BOTH persisted `tokenizer_id=99106e3255c5`,
  BOTH measuring sample `f8e66f9c7b50` under checkpoint
  `0511de4c7372`, ASC `(created_at, evaluation_id)`:
  `8ff910cf2a9e`, `31a283413c75`; `b5bc905326b6` zero measurements
  (natural model-scoped empty case).
* `SampleQualityEngine` inspected FIRST: `__init__` ALREADY composes
  `self.tokenizers = TokenizerEngine(storage)` — so M33 needed
  **zero** new composition lines.

## 3. Implementation

`ai-model-forge/app/sample_quality.py`

* `list_sample_evaluations_for_tokenizer(model_id, tokenizer_id)`
  directly after `list_sample_evaluations_for_checkpoint` (full
  docstring documenting the grouping unit, VERBATIM
  persisted-identity matching, the recorded-state semantics, global-
  tokenizer/model-scoping, and read-only contract): validates the
  tokenizer through the EXISTING `self.tokenizers.load` handle
  (unknown → `FileNotFoundError`, the M2 registry contract), then
  returns the authoritative `list_sample_evaluations()` filtered by
  the persisted top-level `tokenizer_id` — same objects, same
  `(created_at, evaluation_id)` order, no new sort rule, no
  inference, no substitution, no writes. No `__init__` change
  (handle already existed).

`ai-model-forge/app/engine.py` — thin facade passthrough
`list_sample_evaluations_for_tokenizer` (delegates to
`self.sample_quality.list_sample_evaluations_for_tokenizer`).

`ai-model-forge/app/api.py` — ONE route
`GET /models/{model_id}/sample-quality/by-tokenizer/{tokenizer_id}`:
summary "List sample quality by tokenizer", tag `sample-quality`,
GET-only, `response_model=list[SampleEvaluationRecord]`, 404 mapping
identical to the M19/M20 siblings; registered after the M20
by-checkpoint route and BEFORE the generic
`/sample-quality/{evaluation_id}` detail getter (order verified: M20
by-checkpoint < by-tokenizer < generic). Landing-page M33 bullet +
endpoint `<li>` + sample-quality-section header comment updated.

OpenAPI: **64 → 65 paths** (new path exactly once, GET-only, tag
`sample-quality`). README: new M33 section + test counts 470→475 (2
sites) + `sample_quality.py` layout line extended to
`(M19/M20/M33)`.

No schema change; no new storage, cache, index, worker, or second
registry anywhere.

## 4. Tests

**5 new tests** (475 total = 470 + 5) in
`tests/test_sample_quality.py` (both sides of the existing module:
engine env + HTTP client), reusing existing fixtures/helpers
(`_make_sample`, `env.sq.run`, `_http_env`, `_quality_url`,
`_corpus`); production never touched:

* `test_m33_engine_grouping_parity_order_verbatim` — for each
  (model, tokenizer) pair: parity with the M16 listing filtered by
  persisted `tokenizer_id`; `(created_at, evaluation_id)` order;
  unique ids; `model_id`/`tokenizer_id` on every record; verbatim
  `get_sample_evaluation` parity; the M33-created measurements sit in
  exactly their groups; full discovery-based partition over every
  tokenizer that actually has records (disjoint groups covering the
  listing, each record exactly once).
* `test_m33_engine_empty_404s_cross_model_read_only` — fresh
  tokenizer → `[]`; a model-scoped empty case GUARANTEED by the
  platform vocabulary rule (small model vocab 300 can never hold a
  tok_big-600 record); cross-model isolation (disjoint listings; no
  group ever contains the other model's ids); unknown
  model/tokenizer → `FileNotFoundError`; whole-storage file set
  byte-identical before/after (read-only).
* `test_m33_engine_repeated_calls_identical` — 3 repeats,
  `model_dump(mode="json")` equality.
* `test_m33_api_by_tokenizer_grouping_partition_determinism` — 2
  records under the original tokenizer + 2 under a fresh tok2 (one
  sample measured twice → distinct evaluation ids, same persisted
  tokenizer) + fresh tok3; by-tokenizer == listing filtered, exact
  membership/order, verbatim POST-payload and detail-getter parity, 3
  raw-byte-identical repeats; disjoint partition covering the
  listing; tok3 → `200 []`; filter adds no records (listing still 4).
* `test_m33_api_404s_isolation_regressions_openapi` — unknown model
  404 (alone and combined with unknown tokenizer), unknown
  well-formed + malformed tokenizer 404; second real model →
  `200 []` (cross-model isolation); M16 listing/getter, M18 records
  (`== listing`), M19 by-sample, M20 by-checkpoint, M30
  evaluations-by-tokenizer parity, M32 samples-by-tokenizer parity,
  generic ghost eval id 404 (no route capture); OpenAPI 65, path
  once, GET-only, tag `sample-quality`, `SampleEvaluationRecord`
  items, route order M20 by-checkpoint < by-tokenizer < generic.

No fixture/code issues were discovered this milestone (first focused
run 5/5; module full 45/45; the M31/M32 lesson about lazy
module-state was applied up front: `_m33_state` seeds its own
records and partition assertions are discovery-based).

All 15 prompt-listed areas covered. OpenAPI count assertions updated
**64 → 65 at all 17 `len(spec["paths"]) == 64` sites** across 7 test
modules with adjacent comment ladders extended
(`+ 1 (M33 sample-quality by-tokenizer)`); the ~30 other `== 64`
occurrences in tests are hash lengths (64-hex), vocab sizes and
token counts — inspected individually and deliberately LEFT
UNCHANGED. Grep-verified: 0 `len(spec["paths"]) == 64` remain, 17
`== 65` now.

**Measured:** focused 5/5 (`-k m33`); full module 45/45; full suite
**475 passed** in **85.07 s**, and after stale-`/tmp` cleanup
**475 passed** again in **81.90 s**. compileall clean (app, tests,
`smoke_m33_live.py`). pyflakes clean on all changed/new files
(pre-existing legacy findings only: 2, confined to
`smoke_m9_live.py` / `smoke_m11_live.py`, reported separately).

## 5. Live Smoke

`smoke_m33_live.py` (new, port **8754**, baseline inventory
`/tmp/m33-smoke-baseline-inventory.json`), run against the production
`FORGE_ROOT`: **35/35 checks PASS first run** (repeated 3× total,
exit 0 each). The smoke DISCOVERS the distribution live, then checks
A–W:

* **A** baseline audit 96/4,002,745/0 + SHA256 inventory saved (96) +
  both models resolve + 3 known checkpoints + M11/M16/M18–M32
  pre-state + dashboard hash + OpenAPI 65 pre-state + known pair 200.
* **B** discovery: 1 tokenizer `99106e3255c5`; 2 measurements of
  `4a0a871886ef`, BOTH under it, BOTH measuring sample `f8e66f9c7b50`
  under checkpoint `0511de4c7372`, ASC order exactly
  `8ff910cf2a9e, 31a283413c75`; `b5bc905326b6` none.
* **C** exact parity: response == M16 listing filtered by persisted
  `tokenizer_id` (no missing/extra/duplicate; every record EXACTLY
  ONCE).
* **D** persisted identity VERBATIM on both records + verbatim
  detail-getter parity (loss_nats/perplexity included).
* **E** `(created_at, evaluation_id)` ASCENDING exactly as the M16
  listing.
* **F** three GETs raw-byte-identical.
* **G** `b5bc905326b6` + `99106e3255c5` → `200 []`
  (registry-verified model-scoped empty).
* **H** unknown model → 404 (even with the real tokenizer id).
* **I** unknown well-formed + malformed tokenizer → 404.
* **J** M2 tokenizer registry unchanged (list + get verbatim; unknown
  id still 404).
* **K** M16 generic listing + both detail getters unchanged.
* **L** M18 records listing unchanged (identical to the M16 listing).
* **M** M19 by-sample unchanged (2 under `f8e66f9c7b50`).
* **N** M20 by-checkpoint unchanged (2 under `0511de4c7372`).
* **O** M30 evaluations by-tokenizer unchanged (16).
* **P** M31 comparisons by-tokenizer unchanged (8).
* **Q** M32 samples by-tokenizer unchanged (4).
* **R** M24/M28 evaluation histories unchanged (3/3/3; 16
  by-dataset with filtered-listing parity).
* **S** M26/M29 comparison histories unchanged (025e→6 / 30a8→5 /
  0511→1; 8 by-dataset with filtered-listing parity).
* **T** dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  + full output unchanged.
* **U** M11 workflows, M3 checkpoint registry, M9 policy/probe-suite
  registries, M12/M14 recipe registry unchanged.
* **V** OpenAPI exactly 65, new path once, after M20 by-checkpoint,
  before the generic sample-quality detail route.
* **W** zero drift (final §6).

Server started fresh for the smoke and STOPPED after (uvicorn on
8754, log clean).

## 6. Storage Integrity

* Before: **96 files / 4,002,745 B / 0 `.tmp`**; after: **96 files /
  4,002,745 B / 0 `.tmp`**.
* Smoke internal per-file SHA256: 0 changed / 0 missing / 0 new.
* Final audit `sha256sum -c m33_pre.sha256` from the data root:
  **96/96 OK, 0 non-OK** — every pre-existing production file
  byte-identical. M33 wrote NOTHING to production storage (strictly
  read-only; all test writes went to pytest tmp dirs).

## 7. Regression / Compatibility

* Full suite green before (470) and after (475); no existing test
  weakened; only genuinely affected OpenAPI-count assertions updated
  (64 → 65 ×17 sites) with adjacent comments; unrelated `== 64`
  assertions (hash lengths, vocab sizes, token counts) untouched.
* M2 tokenizer registry contract intact (unknown → 404, list/get
  verbatim — live J).
* M16 sample quality: run/list/get unchanged (live K; module 45/45).
* M18 records / M19 by-sample / M20 by-checkpoint unchanged (live
  L/M/N).
* M30 evaluation-by-tokenizer (16), M31 comparison-by-tokenizer (8),
  M32 sample-by-tokenizer (4) unchanged (live O/P/Q).
* M24/M28 evaluation histories (3/3/3; 16 by-dataset) and M26/M29
  comparison histories (6/5/1; 8 by-dataset) unchanged (live R/S).
* M6 gates, M21/M22/M25 suite-runs, M17 dashboard hash/output,
  M11/M12/M14 registries unchanged (live A4/T/U).
* Generic `/sample-quality/{evaluation_id}` detail getter still 404s
  ghost ids (no route capture — by-tokenizer registered before it);
  OpenAPI 65 with correct route order.
* README + landing page + endpoint list extended; no redundant
  documentation created.

## 8. Commit / Final Status

* Implementation commit: **`M33: add sample-quality history by
  tokenizer`** on branch `arena/01a071e9-code-forge` (files:
  README.md, app/api.py, app/engine.py, app/sample_quality.py,
  tests/test_sample_quality.py + 6 further test modules'
  OpenAPI assertions, smoke_m33_live.py, M33_final_report.md;
  egg-info NOT committed).
* Follow-up inventory commit (repo convention M21–M32):
  `M33: add pre-milestone storage inventory` tracking
  `m33_pre.sha256`.
* Push verified: local == FETCH_HEAD after both commits; working
  tree clean except expected `ai_model_forge.egg-info/`.
* All gates passed in order: baseline (470 @ 82.80 s, OpenAPI 64,
  SHA256 pre-inventory) → focused (5/5; module 45/45) → full suite
  (475 @ 85.07 s) → stale-`/tmp` cleanup (5 dirs removed after
  confirming 0 active processes) → full suite rerun (475 @ 81.90 s)
  → compileall → pyflakes → standalone OpenAPI verification (65,
  order+GET+schema) → live smoke (35/35, zero drift) → final SHA256
  audit (96/96 OK).
* **M33 is certified complete.**

## 9. Next Milestone Prompt

```
# MILESTONE 34 — GATE-DECISION HISTORY BY COMPARISON

Continue the existing **AI Model Forge** project.

M33 is the current certified milestone.

The goal of M34 is to add one small, read-only API capability for
inspecting the existing M6 gate-decision history grouped by the M5
comparison each decision judged.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and
verify the actual M33 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 475 tests passing
* OpenAPI: 65 paths
* M33 sample-quality by-tokenizer endpoint working
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
  at most ONE new `__init__` composition line (only if the gate
  engine lacks a comparison-registry handle — inspect first; the
  gates engine already composes a comparison path for its run —
  verify the exact handle).
* No new gate evaluations, comparisons, metrics, scoring, ranking,
  training/rollback/optimization/HPO, no workers, databases, caches,
  indexes, no new persistence, no automatic decisions, no workflow
  changes, no dashboard redesign, no Gemini, no model improvement
  logic.
* Do not modify production data to manufacture fixtures; the empty
  case comes from the REAL production comparisons with ZERO
  decisions (verified at M33: baa361012e00, d683f9b81195,
  729f9c55ea89 under 4a0a871886ef all have 0 decisions).
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (65 -> 66) and their adjacent stale comments; leave
  unrelated `== 65` assertions (hash lengths etc.) untouched after
  inspecting each.
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 2. M34 OBJECTIVE

Add exactly ONE read-only endpoint:

`GET /api/v1/models/{model_id}/gates/decisions/by-comparison/{comparison_id}`

answering "which immutable M6 gate decisions of this model judged
this M5 comparison?" and nothing else.

Membership rules (identical philosophy to M23/M26/M29-M33):

* The authoritative source is the model's OWN M6 gate-decision
  listing (the same one `GET /models/{id}/gates/decisions` and the
  M23 by-policy grouping use); never re-derive from filenames,
  paths, checkpoint ids, policy ids, hashes or the comparison's
  current content.
* Each persisted GateDecisionRecord carries a top-level
  `comparison_id` (verified live at M33: 9 of the 11 production
  decisions persist one; 2 persist `null` — legacy direct-evaluation
  decisions from before comparisons existed). Match the requested id
  against the persisted `comparison_id` VERBATIM; records whose
  persisted value is `null` belong to NO by-comparison group (never
  match any requested id) and must NOT be dropped from the generic
  listing.
* Each decision appears EXACTLY ONCE.
* Inherit the exact existing M6 listing order (verified at M33:
  deterministic (created_at, decision_id) ASCENDING; do not invent a
  new sort rule).
* Return verbatim GateDecisionRecord payloads (verdict, decision,
  losses, delta, reason included).
* Comparisons are MODEL-SCOPED: validate the comparison id through
  the model's own M5 registry (`ComparisonEngine`/`get_comparison`
  — the same getter `GET /models/{id}/comparisons/{cid}` exposes);
  an unknown comparison id, or one belonging to another model, is
  404 (exactly like the M26 by-checkpoint pattern validated through
  the M3 checkpoint registry).
* A VALID comparison of the model with ZERO decisions -> `200 []`
  (NEVER 404); unknown model or unknown/other-model comparison ->
  404.
* Register the route before the generic gate-decision detail getter
  if one exists, following the M23 pattern; M6 (run, listing) and
  M23 by-policy routes must stay intact and unconfused.
* Strictly read-only: no gate evaluation, no comparison execution,
  no new metrics, no manifest writes, no index/cache/db/duplicated
  storage.
* Required chain: M5 comparison registry validation -> M6
  authoritative decision listing -> persisted comparison_id filter
  -> thin facade -> route.

## 3. PRODUCTION FACTS (verified at M33)

* Model `4a0a871886ef` owns 11 gate decisions in ASCENDING
  (created_at, decision_id) order: 8931835d4af6, baf767bdcbe1,
  9d4facca5153, ef8ba75f9e43, 5b0493c0fbed, 8968151a08bd,
  a82c95374a5a, e33c99f2f8ca, 0dae7b2c456e, 5c86e4494d2f,
  6921d3b29b9d.
* Distribution by persisted comparison_id: fc379bfcb50f -> 4,
  d62f89e97c85 -> 2, 786de08efe4c -> 1, 5c5ff22151ed -> 1,
  d9a62dde016b -> 1, null -> 2.
* The model's 8 M5 comparisons (dataset ee1a716c4573) include
  baa361012e00, d683f9b81195, 729f9c55ea89 with ZERO decisions (the
  natural valid-id empty case -> 200 []).
* Exactly ONE policy-attributed decision exists (m9-live-policy ->
  1; the M23 surface). Model `b5bc905326b6` has NO decisions.
* Dashboard hash (unchanged since M17, re-verified at M33):
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838.
* Storage: 96 files / 4,002,745 B / 0 `.tmp`.

## 4. REQUIRED BEHAVIOR

1. `200` with the exact filtered array for the known pair
   (`4a0a871886ef` + `fc379bfcb50f` -> the 4 decisions).
2. Exact parity with the authoritative M6 listing filtered locally by
   persisted `comparison_id` (no missing, no extra, no duplicate).
3. Persisted identity travels VERBATIM (no substitution/rewriting);
   `null`-comparison decisions never appear in any group.
4. Authoritative (created_at, decision_id) ordering preserved.
5. Deterministic: repeated GETs byte-identical (x3).
6. Valid comparison + zero decisions -> `200 []`
   (baa361012e00 etc.).
7. Unknown model -> 404; unknown or other-model comparison id ->
   404 (well-formed AND malformed ids).
8. Cross-model isolation: each model's group draws only from its own
   listing.
9. M6 gate run/listing, M23 by-policy, M26/M29/M31 comparison
   surfaces, M30/M32/M33 by-tokenizer surfaces byte-identical before
   and after.
10. Zero production storage growth.

## 5. ENGINE IMPLEMENTATION

* Inspect the gates engine FIRST (its `__init__` handles, its
  listing method, the M23 by-policy method for the established
  pattern).
* Add ONE method `list_gate_decisions_for_comparison(model_id,
  comparison_id)` that validates the comparison through the model's
  M5 registry path (compose ONE line in `__init__` only if a
  suitable handle is missing) and filters the authoritative listing
  by the persisted top-level `comparison_id`; the facade and route
  are thin pass-throughs.
* Docstring must explain the grouping unit, that identity is matched
  VERBATIM from the persisted record, and that null-comparison
  legacy decisions belong to no group.

## 6. TESTS

* Target: 475 -> 480 tests (~5 new), in the existing gates test
  modules, REUSING their fixtures/helpers (never modify production;
  never inflate the corpus; follow the established second-record
  fixture pattern if a second comparison is needed).
* Cover: authoritative-filter parity; verbatim identity; exclusion
  of unrelated comparisons; null-comparison exclusion; empty `[]`
  for a valid comparison; 404s (unknown model / unknown + malformed
  comparison id); cross-model isolation (another model's comparison
  id is 404); ordering; determinism; no storage writes; M6/M23
  regressions; OpenAPI 65 -> 66 + route order + ghost id 404.
* Search EVERY hard-coded OpenAPI count (grep `== 65` and
  `len(spec["paths"])`) and update only genuinely affected
  assertions + adjacent stale comments.
* Gates in order: baseline tests -> focused -> full suite -> full
  suite again after stale `/tmp/forge-tests-*` cleanup (verify no
  process first) -> compileall (incl. the new smoke) -> pyflakes
  (report pre-existing findings separately) -> OpenAPI verification
  -> live smoke -> final SHA256 audit; commit only after all gates
  pass.

## 7. LIVE TEST

Write `smoke_m34_live.py` (mirror `smoke_m33_live.py`), port
**8755**, baseline inventory
`/tmp/m34-smoke-baseline-inventory.json`; capture the baseline FIRST
(files/bytes/tmp/SHA256/OpenAPI/test count/registries/decision
distribution by comparison/policy), then checks A-W: A known pair
200; B discover ids; C listing-parity; D persisted id verbatim (+
null-exclusion); E listing order; F >= 3 byte-identical repeats; G
empty `200 []` for a valid zero-decision comparison (baa361012e00);
H unknown model 404; I unknown + malformed comparison 404; J
tokenizer registry unchanged; K M6 decision listing + getters; L M23
by-policy unchanged; M M30 evaluations-by-tokenizer; N M31
comparisons-by-tokenizer + M26 by-checkpoint; O M32
samples-by-tokenizer + M33 sample-quality-by-tokenizer; P M27/M28/
M29 histories; Q dashboard; R policy/probe/suite/recipe registries;
S OpenAPI 66; T storage zero drift; U final summary. Storage must
stay 96/4,002,745/0 with SHA256 before/after (0 changed/new/missing;
investigate + report ANY change honestly).

## 8. FINAL AUDIT

* `sha256sum -c` the pre-milestone inventory from the data root:
  96/96 OK, 0 changed/new/missing.
* Recovery runbook: restore zip -> verify pre-milestone SHA256 ->
  rebuild venv -> reconcile git with the authoritative remote ->
  re-prove baseline; NEVER force-push.

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; no gate evaluation, no comparison
  execution, no scoring, no rankings, no automatic decisions, no
  schema change, no duplication of the authoritative listing, no
  caches/indexes/databases/workers/queues, no
  training/rollback/optimization/HPO/Gemini, no unrelated
  refactoring.
* Report honestly; never certify from a partial run.

## 10. COMMIT / PUSH / REPORT

* Commit message: `M34: add gate-decision history by comparison`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M34_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (decision ids, per-comparison
  counts, ordering, determinism, 404s, cross-model/scoping, SHA256,
  dashboard hash, commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M35 prompt,
  grounded ONLY in facts discovered and verified during M34 (no
  invented ids, counts, or endpoints), defining the next small
  additive read-only capability, including automatic M36 prompt
  generation.
```
