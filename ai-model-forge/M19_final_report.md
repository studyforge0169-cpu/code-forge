# Milestone 19 — Final Report
**Per-sample quality history access (`GET /models/{id}/sample-quality/by-sample/{sample_id}`)**
Date: 2026-09-05 · Project root: `/home/user/ai-model-forge` · Production storage: `/home/user/ai-model-forge-data`

## 1. Milestone objective and scope decision
M19 added **exactly one narrow read-only access path** —
`GET /api/v1/models/{model_id}/sample-quality/by-sample/{sample_id}` — exposing M16
sample-quality history grouped by **one known generated sample** of the model.
It answers only: *"what did we measure for this particular sample?"*

Semantics bound (unchanged from the authoritative instruction):
- model must exist (existing 404); the sample must belong to that model — an
  unknown sample **or another model's sample id** is a 404;
- returns only that sample's evaluation records — **full verbatim
  `SampleEvaluationRecord` payloads**, all fields including `loss_nats` and
  `perplexity`;
- deterministic ordering identical to the authoritative M16/M18 listing:
  `(created_at, evaluation_id)` ASCENDING;
- a valid sample with no measurements is a deterministic `200 []`;
- no derived fields, no statistics, no aggregation, repeated requests
  byte-identical, read-only (zero files written, zero `.tmp`, zero storage growth).

Implementation choice (M19 architecture inspection): **no second engine, no
duplicate manifest parsing.** The M16 engine was extended with one read-only
method, `list_sample_evaluations_for_sample(model_id, sample_id)`, that resolves
the sample through the authoritative M15 `sampling.get_sample(model_id,
sample_id)` (persisted identity — never a filename or graph edge — so unknown or
cross-model samples raise `FileNotFoundError` → 404) and then filters the
authoritative M16 `list_sample_evaluations()` by the persisted `sample_id`.
Because M19's grouping is a strict per-sample filter of M18's full listing, the
generic listing was reused rather than duplicated — exactly as the instruction
allowed.

## 2. Baseline verification (before any code change)
Verified before the first M19 edit, per chain discipline:

| Item | Value |
|---|---|
| Production root | `/home/user/ai-model-forge-data` |
| Files / bytes / `.tmp` | **96 / 4,002,745 / 0** |
| Top-level families | datasets 1, models 2, policies 1, probe-suites 1, samples 1 (4 sample dirs under model `4a0a871886ef`), sample-evaluations 1 (2 eval dirs), suite-runs 10, tokenizers 1, workflow-recipes 7 |
| M16 evals | `evaluation-8ff910cf2a9e`, `evaluation-31a283413c75` (both of sample `f8e66f9c7b50`, checkpoint `0511de4c7372`, loss 6.191012, perplexity 488.340243) |
| M17 dashboard hash | `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` |
| Baseline inventory | saved `/tmp/m19-baseline-inventory.sha256` (96 entries, pre-change) |

Pre-change gates: `compileall` OK; `pyflakes` OK; OpenAPI surface 50 paths.
Pre-change full suite (serial, after the disclosed flake fix below):
**394 passed / 0 failed / 0 errors.** Sandbox dependency loss recurred before the
pre-change gates (pyflakes absent, then fastapi/torch absent); the toolchain was
reinstalled only — an environment issue, no code changed for it.

## 3. Implementation
- **`app/sample_quality.py`** — module docstring extended with M19 semantics;
  new engine method `list_sample_evaluations_for_sample(model_id, sample_id)`
  added directly after `get_sample_evaluation`: ownership check via
  `self.sampling.get_sample(model_id, sample_id)` (raises `FileNotFoundError`
  for an unknown sample or a sample of another model — nothing is ever returned
  for a foreign sample), then filters the authoritative
  `list_sample_evaluations(model_id)` by the persisted `sample_id`; ordering and
  payloads are the M16 record payloads untouched; empty → `[]`.
- **`app/engine.py`** — facade `ModelForge.list_sample_evaluations_for_sample`
  (thin read-only pass-through, after `get_sample_evaluation`).
- **`app/api.py`** — one route
  `GET /models/{model_id}/sample-quality/by-sample/{sample_id}`
  (`response_model=list[SampleEvaluationRecord]`, tag `sample-quality`,
  `FileNotFoundError` → 404), registered **between** the M18 `/records` route and
  the M16 `/{evaluation_id}` getter so the literal segments are never swallowed
  (order verified: `/records` → `/by-sample/…` → `/{evaluation_id}`).
  Interactive HTML docs gained one M19 bullet; the REST endpoint list gained the
  by-sample line. No other route, schema, or behavior touched.
- **README.md** — new "Milestone 19" section; test counts updated 394 → 401
  (two sites). No production artifact was modified at any point.

## 4. Test changes — full disclosure (pre-existing test-file touches)
Three categories, all disclosed:
1. **OpenAPI path-count assertions 50 → 51 — four sites** (surface fact: one new
   documented route): `tests/test_sample_quality.py` lines 826, 1052, 1269 and
   `tests/test_dashboards.py` line 1522, each with an explanatory comment
   (`46 (M15 era) + 3 (M16) + 0 (M17) + 1 (M18) + 1 (M19) = 51`). No other M16/
   M17/M18 assertions were touched.
2. **Latent-flake root-cause fix in `test_corrupt_checkpoint_refused`
   (M16-era, untouched by M18/M19).** Pre-change serial run failed there with
   `ValueError: tokenized split 'validation' is empty`. Root cause: record splits
   are deterministic by record-hash bucket (90/5/5), the 60-record corpus made an
   empty validation bucket a ~4.6% event, and salted process `hash()` varies
   corpus content per interpreter (verified 6192–6779 B across fresh runs). Fixed
   at the source: `_corpus(60, "m16-corrupt")` → `_corpus(400, "m16-corrupt")`
   (P(empty) ≈ 0.95^400 ≈ 1e-9) with a root-cause comment added. **Assertions
   unchanged — the test was not weakened.**
3. **Seven new M19 tests appended** to `tests/test_sample_quality.py`
   (33 functions in the file):
   - `test_engine_by_sample_exact_filter_ordering_and_payloads` — exact filter,
     ASCENDING `(created_at, evaluation_id)` order, full payload parity with the
     authoritative listing, metric values, no cross-sample leakage;
   - `test_engine_by_sample_empty_history` — engine returns `[]`;
   - `test_engine_by_sample_resolution_404s` — unknown model / unknown sample /
     foreign sample raise `FileNotFoundError` (the 404 contract) at engine level;
   - `test_api_by_sample_history_parity_and_metric_values` — HTTP 200 + exactly
     the two records for sample `f8e66f9c7b50`, equality with the M18 records
     listing filtered to that sample **and** with the M16 reference listing,
     `loss_nats=6.191012`, `perplexity=488.340243` present, banned vocabulary
     (aggregate/statistics fields) absent;
   - `test_api_by_sample_empty_unknown_and_model_isolation` — valid unmeasured
     sample → `200 []`; unknown model → 404; cross-model isolation with a crafted
     **schema-valid** foreign sample + foreign evaluation record (the sample
     manifest is a full copy of the real record with identity fields rewritten so
     `sampling.get_sample` resolves it cleanly — never a minimal dict);
   - `test_api_by_sample_deterministic_repeat_and_read_only` — repeated GET
     byte-identical, filesystem byte audit zero delta / zero `.tmp`, M17
     dashboard hash `f48557fe…` preserved, M16/M17/M18 surfaces unchanged;
   - `test_api_openapi_by_sample_route` — OpenAPI documents the route with the
     `SampleEvaluationRecord` array schema.

## 5. Focused tests
`pytest tests/test_sample_quality.py -k "by_sample or bysamp or openapi"`
(9 collected incl. the two pre-existing OpenAPI tests) — **exit 0, 9 passed,
0 failed, 0 errors.**

## 6. Full regression, static gates, OpenAPI
- Full suite (serial, clean `/tmp/pytest-of-user` first): **401 passed /
  0 failed / 0 errors, exit 0** — exactly the predicted 394 + 7.
- `python3 -m compileall -q app` — OK.
- `pyflakes app tests smoke_m18_live.py smoke_m19_live.py` — clean.
- OpenAPI: **51 paths**; by-sample route present as documented `GET`, tag
  `sample-quality`, `200` array of `#/components/schemas/SampleEvaluationRecord`;
  registration order `/records` → `/by-sample/…` → `/{evaluation_id}` verified.

## 7. Live smoke — `smoke_m19_live.py`
Ran against production uvicorn (`FORGE_ROOT=/home/user/ai-model-forge-data`,
port 8741). **Exit 0 — 32/32 checks PASS** (phases A–D):
- **A baseline**: exact 96/4,002,745/0 audit; per-file inventory saved to
  `/tmp/m19-smoke-baseline-inventory.json`; 4 samples / 2 sample-evaluations /
  2 M18 records; M16 listing == M18 records listing; dashboard hash begins
  `f48557fe8ab1`.
- **B new endpoint**: `by-sample/f8e66f9c7b50` → 200 with exactly the two
  records, ids `[8ff910cf2a9e, 31a283413c75]` in ASCENDING order; payload parity
  with the M18 records listing filtered to the sample **and** with the M16
  listing filtered to it; per-record M16 getter parity; loss 6.191012 / ppl
  488.340243 exact; payload field sets identical to M16/M18 records (verbatim,
  no derived statistics); **two repeated GETs raw-byte-identical**; valid
  unmeasured samples `05820e8bc68a`, `4e8463e0df06`, `e2b5166fa549` →
  `200 []`; M16/M17/M18 surfaces unchanged by the reads.
- **C 404 semantics**: unknown model → 404; unknown well-formed sample id
  (`ffffffffffff`) → 404; **cross-model isolation** — real forge sample id
  `f8e66f9c7b50` against model `b5bc905326b6` (exists, owns no samples) → 404;
  M18 records unknown model still 404; over-long id → 404 (no format crash).
- **D final audit**: every pre-existing file byte-identical (0 changed), zero
  new files, 96/4,002,745/0 unchanged, artifact families unchanged, dashboard
  `result_hash` == `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`,
  M16 listing and M18 records listing identical to pre-state.

## 8. Final audit — byte accounting, determinism, read-only proof
Independent audit after server shutdown vs the **pre-change** baseline inventory:

| Metric | Pre-change baseline | Final | Delta |
|---|---|---|---|
| Files | 96 | 96 | **0 new / 0 missing** |
| Bytes | 4,002,745 | 4,002,745 | **0** |
| `.tmp` | 0 | 0 | **0** |
| SHA-256 inventory | `/tmp/m19-baseline-inventory.sha256` (96) | recomputed | **all 96 identical** |

- M17 dashboard hash `f48557fe…3838` unchanged (smoke A6/B11/D5 and tests);
  M16 listing/getter/metrics/hashes/manifests and M18 records endpoint
  byte-identical; M1–M18 behavior intact (full 401-test suite green).
- Determinism: engine test, API test, and two live raw-body repeats all
  byte-identical.
- Read-only: zero files created/changed/removed anywhere in production storage
  across ~30 live requests including all 404 paths (server log verified 200/404
  codes; post-audit hash diff empty).
- Completion gate: baseline verified · architecture inspected · implementation
  complete · focused tests green · full regression green · compileall green ·
  pyflakes green · OpenAPI valid · live smoke green · deterministic · read-only ·
  zero growth · zero `.tmp` · historical byte-identical · M1–M18 intact — **all
  satisfied**.

## 9. Closure state + copy-ready M20 prompt
M19 is closed and certified. Final state: repo at `/home/user/ai-model-forge`
(401 tests / 20 suites, all green), production storage
`96 files / 4,002,745 B / 0 .tmp` byte-identical to the M19 pre-change baseline,
server stopped, `smoke_m19_live.py` exit 0 retained as the M19 live proof,
`smoke_m18_live.py` untouched. No production artifact was modified during M19.

### Copy-ready M20 instruction (proposal — adopt, edit, or replace)

> **Milestone 20 (M20).** Close M18 and M19; the authoritative M20 instruction
> now governs. Add exactly ONE narrow read-only access path:
> `GET /api/v1/models/{model_id}/sample-quality/by-checkpoint/{checkpoint_id}`,
> exposing the M16 sample-quality history of a model grouped by ONE recorded
> checkpoint under which its sample measurements were taken.
>
> Binding requirements:
> - Verify the model exists; verify the checkpoint belongs to that model via the
>   existing M3 checkpoint registry (unknown checkpoint, or a checkpoint id of
>   another model → 404); return only evaluation records whose recorded
>   `checkpoint_id` equals the requested one, full verbatim
>   `SampleEvaluationRecord` payloads (all fields incl. `loss_nats`/`perplexity`),
>   deterministic ordering identical to authoritative M16/M18 `(created_at,
>   evaluation_id)` ASCENDING; `[]` when the checkpoint has no measurements;
>   existing 404 for an unknown model.
> - Engine approach: extend the M16 engine with read-only
>   `list_sample_evaluations_for_checkpoint(model_id, checkpoint_id)` filtering
>   the authoritative M16 `list_sample_evaluations()` by the persisted recorded
>   `checkpoint_id` (never filenames/graph edges); facade pass-through on
>   `ModelForge`; no second engine, no duplicate manifest parsing.
> - Strict isolation: never return another model's/checkpoint's/sample's/
>   dataset's/probe-suite's records.
> - Do not change: M16 loss/perplexity/hashes/manifests/endpoints, M18 records,
>   M19 by-sample, M17 dashboard behavior; M17 dashboard hash
>   `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` stays
>   byte-identical. No derived fields/statistics (no per-checkpoint aggregation,
>   no cross-checkpoint comparison, no ranking); repeated requests byte-identical;
>   read-only (zero files, zero new, zero `.tmp`, zero storage growth).
> - Focused tests: production checkpoint `0511de4c7372` of model
>   `4a0a871886ef` returns the two known M16 measurements (loss 6.191012,
>   perplexity 488.340243; ids `8ff910cf2a9e` + `31a283413c75`); checkpoint
>   isolation (a checkpoint id with no measurements → `[]`, incl. the model's
>   other verified checkpoints); model isolation; unknown model/checkpoint/
>   foreign checkpoint → 404; determinism; read-only; M16/M17/M18/M19 parity;
>   OpenAPI valid (surface count 51 → 52, assertions updated with disclosure).
> - Live smoke `smoke_m20_live.py` against
>   `FORGE_ROOT=/home/user/ai-model-forge-data` (M19 final baseline verified
>   first: 96/4,002,745/0), HTTP 200 + exactly two records for checkpoint
>   `0511de4c7372`, parity with the M18 listing filtered to that checkpoint,
>   both eval ids, byte-identical repeat, unknown model/checkpoint → 404, valid
>   empty checkpoint → 200 + `[]`, failure requests storage-neutral, final audit
>   with all historical files byte-identical, zero new files, zero `.tmp`, zero
>   growth, M17 dashboard hash unchanged, M18/M19 listings unchanged, M16
>   records unchanged; non-zero exit on any mismatch.
> - Completion gate (all before declaring done): baseline verified + architecture
>   inspected + implementation complete + focused tests green + full regression
>   green + compileall green + pyflakes green + OpenAPI valid + live smoke green
>   + deterministic + read-only + zero growth + zero `.tmp` + historical
>   byte-identical + M1–M19 intact + exact nine-section report + complete M21
>   prompt.
