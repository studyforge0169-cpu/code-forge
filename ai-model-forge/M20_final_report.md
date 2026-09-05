# Milestone 20 — Final Report
**Per-checkpoint quality history access (`GET /models/{id}/sample-quality/by-checkpoint/{checkpoint_id}`)**
Date: 2026-09-05 · Project root: `/home/user/ai-model-forge` · Production storage: `/home/user/ai-model-forge-data`

## §1 Summary
M20 delivered **exactly one narrow read-only access path** —
`GET /api/v1/models/{model_id}/sample-quality/by-checkpoint/{checkpoint_id}` —
answering only *"which immutable sample-quality evaluations belong to this
checkpoint?"*

- Model must exist; the checkpoint must be registered in that model's M3
  checkpoint registry — an unknown checkpoint **or a checkpoint id belonging to
  another model** is a 404.
- Returns only that checkpoint's evaluation records: **full verbatim
  `SampleEvaluationRecord` payloads** (all fields incl. `loss_nats` /
  `perplexity`) in the authoritative M16/M18/M19 `(created_at, evaluation_id)`
  ASCENDING order.
- A valid checkpoint with no measurements is a deterministic `200 []`.
- No derived fields, no statistics, no checkpoint comparison/judgment;
  repeated requests byte-identical; read-only (zero files written, zero
  `.tmp`, zero storage growth).
- Implementation reuses the M16 engine + M3 registry: no second engine, no new
  parser, no new registry, no new persistence.

## §2 Baseline verification
All checks ran **before any code change**, on the actual workspace (not assumed
from the M19 report):

| Item | Result |
|---|---|
| Production root | `/home/user/ai-model-forge-data` — files **96 / 4,002,745 B / `.tmp` 0** |
| SHA256 inventory | `/tmp/m20-baseline-inventory.sha256` (96 entries, all families) |
| `/tmp` capacity | 993 MB tmpfs, 306 MB free (sufficient; later event §6) |
| compileall / pyflakes | OK / OK |
| OpenAPI | **51 paths**; by-sample + records present; by-checkpoint **absent** (404 pre-change, verified live) |
| Full suite (serial) | **401 passed / 0 failed / 0 errors**, exit 0 — M19 baseline reproduced exactly; the M19 flake did not recur |
| M19 by-sample (live) | 200, exactly `[8ff910cf2a9e, 31a283413c75]` |
| M18 records / M16 listing (live) | 200, same two records, payload-identical |
| Checkpoint ownership (live) | both records carry `checkpoint_id=0511de4c7372`, `sample_id=f8e66f9c7b50`, loss 6.191012, perplexity 488.340243 |
| Model checkpoints (live) | `4a0a871886ef`: `025e6d8d8f15`, `0511de4c7372`, `30a8bc5b82ab`; `b5bc905326b6`: none |
| M17 dashboard hash (live) | `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` |
| Architecture findings | M16 `SampleQualityEngine` already exposes the authoritative listing + M3 `TrainingEngine.get_checkpoint(model_id, ckpt_id)` is the model-scoped registry getter (FileNotFoundError ⇒ 404) — the M20 filter can reuse both; no new abstraction needed. Route-order constraint re-verified: literal `by-sample`/`by-checkpoint`/`records` segments must precede `/{evaluation_id}`. Observation: an **empty `tmp/` directory** exists in the production root, created by `Storage.__init__` (`storage.py:125`) on server start (since earlier milestone smokes). It contains **zero files**, has never affected any file-based audit, and no file in it is ever written by read-only endpoints. |

## §3 Implementation
Every modified file (repository only — **no production artifact was ever
touched**):

1. **`app/sample_quality.py`** — module docstring extended with M20 semantics;
   new engine method `list_sample_evaluations_for_checkpoint(model_id,
   checkpoint_id)`, placed after the M19 method:
   - ownership first: `self.training.get_checkpoint(model_id, checkpoint_id)`
     (M3 registry; raises `FileNotFoundError` for an unknown checkpoint or a
     checkpoint registered under another model — model-scoped paths, nothing
     inferred from filenames) → 404 at the API;
   - then filters the authoritative `list_sample_evaluations(model_id)` by the
     **persisted** `checkpoint_id` recorded in each `SampleEvaluationRecord`;
   - ordering/payloads are the M16 records untouched; empty → `[]`; never
     writes.
2. **`app/engine.py`** — facade `ModelForge.list_sample_evaluations_for_checkpoint`
   (thin read-only pass-through, after the M19 facade).
3. **`app/api.py`** — one route
   `GET /models/{model_id}/sample-quality/by-checkpoint/{checkpoint_id}`
   (`response_model=list[SampleEvaluationRecord]`, tag `sample-quality`,
   `FileNotFoundError` → 404) registered between the M19 `/by-sample` route and
   the M16 `/{evaluation_id}` getter (order verified: `records` 1115 →
   `by-sample` 1141 → `by-checkpoint` 1166 → `/{evaluation_id}` 1194 — literals
   can never be swallowed as evaluation ids). Interactive HTML docs gained one
   M20 bullet; REST list gained one line. No other route/schema touched.
4. **`tests/test_sample_quality.py`** — three OpenAPI path-count assertions
   51→52 (lines 827, 1054, 1272; comments updated) + 7 new M20 tests appended
   (33→40 functions).
5. **`tests/test_dashboards.py`** — one OpenAPI path-count assertion 51→52
   (line 1522, comment updated).
6. **`README.md`** — "Milestone 20" section; test counts 401→408 (two sites).
7. **`smoke_m20_live.py`** — new live smoke (30 checks, phases A–D).

### Full disclosure — pre-existing test-file touches
- **Four** OpenAPI path-count assertions updated 51→52 (three in
  `tests/test_sample_quality.py`, one in `tests/test_dashboards.py`) — factual
  surface change, commented, no other assertion in any M16/M17/M18/M19 test was
  altered.
- **No repeat of the M19 flake fix** was needed: the pre-change baseline (401)
  reproduced cleanly on the first run; the M19 root-cause fix held.
- One M20-internal test bug surfaced in full regression and was fixed within
  the milestone (see §5): my new engine filter test initially asserted that a
  checkpoint's history equals only the records created inside that test,
  forgetting the shared module env already holds earlier M16-era records under
  `big_ckpts[0]`. Corrected to assert exact full-listing parity + containment +
  exclusivity (assertions strengthened, nothing weakened).
- Environmental note: the first post-change full run aborted with
  `OSError: [Errno 28] No space left on device` because `/tmp` (tmpfs) had
  accumulated legacy `forge-tests-*` session roots (~140 MB each) from
  `tempfile.mkdtemp` in `tests/conftest.py` across earlier milestone sessions,
  plus the aborted run's `pytest-of-user`. This was **pure environment disk
  pressure, not a code/test defect**: legacy roots were deleted (no test data
  lost — every run creates its own root) and the rerun passed 408/408. No code
  changed for it.

## §4 Focused tests
Seven new tests appended to `tests/test_sample_quality.py` (mirroring the M19
conventions; engine tests ordered so the empty-checkpoint assertion precedes
the first measurement under that checkpoint):

1. `test_engine_by_checkpoint_empty_history` — valid registered checkpoint of
   the big model with zero measurements → `[]`.
2. `test_engine_by_checkpoint_exact_filter_ordering_and_payloads` — filter
   equals the full authoritative listing filtered by the persisted
   checkpoint_id; every record carries the requested checkpoint; getter parity
   per record; a second checkpoint's measurement never appears in the first
   checkpoint's history (and vice versa); `(created_at, evaluation_id)`
   ASCENDING.
3. `test_engine_by_checkpoint_resolution_404s` — unknown model / unknown
   checkpoint / a checkpoint registered under another model → `FileNotFoundError`.
4. `test_api_by_checkpoint_history_parity_and_metric_values` — HTTP 200 +
   exactly the records of the model's measured checkpoint; equality with the
   M18 records listing and the M16 engine listing filtered by that checkpoint;
   `loss_nats`/`perplexity` present; banned vocabulary absent (average,
   aggregate, rank, score, trend, verdict, mean, delta, statistic, best,
   worst); M19 by-sample of the same sample unchanged.
5. `test_api_by_checkpoint_empty_unknown_and_model_isolation` — valid
   unmeasured checkpoint → `200 []`; unknown model / unknown checkpoint → 404;
   model B (no checkpoints) + A's real checkpoint id → 404; then a **crafted
   schema-valid checkpoint-registry manifest under model B** (verbatim copy of
   A's real manifest with identity fields rewritten — M19-craft precedent) with
   one crafted measurement: B sees exactly its own record under its own
   checkpoint, A never sees it, and A answers 404 for B's checkpoint id even
   though that id exists in storage (ownership rule). Crafted dirs removed in
   `finally`.
6. `test_api_by_checkpoint_deterministic_repeat_and_read_only` — repeated GET
   byte-identical; filesystem byte-audit around the reads (zero new/changed
   files, zero `.tmp`); M16 listing, M18 records, M19 by-sample and dashboard
   `result_hash` unchanged.
7. `test_api_openapi_by_checkpoint_route` — route documented with the array-of-
   `SampleEvaluationRecord` schema, tag `sample-quality`, neighbouring routes
   incl. both literals still present, surface count 52.

Focused run `-k "by_checkpoint or openapi"`: **exit 0, 11 passed / 0 failed /
0 errors** (7 new + 4 pre-existing OpenAPI tests).

## §5 Full regression and live smoke
- **Full suite** (serial, clean `/tmp/pytest-of-user`): **408 collected, exit
  0, zero failure/error marks** (401 + 7; pytest 9.0.3's `-q` green count line
  is not captured to the redirected log, so the count is taken from the exit
  code + progress marks + empty summary — 408 items, all green).
- First attempt aborted on `/tmp` exhaustion (see §3 disclosure); rerun after
  environmental cleanup green. M1–M19 behavior intact (all pre-existing tests
  unchanged except the four disclosed 51→52 counts).
- **compileall** OK (app + tests). **pyflakes** OK (app, tests, both smokes).
- **OpenAPI**: 52 paths; by-checkpoint documented as `GET` (tag
  `sample-quality`) with `200 → array of
  `#/components/schemas/SampleEvaluationRecord``; by-sample and records still
  documented.
- **Live smoke** `smoke_m20_live.py` against production uvicorn
  (`FORGE_ROOT=/home/user/ai-model-forge-data`, port 8741): **exit 0 — 30/30
  PASS** (phases A–D):
  - A: exact 96/4,002,745/0 audit + inventory saved
    (`/tmp/m20-smoke-baseline-inventory.json`); 4 samples / 2 sample-
    evaluations / 2 M18 records / 3 checkpoints; M18 records == M16 listing ==
    M19 by-sample; dashboard hash `f48557fe…`.
  - B: `by-checkpoint/0511de4c7372` → 200 with exactly `[8ff910cf2a9e,
    31a283413c75]` ASCENDING; every record carries `0511de4c7372`; payload
    parity with M18-records-filtered, M16-listing-filtered and the M19
    by-sample response; per-record M16 getter parity; loss 6.191012 /
    perplexity 488.340243 exact; payload field sets identical to M16/M18/M19
    (verbatim, no derived statistics); **two repeated GETs raw-byte-identical**;
    valid unmeasured checkpoints `025e6d8d8f15` and `30a8bc5b82ab` → `200 []`;
    M16/M17/M18/M19 surfaces unchanged by the reads.
  - C: unknown model → 404; unknown well-formed checkpoint (`ffffffffffff`) →
    404; **cross-model isolation** — model `b5bc905326b6` + the forge's real
    checkpoint id `0511de4c7372` → 404; M19 by-sample unknown model still 404;
    malformed id → 404 (no crash).
  - D: every pre-existing file byte-identical (0 changed), zero new files,
    96/4,002,745/0 unchanged, M16 listing / M18 records / M19 by-sample
    byte-identical to pre-state, dashboard hash preserved.
  - Server stopped cleanly; server log verified all intended 200/404 codes.

## §6 Storage and determinism audit
Independent final audit (post-shutdown, vs the **pre-change** baseline):

| Metric | Pre-change baseline | Final | Delta |
|---|---|---|---|
| Files | 96 | 96 | **0 new / 0 missing** |
| Bytes | 4,002,745 | 4,002,745 | **0** |
| `.tmp` | 0 | 0 | **0** |
| SHA-256 | `/tmp/m20-baseline-inventory.sha256` (96) | recomputed | **all 96 identical** |

- Determinism: two live raw-body repeats byte-identical (B10/B11) + engine/API
  test repeats byte-identical; ordering `(created_at, evaluation_id) ASCENDING`
  asserted against the sorted key.
- Read-only: ~25 live requests including every 404 path wrote nothing; the
  post-audit hash diff is empty; `.tmp` count 0; the empty `tmp/` staging
  directory (see §2) contains zero files before and after and is not part of
  any file count.
- M17 dashboard `result_hash` `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
  unchanged; M16 listing/getter/metrics/hashes/manifests, M18 records, M19
  by-sample byte-identical (smoke A5/B13/D4 + tests).

## §7 Honest limitations
M20 measures nothing and judges nothing. It:
- does **not** aggregate, average, rank, score, compare, or trend across
  checkpoints, samples, or models — it only filters already-recorded immutable
  M16 measurements by one persisted checkpoint identity;
- does **not** infer checkpoint/sample relationships — it uses only the
  persisted `checkpoint_id` recorded in each `SampleEvaluationRecord` and the
  M3 registry for ownership;
- does **not** say whether any checkpoint is "better" or "worse": two records
  under one checkpoint merely mean that sample was measured twice against that
  state; `loss_nats`/`perplexity` remain likelihood measurements under one
  causal-LM objective, never an overall quality score;
- does **not** verify checkpoint weights integrity (the M3 `verify_checkpoint`
  path loads weights; this read path touches manifests only, so a corrupt
  weights file is irrelevant to a listing that returns recorded facts);
- does **not** train, optimize, select, recommend, or write anything; it
  performs no semantic/safety/human evaluation and calls no external AI.
- Empty-history `[]` means "no recorded measurements under that checkpoint in
  this model's sample-evaluation family" — it says nothing about whether the
  model was ever trained against that checkpoint otherwise.

## §8 Scope compliance
- **M1–M19 behavior intact**: full 408-test suite green (only the four
  disclosed 51→52 OpenAPI surface counts touched); M16 loss/perplexity/hashes/
  manifests/endpoints, M18 records, M19 by-sample, M17 dashboard
  (`f48557fe…3838`) byte-identical before and after (tests + smoke A/B/D);
  production artifacts untouched — all 96 pre-change files byte-identical,
  zero new, zero missing, zero growth, zero `.tmp`.
- **No forbidden scope introduced**: no training/fine-tuning/LoRA/DPO/RL/
  distillation/quantization/HPO, no automatic selection, no quality scoring/
  ranking/trend analysis, no Gemini/external AI, no workflow/recipe/dashboard/
  frontend changes, no database/indexes/caches/workers/queues/scheduling/
  distributed execution, no new production storage. M20 is ONE read-only
  endpoint.
- Completion gate: baseline verified · architecture inspected · implementation
  complete · focused tests green · full regression green · compileall green ·
  pyflakes green · OpenAPI valid (52) · live smoke green (30/30) ·
  deterministic · read-only · zero production storage growth · zero `.tmp` ·
  all historical production files byte-identical · M1–M19 intact · nine-section
  report with complete M21 prompt — **all satisfied**.

## §9 Copy-ready M21 prompt
M20 is closed and certified. Final state: repo `/home/user/ai-model-forge`
(408 tests / 20 suites green), production storage `96 files / 4,002,745 B /
0 .tmp` byte-identical to the M20 pre-change baseline, servers stopped,
`smoke_m20_live.py` exit 0 retained as the M20 live proof, earlier smokes
untouched.

### Copy-ready M21 instruction (proposal — adopt, edit, or replace)

> **Milestone 21 (M21).** Close M19 and M20; the authoritative M21
> instruction now governs. Add exactly ONE narrow read-only access path:
> `GET /api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}`, exposing
> the M10 multi-probe suite-run history of a model grouped by ONE named
> probe suite (M9 registry).
>
> Binding requirements:
> - Inspect the M9/M10 implementations first: verify the probe-suite
>   registry getter (`policies`/`probe-suites` family) and the model-scoped
>   M10 suite-run listing already exposed by the engine/facade; verify the
>   persisted field each SuiteRunRecord uses to record its suite identity
>   and the authoritative listing order — reuse them, do not re-parse or
>   build a second listing.
> - Verify the model exists and the suite exists in the M9 probe-suite
>   registry (unknown model or unknown suite -> 404); return only that
>   model's suite-run records whose persisted recorded suite identity
>   matches the requested suite — full verbatim SuiteRunRecord payloads —
>   in the authoritative M10 listing order (deterministic; do not invent a
>   new order); `[]` when the model has no recorded runs under that suite;
>   existing 404 for unknown models. Records stay model-scoped: a run
>   recorded under another model is unreachable. Nothing inferred from
>   filenames; use the persisted identities.
> - Do not change: M9 definitions, M10 run endpoints/records/hashes, M16
>   loss/perplexity/records, M17 dashboard (hash
>   `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
>   stays byte-identical), M18 records, M19 by-sample, M20 by-checkpoint.
>   No derived fields/statistics (no per-suite scoring, no pass-rate
>   recomputation, no ranking, no aggregation); repeated requests
>   byte-identical; read-only (zero files, zero new, zero `.tmp`, zero
>   storage growth).
> - Focused tests: production model `4a0a871886ef` + production suite
>   `m9-live-suite` returns all 10 recorded suite-runs
>   (`2af508035191`, `2d203398304b`, `2dc9f7dff400`, `4cb388a21ce2`,
>   `5684649d0ced`, `8f8aee834c9f`, `aae8e8a8ba9c`, `bb114d2ce42e`,
>   `d9742459017b`, `e4b1c2a7fb2d`) in the exact authoritative order;
>   exact suite filtering; payload parity with the M10 listing; empty
>   history `[]` for a real registered suite with no runs under the model
>   (create none in production — use a real registered suite without runs,
>   or a real model without runs, per what production offers); unknown
>   model -> 404; unknown suite -> 404; model isolation (another model's
>   runs never appear); determinism; read-only; M9/M10/M16/M17/M18/M19/M20
>   parity; OpenAPI valid (surface count 52 -> 53; update the path-count
>   assertions with disclosure comments — the exact sites will be
>   discovered by grep `== 52`).
> - Route registration order: the literal `/by-suite/` route must be
>   registered before any `/suite-runs/{run_id}` parameter route (follow
>   the M18/M19/M20 precedent inside the suite-runs route group).
> - Live smoke `smoke_m21_live.py` against
>   `FORGE_ROOT=/home/user/ai-model-forge-data` (M20 final baseline verified
>   first: 96/4,002,745/0), HTTP 200 + exactly 10 records for model
>   `4a0a871886ef` / suite `m9-live-suite`, payload parity with the M10
>   listing filtered to that suite, byte-identical repeat, unknown
>   model/suite -> 404, valid empty combination -> 200 + `[]`, failure
>   requests storage-neutral, final audit with all historical files
>   byte-identical, zero new files, zero `.tmp`, zero growth, M17 dashboard
>   hash unchanged, M16/M18/M19/M20 listings unchanged; non-zero exit on
>   any mismatch.
> - Run: baseline verification (exact file/byte/`.tmp` audit + full SHA256
>   inventory + `/tmp` capacity + serial full suite + compileall + pyflakes
>   + OpenAPI) BEFORE any code change; never modify production artifacts to
>   pass a baseline; never weaken tests; never run pytest suites
>   concurrently; clean `/tmp/pytest-of-user` first; if an existing test
>   exposes genuine latent nondeterminism, diagnose and disclose any
>   root-cause fix prominently and do not repeat earlier fixes unless the
>   baseline independently demonstrates the same problem.
> - Forbidden scope: no training/retraining/SFT/LoRA/QLoRA/PEFT/DPO/RL/
>   RLHF/RLAIF/distillation/quantization/HPO, no automatic model or
>   checkpoint selection, no quality scoring, no semantic/safety/human
>   evaluation, no rankings/leaderboards/trends, no Gemini/external AI,
>   no workflow/recipe changes, no dashboard redesign/frontend work, no
>   database/indexes/caches/workers/queues/scheduling/distributed
>   execution, no new production storage, no suite execution. M21 is ONE
>   read-only endpoint.
> - Honest limitations section: exactly what M21 does not measure, infer,
>   score, aggregate, or automate (e.g., it answers which recorded runs
>   used a suite — not how the model performed, and never re-judges
>   recorded verdicts).
> - Completion gate (all before declaring done): baseline verified +
>   architecture inspected + implementation complete + focused tests green
>   + full regression green + compileall green + pyflakes green + OpenAPI
>   valid + live smoke green + deterministic + read-only + zero growth +
>   zero `.tmp` + historical byte-identical + M1–M20 intact + exact
>   nine-section report (§1 Summary, §2 Baseline verification, §3
>   Implementation, §4 Focused tests, §5 Full regression and live smoke,
>   §6 Storage and determinism audit, §7 Honest limitations, §8 Scope
>   compliance, §9 Copy-ready M22 prompt that itself requires the same
>   gates, baseline verification, forbidden scope, honest limitations and
>   automatically generates the next complete copy-ready milestone
>   prompt).
