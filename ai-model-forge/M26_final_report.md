# MILESTONE 26 — COMPARISON HISTORY BY CHECKPOINT — FINAL REPORT

## 1. Objective

Add exactly ONE new read-only endpoint to the existing **AI Model Forge**:

`GET /api/v1/models/{model_id}/comparisons/by-checkpoint/{checkpoint_id}`

answering *"which immutable M5 comparisons involve this checkpoint
state?"* with **two-sided semantics**: a comparison has `state_a` and
`state_b`, each a persisted `ComparisonSide` (`state_kind: EvalStateKind`,
nullable `checkpoint_id`); the record is returned when EITHER side
records `state_kind == "checkpoint"` with the requested id. Required
behavior, all delivered:

- **Never** treat a checkpoint_id-looking value on a non-checkpoint side
  as a match — current-state sides keep `checkpoint_id=null` and never
  match; nothing is inferred from hashes, timestamps or filenames.
- **Same-checkpoint A=B rule**: such a record appears exactly ONCE
  (dedup by comparison identity, not by hashes/timestamps/side
  combinations/paths).
- Checkpoint validation through the M3 registry
  (`TrainingEngine.get_checkpoint`): unknown model, unknown checkpoint,
  OR another model's checkpoint id → 404 (model-scoped ownership; no raw
  filesystem existence checks).
- Reuse the authoritative M5 `list_comparisons` ordering
  `(created_at, comparison_id)` ASCENDING; verbatim `ComparisonRecord`
  payloads (verdict/per-side losses included).
- Valid checkpoint with no comparisons → `200 + []` (never 404 — "no
  comparisons" is not "does not exist").
- Read-only: no training, rollback, optimization, HPO, ranking, new
  scoring, new comparison metrics, workers, databases, caches, indexes,
  new persistence, parallel registry or duplicate manifest parsing.

**Honesty note (prompt §6 data discrepancy):** the M26 prompt's §6
production counts were TRANSPOSED relative to the authoritative
manifests: the prompt claimed `0511de4c7372` → 5 and `30a8bc5b82ab` → 1
comparisons, while the persisted records (read directly from
`models/4a0a871886ef/comparisons/comp-*/manifest.json` and confirmed by
the live M5 listing) give `0511de4c7372` → **1** and `30a8bc5b82ab` →
**5** (`025e6d8d8f15` → 6 matched). Per the prompt's own rule
("Do not invent IDs from these counts" — resolve authoritative data), M26
was implemented and tested against the authoritative facts; all
expectations below use the authoritative counts.

## 2. Baseline

Verified BEFORE any code change:

- HEAD = **`5af8d8c`** (`M25: add suite-run history by checkpoint`),
  branch `arena/01a071e9-code-forge`, working tree clean.
- Production root `/home/user/ai-model-forge-data`: **96 files /
  4,002,745 bytes / 0 `.tmp`**.
- Full suite serially: **435 passed** (92.7 s).
- `compileall` clean; `pyflakes` clean (only the long-standing notes in
  legacy `smoke_m9_live.py` / `smoke_m11_live.py`).
- OpenAPI: **57 paths**, the new `comparisons/by-checkpoint` path
  absent.
- `m26_pre.sha256` created at the repo root; `diff` vs `m25_pre.sha256`
  → **empty** (M25 certification state intact, zero drift).
- Authoritative comparison facts read directly from the 8 production
  manifests (`models/4a0a871886ef/comparisons/`, in `(created_at,
  comparison_id)` order):

| comparison | verdict | state_a | state_b |
|---|---|---|---|
| `fc379bfcb50f` | improved | 30a8bc5b82ab | 025e6d8d8f15 |
| `baa361012e00` | improved | 30a8bc5b82ab | 025e6d8d8f15 |
| `d683f9b81195` | improved | current | 025e6d8d8f15 |
| `786de08efe4c` | unchanged | 025e6d8d8f15 | 025e6d8d8f15 (A=B) |
| `5c5ff22151ed` | unchanged | current | 0511de4c7372 |
| `d9a62dde016b` | regressed | 025e6d8d8f15 | 30a8bc5b82ab |
| `729f9c55ea89` | unchanged | 30a8bc5b82ab | 30a8bc5b82ab (A=B) |
| `d62f89e97c85` | regressed | 025e6d8d8f15 | 30a8bc5b82ab |

Membership on either side (unique): `025e6d8d8f15` → **6**
[fc379bfcb50f, baa361012e00, d683f9b81195, 786de08efe4c, d9a62dde016b,
d62f89e97c85]; `30a8bc5b82ab` → **5** [fc379bfcb50f, baa361012e00,
d9a62dde016b, 729f9c55ea89, d62f89e97c85]; `0511de4c7372` → **1**
[5c5ff22151ed]. Side totals: 14 checkpoint sides + 2 current sides; 4
records involve BOTH 025e and 30a8 (union = 8 unique comparisons, never
12 slots). No production checkpoint has zero comparisons → the `200 []`
case is covered by a unit/API fixture (second trained model), as the
prompt allows.

## 3. Implementation

Reuse, not a second engine — 44 lines of engine code, thin facade,
thin route:

- **`app/comparison.py`** — `ComparisonEngine.
  list_comparisons_for_checkpoint(model_id, checkpoint_id)` in a new
  "M26: per-checkpoint access (read-only)" section directly after
  `get_comparison`. It (1) validates existence + ownership via the
  already-composed `self.training.get_checkpoint` (M3 registry — NO new
  wiring; a checkpoint id belonging to another model is not registered
  under this model and raises `FileNotFoundError` exactly like an
  unknown one), then (2) returns the authoritative
  `self.list_comparisons(model_id)` filtered by an `involves()` check
  over BOTH persisted sides (`state_kind == EvalStateKind.CHECKPOINT and
  checkpoint_id == id`). Because the listing holds each record exactly
  once, the A=B dedup falls out naturally — the response is a unique
  list of comparison identities; ordering is inherited from the M5
  `(created_at, comparison_id)` sort; `[]` when nothing matches;
  never writes. No changes to `run_comparison`, the manifest format, or
  the storage layout.
- **`app/engine.py`** — facade `ModelForge.list_comparisons_for_checkpoint`
  directly after `get_comparison` (delegation + docstring; import of
  `ComparisonRecord` added to the existing `from .schemas import (...)`).
- **`app/api.py`** — route
  `GET /models/{model_id}/comparisons/by-checkpoint/{checkpoint_id}`
  (`response_model=list[ComparisonRecord]`, tags `["comparison"]`,
  `FileNotFoundError → 404`) placed after `list_comparisons` and
  **before** the generic `GET /models/{model_id}/comparisons/
  {comparison_id}` (the literal `by-checkpoint` segment is not a
  comparison id). Comparison section header extended; landing-page M26
  bullet + one endpoint-list `<li>` added.
- **`README.md`** — Milestone 26 section (two-sided semantics, A=B once,
  current-side exclusion, `200 []`, 404s, read-only, registry-first
  ownership, route-before-detail, M5/M6/M16–M25 unchanged); test-count
  435 → 440 in Quickstart + Layout; `comparison.py` layout line extended
  with the M26 grouping.
- No new modules, schemas, storage structures, caches, indexes,
  workers, databases or background processes.

## 4. Tests

Five new tests (fixture-based, prompts §16 items 5–14; items 1–4 are
production-facts checks executed by the live smoke in §5):

- `tests/test_comparison.py` (engine, 3) — shared module env extended by
  a cached `_m26_env` (a cross-checkpoint A/B, a same-checkpoint A=B, a
  current-vs-checkpoint and a current-vs-current comparison on the
  module model, plus a second fully trained model whose checkpoint has
  zero comparisons):
  `test_m26_engine_filters_by_persisted_sides_and_model` (parity with
  the live M5 listing filtered by the persisted sides for all three
  checkpoints, deterministic order, involvement on ≥1 side, unique
  identities, verbatim `get_comparison` parity, A=B exactly once,
  current-vs-checkpoint exactly once under its checkpoint,
  current-vs-current under none),
  `test_m26_engine_empty_404s_cross_model_and_read_only` (`[]` for the
  valid-but-empty checkpoint; `FileNotFoundError` cross-model both
  directions, unknown model, unknown checkpoint; comparison-manifest
  directory byte-set unchanged by reads),
  `test_m26_engine_repeated_calls_identical` (4 identical calls).
- `tests/test_comparison_api.py` (API, 2):
  `test_m26_api_by_checkpoint_grouping_dedup_and_determinism` (four
  comparisons via `POST /comparisons/run`; exact ids/order for both
  checkpoints; A=B once; current-vs-current in neither; JSON + raw-byte
  identical repeats; parity with the filtered `GET /comparisons`
  listing; `200 + []` for a fresh model's checkpoint) and
  `test_m26_api_404s_isolation_and_prior_surfaces` (unknown model /
  unknown checkpoint / cross-model 404s; M5 listing + detail unchanged
  and not shadowed; M6 decisions + by-policy ghost 404; M20/M24/M25
  by-checkpoint surfaces 200; M21/M22 suite registration + empty
  by-suite/summary; OpenAPI: path present exactly once, GET-only,
  `comparison` tag, `array` of `$ref ComparisonRecord`, **58 paths**).
- OpenAPI count assertions 57 → 58 updated at the exact 10 sites only
  (`test_sample_quality.py` ×4, `test_dashboards.py` ×1,
  `test_suite_runs.py` ×3, `test_gates_api.py` ×1,
  `test_evaluation_api.py` ×1); grep-verified 0 `== 57` remain. No
  existing test weakened or removed.

Honest failure log during development (all first failures reported,
none hidden):

1. First `pyflakes` of the appended engine tests FAILED: undefined
   `_WORDS`/`_bytes`/`_word_soup` (invented upload-helper names) and
   missing `EvalStateKind` import — fixed by reusing the module's real
   `_domain_bytes(TAIL_A, 240)` helper and extending the existing
   schemas import; re-ran pyflakes before pytest.
2. First combined run: 1 failure — my engine test asserted an exact
   singleton list under `reg_final`'s checkpoint, which is
   order-dependent because earlier module tests legitimately add
   comparisons on the shared env. The authoritative parity assertions
   passed; I relaxed the singleton claim to the count-based
   "exactly once" check (matching the prompt's actual requirement).
3. First facade run: `pyflakes` caught `ComparisonRecord` undefined in
   `app/engine.py` (annotation) — import added before any pytest run.

Final results: focused comparison suites **32 passed** (27 pre-existing
+ 5 new); full suite **440 passed** serially (94.12 s; 435 pre-existing
intact + 5 new); `compileall` clean; `pyflakes` clean (0 new findings;
only the pre-existing legacy smoke_m9/m11 notes); OpenAPI standalone
post-check: **58 paths**, endpoint exactly once, GET-only, before the
generic detail route.

## 5. Live Smoke

`smoke_m26_live.py` (new, committed) against the production
`FORGE_ROOT` on **port 8747**: **51/51 PASS**, exit 0. Baseline
per-file SHA256 inventory saved to
`/tmp/m26-smoke-baseline-inventory.json` (96 files). LIVE A–K:

- **A baseline**: 96/4,002,745/0 exact; both models resolve; 3
  checkpoints registered (other model none); M5 listing = exactly the 8
  known ids in ASCENDING order, other model `[]`; persisted side facts
  14 checkpoint + 2 current sides, both A=B records (786de08efe4c under
  025e, 729f9c55ea89 under 30a8) and both current-vs-checkpoint records
  (d683f9b81195 → 025e, 5c5ff22151ed → 0511) exactly as audited, no
  both-sides-current record; M11/M16/M18–M25 pre-state intact; dashboard
  hash `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`;
  OpenAPI 58 with the new path once (GET-only).
- **B `025e6d8d8f15`**: 200, exactly **6** records — the exact audited
  ids [fc379bfcb50f, baa361012e00, d683f9b81195, 786de08efe4c,
  d9a62dde016b, d62f89e97c85] in M5 order; every record involves the
  checkpoint on ≥1 persisted side; parity with the filtered live
  listing; the A=B record 786de08efe4c EXACTLY ONCE; verbatim detail
  getter parity for all 6 (verdict/losses included).
- **C `30a8bc5b82ab`**: 200, exactly **5** records [fc379bfcb50f,
  baa361012e00, d9a62dde016b, 729f9c55ea89, d62f89e97c85]; A=B record
  729f9c55ea89 EXACTLY ONCE; listing parity.
- **D `0511de4c7372`**: 200, exactly **1** record (5c5ff22151ed),
  included via its checkpoint side B only (side A current, id null);
  parity.
- **E dedup/partition**: 6 + 5 + 1 = 12 membership slots over **8
  unique** comparisons; no duplicate id within any checkpoint history;
  the 4 double-membership records (fc379, baa36, d9a62, d62f8 under BOTH
  025e and 30a8) each appear once per checkpoint; each A=B record lives
  only under its own checkpoint.
- **F current-state exclusion**: d683f9b81195 and 5c5ff22151ed appear
  under exactly their checkpoint side's history and no other; no
  both-sides-current record exists and current sides keep
  `checkpoint_id` null.
- **G determinism**: repeated GETs raw-byte-identical for all three
  checkpoints.
- **H unknown model**: 404 (even with a real checkpoint id).
- **I unknown checkpoint**: well-formed 12-hex → 404; malformed
  (percent-encoded spaces/`!!`) → 404, no crash.
- **J cross-model + no shadowing**: `b5bc905326b6` + this model's real
  registry-valid `0511de4c7372` → 404 (model-scoped M3 registry; only
  registry-valid ids probed, no guessed statuses); the other model's
  comparison history is `[]` (no id leaks); detail getter
  `fc379bfcb50f` resolves verbatim (by-checkpoint does not shadow it);
  ghost comparison id → 404.
- **K regression + final audit**: M5 listing + all 8 detail getters,
  M6 decisions, M16/M18–M20 sample-quality (incl. by-checkpoint), M21
  by-suite (10), M22 summary (10), M23 by-policy (1), M24 evaluations
  by-checkpoint (3), M25 suite-runs by-checkpoint (10), M17 dashboard
  (full hash, empty diagnostics), M11 workflows, M3 checkpoint + M9
  policy registries, OpenAPI still 58 — all byte-identical to pre-state;
  by-checkpoint deterministic at the end; final storage 96/4,002,745/0
  with every pre-existing file byte-identical, 0 new files, 0 missing.

Honest failure log: the first two smoke runs FAILED 3 checks — two
script bugs (my URL slice `BASE[:-6]` fetched `/api/openapi.json`
instead of `/openapi.json`, and my A5 side-count claim said 10+6 where
the audited truth is 14+2). Production behavior was never wrong (the
same runs passed all 48 behavioral checks); after fixing the script's
URL constant and count claim against the live listing, 51/51 PASS. The
uvicorn server was stopped after the smoke; no server left running.

## 6. Storage Integrity

- Pre-change inventory `m26_pre.sha256` (repo root, committed) — diff vs
  `m25_pre.sha256` empty at baseline.
- Post-everything audit: **96 files / 4,002,745 bytes / 0 `.tmp`**;
  `sha256sum -c m26_pre.sha256` → **96 OK / 0 failed**; smoke inventory
  diff → **0 changed / 0 new / 0 missing / 0 bytes growth**.
- The endpoint performs zero writes (engine filter over parsed
  manifests only; verified live by the byte-identical inventory).

## 7. Regression / Compatibility

- Full suite: 440/440 with all 435 pre-existing tests intact (only the
  10 exact OpenAPI-count assertions updated 57 → 58; grep-verified).
- OpenAPI 57 → 58: exactly the one new path, GET-only, `comparison`
  tag, `list[ComparisonRecord]` response, registered before the generic
  `/comparisons/{comparison_id}` route (no shadowing in either
  direction — live-verified).
- Live byte-identical before/after: M5 run/list/get semantics, M6 gate
  decisions (+ by-policy), M15 samples, M16 sample-quality, M18–M20
  (incl. sample-quality by-checkpoint), M21/M22 suite surfaces, M23
  by-policy, M24 evaluations by-checkpoint, M25 suite-runs
  by-checkpoint, M11 workflows, M13/M17 dashboard
  (`f48557fe…a3838` preserved), M3 checkpoint registry, M9 policy/suite
  registries.
- Prompt §16 items 1–4 (production facts) verified live in §5; items
  5–14 verified by the §4 fixtures; M5 run/list/get + validation and M6
  gate regression explicitly re-tested.

## 8. Commit / Final Status

- Commit: **`M26: add comparison history by checkpoint`** on branch
  **`arena/01a071e9-code-forge`**, pushed to
  `origin/arena/01a071e9-code-forge`; local HEAD == remote HEAD
  (FETCH_HEAD diff-verified), working tree clean after the push (commit
  hash recorded in the git log; no unrelated commits amended).
- Files changed: `app/comparison.py`, `app/engine.py`, `app/api.py`,
  `tests/test_comparison.py`, `tests/test_comparison_api.py`,
  `tests/test_sample_quality.py`, `tests/test_dashboards.py`,
  `tests/test_suite_runs.py`, `tests/test_gates_api.py`,
  `tests/test_evaluation_api.py`, `README.md`,
  `smoke_m26_live.py` (new), `m26_pre.sha256` (new, audit artifact),
  `M26_final_report.md` (new).
- **Final status: M26 COMPLETE AND CERTIFIED.** All gates pass:
  440/440 tests serially (435 pre-existing intact + 5 new); compileall
  clean; pyflakes 0 new findings; OpenAPI 57 → **58 paths** with the
  endpoint documented exactly once; live smoke **51/51 PASS** on
  production (6/5/1 records for `025e6d8d8f15`/`30a8bc5b82ab`/
  `0511de4c7372` with the exact audited ids, A=B exactly once,
  current-state sides never matching, union = 8 unique comparisons,
  clean 404s, cross-model isolation, byte-identical repeats); storage
  audit **96 files / 4,002,745 bytes / 0 `.tmp` / 0 changed / 0 new /
  0 missing / 0 bytes growth**; dashboard hash preserved.

## 9. Next Milestone Prompt

```
# MILESTONE 27 — SAMPLE HISTORY BY CHECKPOINT

Continue the existing **AI Model Forge** project.

M26 is the current certified milestone.

The goal of M27 is to add one small, read-only API capability for
inspecting the existing M15 sample history grouped by the checkpoint
each sample was generated from.

Do not redesign the architecture or start future training/improvement
features.

---

## 1. VERIFY THE BASELINE FIRST

Before changing any code, inspect the existing implementation and verify
the actual M26 state.

Expected baseline:

* Production root: `/home/user/ai-model-forge-data`
* 96 production files
* 4,002,745 bytes
* 0 `.tmp` files
* 440 tests passing
* OpenAPI: 58 paths
* M26 comparisons by-checkpoint endpoint working
* M25 suite-runs by-checkpoint endpoint working
* M24 evaluations by-checkpoint endpoint working
* M23 gate decisions by-policy endpoint working
* M22 suite-run summary / M21 by-suite working
* M20/M19/M18 sample-quality endpoints working
* M15 sample generation / listing / getter working

Also inspect the existing implementations for:

* the M15 sampling engine: `list_samples` / `get_sample` — the
  authoritative listing and its deterministic `(created_at, sample_id)`
  ordering
* the persisted `SampleRecord` (`app/schemas.py`): every sample carries
  a REQUIRED non-nullable `checkpoint_id: str` plus
  `checkpoint_weights_sha256` — each sample is generated from ONE
  explicit verified checkpoint; there is no state_kind enum and no
  current-state sample (M15 has no current-state generation), so the
  filter is single-field, unlike M26's two-sided comparison semantics
* the M3 checkpoint registry (`TrainingEngine.get_checkpoint`) — the
  same model-scoped ownership validation M20/M24/M25/M26 use
* the M18–M26 read-only API/engine patterns (registry validation +
  authoritative-listing filter + thin facade + thin route registered
  BEFORE any literal-shadowing path)

Create a SHA256 inventory of the production files before making changes.

Run the full test suite **serially** before modifying code.

Also run:

* `compileall`
* `pyflakes`
* OpenAPI validation

If the actual baseline differs from the expected values, investigate it
before proceeding.

---

## 2. M27 OBJECTIVE

Add exactly ONE new read-only endpoint:

`GET /api/v1/models/{model_id}/samples/by-checkpoint/{checkpoint_id}`

It answers ONE question: "which immutable M15 samples were generated
from this checkpoint state?" — the model's authoritative M15 listing
filtered by the persisted `checkpoint_id` recorded in each
`SampleRecord`, after checkpoint ownership is validated through the M3
registry. Nothing else.

Do NOT add: generation of new samples, quality measurement, training,
retraining, rollback, optimization, HPO, ranking, new scoring, new
metrics, background workers, databases, caches, indexes, new
persistence structures, automatic decisions, workflow changes,
dashboard redesign, Gemini, model improvement logic. No future
milestones early (no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/Gemini/
auto-training/auto-rollback/databases/distributed/rankings).

---

## 3. PRODUCTION FACTS (verified at M26)

Authoritative facts, read from the persisted manifests — do NOT invent
ids or counts; if a stated count disagrees with the manifests or the
live listing, the data wins and the discrepancy must be reported.

* Samples live at `<root>/samples/<model_id>/sample-<id>/manifest.json`
  (NOT under `models/`).
* Model `4a0a871886ef` owns exactly 4 samples, ALL generated from
  checkpoint `0511de4c7372`, in ASCENDING `(created_at, sample_id)`
  order: `f8e66f9c7b50` (greedy, 2026-09-04T12:13:33.461341Z),
  `e2b5166fa549` (greedy, .479056Z), `05820e8bc68a` (temperature,
  .494498Z), `4e8463e0df06` (temperature, .511219Z).
* Checkpoints `025e6d8d8f15` and `30a8bc5b82ab` are valid, registered,
  and have ZERO samples -> natural live `200 + []` cases.
* Model `b5bc905326b6` exists with NO checkpoints and NO samples
  (cross-model probe).
* Storage at M26 certification: 96 files / 4,002,745 bytes / 0 `.tmp`.
* M17 dashboard `result_hash`
  `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`.

---

## 4. REQUIRED BEHAVIOR

* Validate the model (unknown -> 404) and resolve the checkpoint
  through the model-scoped M3 registry `TrainingEngine.get_checkpoint`
  (unknown checkpoint OR a checkpoint id belonging to another model ->
  404; NEVER a raw filesystem existence check; never infer from
  filenames).
* Return the model's authoritative M15 `list_samples` output filtered
  by the persisted `checkpoint_id` field, in the exact M15
  `(created_at, sample_id)` ASCENDING order, as verbatim
  `SampleRecord` payloads (prompt, token ids, output text, strategy,
  result_hash included).
* A valid checkpoint with zero samples -> `200 + []` (never 404 —
  "no samples" is not "does not exist").
* Read-only and deterministic: repeated GETs byte-identical; the
  endpoint never writes; no duplicate manifest parsing beyond the
  authoritative listing; no new storage.
* The route MUST be registered BEFORE
  `GET /models/{model_id}/samples/{sample_id}` (the literal
  `by-checkpoint` path segment is not a sample id) and must not
  shadow or be shadowed by it. Also keep M20's
  `sample-quality/by-checkpoint` (a DIFFERENT surface — M16 quality
  evaluations of samples) fully intact and unconfused with this one.

---

## 5. ENGINE IMPLEMENTATION

Follow the M20/M24/M25/M26 pattern exactly:

* ONE method on the M15 sampling engine
  (`list_samples_for_checkpoint(model_id, checkpoint_id)`), placed
  directly after `get_sample`: validate via the already-composed M3
  registry handle (add no new wiring), then filter the authoritative
  `list_samples` listing by the persisted `checkpoint_id`; `[]` when
  none; never writes. Docstring documents semantics + 404 mapping.
* Thin facade in `app/engine.py` after the existing sample facade
  methods.
* Thin route in `app/api.py` (tags `["sampling"]`,
  `response_model=list[SampleRecord]`, `FileNotFoundError -> 404`),
  placed after `list_samples` and before the generic sample detail
  getter; extend the sampling section-header comment, the landing-page
  milestone list (one bullet) and the endpoint list (one `<li>`).
* README: minimal Milestone 27 section (registry-first ownership,
  single-field persisted-identity filter, verbatim payloads, exact M15
  ordering, `200 []`, 404s, read-only) + update the two test-count
  mentions (440 -> 445) + the sampling layout line.
* OpenAPI 58 -> 59; update ONLY the affected exact-count assertions
  (grep for `== 58`).

---

## 6. TESTS

Fixture tests (pytest, in the existing sampling test module, reusing
its helpers/fixtures; never rebuild shared state per test — cache it on
the module-scoped fixture env):

1. grouping parity: for >= 2 checkpoints with samples and >= 1 with
   none, the endpoint output equals the authoritative M15 listing
   filtered by the persisted `checkpoint_id`, exact ids, exact
   `(created_at, sample_id)` order
2. verbatim payloads (equal to `get_sample` for each returned record)
3. deterministic repeated GETs: raw bytes identical
4. valid checkpoint with no samples -> `200 + []`
5. unknown model -> 404; unknown checkpoint -> 404
6. cross-model: another real model + this model's real checkpoint id ->
   404 (model-scoped M3 registry)
7. M15 generate/list/get + validation unchanged (regression)
8. M16 sample-quality surfaces + M20 by-checkpoint unchanged
9. OpenAPI: 59 paths, the new path exactly once, GET-only, `sampling`
   tag, `array` of `$ref SampleRecord`
10. route order: detail getter for a real sample id still resolves;
    ghost sample id still 404

Expected: 440 -> 445 tests passing.

---

## 7. LIVE TEST

Write `smoke_m27_live.py` (pattern: `smoke_m26_live.py`) and run it
against a live server on **port 8748** with
`FORGE_ROOT=/home/user/ai-model-forge-data`. LIVE A–K:

* A baseline: exact 96/4,002,745/0 audit + per-file SHA256 inventory
  saved to `/tmp/m27-smoke-baseline-inventory.json` + model/checkpoint/
  sample registries + M15 listing pre-state (exactly the 4 known ids,
  ALL `checkpoint_id` `0511de4c7372`) + M16/M18–M26 pre-state +
  dashboard hash + OpenAPI 59 pre-state
* B known checkpoint `0511de4c7372` -> 200 with EXACTLY 4 records —
  the exact known ids in ASCENDING `(created_at, sample_id)` order,
  every record's persisted `checkpoint_id` verified, parity with the
  filtered M15 listing, verbatim detail-getter parity
* C empty checkpoint `025e6d8d8f15` -> 200 + []
* D empty checkpoint `30a8bc5b82ab` -> 200 + []
* E partition: 4 + 0 + 0 = 4, no duplicates, union equals the full M15
  listing
* F deterministic repeat: raw-byte-identical GETs
* G unknown model -> 404
* H unknown checkpoint (well-formed + malformed) -> 404
* I cross-model: `b5bc905326b6` + `0511de4c7372` -> 404 (only
  registry-valid ids; no guessed statuses); no sample id leaks under
  the other model; the sample detail getter still resolves (no
  shadowing); ghost sample id -> 404
* J regression: M15 list/get, M16 quality, M18–M20 sample-quality
  (incl. by-checkpoint — a different surface), M21/M22, M23, M24, M25,
  M26 comparisons by-checkpoint, dashboard hash, policy/checkpoint
  registries, OpenAPI 59 — all unchanged
* K final audit: every pre-existing file byte-identical, ZERO new
  files, zero `.tmp`, zero storage growth

Exit code 0 = pass. Stop the server afterwards.

---

## 8. FINAL AUDIT

Static sequence — run in this order and report the FIRST failure
honestly if anything fails:

1. focused tests (sampling module)
2. full suite serially (expect 445 passed)
3. live smoke (LIVE A–K, port 8748)
4. `compileall` (app, tests, smoke script)
5. `pyflakes` (app, tests, smoke script; 0 new findings)
6. OpenAPI post-check: 59 paths, endpoint exactly once, before the
   generic sample route
7. SHA256 audit: production inventory vs the pre-M27 inventory
   (0 changed / 0 new / 0 missing / 0 bytes growth)
8. tree audit: only the intended files changed
9. remote sync check

---

## 9. KEEP THE SCOPE SMALL

* ONE read-only endpoint; ONE engine method; ONE facade; ONE route.
* No generation, no quality measurement, no training/rollback/
  optimization/HPO/ranking, no new scoring or metrics, no workers,
  databases, caches, indexes, no new persistence, no automatic
  decisions, no workflow changes, no dashboard redesign, no Gemini, no
  model improvement logic.
* Do not modify production data to manufacture fixtures; empty cases
  come from the real `025e6d8d8f15` / `30a8bc5b82ab` checkpoints.
* Do not weaken existing tests; update only exact OpenAPI-count
  assertions (58 -> 59).
* Verify the baseline BEFORE changing code; if anything differs,
  investigate and fix before certification; report honestly at all
  times.

---

## 10. COMMIT / PUSH / REPORT

* Commit message: `M27: add sample history by checkpoint`
* Push to `arena/01a071e9-code-forge`; verify local == remote; clean
  tree.
* Write `M27_final_report.md` with exactly 9 sections (objective,
  baseline, implementation, tests, live smoke, storage integrity,
  regression/compatibility, commit/final status, next-milestone
  prompt) with EXACT measured values (sample ids, per-checkpoint
  counts, determinism, 404s, cross-model, SHA256, dashboard hash,
  commit/branch/sync/tree).
* Section 9 must contain the COMPLETE copy-ready M28 prompt, grounded
  ONLY in facts discovered and verified during M27 (no invented ids,
  counts or endpoints — verify a surface exists before promising it),
  small/additive/read-only, no LoRA/QLoRA/DPO/RL/RLHF/RLAIF/HPO/Gemini/
  auto-training/auto-rollback/databases/distributed/rankings, including
  automatic M29 prompt generation in its own report section.
```
