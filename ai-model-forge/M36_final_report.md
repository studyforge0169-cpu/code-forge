# M36 Final Report — Evaluation History by Split

## Milestone

M36 adds ONE narrow read-only MODEL-SCOPED access path to ai-model-forge:

```
GET /api/v1/models/{model_id}/evaluations/by-split/{split}
```

It answers "which immutable M4 evaluation records of this model were run
against this dataset split?" — the model's authoritative M4 listing
filtered by the persisted top-level `split` field recorded in each
`EvaluationRecord` (matched VERBATIM against the `EvaluationSplit` schema
enum `train`/`validation`/`test`; membership NEVER inferred from
filenames, directories, timestamps, dataset names or ids, and the
persisted value is never resolved or rewritten). The route is registered
BEFORE the generic `/evaluations/{eval_id}` getter and alongside the
by-checkpoint (M24), by-dataset (M28) and by-tokenizer (M30) groupings.
OpenAPI 67 -> 68.

Contract (exactly as specified):

- ordering inherited: the exact authoritative M4 `(created_at, eval_id)`
  ASCENDING order — no re-sorting, no new ordering rules;
- valid split + zero matching records -> `200` + `[]` (NEVER 404);
- unknown model -> `404` (exactly like the sibling groupings);
- unsupported split value -> `422` via the schema enum path parameter
  (native FastAPI/Pydantic validation — never manually converted to a
  registry-style 404), and because validation fires pre-handler the 422
  takes precedence even when the model id is also unknown;
- minimum-files rule honored: no persistence, cache, index, DB,
  migration or manifest writes — the endpoint derives everything from
  the existing M4 manifests; repeated GETs are byte-identical.

## Baseline

Session start was ANOTHER out-of-band reset (reset #8): sandbox HEAD was
`f86b670` ("Add files via upload") while the authoritative remote
FETCH_HEAD was `6fc989c` (M35 inventory commit). The standard recovery
runbook was executed before any coding:

1. `git reset --hard 6fc989c` on the session branch;
2. production data restored from `code-forge/code forge.zip`;
3. `sha256sum -c m35_pre.sha256` from the data root -> **0 non-OK**
   (byte-identical through M35);
4. venv rebuilt (`~/.venv/bin/pip install -e ai-model-forge[dev]
   pyflakes`);
5. reconciled with the authoritative remote (no force-push; FETCH_HEAD
   used — the repo has no origin ref).

Baseline then re-proved BEFORE implementation:

- full suite: **485 passed @ 128.62 s** (exact M35 ladder);
- OpenAPI: **67 paths**;
- `m36_pre.sha256` (full per-file inventory of the production root)
  diff vs the restored tree: **EMPTY**;
- stale test tmp dirs: **0**;
- live grounding re-verified: model `4a0a871886ef` owns 16 evaluation
  records with persisted split distribution `validation -> 14`,
  `train -> 2` (and `test -> 0` — no records); model `b5bc905326b6`
  owns none; both models exist, so the 404/200-[] axes are natural.

## Implementation

Four files (plus tests/docs), mirroring the M24/M28/M30/M35 pattern:

- `app/evaluation.py` — new engine method `list_evaluations_for_split(
  model_id, split: EvaluationSplit)` placed directly after
  `list_evaluations_for_tokenizer`. It reuses the model's authoritative
  M4 listing (exact engine parse + deterministic `(created_at, eval_id)`
  ASC order) and filters by the persisted top-level `split` VERBATIM.
  Docstring records the contract: splits have NO registry (unlike the
  checkpoint/dataset/tokenizer axes) — the enum IS the contract, so
  unsupported values are rejected at the API boundary with 422 while an
  unknown model raises FileNotFoundError exactly like the siblings; a
  valid split with zero evaluations returns `[]`; read-only, never
  writes. `EvaluationSplit` added to the schema import block.
- `app/engine.py` — `ModelForge` facade delegating to the new method
  (+`EvaluationSplit` import).
- `app/api.py` — route `@api.get("/models/{model_id}/evaluations/
  by-split/{split}", response_model=list[EvaluationRecord],
  tags=["evaluation"])` with `split: EvaluationSplit` as the path
  parameter (this is what makes unsupported values a native pre-handler
  422), placed after the by-tokenizer route and before the generic
  `/evaluations/{eval_id}` getter; landing-page bullet, endpoint `<li>`
  in the documented layout, and a section comment (+import). One
  hard-won lesson: the route annotation references `EvaluationSplit`,
  so the import must exist in `api.py` BEFORE `app.openapi()` builds —
  an initial miss raised `PydanticUserError: ForwardRef
  ('EvaluationSplit') is not fully defined`; fixed by adding the name to
  the existing `.schemas` import (typed-parameter imports must land in
  ALL THREE files: evaluation.py, engine.py, api.py).
- `README.md` — M36 section, endpoint counts 485 -> 490 (two places),
  and the API layout line now reads M24/M28/M30/M36 for the evaluation
  groupings.

No other production code changed; no persistence, cache, index or
migration logic was added anywhere.

## Tests

Five new tests (485 -> 490, exactly as targeted):

- `tests/test_evaluation.py` — 3 engine tests via the cached
  `_m36_env(env)` fixture (main model with validation x2 / train x1 /
  test x1 evaluations plus an empty second model): exact filtered-listing
  parity + verbatim payloads + ordering; valid-enum empty (`test`, and
  all splits under the empty model) -> `[]`; unknown model ->
  `FileNotFoundError`.
- `tests/test_evaluation_api.py` — 2 API tests via the existing
  `_prepare`/`_eval_cfg`/`EVAL_RUN`/`MODELS` helpers: 200 + exact
  records + detail-getter parity + 422 for unsupported values (incl.
  case variant) + 404 unknown model + `200 []` for the natural empty
  cases + byte-identical repeat + M24/M28/M30 sibling regressions.

One iteration during bring-up (fixed before any full run): the M24
regression inside the API test initially asserted the checkpoint group
equals `[v1]`, but `v1` is a CURRENT-state evaluation (`checkpoint_id:
null`), so the honest check is parity with the locally filtered listing
— rewritten to derive the expectation from the listing itself.

Results (nothing certified from partial runs):

- focused: engine 3/3, API 2/2 (first engine run green; API green after
  the assert fix above);
- full suite, three consecutive complete runs after the fix:
  **490 passed @ 122.11 s**, **490 passed @ 124.83 s**, **490 passed @
  115.00 s** (+ a partitioned 38 + 452 = 490 confirmation);
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings** (no new, no
  pre-existing reclassified);
- OpenAPI count sweep: every hard-coded count assertion searched and
  updated — **21 occurrences of `== 68`, 0 remaining `== 67`** across 8
  test modules, plus 13 ladder comments (490/485/480/…);
- standalone OpenAPI verification (no server): 68 paths; new path
  registered exactly once, GET-only, tag `evaluation`; route order
  by-tokenizer < by-split < generic `{eval_id}`; `split` parameter
  `$ref`s `EvaluationSplit`; items `$ref` `EvaluationRecord`.
- stale test tmp: verified no live pytest/uvicorn process, then cleaned
  8 accumulated `/tmp/forge-tests-*` sandboxes. Observed and
  characterized a PRE-EXISTING suite behavior: each full run leaves
  exactly 1 shared-fixture sandbox behind (reproduced identically with
  the M36 modules excluded — 38-test and 452-test runs each left 1), so
  it is not caused by M36; workspace cleaned again before the live
  smoke.

## Live Smoke

`smoke_m36_live.py` (committed) — 33 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
33/33 PASS each, exit 0**, on port **8757** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m36-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + M4/M5/M6/M11/M16/
  M18–M35 pre-state + dashboard hash + OpenAPI 68 pre-state + known
  pair 200.
- B (DISCOVERY, never assumed): from the live M4 listing + the OpenAPI
  schema — 16 evaluations, split distribution `validation -> 14,
  train -> 2, test -> 0`, ASC `(created_at, eval_id)`; enum values
  `['train', 'validation', 'test']`.
- C: `validation` -> 200 + EXACT 14 records == listing filtered locally;
  verbatim detail-getter parity for all 14 (metrics + split included).
- D: three GETs raw-byte-identical.
- E: `test` (valid enum, zero records) -> 200 + [] on the known model;
  ALL enum splits -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid split -> 404; unsupported values
  (interior-space, `VALIDATION`, `3`) -> 422 with the enum detail;
  unsupported split + UNKNOWN model -> 422 (pre-handler precedence).
- G: the three groups partition the full listing exactly (14+2+0 == 16,
  every record exactly once, no record in two groups, persisted split
  verbatim).
- H: every group preserves the authoritative M4 order exactly.
- I–K: M24/M28/M30 (3/3/3, 16 by-dataset, 16 by-tokenizer + parity),
  M22 (10/10/summary), M26/M29/M31 (6/5/1, 8, 8 + parity), M27/M32/
  M33 (4/0/0, 4, 2), M34/M23 (4 by-comparison, 1 by-policy), M12/M35
  (registry 7 verbatim, global runs 2, by-recipe 2, zero-run recipes
  200 + [], unknown recipe 404), M16 (evaluation ids paired in
  sample-quality) — all unchanged.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe registries
  unchanged.
- M: OpenAPI exactly 68 paths, new path once, order
  by-tokenizer < by-split < generic, M35 by-recipe path still present
  once; storage zero drift (below).

Honest note on smoke bring-up: three script-side defects were found and
fixed during the first smoke iterations (a zero-count enum key can never
appear in a record-derived dict; a trailing-space path value is stripped
by the HTTP request-line parser before the server sees it — replaced
with an interior-space value; a double `json.loads`). All three were
smoke-script bugs; the application behavior was correct throughout, and
the committed script passed 33/33 three times.

## Production Storage

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- final audit after the server stopped: `sha256sum -c m36_pre.sha256`
  from the data root -> **0 non-OK** (every pre-existing file
  byte-identical);
- smoke M3/M4/M5: 0 changed, 0 missing, 0 new files across all three
  runs;
- M36 wrote NOTHING to production: the endpoint is pure data access
  over existing M4 manifests. Production remains byte-identical through
  M36.

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5 listings + detail getters, M22 suite-run
  histories, M23 gate decisions by policy, M24/M28/M30 evaluation
  groupings, M26/M29/M31 comparison groupings, M27/M32/M33 sample and
  sample-quality groupings, M34 gate decisions by comparison, M35
  workflow history by recipe (model-scoped + global M12 lineage), M16
  pairings, M17 dashboard (full hash), M2/M9/M12/M14 registries.
- Full suite green at 490 (no test weakened; the one edited assertion
  was corrected to derive its expectation from the authoritative
  listing rather than a hard-coded id list).
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no change to any other route's
  behavior; existing clients of sibling endpoints see identical
  responses (byte-verified).
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the codebase.

## Git / Commit

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M36: add evaluation history by split` — `app/evaluation.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 9 touched test
   modules (2 with new tests, 7 with the OpenAPI 68 sweep + ladders),
   `smoke_m36_live.py`, `M36_final_report.md`.
2. follow-up inventory commit adding `m36_pre.sha256` at the repo root
   (M21–M36 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed).

## Next Milestone

End-of-M36 live inspection (production server, real records — fields
read from the actual payloads): `ComparisonRecord` persists a top-level
`split` field — all 8 production comparisons of `4a0a871886ef` carry
`split: "validation"` (train/test have zero comparisons: natural
valid-empty cases); `SampleRecord`/`SampleQualityRecord`/`SuiteRun`
persist NO split; gate decisions expose `suggested_checkpoint_id`
(None -> 8, `025e6d8d8f15` -> 3) but that is a suggested-not-measured
axis. Comparisons already group by checkpoint (M26), dataset (M29) and
tokenizer (M31) — the split axis for comparisons is the one remaining
smallest read-only sibling surface.

Copy-ready M37 prompt:

> M37: add comparison history by split. ONE read-only MODEL-SCOPED
> endpoint `GET /api/v1/models/{model_id}/comparisons/by-split/{split}`
> returning the authoritative M5 comparison listing of the model
> filtered by the persisted top-level `ComparisonRecord.split` field
> matched VERBATIM against the existing `EvaluationSplit` schema enum
> (train/validation/test — never inferred from filenames, datasets,
> timestamps or ids; never resolved or rewritten). Inherit the exact M5
> `(created_at, comparison_id)` ASC ordering; valid split + zero
> records -> `200 []`; unknown model -> `404`; unsupported split value
> -> `422` via the schema enum path parameter (native FastAPI
> validation, never a registry-style 404; validation fires pre-handler
> even for an unknown model). Register the route after
> `comparisons/by-tokenizer/{tokenizer_id}` and BEFORE the generic
> `/comparisons/{comparison_id}`. OpenAPI 68 -> 69 (search and update
> EVERY hard-coded count); tests 490 -> ~495 (report the actual);
> full suite twice, compileall, pyflakes, stale `/tmp/forge-tests-*`
> cleanup (verify no live process first). Live smoke on port 8758 with
> >= 17 named checks: DISCOVER the split distribution live (grounding
> facts from M36: `4a0a871886ef` owns 8 comparisons, all
> `split="validation"`; `b5bc905326b6` owns none; train/test are
> valid-empty), partition + parity + ordering + byte-identical x3,
> 404/422/200-[] separation, and M2–M36 regressions (dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`,
> 96/4,002,745/0 storage, SHA256 audit 0/0/0). Minimum-files rule: no
> persistence/cache/index/DB/migration writes. Final report with exactly
> the 9 sections Milestone, Baseline, Implementation, Tests, Live
> Smoke, Production Storage, Regression / Compatibility, Git / Commit,
> Next Milestone — §9 a complete copy-ready M38 prompt grounded ONLY in
> facts you discover live at the end of M37 (inspect the remaining
> persisted record relationships yourself; pick the next smallest
> read-only surface; never invent distribution facts), including the
> same M38 auto-generation requirement. Commit `M37: add comparison
> history by split`, push (never force-push), then the follow-up
> inventory commit `m37_pre.sha256` per the M21–M36 convention; only
> egg-info may remain untracked.
