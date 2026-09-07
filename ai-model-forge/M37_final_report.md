# M37 Final Report — Comparison History by Split

## Milestone

M37 adds ONE narrow read-only MODEL-SCOPED access path to ai-model-forge:

```
GET /api/v1/models/{model_id}/comparisons/by-split/{split}
```

It answers "which immutable M5 comparison records of this model measured
this dataset split?" — the model's authoritative M5 listing filtered by
the persisted top-level `split` field recorded in each
`ComparisonRecord` (the shared-probe contract: a comparison exists only
when BOTH sides measure the SAME dataset/version/split/tokenizer/
window/seed probe, so the split is a property of the comparison itself;
matched VERBATIM against the `EvaluationSplit` schema enum — membership
NEVER inferred from filenames, checkpoint ids, dataset identities,
nested evaluation records or timestamps, and the persisted value is
never resolved or rewritten). The route is registered BEFORE the
generic `/comparisons/{comparison_id}` getter and alongside the
by-checkpoint (M26), by-dataset (M29) and by-tokenizer (M31) groupings.
OpenAPI 68 -> 69.

Contract (exactly as specified):

- ordering inherited: the exact authoritative M5 `(created_at,
  comparison_id)` ASCENDING order;
- valid split + zero matching records -> `200` + `[]` (NEVER 404);
- unknown model + valid split -> `404` (exactly like the sibling
  groupings);
- unsupported split value -> `422` via the schema enum path parameter
  (native FastAPI/Pydantic validation, never manually converted to a
  registry-style 404) — validation fires pre-handler, so
  unknown-model + invalid-split is also 422, matching M36;
- minimum-files rule honored: no persistence, cache, index, DB,
  migration or manifest writes — everything derives from the existing
  M5 comparison manifests; repeated GETs are byte-identical.

## Baseline

No sandbox reset this session: HEAD == FETCH_HEAD == `80bd46a` (M36
implementation `f0f07f2` + inventory `80bd46a`), tree clean except the
known egg-info. Baseline re-proved BEFORE any changes:

- full suite: **490 passed @ 124.70 s** (exact M36 ladder);
- OpenAPI: **68 paths**;
- production: **96 files / 4,002,745 bytes / 0 .tmp**;
- `m37_pre.sha256` captured pre-implementation; `m36_pre.sha256`
  check -> **0 non-OK** (zero drift);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes;
- venv intact (fastapi + pyflakes import checks).

## Implementation

Three production files + docs, mirroring the M26/M29/M31/M36 pattern:

- `app/comparison.py` — new engine method `list_comparisons_for_split(
  model_id, split: EvaluationSplit)` placed directly after
  `list_comparisons_for_tokenizer`. It reuses the model's authoritative
  M5 listing (exact engine parse + deterministic `(created_at,
  comparison_id)` ASC order) and filters by the persisted shared-probe
  `split` VERBATIM. Docstring records the contract: the split is one
  half of the shared-probe identity (both sides measure the same probe
  by construction, so it is a property of the comparison, never of a
  side); splits have NO registry (unlike the checkpoint/dataset/
  tokenizer axes) — the enum IS the contract, so unsupported values
  are rejected with 422 at the API boundary and never reach the
  engine, while an unknown model raises FileNotFoundError exactly like
  the siblings; a valid split with zero comparisons returns `[]`;
  read-only, never writes. `EvaluationSplit` added to the schemas
  import (all-three-files rule honored: comparison.py, engine.py —
  already imported since M36 — api.py, already imported since M36; no
  ForwardRecurrence of the M36 openapi() crash).
- `app/engine.py` — `ModelForge` facade `list_comparisons_for_split`
  delegating to the engine, placed after the by-tokenizer delegation.
- `app/api.py` — route `@api.get("/models/{model_id}/comparisons/
  by-split/{split}", response_model=list[ComparisonRecord],
  tags=["comparison"])` with `split: EvaluationSplit` as the path
  parameter (this is what makes unsupported values a native pre-handler
  422), placed after the by-tokenizer route and before the generic
  `/comparisons/{comparison_id}` getter; landing-page bullet; endpoint
  `<li>` in the documented layout; route-family section comment.
  One honest correction here: the section comment above the comparison
  routes claimed since M36 that "Milestone 36 adds the read-only
  by-split grouping" — an M36 comment slip (M36 was evaluation-by-split;
  discovered during this milestone's inspection). M37 is exactly the
  comparison by-split grouping, so the line now correctly names
  Milestone 37.
- `README.md` — M37 section, counts 490 -> 495 (two places), layout
  line `M26/M29/M31/M37`.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere.

## Tests

Five new tests (490 -> 495, exactly as targeted):

- `tests/test_comparison.py` — 3 engine tests via the cached
  `_m37_env(env)` fixture (one TRAIN-split + one fresh VALIDATION-split
  comparison on the shared model; `test` stays the natural valid-enum
  empty): exact filtered-listing parity for ALL THREE enum splits +
  verbatim detail-getter payloads + ordering; explicit disjoint
  partition whose union is the full listing; valid-enum empty (`test`,
  and all splits under the empty second model) -> `[]`; unknown model
  -> `FileNotFoundError`; cross-model isolation; read-only (manifest
  file-set snapshots unchanged); repeated-call determinism.
- `tests/test_comparison_api.py` — 2 API tests via the existing
  `_prepare`/`_comp` helpers: 200 + exact records + detail-getter
  parity + raw-byte-identical x3 + partition over all enum splits +
  `train`/`test` -> `200 []`; unknown model + valid split -> 404;
  unsupported values (case variant, interior-space, numeric) -> 422
  and unknown-model + invalid-split -> 422 (pre-handler precedence);
  cross-model isolation under every split; M5 listing/getter + ghost-id
  404 (no route capture); M26/M29/M31 sibling regressions; M36
  evaluations-by-split regression; OpenAPI 69 assertions (route order
  by-tokenizer < by-split < generic, GET-only, tag `comparison`, items
  `$ref ComparisonRecord`, split param `$ref EvaluationSplit`).

Two honest bring-up iterations (both fixed before any full run, same
lesson as M36): (1) the engine fixture must self-seed BOTH split
groups — with `-k m37` the earlier tests' comparisons don't exist, so
the fixture now creates its own validation + train records; (2) the
full module revealed an earlier M5 test already creates a train-split
comparison on the shared model, so "train group == exactly the
fixture's record" was replaced by membership + parity derived from the
authoritative listing.

Results (nothing certified from partial runs):

- focused: engine 3/3, API 2/2;
- full suite after the fixes, twice consecutively: **495 passed @
  139.39 s**, then (after cleaning 10 accumulated sandboxes, no live
  process) **495 passed @ 138.94 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**;
- OpenAPI count sweep: every hard-coded count assertion searched and
  updated — **21 assertions `== 68` -> `== 69` across 8 test modules,
  0 `== 68` remaining**, plus 14 ladder comments extended with
  `+ 1 (M37 comparisons by-split)`;
- stale test tmp: verified no live pytest/uvicorn process, then
  cleaned 10 accumulated `/tmp/forge-tests-*` sandboxes before the
  certifying full run (the pre-existing 1-per-run shared-fixture
  leftover behavior is unchanged and not caused by M37).

## Live Smoke

`smoke_m37_live.py` (committed) — 33 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
33/33 PASS each, exit 0, first attempt**, on port **8758** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m37-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + M4/M5/M6/M11/M16/
  M18–M36 pre-state (including 14 validation-split evaluations via the
  M36 surface) + dashboard hash + OpenAPI 69 pre-state + known pair
  200.
- B (DISCOVERY, never assumed): from the live M5 listing + the OpenAPI
  schema — **8 comparisons, ALL `split="validation"`**
  (`validation -> 8, train -> 0, test -> 0`), ASC `(created_at,
  comparison_id)`; enum values `['train', 'validation', 'test']` —
  matching the M36-end grounding exactly.
- C: `validation` -> 200 + EXACT 8 records == listing filtered locally;
  verbatim detail-getter parity for all 8 (verdict/per-side losses +
  split included).
- D: three GETs raw-byte-identical.
- E: `train` AND `test` (valid enum, zero records) -> 200 + [] on the
  known model; ALL enum splits -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid split -> 404; unsupported values
  (interior-space, `VALIDATION`, `3`) -> 422 with the enum detail;
  unsupported split + UNKNOWN model -> 422 (pre-handler precedence).
- G: the three groups partition the full listing exactly (8+0+0 == 8,
  every record exactly once, no record in two groups, persisted split
  verbatim).
- H: every group preserves the authoritative M5 order exactly.
- I: M26/M29/M31 comparison histories unchanged (025e->6 / 30a8->5 /
  0511->1 by-checkpoint, 8 by-dataset, 8 by-tokenizer + parity).
- J: M22 suite runs (10/10/summary), M24/M28/M30 evaluation histories
  (3/3/3, 16, 16), **M36 evaluations-by-split unchanged
  (validation->14 / train->2 / test->0 + parity)**, M27/M32/M33
  samples (4/0/0, 4, 2), M34/M23 gate decisions (4 by-comparison,
  1 by-policy).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe registries
  unchanged.
- M: OpenAPI exactly 69 paths, new path once, order by-tokenizer <
  by-split < generic, M36 evaluations-by-split path still present once;
  storage zero drift (below).

## Production Storage

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: 0 changed, 0 missing, 0 new files across all three
  runs;
- final audit after the server stopped: `sha256sum -c m37_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- M37 wrote NOTHING to production: the endpoint is pure data access
  over existing M5 comparison manifests. Production remains
  byte-identical through M37.

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5 listings + detail getters, M22 suite-run
  histories, M23 gate decisions by policy, M24/M28/M30/M36 evaluation
  groupings (including the M36 by-split twin), M26/M29/M31 comparison
  groupings, M27/M32/M33 sample and sample-quality groupings, M34 gate
  decisions by comparison, M35 workflow history by recipe
  (model-scoped + global M12 lineage), M16 pairings, M17 dashboard
  (full hash), M2/M9/M12/M14 registries.
- Full suite green at 495 with no weakened tests; the two edited new
  assertions were corrected to derive expectations from the
  authoritative listing (the established self-seeding/deselection
  discipline), never by relaxing coverage.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged.
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M37: add comparison history by split` — `app/comparison.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 11 touched test
   modules (2 with new tests, 9 with the OpenAPI 69 sweep + ladders),
   `smoke_m37_live.py`, `M37_final_report.md`.
2. follow-up inventory commit adding `m37_pre.sha256` at the repo root
   (M21–M37 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed).

## Next Milestone

End-of-M37 live inspection (production server, real payloads): every
`EvaluationRecord` persists a top-level `state_kind: EvalStateKind` —
a TWO-value schema enum (`current` = the model's published weights,
`checkpoint` = an immutable stored checkpoint) — with production
distribution **checkpoint -> 9, current -> 7** (both groups non-empty;
`checkpoint_id` is null exactly for the 7 current-state evaluations).
`ComparisonRecord`'s split axis is now fully exposed (M37); samples/
sample-quality/suite-runs persist no split; gate decisions' remaining
`suggested_checkpoint_id` axis is advisory (and its dominant value is
null — 8 of 11); sample `strategy` and workflow/suite `status` are
plain lifecycle strings, not schema enums. The one remaining clean
enum-contract axis over an authoritative listing is the evaluation
state kind.

Copy-ready M38 prompt:

> M38: add evaluation history by state kind. ONE read-only
> MODEL-SCOPED endpoint
> `GET /api/v1/models/{model_id}/evaluations/by-state/{state_kind}`
> returning the authoritative M4 evaluation listing of the model
> filtered by the persisted top-level `EvaluationRecord.state_kind`
> matched VERBATIM against the existing `EvalStateKind` schema enum
> (current/checkpoint — never inferred from checkpoint_id nullability,
> filenames, timestamps or hashes; never resolved or rewritten).
> Inherit the exact M4 `(created_at, eval_id)` ASC ordering; valid
> state kind + zero records -> `200 []`; unknown model -> `404`;
> unsupported state-kind value -> `422` via the schema enum path
> parameter (native FastAPI validation, never a registry-style 404;
> validation fires pre-handler even for an unknown model — the M36/
> M37 contract). Register the route after `evaluations/by-split/
> {split}` and BEFORE the generic `/evaluations/{eval_id}`. OpenAPI
> 69 -> 70 (search and update EVERY hard-coded count); tests 495 ->
> ~500 (report the actual); full suite twice, compileall, pyflakes,
> stale `/tmp/forge-tests-*` cleanup (verify no live process first).
> Live smoke on port 8759 with >= 17 named checks: DISCOVER the
> state-kind distribution live (grounding facts from M37: `4a0a871886ef`
> owns 16 evaluations, checkpoint -> 9 / current -> 7; `b5bc905326b6`
> owns none — both enum groups empty there), partition + parity +
> ordering + byte-identical x3, 404/422/200-[] separation, and M2–M37
> regressions (dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`,
> 96/4,002,745/0 storage, SHA256 audit 0/0/0). Minimum-files rule: no
> persistence/cache/index/DB/migration writes. Final report with
> exactly the 9 sections Milestone, Baseline, Implementation, Tests,
> Live Smoke, Production Storage, Regression / Compatibility, Git /
> Commit, Next Milestone — §9 a complete copy-ready M39 prompt
> grounded ONLY in facts you discover live at the end of M38 (inspect
> the remaining persisted record relationships yourself; pick the next
> smallest read-only surface; never invent distribution facts),
> including the same M39 auto-generation requirement. Commit
> `M38: add evaluation history by state kind`, push (never
> force-push), then the follow-up inventory commit `m38_pre.sha256`
> per the M21–M37 convention; only egg-info may remain untracked.
