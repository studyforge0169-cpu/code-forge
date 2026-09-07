# M39 Final Report — Comparison History by Verdict

## Milestone

M39 adds ONE narrow read-only MODEL-SCOPED access path to ai-model-forge:

```
GET /api/v1/models/{model_id}/comparisons/by-verdict/{verdict}
```

It answers "which immutable M5 comparison records of this model produced
this verdict?" — the model's authoritative M5 listing filtered by the
persisted top-level `verdict` field recorded in each `ComparisonRecord`
(the `ComparisonVerdict` schema enum: improved / regressed / unchanged —
the immutable loss-only judgment persisted at run time by the M5
comparison flow; matched VERBATIM — membership NEVER recalculated from
loss deltas, per-side losses, tolerances, checkpoint ids or hashes, no
comparison executed and no evaluation rerun; the persisted value is
never resolved or rewritten). The route is registered BEFORE the
generic `/comparisons/{comparison_id}` getter and alongside the
by-checkpoint (M26), by-dataset (M29), by-tokenizer (M31) and by-split
(M37) groupings. OpenAPI 70 -> 71.

Contract (exactly as specified):

- ordering inherited: the exact authoritative M5 `(created_at,
  comparison_id)` ASCENDING order;
- valid verdict + zero matching records -> `200` + `[]` (NEVER 404);
- unknown model + valid verdict -> `404` (exactly like the sibling
  groupings);
- unsupported verdict value -> `422` via the schema enum path
  parameter (native FastAPI/Pydantic validation, never manually
  converted to a registry-style 404) — validation fires pre-handler,
  so unknown-model + invalid-verdict is also 422, matching
  M36/M37/M38;
- minimum-files rule honored: no persistence, cache, index, DB,
  migration or manifest writes — everything derives from the existing
  M5 comparison manifests; repeated GETs are byte-identical.

## Baseline

No sandbox reset this session: HEAD == FETCH_HEAD == `2832b91` (M38
implementation `37f8e52` + inventory `2832b91`), tree clean except the
known egg-info. Baseline re-proved BEFORE any changes:

- full suite: **500 passed @ 126.23 s** (exact M38 ladder);
- OpenAPI: **70 paths**;
- production: **96 files / 4,002,745 bytes / 0 .tmp**;
- `m39_pre.sha256` captured pre-implementation; `m38_pre.sha256`
  check -> **0 non-OK** (zero drift);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes.

## Implementation

Three production files + docs, mirroring the M26/M29/M31/M37 pattern:

- `app/comparison.py` — new engine method `list_comparisons_for_verdict(
  model_id, verdict: ComparisonVerdict)` placed directly after
  `list_comparisons_for_split`. It reuses the model's authoritative M5
  listing (exact engine parse + deterministic `(created_at,
  comparison_id)` ASC order) and filters by the persisted `verdict`
  VERBATIM. Docstring records the contract: the verdict is the
  immutable loss-only judgment ("ONLY about measured loss on one
  probe") — NEVER recalculated from loss deltas and the execution
  engine is never invoked; verdicts have NO registry (unlike the
  checkpoint/dataset/tokenizer axes) — the enum IS the contract, so
  unsupported values are rejected with 422 at the API boundary and
  never reach the engine, while an unknown model raises
  FileNotFoundError exactly like the siblings; a valid verdict with
  zero comparisons returns `[]`; read-only, never writes.
  `ComparisonVerdict` was already imported here (used by the run
  path).
- `app/engine.py` — `ModelForge` facade `list_comparisons_for_verdict`
  delegating to the engine, placed after the by-split delegation
  (+`ComparisonVerdict` import).
- `app/api.py` — route `@api.get("/models/{model_id}/comparisons/
  by-verdict/{verdict}", response_model=list[ComparisonRecord],
  tags=["comparison"])` with `verdict: ComparisonVerdict` as the path
  parameter (this is what makes unsupported values a native
  pre-handler 422), placed after the by-split route and before the
  generic `/comparisons/{comparison_id}` getter; landing-page bullet;
  endpoint `<li>`; comparison-routes section comment extended with
  the M39 line (+`ComparisonVerdict` import; all-three-files import
  rule honored).
- `README.md` — M39 section, counts 500 -> 505 (two places), layout
  line `M26/M29/M31/M37/M39`.

One honest note: the first README edit attempt aborted safely on its
own anchor assertion (the anchor said "evaluations-by-split" where the
M38 section's actual text reads "M37 comparisons-by-split"); the
corrected anchor was applied — no partial write ever reached the file.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere.

## Tests

Five new tests (500 -> 505, exactly as targeted):

- `tests/test_comparison.py` — 3 engine tests via the cached
  `_m39_env(env)` fixture (deterministic all-three-verdict coverage:
  one A=B record -> unchanged by construction, plus the two directions
  of the same checkpoint pair under a tiny tolerance -> opposite
  non-unchanged verdicts): explicit three-distinct-verdicts guard;
  exact filtered-listing parity for ALL THREE verdicts + verbatim
  detail-getter payloads + ordering; explicit disjoint three-group
  partition whose union is the full listing; empty model -> `[]` for
  every verdict; unknown model -> `FileNotFoundError`; cross-model
  isolation; read-only (manifest file-set snapshots unchanged);
  repeated-call determinism. (`ComparisonVerdict` added to the test
  module's schema imports.)
- `tests/test_comparison_api.py` — 2 API tests via the existing
  `_prepare`/`_comp` helpers: singleton verdict groups derived from
  the authoritative listing (3 posted records, one per verdict) +
  detail-getter parity + raw-byte-identical x3 for every verdict +
  partition + no-side-effects; unknown model + valid verdict -> 404;
  unsupported values (case variant, interior-space, numeric,
  `better`) -> 422 and unknown-model + invalid-verdict -> 422
  (pre-handler precedence); cross-model isolation under every
  verdict; M5 listing/getter + ghost-id 404 (no route capture);
  M26/M29/M31/M37 sibling regressions; M38 evaluations-by-state-kind
  regression (both kinds, listing parity); OpenAPI 71 assertions
  (route order by-split < by-verdict < generic, GET-only, tag
  `comparison`, items `$ref ComparisonRecord`, `verdict` param `$ref
  ComparisonVerdict`).

Zero bring-up iterations: engine 3/3 focused AND full module 37/37,
API 2/2 focused AND full module 15/15 — all first runs.

Results (nothing certified from partial runs):

- full suite, twice consecutively: **505 passed @ 131.90 s**, then
  (after cleaning accumulated sandboxes, no live process)
  **505 passed @ 129.48 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**;
- OpenAPI count sweep: every hard-coded count assertion searched and
  updated — **23 assertions `== 70` -> `== 71` across 8 test modules
  (one more than M38's 22: M38's own new API test added an
  assertion), 0 `== 70` remaining**, plus 15 ladder lines extended
  with `+ 1 (M39 comparisons by-verdict)`.

## Live Smoke

`smoke_m39_live.py` (committed) — 32 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
32/32 PASS each, exit 0, first attempt**, on port **8760** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m39-smoke-baseline-inventory.json`. (32 rather than 33 named
checks: the three per-verdict parity responses are certified together
in one C1 check with per-group detail, plus a separate all-records
detail-parity check.)

- A: exact 96/4,002,745/0 audit + inventory saved + M4/M5/M6/M11/M16/
  M18–M38 pre-state (including the M36/M37 by-split and M38
  by-state-kind surfaces) + dashboard hash + OpenAPI 71 pre-state +
  known pair 200.
- B (DISCOVERY, never assumed): from the live M5 listing + the OpenAPI
  schema — **8 comparisons, verdict distribution `improved -> 3,
  unchanged -> 3, regressed -> 2` (ALL THREE groups non-empty)**, ASC
  `(created_at, comparison_id)`; enum values
  `['improved', 'regressed', 'unchanged']` — matching the M38-end
  grounding exactly.
- C: ALL THREE verdicts -> 200 + EXACT records == listing filtered
  locally (3/3/2); verbatim detail-getter parity for all 8 records
  (verdict/per-side losses included).
- D: three GETs raw-byte-identical.
- E: ALL enum verdicts -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid verdict -> 404; unsupported values
  (interior-space, `IMPROVED`, `1`) -> 422 with the enum detail;
  unsupported verdict + UNKNOWN model -> 422 (pre-handler
  precedence).
- G: the three groups partition the full listing exactly (3+3+2 == 8,
  every record exactly once, no record in two groups, persisted
  verdict verbatim).
- H: every group preserves the authoritative M5 order exactly.
- I: M26/M29/M31/M37 comparison histories unchanged (6/5/1, 8, 8,
  by-split 8/0/0 — all with listing parity).
- J: M22 suite runs (10/10/summary), M24/M28/M30/M36 evaluation
  histories (3/3/3, 16, 16, by-split 14/2/0), **M38
  by-state-kind unchanged (checkpoint 9 / current 7 + parity)**,
  M27/M32/M33 samples (4/0/0, 4, 2), M34/M23 gate decisions (4
  by-comparison, 1 by-policy).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe
  registries unchanged.
- M: OpenAPI exactly 71 paths, new path once, order by-split <
  by-verdict < generic, M38 evaluations-by-state-kind path still
  present once; storage zero drift (below).

## Production Storage

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: 0 changed, 0 missing, 0 new files across all three
  runs;
- final audit after the server stopped: `sha256sum -c m39_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- M39 wrote NOTHING to production: the endpoint is pure data access
  over existing M5 comparison manifests. Production remains
  byte-identical through M39.

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5 listings + detail getters, M22 suite-run
  histories, M23 gate decisions by policy, M24/M28/M30/M36/M38
  evaluation groupings, M26/M29/M31/M37 comparison groupings, M27/
  M32/M33 sample and sample-quality groupings, M34 gate decisions by
  comparison, M35 workflow history by recipe (model-scoped + global
  M12 lineage), M16 pairings, M17 dashboard (full hash), M2/M9/M12/
  M14 registries.
- Full suite green at 505 with no weakened tests.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged.
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M39: add comparison history by verdict` — `app/comparison.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 11 touched test
   modules (2 with new tests, 9 with the OpenAPI 71 sweep + ladders),
   `smoke_m39_live.py`, `M39_final_report.md`.
2. follow-up inventory commit adding `m39_pre.sha256` at the repo root
   (M21–M39 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed).

## Next Milestone

End-of-M39 live inspection (production server, real payloads + schema
inspection): every `SampleRecord` persists a top-level `strategy:
SampleStrategy` — a TWO-value schema enum (greedy = deterministic
argmax with no RNG; temperature = one deterministic RNG stream seeded
by the request's explicit seed; "a request never mixes them") — with
production distribution **greedy -> 2, temperature -> 2 (both groups
non-empty)**, and the M15 listing is documented-deterministic in
`(created_at, sample_id)` order; `b5bc905326b6` owns no samples
(natural valid-empty for both strategies). Runner-up axes inspected:
`GateDecision.decision: GateDecisionResult` (passed 7 / failed 4 —
viable, slightly larger records), `WorkflowStatus` (completed 9 /
failed 3 / stopped 1 — viable, but the workflow family gained M35
recently), `SuiteRunStatus` (only `completed` exists in production —
single-valued, weak grounding). The M15 sample listing's strategy axis
is the smallest logically justified next surface and the exact twin of
the certified M36–M39 enum-contract groupings.

Copy-ready M40 prompt:

> M40: add sample history by strategy. ONE read-only MODEL-SCOPED
> endpoint
> `GET /api/v1/models/{model_id}/samples/by-strategy/{strategy}`
> returning the authoritative M15 sample listing of the model
> filtered by the persisted top-level `SampleRecord.strategy` matched
> VERBATIM against the existing `SampleStrategy` schema enum
> (greedy/temperature — never inferred from temperature values,
> seeds, filenames or timestamps; never resolved or rewritten; no
> sample is regenerated). Inherit the exact M15 `(created_at,
> sample_id)` ASC ordering; valid strategy + zero records -> `200 []`;
> unknown model -> `404`; unsupported strategy value -> `422` via the
> schema enum path parameter (native FastAPI validation, never a
> registry-style 404; validation fires pre-handler even for an
> unknown model — the M36–M39 contract). Register the route after
> `samples/by-tokenizer/{tokenizer_id}` and BEFORE the generic
> `/samples/{sample_id}` detail route if that ordering exists in the
> family (inspect first; keep the family coherent). OpenAPI 71 -> 72
> (search and update EVERY hard-coded count); tests 505 -> ~510
> (report the actual); full suite twice, compileall, pyflakes, stale
> `/tmp/forge-tests-*` cleanup (verify no live process first). Live
> smoke on port 8761 with >= 17 named checks: DISCOVER the strategy
> distribution live (grounding facts from M39: `4a0a871886ef` owns 4
> samples, greedy -> 2 / temperature -> 2 — both groups non-empty;
> `b5bc905326b6` owns none — natural valid-empty for both
> strategies), two-group partition + parity + ordering +
> byte-identical x3, 404/422/200-[] separation, and M2–M39
> regressions (dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`,
> 96/4,002,745/0 storage, SHA256 audit 0/0/0). Minimum-files rule: no
> persistence/cache/index/DB/migration writes; no sample regeneration.
> Final report with exactly the 9 sections Milestone, Baseline,
> Implementation, Tests, Live Smoke, Production Storage, Regression /
> Compatibility, Git / Commit, Next Milestone — §9 a complete
> copy-ready M41 prompt grounded ONLY in facts you discover live at
> the end of M40 (inspect the remaining persisted record
> relationships yourself — e.g. `GateDecision.decision:
> GateDecisionResult` passed 7 / failed 4 and `WorkflowStatus`
> completed 9 / failed 3 / stopped 1 were the M39 runner-ups; pick
> the next smallest read-only surface; never invent distribution
> facts), including the same M41 auto-generation requirement. Commit
> `M40: add sample history by strategy`, push (never force-push),
> then the follow-up inventory commit `m40_pre.sha256` per the
> M21–M39 convention; only egg-info may remain untracked.
