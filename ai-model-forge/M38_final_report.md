# M38 Final Report — Evaluation History by State Kind

## Milestone

M38 adds ONE narrow read-only MODEL-SCOPED access path to ai-model-forge:

```
GET /api/v1/models/{model_id}/evaluations/by-state-kind/{state_kind}
```

It answers "which immutable M4 evaluation records of this model measured
which kind of model state?" — the model's authoritative M4 listing
filtered by the persisted top-level `state_kind` field recorded in each
`EvaluationRecord` (the `EvalStateKind` schema enum: `current` measured
the model's published weights, `checkpoint` measured one immutable
stored checkpoint; matched VERBATIM — membership NEVER inferred from
filenames, directories, timestamps, eval ids or hashes, and NEVER from
`checkpoint_id` nullability: that nullability is a schema CONSEQUENCE
of the persisted state kind, not its source; the persisted value is
never resolved or rewritten). The route is registered BEFORE the
generic `/evaluations/{eval_id}` getter and alongside the by-checkpoint
(M24), by-dataset (M28), by-tokenizer (M30) and by-split (M36)
groupings. OpenAPI 69 -> 70.

Contract (exactly as specified):

- ordering inherited: the exact authoritative M4 `(created_at, eval_id)`
  ASCENDING order;
- valid state kind + zero matching records -> `200` + `[]` (NEVER 404);
- unknown model + valid state kind -> `404` (exactly like the sibling
  groupings);
- unsupported state-kind value -> `422` via the schema enum path
  parameter (native FastAPI/Pydantic validation, never manually
  converted to a registry-style 404) — validation fires pre-handler,
  so unknown-model + invalid-state-kind is also 422, matching
  M36/M37;
- minimum-files rule honored: no persistence, cache, index, DB,
  migration or manifest writes — everything derives from the existing
  M4 evaluation manifests; repeated GETs are byte-identical.

## Baseline

No sandbox reset this session: HEAD == FETCH_HEAD == `aaf66af` (M37
implementation `a9c3843` + inventory `aaf66af`), tree clean except the
known egg-info. Baseline re-proved BEFORE any changes:

- full suite: **495 passed @ 130.36 s** (exact M37 ladder);
- OpenAPI: **69 paths**;
- production: **96 files / 4,002,745 bytes / 0 .tmp**;
- `m38_pre.sha256` captured pre-implementation; `m37_pre.sha256`
  check -> **0 non-OK** (zero drift);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes.

## Implementation

Three production files + docs, mirroring the M24/M28/M30/M36 pattern:

- `app/evaluation.py` — new engine method
  `list_evaluations_for_state_kind(model_id, state_kind:
  EvalStateKind)` placed directly after `list_evaluations_for_split`.
  It reuses the model's authoritative M4 listing (exact engine parse +
  deterministic `(created_at, eval_id)` ASC order) and filters by the
  persisted `state_kind` VERBATIM. Docstring records the contract:
  state kinds have NO registry (unlike the checkpoint/dataset/
  tokenizer axes) — the enum IS the contract, so unsupported values
  are rejected with 422 at the API boundary and never reach the
  engine, while an unknown model raises FileNotFoundError exactly like
  the siblings; membership never comes from `checkpoint_id`
  nullability; a valid kind with zero evaluations returns `[]`;
  read-only, never writes. `EvalStateKind` was already imported here
  (used by the run path).
- `app/engine.py` — `ModelForge` facade `list_evaluations_for_state_kind`
  delegating to the engine, placed after the by-split delegation
  (+`EvalStateKind` import).
- `app/api.py` — route `@api.get("/models/{model_id}/evaluations/
  by-state-kind/{state_kind}", response_model=list[EvaluationRecord],
  tags=["evaluation"])` with `state_kind: EvalStateKind` as the path
  parameter (this is what makes unsupported values a native
  pre-handler 422), placed after the by-split route and before the
  generic `/evaluations/{eval_id}` getter; landing-page bullet;
  endpoint `<li>`; evaluation-routes section comment extended with the
  M36 and M38 lines (the banner had only listed M28/M30 — now
  complete) (+`EvalStateKind` import; all-three-files import rule
  honored, so no repeat of the M36 ForwardRef openapi() crash).
- `README.md` — M38 section, counts 495 -> 500 (two places), layout
  line `M24/M28/M30/M36/M38`.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere.

## Tests

Five new tests (495 -> 500, exactly as targeted):

- `tests/test_evaluation.py` — 3 engine tests via the cached
  `_m38_env(env)` fixture (a trained model with checkpoint x2 + current
  x1 evaluations; the M28 empty model for the empty groups): exact
  filtered-listing parity for BOTH enum kinds + verbatim detail-getter
  payloads + ordering; explicit disjoint partition over both kinds
  whose union is the full listing; the nullability-is-a-consequence
  check; empty model -> `[]` for both kinds; unknown model ->
  `FileNotFoundError`; cross-model isolation; read-only (manifest
  file-set snapshots unchanged); repeated-call determinism.
  (`EvalStateKind` added to the test module's schema imports.)
- `tests/test_evaluation_api.py` — 2 API tests via the existing
  `_prepare`/`_eval_cfg` helpers: 200 + exact records + listing parity
  + detail-getter parity + raw-byte-identical x3 + partition +
  no-side-effects; unknown model + valid kind -> 404; unsupported
  values (case variant, interior-space, numeric, `weights`) -> 422 and
  unknown-model + invalid-kind -> 422 (pre-handler precedence);
  cross-model isolation under both kinds; M4 listing/getter + ghost-id
  404 (no route capture); M24/M28/M30/M36 sibling regressions
  (listing-derived parity); OpenAPI 70 assertions (route order
  by-split < by-state-kind < generic, GET-only, tag `evaluation`,
  items `$ref EvaluationRecord`, `state_kind` param `$ref
  EvalStateKind`).

No bring-up iterations this time: engine 3/3 and API 2/2 passed on the
first focused runs (the M36/M37 lessons — self-seeding fixtures and
listing-derived expectations — were applied from the start).

Results (nothing certified from partial runs):

- focused: engine 3/3, API 2/2;
- full suite, twice consecutively: **500 passed @ 132.81 s**, then
  (after cleaning 4 accumulated sandboxes, no live process)
  **500 passed @ 132.49 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**;
- OpenAPI count sweep: every hard-coded count assertion searched and
  updated — **22 assertions `== 69` -> `== 70` across 8 test modules
  (one more than M37's 21: M37's own new API test added an
  assertion), 0 `== 69` remaining**, plus 15 ladder lines extended
  with `+ 1 (M38 evaluations by-state-kind)`.

## Live Smoke

`smoke_m38_live.py` (committed) — 33 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
33/33 PASS each, exit 0, first attempt**, on port **8759** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m38-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + M4/M5/M6/M11/M16/
  M18–M37 pre-state (including the M36 evaluations-by-split and M37
  comparisons-by-split surfaces) + dashboard hash + OpenAPI 70
  pre-state + known pair 200.
- B (DISCOVERY, never assumed): from the live M4 listing + the OpenAPI
  schema — **16 evaluations, state-kind distribution
  `checkpoint -> 9, current -> 7`**, ASC `(created_at, eval_id)`;
  enum values `['current', 'checkpoint']` — matching the M37-end
  grounding exactly.
- C: BOTH kinds -> 200 + EXACT records == listing filtered locally
  (9 and 7); `checkpoint_id` null exactly for the current records
  (consequence, never source); verbatim detail-getter parity for all
  16 records.
- D: three GETs raw-byte-identical.
- E: ALL enum kinds -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid kind -> 404; unsupported values
  (interior-space, `CHECKPOINT`, `2`) -> 422 with the enum detail;
  unsupported kind + UNKNOWN model -> 422 (pre-handler precedence).
- G: the two groups partition the full listing exactly (9+7 == 16,
  every record exactly once, no record in two groups, persisted kind
  verbatim).
- H: every group preserves the authoritative M5 order exactly.
- I: M24/M28/M30/M36 evaluation histories unchanged (3/3/3, 16, 16,
  by-split 14/2/0 — all with listing parity).
- J: M22 suite runs (10/10/summary), M26/M29/M31/M37 comparison
  histories (6/5/1, 8, 8, by-split 8/0/0), M27/M32/M33 samples
  (4/0/0, 4, 2), M34/M23 gate decisions (4 by-comparison,
  1 by-policy).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe registries
  unchanged.
- M: OpenAPI exactly 70 paths, new path once, order by-split <
  by-state-kind < generic, M37 comparisons-by-split path still present
  once; storage zero drift (below).

## Production Storage

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: 0 changed, 0 missing, 0 new files across all three
  runs;
- final audit after the server stopped: `sha256sum -c m38_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- M38 wrote NOTHING to production: the endpoint is pure data access
  over existing M4 evaluation manifests. Production remains
  byte-identical through M38.

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5 listings + detail getters, M22 suite-run
  histories, M23 gate decisions by policy, M24/M28/M30/M36 evaluation
  groupings, M26/M29/M31/M37 comparison groupings, M27/M32/M33 sample
  and sample-quality groupings, M34 gate decisions by comparison, M35
  workflow history by recipe (model-scoped + global M12 lineage), M16
  pairings, M17 dashboard (full hash), M2/M9/M12/M14 registries.
- Full suite green at 500 with no weakened tests.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged.
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M38: add evaluation history by state kind` — `app/evaluation.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 11 touched test
   modules (2 with new tests, 9 with the OpenAPI 70 sweep + ladders),
   `smoke_m38_live.py`, `M38_final_report.md`.
2. follow-up inventory commit adding `m38_pre.sha256` at the repo root
   (M21–M38 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed).

## Next Milestone

End-of-M38 live inspection (production server, real payloads + schema
inspection): every `ComparisonRecord` persists a top-level `verdict:
ComparisonVerdict` — a THREE-value schema enum (improved / regressed /
unchanged, "ONLY about measured loss on one probe") — with production
distribution **improved -> 3, unchanged -> 3, regressed -> 2: ALL
THREE groups non-empty** (the richest partition grounding yet; `b5bc905326b6`
owns no comparisons, so every verdict there is a natural valid-empty).
Other candidates inspected and set aside: `SampleStrategy`
(greedy 2 / temperature 2 — viable but a younger listing),
`GateDecision.decision` (passed 7 / failed 4), `SuiteRunStatus` (only
`completed` exists in production — single-valued), `WorkflowStatus`
(completed 9 / failed 3 / stopped 1 — viable, but the M11 workflow
family already gained M35 recently). The comparison listing's verdict
axis is the smallest logically justified next surface: the exact M5
twin of the just-certified M36/M37/M38 enum-contract groupings.

Copy-ready M39 prompt:

> M39: add comparison history by verdict. ONE read-only MODEL-SCOPED
> endpoint
> `GET /api/v1/models/{model_id}/comparisons/by-verdict/{verdict}`
> returning the authoritative M5 comparison listing of the model
> filtered by the persisted top-level `ComparisonRecord.verdict`
> matched VERBATIM against the existing `ComparisonVerdict` schema
> enum (improved/regressed/unchanged — never inferred from loss
> values, deltas, tolerances, per-side losses or hashes; never
> recalculated or rewritten). Inherit the exact M5 `(created_at,
> comparison_id)` ASC ordering; valid verdict + zero records -> `200
> []`; unknown model -> `404`; unsupported verdict value -> `422` via
> the schema enum path parameter (native FastAPI validation, never a
> registry-style 404; validation fires pre-handler even for an
> unknown model — the M36/M37/M38 contract). Register the route after
> `comparisons/by-split/{split}` and BEFORE the generic
> `/comparisons/{comparison_id}`. OpenAPI 70 -> 71 (search and update
> EVERY hard-coded count); tests 500 -> ~505 (report the actual);
> full suite twice, compileall, pyflakes, stale `/tmp/forge-tests-*`
> cleanup (verify no live process first). Live smoke on port 8760
> with >= 17 named checks: DISCOVER the verdict distribution live
> (grounding facts from M38: `4a0a871886ef` owns 8 comparisons,
> improved -> 3 / unchanged -> 3 / regressed -> 2 — all three groups
> non-empty; `b5bc905326b6` owns none — natural valid-empty for every
> verdict), three-group partition + parity + ordering +
> byte-identical x3, 404/422/200-[] separation, and M2–M38
> regressions (dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`,
> 96/4,002,745/0 storage, SHA256 audit 0/0/0). Minimum-files rule: no
> persistence/cache/index/DB/migration writes; the verdict is NEVER
> recalculated from the persisted losses — filtered only. Final
> report with exactly the 9 sections Milestone, Baseline,
> Implementation, Tests, Live Smoke, Production Storage, Regression /
> Compatibility, Git / Commit, Next Milestone — §9 a complete
> copy-ready M40 prompt grounded ONLY in facts you discover live at
> the end of M39 (inspect the remaining persisted record
> relationships yourself; pick the next smallest read-only surface;
> never invent distribution facts), including the same M40
> auto-generation requirement. Commit `M39: add comparison history by
> verdict`, push (never force-push), then the follow-up inventory
> commit `m39_pre.sha256` per the M21–M38 convention; only egg-info
> may remain untracked.
