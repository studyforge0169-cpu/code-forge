# M40 Final Report — Sample History by Strategy

## Milestone

M40 adds ONE narrow read-only MODEL-SCOPED access path to ai-model-forge:

```
GET /api/v1/models/{model_id}/samples/by-strategy/{strategy}
```

It answers "which immutable M15 sample records of this model were
generated with this decoding strategy?" — the model's authoritative M15
listing filtered by the persisted top-level `strategy` field recorded
in each `SampleRecord` (the `SampleStrategy` schema enum: `greedy` =
deterministic argmax with no RNG, `temperature` = one deterministic
RNG stream seeded by the request's explicit seed; matched VERBATIM —
membership NEVER inferred from sample ids, prompt text, generated token
ids, temperature values, seed presence, filenames or manifest paths,
and NEVER recalculated from temperature/seed/other fields; no sample
is regenerated). The route is registered BEFORE the generic
`/samples/{sample_id}` getter and alongside the by-checkpoint (M27) and
by-tokenizer (M32) groupings. OpenAPI 71 -> 72.

Contract (exactly as specified):

- ordering inherited: the exact authoritative M15 `(created_at,
  sample_id)` ASCENDING order;
- valid strategy + zero matching records -> `200` + `[]` (NEVER 404);
- unknown model + valid strategy -> `404` (exactly like the sibling
  groupings);
- unsupported strategy value -> `422` via the schema enum path
  parameter (native FastAPI/Pydantic validation, never manually
  converted to a registry-style 404) — validation fires pre-handler,
  so unknown-model + invalid-strategy is also 422, matching
  M36–M39;
- minimum-files rule honored: no persistence, cache, index, DB,
  migration or manifest writes — everything derives from the existing
  M15 sample manifests; repeated GETs are byte-identical.

## Baseline

The sandbox had reset AGAIN before this milestone (reset #9, the same
signature as before: HEAD was `f86b670` "Add files via upload" with
the whole tree untracked, the production data directory GONE entirely,
and the venv gone too). The standard recovery runbook was executed
before any coding:

1. `git fetch` -> FETCH_HEAD `1a2b1d2` (the authoritative remote tip),
   `git reset --hard FETCH_HEAD` -> HEAD == `1a2b1d2`, tree clean
   except the known egg-info (no force-push; nothing reconstructed
   from assumptions);
2. production restored from `code-forge/code forge.zip` by extracting
   ONLY its `ai-model-forge-data/` prefix (96 members);
3. `sha256sum -c m39_pre.sha256` from the data root -> **0 non-OK**
   (byte-identical through M39: 96 files / 4,002,745 bytes / 0 .tmp);
4. venv rebuilt from scratch (`python3 -m venv ~/.venv` +
   `pip install -e ai-model-forge[dev] pyflakes`), import checks OK.

Baseline then re-proved BEFORE implementation:

- full suite: **505 passed @ 142.36 s** (exact M39 ladder);
- OpenAPI: **71 paths**;
- `m40_pre.sha256` captured pre-implementation (96 entries);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes.

## Implementation

Three production files + docs, mirroring the M27/M32/M36–M39 pattern:

- `app/sampling.py` — new engine method `list_samples_for_strategy(
  model_id, strategy: SampleStrategy)` placed directly after
  `list_samples_for_tokenizer`. It reuses the model's authoritative
  M15 listing (exact engine parse + deterministic `(created_at,
  sample_id)` ASC order) and filters by the persisted `strategy`
  VERBATIM. Docstring records the contract: strategies have NO
  registry (unlike the checkpoint/tokenizer axes) — the enum IS the
  contract, so unsupported values are rejected with 422 at the API
  boundary and never reach the engine, while an unknown model raises
  FileNotFoundError exactly like the siblings; the strategy is never
  recalculated and generation is never invoked; a valid strategy with
  zero samples returns `[]`; read-only, never writes.
  `SampleStrategy` was already imported here (used by the generation
  path).
- `app/engine.py` — `ModelForge` facade `list_samples_for_strategy`
  delegating to the engine, placed after the by-tokenizer delegation
  (+`SampleStrategy` import).
- `app/api.py` — route `@api.get("/models/{model_id}/samples/
  by-strategy/{strategy}", response_model=list[SampleRecord],
  tags=["sampling"])` with `strategy: SampleStrategy` as the path
  parameter (this is what makes unsupported values a native
  pre-handler 422), placed after the by-tokenizer route and before
  the generic `/samples/{sample_id}` getter; landing-page bullet;
  endpoint `<li>`; sampling-routes section comment extended with the
  M40 line (+`SampleStrategy` import; all-three-files import rule
  honored).
- `README.md` — M40 section, counts 505 -> 510 (two places), layout
  line `M27/M32/M40`.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere.

## Tests

Five new tests (505 -> 510, exactly as targeted):

- `tests/test_sampling.py` — 3 engine tests via the cached
  `_m40_state(env)` fixture (greedy x1 + temperature x1 on the big
  model — self-seeding both groups — plus a fresh model created with
  NO samples at all as the natural valid-empty case): exact
  filtered-listing parity for BOTH enum strategies + verbatim
  detail-getter payloads + ordering + the greedy-records-persist-
  temperature-None check; explicit disjoint partition over both
  strategies whose union is the full listing; empty model -> `[]` for
  both; unknown model -> `FileNotFoundError`; cross-model isolation
  (each model's group a subset of its own listing, listings disjoint);
  read-only (shared root-level `samples/` manifest tree snapshot
  unchanged); repeated-call determinism.
- `tests/test_sampling.py` (API) — 2 API tests via the existing
  `_http_env`/`_gen_body` helpers: 200 + exact records + listing
  parity + detail-getter parity + raw-byte-identical x3 for both
  strategies + partition + no-side-effects; unknown model + valid
  strategy -> 404; unsupported values (case variant, interior-space,
  numeric, `top_k`) -> 422 and unknown-model + invalid-strategy ->
  422 (pre-handler precedence); cross-model isolation under both
  strategies; M15 listing/getter + ghost-id 404 (no route capture);
  M27 by-checkpoint + M32 by-tokenizer regressions (listing-derived
  parity); M38 evaluations-by-state-kind + M39 comparisons-by-verdict
  regressions; OpenAPI 72 assertions (route order by-tokenizer <
  by-strategy < generic, GET-only, tag `sampling`, items `$ref
  SampleRecord`, `strategy` param `$ref SampleStrategy`).

One honest bring-up iteration: the first API test run failed on my
M38-regression premise — this harness's `_http_env` training run
persists NO M4 evaluation records, so the evaluations listing is empty
and the regression is the natural-valid-empty form, not
"training-internal evals exist". Fixed to listing-parity form (both
kinds -> 200 + [] == filtered empty listing) before any full run; all
other checks passed unchanged.

Results (nothing certified from partial runs):

- focused: engine 3/3 and full module 27/27 first run; API 5/5 after
  the one fix above;
- full suite, twice consecutively: **510 passed @ 142.36 s**, then
  (after cleaning 10 accumulated sandboxes, no live process)
  **510 passed @ 153.32 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**;
- OpenAPI count sweep: every hard-coded count assertion searched and
  updated — **24 assertions `== 71` -> `== 72` across 8 test modules
  (one more than M39's 23: M39's own new API test added an
  assertion), 0 `== 71` remaining**, plus 16 ladder lines extended
  with `+ 1 (M40 samples by-strategy)`.

## Live Smoke

`smoke_m40_live.py` (committed) — 33 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
33/33 PASS each, exit 0, first attempt**, on port **8761** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m40-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + M4/M5/M6/M11/M15/
  M16/M18–M39 pre-state (including the M36/M37 by-split, M38
  by-state-kind and M39 by-verdict surfaces) + dashboard hash +
  OpenAPI 72 pre-state + known pair 200.
- B (DISCOVERY, never assumed): from the live M15 listing + the
  OpenAPI schema — **4 samples, strategy distribution
  `greedy -> 2, temperature -> 2` (both groups non-empty)**, ASC
  `(created_at, sample_id)`; enum values `['greedy', 'temperature']`
  — matching the M39-end grounding exactly.
- C: BOTH strategies -> 200 + EXACT records == listing filtered
  locally (2/2); verbatim detail-getter parity for all 4 records
  (prompt, token ids, output text, temperature included).
- D: three GETs raw-byte-identical.
- E: ALL enum strategies -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid strategy -> 404; unsupported values
  (interior-space, `GREEDY`, `3`) -> 422 with the enum detail;
  unsupported strategy + UNKNOWN model -> 422 (pre-handler
  precedence).
- G: the two groups partition the full listing exactly (2+2 == 4,
  every record exactly once, no record in two groups, persisted
  strategy verbatim).
- H: every group preserves the authoritative M15 order exactly.
- I: M27/M32 sample histories + the M15 listing itself unchanged
  (4/0/0 by-checkpoint, 4 by-tokenizer, listing parity).
- J: M22 suite runs (10/10/summary), M24/M28/M30/M36/M38 evaluation
  histories (3/3/3, 16, 16, by-split 14/2/0, by-state-kind 9/7),
  M26/M29/M31/M37/M39 comparison histories (6/5/1, 8, 8, by-split
  8/0/0, by-verdict 3/3/2), M33 sample-quality (2), M34/M23 gate
  decisions (4 by-comparison, 1 by-policy).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe
  registries unchanged.
- M: OpenAPI exactly 72 paths, new path once, order by-tokenizer <
  by-strategy < generic, M39 comparisons-by-verdict path still
  present once; storage zero drift (below).

## Production Storage

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: 0 changed, 0 missing, 0 new files across all three
  runs;
- final audit after the server stopped: `sha256sum -c m40_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- M40 wrote NOTHING to production: the endpoint is pure data access
  over existing M15 sample manifests. Production remains
  byte-identical through M40 (including across the reset-#9
  restoration).

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5/M15 listings + detail getters, M22 suite-run
  histories, M23 gate decisions by policy, M24/M28/M30/M36/M38
  evaluation groupings, M26/M29/M31/M37/M39 comparison groupings,
  M27/M32 sample groupings, M33 sample-quality, M34 gate decisions by
  comparison, M35 workflow history by recipe (model-scoped + global
  M12 lineage), M16 pairings, M17 dashboard (full hash), M2/M9/M12/
  M14 registries.
- Full suite green at 510 with no weakened tests; the one edited
  assertion was a wrong premise in MY new regression check (corrected
  to the natural-empty form), never a weakening of existing coverage.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged.
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M40: add sample history by strategy` — `app/sampling.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 10 touched test
   modules (1 with new tests, 9 with the OpenAPI 72 sweep + ladders),
   `smoke_m40_live.py`, `M40_final_report.md`.
2. follow-up inventory commit adding `m40_pre.sha256` at the repo root
   (M21–M40 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed).

## Next Milestone

End-of-M40 live inspection (production server, real payloads + schema
inspection): every `GateDecision` persists a top-level `decision:
GateDecisionResult` — a TWO-value schema enum (passed/failed) — with
production distribution **passed -> 7, failed -> 4 (both groups
non-empty)** across the 11 gate decisions of `4a0a871886ef`; the M6
gate-decision listing is deterministic in `(created_at, decision_id)`
ASC order (verified live), and the family already groups by-policy
(M23) and by-comparison (M34) — the decision axis is the missing
enum-contract twin. `b5bc905326b6` owns NO gate decisions (natural
valid-empty for both values). Runner-up inspected: `WorkflowStatus`
(completed 9 / failed 3 / stopped 1 — viable, but the workflow family
gained M35 recently and its three-value split includes a singleton
group); `SuiteRunStatus` remains single-valued in production.

Copy-ready M41 prompt:

> M41: add gate-decision history by decision. ONE read-only
> MODEL-SCOPED endpoint
> `GET /api/v1/models/{model_id}/gates/decisions/by-decision/{decision}`
> returning the authoritative M6 gate-decision listing of the model
> filtered by the persisted top-level `GateDecision.decision` matched
> VERBATIM against the existing `GateDecisionResult` schema enum
> (passed/failed — never recalculated from verdicts, hints, losses or
> policies; never resolved or rewritten; no gate is re-evaluated).
> Inherit the exact M6 `(created_at, decision_id)` ASC ordering; valid
> decision + zero records -> `200 []`; unknown model -> `404`;
> unsupported decision value -> `422` via the schema enum path
> parameter (native FastAPI validation, never a registry-style 404;
> validation fires pre-handler even for an unknown model — the
> M36–M40 contract). Register the route after
> `gates/decisions/by-comparison/{comparison_id}` and BEFORE the
> generic `/gates/decisions/{decision_id}` detail route if that
> ordering exists in the family (inspect first; keep the family
> coherent with by-policy/by-comparison). OpenAPI 72 -> 73 (search and
> update EVERY hard-coded count); tests 510 -> ~515 (report the
> actual); full suite twice, compileall, pyflakes, stale
> `/tmp/forge-tests-*` cleanup (verify no live process first). Live
> smoke on port 8762 with >= 17 named checks: DISCOVER the decision
> distribution live (grounding facts from M40: `4a0a871886ef` owns 11
> gate decisions, passed -> 7 / failed -> 4 — both groups non-empty;
> `b5bc905326b6` owns none — natural valid-empty for both values),
> two-group partition + parity + ordering + byte-identical x3,
> 404/422/200-[] separation, and M2–M40 regressions (M23 by-policy +
> M34 by-comparison gate surfaces, dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`,
> 96/4,002,745/0 storage, SHA256 audit 0/0/0). Minimum-files rule: no
> persistence/cache/index/DB/migration writes; no gate re-evaluation —
> the persisted decision is filtered only. Final report with exactly
> the 9 sections Milestone, Baseline, Implementation, Tests, Live
> Smoke, Production Storage, Regression / Compatibility, Git / Commit,
> Next Milestone — §9 a complete copy-ready M42 prompt grounded ONLY
> in facts you discover live at the end of M41 (inspect the remaining
> persisted record relationships yourself — e.g. `WorkflowStatus`
> completed 9 / failed 3 / stopped 1 was the M40 runner-up; pick the
> next smallest read-only surface; never invent distribution facts),
> including the same M42 auto-generation requirement. Commit
> `M41: add gate-decision history by decision`, push (never
> force-push), then the follow-up inventory commit `m41_pre.sha256`
> per the M21–M40 convention; only egg-info may remain untracked.
