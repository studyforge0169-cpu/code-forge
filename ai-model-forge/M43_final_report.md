# M43 Final Report — Gate-Decision History by Verdict

## Recovery & Baseline

No reset this session (recovery runbook standby not needed): HEAD ==
FETCH_HEAD == `23c7bea` (M42 implementation `ee2daca` + inventory
`23c7bea`), tree clean except the known egg-info. Pre-implementation
state verified and recorded BEFORE any change:

- full suite: **520 passed @ 153.17 s** (exact M42 ladder);
- OpenAPI: **74 paths**;
- production: **96 files / 4,002,745 bytes / 0 .tmp**;
- `m42_pre.sha256` check -> **0 non-OK** (zero drift);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes, venv
  intact;
- `m43_pre.sha256` captured (96 entries) as the pre-milestone
  inventory;
- expected live grounding (verdicts improved -> 4 / regressed -> 3 /
  unchanged -> 2 + 2 null) re-verified by discovery in the smoke.

## Implementation

Three production files + docs, mirroring the M41/M42 pattern:

- `app/gates.py` — new engine method `list_decisions_for_verdict(
  model_id, verdict: ComparisonVerdict)` placed directly after
  `list_decisions_for_decision`. It reuses the model's authoritative
  M6 listing (exact engine parse + deterministic `(created_at,
  decision_id)` ASC order) and filters by the persisted `verdict`
  VERBATIM. Docstring records the full contract: the verdict is the
  LOSS-ONLY comparison verdict recorded verbatim by the M6 run
  (deliberately DISTINCT from the M41 policy decision — an improved
  candidate can still fail a `minimum_loss` ceiling); the field is
  OPTIONAL — threshold-only gates (`baseline_type="minimum_loss"`)
  judge NO comparison and keep `verdict=None`; None is not an enum
  value and NEVER matches any request (there is deliberately NO route
  for None — threshold-only decisions belong to NO by-verdict group
  and stay in the generic M6 listing untouched); verdicts have NO
  registry (the enum IS the contract, exactly like M41) so unsupported
  values are rejected with 422 at the API boundary and never reach the
  engine, while an unknown model raises FileNotFoundError exactly like
  the siblings; a valid verdict with zero matching decisions returns
  `[]`; read-only, never writes. `ComparisonVerdict` was already
  imported here.
- `app/engine.py` — facade `list_gate_decisions_for_verdict` delegating
  to the engine, placed after the by-decision delegation
  (`ComparisonVerdict` already imported; facade signature matches the
  unannotated sibling style).
- `app/api.py` — route `@api.get("/models/{model_id}/gates/decisions/
  by-verdict/{verdict}", response_model=list[GateDecision],
  tags=["gates"])` with `verdict: ComparisonVerdict` as the path
  parameter (native pre-handler 422 for unsupported values; "none"/
  "null" are NOT enum values and 422 — no route represents None),
  placed after the M41 by-decision route and before the generic
  `/gates/decisions/{decision_id}` getter; landing-page bullet;
  endpoint `<li>`; gate-routes section comment extended with the M43
  line (`ComparisonVerdict` already imported for M39; no new import
  needed anywhere in api.py).
- `README.md` — M43 section, counts 520 -> 525 (two places), gates.py
  layout line extended to `M23/M34/M41/M43`.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere. The gate-decision history family is now
complete: M23 by-policy, M34 by-comparison, M41 by-decision, M43
by-verdict.

## Tests

Five new tests (520 -> 525, exactly as targeted):

- `tests/test_gates.py` — 3 engine tests via the cached `_m43_state(
  env)` fixture, which deterministically creates ALL THREE verdicts
  (improved: early -> final A checkpoint; regressed: final A ->
  final-B-regressed checkpoint on the A probe; unchanged: same
  checkpoint both sides) plus TWO threshold-only gates
  (`baseline_type="minimum_loss"`: one impossible ceiling -> failed,
  one generous ceiling -> passed) which judge NO comparison and keep
  verdict None, plus a fresh model with NO gate decisions.
  Assertions: three-group filtered-listing parity + verbatim
  detail-getter payloads + `(created_at, decision_id)` ordering; None
  NEVER matches — the threshold-only records are excluded from ALL
  THREE groups; explicit pairwise-disjoint partition over ALL THREE
  enum values whose union is EXACTLY the non-null-verdict decisions,
  with the null-verdict records still listed in the generic M6
  history; empty-listing model -> `[]` for all three; unknown model
  -> `FileNotFoundError`; cross-model isolation against the M23 second
  model (both hold decisions, listings disjoint, each group a subset
  of its own listing); read-only (gate manifest file-set snapshots for
  all three models unchanged); repeated-call determinism.
  (+`ComparisonVerdict` import added to the module's schemas import
  block.)
- `tests/test_gates_api.py` — 2 API tests: (1) grouping/partition/
  determinism — one gate per verdict plus two threshold-only
  null-verdict gates (the regressed record uses the established
  domain-B continued-training recipe from the lifecycle test); 200 +
  exact records for ALL THREE verdicts + listing parity +
  detail-getter parity + raw-byte-identical x3 per verdict + None
  exclusion + partition of the non-null decisions + no side effects;
  (2) 404/422/isolation/regressions/OpenAPI — 404 unknown model +
  valid verdict; 422 unsupported values (case variant, interior-space,
  numeric, non-enum word, AND the None-contract probes "none"/"null")
  and unknown-model + invalid-verdict -> 422 (pre-handler
  precedence); cross-model isolation (second model with no gates ->
  all three groups `[]`); M6 listing/getter + ghost decision id 404
  (no route capture); M23 by-policy + M34 by-comparison + M41
  by-decision regressions with listing parity; M42 workflows-by-status
  regression (natural valid-empty in this harness); OpenAPI 75
  assertions (route order by-decision < by-verdict < generic,
  GET-only, tag `gates`, items `$ref GateDecision`, `verdict` param
  `$ref ComparisonVerdict`).

One bring-up iteration (fixed before any full run): the domain-B
dataset in API test 1 was named `api6-m43b-ds`, colliding with test
2's `_make_model(api_client, "m43b", ...)` upload (HTTP 409
duplicate-name); renamed to `api6-m43reg-ds`.

Results (nothing certified from partial runs):

- full suite, twice consecutively: **525 passed @ 160.11 s**, then
  (after cleaning accumulated sandboxes, no live process)
  **525 passed @ 165.69 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**.

## OpenAPI

- OpenAPI 74 -> **75 paths** (exactly one new path).
- Standalone verification (no server): the new path registered exactly
  once, GET-only, tag `gates`; route order by-decision < by-verdict <
  generic `{decision_id}`; `verdict` parameter `$ref
  ComparisonVerdict`; response items `$ref GateDecision`; no duplicate
  or conflicting gate routes, no unintended methods.
- Repository sweep: every hard-coded count assertion searched and
  updated — **27 assertions `== 74` -> `== 75` across 8 test modules
  (one more than M42's 26: M42's own new API test added an
  assertion), 0 `== 74` remaining**, plus 19 ladder lines extended
  with `+ 1 (M43 gate decisions by-verdict)`; no existing OpenAPI
  assertion weakened.

## Live Smoke

`smoke_m43_live.py` (committed) — 35 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
35/35 PASS each, exit 0, first attempt**, on port **8764** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m43-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + full M4/M5/M6/
  M11/M15/M16/M18–M42 pre-state (including the M36–M42 groupings, the
  M41 by-decision counts and the M42 by-status counts) + dashboard
  hash + OpenAPI 75 pre-state + known pair 200.
- B (DISCOVERY, never assumed): from the live M6 listing + the OpenAPI
  schema — **11 gate decisions, verdict distribution
  `improved -> 4, regressed -> 3, unchanged -> 2` (all three groups
  non-empty) plus `null -> 2` threshold-only decisions**, ASC
  `(created_at, decision_id)`; enum values
  `['improved', 'regressed', 'unchanged']` — matching the M42-end
  grounding exactly.
- C: ALL THREE verdicts -> 200 + EXACT records == listing filtered
  locally (4/3/2); verbatim detail-getter parity for all 9 non-null
  records (decision/evidence chain/rollback suggestion included).
- D: three GETs raw-byte-identical.
- E: ALL enum verdicts -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid verdict -> 404; unsupported values (case
  variant, interior-space, numeric, non-enum word) -> 422 with the
  enum detail; the None contract — "none"/"null" -> 422 (no route
  represents the null verdict); unsupported verdict + UNKNOWN model ->
  422 (pre-handler precedence).
- G: the three groups partition the NON-NULL part of the listing
  exactly (4+3+2 == 9; total stays 11; every record exactly once; the
  2 null-verdict threshold-only decisions in NO group but listed).
- H: every group preserves the authoritative M6 order exactly.
- I: M23 by-policy (1) + M34 by-comparison (4) + M41 by-decision
  (passed 7 / failed 4) + the M6 listing unchanged, all with listing
  parity.
- J: M22 suite runs (10/10/summary), M24/M28/M30/M36/M38 evaluation
  histories (3/3/3, 16, 16, by-split 14/2/0, by-state-kind 9/7),
  M26/M29/M31/M37/M39 comparison histories (6/5/1, 8, 8, by-split
  8/0/0, by-verdict 3/3/2), M27/M32/M40 sample histories (4/0/0, 4,
  by-strategy 2/2), M33 sample-quality (2), M42/M35 workflow
  histories (by-status 9/3/1 with listing parity, 13 total, 2
  by-recipe with listing parity).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe
  registries unchanged.
- M: OpenAPI exactly 75 paths, new path once, order by-decision <
  by-verdict < generic, the M42 workflows-by-status and M41
  by-decision paths still present exactly once; storage zero drift
  (below).

## Storage Integrity

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: **0 changed / 0 missing / 0 new** files across all
  three runs;
- final audit after the server stopped: `sha256sum -c m43_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- stale test sandboxes (`/tmp/forge-tests-*`) cleaned after verifying
  no live pytest/uvicorn process, before the certifying second
  full-suite run and the smoke;
- M43 wrote NOTHING to production: the endpoint is pure data access
  over existing M6 gate manifests. Production remains byte-identical
  through M43.

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5/M6/M11/M15 listings + detail getters, M22
  suite-run histories, M23/M34/M41 gate-decision groupings, M24/M28/
  M30/M36/M38 evaluation groupings, M26/M29/M31/M37/M39 comparison
  groupings, M27/M32/M40 sample groupings, M33 sample-quality, M35
  workflow history by recipe (model-scoped + global M12 lineage), M42
  workflow history by status, M16 pairings, M17 dashboard (full
  hash), M2/M9/M12/M14 registries.
- Full suite green at 525 with no weakened tests; the natural-
  valid-empty discipline applied (workflows under the gate harness),
  never by fabricating artifacts.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged; the optional
  `verdict` field was NOT turned into a required schema field and no
  None route exists.
- README/landing updated only additively; OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit State

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M43: add gate-decision history by verdict` — `app/gates.py`,
   `app/engine.py`, `app/api.py`, `README.md`, the 10 touched test
   modules (2 with new tests, 8 with the OpenAPI 75 sweep + ladders),
   `smoke_m43_live.py`, `M43_final_report.md`.
2. follow-up inventory commit adding `m43_pre.sha256` at the repo root
   (M21–M43 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed; treated
consistently with every prior milestone).

## Next Milestone

End-of-M43 live inspection (production server, real payloads + schema
inspection): every `ComparisonRecord` persists BOTH comparison sides
(`state_a` / `state_b`, each with a `state_kind: EvalStateKind` —
current = the model's published weights, checkpoint = an immutable
stored checkpoint; current-state sides keep `checkpoint_id` None), and
the M26 by-checkpoint grouping already established the family's
either-side semantics ("a comparison involves the checkpoint when
EITHER persisted side records that id"). The state-kind axis of the
same sides is not yet exposed: production either-side distribution
**checkpoint -> 8, current -> 2 (both groups non-empty)** across the
8 comparisons of `4a0a871886ef` (baseline-side split: checkpoint 6 /
current 2); the M5 listing is deterministic in `(created_at,
comparison_id)` ASC order (verified live), and the comparison family
already groups by-checkpoint (M26), by-dataset (M29), by-tokenizer
(M31), by-split (M37), by-verdict (M39) — by-state-kind is the missing
enum-contract twin of M38 (which exposed the same `EvalStateKind`
axis for evaluations). `b5bc905326b6` owns NO comparisons (natural
valid-empty for both values). Runners-up inspected and set aside:
M16 sample-quality by `sample_id` (persisted top-level field with
clean M15 registry validation, but production holds only ONE
non-empty group — both records share sample `f8e66f9c7b50` — weaker
on the two-meaningful-groups criterion), `SuiteRunRecord.state`
(all 10 production runs evaluate a checkpoint — single-valued),
`WorkflowRecord.failed_stage_id` (free strings, no enum/registry),
global suite-runs lineage (the registry holds ONE suite with runs on
one model — single group).

Copy-ready M44 prompt:

> M44 — COMPARISON HISTORY BY STATE KIND
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped M5 comparison records by the persisted state kind of
> their comparison sides (either-side semantics, exactly like M26
> by-checkpoint). Do NOT jump ahead into aggregation, rankings,
> auto-gating, rollback, retraining, optimization, HPO, RL, Gemini
> integration, inference, deployment, or autonomous improvement.
> Preserve the minimum-files/minimum-storage architecture and the
> established authoritative-listing -> persisted-field-filter ->
> response pattern.
>
> STEP 0 — VERIFY STATE AND BASELINE. Inspect the repository and git
> state. If the familiar reset signature appears (HEAD `f86b670` /
> "Add files via upload", files untracked, production data missing,
> venv missing), execute the established recovery runbook: `git
> fetch` + `git reset --hard FETCH_HEAD` (never force-push, never
> discard authoritative commits), restore `ai-model-forge-data/` from
> the workspace zip, verify against `m43_pre.sha256`, rebuild the
> venv (`python3 -m venv ~/.venv` + `pip install -e
> ai-model-forge[dev] pyflakes`), then re-prove the baseline.
> Expected baseline: M43 certified; HEAD/FETCH_HEAD inventory commit
> = the M43 inventory hash; 525 tests passing; OpenAPI 75; production
> 96 files / 4,002,745 bytes / 0 tmp; zero SHA drift; production gate
> verdict distribution of `4a0a871886ef`: improved -> 4 / regressed
> -> 3 / unchanged -> 2 + 2 null. Capture `m44_pre.sha256` and record
> the exact baseline before changing anything. Inspect
> `app/comparison.py` (the M5 listing + M26
> `list_comparisons_for_checkpoint()` — note its EITHER-side
> semantics), `app/engine.py`, `app/api.py` (the
> `/models/{id}/comparisons/...` family), `app/schemas.py`
> (`ComparisonRecord.state_a/state_b: ComparisonSide` with
> `state_kind: EvalStateKind` — current/checkpoint), and the M38
> evaluations-by-state-kind implementation (the enum-contract twin);
> run the complete suite before editing.
>
> GROUNDED SELECTION (live-verified at M43 end): the persisted
> `state_kind` of each comparison side (`ComparisonSide.state_kind:
> EvalStateKind`) with EITHER-side membership exactly like M26 — a
> comparison belongs to the request when EITHER persisted side
> records the requested state kind (nothing inferred from hashes,
> checkpoint ids, losses or filenames); production either-side
> distribution checkpoint -> 8 / current -> 2 (both non-empty;
> baseline-side split checkpoint 6 / current 2); M5 listing order
> `(created_at, comparison_id)` ASC; no by-state-kind comparison
> surface exists in M1–M43 (M38 exposed the same axis for
> evaluations).
>
> IMPLEMENTATION: In `app/comparison.py` add
> `list_comparisons_for_state_kind(model_id, state_kind)` following
> the M26/M43 thin pattern: validate the model via the authoritative
> M5 listing mechanism (`list_comparisons(model_id)`); filter ONLY
> the persisted `record.state_a.state_kind == state_kind or
> record.state_b.state_kind == state_kind` (either side, verbatim —
> never inferred from checkpoint ids, state hashes or evaluation
> results; never resolved or rewritten); preserve the M5
> `(created_at, comparison_id)` ASC ordering; `[]` for a valid model
> with no matching records; unknown model -> existing
> FileNotFoundError/404; invalid state kind -> native FastAPI/
> Pydantic 422 via an `EvalStateKind` path parameter (the enum IS
> the contract — no registry, matching M36–M43); no cross-model
> leakage; no writes; no cache/index/DB/secondary registry. In
> `app/engine.py` expose it through the facade (sibling-facade
> style, no duplicated filtering). In `app/api.py` add exactly:
> `GET /api/v1/models/{model_id}/comparisons/by-state-kind/
> {state_kind}` with `response_model=list[ComparisonRecord]`, tags
> matching the comparison family, registered after the M39
> by-verdict route and BEFORE the generic
> `/comparisons/{comparison_id}` detail route (so "by-state-kind"
> can never be interpreted as a comparison id). Documentation:
> landing bullet + endpoint `<li>` + section comment (verify M42/M43
> anchors before editing); README M44 section + endpoint count
> 525 -> ~530 + layout line update.
>
> TESTS: ~3 engine tests + 2 API tests following the M43 conventions
> (comparison-family conventions in `tests/test_comparison_api.py` /
> the comparison module): both enum groups with either-side listing
> parity (derive membership from the authoritative M5 listing, never
> hard-coded ids; include a checkpoint-vs-checkpoint record, a
> current-vs-checkpoint record and a checkpoint-vs-current record to
> pin the either-side semantics); deterministic ordering; persisted
> state kind verbatim; partition is NOT disjoint by design (a record
> with both sides checkpoint belongs to the checkpoint group once) —
> instead assert each group == listing filtered by either-side
> membership and every record belongs to at least one group;
> valid-empty (a model with no comparisons -> 200 []); unknown model
> -> 404; invalid state kind -> 422 (case variant, interior-space,
> numeric) incl. unknown-model + invalid -> 422 pre-handler;
> cross-model isolation; byte-identical repeats; no writes; M26
> by-checkpoint + M37 by-split + M39 by-verdict regressions
> (parity or natural-valid-empty per the harness — never manufacture
> artifacts). Target 525 -> ~530 (report the actual).
>
> OPENAPI: 75 -> 76 paths. Update every genuine path-count assertion
> (search the whole repo; do not blindly replace unrelated numbers).
> Verify: exactly one new path, GET only, comparison-family tag,
> items $ref ComparisonRecord, state_kind parameter $ref
> EvalStateKind, family ordering by-verdict < by-state-kind <
> generic detail, no duplicate routes, no unintended methods. Do not
> weaken existing assertions.
>
> FULL VERIFICATION: focused comparison tests; API tests; complete
> suite; complete suite a second time; compileall; pyflakes; OpenAPI
> verification; production storage audit; SHA verification against
> `m44_pre.sha256`. Clean stale `/tmp/forge-tests-*` after verifying
> no live process. Expected: tests 525 -> ~530, OpenAPI 76, storage
> unchanged 96/4,002,745/0, zero drift.
>
> LIVE SMOKE: create `smoke_m44_live.py`, port **8765**, following
> the established A–M structure: baseline inventory capture; live
> DISCOVERY of the either-side state-kind distribution (expected
> grounding: checkpoint -> 8 / current -> 2 among 8 comparisons;
> report the actual if legitimately different — never invent);
> both-group either-side parity; ordering; byte-identical repeats;
> unknown model -> 404; invalid state kind -> 422; second model ->
> 200 []; cross-model isolation; M26 by-checkpoint + M37 by-split +
> M39 by-verdict unchanged; M38 evaluations-by-state-kind unchanged
> (the enum twin); M43 gate by-verdict + M41 by-decision + M42
> by-status unchanged; M18–M42 regressions unchanged; dashboard
> hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
> unchanged; policy/probe/recipe registries unchanged; storage
> remains 96 files / 4,002,745 bytes / 0 tmp; SHA256 0 changed / 0
> new / 0 missing; no new production artifacts.
>
> FINAL AUDIT + REPORT: inspect git diff/status; verify only
> intended files changed; verify production unchanged; no
> force-push; local HEAD == authoritative remote HEAD; only egg-info
> untracked. Commit the implementation (`M44: add comparison
> history by state kind`) and the inventory (`m44_pre.sha256`)
> separately per the M21–M43 convention; push normally. Final report
> with EXACTLY these 9 sections: 1. Recovery & Baseline, 2.
> Implementation, 3. Tests, 4. OpenAPI, 5. Live Smoke, 6. Storage
> Integrity, 7. Regression / Compatibility, 8. Git / Commit State,
> 9. Next Milestone. Section 9 must automatically select M45 from
> LIVE persisted production relationships discovered at the end of
> M44 (inspect every remaining persisted enum/identity axis across
> evaluations, comparisons, samples, sample-quality, suite-runs,
> gates, workflows, recipes — e.g. the M43 runner-up M16
> sample-quality by sample_id held only ONE non-empty production
> group; re-verify; prefer an existing enum/registry with >= 2
> non-empty live groups, thin filter, no schema redesign, no new
> storage; mention the runner-up) and include a COMPLETE copy-ready
> M45 prompt with the same structure as this one (Step 0
> baseline/recovery, grounded selection, exact endpoint/engine
> method/field/enum, route ordering, tests, OpenAPI count, smoke
> port 8766, storage invariants, the exact 9-section report
> requirement, automatic M46 selection). Do not jump to advanced
> training, autonomous improvement, HPO, RL, Gemini integration,
> inference, deployment, or other future architecture unless the
> live evidence specifically makes that the next justified
> milestone.
