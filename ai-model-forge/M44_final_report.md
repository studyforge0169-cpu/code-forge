# M44 Final Report — Comparison History by State Kind

## Recovery & Baseline

**Reset #10 occurred** (the familiar signature: HEAD `f86b670` "Add
files via upload" with the whole tree untracked, the production data
directory GONE, and the venv GONE). The established recovery runbook
was executed before any coding:

1. `git fetch` -> FETCH_HEAD `9d06deb` (the authoritative M43
   inventory tip), `git reset --hard FETCH_HEAD` -> HEAD == `9d06deb`,
   tree clean except the known egg-info (no force-push; nothing
   reconstructed from assumptions);
2. production restored from `code-forge/code forge.zip` by extracting
   ONLY its `ai-model-forge-data/` prefix (96 members);
3. audit: **96 files / 4,002,745 bytes / 0 .tmp**;
   `sha256sum -c m43_pre.sha256` -> **0 non-OK** (byte-identical
   through M43);
4. venv rebuilt from scratch (`python3 -m venv ~/.venv` + `pip
   install -e ai-model-forge[dev] pyflakes`, 104 s), import checks
   OK.

Baseline then re-proved BEFORE implementation:

- full suite: **525 passed @ 167.44 s** (exact M43 ladder);
- OpenAPI: **75 paths**;
- `m44_pre.sha256` captured (96 entries);
- no stale `/tmp/forge-tests-*`, no live server/pytest processes.

## Implementation

Three production files + docs, mirroring the M26/M38/M43 pattern with
M26's either-side semantics as the semantic authority:

- `app/comparison.py` — new engine method
  `list_comparisons_for_state_kind(model_id, state_kind:
  EvalStateKind)` placed directly after
  `list_comparisons_for_checkpoint` (the M26 method it mirrors). It
  reuses the model's authoritative M5 listing (exact engine parse +
  deterministic `(created_at, comparison_id)` ASC order) and filters
  by the persisted side state kinds with the SAME nested `involves()`
  shape as M26: a comparison belongs when EITHER persisted side
  (`state_a` / `state_b`) records the requested `state_kind`, matched
  VERBATIM — never inferred from checkpoint ids alone, state hashes,
  losses or evaluation results; because the listing holds each record
  exactly once, a both-sides match appears EXACTLY ONCE. Docstring
  records the full contract: state kinds have NO registry (the enum IS
  the contract, exactly like M38) so unsupported values are rejected
  with 422 at the API boundary and never reach the engine, while an
  unknown model raises FileNotFoundError exactly like the siblings; a
  valid state kind with no matching comparisons returns `[]`;
  read-only, never writes. `EvalStateKind` was already imported here
  (used by the M26 filter).
- `app/engine.py` — facade `list_comparisons_for_state_kind` delegating
  to the engine, placed after the by-verdict delegation
  (`EvalStateKind` already imported; facade signature matches the
  unannotated sibling style).
- `app/api.py` — route `@api.get("/models/{model_id}/comparisons/
  by-state-kind/{state_kind}", response_model=list[ComparisonRecord],
  tags=["comparison"])` with `state_kind: EvalStateKind` as the path
  parameter (native pre-handler 422 for unsupported values), placed
  after the M39 by-verdict route and before the generic
  `/comparisons/{comparison_id}` getter; landing-page bullet (worded
  "involving the requested state kind on either side", per the
  milestone's wording requirement); endpoint `<li>`; comparison-routes
  section comment extended with the M44 line (`EvalStateKind` already
  imported for M38; no new import needed anywhere).
- `README.md` — M44 section, counts 525 -> 530 (two places),
  comparison.py layout line extended to
  `M26/M29/M31/M37/M39/M44`.

No other production code changed; nothing persisted, cached, indexed
or migrated anywhere.

## Tests

Five new tests (525 -> 530, exactly as targeted), REUSING the existing
M26 fixtures rather than creating a second comparison model:

- `tests/test_comparison.py` — 3 engine tests on top of the cached
  `_m26_env(env)` fixture (cross-checkpoint A/B, same-checkpoint A=B,
  current-vs-checkpoint, current-vs-current, plus the second model
  with a trained checkpoint but ZERO comparisons):
  either-side parity for BOTH enum values against the authoritative
  M5 listing independently filtered + `(created_at, comparison_id)`
  ordering + verbatim detail-getter payloads + both-sides-match-
  exactly-once; explicit either-side pins (the current-vs-checkpoint
  record in BOTH groups; checkpoint-only / current-only records in
  exactly one; the two groups COVER the full listing — overlap by
  design, never duplication); natural valid-empty (the zero-
  comparison model -> `[]` for both kinds); unknown model ->
  `FileNotFoundError`; cross-model isolation; read-only (comparison
  manifest file-set snapshots for both models unchanged);
  repeated-call determinism.
- `tests/test_comparison_api.py` — 2 API tests: (1) grouping/
  either-side/determinism — the four M26 comparison shapes over HTTP;
  200 + exact records for BOTH kinds + either-side listing parity +
  detail-getter parity + raw-byte-identical x3 per kind + coverage
  (union == listing) + either-side pins + no side effects; (2)
  404/422/isolation/regressions/OpenAPI — 404 unknown model + valid
  state kind; 422 unsupported values (case variant, interior-space,
  numeric, non-enum word) and unknown-model + invalid-kind -> 422
  (pre-handler precedence); cross-model isolation (second model with
  no comparisons -> both groups `[]`); M5 listing/getter + ghost id
  404 (no route capture); M26/M29/M31/M37/M39 comparison-family
  regressions with listing parity; M38 evaluations-by-state-kind
  regression (the enum-contract twin, listing parity); OpenAPI 76
  assertions (route order by-verdict < by-state-kind < generic,
  GET-only, tag `comparison`, items `$ref ComparisonRecord`,
  `state_kind` param `$ref EvalStateKind`).

No bring-up iterations — all five tests passed on the first focused
run.

Results (nothing certified from partial runs):

- full suite, twice consecutively: **530 passed @ 171.54 s**, then
  (after cleaning accumulated sandboxes, no live process)
  **530 passed @ 174.95 s**;
- `python -m compileall app tests` -> OK;
- `python -m pyflakes app tests` -> **0 findings**.

## OpenAPI

- OpenAPI 75 -> **76 paths** (exactly one new path).
- Standalone verification (no server): the new path registered exactly
  once, GET-only, tag `comparison`; route order by-verdict <
  by-state-kind < generic `{comparison_id}`; `state_kind` parameter
  `$ref EvalStateKind`; response items `$ref ComparisonRecord`; no
  duplicate or conflicting comparison routes, no unintended methods.
- Repository sweep: every hard-coded count assertion searched and
  updated — **28 assertions `== 75` -> `== 76` across 8 test modules
  (one more than M43's 27: M43's own new API test added an
  assertion), 0 `== 75` remaining**, plus 20 ladder lines extended
  with `+ 1 (M44 comparisons by-state-kind)`; no existing OpenAPI
  assertion weakened.

## Live Smoke

`smoke_m44_live.py` (committed) — 34 named checks in sections LIVE A–M
against the real production FORGE_ROOT over HTTP, run **three times,
34/34 PASS each, exit 0, first attempt**, on port **8765** (server
`FORGE_ROOT=/home/user/ai-model-forge-data uvicorn app.api:app`,
stopped afterwards). Baseline per-file inventory saved to
`/tmp/m44-smoke-baseline-inventory.json`.

- A: exact 96/4,002,745/0 audit + inventory saved + full M4/M5/M6/
  M11/M15/M16/M18–M43 pre-state (including the M36–M43 groupings, the
  M43 by-verdict gate counts and the M42 by-status counts) + dashboard
  hash + OpenAPI 76 pre-state + known pair 200.
- B (DISCOVERY, never assumed): from the live M5 listing + the OpenAPI
  schema — **8 comparisons, either-side state-kind distribution
  `checkpoint -> 8, current -> 2` (both groups non-empty; the two
  current-side records also involve a checkpoint side, so the groups
  overlap by design)**, ASC `(created_at, comparison_id)`; enum values
  `['current', 'checkpoint']` — matching the M43-end grounding
  exactly.
- C: BOTH state kinds -> 200 + EXACT records == the either-side
  membership derived INDEPENDENTLY from the listing (M26 parity — both
  sides inspected locally); verbatim detail-getter parity for all
  group records (both sides, losses, verdict verbatim).
- D: three GETs raw-byte-identical.
- E: ALL enum state kinds -> 200 + [] under `b5bc905326b6`.
- F: unknown model + valid state kind -> 404; unsupported values (case
  variant, interior-space, numeric, non-enum word) -> 422 with the
  enum detail; unsupported state kind + UNKNOWN model -> 422
  (pre-handler precedence).
- G: the two groups COVER the full listing exactly (every record in at
  least one group; each group == the derived either-side membership;
  every record EXACTLY ONCE per group; overlap by either-side design,
  never duplication).
- H: every group preserves the authoritative M5 order exactly.
- I: M26 by-checkpoint (6/5/1) + M29 by-dataset (8) + M31 by-tokenizer
  (8) + M37 by-split (8/0/0) + M39 by-verdict (3/3/2) + the M5 listing
  unchanged, all with listing parity.
- J: M22 suite runs (10/10/summary), M24/M28/M30/M36/M38 evaluation
  histories (3/3/3, 16, 16, by-split 14/2/0, by-state-kind 9/7 — the
  M38 twin), M27/M32/M40 sample histories (4/0/0, 4, by-strategy
  2/2), M33 sample-quality (2), M41/M43/M34/M23 gate-decision
  histories (by-decision passed 7, by-verdict improved 4 / regressed 3
  / unchanged 2 with the 2 null-verdict records in NO group, 4
  by-comparison, 1 by-policy), M42/M35 workflow histories (by-status
  9/3/1 with listing parity, 13 total, 2 by-recipe with listing
  parity).
- K: M12/M35 workflow surfaces (registry 7 verbatim, global runs 2,
  by-recipe 2, zero-run recipes 200 + [], unknown recipe 404) + M16
  pairings.
- L: dashboard `result_hash
  f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838` +
  full output unchanged; tokenizer/policy/probe-suite/recipe
  registries unchanged.
- M: OpenAPI exactly 76 paths, new path once, order by-verdict <
  by-state-kind < generic, the M43 gate-decisions-by-verdict, M42
  workflows-by-status and M41 by-decision paths still present exactly
  once; storage zero drift (below).

## Storage Integrity

- Before AND after every smoke run: **96 files / 4,002,745 bytes /
  0 .tmp** — totals unchanged;
- smoke M3/M4/M5: **0 changed / 0 missing / 0 new** files across all
  three runs;
- final audit after the server stopped: `sha256sum -c m44_pre.sha256`
  from the data root -> **0 non-OK** (every file byte-identical to the
  pre-milestone inventory);
- stale test sandboxes (`/tmp/forge-tests-*`) cleaned after verifying
  no live pytest/uvicorn process, before the certifying second
  full-suite run and the smoke;
- M44 wrote NOTHING to production: the endpoint is pure data access
  over existing M5 comparison manifests. Production remains
  byte-identical through M44 (including across the reset-#10
  restoration).

## Regression / Compatibility

- All pre-existing surfaces re-verified live and unchanged (smoke
  sections I–L): M4/M5/M6/M11/M15 listings + detail getters, M22
  suite-run histories, M23/M34/M41/M43 gate-decision groupings, M24/
  M28/M30/M36/M38 evaluation groupings, M26/M29/M31/M37/M39 comparison
  groupings, M27/M32/M40 sample groupings, M33 sample-quality, M35
  workflow history by recipe (model-scoped + global M12 lineage), M42
  workflow history by status, M16 pairings, M17 dashboard (full
  hash), M2/M9/M12/M14 registries.
- Full suite green at 530 with no weakened tests; the M26 fixtures
  were reused rather than duplicating a comparison model.
- The 422 semantic is native FastAPI path-parameter validation — no
  middleware, no manual conversion, no behavior change for any other
  route; sibling endpoints byte-verified unchanged.
- README/landing updated only additively (worded "involving the
  requested state kind on either side"); OpenAPI additions verified
  against every hard-coded count in the repository.

## Git / Commit State

On branch `arena/01a071e9-code-forge` (session branch; never force
pushed):

1. `M44: add comparison history by state kind` —
   `app/comparison.py`, `app/engine.py`, `app/api.py`, `README.md`,
   the 11 touched test modules (2 with new tests, 9 with the OpenAPI
   76 sweep + ladders), `smoke_m44_live.py`, `M44_final_report.md`.
2. follow-up inventory commit adding `m44_pre.sha256` at the repo root
   (M21–M44 convention).

Exact commit hashes and the HEAD == FETCH_HEAD verification are
reported in the session reply. The only untracked path left is
`ai-model-forge/ai_model_forge.egg-info/` (never committed; treated
consistently with every prior milestone).

## Next Milestone

End-of-M44 live inspection (production server, real payloads + schema
inspection): every `GateDecision` embeds its `GatePolicy` VERBATIM
("policy is embedded as evaluated"), and every policy carries a
required `baseline_type: GateBaselineType` — a FOUR-value schema enum
(checkpoint = a specific immutable checkpoint; current = the model's
published current weights; evaluation_result_hash = a past immutable
evaluation; minimum_loss = absolute loss threshold only, no state) —
persisted with each decision. Production distribution across the 11
gate decisions of `4a0a871886ef`: **checkpoint -> 7, current -> 2,
minimum_loss -> 2 (THREE non-empty groups; evaluation_result_hash ->
0, a natural valid-empty)**. Because `baseline_type` is required,
every decision falls in exactly ONE group — a TRUE disjoint partition
(no None case, unlike M43). The M6 listing is deterministic in
`(created_at, decision_id)` ASC order (verified live through M43), and
the gate-decision family already groups by-policy (M23), by-comparison
(M34), by-decision (M41), by-verdict (M43) — by-baseline-type is the
remaining enum axis of the embedded policy. `b5bc905326b6` owns NO
gate decisions (natural valid-empty for all four values). Runners-up
inspected and set aside: M16 sample-quality by `sample_id`
(registry-validated persisted identity, but production holds only ONE
non-empty group — both records share sample `f8e66f9c7b50` — weaker on
the two-meaningful-groups criterion), `SuiteRunRecord.state` (all 10
production runs evaluate a checkpoint — single-valued), dataset/model
registries (1 dataset / 2 models — no meaningful groups).

Copy-ready M45 prompt:

> M45 — GATE-DECISION HISTORY BY BASELINE TYPE
>
> Build exactly the next smallest read-only history surface: filter
> model-scoped gate-decision records by the persisted `baseline_type`
> of the policy embedded verbatim in each decision. Do NOT jump ahead
> into aggregation, rankings, auto-gating, rollback, retraining,
> optimization, HPO, RL, Gemini integration, inference, deployment, or
> autonomous improvement. Preserve the minimum-files/minimum-storage
> architecture and the established authoritative-listing ->
> persisted-field-filter -> response pattern.
>
> STEP 0 — VERIFY STATE AND BASELINE. Inspect the repository and git
> state. If the familiar reset signature appears (HEAD `f86b670` /
> "Add files via upload", files untracked, production data missing,
> venv missing), execute the established recovery runbook: `git
> fetch` + `git reset --hard FETCH_HEAD` (never force-push, never
> discard authoritative commits), restore `ai-model-forge-data/` from
> the workspace zip by extracting ONLY its `ai-model-forge-data/`
> prefix, verify against `m44_pre.sha256`, rebuild the venv
> (`python3 -m venv ~/.venv` + `pip install -e ai-model-forge[dev]
> pyflakes`), then re-prove the baseline. Expected baseline: M44
> certified; HEAD/FETCH_HEAD inventory commit = the M44 inventory
> hash; 530 tests passing; OpenAPI 76; production 96 files /
> 4,002,745 bytes / 0 tmp; zero SHA drift; production comparison
> either-side state-kind distribution of `4a0a871886ef`: checkpoint
> -> 8 / current -> 2. Capture `m45_pre.sha256` and record the exact
> baseline before changing anything. Inspect `app/gates.py` (the M41
> `list_decisions_for_decision()` and M43
> `list_decisions_for_verdict()` implementations), `app/engine.py`,
> `app/api.py` (the `/models/{id}/gates/decisions/...` family),
> `app/schemas.py` (`GatePolicy.baseline_type: GateBaselineType` —
> required, embedded verbatim in each `GateDecision` as
> `decision.policy`; enum values checkpoint / current /
> evaluation_result_hash / minimum_loss), and the M41/M43 test blocks
> in `tests/test_gates.py` / `tests/test_gates_api.py`; run the
> complete suite before editing.
>
> GROUNDED SELECTION (live-verified at M44 end):
> `GateDecision.policy.baseline_type` — the persisted required enum
> recorded verbatim inside the embedded policy of each decision (the
> M6 flow persists the policy exactly as evaluated); production
> distribution checkpoint -> 7 / current -> 2 / minimum_loss -> 2
> (three non-empty groups) + evaluation_result_hash -> 0 (natural
> valid-empty); a TRUE disjoint partition (baseline_type is required —
> no None case); M6 listing order `(created_at, decision_id)` ASC; no
> by-baseline-type surface exists in M1–M44. NOTE the field is
> NESTED (`decision.policy.baseline_type`), the first such grouping:
> filter the persisted embedded value VERBATIM — never re-derived
> from the baseline checkpoint id, hash or loss, never resolved or
> rewritten, and never re-evaluated.
>
> IMPLEMENTATION: In `app/gates.py` add
> `list_decisions_for_baseline_type(model_id, baseline_type)`
> following the M43 thin pattern: validate the model via the
> authoritative M6 listing mechanism (`list_decisions(model_id)`);
> filter ONLY the persisted `record.policy.baseline_type ==
> baseline_type` (verbatim; each decision appears EXACTLY ONCE; the
> disjoint groups partition the full listing); preserve the M6
> `(created_at, decision_id)` ASC ordering; `[]` for a valid model
> with no matching records; unknown model -> existing
> FileNotFoundError/404; invalid baseline type -> native FastAPI/
> Pydantic 422 via a `GateBaselineType` path parameter (the enum IS
> the contract — no registry, matching M36–M44); no cross-model
> leakage; no writes; no cache/index/DB/secondary registry. In
> `app/engine.py` expose it through the facade (sibling-facade
> style, no duplicated filtering; +`GateBaselineType` import where
> needed). In `app/api.py` add exactly:
> `GET /api/v1/models/{model_id}/gates/decisions/by-baseline-type/
> {baseline_type}` with `response_model=list[GateDecision]`, tags
> `["gates"]`, registered after the M43 by-verdict route and BEFORE
> the generic `/gates/decisions/{decision_id}` route (so
> "by-baseline-type" can never be interpreted as a decision id).
> Documentation: landing bullet + endpoint `<li>` + section comment
> (verify M43/M44 anchors before editing); README M45 section +
> endpoint count 530 -> ~535 + layout line update.
>
> TESTS: ~3 engine tests + 2 API tests following the M43 conventions
> in `tests/test_gates.py` / `tests/test_gates_api.py`: at least
> THREE baseline-type groups with listing parity (checkpoint-baseline
> gates, current-baseline gates, minimum_loss threshold-only gates;
> derive membership from the authoritative M6 listing, never
> hard-coded ids); the evaluation_result_hash group as the natural
> valid-empty (or create one evaluation-hash-baseline gate if the
> existing fixture helpers support it cheaply); deterministic
> ordering; persisted baseline type verbatim; TRUE disjoint partition
> over ALL FOUR enum values (union == full listing, no None case);
> valid-empty (a model with no gate decisions -> 200 []); unknown
> model -> 404; invalid baseline type -> 422 (case variant,
> interior-space, numeric) incl. unknown-model + invalid -> 422
> pre-handler; cross-model isolation; byte-identical repeats; no
> writes; M23 by-policy + M34 by-comparison + M41 by-decision + M43
> by-verdict regressions (parity or natural-valid-empty per the
> harness — never manufacture artifacts). Target 530 -> ~535 (report
> the actual).
>
> OPENAPI: 76 -> 77 paths. Update every genuine path-count assertion
> (search the whole repo; do not blindly replace unrelated numbers).
> Verify: exactly one new path, GET only, tag gates, items $ref
> GateDecision, baseline_type parameter $ref GateBaselineType,
> family ordering by-verdict < by-baseline-type < generic detail, no
> duplicate routes, no unintended methods. Do not weaken existing
> assertions.
>
> FULL VERIFICATION: focused gate tests; API tests; complete suite;
> complete suite a second time; compileall; pyflakes; OpenAPI
> verification; production storage audit; SHA verification against
> `m45_pre.sha256`. Clean stale `/tmp/forge-tests-*` after verifying
> no live process. Expected: tests 530 -> ~535, OpenAPI 77, storage
> unchanged 96/4,002,745/0, zero drift.
>
> LIVE SMOKE: create `smoke_m45_live.py`, port **8766**, following
> the established A–M structure: baseline inventory capture; live
> DISCOVERY of the baseline-type distribution from the M6 listing
> (expected grounding: checkpoint -> 7 / current -> 2 /
> minimum_loss -> 2 / evaluation_result_hash -> 0 among 11 gate
> decisions; report the actual if legitimately different — never
> invent); three-group parity; TRUE disjoint partition 7+2+2+0 == 11;
> ordering; byte-identical repeats; unknown model -> 404; invalid
> baseline type -> 422; second model -> 200 []; cross-model
> isolation; M23/M34/M41/M43 gate surfaces unchanged; M44
> comparisons-by-state-kind unchanged (either-side parity); M18–M43
> regressions unchanged; dashboard hash
> `f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838`
> unchanged; policy/probe/recipe registries unchanged; storage
> remains 96 files / 4,002,745 bytes / 0 tmp; SHA256 0 changed / 0
> new / 0 missing; no new production artifacts.
>
> FINAL AUDIT + REPORT: inspect git diff/status; verify only
> intended files changed; verify production unchanged; no
> force-push; local HEAD == authoritative remote HEAD; only egg-info
> untracked. Commit the implementation (`M45: add gate-decision
> history by baseline type`) and the inventory (`m45_pre.sha256`)
> separately per the M21–M44 convention; push normally. Final report
> with EXACTLY these 9 sections: 1. Recovery & Baseline, 2.
> Implementation, 3. Tests, 4. OpenAPI, 5. Live Smoke, 6. Storage
> Integrity, 7. Regression / Compatibility, 8. Git / Commit State,
> 9. Next Milestone. Section 9 must automatically select M46 from
> LIVE persisted production relationships discovered at the end of
> M45 (inspect every remaining persisted enum/identity axis across
> evaluations, comparisons, samples, sample-quality, suite-runs,
> gates, workflows, recipes — e.g. the M44 runner-up M16
> sample-quality by sample_id held only ONE non-empty production
> group; re-verify; prefer an existing enum/registry with >= 2
> non-empty live groups, thin filter, no schema redesign, no new
> storage; mention the runner-up) and include a COMPLETE copy-ready
> M46 prompt with the same structure as this one (Step 0
> baseline/recovery, grounded selection, exact endpoint/engine
> method/field/enum, semantics, route ordering, tests, OpenAPI
> count, smoke port 8767, storage invariants, the exact 9-section
> report requirement, automatic M47 selection). Do not jump to
> advanced training, autonomous improvement, HPO, RL, Gemini
> integration, inference, deployment, or other future architecture
> unless the live evidence specifically makes that the next
> justified milestone.
