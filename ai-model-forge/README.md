# AI Model Forge

A platform for **creating, training, evaluating, improving, versioning and deploying an AI model that is owned by the user**. Built incrementally as a modular monolith — milestone by milestone, each verified before the next begins.

**Status: Milestone 15 complete** — deterministic checkpoint sampling
(generation). The platform's first model-USAGE primitive: pure CPU-first
inference that generates text from ONE explicitly selected, verified, immutable
checkpoint — every input explicit (model_id + checkpoint_id + tokenizer_id +
prompt + strategy + parameters), nothing auto-selected. Exactly two decoding
strategies: `greedy` (deterministic argmax, no RNG) and seeded `temperature`
sampling (one deterministic RNG stream from the request's integer seed).
Checkpoints pass the existing content-hash integrity verification, the
tokenizer/model vocabulary compatibility follows the exact M3/M4 platform rule,
and logits are restricted to the tokenizer's vocab so every generated token
decodes. Each successful request writes exactly ONE immutable manifest under the
new root-level `samples/` family with a deterministic `result_hash`; failures
write nothing. M14 composable recipes and everything earlier remain untouched —
generation never trains, evaluates, scores, ranks or judges output.

---

## What works today

### Milestone 1 — model foundation
- **Configurable decoder-only Transformer** (`app/model_builder.py`), fully config-driven:
  embeddings · RoPE / learned / no positions · RMSNorm · causal MHA **and GQA** · fused or separate QKV · SwiGLU/GELU/ReLU MLP · residual + final norm · LM head (optional tying) · soft-cap · dropout · fp32/fp16/bf16 · flash-attention config (requires CUDA at build) · **KV-cache generation**
- **Model config schema with pairwise validation** (`app/schemas.py`) — invalid combos never reach disk
- **Model persistence**: one manifest + one weights file + sha256 sidecar per model; atomic writes; content hashes; integrity verification
- **REST API**: health / project / system + full model registry (create, validate, list, get, verify, delete, weights download)
- **Hardware detection** (CPU/CUDA/VRAM/RAM/disk) ready for training auto-configuration

### Milestone 2 — data engine foundation
- **Tokenizer engine** (`app/tokenizer.py`): deterministic **byte-level BPE** (HF `tokenizers`),
  `tokenizer.json` + `manifest.json` per tokenizer; trains from uploaded files **or** a stored dataset;
  exact `decode(encode(x)) == x` for real text (verified multilingual/emoji/whitespace); vocab ceiling honoured
- **Dataset engine** (`app/dataset.py`):
  - ingestion: `.txt` `.md` (paragraphs) · `.csv` (rows; header-aware, ragged rows reported) · `.json` (arrays, `text` field, nested, depth-capped)
  - per-file reports — mixed valid/invalid uploads never fail opaquely
  - normalization → empty-record removal → **sha256 dedup** → quality stats → **deterministic 90/5/5 splits** (hash buckets, reproducible, no overlap)
  - **append-only versioning**: `datasets/<id>/v<N>/` self-contained immutable snapshots (`records.jsonl.gz` deterministic gzip); v1 byte-identical after v2
  - dedup index (`records.sha256`), dataset pointer (`dataset.json`), source metadata + hashes — source files never copied
  - **tokenization**: `v<N>/tokenized/<tokenizer_id>/{train,validation,test}.bin` in **uint16** (uint32 if vocab > 65536) with per-split token counts + sha256
  - **verification**: manifest validity, record count, per-record rehash, split recomputation, tokenized size/hash — tampering ⇒ `status: "failed"`, never silently repaired
  - deletion safety: datasets train-referenced by a tokenizer refuse deletion (409)
- **REST API**: dataset upload/list/get/verify/tokenize/delete + tokenizer train/list/get/delete

### Milestone 3 — training engine foundation
- **`app/training.py`** — first real training layer, synchronous and deterministic:
  - **methods**: `continued_pretraining` and `sft` — both are *standard causal-LM sequence
    training* over M2 tokenized `.bin` streams (shift-by-one cross-entropy, fp32).
    **Honest scope**: no chat templates, role masks, preference labels or instruction loss;
    `sft` differs only in name/records for now — documented.
  - **data policy**: token stream is read in file order via memmap and cut into
    `max_seq_len` sequences; the incomplete tail is **dropped** (deterministic; chunks may
    straddle documents; no cross-document packing yet); batch order is fixed (no shuffle)
  - **optimizer**: AdamW (configurable betas, weight decay); **schedules**: constant /
    linear / cosine with linear warmup (recorded per checkpoint); optional grad clipping;
    **gradient accumulation** (effective update batch = batch × accumulation)
  - **hardware-aware batches**: memory estimate (16 B/param optimizer state + conservative
    activations + 32 MiB overhead) vs 60 % of RAM/VRAM; batch auto-halving is *recorded* in
    the report (`requested` vs `actual`), never silent
  - **determinism**: seeds from config (default = sha256 of config), same config + seed ⇒
    matching loss history (≈1e-5) and matching checkpoint hashes
  - **evaluation gate**: baseline validation loss/perplexity before any update; eval after
    every `eval_every_steps` and at the final step; each checkpoint is `accept` (beat the
    running best) or `not_best` — history is never destroyed
  - **keep-best**: `keep_best=true` leaves the model weights at the best checkpoint of the
    run (`rolled_back_to` in the report); `false` publishes the final state
- **Immutable, content-addressed checkpoints**:
  `models/<id>/checkpoints/<ckpt_id>/{manifest.json, weights.pt}`, each with
  `weights_sha256` + lineage (`run_id`, method, dataset/version, step/epoch, metrics,
  parent checkpoint). Before writing, the content hash is computed: an identical artifact
  ⇒ the new file is a **hard link** (one physical copy of byte-identical states).
  Model manifests atomically gain `latest_checkpoint` / `best_checkpoint` and an
  append-only `training_provenance` list; the model's own `weights.pt` is only replaced
  after the run completes (or by explicit verified rollback).
- **Rollback endpoint**: `verify` the checkpoint's content hash against its manifest first;
  corrupted weights ⇒ refusal (409), weights and pointers untouched.
- **REST API**: `POST /training/run` (synchronous `TrainingReport`),
  `GET /models/{id}/checkpoints`, `GET /models/{id}/checkpoints/{ckpt}`,
  `POST /models/{id}/rollback`. Error mapping: 404 missing artifacts, 409 integrity,
  422 invalid configuration, preflight runs before anything is written.

### Milestone 4 — evaluation engine foundation
- **`app/evaluation.py`** — read-only measurement of an *existing* model state:
  - state selection: `model_id` is always required; `checkpoint_id=None` evaluates the
    model's current published `weights.pt` (verified against its `weights.sha256`
    sidecar), `checkpoint_id=<id>` evaluates that verified immutable checkpoint
    (reuses M3 `verify_checkpoint` — corrupted ⇒ refusal **before** the eval loop)
  - objective: the exact M3 causal-LM metric — fp32 token-weighted mean cross-entropy
    over `tokens[:-1] → tokens[1:]`, `perplexity = exp(min(loss, 100))`; no gradients,
    no RNG, no sampling, eval-mode dropout off
  - data policy identical to M3: tokenized `.bin` via memmap, window length =
    `max_seq_len` (default: the model's context length), file order, incomplete tail
    dropped, final partial batch handled like M3
  - `max_eval_tokens` caps the evaluated *target* tokens deterministically (whole
    windows then a partial final window) and sets `truncated=True`; because M2 streams
    are flat (no per-token record boundaries), a truncated evaluation reports
    `records_covered=None` — never an estimate. Full-split evaluations report the exact
    split record count from the M2 version manifest
  - **read-only contract**: evaluation only creates one immutable manifest under
    `models/<id>/evaluations/eval-<id>/manifest.json` (no weights/checkpoint/data
    copies); weights, checkpoints, provenance, datasets and earlier records are never
    touched (tested byte-for-byte)
  - **determinism**: two identical evaluations give identical metrics, counts and
    `result_hash`; the hash covers state/data/config/metrics only — never eval_id,
    timestamps or paths — so re-runs are comparable across records
  - **honest comparison semantics**: lower loss/perplexity is better *for this
    objective only*; train-loss vs validation-loss comparisons are a deliberate
    overfitting probe; no quality scores/leaderboards/rankings
- **REST API**: `POST /evaluations/run` (EvaluationConfig → EvaluationRecord),
  `GET /models/{id}/evaluations` (404 unknown model, `[]` when none),
  `GET /models/{id}/evaluations/{eval_id}`. Error mapping matches M3: 404 missing
  artifacts, 409 integrity/corruption, 422 invalid configuration, 500 only unexpected.

### Milestone 5 — comparison engine foundation
- **`app/comparison.py`** — evidence-based A/B comparison sitting above M3+M4:
  - question answered: *two states of a model, measured under IDENTICAL probe
    conditions — did the loss improve, regress or stay unchanged?*
  - **one shared probe** per comparison: dataset/version/split/tokenizer/window/
    token-cap/batch/seed are identical for A and B by construction — only the model
    state differs (`state_kind` current|checkpoint + `checkpoint_id`)
  - **both states verified up front** (M1 sidecar check for current weights, M3
    `verify_checkpoint` for checkpoints) — missing → 404, corrupt → 409, all before
    any evaluation or comparison artifact exists
  - **evaluation reuse**: each side resolves to an existing immutable M4 evaluation
    only when its complete identity matches (state kind/id + canonical state hash +
    probe fields); otherwise exactly one new evaluation is created via M4. Repeating
    a comparison creates **no** duplicate evaluations; the two referenced evaluation
    ids are recorded per side
  - **verdict is loss-only**: `|loss_B − loss_A| ≤ tolerance` ⇒ `unchanged`;
    `loss_B − loss_A < −tolerance` ⇒ `improved`; `> tolerance` ⇒ `regressed`.
    Perplexity is supporting information. **"Improved/regressed" describes measured
    loss on the specified probe — it is not a universal judgment of model quality.**
  - **guards**: comparisons across different probes (split/dataset/version/tokenizer/
    window/cap/seed) or across models with different configurations are refused
    (422) — no artifacts are created
  - **immutable history**: `models/<id>/comparisons/comp-<id>/manifest.json`,
    append-only (each request is auditable; only evaluations are deduplicated),
    with a deterministic `result_hash` over inputs/results only (never ids/timestamps)
- **REST API**: `POST /comparisons/run` (ComparisonRequest → ComparisonRecord),
  `GET /models/{id}/comparisons` (404 unknown model, `[]` when none),
  `GET /models/{id}/comparisons/{comparison_id}`. Error mapping: 404 / 409 / 422.

----
### Milestone 6 — stage gates & policy-driven run decisions
- **`app/gates.py`** — the stage gate sits ON TOP of M5/M4 and reuses them (no duplicated
  math): it answers one explicit question — *given this policy (fixed probe + baseline
  form + tolerance/constraints), this candidate state and this exact evaluation probe,
  should the candidate be accepted?*
  - **inline policies, no registry**: a `GatePolicy` (probe fields, `baseline_type`,
    `tolerance`, optional `max_regression_delta` / absolute `minimum_loss` ceiling) is
    supplied with every request and **embedded verbatim in the decision** — every gate
    request is self-contained and auditable; there is no mutable policy store to drift
  - **four baseline forms**: (A) a specific immutable **checkpoint**; (B) the model's
    **current** published weights; (C) a **previously recorded evaluation result hash**
    (accepted only when the full probe identity — model, dataset/version/split/
    tokenizer/window/cap/batch/seed — matches the policy; an unrelated evaluation is
    never the baseline); (D) an **absolute minimum-loss threshold** with no state
    baseline and therefore no fabricated comparison. Ambiguous/contradictory baseline
    combinations are rejected by the schema
  - **explicit decision semantics** (verdict is M5's loss-only verdict; the gate adds
    policy semantics on top): `improved` → pass · `unchanged` (within tolerance) → pass ·
    `regressed` → pass **only** when `delta ≤ max_regression_delta` is configured, else
    fail · an absolute `minimum_loss` ceiling can additionally fail any candidate
    (`delta_loss_nats = candidate − baseline`; negative = candidate better)
  - **evidence chain is complete and honest**: each `GateDecision` records per-side
    state kind/checkpoint id/state hash/evaluation id + result hash, losses, delta,
    verdict, `comparison_id` + `comparison_result_hash` (when a state baseline exists —
    threshold-only gates never invent one), decision, reason, and the full policy
  - **safe on failure**: a failed gate against a checkpoint baseline records
    `suggested_checkpoint_id` + `hint: "rollback recommended"` — it **never rolls back,
    retrains or selects**. The user executes the rollback via the existing M3 endpoint
  - **verification order & integrity**: model → dataset/version → candidate state hash
    (live recomputation, never stale) → baseline resolution → evaluations → comparison.
    Corrupt/missing states are refused (409/404) before any artifact exists; weights are
    never touched by a gate
  - **reuse & auditability**: repeated identical requests create **no duplicate
    evaluations or comparisons** (exact identity reuse) yet **append a fresh immutable
    decision** at `models/<id>/gates/gate-<decision_id>/manifest.json`. `result_hash`
    is deterministic over semantic inputs/results only (ids/timestamps excluded)
- **REST API**: `POST /gates/evaluate` (GateRequest → GateDecision),
  `GET /models/{id}/gates/decisions`, `GET /models/{id}/gates/decisions/{decision_id}`.
  Error mapping: 404 missing artifacts, 409 corrupt states, 422 invalid policy/probe.

----
### Milestone 7 — orchestrated workflows over stage gates
- **`app/workflows.py`** — `WorkflowEngine` composes the existing M1–M6 engines into
  ordered, auditable, fully synchronous workflows. It orchestrates; it never replaces:
  no new training/evaluation/comparison/gate logic exists anywhere in the milestone —
  each stage runs the *actual* M3/M4/M5/M6 engines and records what happened
- **plans are inline and immutable**: a `WorkflowPlan` (name, `model_id`, ordered
  stages with unique ids, optional description) is submitted with every run and
  **embedded verbatim in the run record** — there is no plan registry, nothing to
  edit/delete, and each run stays fully auditable on its own. All stages operate on
  ONE subject model; every embedded config must agree (schema-enforced)
- **stage types**: `train` (literal M3 `TrainingConfig`, run exactly as supplied —
  no invented configs, no retries, no auto-tuning), `evaluate` (M4 probe + optional
  `checkpoint_from_stage`), `compare` (M5-style two-state pair over one shared probe),
  `gate` (full M6 `GatePolicy` + candidate state). Explicit stage references point at
  earlier stages by id (`from_stage`, `on_pass`, `on_fail`); unknown references and
  references to stages that produced no artifact fail validation/execution cleanly —
  ids are never fabricated
- **conditional execution on real M6 decisions**: a gate stage branches on the actual
  persisted `passed|failed` decision (`on_pass`/`on_fail` → forward jumps, the skipped
  middle stages are recorded as skipped). A failed gate with **no** `on_fail` STOPS the
  workflow: status `stopped`, with `suggested_checkpoint_id` + hint read back from the
  persisted M6 decision. Rollback is recorded as a suggestion and **never executed** —
  only the user can call the M3 rollback endpoint
- **run records are one immutable manifest per run** at
  `models/<id>/workflows/workflow-<workflow_id>/manifest.json` (atomic write, mkdir
  `exist_ok=False`): status `completed|failed|stopped`, per-stage executed/skipped +
  artifact or error, transitions `(stage, decision, to_stage)`, failed stage + terminal
  reason, referenced artifact ids, deterministic `result_hash` (ids/timestamps/paths
  excluded → identical plans over identical evidence reproduce it). Partial history is
  legitimate: `stage1 ok, stage2 ok, stage3 failed` records the earlier artifacts and
  the failing point; successful artifacts are never deleted
- **evidence is reused, never duplicated**: evaluations/comparisons resolve through the
  same exact-identity reuse the M4/M5/M6 engines use (identical states + probes share
  artifacts); each gate execution and each run appends exactly one new immutable
  audit manifest
- **REST API**: `POST /workflows/run` (synchronous), `GET /models/{id}/workflows`,
  `GET /models/{id}/workflows/{workflow_id}`; no update/delete/cancel/background/schedule
  endpoints exist. Error mapping reuses M1–M6 conventions: 404 missing
  model/artifact/stage/checkpoint/evaluation/comparison, 409 corruption/integrity,
  422 invalid workflow/stage/dependency/transition/reference

### Milestone 8 — read-only dashboards over the immutable histories
- **`app/dashboards.py`** — `DashboardEngine` is a deterministic, read-only observer: it
  scans the persisted M1–M7 manifests under one model (`checkpoints/`, `evaluations/`,
  `comparisons/`, `gates/`, `workflows/` + the model manifest's training provenance) and
  recomputes one JSON dashboard per GET. It never writes — no cache, no database, no
  new files, no control over any M1–M7 artifact
- **nothing is invented**: every reported value comes from a persisted manifest, and
  every series/group/key is derived deterministically from recorded fields:
  - **checkpoint lineage** — chronological `(created_at, id)` order (the true chain
    order: every recorded `parent_checkpoint_id` precedes its child), each checkpoint
    tied to its recorded run via the model manifest's `training_provenance`
  - **evaluation series** — grouped by the exact M4/M5/M6 probe identity (state
    kind/id/content-hash + dataset/version/split/tokenizer + window/token-cap/batch/
    seed); series only ever merge evaluations that are genuinely identical, and expose
    the recorded per-evaluation values verbatim
  - **comparison series** — grouped by ordered state pair + probe + tolerance; M5
    semantics preserved verbatim (`delta_loss_nats = loss_B − loss_A`, recorded verdict,
    tolerance, hashes, timestamps) — no new scores or ranking is added
  - **gate decision series** — grouped by the recorded policy configuration (sha256 of
    the persisted policy); per-policy statistics are descriptive `passed/failed` counts,
    never a model-quality score
  - **workflow summary** — all run records in persisted order plus descriptive status
    counts; duration only when derivable from persisted fields
  - **artifact reference graph** — deterministic nodes/edges from recorded references
    only (checkpoint parents, run-produced/final checkpoints, evaluation state,
    comparison state_a/state_b, gate candidate/baseline/evidence/suggestion, workflow
    stage artifacts); every node id resolves through the artifact system; an internal
    reference to a missing artifact becomes a diagnostic, never an invented edge
- **determinism**: identical storage → identical semantic JSON. Ordering never depends
  on filesystem traversal (records sort by persisted `(created_at, id)`; groups/nodes/
  edges sort canonically), no volatile ids or request timestamps are generated, and
  `result_hash` covers the canonical JSON of the dashboard itself (excluding the hash —
  field docstring documents the exact coverage), so repeated GETs are byte-stable
- **corruption resilience**: an unreadable manifest (missing file / invalid JSON /
  schema mismatch) is skipped and reported in `diagnostics` (family + directory-derived
  id + deterministic issue); valid artifacts stay visible and the dashboard never
  crashes. The same applies to references into skipped artifacts
- **REST API**: `GET /models/{id}/dashboard` returns the full per-model dashboard
  (sections A–F above + diagnostics + `result_hash`). Error mapping reuses M1–M7
  conventions: 404 unknown model, 409 corrupt model manifest, 422 invalid input — the
  endpoint performs **no writes of any kind** and no update/delete/edit endpoints exist
- **scope is observation only**: M8 adds no measurement semantics of its own (no
  quality scores, no cross-model leaderboards, no recommendations, no rollback/
  retraining/selection), executes no writes, and never modifies, moves or deletes a
  manifest — dashboards are views over the append-only evidence, recomputed live

### Milestone 9 — stable policy registry & named probe suites
- **`app/policies.py`** — `PolicyEngine` manages two kinds of immutable DEFINITIONS at
  the storage root (`policies/<policy_id>/manifest.json`,
  `probe-suites/<suite_id>/manifest.json`, one atomic manifest each; no database, no
  registry service, no duplicated decision history). It organizes and names what M6/M4
  already know; it never invents decision mathematics, scores or probe identity
- **policies**: `PolicyDefinition` = stable user-chosen `policy_id` + the exact M6
  `GatePolicy` + `config_hash` (sha256 over the canonical JSON of the policy alone —
  no ids/timestamps/description/paths, so identical semantics always reproduce it).
  Registration is immutable: the same id with identical content resolves idempotently
  to the existing definition; the same id with any different content is rejected
  (API 409) — never silently overwritten. Definitions are separate from execution:
  a `GateDecision` remains the audit event; policy records never contain decision
  history
- **probe suites**: `ProbeSuite` = stable `suite_id` + a set of `SuiteProbe`s. A probe
  carries the exact M4 probe-identity fields (dataset/version/split/tokenizer/window/
  token-cap/batch/seed — mirroring `EvaluationConfig`, `extra="forbid"`) and resolves
  via `to_evaluation_config(model_id, checkpoint_id)` into an ordinary M4
  `EvaluationConfig`, so evaluations produced from a suite are indistinguishable from
  directly configured ones (same result hashes, same dashboard series keys). Probes
  are canonicalized on registration (order carries no evaluation meaning and never
  influences identity or output); duplicate probes and empty suites are rejected;
  `probes_hash` is sha256 over the canonical sorted probe set. A suite computes no
  score — each probe stays an independent exact M4 probe
- **gate integration**: `GateRequest` and `WorkflowGateStage` accept EXACTLY ONE policy
  source — inline `policy` (M6, unchanged) or registry `policy_id` (resolved by
  `GateEngine` through `PolicyEngine` before anything runs; unknown ids fail cleanly
  with 404 semantics, a definition targeting another model is rejected). Both paths
  execute the identical M6 logic and embed the executed policy in the decision; a
  registry decision additionally records `policy_id` + `policy_config_hash`
  provenance so it stays interpretable even as future definitions are added. Inline
  policies keep producing decisions byte-identical to historical M6 behaviour (no
  provenance fields), so every existing decision/manifest stays untouched
- **workflow integration**: M7 gate stages may set `policy_id` instead of an inline
  policy (schema-enforced XOR); references point at immutable definitions, unknown
  references fail the run cleanly at the failing stage. Workflows never auto-select
  policies or probe suites — the plan states exactly what it wants
- **dashboard integration**: historical M6 decisions render exactly as before; a
  decision that used a registry policy additionally carries `policy_id` and
  `policy_config_hash` (recorded provenance flows through unchanged). No historical
  manifest is rewritten and nothing is recalculated under new definitions
- **REST API**: `POST/GET /policies`, `GET /policies/{id}` and `POST/GET
  /probe-suites`, `GET /probe-suites/{id}` — create (201, idempotent for identical
  content), list and get only. No PUT/PATCH/DELETE: a new version is a new immutable
  id. Errors follow the existing mapping: 404 unknown id, 409 conflicting immutable
  definition, 422 malformed
- **scope is definitions only**: no automatic execution of suites, no multi-probe
  aggregation or averaging, no quality scores, no rankings, no auto-gating — a suite
  is a named evidence set and a policy a named rule, both explicit, deterministic,
  immutable and user-controlled

### Milestone 10 — explicit multi-probe evaluation batches over named probe suites
- **`app/suite_runs.py`** — `SuiteRunEngine` is a THIN ORCHESTRATION LAYER over M4:
  one `SuiteRunRequest` (model_id + suite_id + one explicit state using the M4/M5
  `ComparisonState` convention) resolves the M9 suite, then executes each probe
  independently in canonical suite order through the existing M4 engine. Loss,
  perplexity, batching, truncation, seeds, state hashing and verification stay owned
  by M4 — nothing here duplicates evaluation mathematics
- **evidence reuse is exact M4 identity**: each probe reuses an existing evaluation
  only when state (kind/id/canonical hash), dataset/version/split/tokenizer, window,
  token cap, batch size and seed all match — the same reuse semantics M5–M9 use.
  A repeated identical suite run appends only the new run manifest; no duplicate
  evaluation manifests are ever created
- **a suite is a set of probes, NOT a benchmark score**: the run record exposes each
  probe and its evaluation id independently plus execution bookkeeping counts
  (`probe_count`/`completed_count`/`reused_count`/`failed_count`). No average loss,
  no perplexity aggregate, no score, no pass percentage, no ranking exists anywhere —
  comparisons and gates still operate one probe at a time, explicitly requested
- **deterministic failure semantics**: model, state (verified canonical hash) and
  suite are resolved BEFORE anything executes — unknown model/suite/checkpoint and
  corrupt suites fail cleanly (404/409 semantics) with nothing persisted. A probe
  that cannot execute because of a missing/invalid underlying artifact (dataset,
  tokenizer, …) is recorded per-probe as `failed` with deterministic error text and
  `evaluation_id: null` — no fabricated result, no silent skip; the run persists with
  status `failed` and remaining probes still execute in order
- **immutable persistence**: one atomic manifest per run at
  `suite-runs/<suite_run_id>/manifest.json` (storage root, next to the M9 definition
  dirs) containing only references — evaluations, weights, datasets and tokenizers
  are never copied. `result_hash` covers the semantic execution (model + config
  identity, state incl. canonical hash, suite id + probes hash, canonical
  probe→evaluation mapping, status, failure text) — never the run id, timestamps,
  durations, paths or created/reused bookkeeping labels — so two logically identical
  runs over identical immutable evidence reproduce it exactly
- **REST API**: `POST /suite-runs` (execute one suite against one state),
  `GET /models/{id}/suite-runs` (deterministic oldest-first list) and
  `GET /models/{id}/suite-runs/{suite_run_id}` (one persisted run; 404 for unknown
  model/run). No update/delete endpoints exist (405) — runs are immutable evidence.
  Error mapping follows M1–M9: 404 unknown model/suite/state, 409 corrupt
  definitions/evidence, 422 malformed requests
- **compatibility**: M8 dashboards keep rendering the underlying M4 evaluations
  exactly as before (suite runs simply reference those evaluations); M5 comparisons,
  M6 gates and M7 workflows are untouched — suites never trigger them automatically

### Milestone 11 — suite-run stages inside M7 workflows
- **`suite_run` is a new M7 stage type** (schema + engine only): a stage declares
  exactly `{suite_id, state}` where `state` reuses the M7 `StageStateRef`
  representation — current weights, a literal immutable checkpoint id, or
  `from_stage` pointing at an EARLIER train stage whose final checkpoint is the
  evaluated state. Nothing is ever guessed: no implicit "latest", no history search,
  no positional inference; dependency validation reuses the exact M7 rules (unknown /
  non-earlier / non-train references are rejected at plan validation)
- **execution is a thin adapter over M10**: when the workflow reaches the stage it
  resolves the state, constructs the existing M10 `SuiteRunRequest` and calls the
  existing `SuiteRunEngine` — the same engine, persistence layout, exact-M4 identity
  reuse, canonical probe order, deterministic result hashes and per-probe failure
  semantics as a standalone suite run. No second suite-run implementation exists
- **the workflow record references the suite-run artifact**: the stage result stores
  `kind="suite_run"`, the immutable `suite_run_id`, its result hash and the evaluated
  checkpoint (when checkpoint-based); the full per-probe detail stays in the
  suite-run manifest — no duplicated evidence in the workflow record
- **failure semantics follow M7, with M10's distinction preserved**: a fatal preflight
  error (unknown/corrupt suite, invalid state) makes the stage fail — the workflow
  persists a `failed` run identifying the stage and no fabricated suite-run artifact
  exists. An underlying suite run that persisted with status `failed` (individual
  probes failed) is a REAL artifact and is referenced as such — the workflow never
  converts it into success and never invents a suite-level quality judgment
- **control flow unchanged**: a suite stage is a normal non-branching stage
  (transition `next`); gate stages after it still branch on their real M6 decisions
  via the existing `on_pass`/`on_fail` mechanism — no custom suite branching exists
- **no automatic gates**: the suite produces per-probe evaluations only; gating any
  probe requires an explicit later M6 gate stage exactly as before
- **REST API**: no new endpoints — `POST /workflows/run` accepts plans containing the
  stage (OpenAPI reflects it automatically); the M10 suite-run endpoints are
  unchanged. Storage per first workflow run: one workflow manifest + one suite-run
  manifest + zero-or-more genuinely missing M4 evaluations; repeated equivalent
  workflows add one workflow + one suite-run manifest and zero evaluations

### Milestone 12 — named workflow recipes (immutable reusable M7 plans)
- **concept**: a workflow recipe is INERT DATA — a user-chosen `recipe_id`, an
  optional description and an ordered `WorkflowStage` list (the exact M7 union:
  train / evaluate / compare / gate / suite_run — no recipe-specific stage language).
  It is registered once at `workflow-recipes/<recipe_id>/manifest.json` (one atomic
  manifest, M9 conventions) and executes only when an explicit
  `POST /workflows/recipes/{id}/runs` binds ONE model. There is no automatic
  execution, scheduling, repetition, or model/checkpoint/suite selection anywhere
- **one validator, one executor**: registration validates the stage list through the
  SAME shared structural rules as inline `WorkflowPlan` validation
  (`validate_plan_stages` — duplicate ids, unknown/self/forward `from_stage` refs,
  non-train targets, forward-only gate branches; identical error messages by
  construction). Running a recipe resolves it, verifies the explicitly supplied
  model exists, converts the stages into an existing `WorkflowPlan` (name = recipe
  id) — full plan validation again, including the rule that every embedded stage
  config must target the bound model — and delegates to the existing
  `WorkflowEngine`. `RecipeEngine` is a thin registry/resolver; it never executes a
  stage itself
- **model binding is the only runtime parameter**: `state_kind: current` is resolved
  against the bound model at run time; literal checkpoints are honored exactly;
  `from_stage` resolves inside the run against the recipe's own earlier train stages
  (no history search). Every run record snapshots what actually happened
  (checkpoint id + state hash per stage) so recipes stay immutable while runs stay
  independently auditable — the same recipe can legitimately snapshot different
  states across runs
- **recipe provenance is additive on run records**: recipe-generated workflow runs
  record `recipe_id` + `recipe_hash` (the recipe's deterministic config hash) while
  historical inline runs keep `recipe_id`/`recipe_hash` null. The recipe is never
  copied into the run and suite/evaluation data is never duplicated — records stay
  reference-oriented. `config_hash` is sha256 over the canonical ORDERED stage JSON
  only (no recipe id, description, timestamps, paths): identical semantics across
  different ids reproduce the same hash, any semantic stage change changes it
- **identity/immutability**: same id + identical content → returns the existing
  definition (no write); same id + different content → 409; no update/delete/rename/
  version endpoints exist (405); registration and runs never rewrite any manifest
- **evidence reuse unchanged**: repeated recipe runs produce distinct workflow and
  suite-run records (each explicit invocation is an auditable event) while exact M4
  evidence is reused — zero duplicate evaluation manifests when evidence exists
- **failure semantics unchanged**: unknown recipe/model → 404 with nothing persisted;
  a recipe whose embedded configs pin a different model than the bound one is
  rejected (422) before any write; runtime stage failures (missing suite/checkpoint,
  corrupt evidence) persist a `failed` run record with recipe provenance exactly like
  inline plans — no fabricated artifacts; a persisted suite run with per-probe
  failures is referenced as-is
- **dashboard**: M8 unchanged; recipe runs appear in workflow history with their
  provenance fields (additive, deterministic) — no recipe scores, rankings or
  benchmark numbers are invented
- **REST API**: `POST /workflows/recipes` (201; idempotent repeats return the
  existing definition), `GET /workflows/recipes`, `GET /workflows/recipes/{id}`
  (404 unknown), `POST /workflows/recipes/{id}/runs` with body `{"model_id": "..."}`
  → the workflow run record. OpenAPI exposes all four. Storage: registration adds
  exactly one recipe manifest; each successful run adds one workflow manifest plus
  one suite-run manifest per suite stage, and zero M4 evaluations when evidence
  already exists; a run that fails before persistence adds nothing

----
### Milestone 13 — read-only suite-run dashboard views & recipe-run lineage
- **concept**: M13 adds observation, never action. Two read-only surfaces extend the
  existing deterministic views — the M8 dashboard shows the model's M10 suite-run
  history (the records the evaluations it already renders belong to), and recipe-run
  provenance becomes queryable across models. Nothing new is stored, indexed or
  computed into a judgment
- **dashboard `suite_runs` section**: a per-model summary shaped like the `workflows`
  section (`counts` + `records`). Records are the model's `SuiteRunRecord`s scanned
  live from the STORAGE-ROOT `suite-runs/` family (the M10 layout — the dashboard
  never copies suite runs under model dirs) and filtered by the manifest's persisted
  `model_id`. Ordering is `(created_at, suite_run_id)`; `counts` hold only the exact
  M10 statuses (`completed`/`failed`) of visible records. Every record keeps its
  per-probe data exactly as recorded: probe definition, `evaluation_id`, outcome
  (`created` = this run created the exact M4 evaluation, `reused` = it reused
  existing exact evidence, `failed` = recorded per-probe failure) and error where
  M10 persists it. No suite score, average, pass percentage, benchmark, quality
  ranking or leaderboard exists anywhere — `reused` describes evaluation-artifact
  reuse, never probe success. A model without suite runs gets the deterministic
  empty section `{"counts": {}, "records": []}` (never null/omitted); corrupt or
  manifest-less suite-run dirs are skipped with the same deterministic diagnostics
  used by every other family, and valid records stay visible
- **artifact graph**: suite runs become real graph nodes (`suite_run` family, bare
  run id — the existing node conventions); a workflow stage whose artifact is a
  suite run yields a deterministic `workflow → suite_run` edge (`stage_artifact`),
  and each recorded per-probe `evaluation_id` yields a `suite_run → evaluation`
  edge (`probe`) — only when the referenced evaluation really exists. Missing
  evaluation or corrupt manifest → the standard deterministic diagnostic; no node
  or edge is ever fabricated. The graph stays model-scoped and hash-covered
- **recipe provenance unchanged**: M12's additive `recipe_id`/`recipe_hash` fields
  keep their exact semantics and remain visible in workflow history; M13 invents no
  recipe success rate, quality number, benchmark or ranking
- **recipe-run lineage**: `GET /workflows/recipes/{recipe_id}/runs` lists every
  persisted workflow run whose manifest carries that `recipe_id` — a LIVE scan of
  the existing per-model workflow manifests, therefore cross-model (a recipe may be
  bound to different models over time). Each record is the full workflow run record
  annotated with `workflow_id`, `model_id`, `recipe_id`, `recipe_hash`, `status`,
  `created_at` etc. Ordering is `(created_at, workflow_id)`, never filesystem or
  model order. Unknown recipe → 404 (the recipe definition is verified to exist
  first); existing recipe with no runs → deterministic `[]`. No index, cache, new
  storage family or write of any kind; the existing per-model endpoints are
  untouched and keep returning identical shapes
- **zero-write guarantee**: dashboards and lineage are recomputed from immutable
  manifests on every call — M13 performs zero writes (no manifests, indexes,
  caches, timestamps, mutations). Repeated calls over identical storage produce
  byte-identical responses, and `result_hash` (sha256 over the complete canonical
  dashboard JSON excluding itself) changes if and only if the underlying storage
  state changes — adding the suite-run artifacts changed it exactly once
- **REST API**: one new read-only endpoint, `GET /workflows/recipes/{id}/runs`
  (200 list / 404 unknown recipe); the dashboard shape gains only the additive
  `suite_runs` section. No schema-version bump, no rewritten history, no new error
  framework; PUT/PATCH/DELETE remain 405 on every immutable path
- **no suite scoring / no recipe scoring**: M13 renders recorded execution facts
  and references; any view that would require computing a quality judgment,
  aggregate score, ranking or new storage is out of scope by definition

### Milestone 14 — composable workflow recipes (`recipe` stages)
- **concept**: M12 recipes are inert ordered stage lists; M14 lets one registered
  recipe REUSE others through a `recipe` stage (one `recipe_id` reference, the
  conventional `WorkflowStage` shape — no parameters, no runtime substitution,
  no dynamic execution). A composition is a STATIC definition: it expands into
  exactly ONE fully validated M7 `WorkflowPlan` executed by ONE
  `WorkflowEngine.run()` producing ONE workflow record — never nested runs or
  per-subrecipe records
- **registration order (any failure ⇒ nothing persisted)**: shared schema rules
  (recipe stages allowed only in registered recipe definitions — inline
  `WorkflowPlan`s reject them) → every referenced recipe must ALREADY exist in
  the immutable registry (422, never deferred to runtime) → cycle rejection
  (self, A→B→A, longer — reachable only through registry states manipulated
  outside the API; the engine still detects them) → depth limit: compositions
  may chain at most 32 recipes (32 accepted, 33 rejected) → deterministic
  expansion → the SHARED `validate_plan_stages` over the FULLY EXPANDED stage
  list → composite config hash → one immutable manifest write
- **deterministic expansion convention**: stages of the top-level recipe keep
  their declared ids and order; a referenced recipe's stages are spliced in
  parent order at the call position, each id qualified by the dot-joined chain
  of call-stage ids that led to it (`call.stage`, nested: `a.b.stage`). All
  INTERNAL references of the referenced recipe — `from_stage` state pointers
  and gate `on_pass`/`on_fail` branches to its own non-call stages — are
  rewritten to the qualified ids, so they keep binding inside the same
  expansion; targets that would cross a boundary (e.g. branching into a call
  stage that vanishes during expansion) are rejected at registration — rules
  are never invented at runtime. `current`/checkpoint/`from_stage` semantics
  are the existing M7/M10/M12 ones
- **hashing & provenance**: plain recipes keep the exact M12 config hash;
  composite config hashes are sha256 over the DECLARED ordered stages PLUS the
  referenced recipes' ids and config hashes (declared order) — immutable
  dependencies pin their consumers deterministically (same content → same
  hash, composite id/description/timestamps/paths excluded; changed stage or
  changed dependency → changed hash). Composite manifests store the declared
  stage list (recipe stages intact) plus a reference-oriented `composition`
  (recipe_id + config_hash per DIRECT dependency) — referenced manifests are
  never copied. Run records keep M12 `recipe_id`/`recipe_hash` (the top-level
  recipe the caller invoked) plus an additive depth-first `composition` trace;
  inline-plan runs keep all three null
- **execution & model binding**: exactly one explicit `model_id` per run (M12
  unchanged); nested recipes never bind or auto-select a model; a model pinned
  inside a referenced recipe's config stays intact after expansion and a
  conflict with the bound model is rejected BEFORE execution (nothing
  rewritten, nothing persisted). Suites inside composed recipes run through the
  existing M10 engine with exact M4 evidence reuse — zero duplicate evaluation
  manifests (same rules as M11/M12)
- **lineage truth (M13 semantics unchanged)**: the top-level composite run
  appears only under the invoked recipe's lineage; referenced recipes never
  claim it — composition provenance is not execution ownership. Ordering
  stays `(created_at, workflow_id)`; repeats stay byte-identical
- **REST API**: the five existing recipe endpoints are unchanged; POST
  registration accepts the new stage (idempotent identical re-registration
  stays 201; conflict stays 409 with original bytes untouched; unknown
  reference / cycle / depth / invalid-expanded-plan → 422 with no manifest);
  POST run executes the expansion; OpenAPI exposes the `recipe` stage and the
  reference model. No `/compose` or `/expand` endpoints exist
- **no new machinery**: no second registry/validator/executor, no scheduling,
  queues, workers, runtime parameters, editing/deletion/versioning, no suite
  aggregation or quality numbers, no dashboard redesign — dashboard
  determinism and storage accounting follow the exact M8/M13 rules

### Milestone 15 — deterministic checkpoint sampling (`samples/`, M15)
- **concept**: given ONE explicit model + checkpoint + tokenizer + prompt +
  strategy, the sampling engine verifies the checkpoint (content hash vs its
  manifest `weights_sha256`, through the existing M3 machinery), rebuilds the
  model from its stored config with the existing builder/restore primitives
  (the exact M4 convention), and decodes autoregressively under
  `torch.no_grad()` through the existing forward path with the incremental
  KV cache. One immutable manifest per request under `samples/<model_id>/
  sample-<id>/manifest.json`; nothing else is ever written
- **exactly two strategies**: `greedy` = per-step argmax over the logits with
  no RNG anywhere; `temperature` = logits/temperature -> softmax -> one sample
  per step from a CPU `torch.Generator` seeded with the request's explicit
  integer seed (`0 <= seed < 2**32`). `temperature` must lie in (0, 1];
  `temperature=0` is REJECTED, never reinterpreted as greedy. Greedy forbids
  temperature/seed fields; temperature requires both. Same checkpoint +
  tokenizer + prompt + parameters + seed => identical token ids, identical
  text, identical `result_hash`
- **vocabulary rule (the platform convention, not a new one)**: the model's
  `vocab_size` must be >= the tokenizer's actual vocab (the exact check M3/M4
  already document); logits are restricted to the tokenizer's actual vocab at
  every decode step so every generated token id is decodable — a checkpoint
  this platform trained is always sampleable through its tokenizer. Requests
  violating the rule are rejected 422 before any decode
- **no EOS policy**: generation stops after exactly `max_new_tokens` (the
  stored tokenizers define no post-processor and the models were not trained
  with EOS boundaries; inventing one would be a new rule). Every request must
  fit the model's context: `prompt_tokens + max_new_tokens <= context_length`
  is enforced at preflight (422) — never silently truncated
- **reproducibility**: `result_hash` = sha256 over model/checkpoint identity +
  the VERIFIED checkpoint weights content hash + tokenizer id + tokenizer.json
  content hash + prompt verbatim + strategy + temperature + seed +
  max_new_tokens + generated token ids + output text. Sample id, timestamps,
  duration, hardware and paths are excluded, so identical requests over
  identical immutable inputs reproduce the hash byte-for-byte
- **REST API**: `POST /samples/generate` (one request -> one manifest; unknown
  model/checkpoint/tokenizer -> 404, corrupt checkpoint -> 409, vocabulary/
  parameter/context violations -> 422, all with zero writes),
  `GET /models/{id}/samples` (deterministic (created_at, sample_id) order,
  read-only, 404 unknown model), `GET /models/{id}/samples/{sample}` (404
  unknown). No update/delete/regenerate endpoints, no dashboard/graph/workflow/
  recipe integration
- **quality boundary**: M15 is an inference primitive — it does NOT evaluate,
  score, rank, judge, benchmark or improve generated text, and nothing
  auto-retries or feeds back into the model

### Milestone 16 — per-sample quality measurement (`sample-evaluations/`, M16)
- **concept**: the platform's first MEASUREMENT layer over generated text.
  Given ONLY `model_id` + `sample_id` (no request body config), the engine
  resolves everything from the immutable M15 sample manifest — the recorded
  checkpoint and tokenizer are verified (content-hash integrity through the
  existing M3/M15 machinery + the platform vocabulary convention
  `model vocab_size >= tokenizer actual vocab`) and the full recorded
  prompt + generated token sequence is scored in ONE context window under
  the EXISTING M4 causal-LM objective. Nothing is auto-selected; nothing is
  ever supplied separately
- **exact target accounting**: causal shift-by-one cross-entropy is
  accumulated ONLY over the generated continuation targets — the first
  generated token is conditioned on the full prompt (scored from the last
  prompt position's logits), prompt tokens are NEVER scored as
  generated-text targets, and every generated target is counted exactly
  once (`evaluated_token_count == generated_token_count`). One forward
  pass under `torch.no_grad()`, fp32, no RNG
- **metrics**: `loss_nats` (mean causal cross-entropy over the generated
  targets, rounded 6 decimals like M4) and `perplexity = exp(min(loss, 100))`
  — the exact M4 convention. Perplexity is a likelihood measurement under
  one causal-LM objective; it is NOT an overall quality judgment and lower
  values do not by themselves mean better/safer/more useful/more human-like
  text
- **window rule (ONE deterministic rule, no silent truncation)**: the full
  sequence must fit ONE model context window — every M15 sample satisfies
  this by M15's own generation-time preflight. An overlong (edited/invalid)
  sample is rejected at preflight 422 with zero writes rather than silently
  truncated or scored under invented sliding-window accounting
- **verification, in fixed order**: unknown model/sample/checkpoint/
  tokenizer -> 404; unreadable manifest, identity-field mismatch, recorded
  checkpoint/tokenizer hash mismatch, or a sample whose own `result_hash`
  does not reproduce -> 409; window overflow, vocabulary violation,
  count/id inconsistencies -> 422. Internally inconsistent samples FAIL,
  never silently repaired. Every failure writes nothing
- **storage**: one successful measurement -> exactly ONE immutable manifest
  under `sample-evaluations/<model_id>/evaluation-<id>/manifest.json`
  (atomic). Samples, checkpoints, tokenizers, M4 evaluations, workflows,
  recipes and dashboards are never modified. Repeated identical requests
  create separate records (no silent deduplication) with a byte-identical
  semantic `result_hash` (model/sample identity + sample result_hash +
  canonical digest of the exact measured token sequence + checkpoint/
  tokenizer identity/hashes + target accounting + window rule/counts +
  metrics; evaluation id/timestamps/duration/hardware/paths excluded)
- **REST API**: `POST /models/{id}/samples/{sample}/quality` (body-less;
  the sample is the source of truth), `GET /models/{id}/sample-quality`
  (deterministic (created_at, evaluation_id) order), `GET /models/{id}/
  sample-quality/{eval}`. No edit/delete/rerun endpoints; no dashboard,
  workflow, recipe or graph integration; M15 generation endpoints unchanged
- **honest boundary**: M16 measures the likelihood of generated text under
  the existing causal-LM objective ONLY — no semantic quality, safety,
  hallucination, human-preference or overall-quality judgment of any kind,
  and no cross-sample averages or rankings

### Milestone 17 — sample-quality observability on the dashboard (`sample_quality`)
- **concept**: the M8/M13 model dashboard now exposes a deterministic,
  read-only `sample_quality` section derived LIVE from the model's immutable
  M16 `sample-evaluations/<model_id>/` manifests — no persistence, no cache,
  no index, zero storage growth (the dashboard stays a pure recomputation)
- **section content**: `total_count` (valid M16 records of the model),
  `by_sample` (per-sample history references ordered by `sample_id`;
  sample_result_hash as recorded by the newest evaluation + evaluation count
  + newest evaluation id/timestamp) and `latest` (every record newest-first,
  ordered by (created_at, evaluation_id) DESCENDING, references only)
- **references only, never metrics**: the section deliberately carries no
  loss/perplexity values and computes no averages, minimums/maximums,
  rankings, "best sample", cross-sample comparisons or quality verdicts —
  M16 records are referenced by id, their metric payload is not copied
- **ordering**: fully deterministic from persisted fields — records sorted
  by (created_at, evaluation_id), by_sample by sample_id, latest
  descending — never filesystem traversal order
- **corruption handling** (existing M8/M13 convention): malformed/
  unreadable/schema-invalid sample-evaluation manifests are skipped with a
  deterministic diagnostic entry (family + directory-derived id + issue);
  an evaluation referencing a missing sample directory is likewise
  diagnosed; nothing is repaired, fabricated or written
- **model isolation**: only `sample-evaluations/{model_id}/` contributes
  (records are model-scoped by their persisted `model_id` — a stray record
  under the wrong model directory is silently skipped, exactly like the M13
  suite-runs convention); `result_hash` covers the new section, so identical
  storage reproduces byte-identical dashboard JSON and hash; GET remains
  completely read-only

### Milestone 18 — sample-quality record access (`/sample-quality/records`)
- **concept**: one dedicated read-only listing route,
  `GET /models/{id}/sample-quality/records`, returns the model's FULL
  immutable M16 `SampleEvaluationRecord` payloads — including the recorded
  `loss_nats` and `perplexity` values — in the exact M16 authoritative
  order ((created_at, evaluation_id) ASCENDING). It is a pure pass-through
  of `SampleQualityEngine.list_sample_evaluations()`: no second engine, no
  manifest re-parsing, no duplicate representation in storage, zero storage
  growth
- **no statistics, no interpretation**: the endpoint returns the records
  verbatim and computes no averages, minimums/maximums, rankings, deltas,
  trends, cross-sample statistics or quality scores. `loss_nats` and
  `perplexity` remain what M16 recorded — likelihood measurements of one
  generated sample under the existing causal-LM objective — and are NOT an
  overall model-quality score
- **semantics**: empty history -> deterministic `[]`; unknown model ->
  existing 404; records are model-scoped (never aggregated across models);
  repeated GETs are byte-identical; the endpoint never writes. The M16
  reference listing `GET /models/{id}/sample-quality`, the individual M16
  getter and the M17 dashboard `sample_quality` section are unchanged

### Milestone 19 — per-sample quality history access (`by-sample`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/sample-quality/by-sample/{sample_id}` — exposing M16
  sample-quality history grouped by ONE known generated sample of the
  model. It answers: "what did we measure for this particular sample?"
  and nothing else
- **exact filtering**: the sample id must belong to the model — an unknown
  sample, or the sample id of a different model, is a 404 (the sample is
  resolved through the authoritative M15 `sampling.get_sample(model_id,
  sample_id)`; its persisted identity, never a filename or graph edge, is
  used as the filter). Returned records are that sample's M16
  `SampleEvaluationRecord` payloads verbatim — all fields including
  `loss_nats` and `perplexity` — in the exact M16/M18 authoritative order
  ((created_at, evaluation_id) ASCENDING); a known sample with no
  measurements is a deterministic `[]`
- **implementation is a reuse, not a second engine**: the engine method
  `list_sample_evaluations_for_sample(model_id, sample_id)` filters the
  authoritative M16 `list_sample_evaluations()` by the persisted sample id
  with the exact M16 listing/getter semantics; the facade and route are
  thin pass-throughs. No duplicate manifest parsing, no new storage, no
  caches, indexes, workers or derived statistics — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: a model can only ever see its own samples'
  measurements; another model's/sample's/dataset's/probe-suite's records
  are unreachable. Unknown model -> existing 404; unknown or foreign
  sample -> 404; empty history -> `[]`. M16 (listing, getter, metrics,
  hashes, manifests), M18 records and the M17 dashboard (hash
  `f48557fe...3838`) are byte-identical before and after

### Milestone 20 — per-checkpoint quality history access
  (`by-checkpoint`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/sample-quality/by-checkpoint/{checkpoint_id}` — answers
  "which immutable sample-quality evaluations belong to this checkpoint?"
  and nothing else: the model's authoritative M16 listing filtered by the
  persisted `checkpoint_id` recorded in each `SampleEvaluationRecord`
- **exact filtering**: ownership is validated through the model's M3
  checkpoint registry first — an unknown checkpoint, or a checkpoint id
  that belongs to another model, is a 404 (nothing is inferred from
  filenames or graph edges). Returned records are verbatim M16
  `SampleEvaluationRecord` payloads (all fields incl. `loss_nats` and
  `perplexity`) in the exact M16/M18/M19 authoritative order
  ((created_at, evaluation_id) ASCENDING); a valid checkpoint with no
  measurements is a deterministic `[]`
- **implementation is a reuse, not a second engine**: the engine method
  `list_sample_evaluations_for_checkpoint(model_id, checkpoint_id)`
  validates through the existing M3 `get_checkpoint` registry and filters
  the authoritative M16 `list_sample_evaluations()` by the persisted
  checkpoint identity; the facade and route are thin pass-throughs. No
  duplicate manifest parsing, no new storage, no caches, indexes, workers
  or derived statistics — repeated GETs are byte-identical and the
  endpoint never writes; it answers which measurements exist under one
  checkpoint and never judges the checkpoint
- **isolation & boundaries**: a model can only ever see its own
  checkpoints' measurements; another model's/checkpoint's/sample's/
  dataset's/probe-suite's records are unreachable. Unknown model ->
  existing 404; unknown or foreign checkpoint -> 404; empty history ->
  `[]`. M16 (listing, getter, metrics, hashes, manifests), M18 records,
  M19 by-sample and the M17 dashboard (hash `f48557fe...3838`) are
  byte-identical before and after

### Milestone 21 — per-suite suite-run history access (`by-suite`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/suite-runs/by-suite/{suite_id}` — answers "which
  immutable suite-run records belong to this model and this named suite?"
  and nothing else: the model's authoritative M10 listing filtered by the
  persisted `suite_id` recorded in each `SuiteRunRecord`
- **exact filtering**: suite existence is verified through the existing M9
  probe-suite registry first (an unknown suite is a 404 — suite identity is
  the persisted suite manifest, never inferred from run-directory names);
  the model is verified through the existing model registry (unknown model
  is a 404). Returned records are verbatim `SuiteRunRecord` payloads in the
  exact M10 authoritative order ((created_at, suite_run_id) ASCENDING); a
  valid suite with no runs for this model is a deterministic `[]`
- **implementation is a reuse, not a second engine**: the engine method
  `list_suite_runs_for_suite(model_id, suite_id)` resolves the suite via the
  existing M9 `get_suite` registry getter and filters the authoritative M10
  `list_suite_runs()` by the persisted suite identity; the facade and route
  are thin pass-throughs. No second suite-run system, no duplicate manifest
  parsing, no new storage, caches, indexes or aggregation — repeated GETs
  are byte-identical and the endpoint never writes; it answers which runs
  exist under one suite and never scores, ranks or compares them
- **isolation & boundaries**: a model only ever sees its own suite-run
  records; another model's runs (even of the same suite) are unreachable.
  Unknown model/suite -> existing 404s; valid suite without runs -> `[]`.
  M10 (run, listing, getter, hashes, manifests), the M13 dashboard and the
  M16–M20 sample-quality surface are byte-identical before and after

### Milestone 22 — per-suite bookkeeping summary (`by-suite/summary`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/suite-runs/by-suite/{suite_id}/summary` — answers "how
  many suite runs exist for this model and suite, which ones, and when did
  they run?" — a pure derived counting view over the M21 grouping
- **bookkeeping only**: the summary contains exactly `model_id`,
  `suite_id`, `total_count`, `run_ids` (the deterministic M21 ASCENDING
  order) and `earliest_created_at`/`latest_created_at` — nothing else. A
  valid suite with no runs returns a ZERO summary (count 0, empty ids,
  null timestamps), not a 404. No scores, rankings, averages, trends,
  regression judgments, verdicts or new metrics — ever
- **implementation is a reuse, not a second engine**: the engine method
  `list_suite_run_summary_for_suite(model_id, suite_id)` calls the M21
  `list_suite_runs_for_suite` filter (model + M9 suite-registry
  validation, persisted `suite_id`, exact M10 ordering) and derives the
  counts/min/max from the parsed records; the facade and route are thin
  pass-throughs. Nothing is persisted — the summary is recomputed
  deterministically per request; repeated GETs are byte-identical and the
  endpoint never writes
- **isolation & boundaries**: a model only ever sees its own runs;
  another model's runs are never counted (cross-model yields a zero
  summary). Unknown model/suite -> existing 404s. M21 by-suite (full
  record payloads), M10 (run/listing/getter), the M13 dashboard and the
  M16–M20 sample-quality surface are byte-identical before and after

### Milestone 23 — gate-decision history by policy (`by-policy`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/gates/decisions/by-policy/{policy_id}` — answers
  "which immutable gate decisions of this model were produced under this
  registered policy?" — the model's authoritative M6 listing filtered by
  the persisted `policy_id` recorded in each `GateDecision`
- **exact filtering**: policy existence is verified through the existing
  M9 policy registry first (an unknown policy is a 404 — policy identity
  is the persisted definition manifest, never inferred from
  gate-directory names); the model is verified through the existing model
  registry (unknown model is a 404). Returned records are verbatim
  `GateDecision` payloads in the exact M6 authoritative order
  ((created_at, decision_id) ASCENDING); a valid policy with no decisions
  for this model is a deterministic `[]`; inline-policy decisions keep
  `policy_id=null` and never appear
- **implementation is a reuse, not a second engine**: the engine method
  `list_decisions_for_policy(model_id, policy_id)` resolves the policy via
  the existing M9 `get_policy` registry getter and filters the
  authoritative M6 `list_decisions()` by the persisted policy identity;
  the facade and route are thin pass-throughs. No second gate system, no
  duplicate manifest parsing, no new storage, caches, indexes or
  aggregation — repeated GETs are byte-identical and the endpoint never
  writes; it answers which decisions exist under one policy and never
  judges, scores or ranks them
- **isolation & boundaries**: a model only ever sees its own decisions;
  another model's decisions (even under the same policy) are unreachable.
  Unknown model/policy -> existing 404s; valid policy without decisions
  -> `[]`. M6 (evaluate, listing, getter, hashes, manifests), the M9
  policy registry, the M13 dashboard and the M16–M22 sample-quality /
  suite-run surfaces are byte-identical before and after

### Milestone 24 — evaluation history by checkpoint
  (`evaluations/by-checkpoint`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/evaluations/by-checkpoint/{checkpoint_id}` — answers
  "which immutable M4 evaluations measured this checkpoint?" and nothing
  else: the model's authoritative M4 listing filtered by the persisted
  `checkpoint_id` recorded in each `EvaluationRecord`
- **exact filtering**: ownership is validated through the model's M3
  checkpoint registry first (an unknown checkpoint, or a checkpoint id
  belonging to another model, is a 404 — checkpoint ids are model-scoped;
  nothing is inferred from filenames). Returned records are verbatim M4
  `EvaluationRecord` payloads (all fields incl. `loss_nats` and
  `perplexity`) in the exact M4 authoritative order
  ((created_at, eval_id) ASCENDING); current-state evaluations keep
  `checkpoint_id=null` and never appear; a valid checkpoint with no
  evaluations is a deterministic `[]`
- **implementation is a reuse, not a second engine**: the engine method
  `list_evaluations_for_checkpoint(model_id, checkpoint_id)` validates
  through the existing M3 `get_checkpoint` registry (the same ownership
  path M20 uses) and filters the authoritative M4 `list_evaluations()`
  by the persisted checkpoint identity; the facade and route are thin
  pass-throughs. No duplicate manifest parsing, no new storage, no
  caches, indexes or derived statistics — repeated GETs are
  byte-identical and the endpoint never writes; it answers which
  evaluations exist under one checkpoint and never compares or judges
  them
- **isolation & boundaries**: a model only ever sees its own
  evaluations; another model's evaluations (even mentioning a matching
  checkpoint id) are unreachable because ownership resolves through the
  model-scoped M3 registry. Unknown model/checkpoint -> existing 404s;
  valid checkpoint without evaluations -> `[]`. M4 (run, listing,
  getter, hashes, manifests), M3 checkpoint resolution, the M13
  dashboard and the M16–M23 surfaces are byte-identical before and after

### Milestone 25 — suite-run history by checkpoint
  (`suite-runs/by-checkpoint`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/suite-runs/by-checkpoint/{checkpoint_id}` — answers
  "which immutable M10 suite runs executed against this checkpoint
  state?" and nothing else: the model's authoritative M10 listing
  filtered by the persisted run state recorded in each `SuiteRunRecord`
  (`state.state_kind="checkpoint"` + `state.checkpoint_id`)
- **exact filtering**: ownership is validated through the model's M3
  checkpoint registry first (an unknown checkpoint, or a checkpoint id
  belonging to another model, is a 404 — checkpoint ids are
  model-scoped; nothing is inferred from filenames). Returned records
  are verbatim `SuiteRunRecord` payloads in the exact M10 authoritative
  order ((created_at, suite_run_id) ASCENDING); current-state runs keep
  `state.checkpoint_id=null` and never appear; a valid checkpoint with
  no suite runs is a deterministic `[]`
- **implementation is a reuse, not a second engine**: the engine method
  `list_suite_runs_for_checkpoint(model_id, checkpoint_id)` validates
  through the existing M3 `get_checkpoint` registry (the same ownership
  path M20/M24 use) and filters the authoritative M10
  `list_suite_runs()` by the persisted run state; the facade and route
  are thin pass-throughs. No duplicate manifest parsing, no new storage,
  caches or indexes — repeated GETs are byte-identical and the endpoint
  never writes
- **isolation & boundaries**: a model only ever sees its own suite
  runs; another model's runs are unreachable (ownership resolves through
  the model-scoped M3 registry). Unknown model/checkpoint -> existing
  404s; valid checkpoint without runs -> `[]`. M10 (run, listing,
  getter), M21 by-suite, M22 summary, the M13 dashboard and the
  M16–M24 surfaces are byte-identical before and after

### Milestone 26 — comparison history by checkpoint
  (`comparisons/by-checkpoint`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/comparisons/by-checkpoint/{checkpoint_id}` —
  answers "which immutable M5 comparisons involve this checkpoint
  state?" and nothing else
- **two-sided semantics**: a comparison has `state_a` and `state_b`,
  each a persisted `ComparisonSide`; a record belongs to the
  checkpoint when **either** side records
  `state_kind="checkpoint"` with the requested id. Same-checkpoint
  A=B comparisons appear **exactly once** (dedup by comparison
  identity, never by hash/timestamp/side combinations); current-state
  sides (`checkpoint_id=null`) never match — nothing is inferred from
  hashes, timestamps or filenames
- **exact filtering**: ownership is validated through the model's M3
  checkpoint registry first (an unknown checkpoint, or a checkpoint id
  belonging to another model, is a 404 — checkpoint ids are
  model-scoped). Returned records are verbatim `ComparisonRecord`
  payloads (verdict and per-side losses included) in the exact M5
  authoritative order ((created_at, comparison_id) ASCENDING); a valid
  checkpoint with no comparisons is a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine method
  `list_comparisons_for_checkpoint(model_id, checkpoint_id)` validates
  through the existing M3 `get_checkpoint` registry (the same
  ownership path M20/M24/M25 use) and filters the authoritative M5
  `list_comparisons()` by the persisted side states; the facade and
  route are thin pass-throughs registered **before** the generic
  `/comparisons/{comparison_id}` detail getter. No duplicate manifest
  parsing, no new storage, caches or indexes — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: a model only ever sees its own
  comparisons; another model's comparisons are unreachable (ownership
  resolves through the model-scoped M3 registry). Unknown
  model/checkpoint -> existing 404s; valid checkpoint without
  comparisons -> `[]`. M5 (run, listing, getter), M6 gates, M3
  checkpoint resolution, the M13 dashboard and the M16–M25 surfaces
  are byte-identical before and after

### Milestone 27 — sample history by checkpoint
  (`samples/by-checkpoint`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/samples/by-checkpoint/{checkpoint_id}` — answers
  "which immutable M15 samples were generated from this checkpoint
  state?" and nothing else
- **exact filtering**: membership comes from the persisted sample
  identity ONLY — every `SampleRecord` carries a required non-nullable
  `checkpoint_id` (M15 generation always binds ONE explicit verified
  checkpoint; there is no state_kind enum and no current-state sample),
  and a sample belongs to the request only when that persisted id
  matches. Nothing is inferred from directory names, timestamps,
  hashes, prompt text or tokenizer identity. Ownership is validated
  through the model's M3 checkpoint registry first (an unknown
  checkpoint, or a checkpoint id belonging to another model, is a 404
  — checkpoint ids are model-scoped). Returned records are verbatim
  `SampleRecord` payloads (prompt, token ids, output text, strategy,
  `result_hash` included) in the exact M15 authoritative order
  ((created_at, sample_id) ASCENDING); a valid checkpoint with no
  samples is a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine method
  `list_samples_for_checkpoint(model_id, checkpoint_id)` validates
  through the existing M3 `get_checkpoint` registry (the same ownership
  path M20/M24/M25/M26 use) and filters the authoritative M15
  `list_samples()` by the persisted checkpoint identity; the facade
  and route are thin pass-throughs registered **before** the generic
  `/samples/{sample_id}` detail getter. No duplicate manifest parsing,
  no new storage, caches or indexes — repeated GETs are byte-identical
  and the endpoint never writes; the existing M20
  `sample-quality/by-checkpoint` surface (M16 quality *measurements*)
  is a different surface and stays untouched
- **isolation & boundaries**: a model only ever sees its own samples;
  another model's samples are unreachable (ownership resolves through
  the model-scoped M3 registry). Unknown model/checkpoint -> existing
  404s; valid checkpoint without samples -> `[]`. M15 (generate,
  listing, getter), M16–M26 surfaces, the M13/M17 dashboards and the
  M3 checkpoint registry are byte-identical before and after

### Milestone 28 — evaluation history by dataset
  (`evaluations/by-dataset`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/evaluations/by-dataset/{dataset_id}` — answers
  "which immutable M4 evaluations of this model measured this
  dataset?" and nothing else
- **exact filtering**: membership comes from the persisted dataset
  identity ONLY — every `EvaluationRecord` carries top-level
  `dataset_id` and `dataset_version`, and an evaluation belongs to
  the request only when its persisted `dataset_id` matches. Nothing
  is inferred from filenames, dataset directory names, tokenizer ids,
  eval ids, checkpoint ids, hashes or timestamps. The persisted
  `dataset_version` travels VERBATIM inside every returned record:
  all versions of the dataset are returned, each exactly as persisted
  — versions are never collapsed, resolved to the latest, or rewritten
- **global datasets, model-scoped history**: the dataset is validated
  through the M2 registry first (`DatasetEngine.load_meta` — the same
  registry call M4's own run preflight uses; unknown dataset → 404).
  Datasets are GLOBAL (any model may be evaluated on any dataset), so
  model scoping comes from the model's own authoritative M4 listing —
  a model never sees another model's evaluations. Returned records
  are verbatim `EvaluationRecord` payloads (loss/perplexity/state
  identity included) in the exact M4 authoritative order
  ((created_at, eval_id) ASCENDING); a valid dataset with no
  evaluations for the model is a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_evaluations_for_dataset(model_id, dataset_id)`
  validates through the already-composed M2 dataset handle and
  filters the authoritative M4 `list_evaluations()` by the persisted
  dataset identity; the facade and route are thin pass-throughs
  registered **before** the generic `/evaluations/{eval_id}` detail
  getter (and after the M24 by-checkpoint route — a different
  grouping of the same listing, both intact). No dataset-evaluation
  index, no caches, no new storage — repeated GETs are byte-identical
  and the endpoint never writes
- **isolation & boundaries**: unknown model/dataset -> existing 404s;
  valid dataset without evaluations for the model -> `[]`. M4 (run,
  listing, getter), M24 by-checkpoint, M2 dataset registry, the
  M13/M17 dashboards and the M16–M27 surfaces are byte-identical
  before and after

### Milestone 29 — comparison history by dataset
  (`comparisons/by-dataset`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/comparisons/by-dataset/{dataset_id}` — answers
  "which immutable M5 comparisons of this model measured this
  dataset?" and nothing else
- **exact filtering (shared-probe identity)**: a comparison is valid
  only when BOTH sides measure the SAME probe, so `ComparisonRecord`
  persists exactly ONE top-level `dataset_id` + `dataset_version`
  pair — the dataset identity of the shared probe both sides measured
  by construction (M5 refuses cross-probe requests — different
  dataset/version/split/tokenizer/window/seed — with 422 before any
  artifact exists; per-side dataset identities cannot occur).
  Membership matches the persisted `dataset_id` only — never
  filenames, dataset directory names, checkpoint ids, state/result
  hashes, timestamps or eval ids
- **once per comparison + versions verbatim**: because the comparison
  (not the side) is the unit of grouping and the authoritative M5
  listing holds each record exactly once, a comparison whose two
  sides measure the requested dataset (true for EVERY matching
  record, including same-checkpoint A=B) appears EXACTLY ONCE —
  dedup by comparison identity, never by dataset id, hash, timestamp,
  side equality or path. Every version of the dataset is returned,
  each with its persisted `dataset_version` VERBATIM (never
  collapsed, resolved to the latest, aliased or rewritten)
- **global datasets, model-scoped history**: the dataset is validated
  through the M2 registry first (`DatasetEngine.load_meta` — the same
  registry call M4's run preflight and M28's by-dataset grouping use;
  unknown dataset → 404). Datasets are GLOBAL, so model scoping comes
  from the model's own authoritative M5 listing — a model never sees
  another model's comparisons. Returned records are verbatim
  `ComparisonRecord` payloads (verdict/per-side losses included) in
  the exact M5 authoritative order ((created_at, comparison_id)
  ASCENDING); a valid dataset with no comparisons for the model is a
  deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_comparisons_for_dataset(model_id, dataset_id)`
  validates through the already-composed M2 dataset handle and
  filters the authoritative M5 `list_comparisons()` by the persisted
  dataset identity; the facade and route are thin pass-throughs
  registered after the M26 by-checkpoint route and **before** the
  generic `/comparisons/{comparison_id}` detail getter. No
  dataset-comparison index, no caches, no new storage — repeated GETs
  are byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model/dataset -> existing 404s;
  valid dataset without comparisons for the model -> `[]`. M5 (run,
  listing, getter), M26 by-checkpoint, M28 by-dataset, the M2 dataset
  registry and the M16–M28 surfaces are byte-identical before and
  after

### Milestone 30 — evaluation history by tokenizer
  (`evaluations/by-tokenizer`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/evaluations/by-tokenizer/{tokenizer_id}` —
  answers "which immutable M4 evaluations of this model measured with
  this tokenizer?" and nothing else
- **exact filtering**: membership comes from the persisted tokenizer
  identity ONLY — every `EvaluationRecord` carries a top-level
  `tokenizer_id`, and an evaluation belongs to the request only when
  that persisted id matches VERBATIM. Nothing is inferred from
  filenames, eval ids, checkpoint/dataset identities, or the tokenizer
  currently registered; no latest-tokenizer substitution; tokenizer
  ids are opaque ids (M2/M4 have no tokenizer versioning and none is
  introduced)
- **global tokenizers, model-scoped history**: the tokenizer is
  validated through the existing registry first
  (`TokenizerEngine.load` — the same getter `GET /tokenizers/{id}`
  exposes; unknown tokenizer → 404). Tokenizers are GLOBAL (any model
  may be evaluated with any tokenizer), so model scoping comes from
  the model's own authoritative M4 listing — a model never sees
  another model's evaluations. Returned records are verbatim
  `EvaluationRecord` payloads (loss/perplexity/state/dataset identity
  included) in the exact M4 authoritative order ((created_at, eval_id)
  ASCENDING); a valid tokenizer with no evaluations for the model is a
  deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_evaluations_for_tokenizer(model_id, tokenizer_id)`
  validates through the already-composed tokenizer handle (M4's
  `__init__` already composes `TokenizerEngine`) and filters the
  authoritative M4 `list_evaluations()` by the persisted
  `tokenizer_id`; the facade and route are thin pass-throughs
  registered after the M28 by-dataset route and **before** the generic
  `/evaluations/{eval_id}` detail getter (M24 by-checkpoint, M28
  by-dataset and M30 by-tokenizer are different groupings of the same
  listing, all intact). No caches, no new storage — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model/tokenizer -> existing
  404s; valid tokenizer without evaluations for the model -> `[]`. M4
  (run, listing, getter), M24 by-checkpoint, M28 by-dataset, the
  tokenizer registry and the M16–M29 surfaces are byte-identical
  before and after

### Milestone 31 — comparison history by tokenizer
  (`comparisons/by-tokenizer`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/comparisons/by-tokenizer/{tokenizer_id}` —
  answers "which immutable M5 comparisons of this model measured with
  this tokenizer?" and nothing else
- **exact filtering (shared-probe identity)**: a comparison is valid
  only when BOTH sides measure the SAME probe, so `ComparisonRecord`
  persists exactly ONE top-level `tokenizer_id` — the tokenizer
  identity of the shared probe both sides measured by construction
  (M5 refuses cross-probe requests with 422; per-side tokenizer
  identities cannot occur; `ComparisonSide` carries no tokenizer
  fields). Membership matches the persisted `tokenizer_id` VERBATIM —
  never filenames, checkpoint ids, dataset identities, hashes, the
  tokenizer currently registered, or a latest-tokenizer substitution;
  no tokenizer versioning exists and none is introduced
- **once per comparison**: because the comparison (not the side) is
  the unit of grouping and the authoritative M5 listing holds each
  record exactly once, every matching comparison — including
  same-checkpoint A=B records (both sides measured the requested
  tokenizer by the shared-probe rule) — appears EXACTLY ONCE (dedup
  by comparison identity)
- **global tokenizers, model-scoped history**: the tokenizer is
  validated through the existing registry first
  (`TokenizerEngine.load` — the same getter `GET /tokenizers/{id}`
  exposes; unknown tokenizer → 404). Tokenizers are GLOBAL, so model
  scoping comes from the model's own authoritative M5 listing — a
  model never sees another model's comparisons. Returned records are
  verbatim `ComparisonRecord` payloads (verdict/per-side losses
  included) in the exact M5 authoritative order ((created_at,
  comparison_id) ASCENDING); a valid tokenizer with no comparisons
  for the model is a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_comparisons_for_tokenizer(model_id, tokenizer_id)`
  validates through the tokenizer registry handle (composed once in
  `ComparisonEngine.__init__`, following the existing style) and
  filters the authoritative M5 `list_comparisons()` by the persisted
  `tokenizer_id`; the facade and route are thin pass-throughs
  registered after the M29 by-dataset route and **before** the
  generic `/comparisons/{comparison_id}` detail getter (M26/M29/M31
  are different groupings of the same listing, all intact). No
  caches, no new storage — repeated GETs are byte-identical and the
  endpoint never writes
- **isolation & boundaries**: unknown model/tokenizer -> existing
  404s; valid tokenizer without comparisons for the model -> `[]`. M5
  (run, listing, getter), M26 by-checkpoint, M29 by-dataset, M30
  by-tokenizer, the tokenizer registry and the M16–M30 surfaces are
  byte-identical before and after

### Milestone 32 — sample history by tokenizer
  (`samples/by-tokenizer`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/samples/by-tokenizer/{tokenizer_id}` — answers
  "which immutable M15 samples of this model were generated with this
  tokenizer?" and nothing else
- **exact filtering (persisted sample tokenizer identity)**: every
  `SampleRecord` carries a required non-nullable top-level
  `tokenizer_id` (M15 generation always takes ONE explicit tokenizer
  from the request; the record also persists the matching
  `tokenizer_hash` audit field, preserved verbatim and never
  re-derived), and a sample belongs to the request only when that
  persisted id matches VERBATIM — never filenames, paths, checkpoint
  metadata, prompt text, generated tokens, hashes, the tokenizer
  currently registered, or a latest-tokenizer substitution (tokenizer
  ids are opaque; M2/M15 have no tokenizer versioning and none is
  introduced; NO M15 schema change is made for this endpoint). Each
  matching sample appears EXACTLY ONCE (the authoritative listing
  holds each record exactly once)
- **global tokenizers, model-scoped history**: the tokenizer is
  validated through the existing GLOBAL M2 registry first
  (`TokenizerEngine.load` — the same getter `GET /tokenizers/{id}`
  exposes; unknown tokenizer → 404). Model scoping comes from the
  model's own authoritative M15 listing — a model never sees another
  model's samples. Returned records are verbatim `SampleRecord`
  payloads (prompt, token ids, output text, strategy, `result_hash`
  included) in the exact M15 authoritative order ((created_at,
  sample_id) ASCENDING); a valid tokenizer with no samples for the
  model is a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_samples_for_tokenizer(model_id, tokenizer_id)`
  validates through the tokenizer registry handle ALREADY composed in
  `SamplingEngine.__init__` (M15 generation uses the same handle; no
  new composition) and filters the authoritative M15
  `list_samples()` by the persisted `tokenizer_id`; the facade and
  route are thin pass-throughs registered after the M27
  by-checkpoint route and **before** the generic
  `/samples/{sample_id}` detail getter (M27/M32 are different
  groupings of the same listing, both intact). No caches, no new
  storage — repeated GETs are byte-identical and the endpoint never
  writes
- **isolation & boundaries**: unknown model/tokenizer -> existing
  404s; valid tokenizer without samples for the model -> `[]`. M15
  (generate, listing, getter), M27 by-checkpoint, M30 evaluations
  by-tokenizer, M31 comparisons by-tokenizer, the tokenizer registry
  and the M16–M31 surfaces are byte-identical before and after

### Milestone 33 — sample-quality history by tokenizer
  (`sample-quality/by-tokenizer`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/sample-quality/by-tokenizer/{tokenizer_id}` —
  answers "which immutable M16 sample-quality measurements of this
  model measured samples generated with this tokenizer?" and nothing
  else
- **exact filtering (persisted measurement tokenizer identity)**:
  every `SampleEvaluationRecord` carries a required non-nullable
  top-level `tokenizer_id` (M16 measures an immutable M15 sample
  under its own RECORDED state; the record persists that state's
  tokenizer identity), and a measurement belongs to the request only
  when that persisted id matches VERBATIM — never filenames, paths,
  sample ids, checkpoint ids, tokenizer contents or hashes, the
  tokenizer currently registered, or a latest-tokenizer substitution
  (tokenizer ids are opaque; M2/M16 have no tokenizer versioning and
  none is introduced; `SampleEvaluationRecord` is NOT altered for
  this endpoint). Each matching record appears EXACTLY ONCE (the
  authoritative listing holds each record exactly once)
- **global tokenizers, model-scoped history**: the tokenizer is
  validated through the existing GLOBAL M2 registry first
  (`TokenizerEngine.load` — the same getter `GET /tokenizers/{id}`
  exposes; unknown tokenizer → 404). Model scoping comes from the
  model's own authoritative M16 listing — a model never sees another
  model's measurements. Returned records are verbatim
  `SampleEvaluationRecord` payloads (loss/perplexity included) in the
  exact M16 authoritative order ((created_at, evaluation_id)
  ASCENDING); a valid tokenizer with no measurements for the model is
  a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_sample_evaluations_for_tokenizer(model_id,
  tokenizer_id)` validates through the tokenizer registry handle
  ALREADY composed in `SampleQualityEngine.__init__` (M16 measurement
  uses the same handle; no new composition) and filters the
  authoritative M16 `list_sample_evaluations()` by the persisted
  `tokenizer_id`; the facade and route are thin pass-throughs
  registered after the M20 by-checkpoint route and **before** the
  generic `/sample-quality/{evaluation_id}` detail getter (M19/M20/
  M33 are different groupings of the same listing, all intact). No
  caches, no new storage — repeated GETs are byte-identical and the
  endpoint never writes
- **isolation & boundaries**: unknown model/tokenizer -> existing
  404s; valid tokenizer without measurements for the model -> `[]`.
  M16 (measure, listing, records, getter), M18 records, M19
  by-sample, M20 by-checkpoint, M30/M31/M32 by-tokenizer surfaces,
  the tokenizer registry and the M16–M32 surfaces are byte-identical
  before and after

### Milestone 34 — gate-decision history by comparison
  (`gates/decisions/by-comparison`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/gates/decisions/by-comparison/{comparison_id}` —
  answers "which immutable M6 gate decisions of this model judged
  this M5 comparison?" and nothing else
- **exact filtering (persisted decision comparison identity)**: every
  `GateDecision` carries a top-level `comparison_id` (the M5 record
  it judged — the M6 run persists it verbatim from the request), and
  a decision belongs to the request only when that persisted id
  matches VERBATIM — never filenames, gate-directory names, checkpoint
  ids, policy ids, hashes, or re-derivation from the comparison's
  current content. Legacy direct-evaluation decisions keep
  `comparison_id=None`, belong to NO by-comparison group, and stay in
  the generic M6 listing untouched. Each matching decision appears
  EXACTLY ONCE
- **model-scoped ownership first**: the comparison is validated
  through the model's OWN M5 registry first
  (`ComparisonEngine.get_comparison` — the same getter
  `GET /models/{id}/comparisons/{comparison_id}` exposes; unknown
  comparison, or one belonging to another model, → 404 — comparisons
  are model-scoped). Returned records are verbatim `GateDecision`
  payloads (verdict, decision, losses, delta, reason included) in the
  exact M6 authoritative order ((created_at, decision_id) ASCENDING);
  a valid comparison with zero decisions is a deterministic `[]` —
  never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_decisions_for_comparison(model_id, comparison_id)`
  validates through the comparison registry handle ALREADY composed
  in `GateEngine.__init__` (the M6 run uses the same handle; no new
  composition) and filters the authoritative M6 `list_decisions()` by
  the persisted `comparison_id`; the facade and route are thin
  pass-throughs registered after the M23 by-policy route and
  **before** the generic `/gates/decisions/{decision_id}` detail
  getter (M23/M34 are different groupings of the same listing, both
  intact). No caches, no new storage — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model/comparison -> existing
  404s; valid comparison without decisions -> `[]`. M6 (run, listing,
  getter), M23 by-policy, M26/M29/M31 comparison surfaces, the M5
  registry and the M16–M33 surfaces are byte-identical before and
  after

### Milestone 35 — workflow history by recipe, model-scoped
  (`workflows/by-recipe`)
- **concept**: one narrow read-only MODEL-SCOPED access path —
  `GET /models/{id}/workflows/by-recipe/{recipe_id}` — answers "which
  immutable M11 workflow runs of this model were executed from this
  registered recipe?" and nothing else. It is the model-scoped
  complement of the GLOBAL M12 cross-model
  `GET /workflows/recipes/{recipe_id}/runs` lineage surface, which
  stays unchanged
- **exact filtering (persisted run recipe identity)**: every
  `WorkflowRecord` carries a top-level `recipe_id` plus its matching
  `recipe_hash` provenance (M12 records both verbatim when a
  registered recipe is executed; the provenance is preserved exactly
  and never re-derived), and a run belongs to the request only when
  that persisted id matches VERBATIM — never filenames, paths, stage
  ids, stage contents, statuses, recipe hashes, or the recipe's
  current definition. Ad-hoc runs keep `recipe_id=None`, belong to NO
  by-recipe group, and stay in the generic M11 listing untouched (no
  `by-ad-hoc` pseudo-recipe exists). Each matching run appears EXACTLY
  ONCE
- **global recipes, model-scoped history**: the recipe is validated
  through the existing GLOBAL M12/M14 registry first
  (`RecipeEngine.get` — the same resolution
  `GET /workflows/recipes/{recipe_id}` uses; unknown recipe → 404 —
  never `[]`). Model scoping comes from the model's own authoritative
  M11 listing — a model never sees another model's runs, and a valid
  recipe never makes an unknown model valid. Returned records are
  verbatim `WorkflowRecord` payloads (status, stages, transitions,
  `result_hash` included) in the exact M11 authoritative order
  ((created_at, workflow_id) ASCENDING); a valid registered recipe
  with zero runs for the model is a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_workflows_for_recipe(model_id, recipe_id)` filters the
  authoritative M11 `list_workflows()` by the persisted `recipe_id`;
  the recipe handle is ONE new `__init__` composition line in
  `WorkflowEngine` (`RecipeEngine(storage, workflows=self)` — the
  local import avoids the circular module dependency and the
  injection keeps THIS engine the sole workflow executor); the facade
  and route are thin pass-throughs registered after the M11 listing
  route and **before** the generic `/workflows/{workflow_id}` detail
  getter. No caches, no new storage, no recipe expansion or execution
  — repeated GETs are byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model/recipe -> existing 404s;
  valid recipe without runs for the model -> `[]`. M7 (run, listing,
  getter), the M12/M14 registry + GLOBAL recipe-runs lineage, M23/M34
  gate groupings and the M16–M34 surfaces are byte-identical before
  and after

### Milestone 36 — evaluation history by split
  (`evaluations/by-split`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/evaluations/by-split/{split}` — answers "which
  immutable M4 evaluations of this model measured this dataset
  split?" and nothing else
- **exact filtering (persisted split identity)**: every
  `EvaluationRecord` carries a top-level `split` (the schema enum
  train/validation/test, persisted verbatim at run time from the
  evaluation request), and an evaluation belongs to the request only
  when that persisted value matches VERBATIM — never filenames,
  directories, timestamps, dataset names, eval ids or hashes, and
  never resolved or rewritten. Each matching evaluation appears
  EXACTLY ONCE; verbatim `EvaluationRecord` payloads
  (loss/perplexity/state/dataset identity included) in the exact M4
  authoritative order ((created_at, eval_id) ASCENDING)
- **the 422-vs-404 contract (no split registry)**: splits are a
  SCHEMA ENUM, not a registry — unlike the M24 checkpoint / M28
  dataset / M30 tokenizer axes there is nothing to 404 for an
  unknown split id. An unsupported split value (`test-set`, `TEST`,
  …) is rejected at the API boundary with **422** (FastAPI enum
  path-param validation), while an unknown model stays **404**
  exactly like every sibling grouping; the enum IS the contract
- **model-scoped history**: model validation comes from the
  authoritative M4 listing path (unknown model → 404); a model never
  sees another model's evaluations. A valid split with no
  evaluations for the model is a deterministic `[]` — never a 404
  (e.g. the `test` split currently has zero production evaluations,
  and `b5bc905326b6` has none at all)
- **implementation is a reuse, not a second engine**: the engine
  method `list_evaluations_for_split(model_id, split)` filters the
  authoritative M4 `list_evaluations()` by the persisted `split`
  (ZERO new `__init__` composition lines — no registry handle is
  needed); the facade and route are thin pass-throughs registered
  after the M30 by-tokenizer route and **before** the generic
  `/evaluations/{eval_id}` detail getter (M24/M28/M30/M36 are
  different groupings of the same listing, all intact). No caches, no
  new storage — repeated GETs are byte-identical and the endpoint
  never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported split value -> 422; valid split without evaluations
  for the model -> `[]`. M4 (run, listing, getter), M24
  by-checkpoint, M28 by-dataset, M30 by-tokenizer and the M16–M35
  surfaces are byte-identical before and after

### Milestone 37 — comparison history by split
  (`comparisons/by-split`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/comparisons/by-split/{split}` — answers "which
  immutable M5 comparisons of this model measured this dataset
  split?" and nothing else
- **exact filtering (persisted shared-probe split identity)**: every
  `ComparisonRecord` carries a top-level `split` (the schema enum
  train/validation/test — one half of the shared-probe contract: a
  comparison exists only when BOTH sides measure the SAME
  dataset/version/split/tokenizer/window/seed probe, so the split is
  a property of the comparison itself, never of a side), and a
  comparison belongs to the request only when that persisted value
  matches VERBATIM — never filenames, checkpoint ids, dataset
  identities, nested evaluation records or hashes, and never
  resolved or rewritten. Each matching comparison appears EXACTLY
  ONCE (including same-checkpoint A=B records); verbatim
  `ComparisonRecord` payloads (verdict/per-side losses included) in
  the exact M5 authoritative order ((created_at, comparison_id)
  ASCENDING)
- **the 422-vs-404 contract (no split registry)**: splits are a
  SCHEMA ENUM, not a registry — unlike the M26 checkpoint / M29
  dataset / M31 tokenizer axes there is nothing to 404 for an
  unsupported split value, so it is rejected with 422 by schema
  validation at the API boundary (before the handler, matching M36:
  even unknown-model + invalid-split is 422); unknown model with a
  VALID split -> existing 404; a valid split with zero comparisons
  for the model is a deterministic `[]` — never a 404 (all 8
  current production comparisons are `validation`; `train`/`test`
  and every split of `b5bc905326b6` are natural valid-empty cases)
- **implementation is a reuse, not a second engine**: the engine
  method `list_comparisons_for_split(model_id, split)` filters the
  authoritative M5 `list_comparisons()` by the persisted `split`
  (ZERO new `__init__` composition lines — no registry handle is
  needed); the facade and route are thin pass-throughs registered
  after the M31 by-tokenizer route and **before** the generic
  `/comparisons/{comparison_id}` detail getter (M26/M29/M31/M37 are
  different groupings of the same listing, all intact). No caches,
  no new storage — repeated GETs are byte-identical and the endpoint
  never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported split value -> 422; valid split without comparisons
  for the model -> `[]`. M5 (run, listing, getter), M26
  by-checkpoint, M29 by-dataset, M31 by-tokenizer, M36
  evaluations-by-split and the M16–M36 surfaces are byte-identical
  before and after

### Milestone 38 — evaluation history by state kind
  (`evaluations/by-state-kind`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/evaluations/by-state-kind/{state_kind}` —
  answers "which immutable M4 evaluations of this model measured
  which kind of model state?" and nothing else
- **exact filtering (persisted state-kind identity)**: every
  `EvaluationRecord` carries a top-level `state_kind` (the schema
  enum current/checkpoint, persisted verbatim at run time —
  `current` measured the model's published weights, `checkpoint`
  measured one immutable stored checkpoint), and an evaluation
  belongs to the request only when that persisted value matches
  VERBATIM — never filenames, timestamps, eval ids or hashes, and
  NEVER the `checkpoint_id` nullability (that nullability is a
  schema CONSEQUENCE of the persisted state kind, not its source).
  Each matching evaluation appears EXACTLY ONCE; verbatim
  `EvaluationRecord` payloads (loss/perplexity/state identity
  included) in the exact M4 authoritative order ((created_at,
  eval_id) ASCENDING)
- **the 422-vs-404 contract (no state-kind registry)**: state kinds
  are a SCHEMA ENUM, not a registry — unlike the M24 checkpoint /
  M28 dataset / M30 tokenizer axes there is nothing to 404 for an
  unsupported state-kind value, so it is rejected with 422 by
  schema validation at the API boundary (before the handler,
  matching M36/M37: even unknown-model + invalid-state-kind is
  422); unknown model with a VALID state kind -> existing 404; a
  valid state kind with zero evaluations for the model is a
  deterministic `[]` — never a 404 (production currently holds
  checkpoint -> 9 / current -> 7 for `4a0a871886ef`; every state
  kind of `b5bc905326b6` is a natural valid-empty case)
- **implementation is a reuse, not a second engine**: the engine
  method `list_evaluations_for_state_kind(model_id, state_kind)`
  filters the authoritative M4 `list_evaluations()` by the persisted
  `state_kind` (ZERO new `__init__` composition lines — no registry
  handle is needed); the facade and route are thin pass-throughs
  registered after the M36 by-split route and **before** the
  generic `/evaluations/{eval_id}` detail getter
  (M24/M28/M30/M36/M38 are different groupings of the same listing,
  all intact). No caches, no new storage — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported state-kind value -> 422; valid state kind without
  evaluations for the model -> `[]`. M4 (run, listing, getter), M24
  by-checkpoint, M28 by-dataset, M30 by-tokenizer, M36 by-split,
  M37 comparisons-by-split and the M16–M37 surfaces are
  byte-identical before and after

### Milestone 39 — comparison history by verdict
  (`comparisons/by-verdict`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/comparisons/by-verdict/{verdict}` — answers
  "which immutable M5 comparisons of this model produced this
  verdict?" and nothing else
- **exact filtering (persisted verdict identity)**: every
  `ComparisonRecord` carries a top-level `verdict` (the schema enum
  improved/regressed/unchanged — the immutable loss-only judgment
  persisted at run time by the M5 comparison flow: it describes
  measured loss on ONE probe, never a universal quality judgment),
  and a comparison belongs to the request only when that persisted
  value matches VERBATIM — NEVER recalculated from loss deltas,
  per-side losses or tolerances, never resolved or rewritten, no
  comparison executed and no evaluation rerun. Each matching
  comparison appears EXACTLY ONCE; verbatim `ComparisonRecord`
  payloads (verdict/per-side losses included) in the exact M5
  authoritative order ((created_at, comparison_id) ASCENDING)
- **the 422-vs-404 contract (no verdict registry)**: verdicts are a
  SCHEMA ENUM, not a registry — unlike the M26 checkpoint / M29
  dataset / M31 tokenizer axes there is nothing to 404 for an
  unsupported verdict value, so it is rejected with 422 by schema
  validation at the API boundary (before the handler, matching
  M36/M37/M38: even unknown-model + invalid-verdict is 422); unknown
  model with a VALID verdict -> existing 404; a valid verdict with
  zero comparisons for the model is a deterministic `[]` — never a
  404 (production currently holds improved -> 3 / unchanged -> 3 /
  regressed -> 2 for `4a0a871886ef` — all three groups non-empty;
  every verdict of `b5bc905326b6` is a natural valid-empty case)
- **implementation is a reuse, not a second engine**: the engine
  method `list_comparisons_for_verdict(model_id, verdict)` filters
  the authoritative M5 `list_comparisons()` by the persisted
  `verdict` (ZERO new `__init__` composition lines — no registry
  handle is needed, and the execution engine is never invoked); the
  facade and route are thin pass-throughs registered after the M37
  by-split route and **before** the generic
  `/comparisons/{comparison_id}` detail getter
  (M26/M29/M31/M37/M39 are different groupings of the same listing,
  all intact). No caches, no new storage — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported verdict value -> 422; valid verdict without
  comparisons for the model -> `[]`. M5 (run, listing, getter), M26
  by-checkpoint, M29 by-dataset, M31 by-tokenizer, M37 by-split, M38
  evaluations-by-state-kind and the M16–M38 surfaces are
  byte-identical before and after

### Milestone 40 — sample history by strategy
  (`samples/by-strategy`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/samples/by-strategy/{strategy}` — answers "which
  immutable M15 samples of this model were generated with this
  decoding strategy?" and nothing else
- **exact filtering (persisted strategy identity)**: every
  `SampleRecord` carries a top-level `strategy` (the schema enum
  greedy/temperature, persisted verbatim at generation time from the
  explicit request — greedy is deterministic argmax with no RNG,
  temperature draws from the temperature-scaled distribution using
  ONE deterministic RNG stream seeded by the request's explicit
  seed), and a sample belongs to the request only when that persisted
  value matches VERBATIM — never sample ids, prompt text, generated
  token ids, temperature values, seed presence, filenames or manifest
  paths, and NEVER recalculated from temperature/seed/other fields;
  no sample is regenerated. Each matching sample appears EXACTLY
  ONCE; verbatim `SampleRecord` payloads (prompt, token ids, output
  text, temperature, result_hash included) in the exact M15
  authoritative order ((created_at, sample_id) ASCENDING)
- **the 422-vs-404 contract (no strategy registry)**: strategies are
  a SCHEMA ENUM, not a registry — unlike the M27 checkpoint / M32
  tokenizer axes there is nothing to 404 for an unsupported strategy
  value, so it is rejected with 422 by schema validation at the API
  boundary (before the handler, matching M36–M39: even
  unknown-model + invalid-strategy is 422); unknown model with a
  VALID strategy -> existing 404; a valid strategy with zero samples
  for the model is a deterministic `[]` — never a 404 (production
  currently holds greedy -> 2 / temperature -> 2 for `4a0a871886ef`
  — both groups non-empty; every strategy of `b5bc905326b6` is a
  natural valid-empty case)
- **implementation is a reuse, not a second engine**: the engine
  method `list_samples_for_strategy(model_id, strategy)` filters the
  authoritative M15 `list_samples()` by the persisted `strategy`
  (ZERO new `__init__` composition lines — no registry handle is
  needed, and generation is never invoked); the facade and route are
  thin pass-throughs registered after the M32 by-tokenizer route and
  **before** the generic `/samples/{sample_id}` detail getter
  (M27/M32/M40 are different groupings of the same listing, all
  intact). No caches, no new storage — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported strategy value -> 422; valid strategy without samples
  for the model -> `[]`. M15 (generate, listing, getter), M27
  by-checkpoint, M32 by-tokenizer, M38 evaluations-by-state-kind,
  M39 comparisons-by-verdict and the M16–M39 surfaces are
  byte-identical before and after

### Milestone 41 — gate-decision history by decision
  (`gates/decisions/by-decision`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/gates/decisions/by-decision/{decision}` —
  answers "which immutable gate decisions of this model produced
  this decision result?" and nothing else
- **exact filtering (persisted decision identity)**: every
  `GateDecision` carries a top-level `decision` (the schema enum
  passed/failed — the immutable policy verdict persisted at run time
  by the M6 gate flow; the policy semantics may legitimately DIFFER
  from the loss-only comparison verdict, e.g. an improved candidate
  still fails a `minimum_loss` ceiling), and a decision belongs to
  the request only when that persisted value matches VERBATIM —
  never recalculated from loss deltas, tolerances, policy
  thresholds, gate configuration or comparison results, never
  resolved or rewritten, no gate re-evaluated. Each matching
  decision appears EXACTLY ONCE; verbatim `GateDecision` payloads
  (verdict/evidence chain/rollback suggestion included) in the exact
  M6 authoritative order ((created_at, decision_id) ASCENDING)
- **the 422-vs-404 contract (no decision registry)**: decision
  results are a SCHEMA ENUM, not a registry — unlike the M23 policy /
  M34 comparison axes there is nothing to 404 for an unsupported
  decision value, so it is rejected with 422 by schema validation at
  the API boundary (before the handler, matching M36–M40: even
  unknown-model + invalid-decision is 422); unknown model with a
  VALID decision -> existing 404; a valid decision with zero gate
  decisions for the model is a deterministic `[]` — never a 404
  (production currently holds passed -> 7 / failed -> 4 for
  `4a0a871886ef` — both groups non-empty; every decision result of
  `b5bc905326b6` is a natural valid-empty case)
- **implementation is a reuse, not a second engine**: the engine
  method `list_decisions_for_decision(model_id, decision)` filters
  the authoritative M6 `list_decisions()` by the persisted
  `decision` (ZERO new `__init__` composition lines — no registry
  handle is needed, and the gate engine's run path is never
  invoked); the facade and route are thin pass-throughs registered
  after the M34 by-comparison route and **before** the generic
  `/gates/decisions/{decision_id}` detail getter (M23/M34/M41 are
  different groupings of the same listing, all intact). No caches,
  no new storage — repeated GETs are byte-identical and the
  endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported decision value -> 422; valid decision without gate
  decisions for the model -> `[]`. M6 (run, listing, getter), M23
  by-policy, M34 by-comparison, M40 samples-by-strategy and the
  M16–M40 surfaces are byte-identical before and after

### Milestone 42 — workflow history by status
  (`workflows/by-status`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/workflows/by-status/{status}` — answers "which
  immutable workflow runs of this model ended with this terminal
  status?" and nothing else
- **exact filtering (persisted status identity)**: every
  `WorkflowRecord` carries a top-level `status` (the schema enum
  completed/failed/stopped — the terminal state persisted at run end
  by the M7 orchestration: completed = plan executed through its
  last stage; failed = a stage raised a missing/corrupt/invalid
  input; stopped = a gate decision failed and no on_fail branch was
  declared; a mid-flight 'running' state is deliberately never
  modelled), and a run belongs to the request only when that
  persisted value matches VERBATIM — never inferred from stage
  results, failed stage ids, workflow timestamps, artifact existence
  or recipe information, never recalculated or rewritten, nothing
  re-executed. Each matching run appears EXACTLY ONCE; verbatim
  `WorkflowRecord` payloads (stages, transitions, terminal_reason,
  result_hash and recipe provenance included) in the exact M11
  authoritative order ((created_at, workflow_id) ASCENDING)
- **the 422-vs-404 contract (no status registry)**: statuses are a
  SCHEMA ENUM, not a registry — unlike the M35 recipe axis there is
  nothing to 404 for an unsupported status value, so it is rejected
  with 422 by schema validation at the API boundary (before the
  handler, matching M36–M41: even unknown-model + invalid-status is
  422); unknown model with a VALID status -> existing 404; a valid
  status with zero matching runs is a deterministic `[]` — never a
  404 (production currently holds completed -> 9 / failed -> 3 /
  stopped -> 1 for `4a0a871886ef` — all three groups non-empty;
  every status of `b5bc905326b6` is a natural valid-empty case)
- **implementation is a reuse, not a second engine**: the engine
  method `list_workflows_for_status(model_id, status)` filters the
  authoritative M11 `list_workflows()` by the persisted `status`
  (ZERO new composition lines — no registry handle is needed, and
  the orchestration run path is never invoked); the facade and route
  are thin pass-throughs registered after the M35 by-recipe route
  and **before** the generic `/workflows/{workflow_id}` detail
  getter (M35/M42 are different groupings of the same listing, both
  intact). No caches, no new storage — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported status value -> 422; valid status without matching
  runs -> `[]`. M7 (run, listing, getter), M12/M14 recipes + the
  GLOBAL recipe-runs lineage, M35 by-recipe, M41 gate
  decisions-by-decision and the M16–M41 surfaces are byte-identical
  before and after

### Milestone 43 — gate-decision history by verdict
  (`gates/decisions/by-verdict`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/gates/decisions/by-verdict/{verdict}` — answers
  "which immutable gate decisions of this model recorded this
  loss-only comparison verdict?" and nothing else. Completes the
  gate-decision history family alongside M23 (by-policy), M34
  (by-comparison), M41 (by-decision)
- **exact filtering (persisted verdict identity)**: every
  `GateDecision` carries a top-level `verdict:
  Optional[ComparisonVerdict]` (improved / regressed / unchanged —
  the LOSS-ONLY comparison verdict recorded verbatim by the M6 run,
  deliberately DISTINCT from the M41 policy decision: an improved
  candidate can still fail a `minimum_loss` ceiling), and a decision
  belongs to the request only when that persisted value matches
  VERBATIM — never recalculated from losses, deltas, tolerances,
  policies or comparison records, never resolved or rewritten, no
  gate re-evaluated. Each matching decision appears EXACTLY ONCE;
  verbatim `GateDecision` payloads (decision/evidence chain/rollback
  suggestion included) in the exact M6 authoritative order
  ((created_at, decision_id) ASCENDING)
- **the None contract (threshold-only decisions)**: the field is
  OPTIONAL — threshold-only gates (`baseline_type="minimum_loss"`)
  judge NO comparison and keep `verdict=None`; None is not an enum
  value and NEVER matches any request; there is deliberately NO
  route representing None; null-verdict decisions belong to NO
  by-verdict group and stay listed in the generic M6 history
  untouched (production currently holds improved -> 4 / regressed ->
  3 / unchanged -> 2 for `4a0a871886ef` — all three groups non-empty
  — plus 2 threshold-only null-verdict decisions outside every
  group; every verdict of `b5bc905326b6` is a natural valid-empty
  case)
- **the 422-vs-404 contract (no verdict registry)**: verdicts are a
  SCHEMA ENUM, not a registry — exactly like M41, an unsupported
  verdict value is rejected with 422 by schema validation at the API
  boundary (before the handler, matching M36–M42: even
  unknown-model + invalid-verdict is 422); unknown model with a
  VALID verdict -> existing 404; a valid verdict with zero matching
  decisions is a deterministic `[]` — never a 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_decisions_for_verdict(model_id, verdict)` filters the
  authoritative M6 `list_decisions()` by the persisted `verdict`
  (ZERO new composition lines — no registry handle is needed, and
  the gate run path is never invoked); the facade and route are thin
  pass-throughs registered after the M41 by-decision route and
  **before** the generic `/gates/decisions/{decision_id}` detail
  getter (M23/M34/M41/M43 are different groupings of the same
  listing, all intact). No caches, no new storage — repeated GETs
  are byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported verdict value -> 422; valid verdict without matching
  decisions -> `[]`; null-verdict decisions in no group. M6 (run,
  listing, getter), M23 by-policy, M34 by-comparison, M41
  by-decision, M42 workflows-by-status and the M16–M42 surfaces are
  byte-identical before and after

### Milestone 44 — comparison history by state kind
  (`comparisons/by-state-kind`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/comparisons/by-state-kind/{state_kind}` — answers
  "which immutable comparisons of this model involve the requested
  kind of model state on either side?" and nothing else. The
  enum-contract sibling of M38 (evaluation history by state kind),
  with M26's either-side semantics because comparison records persist
  TWO states
- **exact filtering (persisted side state kinds, EITHER side)**:
  every `ComparisonRecord` persists both sides (`state_a` /
  `state_b`, each with `state_kind: EvalStateKind` — current = the
  model's published weights, checkpoint = an immutable stored
  checkpoint), and a comparison belongs to the request only when
  EITHER side's persisted value matches VERBATIM — never inferred
  from checkpoint ids alone, state hashes, losses or evaluation
  results, never recalculated or rewritten, no comparison
  re-executed. Because the listing holds each record exactly once, a
  comparison matching on BOTH sides (A = B = checkpoint) appears
  EXACTLY ONCE. Verbatim `ComparisonRecord` payloads (both sides,
  losses, verdict verbatim) in the exact M5 authoritative order
  ((created_at, comparison_id) ASCENDING)
- **the 422-vs-404 contract (no state-kind registry)**: state kinds
  are a SCHEMA ENUM, not a registry — exactly like M38, an
  unsupported state-kind value is rejected with 422 by schema
  validation at the API boundary (before the handler, matching
  M36–M43: even unknown-model + invalid-state-kind is 422); unknown
  model with a VALID state kind -> existing 404; a valid state kind
  with zero matching comparisons is a deterministic `[]` — never a
  404 (production currently holds checkpoint -> 8 / current -> 2 for
  `4a0a871886ef` — both groups non-empty; every state kind of
  `b5bc905326b6` is a natural valid-empty case)
- **implementation is a reuse, not a second engine**: the engine
  method `list_comparisons_for_state_kind(model_id, state_kind)`
  filters the authoritative M5 `list_comparisons()` by the persisted
  side state kinds (the same either-side `involves()` shape as M26;
  ZERO new composition lines); the facade and route are thin
  pass-throughs registered after the M39 by-verdict route and
  **before** the generic `/comparisons/{comparison_id}` detail getter
  (M26/M29/M31/M37/M39/M44 are different groupings of the same
  listing, all intact). No caches, no new storage — repeated GETs are
  byte-identical and the endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported state-kind value -> 422; valid state kind without
  matching comparisons -> `[]`. M5 (run, listing, getter), M26
  by-checkpoint, M29 by-dataset, M31 by-tokenizer, M37 by-split, M39
  by-verdict, M38 evaluations-by-state-kind and the M16–M43 surfaces
  are byte-identical before and after

### Milestone 45 — gate-decision history by baseline type
  (`gates/decisions/by-baseline-type`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/gates/decisions/by-baseline-type/{baseline_type}`
  — answers "which immutable gate decisions of this model compared
  their candidate against this kind of baseline?" and nothing else.
  The FIRST nested-field grouping: membership comes from the
  persisted `policy.baseline_type` embedded verbatim in each decision
- **exact filtering (persisted NESTED policy field)**: every
  `GateDecision` embeds its `GatePolicy` VERBATIM (the M6 run
  persists the policy exactly as evaluated), and every policy
  carries a REQUIRED `baseline_type: GateBaselineType` (the schema
  enum: checkpoint = a specific immutable checkpoint; current = the
  model's published current weights; evaluation_result_hash = a past
  immutable evaluation; minimum_loss = an absolute loss threshold
  only, no state), and a decision belongs to the request only when
  that persisted nested value matches VERBATIM — never derived from
  checkpoint id presence, comparison or evaluation references, policy
  contents outside baseline_type, the gate result, the verdict, loss
  deltas or timestamps, never resolved or rewritten, no gate
  re-evaluated. Each matching decision appears EXACTLY ONCE; verbatim
  `GateDecision` payloads (embedded policy, verdict, evidence chain
  included) in the exact M6 authoritative order ((created_at,
  decision_id) ASCENDING)
- **true disjoint partition, complete enum contract**: because the
  field is REQUIRED, every decision falls in exactly ONE group — no
  None case (unlike M43's optional verdict). The enum exposes the
  COMPLETE contract including `evaluation_result_hash`: a valid value
  with no matching decisions is a deterministic `[]` — never 404,
  and no route is omitted for empty categories (production currently
  holds checkpoint -> 7 / current -> 2 / minimum_loss -> 2 /
  evaluation_result_hash -> 0 for `4a0a871886ef` — three non-empty
  groups; every baseline type of `b5bc905326b6` is a natural
  valid-empty case)
- **the 422-vs-404 contract (no baseline-type registry)**: baseline
  types are a SCHEMA ENUM, not a registry — exactly like M41/M43, an
  unsupported baseline-type value is rejected with 422 by schema
  validation at the API boundary (before the handler, matching
  M36–M44: even unknown-model + invalid-baseline-type is 422);
  unknown model with a VALID baseline type -> existing 404
- **implementation is a reuse, not a second engine**: the engine
  method `list_decisions_for_baseline_type(model_id, baseline_type)`
  filters the authoritative M6 `list_decisions()` by the persisted
  nested `policy.baseline_type` (ZERO new composition lines — no
  registry handle is needed, and the gate run path is never
  invoked); the facade and route are thin pass-throughs registered
  after the M43 by-verdict route and **before** the generic
  `/gates/decisions/{decision_id}` detail getter (M23/M34/M41/M43/M45
  are different groupings of the same listing, all intact). No
  caches, no new storage — repeated GETs are byte-identical and the
  endpoint never writes
- **isolation & boundaries**: unknown model -> existing 404;
  unsupported baseline-type value -> 422; valid baseline type
  without matching decisions -> `[]`. M6 (run, listing, getter), M23
  by-policy, M34 by-comparison, M41 by-decision, M43 by-verdict, M44
  comparisons-by-state-kind and the M16–M44 surfaces are
  byte-identical before and after

### Milestone 46 — checkpoint history by training run
  (`checkpoints/by-run`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/checkpoints/by-run/{run_id}` — answers "which
  immutable checkpoints did ONE training run of this model produce?"
  and nothing else. Membership comes from each checkpoint's own
  persisted `run_id` (REQUIRED identity field, "lineage within/across
  runs"), and run ownership is validated against the model's OWN
  manifest `training_provenance` — no separate training-run registry
  is introduced; the model manifest IS the registry
- **exact filtering (persisted checkpoint field)**: the listing is
  the authoritative M3 checkpoint store filtered VERBATIM by
  `c.run_id == run_id` — membership NEVER derived from checkpoint
  directories, steps, epochs, timestamps, losses, parent
  relationships or any provenance field other than the checkpoint's
  own `run_id`; the provenance list is used ONLY for existence
  validation (the 404), never for building the response. Each
  checkpoint appears EXACTLY ONCE; verbatim `CheckpointRecord`
  payloads in the exact M3 authoritative order ((step, created_at)
  ascending, preserved — not replaced by another ordering)
- **validation & boundaries**: unknown model -> existing 404;
  unknown run on a valid model -> 404 (the run must appear in that
  model's `training_provenance`); a run id belonging to another
  model -> 404 (model-scoped ownership — the other model's manifest
  does not register it); a registered run with zero checkpoints
  (constructible through the real engine when configured steps
  finish before checkpoint creation) -> `200 []`. `run_id` is a
  persisted identifier, not an enum — no artificial validation rules
  beyond the provenance registry. M3 (training run, listing, detail,
  rollback), M24–M45 history surfaces, dashboards and registries are
  byte-identical before and after. No caches, no new storage —
  repeated GETs are byte-identical and the endpoint never writes

### Milestone 47 — evaluation history by truncation
  (`evaluations/by-truncated`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/evaluations/by-truncated/{truncated}` —
  answers "which immutable evaluations of this model were stopped
  early by the `max_eval_tokens` cap?" and nothing else. Membership
  comes from the persisted REQUIRED boolean `truncated` the engine
  records on every `EvaluationRecord` at run time
- **exact filtering (persisted boolean)**: True = `max_eval_tokens`
  stopped the evaluation before the split ended; False = the
  configured/permitted evaluation stream was consumed without the
  cap cutting it short. The listing is the authoritative M4
  evaluation history filtered VERBATIM by that persisted boolean —
  NEVER recalculated, NEVER derived from `records_covered`, token
  counts, split length, the evaluation configuration, timestamps,
  durations, state kinds or any other field. The boolean carries NO
  quality judgment — it is engine metadata about how far the
  evaluation stream was consumed, nothing more; `truncated=True` is
  not intrinsically better or worse. Each evaluation appears EXACTLY
  ONCE; verbatim `EvaluationRecord` payloads in the exact M4
  authoritative order ((created_at, eval_id) ascending, preserved)
- **boundaries**: the closed two-value contract makes the two groups
  a TRUE disjoint partition of the listing with no None case;
  non-boolean spellings -> 422 (schema-level validation at the API
  boundary, pre-handler — booleans are never silently reinterpreted
  from arbitrary strings); unknown model + valid boolean -> 404; a
  model with no evaluations of one status -> `200 []`. M4 (run,
  listing, getter), M24/M28/M30/M36/M38 evaluation groupings, M46
  checkpoint-by-run and the other M22–M45 surfaces are byte-identical
  before and after. No caches, no re-runs, no automatic coverage
  enforcement, no new storage — repeated GETs are byte-identical and
  the endpoint never writes

### Milestone 48 — evaluation history by seed
  (`evaluations/by-seed`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/evaluations/by-seed/{seed}` — answers "which
  immutable evaluations of this model ran with this effective
  seed?" and nothing else. Membership comes from the REQUIRED
  integer `seed` persisted on every `EvaluationRecord` at run time
  (the effective seed used, default derived from the config)
- **exact filtering (persisted integer)**: the listing is the
  authoritative M4 evaluation history filtered VERBATIM by the
  record's own persisted value — NEVER recalculated, NEVER
  normalized, NEVER derived from the embedded config dict, request
  parameters, dataset identity, splits, tokenizers, state kinds,
  losses, ids, timestamps or any other field. The seed is
  bookkeeping identity, not a quality metric — no seed is better
  than another, and this endpoint is not a model-quality measure.
  Each evaluation appears EXACTLY ONCE; verbatim `EvaluationRecord`
  payloads in the exact M4 authoritative order ((created_at,
  eval_id) ascending, preserved)
- **boundaries**: the seed is an OPEN integer value axis — no
  registry, no enum, no artificial range constraint: any integer is
  type-valid, an unmatched seed on a valid model -> `200 []`
  (natural valid-empty), a non-integer spelling -> 422
  (schema-level validation at the API boundary, pre-handler —
  integers are never silently reinterpreted), unknown model + valid
  integer -> 404; the per-seed groups form a TRUE disjoint
  partition of the listing with no None case. M4 (run, listing,
  getter), M24/M28/M30/M36/M38/M47 evaluation groupings, M46
  checkpoint-by-run and the other M22–M45 surfaces are byte-identical
  before and after. No seed registry/index/cache, no aggregation, no
  new storage — repeated GETs are byte-identical and the endpoint
  never writes

### Milestone 49 — comparison history by seed
  (`comparisons/by-seed`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/comparisons/by-seed/{seed}` — answers "which
  immutable A/B comparisons of this model ran with this effective
  seed?" and nothing else. Membership comes from the REQUIRED
  integer `seed` persisted on every `ComparisonRecord` at run time
  (the seed of the identical-probe measurement)
- **exact filtering (persisted integer)**: the listing is the
  authoritative M5 comparison history filtered VERBATIM by the
  record's own persisted value — NEVER recalculated, NEVER
  normalized, NEVER derived from the comparison configuration,
  either side's evaluation, state payloads, verdicts, loss deltas,
  ids, timestamps or any other field. The seed is bookkeeping
  identity, not a quality metric — no seed produces better
  comparisons. Each comparison appears EXACTLY ONCE; verbatim
  `ComparisonRecord` payloads (both sides, losses, verdict included)
  in the exact M5 authoritative order ((created_at, comparison_id)
  ascending, preserved)
- **boundaries**: the seed is an OPEN integer value axis — no
  registry, no enum, no artificial range constraint: any integer is
  type-valid, an unmatched seed on a valid model -> `200 []`
  (natural valid-empty), a non-integer spelling -> 422
  (schema-level validation at the API boundary, pre-handler —
  integers are never silently reinterpreted), unknown model + valid
  integer -> 404; the per-seed groups form a TRUE disjoint
  partition of the listing with no None case. M5 (run, listing,
  getter), M26/M29/M31/M37/M39/M44 comparison groupings and the
  other M4–M48 surfaces are byte-identical before and after. No seed
  registry/index/cache, no aggregation, no new storage — repeated
  GETs are byte-identical and the endpoint never writes

### Milestone 50 — suite-run history by reused count
  (`suite-runs/by-reused`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/suite-runs/by-reused/{reused_count}` — answers
  "which immutable suite runs of this model satisfied this many
  probes with pre-existing evidence?" and nothing else. Membership
  comes from the REQUIRED integer `reused_count` persisted on every
  `SuiteRunRecord` at run time. This closes the simple grouping
  ladder: every persisted listing field with a genuine multi-group
  identity is now exposed
- **exact filtering (persisted integer)**: the listing is the
  authoritative M10 suite-run history filtered VERBATIM by the
  record's own persisted value — NEVER recalculated, NEVER derived
  from probe outcomes, completed_count, failed_count, probe_count,
  suite size, status, timestamps, artifact ids, evaluation or
  comparison records, configuration or any other field. The count is
  EXECUTION BOOKKEEPING, never a score, a ranking or a quality
  signal — reused evidence is not better or worse, it is how the
  immutable evaluation cache satisfied the suite. Each run appears
  EXACTLY ONCE; verbatim `SuiteRunRecord` payloads in the exact M10
  authoritative order ((created_at, suite_run_id) ascending,
  preserved)
- **boundaries**: the count is an OPEN integer value axis — no
  registry, no enum, no artificial range constraint: any integer is
  type-valid, an unmatched count on a valid model -> `200 []`
  (natural valid-empty), a non-integer spelling -> 422
  (schema-level validation at the API boundary, pre-handler —
  integers are never silently reinterpreted), unknown model + valid
  integer -> 404; the per-count groups form a TRUE disjoint
  partition of the listing with no None case. M10 (run, listing,
  getter, summary), M21 by-suite, M22 summary, M25 by-checkpoint
  and the other M4–M49 surfaces are byte-identical before and after.
  No reused-count registry/index/cache, no aggregation, no new
  storage — repeated GETs are byte-identical and the endpoint never
  writes

### Milestone 51 — recipe resolution preflight
  (`workflows/recipes/.../plan`)
- **concept**: one narrow read-only access path —
  `GET /models/{id}/workflows/recipes/{recipe_id}/plan` — answers
  "what EXACTLY would this registered recipe execute against this
  model?" and nothing else. The architectural completion of the M12
  synchronous execution surface: resolution WITHOUT execution, through
  the SAME `RecipeEngine` resolution code `run()` uses (recipe lookup
  → model validation → M14 deterministic expansion → `WorkflowPlan`
  construction with the FULL M7 validation incl. embedded-config
  model agreement) — one resolution system, never a second executor
- **what it returns**: the expanded, model-bound, fully validated
  `WorkflowPlan` the `WorkflowEngine` WOULD execute (its `plan_hash`
  predicts the executed run's `plan_hash`), plus recipe provenance
  (`recipe_id` + `recipe_hash`) and the additive M14 `composition`
  trace (null for plain recipes). Composite recipes resolve to their
  spliced, id-qualified expanded stage list — visible BEFORE any
  execution, where previously it only materialized inside a run. A
  computed view (`WorkflowRecipeResolution`): NEVER persisted, never
  written — a resolve leaves zero new files
- **boundaries**: error semantics identical to a run request —
  unknown recipe/model -> 404 with nothing persisted; binding/schema
  conflicts (a bound model contradicting a model pinned inside the
  recipe) -> 422. Execution itself stays exactly where it was: the
  synchronous `POST /workflows/recipes/{recipe_id}/runs` through the
  sole `WorkflowEngine` — no scheduling, no background workers, no
  queues, no second record type. M7 (run, records), M12/M14
  (registration, expansion, lineage), M10/M11 (suite stages,
  evidence reuse) and the M18–M50 history surfaces are byte-identical
  before and after

### Milestone 59 — best-checkpoint improvement HISTORY
  (`GET /models/{model_id}/checkpoints/best/history`)
- **concept**: a read-only, model-scoped view of HOW the best-checkpoint
  selection evolved — the chronological sequence of checkpoints that
  BECAME the M52-selected best as the model's checkpoint registry grew.
  Only winners appear (non-winning checkpoints are excluded); each entry
  carries the authoritative persisted fields (checkpoint id, producing
  `run_id`, `step`, `created_at`, `validation_loss`, `perplexity`) plus
  `delta_loss_nats` — the improvement vs the PREVIOUS best under the
  established sign convention (current - previous; negative means
  improvement; None on the first entry; 0.0 only when the canonical
  tie-break actually moved the selection to an equally-good checkpoint)
- **one selection semantics**: the timeline replays the ONE shared M52
  winner rule (minimum persisted `validation_loss`; ties by the
  canonical (step, created_at) ASCENDING order — first among equals;
  non-finite values never candidates) over the authoritative M3 listing
  in chronological (created_at, checkpoint_id) order. No second
  selector, no second comparator; the FINAL entry is always the live
  `GET .../checkpoints/best` answer; losses are monotonically
  non-increasing
- **computed live, zero storage**: derived from persisted manifests on
  every call — never a record, never a cache, no best pointer, no
  history database. A new checkpoint is reflected immediately;
  repeated calls over unchanged storage are byte-identical. Unreadable
  manifests are skipped by the listing (the established corruption
  semantics); unknown model -> 404; a valid model with no selectable
  checkpoints -> an EMPTY history (consistent with the collection
  endpoints)
- **purpose**: observability of the improvement trajectory (e.g. the
  successive advances of an M58 repeated run) — it decides NOTHING: no
  automatic stopping, no convergence detection, no repetition selection
- OpenAPI: path count 84 -> 85 (exactly this one new read-only route)

### Milestone 61 — explicit VERIFIED checkpoint retention (`DELETE .../checkpoints/{id}`)
- **concept**: the storage counterpart of the completed improvement
  loop. Repeated training accumulates immutable content-addressed
  checkpoints with no safe way to reclaim any of them (only whole-model
  delete existed). M61 adds ONE explicit, request-only operation:
  `DELETE /models/{id}/checkpoints/{checkpoint_id}` removes exactly
  ONE checkpoint — never automatically, no background GC, no keep-N,
  no age policy, no bulk mode
- **the guard (fixed order)**: (1) scope through the authoritative
  listing — unknown model/checkpoint, or a checkpoint whose manifest
  is unreadable (the listing skips it), -> 404, nothing deleted;
  (2) integrity verification through the EXISTING M3 verifier
  (corrupt weights, missing weights -> 409: deletion never bypasses
  integrity validation); (3) a LIVE reference-safety analysis over the
  authoritative listings; (4) ONE atomic removal
- **blocking references** (computed live via the SAME selectors and
  the SAME read-only `..._for_checkpoint` filters the listing routes
  use — no stored index, no second registry): the live M52 best; the
  published `latest_checkpoint`; the manifest's stored
  `best_checkpoint` weights reference; and every persisted checkpoint
  id in the immutable EVIDENCE record families — workflow plans and
  stage artifacts (M53/M55/M56/M57/M60 pins included), M4 evaluations,
  M5 comparisons (either side), M6 gate decisions
  (candidate/baseline/suggestion), M10 suite runs, M15 samples and M16
  sample-quality measurements. Protected -> 409 with a deterministic
  ordered blocker list (`best`, `published`, `manifest_reference`,
  `workflow_reference`, `evaluation_reference`, `comparison_reference`,
  `gate_reference`, `suite_run_reference`, `sample_reference`,
  `sample_quality_reference`)
- **deliberately NOT blocking (inspected + documented)**: pure lineage
  metadata — a surviving checkpoint's `parent_checkpoint_id` and run
  provenance (`initial`/`final`/`rolled_back_to`) — and the computed
  views (M59 history, M8 dashboard). Lineage is informational history
  the architecture explicitly tolerates losing (the dashboard emits a
  diagnostic for a missing reference and keeps valid artifacts
  visible); nothing resolves lineage to state; and blocking it would
  protect EVERY checkpoint ever created, because M3 chains checkpoints
  within each run (each checkpoint's parent is its predecessor, the
  run's final is the provenance final) — retention would be dead code.
  Appearing in M59 history protects nothing: after a deletion the
  history recomputes naturally over the survivors
- **atomicity**: the smallest reusable storage primitive
  (`atomic_delete_dir`: one `os.rename` to a hidden `.tmp-delete-*`
  sibling the listings already skip, then rmtree — the mirror of the
  atomic writers). No observer can ever see a half-deleted checkpoint;
  a crash can only leave a hidden residue, never a valid-looking
  partial one. The checkpoint registry IS the directory listing, so
  removal needs no manifest/pointer updates — the model manifest,
  weights.pt and every record family stay untouched
- **result**: a small deterministic response (`files_removed`,
  `bytes_reclaimed`); unrelated checkpoints stay byte-identical; the
  M52 best, the published state, M46 by-run listings and M59 history
  stay coherent over the surviving registry
- OpenAPI: NO new path — the DELETE rides the existing checkpoint
  detail route; path count stays 85 (new operation +
  `CheckpointDeletionResult` schema)

### Milestone 60 — declarative BEST-PUBLICATION stage (`publish`)
- **concept**: the loop's missing ACCEPT step. A workflow/recipe PUBLISH
  stage makes ONE immutable checkpoint the model's published/live state:
  `publish_from_best` declares `PUBLISH(best)` — the canonical terminal
  acceptance action `TRAIN → EVALUATE(best) → GATE(best) → PUBLISH(best)`.
  The stage executes the EXISTING M3 verified rollback machinery (the
  same `TrainingEngine.rollback` behind `POST /models/{id}/rollback`:
  verify the checkpoint's integrity → atomically restore its weights as
  the current weights → update `latest_checkpoint`). A thin workflow
  adapter — never a second publication mechanism, never a second
  weights copy, never a new latest-pointer system, never a mutation of
  the immutable source checkpoint
- **one resolution path**: `publish_from_best` is resolved by the SAME
  M53 resolver as M53 state refs, M55 best-resume, M56 best-evaluation
  and M57 best-baselines — ONE M52 selection per plan (minimum
  persisted `validation_loss`, canonical tie-break), pinned as the
  concrete checkpoint id on `resolved_checkpoint_id`, flowing into the
  immutable run record and its `plan_hash`. The executor ALWAYS
  receives a concrete id and never asks "what is best?"; M58
  repetitions re-resolve per iteration exactly like every other `best`
  declaration; the M51 preflight pins the same id
- **exactly one source**: `publish_from_best` XOR an explicit
  `checkpoint_id` (both set, or neither -> 422); a pinned
  `resolved_checkpoint_id` is only valid with `publish_from_best=True`
  (the resolver's pinned form). The direct M3 rollback route keeps
  naming the concrete checkpoint explicitly — 'best' never leaks into
  M3; the workflow layer owns the declarative → concrete conversion
- **failure safety**: publication is the existing atomic M3 path —
  integrity verification FIRST (a corrupt/unreadable resolved
  checkpoint fails the stage; nothing is published, the previous
  published state and the immutable source stay intact), then the
  atomic weights write. A failing stage persists the normal `failed`
  run record with the error and the skipped remainder; on a gate stop
  (no `on_fail` branch) the publish stage is simply skipped — nothing
  is ever published from a rejected state
- **lineage**: the run record's plan carries the declarative request
  (`publish_from_best=True`) plus the pinned concrete id; the stage
  artifact is reference-oriented — the published checkpoint's id, its
  persisted weights content hash (`state_hash`) and its persisted
  validation loss — no checkpoint payload is duplicated into workflow
  records. The authoritative invariant after a successful
  `PUBLISH(best)`: resolved best checkpoint == published/live
  checkpoint
- **storage**: zero new artifacts per se — publication rewrites the
  model's `weights.pt` (the existing published-state file) and updates
  the model manifest's `latest_checkpoint`/`updated_at`, both through
  the established M3 path; no new checkpoint is created, no second
  best pointer exists, and the checkpoint registry is untouched
- OpenAPI: no new route — the stage surface rides the existing
  `POST /workflows/run`, recipe-run and M51 preflight schemas; path
  count stays 85

### Milestone 58 — bounded finite recipe REPETITIONS
  (`WorkflowRecipeRunRequest.repetitions`)
- **concept**: the recipe-run request may declare `repetitions: N`
  (default 1, bounded 1..16 — a deliberately conservative finite maximum;
  null behaves as omitted): the recipe executes N times SEQUENTIALLY,
  each iteration a FULL independent resolution + execution cycle through
  the EXISTING single-run path (recipe lookup → model validation → M14
  expansion → WorkflowPlan → M53 best resolution → ONE WorkflowEngine
  run → ONE normal immutable record). This is the whole point: every
  `best` declaration (M53 refs, M55 resume, M56 evaluate, M57 gate
  baseline) re-resolves at THAT iteration's plan start, so the
  improvement loop advances across iterations instead of replaying one
  frozen resolution
- **records**: N normal immutable `WorkflowRecord`s (never a wrapper
  record, never nested records, no repetition storage tree, no loop
  state); deterministic execution order 1..k; the registered recipe
  manifest is never mutated — the count is an execution REQUEST
  parameter, so the same recipe runs as 1, 5 or 10 repetitions
- **responses**: `repetitions=1` (default or null) reproduces today's
  exact single-record response; `repetitions>1` returns an ordered
  batch view (`WorkflowRecipeRepetitionRun`: requested/actual counts,
  `stopped_early`, ordered workflow ids, the full records — also
  available read-only via the M35 by-recipe history)
- **abort semantics**: strictly sequential, no concurrency; an
  iteration ending `stopped` (a NORMAL gate outcome) ends the sequence
  and the batch is returned; an iteration whose stage RAISES keeps the
  existing failure semantics (the failed record persists, the error
  maps to 404/409/422, later iterations NEVER run) — no retry, no
  skip, no automatic rollback, no "continue anyway"
- **boundaries**: bounded and explicit only — no unbounded repetition,
  no while-improving, no convergence detection, no background queue;
  M51 preflight stays a SINGLE-iteration resolution preview (later
  iterations cannot be predicted: each re-resolves against the state
  its predecessors created); evidence reuse stays fully active
  (identical replays reuse exact evaluations/comparisons)
- OpenAPI: path count UNCHANGED (84); the request schema gains the
  bounded field and the batch response is a new schema component

### Milestone 57 — declarative BEST-BASELINE gate policies
  (`GatePolicy.baseline_from_best`)
- **concept**: a workflow/recipe GATE stage's INLINE policy may declare
  `baseline_from_best: true` — "judge the candidate against the model's
  best checkpoint" — closing the last non-declarative slot of the
  canonical improvement loop. The SAME workflow resolver that pins M53
  state refs, M55 best-resume train stages and M56 best-evaluation
  stages resolves the SAME M52 selection (minimum persisted
  `validation_loss`) ONCE per plan at execution/M51-preflight time and
  pins the concrete id (`resolved_baseline_checkpoint_id`, the M53
  pinned form); the gate engine then receives a PURE M6 policy with an
  explicit `baseline_checkpoint_id` — one selection system, one
  executor, the gate layer never queries "best" and M6 decision
  semantics (improved/unchanged/regressed, tolerance,
  `max_regression_delta`, `minimum_loss`, suggestion-on-fail) are
  untouched
- **semantics**: XOR with the explicit `baseline_checkpoint_id` and
  only valid with `baseline_type='checkpoint'` (the best IS a checkpoint
  baseline; contradictions -> 422, never a silent preference;
  `resolved_baseline_checkpoint_id` without best -> 422); a DIRECT
  `POST /gates/evaluate` declaring best -> 422 (no workflow context;
  `GET /models/{id}/checkpoints/best` answers it); a best-baseline
  policy cannot be REGISTERED -> 422 (registry manifests are immutable
  and shared across executions while the selection is resolved and
  pinned per plan — the pin lives in the workflow record, never in the
  registry); default `false` preserves every existing request exactly;
  OpenAPI path count UNCHANGED (84) — schema fields only
- **provenance & evidence**: the persisted decision embeds the EXECUTED
  policy (explicit concrete baseline id, best stripped) and its
  `baseline` side identifies the concrete checkpoint — evaluations and
  the comparison reuse the existing exact-evidence lookups (declaring
  best never duplicates evidence; two identical best-baseline gates in
  one plan share ONE comparison); the declarative recipe manifest is
  never rewritten while the immutable run record's plan pins the
  concrete id and a different resolution means a different `plan_hash`
- **timing**: resolution happens at PLAN START (the unchanged M53
  architecture): the gate judges against the best that existed when the
  workflow started — a checkpoint created later in the SAME plan is
  reachable only via the candidate's explicit `from_stage`. Each NEW
  execution re-resolves, so re-running the canonical loop — train,
  evaluate(best), gate(candidate vs best), train from best,
  evaluate(best) — picks up the improved state. Finite, explicit,
  re-runnable; NO automatic repetition

### Milestone 56 — declarative BEST-EVALUATION stages
  (`WorkflowEvaluationStage.checkpoint_from_best`)
- **concept**: a workflow/recipe EVALUATE stage may declare
  `checkpoint_from_best: true` — "evaluate the model's best checkpoint" —
  closing the last state-reference gap of M53/M55. The SAME workflow
  resolver that pins M53 state refs and M55 best-resume train stages
  resolves the SAME M52 selection (minimum persisted `validation_loss`)
  ONCE per plan at execution/M51-preflight time and pins the concrete id
  (`resolved_checkpoint_id`, the M53 pinned form); the M4 evaluation path
  then receives a PURE explicit-checkpoint probe — one selection system,
  one executor, the evaluation layer never queries "best" and never
  silently falls back to current weights
- **semantics**: XOR with the explicit `config.checkpoint_id` and with
  `checkpoint_from_stage` (a contradiction -> 422, never a silent
  preference; `resolved_checkpoint_id` without best -> 422); the DIRECT
  `POST /evaluations/run` route has no such field (unknown field -> 422) —
  direct runs name the checkpoint explicitly (e.g. the answer of
  `GET /checkpoints/best`); default `false` preserves every existing
  request exactly; OpenAPI path count UNCHANGED (84) — schema fields only
- **evidence**: identity stays the CONCRETE checkpoint + the existing M4
  probe identity — an evaluation for the resolved best checkpoint is
  reused through the existing M5/M6-style exact-evidence lookup (declaring
  `best` never duplicates evidence; two identical `evaluate(best)` stages
  in one plan share ONE evaluation record)
- **immutability & timing**: the declarative recipe stays
  `checkpoint_from_best: true` (manifest never rewritten); the immutable
  run record's plan pins the concrete id and its `plan_hash` differs
  across resolutions of different checkpoints. Resolution happens at PLAN
  START (the documented M53/M55 architecture): every `best` in ONE plan —
  including M55 `resume_from_best` train stages and M56 evaluate stages —
  shares the SAME single per-plan selection computed from the checkpoints
  existing when the workflow starts; in-run stage outputs are referenced
  explicitly via `checkpoint_from_stage` (no hidden state discovery).
  Each NEW execution re-resolves, so re-running the canonical loop picks
  up the improved state — finite, explicit, re-runnable; NO automatic
  repetition, no auto-improvement

### Milestone 55 — declarative BEST-RESUME for train stages
  (`TrainingConfig.resume_from_best`)
- **concept**: a workflow/recipe TRAIN stage may declare
  `resume_from_best: true` — "initialize this run from the model's best
  checkpoint" — connecting M53 (best state references) and M54
  (explicit training resume). The SAME workflow resolver that pins M53
  state refs resolves the SAME M52 selection (minimum persisted
  `validation_loss`) ONCE per plan at execution/M51-preflight time and
  pins the concrete id (`resolved_resume_checkpoint_id`); the training
  engine then receives a PURE M54 `resume_from_checkpoint_id` — one
  selection system, one executor, the training layer never queries
  "best"
- **semantics**: XOR with the explicit id (both set is a contradiction
  -> 422; `resolved_resume_checkpoint_id` without best -> 422); a
  DIRECT `POST /training/run` with `resume_from_best` is rejected —
  direct runs name the checkpoint explicitly (e.g. the answer of
  `GET /checkpoints/best`); default `false` preserves every existing
  request exactly
- **immutability**: the declarative recipe stays `resume_from_best:
  true` (manifest never rewritten); the immutable run record's plan
  pins the concrete id and its `plan_hash` differs across resolutions
  of different checkpoints; the training provenance records the
  concrete start (`initial_checkpoint_id` + config) with the new
  checkpoints descending from it. Resolution happens at PLAN START: a
  best-resume stage sees the checkpoints existing when the workflow
  starts — in-run stage outputs are referenced explicitly via
  `from_stage` (no hidden state discovery)
- **the loop, declaratively**: `train -> evaluate -> train from best
  -> evaluate -> gate` is now ONE registered, immutable, re-runnable
  recipe (finite and explicit — NO automatic repetition, no
  while-improving, no auto-improvement). OpenAPI path count UNCHANGED
  (84) — schema fields only

### Milestone 54 — explicit training RESUME point
  (`TrainingConfig.resume_from_checkpoint_id`)
- **concept**: a training run may name the immutable checkpoint it
  initializes from — `resume_from_checkpoint_id: <checkpoint id>`
  (default `None` = exactly today's behavior: start from the model's
  current published weights). This closes the non-destructive
  improvement loop: train → evaluate → select best (`/checkpoints/best`)
  → continue FROM that checkpoint — WITHOUT `rollback` (which publishes
  the checkpoint as the model's current state just to prepare a run)
- **non-destructive (the M54 invariant)**: the resume checkpoint is
  verified through the EXISTING checkpoint verifier (model-scoped
  lookup, weights content-hash) and its weights initialize THIS run
  only — the model's published `weights.pt` and `latest_checkpoint`
  are NOT touched to prepare the run; publication happens only through
  the normal training completion semantics (keep-best / final adoption)
- **provenance**: the run records its concrete starting checkpoint in
  the EXISTING lineage fields — `RunProvenance.initial_checkpoint_id`
  (and `parent_checkpoint_id`, plus the full config JSON) — and the
  run's first checkpoint descends from it (`parent_checkpoint_id`
  chain). The baseline evaluation (the keep-best acceptance anchor) is
  measured on the RESUMED state, so acceptance decisions stay
  mathematically consistent. Model-WEIGHT resume only: M3 persists no
  optimizer/scheduler state, and none is restored (fresh optimizer,
  documented honestly)
- **boundaries**: unknown checkpoint or a checkpoint of ANOTHER model
  -> the established 404; silently-corrupted weights (content-hash
  mismatch) -> 409; unreadable weights -> 422; empty id -> 422
  (schema). Available everywhere `TrainingConfig` flows — the
  `POST /training/run` route and workflow/recipe `train` stages — with
  NO new endpoint (OpenAPI path count UNCHANGED). `rollback` keeps its
  own meaning: PUBLISH a checkpoint as current state; resume means:
  initialize ONE run from it without publishing. The checkpoint is
  referenced, never copied

### Milestone 53 — best-checkpoint STATE REFERENCES
  (`state_kind: "best"` on workflow stage states)
- **concept**: a workflow stage state (`StageStateRef` — suite-run
  states, comparison sides, gate candidates) may declare
  `state_kind: "best"`: "use the best checkpoint" WITHOUT hand-pinning
  an id. At execution or M51-preflight time the workflow engine's ONE
  resolver invokes the SAME M52 selection (minimum persisted
  `validation_loss` over the authoritative M3 listing, canonical
  `(step, created_at)` tie-break, non-finite values never candidates)
  and pins the CONCRETE checkpoint id on the resolved plan
  (`resolved_checkpoint_id`)
- **immutability**: the RESOLVED plan is what executes, persists in the
  immutable run record and hashes — so the record shows exactly
  `best → concrete id`, its `plan_hash` differs across executions that
  resolved different checkpoints, and history never silently changes
  meaning when a later checkpoint becomes best. The recipe definition
  stays declarative (registered manifests are never rewritten; best is
  re-resolved on every execution/preflight against the checkpoints
  that exist at that moment — no stored pointer, no cross-time lock;
  the run record is authoritative for what actually ran). Downstream
  engines (M4/M5/M6/M10) receive a NORMAL literal checkpoint state —
  their records are byte-identical in shape to explicit-id runs
- **boundaries**: `best` must not set `checkpoint_id`/`from_stage`
  (an explicit id alongside the selection request is a contradiction
  — 422); direct comparison/gate/suite-run API requests still require
  `current`/`checkpoint` (best is a WORKFLOW-stage reference);
  a model with no selectable checkpoints fails resolution with the
  established 404 BEFORE anything executes or persists. No new
  endpoint — inline `POST /workflows/run`, recipe runs and the M51
  `GET .../recipes/{r}/plan` preflight all resolve through the same
  path; M14 composite recipes resolve `best` after deterministic
  expansion into the ONE final record. OpenAPI path count UNCHANGED
  (84) — only the schema gained the enum member + pinned-id field

### Milestone 52 — best-checkpoint selection
  (`checkpoints/best`)
- **concept**: one read-only MODEL-SCOPED selection primitive —
  `GET /models/{id}/checkpoints/best` — answers "which of this
  model's checkpoints has the MINIMUM PERSISTED `validation_loss`?"
  and nothing else. "Best" means exactly this one criterion — the
  endpoint never claims overall model quality (semantic quality,
  factuality, safety or generalization are NOT established by a lower
  validation loss)
- **grounding**: candidates are the model's own checkpoints from the
  AUTHORITATIVE M3 listing (the persisted `checkpoints/<id>/
  manifest.json` is the source; unreadable manifests are already
  skipped there — established M46 semantics); the selected
  `validation_loss` is read VERBATIM from the persisted manifest,
  never recomputed and never derived from perplexity, checkpoint
  `decision`s, evaluations, comparisons, ids or timestamps; persisted
  non-finite values are never candidates
- **determinism + ties**: `min()` over the listing's canonical
  `(step, created_at)` ASCENDING order — an exact tie on the minimum
  resolves to the FIRST checkpoint among equals in that established
  canonical order, and the tie is DISCLOSED via a `tied` flag in the
  response (never silently judged). The response is a minimal
  computed view `CheckpointSelection` (`model_id`, the explicit
  `criterion: "minimum_persisted_validation_loss"`,
  `candidate_count`, `tied`, and the complete verbatim
  `CheckpointRecord`) — never persisted, no selection pointer, no
  rankings/trends/recommendations
- **boundaries**: unknown model -> the family's 404; a VALID model
  with no selectable checkpoints (never trained, or every manifest
  unreadable) -> the established not-found semantic (404, nothing
  manufactured). Declared BEFORE the generic `{checkpoint_id}` detail
  route so "best" can never be captured as an id. Read-only: zero
  storage growth, no evaluation, no training, no weights touched, no
  `decision` changed. M3 listing/detail, M46 by-run, rollback and the
  M4–M51 surfaces are byte-identical before and after; the dashboard
  is untouched (it aggregates persisted facts; a computed selection
  view does not belong there)

## Quickstart

```bash
pip install -e ".[dev]"
pytest                       # 613 tests
python -m uvicorn app.api:app --port 8000   # landing at /, docs at /docs
```

### Data flow example

```bash
# 1. upload text -> versioned dataset
curl -X POST localhost:8000/api/v1/datasets/upload -F "files=@notes.txt" -F "name=my-corpus"
# {"dataset_id": "...", "version": 1, "counts": {...}, "splits": {...}, ...}

# 2. train a tokenizer on it
curl -X POST localhost:8000/api/v1/tokenizers/train \
     -F 'config={"name":"tok-1","vocab_size":2000}' -F "dataset_id=<id>"

# 3. tokenize
curl -X POST localhost:8000/api/v1/datasets/<id>/tokenize -H 'Content-Type: application/json' \
     -d '{"tokenizer_id":"<tok-id>"}'

# 4. integrity
curl localhost:8000/api/v1/datasets/<id>/verify
```

### Training example (M3)

```bash
# create a model whose vocab covers the tokenizer
curl -X POST localhost:8000/api/v1/models -H 'Content-Type: application/json' -d '{
  "config": {"name":"my-model","vocab_size":2048,"context_length":512,"hidden_size":384,
             "n_layers":6,"n_heads":6,"n_kv_heads":2,"intermediate_size":1024}}'

# continued pretraining (synchronous): 200 steps, cosine schedule, eval every 25
curl -X POST localhost:8000/api/v1/training/run -H 'Content-Type: application/json' -d '{
  "name":"run-1","method":"continued_pretraining","model_id":"<model-id>",
  "dataset_id":"<ds-id>","tokenizer_id":"<tok-id>","learning_rate":3e-4,
  "batch_size":8,"max_seq_len":256,"steps":200,"lr_schedule":"cosine",
  "warmup_steps":20,"eval_every_steps":25,"keep_best":true}'

# inspect checkpoints, then roll back to a specific verified one if desired
curl localhost:8000/api/v1/models/<model-id>/checkpoints
curl -X POST localhost:8000/api/v1/models/<model-id>/rollback \
     -H 'Content-Type: application/json' -d '{"checkpoint_id":"<ckpt-id>"}'
```

Train exactly one of `epochs` or `steps` (both → 422). `epochs` derives its step count from
the dataset size; `steps` recycles the data deterministically when the run is longer than
one epoch. The run's report carries `requested_batch_size` vs `actual_batch_size` (memory
auto-fit), baseline/best validation loss + perplexity, every checkpoint with its
accept/not_best decision, and `rolled_back_to` when keep-best restored an earlier state.

### Evaluation example (M4)

```bash
# evaluate the model's current published weights on the validation split
curl -X POST localhost:8000/api/v1/evaluations/run -H 'Content-Type: application/json' -d '{
  "model_id":"<model-id>","dataset_id":"<ds-id>","tokenizer_id":"<tok-id>",
  "split":"validation","batch_size":8,"max_seq_len":256}'

# evaluate a specific immutable checkpoint (same schema, state_kind=checkpoint)
curl -X POST localhost:8000/api/v1/evaluations/run -H 'Content-Type: application/json' -d '{
  "model_id":"<model-id>","checkpoint_id":"<ckpt-id>","dataset_id":"<ds-id>",
  "tokenizer_id":"<tok-id>","split":"validation","batch_size":8,"max_seq_len":256}'

# immutable history (404 for unknown models, [] when none)
curl localhost:8000/api/v1/models/<model-id>/evaluations
curl localhost:8000/api/v1/models/<model-id>/evaluations/<eval-id>
```

Evaluations never modify anything: weights, checkpoints, provenance and datasets stay
byte-identical; only one immutable `manifest.json` under
`models/<id>/evaluations/eval-<id>/` is created per run. `loss_nats` is the token-weighted
mean cross-entropy and `perplexity = exp(min(loss,100))` — lower is better *for this
objective only*; compare records only under the same dataset/version/split/tokenizer and
window length (`max_seq_len`, default = model context). `result_hash` is deterministic
over the evaluation inputs+results (not eval_id/timestamps), so identical evaluations are
recognisable across records. `records_covered` is the exact M2 split record count on full
evaluations and `null` when `max_eval_tokens` truncated the run (never estimated from the
flat token stream).

### Comparison example (M5)

```bash
# compare two checkpoints of a model on one identical probe
curl -X POST localhost:8000/api/v1/comparisons/run -H 'Content-Type: application/json' -d '{
  "model_id":"<model-id>",
  "state_a":{"state_kind":"checkpoint","checkpoint_id":"<ckpt-a>"},
  "state_b":{"state_kind":"checkpoint","checkpoint_id":"<ckpt-b>"},
  "dataset_id":"<ds-id>","tokenizer_id":"<tok-id>","split":"validation",
  "batch_size":8,"max_seq_len":256,"tolerance":0.0001}'
# -> {"verdict":"improved|regressed|unchanged","delta_loss_nats":..., ...}

# current weights vs an older checkpoint (same probe)
curl -X POST localhost:8000/api/v1/comparisons/run -H 'Content-Type: application/json' -d '{
  "model_id":"<model-id>","state_a":{"state_kind":"current"},
  "state_b":{"state_kind":"checkpoint","checkpoint_id":"<ckpt-old>"},
  "dataset_id":"<ds-id>","tokenizer_id":"<tok-id>","split":"validation",
  "batch_size":8,"max_seq_len":256}'

# immutable history (404 for unknown models, [] when none)
curl localhost:8000/api/v1/models/<model-id>/comparisons
curl localhost:8000/api/v1/models/<model-id>/comparisons/<comparison-id>
```

A comparison evaluates (or reuses an exact existing M4 evaluation of) each state under the
**same** dataset/version/split/tokenizer/window/token-cap/seed — only the state differs.
`delta_loss_nats = loss_B - loss_A`; the loss-only verdict uses `tolerance`:
`|delta| ≤ tolerance` → `unchanged`, negative beyond tolerance → `improved`, positive
beyond tolerance → `regressed`. Perplexity is reported as supporting information.
Repeating a comparison creates no duplicate evaluations (identical evaluation ids come
back) but always appends a new comparison record — an auditable history.
**"Improved"/"regressed" describes the measured loss on the specified evaluation probe.
It is not a universal judgment of model quality.** Comparisons across different probes or
across models with different configurations are refused (422).

### Gate example (M6)

```bash
# PASS/FAIL a candidate checkpoint against a fixed checkpoint baseline.
# The policy is inline: probe + baseline form + tolerance/constraints.
curl -X POST localhost:8000/api/v1/gates/evaluate -H 'Content-Type: application/json' -d '{
  "model_id":"<model-id>",
  "policy":{
    "name":"accept-if-no-regression","model_id":"<model-id>",
    "dataset_id":"<ds-id>","tokenizer_id":"<tok-id>","split":"validation",
    "batch_size":8,"max_seq_len":256,"tolerance":0.0001,
    "baseline_type":"checkpoint","baseline_checkpoint_id":"<baseline-ckpt>",
    "max_regression_delta":0.05},
  "candidate":{"state_kind":"checkpoint","checkpoint_id":"<candidate-ckpt>"}}'
# -> {"decision":"passed|failed","verdict":"improved|regressed|unchanged",
#     "delta_loss_nats":...,"reason":"...",
#     "suggested_checkpoint_id":"<baseline-ckpt>","hint":"rollback recommended",...}
#    (suggestion only appears on a failed checkpoint gate; the gate never rolls back)

# baseline forms: current weights | past evaluation result hash | absolute threshold
"baseline_type":"current"
"baseline_type":"evaluation_result_hash","baseline_result_hash":"<64-hex M4 hash>"
"baseline_type":"minimum_loss","minimum_loss":4.5    # no comparison, no baseline side

# immutable decision history (404 for unknown models, [] when none)
curl localhost:8000/api/v1/models/<model-id>/gates/decisions
curl localhost:8000/api/v1/models/<model-id>/gates/decisions/<decision-id>
```

`delta_loss_nats = candidate_loss − baseline_loss` (M5 semantics, negative = candidate
improved). The verdict is M5's loss-only verdict; **passed/failed is the policy's
decision** on top of it. Every decision embeds its full policy and points at its exact
evaluations (+ comparison when a state baseline was used), so the whole chain is
re-auditable. Repeating a request reuses evaluation/comparison evidence by identity and
only appends one new decision manifest.

### Workflow example (M7)
```bash
# one inline plan: train on the explicit M3 config, evaluate the train stage's
# final checkpoint, then gate it against the live baseline it just replaced
curl -X POST localhost:8000/api/v1/workflows/run -H 'Content-Type: application/json' -d '{
  "name": "cpt-then-gate", "model_id": "<model-id>",
  "description": "continued pretraining guarded by a stage gate",
  "stages": [
    {"stage_id": "train1", "type": "train",
     "training": {"method": "continued_pretraining", "model_id": "<model-id>",
                  "dataset_id": "<ds>", "tokenizer_id": "<tok>",
                  "learning_rate": 3e-3, "batch_size": 8, "max_seq_len": 32,
                  "steps": 200, "eval_every_steps": 25}},
    {"stage_id": "eval1", "type": "evaluate",
     "evaluation": {"config": {"model_id": "<model-id>", "dataset_id": "<ds>",
                               "split": "validation", "tokenizer_id": "<tok>",
                               "batch_size": 8, "max_seq_len": 32, "seed": 1},
                    "checkpoint_from_stage": "train1"}},
    {"stage_id": "gate1", "type": "gate",
     "gate": {"policy": {"name": "no-regression", "model_id": "<model-id>",
                         "dataset_id": "<ds>", "split": "validation",
                         "tokenizer_id": "<tok>", "batch_size": 8,
                         "max_seq_len": 32, "seed": 1, "baseline_type": "checkpoint",
                         "baseline_checkpoint_id": "<old-ckpt>",
                         "max_regression_delta": 0.1},
              "candidate": {"state_kind": "checkpoint", "from_stage": "train1"}},
     "on_pass": "eval2", "on_fail": "eval2"},
    {"stage_id": "eval2", "type": "evaluate",
     "evaluation": {"config": {"model_id": "<model-id>", "dataset_id": "<ds>",
                               "split": "validation", "tokenizer_id": "<tok>",
                               "batch_size": 8, "max_seq_len": 32, "seed": 1},
                    "checkpoint_from_stage": "train1"}}
  ]}'
# -> {"workflow_id":"...","status":"completed","result_hash":"...",
#     "stages":[{"stage_id":"train1","executed":true,
#                "artifact":{"kind":"training_report","checkpoint_id":"...",...}},...],
#     "transitions":[["train1","next","eval1"],["eval1","next","gate1"],
#                    ["gate1","passed","eval2"],["eval2","next",null]]}
# A failed gate without on_fail yields {"status":"stopped",
#   "suggested_checkpoint_id":"<baseline-ckpt>","hint":"rollback recommended"} —
#   the workflow never rolls back by itself.
curl localhost:8000/api/v1/models/<model-id>/workflows
curl localhost:8000/api/v1/models/<model-id>/workflows/<workflow-id>
```
Every run record embeds the executed plan verbatim and cites its evidence artifacts;
repeat runs of an identical plan reuse evaluations/comparisons by identity and only
append one new gate decision + one new run manifest (same `result_hash`).

### Dashboard example (M8)

```bash
# read-only history view: recomputed live from the persisted manifests, never writes
curl localhost:8000/api/v1/models/<model-id>/dashboard
# -> {
#      "model_id": "...", "config_hash": "...", "schema_version": 1,
#      "summary": {"id":"...","name":"...","latest_checkpoint":"...",
#                   "training_run_count":2, ...},
#      "checkpoints": [{"checkpoint_id":"...","parent_checkpoint_id":"...",...}],
#      "training_runs": [...],
#      "evaluations": [{"identity":{...exact probe...},"count":2,
#                        "records":[...]}, ...],        # one series per probe identity
#      "comparisons": [{"identity":{...state pair + probe + tolerance...},...}],
#      "gate_decisions": [{"policy_key":"...","policy":{...},"statistics":
#                          {"passed":2,"failed":1},"records":[...]}, ...],
#      "workflows": {"counts":{"completed":2,"stopped":1,"failed":1},
#                     "records":[...]},
#      "artifact_graph": {"nodes":[{"family":"evaluation","artifact_id":"..."},...],
#                          "edges":[{"source":{...},"target":{...},"role":"state"},...]},
#      "diagnostics": [],        # deterministic notes when artifacts are unreadable
#      "result_hash": "..." }   # sha256 over the dashboard JSON itself (minus the hash)
# Repeated GETs return byte-identical JSON until the immutable history changes;
# corrupt manifests are skipped with diagnostics instead of failing the view.
```

### Registry example (M9)

```bash
# register an immutable policy: a stable id for one exact M6 GatePolicy
curl -X POST localhost:8000/api/v1/policies -H 'Content-Type: application/json' -d '{
  "policy_id": "no-regression-on-general",
  "description": "guardrail: no regression vs the kept checkpoint on probe set P",
  "policy": {"name":"gate-no-reg","model_id":"<model-id>","dataset_id":"<ds>",
             "split":"validation","tokenizer_id":"<tok>","batch_size":8,
             "max_seq_len":32,"seed":1,"baseline_type":"checkpoint",
             "baseline_checkpoint_id":"<old-ckpt>","max_regression_delta":0.1}}'
# -> 201 {"policy_id":"no-regression-on-general","config_hash":"<64-hex>",...}
# identical content again -> the SAME definition; changed content under the same
# id -> 409 (policies are immutable: a new rule is a new id)

# register a named probe suite: a set of exact M4 probes (order-insensitive)
curl -X POST localhost:8000/api/v1/probe-suites -H 'Content-Type: application/json' -d '{
  "suite_id": "general-validation",
  "probes": [{"dataset_id":"<ds>","split":"validation","tokenizer_id":"<tok>",
              "batch_size":8,"max_seq_len":32,"seed":1},
             {"dataset_id":"<ds2>","split":"test","tokenizer_id":"<tok>",
              "max_seq_len":64,"seed":2}]}'
# -> 201 {"suite_id":"general-validation","probes_hash":"<64-hex>",...}
# suites are named containers: no score is computed and each probe stays an
# independent M4 evaluation probe

# gates and workflow gate stages may use policy_id instead of an inline policy
curl -X POST localhost:8000/api/v1/gates/evaluate -H 'Content-Type: application/json' -d '{
  "model_id": "<model-id>", "policy_id": "no-regression-on-general",
  "candidate": {"state_kind":"checkpoint","checkpoint_id":"<candidate-ckpt>"}}'
# -> the SAME M6 decision an identical inline policy would produce, plus
#    "policy_id" / "policy_config_hash" provenance on the record
curl localhost:8000/api/v1/policies
curl localhost:8000/api/v1/policies/no-regression-on-general
curl localhost:8000/api/v1/probe-suites
curl localhost:8000/api/v1/probe-suites/general-validation
```

### Suite-run example (M10)

```bash
# execute one named suite against ONE explicit model state (current or checkpoint):
# every probe runs as an independent exact M4 evaluation (reused when it exists)
curl -X POST localhost:8000/api/v1/suite-runs -H 'Content-Type: application/json' -d '{
  "model_id": "<model-id>",
  "suite_id": "general-validation",
  "state": {"state_kind": "checkpoint", "checkpoint_id": "<ckpt-id>"}}'
# -> {
#      "suite_run_id": "...", "model_id": "...", "config_hash": "...",
#      "state": {"state_kind":"checkpoint","checkpoint_id":"..."},
#      "state_hash": "...", "suite_id": "general-validation",
#      "suite_probes_hash": "...", "status": "completed",
#      "probe_count": 2, "completed_count": 2, "reused_count": 0, "failed_count": 0,
#      "results": [
#        {"probe": {"dataset_id":"<ds>","split":"validation","seed":1,...},
#         "evaluation_id": "eval-...", "outcome": "created"},
#        {"probe": {"dataset_id":"<ds2>","split":"test","seed":2,...},
#         "evaluation_id": "eval-...", "outcome": "created"}],
#      "result_hash": "...", "schema_version": 1 }
# Running the SAME request again reuses both evaluations (outcome "reused",
# identical result_hash) and appends only the second run manifest.
# A probe that cannot run (e.g. missing dataset) is recorded with
# "outcome":"failed", "evaluation_id": null and run "status":"failed".
curl localhost:8000/api/v1/models/<model-id>/suite-runs
curl localhost:8000/api/v1/models/<model-id>/suite-runs/<suite-run-id>
```

### Suite-run workflow stage example (M11)

```bash
# one workflow: train -> suite-run on that train stage's final checkpoint
curl -X POST localhost:8000/api/v1/workflows/run -H 'Content-Type: application/json' -d '{
  "name": "train-then-suite", "model_id": "<model-id>",
  "stages": [
    {"stage_id": "train1", "type": "train",
     "training": {"method": "continued_pretraining", "model_id": "<model-id>",
                  "dataset_id": "<ds>", "tokenizer_id": "<tok>",
                  "learning_rate": 3e-3, "batch_size": 8, "max_seq_len": 32,
                  "steps": 200, "eval_every_steps": 25}},
    {"stage_id": "suite1", "type": "suite_run",
     "suite_run": {"suite_id": "general-validation",
                   "state": {"state_kind": "checkpoint",
                             "from_stage": "train1"}}}]}'
# -> {"workflow_id":"...","status":"completed",
#     "stages":[{"stage_id":"train1","executed":true,
#                "artifact":{"kind":"training_report",...}},
#               {"stage_id":"suite1","executed":true,
#                "artifact":{"kind":"suite_run","artifact_id":"<suite-run-id>",
#                            "result_hash":"...","checkpoint_id":"<train1-final-ckpt>"},...}],
#     "transitions":[["train1","next","suite1"],["suite1","next",null]]}
# The workflow record REFERENCES the immutable suite-run artifact — per-probe
# evaluation results live in the suite-run manifest (GET the suite-run id via
# the M10 endpoints). No aggregate score is produced anywhere.
```

### Workflow recipe example (M12)

```bash
# 1) register an immutable recipe: suite-run against the model's current state
curl -X POST localhost:8000/api/v1/workflows/recipes -H 'Content-Type: application/json' -d '{
  "recipe_id": "current-suite-check",
  "description": "Run the general-validation suite on whatever state the bound model publishes",
  "stages": [
    {"stage_id": "suite1", "type": "suite_run",
     "suite_run": {"suite_id": "general-validation",
                   "state": {"state_kind": "current"}}}]}'
# -> 201 {"recipe_id":"current-suite-check","config_hash":"<64-hex>",...}
# identical re-registration -> same definition (no write); same id + different
# stages -> 409. Recipes are immutable: PUT/PATCH/DELETE do not exist (405).

# 2) execute it against ONE explicitly named model (the only runtime binding)
curl -X POST localhost:8000/api/v1/workflows/recipes/current-suite-check/runs \
     -H 'Content-Type: application/json' -d '{"model_id": "<model-id>"}'
# -> 200 {"workflow_id":"...","status":"completed","recipe_id":"current-suite-check",
#         "recipe_hash":"<config-hash>",
#         "stages":[{"stage_id":"suite1","artifact":{"kind":"suite_run",...}}]}
#    the run record snapshots the resolved state (checkpoint/state hash); the
#    recipe definition never changes. Re-running reuses exact M4 evidence:
#    +1 workflow manifest +1 suite-run manifest, 0 duplicate evaluations.
#    Unknown recipe/model -> 404 (nothing persisted); runtime stage failures
#    persist a failed run record exactly like inline M7 plans.
```

### Suite-run dashboard view & recipe lineage example (M13)

```bash
# the per-model dashboard now carries the model's suite-run history next to the
# evaluations those runs reference (records are the M10 manifests, read-only)
curl -s localhost:8000/api/v1/models/<model-id>/dashboard | python3 -m json.tool
# -> { ..., "suite_runs": {"counts": {"completed": 2, "failed": 0},
#      "records": [{"suite_run_id":"...","model_id":"...","status":"completed",
#                    "suite_id":"general-validation",
#                    "results":[{"probe":{...},"evaluation_id":"...",
#                                "outcome":"created|reused|failed",...}],...}]},
#      "artifact_graph": {"nodes":[...,{"family":"suite_run",
#                             "artifact_id":"<suite-run-id>"}],
#        "edges":[...,{"source":{"family":"workflow",...},
#                       "target":{"family":"suite_run",...},
#                       "role":"stage_artifact"},
#                      {"source":{"family":"suite_run",...},
#                       "target":{"family":"evaluation",...},
#                       "role":"probe"}]}, ... }
# records are filtered by the persisted model_id; corrupt manifests are skipped
# with diagnostics; counts are execution bookkeeping only - never a suite score

# cross-model run lineage of one recipe (live scan of workflow manifests)
curl -s localhost:8000/api/v1/workflows/recipes/current-suite-check/runs \
     | python3 -m json.tool
# -> [{"workflow_id":"...","model_id":"<model-a>","recipe_id":"current-suite-check",
#      "recipe_hash":"<config-hash>","status":"completed","created_at":"..."}, ...]
#    ordered (created_at, workflow_id) across every model that bound the recipe;
#    unknown recipe -> 404, registered-but-never-run recipe -> []. Read-only:
#    repeated calls are byte-identical and never write anything.
```

### Workflow recipe composition example (M14)

```bash
# 1) register a leaf recipe (an immutable suite-run stage on 'current')
curl -s -X POST localhost:8000/api/v1/workflows/recipes -H 'Content-Type: application/json' \
  -d '{"recipe_id":"leaf-check","stages":[{"stage_id":"s1","type":"suite_run",
        "suite_run":{"suite_id":"general-validation","state":{"state_kind":"current"}}}]}'
# -> { "recipe_id":"leaf-check","composition":null,"config_hash":"<hash>", ... }

# 2) register a composite that trains, then calls the leaf, then gates
curl -s -X POST localhost:8000/api/v1/workflows/recipes -H 'Content-Type: application/json' \
  -d '{"recipe_id":"train-check","stages":[
        {"stage_id":"t1","type":"train","training":{...}},
        {"stage_id":"leg","type":"recipe","recipe":{"recipe_id":"leaf-check"}}]}'
# -> { "recipe_id":"train-check","stages":[<declared stages, call stage intact>],
#      "composition":[{"recipe_id":"leaf-check","config_hash":"<leaf hash>"}],
#      "config_hash":"<composite hash over declared stages + referenced id/hash>", ... }
#    unknown reference -> 422 (no manifest); re-registering identical content is
#    idempotent (201, no write); different content under the same id -> 409

# 3) run the composite: ONE explicit model, ONE workflow record
curl -s -X POST localhost:8000/api/v1/workflows/recipes/train-check/runs \
  -H 'Content-Type: application/json' -d '{"model_id":"<model-id>"}'
# -> { "workflow_id":"...","status":"completed","recipe_id":"train-check",
#      "composition":[{"recipe_id":"leaf-check","config_hash":"<leaf hash>"}],
#      "plan":{"stages":[
#         {"stage_id":"t1","type":"train",...},
#         {"stage_id":"leg.s1","type":"suite_run",  # qualified deterministically
#          "suite_run":{...}}]}, ... }
#    expansion is pure: stage 'leg.s1' is the leaf's stage qualified by the call
#    path 'leg'; every internal from_stage/on_pass/on_fail reference of a
#    referenced recipe is rewritten to its qualified id; M4 evidence is reused
#    exactly (no duplicate evaluation manifests); lineage GET shows this run
#    under 'train-check' only, never under 'leaf-check'.
```

### Deterministic checkpoint sampling example (M15)

```bash
# generate from ONE explicit verified checkpoint (greedy = argmax, no RNG)
curl -s -X POST localhost:8000/api/v1/samples/generate -H 'Content-Type: application/json' \
  -d '{"model_id":"<model-id>","checkpoint_id":"<ckpt-id>","tokenizer_id":"<tok-id>",
       "prompt":"river mountain cloud forest","strategy":"greedy","max_new_tokens":16}'
# -> { "sample_id":"...","model_id":"...","checkpoint_id":"...",
#      "checkpoint_weights_sha256":"<verified content hash>",
#      "tokenizer_id":"...","tokenizer_hash":"...","prompt":"river mountain cloud forest",
#      "prompt_token_ids":[...],"generated_token_ids":[...],"output_text":"...",
#      "strategy":"greedy","temperature":null,"seed":null,
#      "max_new_tokens":16,"result_hash":"<deterministic>", ... }
#    identical requests -> identical token ids/text/result_hash, distinct sample ids.
#    Seeded temperature: add "strategy":"temperature","temperature":0.8,"seed":7.
#    Every preflight failure writes nothing: unknown model/checkpoint/tokenizer
#    -> 404; corrupt checkpoint -> 409; vocab mismatch / bad params / context
#    overflow / temperature=0 / greedy+seed -> 422.

# immutable history (read-only, deterministic order)
curl -s localhost:8000/api/v1/models/<model-id>/samples | python3 -m json.tool
curl -s localhost:8000/api/v1/models/<model-id>/samples/<sample-id> | python3 -m json.tool
```

### Per-sample quality measurement example (M16)

```bash
# measure ONE generated sample under ITS OWN recorded checkpoint+tokenizer
# (no body: the immutable sample manifest is the source of truth)
curl -s -X POST localhost:8000/api/v1/models/<model-id>/samples/<sample-id>/quality
# -> { "evaluation_id":"...","model_id":"...","sample_id":"...",
#      "sample_result_hash":"<the sample's result_hash>",
#      "token_sequence_sha256":"<digest of the exact measured ids>",
#      "checkpoint_id":"...","checkpoint_weights_sha256":"<verified hash>",
#      "tokenizer_id":"...","tokenizer_hash":"...",
#      "prompt_token_count":24,"generated_token_count":8,
#      "evaluated_token_count":8,"context_length":64,
#      "window_token_count":32,"window_rule":"single_window",
#      "loss_nats":6.438885,"perplexity":625.70858,"result_hash":"...", ... }
#    prompt tokens condition only; generated tokens are scored exactly once;
#    first generated token conditioned on the full prompt; overlong samples
#    -> 422 (never truncated). Unknown model/sample/checkpoint/tokenizer ->
#    404; corrupt/inconsistent records -> 409; every failure writes nothing.
#    Identical repeat -> same loss/perplexity/result_hash, new evaluation_id.

# immutable measurement history (read-only, deterministic order)
curl -s localhost:8000/api/v1/models/<model-id>/sample-quality | python3 -m json.tool
curl -s localhost:8000/api/v1/models/<model-id>/sample-quality/<evaluation-id> | python3 -m json.tool
```

## Layout

```
ai-model-forge/
  app/
    config.py          # paths, env, logging, versions
    schemas.py         # ALL schemas: configs, records, manifests (single source of truth)
    hardware.py        # CPU/GPU/VRAM/RAM/disk detection
    storage.py         # atomic writes, hashing, artifact layout helpers
    model_builder.py   # configurable Transformer + generation + factory
    dataset.py         # data engine: ingestion, dedup, splits, versions, verify, tokenize
    tokenizer.py       # byte-level BPE engine
    training.py        # training engine: schedules, streams, run, checkpoints, rollback
                       # + read-only by-run grouping of the checkpoint history (M46)
                       # + read-only best-checkpoint selection (M52)
                       # + 'best' state references resolved via that selection (M53)
    evaluation.py      # evaluation engine: read-only state measurement (M4)
                       # + read-only by-checkpoint/by-dataset/by-tokenizer/by-split/
                       #   by-state-kind/by-truncated/by-seed grouping
                       #   (M24/M28/M30/M36/M38/M47/M48)
    comparison.py      # comparison engine: A/B states over identical probes (M5)
                       # + read-only by-checkpoint/by-dataset/by-tokenizer/by-split/
                       #   by-verdict/by-state-kind grouping
                       #   (M26/M29/M31/M37/M39/M44/M49)
    gates.py           # stage gates: policy-driven run decisions (M6)
                       # + read-only by-policy/by-comparison/by-decision/
                       #   by-verdict/by-baseline-type grouping
                       #   (M23/M34/M41/M43/M45)
    workflows.py       # workflow engine: ordered orchestration over M3-M6 (M7)
                       # + read-only by-recipe/by-status grouping (M35/M42)
    dashboards.py      # read-only dashboard engine: deterministic views (M8/M13)
    policies.py        # policy registry + probe suites: immutable definitions (M9)
    suite_runs.py      # explicit multi-probe M4 batches over named suites (M10)
                       # + read-only by-suite/by-checkpoint grouping (M21/M22/M25)
                       # + read-only by-reused-count grouping (M50)
    recipes.py         # workflow recipes: immutable plans + M14 composition (M12/M14)
                       # + read-only model-bound recipe resolution preflight (M51)
    sampling.py         # checkpoint sampling: deterministic generation (M15)
                       # + read-only by-checkpoint/by-tokenizer/by-strategy
                       #   grouping of the history (M27/M32/M40)
    sample_quality.py  # per-sample likelihood measurement of samples (M16)
                       # + read-only by-sample/by-checkpoint/by-tokenizer grouping (M19/M20/M33)
    engine.py          # facade composing all engines
    api.py             # FastAPI routes (thin)
  tests/               # 613 tests across 20 suites
```

Forge data lives outside the source tree at `~/ai-model-forge-data` (override `FORGE_ROOT`):

```
<root>/
  project.json
  models/<id>/            manifest.json · weights.pt · weights.sha256
    checkpoints/<ckpt_id>/ manifest.json · weights.pt        (immutable, content-addressed)
    evaluations/eval-<id>/ manifest.json                     (immutable, read-only runs)
    comparisons/comp-<id>/ manifest.json                     (immutable, A/B records)
    gates/gate-<id>/     manifest.json                     (immutable, policy decisions)
    workflows/workflow-<id>/ manifest.json                (immutable, one run per manifest)
  suite-runs/<run_id>/     manifest.json                (immutable, one run per manifest; M10)
  policies/<policy_id>/    manifest.json                (immutable definitions; M9)
  probe-suites/<suite_id>/ manifest.json                (immutable definitions; M9)
  datasets/<ds_id>/       dataset.json · records.sha256 (dedup index)
    v1/                   manifest.json · records.jsonl.gz          (immutable)
      tokenized/<tok_id>/ manifest.json · train.bin · validation.bin · test.bin
    v2/                   ... (appended snapshots never rewrite v1)
  tokenizers/<id>/        manifest.json · tokenizer.json
  samples/<model_id>/sample-<id>/ manifest.json             (immutable; M15)
  sample-evaluations/<model_id>/evaluation-<id>/ manifest.json
                                                            (immutable; M16)
```

## Storage-efficiency rules in force

- metadata exists once; lists are derived, never mirrored
- records stored compressed (deterministic gzip); **no source-file copies** — only hashes + metadata
- versions are append-only and immutable once written; only the pointer + dedup index mutate
- dedup: sha256 record identity; within-upload duplicates dropped and counted; cross-version duplicates reported
- checkpoint states are content-addressed: identical tensor content ⇒ one physical file
  (hard link); checkpoints are immutable once written and never rewritten
- atomic writes everywhere (temp + fsync + rename); temp cleanup on startup

## Roadmap

`MODEL → DATA → TRAINING → EVALUATION → IMPROVEMENT → OPTIMIZATION → DEPLOYMENT`

Milestones 12–18 delivered **named workflow recipes**, **read-only suite-run
observability**, **composable recipes**, **deterministic checkpoint sampling**,
**per-sample quality measurement**, **sample-quality dashboard observability**
and **sample-quality record access** (see above): one-shot inline M7 plans can
be registered once and re-invoked with an explicit model binding, recipes may
reference other registered recipes and are expanded deterministically into one
validated plan at registration time, the M8 dashboard renders the model's
suite-run history (with per-probe evaluation references) and, since M17, a
read-only sample-quality section (per-sample evaluation counts + newest
references, derived live from the immutable M16 records, never stored), recipe
runs expose cross-model lineage over the existing workflow manifests, text
generation from an explicitly verified checkpoint is an immutable, reproducible
inference primitive, any generated sample can be measured deterministically
under the existing causal-LM objective (loss/perplexity over its generated
targets only, sample-driven state resolution, immutable sample-evaluation
records), and — since M18 — the full immutable measurement records are exposed
through one dedicated read-only listing route (metric values included, no
derived statistics).
Remaining natural steps: scheduled or repeatable execution of the synchronous
engines (deliberately out of scope until extended on purpose), suite
aggregation/quality notions (currently forbidden by the no-judgment rule),
richer policy inputs, and an automatic improvement loop.
Interactive/served inference (chat, streaming, multi-turn state) and training
methods beyond CPT/SFT (incl. LoRA) are intentionally later milestones.

### Milestone 14 — composable workflow recipes (`recipe` stages)
- **concept**: a registered recipe may reference OTHER registered recipes through a
  `recipe` stage (conventional stage shape: `stage_id` + `type: "recipe"` + one
  `recipe_id`). Compositions are static definitions: registration expands them
  deterministically into ONE fully validated M7 plan; running one executes the
  existing engines exactly once — no second registry/validator/executor, no nested
  or per-subrecipe runs, no dynamic execution
- **registration order — any failure leaves nothing behind**: shared schema rules
  (recipe stages are legal only inside registered recipe definitions; inline
  `WorkflowPlan`s reject them) → referenced recipes must already exist (rejected at
  registration time, never deferred to run time) → cycle rejection (self / A→B→A /
  longer, over the immutable registry) → depth limit: at most 32 chained recipes
  (32 accepted, 33 rejected) → deterministic expansion → the SHARED M7
  `validate_plan_stages` over the fully expanded list → composite config hash → one
  atomic manifest
- **documented qualification convention**: a referenced recipe's stages are
  spliced in parent order at the call position and their ids are qualified by the
  dot-joined chain of call-stage ids that led to them — `call.stage` for one
  level, `a.b.stage` for nested calls. The top-level recipe's own stages keep
  their declared ids, so a composite can never collide with a callee's stages
- **internal references stay internal**: every `from_stage` / gate
  `on_pass`/`on_fail` target inside a referenced recipe that names one of its own
  non-call stages is rewritten to the qualified id (binding `current` /
  checkpoints / earlier-train-stage semantics exactly as M7/M10/M12 define them).
  Targets that would cross a recipe boundary — e.g. a branch into a call stage,
  which disappears during expansion — are rejected at registration; no rule is
  invented at run time
- **hashing**: plain recipes keep the exact M12 config hash (canonical JSON of the
  ordered stages). Composite config hashes are sha256 over the declared ordered
  stages PLUS the referenced recipes' ids and config hashes (declared order) —
  immutable dependencies therefore pin their consumers; ids, descriptions,
  timestamps and paths are excluded. Same content → same hash; a changed stage or
  a changed dependency hash → a changed composite hash
- **manifests & provenance**: composite manifests store the composition-level
  ordered stage list (recipe-reference stages intact — never a flattened copy)
  plus a reference-oriented `composition` (recipe_id + config_hash per direct
  dependency); referenced manifests are never copied. Run records keep M12
  `recipe_id`/`recipe_hash` (the recipe the caller invoked) plus an additive
  depth-first `composition` trace of the expansion; inline-plan runs keep all
  three null
- **one workflow per execution**: top-level invocation owns the single record;
  referenced recipes never produce records of their own and never claim the run in
  M13 lineage — composition provenance is not execution ownership. Lineage
  ordering and byte-determinism are unchanged
- **explicit binding**: exactly one `model_id` per run request (M12 rule, kept);
  nested recipes never bind or auto-select a model, and a model pinned inside a
  referenced recipe's config survives expansion untouched — a conflict with the
  bound model is rejected BEFORE execution, never rewritten
- **evidence reuse**: suites inside composed recipes run through the existing M10
  engine with the exact M4/M10 reuse rules — repeated identical runs add run +
  suite-run manifests only, never duplicate evaluation manifests
- **storage & API**: writes are limited to exactly one recipe manifest per
  registration and one workflow manifest per execution (+ the M7/M10 artifacts
  those stages legitimately create); identical re-registration is a no-write;
  changed content under the same id is 409 with the original bytes untouched.
  The five existing recipe endpoints are unchanged; POST registration accepts the
  new stage; POST run executes the expansion; OpenAPI documents the `recipe`
  stage. No `/compose` or `/expand` endpoints; no scheduling/queues/workers, no
  runtime parameters or recipe substitution, no editing/deletion/versioning, no
  suite aggregation or quality numbers, no dashboard redesign
## Honest limitations

- `sft` and `continued_pretraining` currently share identical causal-LM mechanics — no
  chat templates, role masks, instruction loss, preference data or RL methods yet
- Training is CPU-first, synchronous and single-machine: no Distributed/queue/GPU
  orchestration yet; CUDA code paths are exercised only when a GPU is present
- M4/M5 metrics are loss/perplexity on the causal-LM objective only — no quality scores,
  sampling-based or human-like evaluation, no capability suites yet. A comparison's
  verdict is only meaningful for the exact probe it used (dataset/version/split/
  tokenizer/window); the same pair of states can legitimately show different verdicts
  on different probes
- M6 gate decisions inherit the same loss-only scope: **passed/failed is a statement
  about one policy on one probe for one candidate state**, never a universal quality
  judgment. A policy that passes here can fail on another probe, another baseline form
  or a different tolerance — that is by design, and each decision records exactly which
  policy/probe/baseline produced it
- M6 gates never execute rollback/retraining/model selection: a failed checkpoint gate
  only records `suggested_checkpoint_id` + a hint. Since M60 the user may AUTHOR an
  explicit PUBLISH stage (resolved/pinned through the same M53/M52 machinery, executed
  by the M3 rollback path) after the gate or on a gate branch of a workflow — but the
  gate itself still executes nothing, and nothing is ever automatic
- Gate policies are inline (no registry/CRUD) and every request appends a decision; the
  "current weights" baseline (B) and evaluation-result-hash baseline (C) mean what they
  say *at request time*: (B) is re-measured live and (C) stays valid only while the
  referenced evaluation manifest exists and matches the policy probe exactly
- Boundary precision: comparison deltas are stored rounded to 6 decimals (M4/M5
  convention); a gate's max-regression/tolerance edge is therefore exact to that
  rounding, and `minimum_loss` compares the candidate evaluation's stored loss
- Absolute `minimum_loss` and `max_regression_delta` bounds are single scalars; probe
  suites, per-split policies or policies over non-loss criteria are future work
- M7 workflows inherit the loss-only scope of the stages they orchestrate: a `completed`
  workflow is a statement about the exact M3–M6 engines, configs, probes and evidence
  embedded in its run record — never a universal quality judgment. Identical workflow
  plans can legitimately stop vs. complete as the model/data/evidence changes
- M7 plans are inline and runs are synchronous: one plan per request, executed start to
  finish (a failed stage persists a `failed` run and surfaces the error). There is no
  plan registry, editing, deletion, cancellation, scheduling, queue or background worker
- A gate that fails without an `on_fail` branch stops the run with a recorded suggestion
  (`suggested_checkpoint_id` + hint) — the workflow NEVER executes rollback, retraining,
  model selection or hyperparameter search; those remain explicit user calls
- M7 does not deduplicate training runs: identical `train` stages execute the supplied
  M3 config each time (each run is its own lineage event); evaluations/comparisons are
  the artifacts reused by identity
- The model manifest's `state_hash` field is recorded at model creation and is not
  rewritten by M3 training finalize (pre-existing M1/M3 behavior). It is never used by
  the M4–M7 resolution paths, which recompute the live weights' content hash at query
  time and verify the raw-byte `weights.sha256` sidecar instead
- M5 comparison verdicts are loss-only and tolerance-based; the default tolerance
  (1e-4 nats) is strict — near-tie comparisons should set a larger tolerance explicitly
- M4 `records_covered` is exact for whole-split evaluations and `null` for capped ones;
  M2's flat token streams carry no per-token record boundaries, so finer coverage
  accounting is not possible without a data-pipeline change
- Current-state integrity relies on the raw-byte `weights.sha256` sidecar (M1/M3
  convention); the model manifest's `state_hash` field reflects creation time unless a
  later milestone refreshes it (the evaluated canonical hash is always recomputed live)
- Checkpoint content hashes cover loaded tensor values (fp32-canonical); a tamper that
  only alters zip-container metadata without changing values is not detected
- Cross-version dataset duplicate *reports* exist; cross-version physical dedup is not
  done. Checkpoint physical dedup (hard links) only triggers on byte-identical states
- Byte-level BPE preserves all printable text incl. tabs/newlines, but raw control bytes
  (e.g. `\x00`) do not round-trip (standard ByteLevel behaviour, documented in `tokenizer.py`)
- No PDF/DOCX/EPUB/image/audio/video ingestion yet; no URLs/APIs/SQL; no tokenizer import
  from HF; no sampling/top-p; no Gemini — later milestones
- M8 dashboards inherit the loss-only, per-probe scope of M4–M7: they aggregate recorded
  loss/perplexity evidence and decisions, and never compute quality scores, universal
  judgments, leaderboards or rankings of their own
- A dashboard is a live view: it reflects exactly the manifests currently on disk (skips
  corrupt ones with diagnostics). Nothing is cached or persisted, so it is only as
  current as the last GET, and `result_hash` changes whenever the underlying storage
  state changes (e.g. a new immutable record appended by M3–M7) — that is expected
  behaviour, not a signature of mutability
- Checkpoints are listed chronologically (creation order — the lineage chain); the M3
  list endpoint's step-first view may interleave concurrent runs differently. Both are
  deterministic views of the same immutable records
- M9 definitions are global to one FORGE_ROOT (not per model); a policy embeds its
  target `model_id` and is only usable by that model's gates (checked at resolution).
  Probe suites are model-agnostic by design — model/state are supplied at use time
- The registry stores definitions only: nothing validates that a policy's referenced
  checkpoint/dataset still exists until the policy is USED in a gate; a definition may
  outlive its evidence, and gate requests against stale definitions fail cleanly at
  execution time (they never auto-update or rewrite anything)
- Re-registering the same id with identical content returns the existing definition
  (idempotent); there is deliberately no update/versioning endpoint — a changed rule
  requires a new id, and old decisions keep referencing their executed (embedded)
  policy, never the newest definition under the same name
- Probe suites are containers: M9 executes nothing automatically, and gate policies
  still apply to ONE probe at a time (the suite probe + candidate state given in the
  gate request). Multi-probe execution and per-suite aggregation are explicitly future
  work and would need their own milestone
- Registry manifest corruption is handled per definition: list() skips unreadable
  definitions with a log warning and resolution reports a RuntimeError (409 at the
  API); a corrupt policy is unusable until the user removes/replaces the directory —
  nothing repairs itself
- M10 suite runs are per-model, per-state executions of a GLOBAL suite: a suite may
  reference any dataset/tokenizer in the root, and a run only records per-probe
  evaluation references — nothing more. Runs never aggregate, and there is no
  "best probe" or "best suite" notion anywhere
- Suite runs execute synchronously in request scope (like M3–M7): no scheduling,
  batching or background execution exists yet
- Failure handling distinguishes preflight errors (unknown model/suite/state or
  corrupt suite → nothing persisted, 404/409) from per-probe execution failures
  (missing/invalid underlying artifacts → recorded per-probe, run status `failed`).
  An unreadable/corrupt EXISTING evaluation manifest is invisible to the exact-reuse
  lookup, so a suite run creates a fresh evaluation for that probe instead of reusing
  the broken record — evidence is treated as unavailable, never guessed at
- A suite run's `result_hash` covers semantic execution only; evidence that changes
  between runs (new dataset versions resolving differently, altered state hashes,
  new/removed evaluations) legitimately changes it — that is a property of the
  immutable evidence, not of the run record
- Suite-run history is listed via its own endpoints AND (M13) in the per-model
  dashboard's `suite_runs` section, which renders each model's own records only —
  there is deliberately no cross-model suite view; the only cross-model read is the
  recipe lineage endpoint
- M11 suite stages may only reference states M7 can represent: current weights, a
  literal checkpoint id, or an earlier TRAIN stage's final checkpoint via
  `from_stage`. Evaluating a checkpoint produced by any other stage type is not
  possible (those stages produce no checkpoints) and is rejected at plan validation
- A `suite_run` stage is a non-branching stage: workflows branch only on M6 gate
  decisions (on_pass/on_fail), so a suite stage cannot conditionally stop a workflow
  by itself — that remains the explicit job of a later gate stage
- The workflow record stores suite-run provenance (id, result hash, evaluated
  checkpoint); the full per-probe detail lives in the suite-run manifest and is
  fetched through the M10 endpoints or, since M13, through the dashboard's
  `suite_runs` section (records verbatim, per probe)
- Underlying per-probe suite failures surface through the referenced suite-run
  record's `failed` status; the M7 workflow itself completes because the suite-run
  artifact genuinely exists — workflows do not convert probe failures into stage
  failures (that distinction is documented in the M10 limitation list)
- A recipe is a frozen ordered stage list: it cannot contain placeholders or
  per-run parameters beyond the model binding, so a recipe that must evaluate a
  different suite per run is expressed as separate recipes. Recipes never compose
  other recipes yet (no recipe-in-recipe expansion)
- Registration validates structure only: suites/checkpoints/policies referenced by a
  recipe are resolved at RUN time, so a recipe may be registered before its
  referenced evidence exists — a missing artifact then fails that run cleanly
  (failed run record with provenance) instead of failing registration
- A recipe whose embedded stage configs name a model (train/evaluate stages, inline
  gate policies) is pinned to that model: binding a run to a different model is
  rejected (422) before anything executes. Only model-agnostic recipes (compare /
  registry-policy gate / suite_run stages, whose state binds per run) execute
  against any model — this keeps binding explicit and never silently rewrites
  stored configs
- Recipe run records keep `recipe_id`/`recipe_hash` provenance fields additive and
  null for inline runs; `result_hash` intentionally covers semantic execution only,
  so an inline run and a recipe run with identical bound plans hash identically
- The M8 dashboard renders recipe runs as workflow history (with provenance); there
  is no dedicated recipe-management UI, no recipe rankings and no aggregate recipe
  quality notion — recipes are definitions, not judgments
- M13 is strictly read-only observability: dashboards and recipe lineage recompute
  live from the immutable manifests on every call — zero writes, zero new storage
  families, no indexes, caches, timestamps or background work of any kind. Repeated
  reads over identical storage are byte-identical by construction and audited as
  such; if a future milestone needs caching it must opt in deliberately
- The dashboard `suite_runs` section is per-model (like every dashboard section):
  it shows the records whose persisted `model_id` is that model's. Suite runs of
  other models are visible only through that model's own dashboard — there is no
  cross-model suite-run view and no suite leaderboard
- The dashboard renders each suite run's execution bookkeeping and per-probe
  references exactly as recorded. `reused` means the run reused an existing exact
  M4 evaluation — it says nothing about probe success, and no suite score, pass
  percentage, average loss/perplexity, quality ranking or benchmark is computed
  anywhere in M13; a `failed` per-probe outcome is shown only where M10 recorded it
- Recipe lineage reflects workflow manifests only: a recipe run that failed before
  persisting a run record is invisible (there is nothing to show), and inline runs
  (null `recipe_id`) never appear under any recipe. Lineage is a live scan, so it
  reflects exactly the manifests on disk at GET time
- Corrupt or manifest-less suite-run directories are skipped with deterministic
  diagnostics (same convention as every dashboard family); valid records next to
  them stay visible. A suite run whose per-probe record references a missing
  evaluation stays visible but its dangling reference surfaces only as a graph
  diagnostic — M13 never fabricates an evaluation node or edge
- M14 compositions are static, definition-time reuse: a recipe call carries no
  parameters and nothing is substituted at run time — if a milestone ever needs
  parameterized pipelines it must be designed on purpose (runtime parameters,
  editing/deletion/versioning/renaming of recipes and marketplace/sharing remain
  forbidden by the current scope)
- The depth limit (32 chained recipes) is enforced at registration with a
  deterministic error; it is a guardrail, not a recommendation — deep chains also
  lengthen qualified stage ids (each nesting level adds its call-stage id + "."),
  and registration rejects any qualified id longer than the 64-character stage-id
  limit. Through the public API a cycle can never even be closed (registrations
  are immutable and may only reference existing recipes); cycle detection is
  defense in depth for registries manipulated outside the API
- Registration-time rejection means a composite is only as current as the registry
  when it is registered: every referenced recipe is pinned by id + config hash at
  registration, and the immutable registry guarantees the expansion can never
  change afterwards. There is no way to "rebind" a composite to newer dependency
  content under the same id — that would require versioning, which is out of scope
- A referenced recipe's model-pinned configs (train/evaluate/gate policy targets)
  survive expansion untouched and are checked against the caller's bound model at
  run start: mismatch → 422 with nothing persisted. Recipes therefore compose
  cleanly only when every stage targets the same explicit model family; composing
  recipes that pin different models in one run is intentionally impossible
- Composite execution is a single workflow run: it appears in dashboard workflow
  history and M13 recipe lineage exactly once, under the invoked top-level recipe.
  The per-run `composition` trace lists referenced recipes and their config hashes
  for auditability; it is provenance (what was expanded), never an ownership claim
- Recipe-stage semantics are definition-level by design: a composite expands into
  its flat stage list before any plan validation or execution — the engine never
  sees `recipe` stages, and inline `WorkflowPlan`s reject them. If a future
  milestone wanted runtime-selected recipe invocation it would be new machinery
  (deliberately out of scope)
- M15 generation is synchronous and CPU-first, single-request decode with an
  incremental KV cache; there is no batching, streaming, queueing or
  background execution, and no GPU path is required (the engine uses CUDA
  only when the host reports one, mirroring M3/M4)
- Determinism is exact within an environment: identical requests reproduce
  byte-identical manifests because decoding is fp32, single-threaded per
  request, seeded by the explicit integer seed (CPU `torch.Generator`) and
  greedy uses no RNG at all. Cross-environment byte-identity additionally
  depends on the same torch/tokenizers versions (hardware/manifest fields are
  recorded but excluded from `result_hash`); floating-point decoding is
  deterministic on the same binary, never claimed bit-perfect across builds
- The decode vocabulary is the TOKENIZER's actual vocab (the platform
  convention: model vocab_size >= tokenizer vocab). If a tokenizer's actual
  vocab is smaller than the model's, generation never emits ids the tokenizer
  cannot decode — but the equality clause sometimes imagined for sampling is
  not enforced here because no production artifact of this platform uses an
  equal-vocab pair (models are created with headroom); enforcing equality
  would make every existing checkpoint unsampleable. This is the documented,
  tested M15 rule
- No EOS policy: decode always produces exactly `max_new_tokens` tokens and
  the special ids (`<unk>`/`<s>`/`</s>`) decode to empty strings if the model
  emits them — the record keeps raw ids so nothing is ever lost; prompts and
  outputs are not trimmed, filtered or post-processed in any way
- M15 never judges output: no perplexity, scores, safety/hallucination
  checks, rankings or feedback loops exist anywhere; quality evaluation of
  generated text is explicitly a later milestone. `output_text` is raw
  tokenizer decode (the documented M2 round-trip caveat about raw control
  bytes applies to prompts/outputs)
- Prompt text is stored verbatim and encoded verbatim (never stripped,
  never auto-completed with special tokens); `max_new_tokens` is capped at
  512 by schema AND at `context_length - prompt_tokens` by preflight — the
  effective per-request cap is therefore model-dependent and documented in
  the rejection message rather than silently truncated
