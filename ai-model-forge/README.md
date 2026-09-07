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

## Quickstart

```bash
pip install -e ".[dev]"
pytest                       # 480 tests
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
    evaluation.py      # evaluation engine: read-only state measurement (M4)
                       # + read-only by-checkpoint/by-dataset/by-tokenizer grouping (M24/M28/M30)
    comparison.py      # comparison engine: A/B states over identical probes (M5)
                       # + read-only by-checkpoint/by-dataset/by-tokenizer grouping (M26/M29/M31)
    gates.py           # stage gates: policy-driven run decisions (M6)
                       # + read-only by-policy/by-comparison grouping (M23/M34)
    workflows.py       # workflow engine: ordered orchestration over M3-M6 (M7)
    dashboards.py      # read-only dashboard engine: deterministic views (M8/M13)
    policies.py        # policy registry + probe suites: immutable definitions (M9)
    suite_runs.py      # explicit multi-probe M4 batches over named suites (M10)
                       # + read-only by-suite/by-checkpoint grouping (M21/M22/M25)
    recipes.py         # workflow recipes: immutable plans + M14 composition (M12/M14)
    sampling.py         # checkpoint sampling: deterministic generation (M15)
                       # + read-only by-checkpoint grouping of the history (M27)
    sample_quality.py  # per-sample likelihood measurement of samples (M16)
                       # + read-only by-sample/by-checkpoint/by-tokenizer grouping (M19/M20/M33)
    engine.py          # facade composing all engines
    api.py             # FastAPI routes (thin)
  tests/               # 480 tests across 20 suites
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
  only records `suggested_checkpoint_id` + a hint. The user (or a future orchestrator
  milestone) must call the M3 rollback endpoint explicitly — nothing is automatic
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
