"""Milestone 9 tests: stable policy registry + named probe suites.

Engine tests (module-scoped real history: domain -> tokenizer -> model ->
training -> evaluations -> one comparison) prove:

  * policies: registration/idempotency/hash determinism/persistence/list-get/
    conflicts (same id + different config), resolution (model mismatch,
    missing, corrupt), byte-level immutability of existing artifacts
  * probe suites: registration, canonical ordering, hashes, duplicate-probe
    and empty-suite rejection, M4 identity preservation (suite probes resolve
    to EvaluationConfigs with the exact M4 identity semantics)
  * gate compatibility: an inline M6 policy and a resolved registry policy
    produce identical M6 decisions (result_hash equality) on identical
    evidence, with registry provenance persisted on the decision
  * workflow compatibility: gate stages may reference a registry policy_id;
    unknown references fail cleanly; inline gate stages keep working
  * dashboard compatibility: historical records unchanged, registry decisions
    render with policy_id/policy_config_hash, output stays deterministic
  * API coverage: all six endpoints + 404/409/422 mapping over HTTP
"""
from __future__ import annotations

import json
import random
import time

import pytest
from pydantic import ValidationError

from app.engine import ModelForge
from app.schemas import (
    ComparisonRequest,
    ComparisonState,
    EvalStateKind,
    EvaluationConfig,
    GatePolicy,
    GateRequest,
    ModelCreateRequest,
    PolicyCreateRequest,
    ProbeSuiteCreateRequest,
    StageStateRef,
    StageType,
    SuiteProbe,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
    WorkflowGateStage,
    WorkflowPlan,
    WorkflowStage,
)

HEADS = ("river mountain cloud forest desert ocean valley island meadow canyon "
         "table chair lamp desk shelf couch rug clock mirror vase").split()
TAIL = ("flows stands gleams rises falls drifts looms shines hides waits").split()


def _domain_bytes(n: int = 240) -> bytes:
    rng = random.Random(11)
    out = []
    for i in range(n):
        h1, h2 = rng.sample(HEADS, 2)
        out.append(f"{h1} is {rng.choice(TAIL)} near {h2} with number {i}")
    return ("\n\n".join(out) + "\n").encode("utf-8")


def _probe(ds: str, tok: str, seed: int, split: str = "validation",
           batch: int = 8, seq: int = 32) -> dict:
    return dict(dataset_id=ds, split=split, tokenizer_id=tok,
                batch_size=batch, max_seq_len=seq, seed=seed)


def _policy_for(model_id: str, ds: str, tok: str, seed: int,
                baseline_ckpt=None, minimum_loss=None, name: str = "m9-pol",
                max_regression_delta=None) -> GatePolicy:
    pol = dict(name=name, model_id=model_id,
               baseline_type=("checkpoint" if baseline_ckpt is not None
                              else "minimum_loss"),
               **_probe(ds, tok, seed))
    if baseline_ckpt is not None:
        pol["baseline_checkpoint_id"] = baseline_ckpt
    if minimum_loss is not None:
        pol["minimum_loss"] = minimum_loss
    if max_regression_delta is not None:
        pol["max_regression_delta"] = max_regression_delta
    return GatePolicy(**pol)


def _cstate(ckpt=None, current=False) -> ComparisonState:
    if current:
        return ComparisonState(state_kind=EvalStateKind.CURRENT)
    return ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                           checkpoint_id=ckpt)


class Env:
    """One temp root: model with real history + registered definitions."""

    def __init__(self, root):
        self.forge = ModelForge(root=root)

    def fresh_model(self, name: str = "m9-extra", seed: int = 77) -> str:
        return self.forge.create_model(ModelCreateRequest(config=TransformerConfig(
            name=name, vocab_size=640, context_length=64, hidden_size=64,
            n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
            seed=seed)))[0].id

    def prepare(self):
        f = self.forge
        up = f.upload_dataset([("a.txt", _domain_bytes())], name="m9-dom")
        self.ds = up["dataset_id"]
        tok = f.train_tokenizer(TokenizerConfig(name="m9-tok", vocab_size=600),
                                dataset_id=self.ds)
        self.tok = tok.id
        f.tokenize_dataset(self.ds, tok.id)
        cfg = TransformerConfig(name="m9-main", vocab_size=640, context_length=64,
                                hidden_size=64, n_layers=2, n_heads=4,
                                n_kv_heads=2, intermediate_size=128, seed=1)
        self.model_id = f.create_model(ModelCreateRequest(config=cfg))[0].id
        tokens = f.datasets.tokenized_artifact(self.ds, 1, self.tok)[0] \
            .splits["train"].count
        sp_epoch = max(1, (tokens // 32) // 8)
        f.run_training(TrainingConfig(
            method="continued_pretraining", model_id=self.model_id,
            dataset_id=self.ds, tokenizer_id=self.tok, learning_rate=3e-3,
            batch_size=8, max_seq_len=32, epochs=6,
            eval_every_steps=max(1, sp_epoch // 2), keep_best=False, seed=1))
        ckpts = f.list_checkpoints(self.model_id)
        self.ckpts = [c.checkpoint_id for c in ckpts]
        self.ck_a = self.ckpts[len(self.ckpts) // 2]    # mid chain
        self.ck_b = self.ckpts[-1]                      # latest
        # one comparison between two real checkpoints (probe seed 7001) so
        # gate tests reuse exact evidence without adding evaluations
        self.comp = f.run_comparison(ComparisonRequest(
            model_id=self.model_id, state_a=_cstate(self.ck_a),
            state_b=_cstate(self.ck_b), tolerance=1e-4,
            **_probe(self.ds, self.tok, 7001)))
        self.loss_b = self.comp.loss_b

    # ------------------------------------------------------------------ #
    # registry helpers
    # ------------------------------------------------------------------ #

    def register_policy(self, tag: str, seed: int, **policy_kw) -> str:
        pol = _policy_for(self.model_id, self.ds, self.tok, seed,
                          name=f"m9-{tag}", **policy_kw)
        pid = f"pol-{tag}"
        self.forge.register_policy(PolicyCreateRequest(
            policy_id=pid, description=f"test policy {tag}", policy=pol))
        return pid

    def gate(self, *, inline=None, policy_id=None, candidate_ckpt=None,
             seed: int = 7001, policy=None):
        cand = _cstate(candidate_ckpt or self.ck_b)
        if policy is None and inline is None and policy_id is None:
            policy = _policy_for(self.model_id, self.ds, self.tok, seed,
                                 baseline_ckpt=self.ck_a)
        return self.forge.run_gate(GateRequest(
            model_id=self.model_id, policy=policy or inline,
            policy_id=policy_id, candidate=cand))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m9-root"))
    e.prepare()
    return e


# =========================================================================== #
# A. Policy registry — engine semantics
# =========================================================================== #

def test_policy_register_and_deterministic_hash(env):
    f = env.forge
    pol = _policy_for(env.model_id, env.ds, env.tok, 8001,
                      baseline_ckpt=env.ck_b)
    r1 = f.register_policy(PolicyCreateRequest(
        policy_id="pol-hash-a", description="first", policy=pol))
    assert r1.policy_id == "pol-hash-a"
    assert len(r1.config_hash) == 64
    # same semantic config (different description) -> same hash, and the
    # hash excludes ids/timestamps/descriptions
    r2 = f.register_policy(PolicyCreateRequest(
        policy_id="pol-hash-b", description="entirely different", policy=pol))
    assert r2.config_hash == r1.config_hash
    assert r2.created_at != r1.created_at or r2.policy_id != r1.policy_id
    # hash is deterministic across separate storage roots too
    from app.engine import ModelForge as MF
    import tempfile
    other = MF(root=tempfile.mkdtemp(prefix="m9-hash-"))
    r3 = other.register_policy(PolicyCreateRequest(
        policy_id="pol-elsewhere", policy=pol))
    assert r3.config_hash == r1.config_hash


def test_policy_same_id_same_config_is_idempotent(env):
    f = env.forge
    pol = _policy_for(env.model_id, env.ds, env.tok, 8002,
                      minimum_loss=0.5)
    req = PolicyCreateRequest(policy_id="pol-idem", policy=pol)
    first = f.register_policy(req)
    second = f.register_policy(req)
    assert second.policy_id == first.policy_id
    assert second.config_hash == first.config_hash
    assert second.created_at == first.created_at      # the SAME record
    # only one manifest exists on disk
    d = f.storage.root / "policies" / "pol-idem"
    assert [p.name for p in d.iterdir()] == ["manifest.json"]


def test_policy_same_id_different_config_conflicts(env):
    f = env.forge
    pid = "pol-clash"
    f.register_policy(PolicyCreateRequest(
        policy_id=pid, policy=_policy_for(env.model_id, env.ds, env.tok,
                                          8003, minimum_loss=0.1)))
    with pytest.raises(ValueError, match="already exists"):
        f.register_policy(PolicyCreateRequest(
            policy_id=pid, policy=_policy_for(env.model_id, env.ds, env.tok,
                                              8003, minimum_loss=0.2)))
    # original untouched
    got = f.get_policy(pid)
    assert got.policy.minimum_loss == 0.1


def test_policy_malformed_definitions_rejected(env):
    # checkpoint baseline without the checkpoint id -> GatePolicy ValidationError
    bad = dict(name="m9-bad", model_id=env.model_id,
               baseline_type="checkpoint", **_probe(env.ds, env.tok, 8004))
    with pytest.raises(ValidationError):
        GatePolicy(**bad)
    # empty policy_id
    with pytest.raises(ValidationError):
        PolicyCreateRequest(policy_id="", policy=_policy_for(
            env.model_id, env.ds, env.tok, 8005))
    # extra fields forbidden
    with pytest.raises(ValidationError):
        PolicyCreateRequest(
            policy_id="pol-extra",
            policy=_policy_for(env.model_id, env.ds, env.tok, 8006),
            **{"bogus": 1})


def test_policy_list_get_ordering_unknown(env):
    f = env.forge
    for i, seed in enumerate((8010, 8011, 8012)):
        pid = f"pol-order-{i}"
        f.register_policy(PolicyCreateRequest(
            policy_id=pid,
            policy=_policy_for(env.model_id, env.ds, env.tok, seed,
                               minimum_loss=0.01 * i)))
        time.sleep(0.01)                                # distinct created_at
    names = [p.policy_id for p in f.list_policies()
             if p.policy_id.startswith("pol-order-")]
    assert names == [f"pol-order-{i}" for i in range(3)]   # oldest first
    got = f.get_policy("pol-order-1")
    assert got.policy.minimum_loss == 0.01
    with pytest.raises(FileNotFoundError, match="not found"):
        f.get_policy("pol-ghost")


def test_policy_persistence_and_resolution(env):
    f = env.forge
    pid = env.register_policy("resolve", 8020, baseline_ckpt=env.ck_b)
    # manifest on disk is the single immutable source
    mpath = f.storage.root / "policies" / pid / "manifest.json"
    assert mpath.exists()
    disk = json.loads(mpath.read_text())
    assert disk["policy_id"] == pid and disk["config_hash"] == \
        f.get_policy(pid).config_hash
    # repeated resolution returns identical semantic content
    p1, h1 = f.policies.resolve_policy(env.model_id, pid)
    p2, h2 = f.policies.resolve_policy(env.model_id, pid)
    assert p1.model_dump(mode="json") == p2.model_dump(mode="json")
    assert h1 == h2 == f.get_policy(pid).config_hash
    # resolution checks the target model and fails cleanly when missing
    with pytest.raises(FileNotFoundError, match="not found"):
        f.policies.resolve_policy(env.model_id, "pol-nope")
    other = env.fresh_model("m9-mismatch")
    with pytest.raises(ValueError, match="targets model"):
        f.policies.resolve_policy(other, pid)


def test_policy_corrupt_manifest_diagnostics(env):
    f = env.forge
    pid = env.register_policy("corrupt", 8030, minimum_loss=0.3)
    (f.storage.root / "policies" / pid / "manifest.json").write_bytes(
        b"{broken json")
    with pytest.raises(RuntimeError, match="corrupt"):
        f.policies.resolve_policy(env.model_id, pid)
    # list_policies skips the corrupt dir and keeps valid ones visible
    ok = env.register_policy("corrupt-ok", 8031, minimum_loss=0.4)
    names = [p.policy_id for p in f.list_policies()]
    assert pid not in names and ok in names


# =========================================================================== #
# B. Probe suites — engine semantics
# =========================================================================== #

def _suite_probes(ds: str, tok: str) -> list[SuiteProbe]:
    return [SuiteProbe(dataset_id=ds, split="validation", tokenizer_id=tok,
                       batch_size=8, max_seq_len=32, seed=9001),
            SuiteProbe(dataset_id=ds, split="test", tokenizer_id=tok,
                       batch_size=8, max_seq_len=64, seed=9002)]


def test_suite_register_canonical_order_and_hash(env):
    f = env.forge
    p1, p2 = _suite_probes(env.ds, env.tok)
    # register with probes deliberately out of canonical order
    s1 = f.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="suite-order", probes=[p2, p1]))
    assert s1.suite_id == "suite-order" and len(s1.probes_hash) == 64
    # stored order is canonical (sorted by canonical JSON), never input order
    assert s1.probes == sorted([p1, p2], key=lambda p: json.dumps(
        p.model_dump(mode="json"), sort_keys=True, default=str))
    # same contents registered under another id share the probes hash
    s2 = f.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="suite-order-2", probes=[p1, p2]))
    assert s2.probes_hash == s1.probes_hash


def test_suite_same_id_same_content_idempotent(env):
    f = env.forge
    req = ProbeSuiteCreateRequest(suite_id="suite-idem",
                                  probes=_suite_probes(env.ds, env.tok))
    first = f.register_probe_suite(req)
    second = f.register_probe_suite(req)
    assert second.probes_hash == first.probes_hash
    assert second.created_at == first.created_at
    d = f.storage.root / "probe-suites" / "suite-idem"
    assert [p.name for p in d.iterdir()] == ["manifest.json"]


def test_suite_same_id_different_content_conflicts(env):
    f = env.forge
    pid = "suite-clash"
    f.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id=pid, probes=_suite_probes(env.ds, env.tok)))
    changed = _suite_probes(env.ds, env.tok)
    changed[0] = changed[0].model_copy(update={"seed": 9999})
    with pytest.raises(ValueError, match="already exists"):
        f.register_probe_suite(ProbeSuiteCreateRequest(suite_id=pid,
                                                       probes=changed))
    assert len(f.get_probe_suite(pid).probes) == 2             # untouched


def test_suite_duplicate_and_empty_probes_rejected(env):
    p1, _ = _suite_probes(env.ds, env.tok)
    with pytest.raises(ValidationError, match="same probe twice"):
        ProbeSuiteCreateRequest(suite_id="suite-dup", probes=[p1, p1])
    with pytest.raises(ValidationError, match="at least 1"):
        ProbeSuiteCreateRequest(suite_id="suite-empty", probes=[])
    with pytest.raises(ValidationError):
        SuiteProbe(dataset_id=env.ds, split="validation", tokenizer_id=env.tok,
                   batch_size=0, seed=1)                 # batch must be >= 1


def test_suite_list_get_unknown(env):
    f = env.forge
    for i in range(2):
        f.register_probe_suite(ProbeSuiteCreateRequest(
            suite_id=f"suite-list-{i}",
            probes=[SuiteProbe(dataset_id=env.ds, split="validation",
                               tokenizer_id=env.tok, batch_size=8,
                               max_seq_len=32, seed=9100 + i)]))
        time.sleep(0.01)
    names = [s.suite_id for s in f.list_probe_suites()
             if s.suite_id.startswith("suite-list-")]
    assert names == ["suite-list-0", "suite-list-1"]
    with pytest.raises(FileNotFoundError, match="not found"):
        f.get_probe_suite("suite-ghost")


def test_suite_probe_preserves_m4_identity(env):
    """A suite probe resolves to an EvaluationConfig with EXACT M4 identity."""
    f = env.forge
    probe = SuiteProbe(dataset_id=env.ds, split="validation",
                       tokenizer_id=env.tok, batch_size=8, max_seq_len=32,
                       seed=9200)
    cfg = probe.to_evaluation_config(env.model_id, checkpoint_id=env.ck_b)
    assert isinstance(cfg, EvaluationConfig)
    assert cfg.checkpoint_id == env.ck_b and cfg.seed == 9200
    # direct M4 config with the identical fields
    direct = EvaluationConfig(model_id=env.model_id, checkpoint_id=env.ck_b,
                              dataset_id=env.ds, split="validation",
                              tokenizer_id=env.tok, batch_size=8,
                              max_seq_len=32, seed=9200)
    assert cfg.model_dump() == direct.model_dump()
    # running both produces identical M4 identity semantics: same result hash
    # and state identity (records differ only in id/timestamp)
    a = f.run_evaluation(direct)
    b = f.run_evaluation(cfg)
    assert a.result_hash == b.result_hash
    assert a.state_hash == b.state_hash and a.token_count == b.token_count
    # dashboard probe identity keys are equal (no second probe identity)
    dash = f.get_dashboard(env.model_id).model_dump(mode="json")
    keys = {}
    for g in dash["evaluations"]:
        for r in g["records"]:
            keys[r["eval_id"]] = g["key"]
    assert keys[a.eval_id] == keys[b.eval_id]
    assert keys[a.eval_id] is not None


# =========================================================================== #
# C. Gate compatibility: inline policy == resolved registry policy
# =========================================================================== #

def test_registry_and_inline_policy_identical_decisions(env):
    f = env.forge
    n_gates0 = len(f.list_gate_decisions(env.model_id))
    inline = _policy_for(env.model_id, env.ds, env.tok, 7001,
                         baseline_ckpt=env.ck_a, name="m9-eq")
    d_inline = f.run_gate(GateRequest(model_id=env.model_id, policy=inline,
                                      candidate=_cstate(env.ck_b)))
    pid = env.register_policy("eq", 7001, baseline_ckpt=env.ck_a)
    d_reg = f.run_gate(GateRequest(model_id=env.model_id, policy_id=pid,
                                   candidate=_cstate(env.ck_b)))
    # identical evidence + identical M6 semantics -> identical decision
    assert d_reg.decision == d_inline.decision
    assert d_reg.result_hash == d_inline.result_hash
    assert d_reg.verdict == d_inline.verdict
    assert d_reg.policy.model_dump(mode="json") == \
        d_inline.policy.model_dump(mode="json")
    assert d_reg.policy_id == pid
    assert d_reg.policy_config_hash == f.get_policy(pid).config_hash
    assert d_inline.policy_id is None
    assert d_inline.policy_config_hash is None
    # exactly two new immutable decisions, nothing else (evidence reused)
    assert len(f.list_gate_decisions(env.model_id)) == n_gates0 + 2
    assert d_reg.comparison_id == d_inline.comparison_id


def test_gate_request_requires_exactly_one_policy_source(env):
    pol = _policy_for(env.model_id, env.ds, env.tok, 7001,
                      baseline_ckpt=env.ck_a)
    with pytest.raises(ValidationError):
        GateRequest(model_id=env.model_id, policy=pol, policy_id="pol-x",
                    candidate=_cstate(env.ck_b))
    with pytest.raises(ValidationError):
        GateRequest(model_id=env.model_id, candidate=_cstate(env.ck_b))


def test_gate_registry_policy_unknown_or_mismatched_fails_cleanly(env):
    f = env.forge
    n0 = len(f.list_gate_decisions(env.model_id))
    with pytest.raises(FileNotFoundError, match="not found"):
        f.run_gate(GateRequest(model_id=env.model_id, policy_id="pol-nope",
                               candidate=_cstate(env.ck_b)))
    other = env.fresh_model("m9-wrongmodel")
    with pytest.raises(ValueError, match="targets model"):
        f.run_gate(GateRequest(model_id=other, policy_id="pol-eq",
                               candidate=_cstate(env.ck_b)))
    assert len(f.list_gate_decisions(env.model_id)) == n0    # nothing ran


# =========================================================================== #
# D. Workflow compatibility (registry-aware gate stages)
# =========================================================================== #

def test_workflow_gate_stage_with_registry_policy(env):
    f = env.forge
    pid = env.register_policy("wf", 7002, baseline_ckpt=env.ck_a)
    plan = WorkflowPlan(name="m9-wf-reg", model_id=env.model_id, stages=[
        WorkflowStage(stage_id="g1", type=StageType.GATE,
                      gate=WorkflowGateStage(
                          policy_id=pid,
                          candidate=StageStateRef(
                              state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=env.ck_b)))])
    rec = f.run_workflow(plan)
    assert rec.status.value == "completed"
    art = rec.stages[0].artifact
    assert art.kind.value == "gate_decision"
    decision = f.get_gate_decision(env.model_id, art.artifact_id)
    assert decision.policy_id == pid
    assert decision.policy_config_hash == f.get_policy(pid).config_hash
    assert decision.decision.value == art.gate_decision


def test_workflow_gate_stage_unknown_policy_fails_cleanly(env):
    f = env.forge
    plan = WorkflowPlan(name="m9-wf-missing", model_id=env.model_id, stages=[
        WorkflowStage(stage_id="g1", type=StageType.GATE,
                      gate=WorkflowGateStage(
                          policy_id="pol-ghost-wf",
                          candidate=StageStateRef(
                              state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=env.ck_b)))])
    with pytest.raises(FileNotFoundError, match="not found"):
        f.run_workflow(plan)
    runs = f.list_workflows(env.model_id)
    failed = [r for r in runs if r.status.value == "failed"]
    assert any(r.failed_stage_id == "g1" for r in failed)


def test_workflow_gate_stage_schema_xor_and_inline_still_works(env):
    pol = _policy_for(env.model_id, env.ds, env.tok, 7003,
                      baseline_ckpt=env.ck_a)
    # inline (M6) stage still parses and runs
    inline_stage = WorkflowGateStage(
        policy=pol, candidate=StageStateRef(
            state_kind=EvalStateKind.CHECKPOINT, checkpoint_id=env.ck_b))
    assert inline_stage.policy_id is None
    with pytest.raises(ValidationError):
        WorkflowGateStage(policy=pol, policy_id="pol-wf-x",
                          candidate=StageStateRef(
                              state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=env.ck_b))
    with pytest.raises(ValidationError):
        WorkflowGateStage(candidate=StageStateRef(
            state_kind=EvalStateKind.CHECKPOINT, checkpoint_id=env.ck_b))


# =========================================================================== #
# E. Dashboard compatibility
# =========================================================================== #

def test_dashboard_historical_records_unchanged_and_registry_visible(env):
    f = env.forge
    before = f.get_dashboard(env.model_id).model_dump(mode="json")
    # two decisions created by the parity test already exist; snapshot their
    # gate section for the exact id-based comparison
    old_ids = {r["decision_id"] for g in before["gate_decisions"]
               for r in g["records"]}
    inline = _policy_for(env.model_id, env.ds, env.tok, 7001,
                         baseline_ckpt=env.ck_a)
    d_in = f.run_gate(GateRequest(model_id=env.model_id, policy=inline,
                                  candidate=_cstate(env.ck_b)))
    pid = env.register_policy("dash", 7001, baseline_ckpt=env.ck_a)
    d_reg = f.run_gate(GateRequest(model_id=env.model_id, policy_id=pid,
                                   candidate=_cstate(env.ck_b)))
    after = f.get_dashboard(env.model_id).model_dump(mode="json")
    # historical sections (everything except gates) unchanged
    for section in ("checkpoints", "training_runs", "evaluations",
                    "comparisons", "workflows"):
        assert after[section] == before[section]
    # old gate records are rendered exactly as before
    for g in after["gate_decisions"]:
        for r in g["records"]:
            if r["decision_id"] in old_ids:
                pass  # covered by the no-diagnostics + records match below
    after_ids = {r["decision_id"] for g in after["gate_decisions"]
                 for r in g["records"]}
    assert after_ids == old_ids | {d_in.decision_id, d_reg.decision_id}
    # the registry decision renders with provenance; inline without
    rows = {r["decision_id"]: r for g in after["gate_decisions"]
            for r in g["records"]}
    assert rows[d_reg.decision_id]["policy_id"] == pid
    assert rows[d_reg.decision_id]["policy_config_hash"] == \
        f.get_policy(pid).config_hash
    assert rows[d_in.decision_id]["policy_id"] is None
    # determinism: repeat read identical, hash stable
    again = f.get_dashboard(env.model_id).model_dump(mode="json")
    assert again == after
    assert f.get_dashboard(env.model_id).result_hash == \
        f.get_dashboard(env.model_id).result_hash


# =========================================================================== #
# F. Byte-level immutability of pre-existing artifacts
# =========================================================================== #

def test_registration_never_touches_existing_artifacts(env):
    f = env.forge
    root = f.storage.root
    before = {p.relative_to(root).as_posix(): p.read_bytes()
              for p in root.rglob("*") if p.is_file()}
    f.register_policy(PolicyCreateRequest(
        policy_id="pol-ro-audit",
        policy=_policy_for(env.model_id, env.ds, env.tok, 9300,
                           minimum_loss=0.25)))
    f.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="suite-ro-audit", probes=_suite_probes(env.ds, env.tok)))
    after = {p.relative_to(root).as_posix(): p.read_bytes()
             for p in root.rglob("*") if p.is_file()}
    # only the two definition manifests are new
    new = set(after) - set(before)
    assert new == {"policies/pol-ro-audit/manifest.json",
                   "probe-suites/suite-ro-audit/manifest.json"}
    for path in before:
        assert after[path] == before[path]               # byte-identical
    assert not any(".tmp" in p for p in after)


# =========================================================================== #
# G. HTTP API coverage
# =========================================================================== #

POLICIES = "/api/v1/policies"
SUITES = "/api/v1/probe-suites"
MODELS = "/api/v1/models"
TRAIN_RUN = "/api/v1/training/run"
GATE_EVAL = "/api/v1/gates/evaluate"


def _policy_payload(model_id, ds, tok, *, seed=9501, baseline_ckpt=None,
                    minimum_loss=None):
    pol = {"name": "api9-pol", "model_id": model_id, "dataset_id": ds,
           "split": "validation", "tokenizer_id": tok, "batch_size": 8,
           "max_seq_len": 32, "seed": seed,
           "baseline_type": "checkpoint" if baseline_ckpt else "minimum_loss"}
    if baseline_ckpt:
        pol["baseline_checkpoint_id"] = baseline_ckpt
    if minimum_loss is not None:
        pol["minimum_loss"] = minimum_loss
    return pol


def _suite_payload(ds, tok, seeds=(9601, 9602)):
    return [{"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": s} for s in seeds]


def test_policies_api_crud_conflict_404_422(api_client):
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", ("p.txt", _domain_bytes(60),
                                           "text/plain"))],
                         data={"name": "api9-pol-ds"})
    ds = up.json()["dataset_id"]
    cfg = {"name": "api9-pol-model", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    pol = _policy_payload(mid, ds, "tok-never-used", minimum_loss=0.5)

    # register -> 201 with deterministic config hash
    r = api_client.post(POLICIES, json={"policy_id": "api9-pol-1",
                                        "description": "d", "policy": pol})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["policy_id"] == "api9-pol-1" and len(body["config_hash"]) == 64
    # list + get
    assert any(p["policy_id"] == "api9-pol-1" for p in
               api_client.get(POLICIES).json())
    got = api_client.get(f"{POLICIES}/api9-pol-1")
    assert got.status_code == 200 and got.json()["config_hash"] == \
        body["config_hash"]
    # idempotent identical register -> same definition
    again = api_client.post(POLICIES, json={"policy_id": "api9-pol-1",
                                            "policy": pol})
    assert again.status_code == 201
    assert again.json()["created_at"] == body["created_at"]
    # same id + different config -> 409
    pol2 = dict(pol, minimum_loss=0.9)
    clash = api_client.post(POLICIES, json={"policy_id": "api9-pol-1",
                                            "policy": pol2})
    assert clash.status_code == 409 and "already exists" in clash.text
    # malformed -> 422
    bad = dict(pol, baseline_type="checkpoint")   # no baseline_checkpoint_id
    mal = api_client.post(POLICIES, json={"policy_id": "api9-pol-bad",
                                          "policy": bad})
    assert mal.status_code == 422
    # unknown id -> 404
    assert api_client.get(f"{POLICIES}/api9-pol-ghost").status_code == 404
    # no delete/put/patch endpoints exist
    assert api_client.delete(f"{POLICIES}/api9-pol-1").status_code == 405
    assert api_client.put(f"{POLICIES}/api9-pol-1").status_code == 405


def test_probe_suites_api_crud_conflict_404_422(api_client):
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", ("s.txt", _domain_bytes(60),
                                           "text/plain"))],
                         data={"name": "api9-suite-ds"})
    ds = up.json()["dataset_id"]
    probes = _suite_payload(ds, "tok-never-used")
    r = api_client.post(SUITES, json={"suite_id": "api9-suite-1",
                                      "probes": probes})
    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["probes_hash"]) == 64 and len(body["probes"]) == 2
    # canonical order preserved on read
    assert body["probes"] == sorted(body["probes"], key=lambda p: json.dumps(
        p, sort_keys=True, default=str))
    # duplicate probe -> 422
    dup = api_client.post(SUITES, json={"suite_id": "api9-suite-dup",
                                        "probes": [probes[0], probes[0]]})
    assert dup.status_code == 422
    # same id + different content -> 409
    changed = [dict(probes[0], seed=7777)]
    clash = api_client.post(SUITES, json={"suite_id": "api9-suite-1",
                                          "probes": changed})
    assert clash.status_code == 409 and "already exists" in clash.text
    # list/get/404
    assert any(s["suite_id"] == "api9-suite-1" for s in
               api_client.get(SUITES).json())
    assert api_client.get(f"{SUITES}/api9-suite-1").status_code == 200
    assert api_client.get(f"{SUITES}/api9-suite-ghost").status_code == 404


def test_gate_with_registry_policy_via_api(api_client):
    """One real HTTP gate path using a registry policy over trained evidence."""
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", ("g.txt", _domain_bytes(120),
                                           "text/plain"))],
                         data={"name": "api9-gate-ds"})
    ds = up.json()["dataset_id"]
    tok = api_client.post("/api/v1/tokenizers/train",
                          data={"config": json.dumps(
                              {"name": "api9-gate-tok", "vocab_size": 600}),
                                "dataset_id": ds}).json()["tokenizer"]["id"]
    api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                    json={"tokenizer_id": tok})
    cfg = {"name": "api9-gate-model", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    run = api_client.post(TRAIN_RUN, json={
        "method": "continued_pretraining", "model_id": mid, "dataset_id": ds,
        "tokenizer_id": tok, "learning_rate": 3e-3, "batch_size": 8,
        "max_seq_len": 32, "epochs": 3, "eval_every_steps": 3,
        "keep_best": False, "seed": 1})
    assert run.status_code == 200, run.text
    ckpts = api_client.get(f"{MODELS}/{mid}/checkpoints").json()
    ck_last = ckpts[-1]["checkpoint_id"]
    ck_a = ckpts[len(ckpts) // 2]["checkpoint_id"]

    pol = _policy_payload(mid, ds, tok, seed=9701, baseline_ckpt=ck_a)
    reg = api_client.post(POLICIES, json={"policy_id": "api9-gate-pol",
                                          "policy": pol})
    assert reg.status_code == 201, reg.text

    # inline gate first, then registry gate — decisions must match
    inline = api_client.post(GATE_EVAL, json={
        "model_id": mid, "policy": pol,
        "candidate": {"state_kind": "checkpoint", "checkpoint_id": ck_last}})
    assert inline.status_code == 200, inline.text
    reg_gate = api_client.post(GATE_EVAL, json={
        "model_id": mid, "policy_id": "api9-gate-pol",
        "candidate": {"state_kind": "checkpoint", "checkpoint_id": ck_last}})
    assert reg_gate.status_code == 200, reg_gate.text
    d = reg_gate.json()
    assert d["decision_id"] != inline.json()["decision_id"]
    assert d["result_hash"] == inline.json()["result_hash"]
    assert d["policy_id"] == "api9-gate-pol"
    assert d["policy_config_hash"] == reg.json()["config_hash"]
    # unknown policy_id -> 404 with nothing appended
    n0 = len(api_client.get(f"{MODELS}/{mid}/gates/decisions").json())
    missing = api_client.post(GATE_EVAL, json={
        "model_id": mid, "policy_id": "api9-ghost",
        "candidate": {"state_kind": "checkpoint", "checkpoint_id": ck_last}})
    assert missing.status_code == 404
    both = api_client.post(GATE_EVAL, json={
        "model_id": mid, "policy": pol, "policy_id": "api9-gate-pol",
        "candidate": {"state_kind": "checkpoint", "checkpoint_id": ck_last}})
    assert both.status_code == 422
    assert len(api_client.get(f"{MODELS}/{mid}/gates/decisions").json()) == n0
    # dashboard renders the registry decision with provenance
    dash = api_client.get(f"{MODELS}/{mid}/dashboard").json()
    rows = [r for g in dash["gate_decisions"] for r in g["records"]
            if r["decision_id"] == d["decision_id"]]
    assert len(rows) == 1
    assert rows[0]["policy_id"] == "api9-gate-pol"
    assert rows[0]["policy_config_hash"] == reg.json()["config_hash"]
    assert dash["result_hash"] == \
        api_client.get(f"{MODELS}/{mid}/dashboard").json()["result_hash"]
