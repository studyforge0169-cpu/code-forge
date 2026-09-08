"""Milestone 12 tests: named workflow recipes (immutable reusable M7 plans).

Engine tests use a module-scoped environment (one dataset + tokenizer + a
trained main model with literal checkpoints, plus small dedicated models for
stages that train) mirroring the M7/M11 test style; API coverage runs over
the shared HTTP TestClient root with per-test model setup (no training
needed — recipes bind current state for HTTP flows).

Covered: registration semantics + M7 validation parity (identical shared
rules/messages), immutability (idempotent / 409 / 405 / byte-level),
retrieval, explicit model binding, current/checkpoint/earlier-train-stage
state semantics with snapshot provenance, recipe provenance on run records
(inline runs stay null), suite_run integration through the existing M10
engine, evidence reuse without duplicate manifests, failure semantics
(unknown recipe/model -> nothing persisted; runtime failures -> failed run
record with provenance, no fabricated artifacts), and dashboard
determinism. M1-M11 behavior is untouched.
"""
from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from app.recipes import RecipeEngine, recipe_config_hash
from app.schemas import (
    EvalStateKind,
    EvaluationConfig,
    GateBaselineType,
    GatePolicy,
    ModelCreateRequest,
    ProbeSuiteCreateRequest,
    StageStateRef,
    StageType,
    SuiteProbe,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
    WorkflowComparisonStage,
    WorkflowEvaluationStage,
    WorkflowGateStage,
    WorkflowPlan,
    WorkflowRecipeCreateRequest,
    WorkflowStage,
    WorkflowStatus,
    WorkflowSuiteRunStage,
)

HEADS = ("river mountain cloud forest desert ocean valley island meadow "
         "canyon").split()
TAIL_A = ("flows stands gleams rises falls drifts looms shines hides "
          "waits").split()


def _domain_bytes(tails, n: int) -> bytes:
    rng = random.Random(11)
    out = []
    for i in range(n):
        h1, h2 = rng.sample(HEADS, 2)
        out.append(f"{h1} is {rng.choice(tails)} near {h2} with number {i}")
    return ("\n\n".join(out) + "\n").encode("utf-8")


def _tiny(name: str, seed: int = 1) -> TransformerConfig:
    return TransformerConfig(name=name, vocab_size=640, context_length=64,
                             hidden_size=64, n_layers=2, n_heads=4,
                             n_kv_heads=2, intermediate_size=128, seed=seed)


class Env:
    """One temp root: domain-A dataset, tokenizer, trained main model."""

    def __init__(self, root):
        from app.engine import ModelForge

        self.forge = ModelForge(root=root)

    def prepare(self):
        f = self.forge
        up = f.upload_dataset([("a.txt", _domain_bytes(TAIL_A, 240))],
                              name="m12-domA")
        self.ds_a = up["dataset_id"]
        tok = f.train_tokenizer(
            TokenizerConfig(name="m12-tok", vocab_size=600),
            dataset_id=self.ds_a)
        self.tok_id = tok.id
        f.tokenize_dataset(self.ds_a, tok.id)
        self.model_id = f.create_model(
            ModelCreateRequest(config=_tiny("m12-main")))[0].id
        rep = f.run_training(self._train(self.ds_a, 6))
        self.ck_main = rep.checkpoints[0]["checkpoint_id"]
        self.ck_late = rep.checkpoints[-1]["checkpoint_id"]
        # suite shared by recipe tests: two exact M4 probes on domain A
        probes = [SuiteProbe(dataset_id=self.ds_a, split="validation",
                             tokenizer_id=self.tok_id, batch_size=8,
                             max_seq_len=32, seed=s)
                  for s in (83001, 83002)]
        f.register_probe_suite(ProbeSuiteCreateRequest(
            suite_id="m12-suite", probes=probes))
        self.recipes = RecipeEngine(f.storage, workflows=f.workflows)

    def _train(self, ds_id: str, epochs: int) -> TrainingConfig:
        return TrainingConfig(method="continued_pretraining",
                              model_id=self.model_id, dataset_id=ds_id,
                              tokenizer_id=self.tok_id, learning_rate=3e-3,
                              batch_size=8, max_seq_len=32, epochs=epochs,
                              eval_every_steps=3, keep_best=False, seed=1)

    def fresh_model(self, name="m12-fresh", seed=9) -> str:
        return self.forge.create_model(
            ModelCreateRequest(config=_tiny(name, seed=seed)))[0].id

    def recipes_root(self):
        root = self.forge.storage.root / "workflow-recipes"
        return root if root.exists() else None

    def manifest_bytes(self, recipe_id: str) -> bytes:
        p = (self.forge.storage.root / "workflow-recipes" / recipe_id
             / "manifest.json")
        return p.read_bytes()

    def no_tmp(self) -> None:
        leftovers = [p for p in self.forge.storage.root.rglob("*")
                     if p.is_file() and ".tmp" in p.name]
        assert leftovers == []


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m12-root"))
    e.prepare()
    return e


# --------------------------------------------------------------------------- #
# Stage builders (explicit references, never guessed)
# --------------------------------------------------------------------------- #

def _suite_stage(sid: str, suite_id: str, *, ckpt=None, from_stage=None,
                 current=False) -> WorkflowStage:
    kind = EvalStateKind.CURRENT if current else EvalStateKind.CHECKPOINT
    ref_kw = {}
    if ckpt is not None:
        ref_kw["checkpoint_id"] = ckpt
    elif from_stage is not None:
        ref_kw["from_stage"] = from_stage
    elif not current:
        raise AssertionError("a checkpoint state needs ckpt or from_stage")
    return WorkflowStage(
        stage_id=sid, type=StageType.SUITE_RUN,
        suite_run=WorkflowSuiteRunStage(
            suite_id=suite_id, state=StageStateRef(state_kind=kind, **ref_kw)))


def _train_stage(sid: str, model_id: str, ds: str, tok: str, epochs=2,
                 seed=1) -> WorkflowStage:
    return WorkflowStage(
        stage_id=sid, type=StageType.TRAIN,
        training=TrainingConfig(name="r-train", model_id=model_id,
                                dataset_id=ds, tokenizer_id=tok,
                                learning_rate=3e-3, batch_size=8,
                                max_seq_len=32, epochs=epochs,
                                eval_every_steps=25, keep_best=False,
                                seed=seed))


def _eval_stage(sid: str, model_id: str, ds: str, tok: str,
                from_stage: str) -> WorkflowStage:
    return WorkflowStage(
        stage_id=sid, type=StageType.EVALUATE,
        evaluation=WorkflowEvaluationStage(
            config=EvaluationConfig(model_id=model_id, dataset_id=ds,
                                    split="validation", tokenizer_id=tok,
                                    batch_size=8, max_seq_len=32, seed=2),
            checkpoint_from_stage=from_stage))


def _recipe(request_id: str, stages, description=None) -> WorkflowRecipeCreateRequest:
    return WorkflowRecipeCreateRequest(recipe_id=request_id,
                                       description=description,
                                       stages=stages)


def _suite_evals(env, model_id: str, suite_run_id: str):
    rec = env.forge.get_suite_run(model_id, suite_run_id)
    return [r.evaluation_id for r in rec.results]


# =========================================================================== #
# Registration + M7 validation parity
# =========================================================================== #

def test_recipe_registration_and_idempotence(env):
    st = _suite_stage("eval_suite", "m12-suite", ckpt=env.ck_main)
    r = env.recipes.register(_recipe("m12-reg-1", [st], "first"))
    assert r.recipe_id == "m12-reg-1" and r.config_hash == recipe_config_hash([st])
    assert r.stages[0].stage_id == "eval_suite"
    b0 = env.manifest_bytes("m12-reg-1")
    # identical content + same id -> existing definition returned, no rewrite
    again = env.recipes.register(_recipe("m12-reg-1", [st], "other desc"))
    assert again.config_hash == r.config_hash
    assert env.manifest_bytes("m12-reg-1") == b0
    # same id + different semantic content -> ValueError (409 upstream)
    st2 = _suite_stage("eval_suite", "m12-suite", current=True)
    with pytest.raises(ValueError, match="already exists"):
        env.recipes.register(_recipe("m12-reg-1", [st2]))
    assert env.manifest_bytes("m12-reg-1") == b0
    # identical content + different id -> identical config hash
    other = env.recipes.register(_recipe("m12-reg-1b", [st]))
    assert other.config_hash == r.config_hash
    env.no_tmp()


def test_recipe_registration_supports_all_stage_types(env):
    fresh = env.fresh_model()
    stages = [
        _train_stage("t1", fresh, env.ds_a, env.tok_id, epochs=1),
        _eval_stage("e1", fresh, env.ds_a, env.tok_id, from_stage="t1"),
        WorkflowStage(
            stage_id="c1", type=StageType.COMPARE,
            comparison=WorkflowComparisonStage(
                state_a=StageStateRef(state_kind=EvalStateKind.CHECKPOINT,
                                      from_stage="t1"),
                state_b=StageStateRef(state_kind=EvalStateKind.CURRENT),
                dataset_id=env.ds_a, split="validation",
                tokenizer_id=env.tok_id, batch_size=8, max_seq_len=32,
                seed=3)),
        WorkflowStage(
            stage_id="g1", type=StageType.GATE,
            gate=WorkflowGateStage(
                policy=GatePolicy(name="min-loss", model_id=fresh,
                                  dataset_id=env.ds_a, split="validation",
                                  tokenizer_id=env.tok_id, batch_size=8,
                                  max_seq_len=32, seed=4,
                                  baseline_type=GateBaselineType.MINIMUM_LOSS,
                                  minimum_loss=1e9),
                candidate=StageStateRef(state_kind=EvalStateKind.CHECKPOINT,
                                        from_stage="t1")),
            on_pass="s1", on_fail="s1"),
        _suite_stage("s1", "m12-suite", from_stage="t1"),
    ]
    r = env.recipes.register(_recipe("m12-all-types", stages))
    kinds = [s.type.value for s in r.stages]
    assert kinds == ["train", "evaluate", "compare", "gate", "suite_run"]
    # idempotent re-registration of the same all-stage content
    again = env.recipes.register(_recipe("m12-all-types", stages))
    assert again.config_hash == r.config_hash


def test_recipe_validation_parity_with_inline_plans(env):
    """Every structural rejection must be IDENTICAL for recipes and plans."""

    def both(stages, expected: str):
        with pytest.raises(ValidationError) as pe:
            WorkflowPlan(name="p", model_id=env.model_id, stages=stages)
        with pytest.raises(ValidationError) as re_:
            _recipe("m12-parity", stages)
        pe_msg = pe.value.errors()[0]["msg"]
        re_msg = re_.value.errors()[0]["msg"]
        assert pe_msg == re_msg, (pe_msg, re_msg)
        assert expected in pe_msg

    sr = _suite_stage("s1", "m12-suite", current=True)
    # duplicate stage ids
    both([sr, WorkflowStage(**sr.model_dump())], "duplicate")
    # unknown from_stage target
    both([_suite_stage("s1", "m12-suite", from_stage="ghost")], "unknown stage")
    # self reference
    both([_suite_stage("s1", "m12-suite", from_stage="s1")], "EARLIER")
    # forward reference (suite stage 2 points at a LATER suite stage)
    s1 = _suite_stage("s1", "m12-suite", current=True)
    s2 = _suite_stage("s2", "m12-suite", from_stage="s3")
    s3 = _suite_stage("s3", "m12-suite", current=True)
    both([s1, s2, s3], "EARLIER")
    # from_stage -> earlier NON-train stage (evaluate before suite)
    e0 = _eval_stage("e0", env.model_id, env.ds_a, env.tok_id,
                     from_stage=None)
    bad = _suite_stage("s1", "m12-suite", from_stage="e0")
    with pytest.raises(ValidationError) as pe:
        WorkflowPlan(name="p", model_id=env.model_id, stages=[e0, bad])
    with pytest.raises(ValidationError) as re_:
        _recipe("m12-parity-nontrain", [e0, bad])
    assert ("not a train stage" in pe.value.errors()[0]["msg"]
            and pe.value.errors()[0]["msg"] == re_.value.errors()[0]["msg"])
    # gate on_pass unknown target / backward branch / on non-gate stages
    gate = WorkflowStage(
        stage_id="g1", type=StageType.GATE,
        gate=WorkflowGateStage(
            policy=GatePolicy(name="min-loss", model_id=env.model_id,
                              dataset_id=env.ds_a, split="validation",
                              tokenizer_id=env.tok_id, batch_size=8,
                              max_seq_len=32, seed=4,
                              baseline_type=GateBaselineType.MINIMUM_LOSS,
                              minimum_loss=1e9),
            candidate=StageStateRef(state_kind=EvalStateKind.CURRENT)),
        on_pass="ghost")
    both([gate], "references unknown stage")
    both([WorkflowStage(**{**gate.model_dump(), "on_pass": "g1",
                           "on_fail": None})], "FORWARD")
    with pytest.raises(ValidationError) as pe:
        WorkflowPlan(name="p", model_id=env.model_id,
                     stages=[WorkflowStage(**{**sr.model_dump(),
                                              "on_pass": "s1"})])
    with pytest.raises(ValidationError) as re_:
        _recipe("m12-parity-nongate", [WorkflowStage(**{**sr.model_dump(),
                                                        "on_pass": "s1"})])
    assert ("gate-only" in pe.value.errors()[0]["msg"]
            and pe.value.errors()[0]["msg"] == re_.value.errors()[0]["msg"])
    # invalid suite-run stage state: current + checkpoint_id conflict
    with pytest.raises(ValidationError) as pe:
        WorkflowSuiteRunStage(
            suite_id="m12-suite",
            state=StageStateRef(state_kind=EvalStateKind.CURRENT,
                                checkpoint_id=env.ck_main))
    assert "must not set checkpoint_id" in pe.value.errors()[0]["msg"]
    # checkpoint state with neither source (or both) is rejected the same way
    with pytest.raises(ValidationError, match="exactly one of"):
        StageStateRef(state_kind=EvalStateKind.CHECKPOINT)
    # gate needs exactly one policy source
    with pytest.raises(ValidationError, match="exactly one policy source"):
        _recipe("m12-parity-gate", [WorkflowStage(
            stage_id="g1", type=StageType.GATE,
            gate=WorkflowGateStage(
                candidate=StageStateRef(state_kind=EvalStateKind.CURRENT)))])


def test_recipe_zero_or_blank_stages_rejected(env):
    with pytest.raises(ValidationError):
        _recipe("m12-empty", [])
    with pytest.raises(ValidationError):
        WorkflowRecipeCreateRequest(
            recipe_id="bad id!", stages=[
                _suite_stage("s1", "m12-suite", current=True)])


def test_recipe_list_get_and_deterministic_order(env):
    l1 = env.recipes.list()
    l2 = env.recipes.list()
    assert [r.recipe_id for r in l1] == [r.recipe_id for r in l2]
    assert "m12-reg-1" in [r.recipe_id for r in l1]
    got = env.recipes.get("m12-reg-1")
    assert got.recipe_id == "m12-reg-1" and got.description == "first"
    assert got.config_hash == recipe_config_hash(got.stages)
    with pytest.raises(FileNotFoundError):
        env.recipes.get("m12-no-such")
    # get/list never write
    before = len(list((env.forge.storage.root / "workflow-recipes").iterdir()))
    env.recipes.get("m12-reg-1")
    env.recipes.list()
    assert len(list((env.forge.storage.root / "workflow-recipes").iterdir())) == before


# =========================================================================== #
# Execution: explicit binding + state semantics + provenance
# =========================================================================== #

def test_current_state_recipe_run_provenance_and_reuse(env):
    from app.schemas import ComparisonState

    st = _suite_stage("eval_suite", "m12-suite", current=True)
    r = env.recipes.register(_recipe("m12-cur", [st]))
    # engine-verified identity of the bound model's current state
    cur_hash = env.forge.comparison.verified_state_hash(
        env.model_id, ComparisonState(state_kind=EvalStateKind.CURRENT))

    rec = env.recipes.run("m12-cur", env.model_id)
    assert rec.status == WorkflowStatus.COMPLETED
    assert rec.recipe_id == "m12-cur" and rec.recipe_hash == r.config_hash
    assert rec.name == "m12-cur" and rec.plan.model_id == env.model_id
    art = rec.stages[0].artifact
    assert art.kind.value == "suite_run" and art.checkpoint_id is None
    sr = env.forge.get_suite_run(env.model_id, art.artifact_id)
    assert sr.suite_id == "m12-suite"
    assert sr.state.state_kind == EvalStateKind.CURRENT
    assert sr.state_hash == cur_hash            # resolved + snapshotted
    assert sr.completed_count == 2 and sr.reused_count == 0
    # record on disk carries provenance (manifest round-trip)
    disk = env.forge.get_workflow(env.model_id, rec.workflow_id)
    assert disk.recipe_id == "m12-cur" and disk.recipe_hash == r.config_hash

    # repeat: distinct run + distinct suite run, same evidence, no dup evals
    n0 = len(env.forge.list_evaluations(env.model_id))
    rec2 = env.recipes.run("m12-cur", env.model_id)
    assert rec2.workflow_id != rec.workflow_id
    sr2 = env.forge.get_suite_run(env.model_id, rec2.stages[0].artifact.artifact_id)
    assert sr2.suite_run_id != sr.suite_run_id
    assert sr2.reused_count == 2 and sr2.completed_count == 2
    assert len(env.forge.list_evaluations(env.model_id)) == n0
    assert sr2.result_hash == sr.result_hash
    assert rec2.result_hash == rec.result_hash
    assert _suite_evals(env, env.model_id, sr.suite_run_id) == \
        _suite_evals(env, env.model_id, sr2.suite_run_id)

    # inline plan runs keep null recipe provenance
    inline = env.forge.run_workflow(WorkflowPlan(
        name="inline", model_id=env.model_id, stages=[st]))
    assert inline.recipe_id is None and inline.recipe_hash is None
    env.no_tmp()


def test_explicit_checkpoint_recipe_state_snapshot(env):
    st = _suite_stage("eval_suite", "m12-suite", ckpt=env.ck_late)
    env.recipes.register(_recipe("m12-ck", [st]))
    rec = env.recipes.run("m12-ck", env.model_id)
    assert rec.status == WorkflowStatus.COMPLETED
    art = rec.stages[0].artifact
    assert art.artifact_id and art.checkpoint_id == env.ck_late
    sr = env.forge.get_suite_run(env.model_id, art.artifact_id)
    ck = env.forge.get_checkpoint(env.model_id, env.ck_late)
    assert sr.state.checkpoint_id == env.ck_late
    assert sr.state_hash == ck.weights_sha256          # provenance snapshot
    assert rec.recipe_id == "m12-ck"


def test_earlier_train_stage_recipe_from_stage_provenance(env):
    fresh = env.fresh_model("m12-train-model")
    st_tr = _train_stage("tr", fresh, env.ds_a, env.tok_id, epochs=2)
    st_sr = _suite_stage("eval_suite", "m12-suite", from_stage="tr")
    env.recipes.register(_recipe("m12-fromstage", [st_tr, st_sr]))
    rec = env.recipes.run("m12-fromstage", fresh)
    assert rec.status == WorkflowStatus.COMPLETED
    tr_art = rec.stages[0].artifact
    sr_art = rec.stages[1].artifact
    assert tr_art.kind.value == "training_report"
    # the suite stage evaluated the final checkpoint of the in-run train stage
    assert sr_art.checkpoint_id == tr_art.checkpoint_id
    sr = env.forge.get_suite_run(fresh, sr_art.artifact_id)
    assert sr.state.checkpoint_id == tr_art.checkpoint_id
    ck = env.forge.get_checkpoint(fresh, tr_art.checkpoint_id)
    assert sr.state_hash == ck.weights_sha256
    assert rec.recipe_id == "m12-fromstage"


def test_all_stage_recipe_executes_through_existing_engines(env):
    """train -> evaluate(from_stage) -> compare(from_stage vs current) ->
    suite_run(from_stage) in ONE recipe on a dedicated model: every stage
    resolves within the run and writes only genuine evidence."""
    fresh = env.fresh_model("m12-stack", seed=11)
    stages = [
        _train_stage("t1", fresh, env.ds_a, env.tok_id, epochs=1),
        _eval_stage("e1", fresh, env.ds_a, env.tok_id, from_stage="t1"),
        WorkflowStage(
            stage_id="c1", type=StageType.COMPARE,
            comparison=WorkflowComparisonStage(
                state_a=StageStateRef(state_kind=EvalStateKind.CHECKPOINT,
                                      from_stage="t1"),
                state_b=StageStateRef(state_kind=EvalStateKind.CURRENT),
                dataset_id=env.ds_a, split="validation",
                tokenizer_id=env.tok_id, batch_size=8, max_seq_len=32,
                seed=3)),
        _suite_stage("s1", "m12-suite", from_stage="t1"),
    ]
    env.recipes.register(_recipe("m12-stack", stages))
    n_wf = len(env.forge.list_workflows(fresh))
    rec = env.recipes.run("m12-stack", fresh)
    assert rec.status == WorkflowStatus.COMPLETED
    kinds = [s.artifact.kind.value if s.artifact else None
             for s in rec.stages]
    assert kinds == ["training_report", "evaluation", "comparison",
                     "suite_run"]
    # from_stage provenance: all downstream stages reference the SAME ckpt
    ck = rec.stages[0].artifact.checkpoint_id
    assert rec.stages[1].artifact.checkpoint_id == ck
    assert rec.stages[2].artifact.checkpoint_id is None  # current comparison
    cmp_rec = env.forge.get_comparison(fresh, rec.stages[2].artifact.artifact_id)
    assert cmp_rec.verdict in ("improved", "regressed", "unchanged")
    assert rec.stages[3].artifact.checkpoint_id == ck
    assert len(env.forge.list_workflows(fresh)) == n_wf + 1
    # comparison state_a measured the SAME train checkpoint (provenance chain)
    assert cmp_rec.state_a.checkpoint_id == ck
    env.no_tmp()


def test_gate_stage_inside_recipe_uses_existing_m6_semantics(env):
    fresh = env.fresh_model("m12-gate-model", seed=13)
    policy = GatePolicy(name="min-loss", model_id=fresh,
                        dataset_id=env.ds_a, split="validation",
                        tokenizer_id=env.tok_id, batch_size=8,
                        max_seq_len=32, seed=4,
                        baseline_type=GateBaselineType.MINIMUM_LOSS,
                        minimum_loss=1e9)
    stages = [
        _train_stage("t1", fresh, env.ds_a, env.tok_id, epochs=1),
        WorkflowStage(
            stage_id="g1", type=StageType.GATE,
            gate=WorkflowGateStage(policy=policy,
                                   candidate=StageStateRef(
                                       state_kind=EvalStateKind.CHECKPOINT,
                                       from_stage="t1"))),
    ]
    env.recipes.register(_recipe("m12-gate", stages))
    rec = env.recipes.run("m12-gate", fresh)
    assert rec.status == WorkflowStatus.COMPLETED
    art = rec.stages[1].artifact
    assert art.kind.value == "gate_decision"
    decision = env.forge.get_gate_decision(fresh, art.artifact_id)
    assert decision.decision.value == "passed"   # loss < 1e9 threshold
    assert art.checkpoint_id == rec.stages[0].artifact.checkpoint_id
    env.no_tmp()


# =========================================================================== #
# Failure semantics
# =========================================================================== #

def test_unknown_recipe_and_model_persist_nothing(env):
    n_wf = len(env.forge.list_workflows(env.model_id))
    n_sr = len(env.forge.list_suite_runs(env.model_id))
    root_files = len(list(env.forge.storage.root.rglob("*")))
    with pytest.raises(FileNotFoundError):
        env.recipes.run("m12-no-recipe", env.model_id)
    with pytest.raises(FileNotFoundError):
        env.recipes.run("m12-cur", "m12-no-model")
    assert len(env.forge.list_workflows(env.model_id)) == n_wf
    assert len(env.forge.list_suite_runs(env.model_id)) == n_sr
    assert len(list(env.forge.storage.root.rglob("*"))) == root_files


def test_binding_mismatch_rejected_before_any_write(env):
    other = env.fresh_model("m12-pin-target", seed=15)
    pinned = _train_stage("t1", other, env.ds_a, env.tok_id, epochs=1)
    env.recipes.register(_recipe("m12-pinned", [pinned]))
    n_wf = len(env.forge.list_workflows(env.model_id))
    with pytest.raises(ValueError, match="training config targets model"):
        env.recipes.run("m12-pinned", env.model_id)   # bound != pinned model
    assert len(env.forge.list_workflows(env.model_id)) == n_wf
    # same recipe against ITS pinned model is fine
    rec = env.recipes.run("m12-pinned", other)
    assert rec.status == WorkflowStatus.COMPLETED


def test_missing_suite_at_runtime_failed_run_no_fabricated_artifact(env):
    st = _suite_stage("eval_suite", "m12-no-suite", current=True)
    env.recipes.register(_recipe("m12-ghost-suite", [st]))
    n_wf = len(env.forge.list_workflows(env.model_id))
    n_sr = len(env.forge.list_suite_runs(env.model_id))
    with pytest.raises(FileNotFoundError):
        env.recipes.run("m12-ghost-suite", env.model_id)
    runs = env.forge.list_workflows(env.model_id)
    assert len(runs) == n_wf + 1
    failed = runs[-1]
    assert failed.status == WorkflowStatus.FAILED
    assert failed.failed_stage_id == "eval_suite"
    assert failed.recipe_id == "m12-ghost-suite"
    assert failed.stages[0].artifact is None
    assert len(env.forge.list_suite_runs(env.model_id)) == n_sr


def test_missing_checkpoint_at_runtime_failed_run(env):
    st = _suite_stage("eval_suite", "m12-suite", ckpt="deadbeef0000")
    env.recipes.register(_recipe("m12-ghost-ck", [st]))
    n_wf = len(env.forge.list_workflows(env.model_id))
    n_sr = len(env.forge.list_suite_runs(env.model_id))
    with pytest.raises(FileNotFoundError):
        env.recipes.run("m12-ghost-ck", env.model_id)
    runs = env.forge.list_workflows(env.model_id)
    assert len(runs) == n_wf + 1 and runs[-1].status == WorkflowStatus.FAILED
    assert runs[-1].stages[0].artifact is None
    assert len(env.forge.list_suite_runs(env.model_id)) == n_sr


def test_per_probe_failed_suite_run_referenced_as_is(env):
    """A suite run that persisted with per-probe failures is a REAL artifact:
    the recipe references it and the workflow completes — never fake success,
    never a stage failure."""
    probes = [SuiteProbe(dataset_id="m12-no-dataset", split="validation",
                         tokenizer_id=env.tok_id, batch_size=8,
                         max_seq_len=32, seed=1)]
    env.forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m12-broken-suite", probes=probes))
    st = _suite_stage("eval_suite", "m12-broken-suite", current=True)
    env.recipes.register(_recipe("m12-broken-run", [st]))
    rec = env.recipes.run("m12-broken-run", env.model_id)
    assert rec.status == WorkflowStatus.COMPLETED
    art = rec.stages[0].artifact
    assert art.kind.value == "suite_run"
    sr = env.forge.get_suite_run(env.model_id, art.artifact_id)
    assert sr.status.value == "failed"
    assert sr.failed_count == 1 and sr.completed_count == 0
    assert rec.recipe_id == "m12-broken-run"


# =========================================================================== #
# Immutability + byte-level guarantees + dashboard
# =========================================================================== #

def test_recipe_definitions_byte_identical_after_runs(env):
    root = env.forge.storage.root / "workflow-recipes"
    before = {p.relative_to(root).as_posix(): p.read_bytes()
              for p in root.rglob("*") if p.is_file()}
    env.recipes.run("m12-cur", env.model_id)
    env.recipes.run("m12-ck", env.model_id)
    after = {p.relative_to(root).as_posix(): p.read_bytes()
             for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_recipe_runs_do_not_rewrite_workflow_history(env):
    wroot = env.forge.storage.model_dir(env.model_id) / "workflows"
    before = {p.relative_to(wroot).as_posix(): p.read_bytes()
              for p in wroot.rglob("*") if p.is_file()}
    rec = env.recipes.run("m12-cur", env.model_id)
    assert (wroot / f"workflow-{rec.workflow_id}" / "manifest.json").exists()
    after = {p.relative_to(wroot).as_posix(): p.read_bytes()
             for p in wroot.rglob("*") if p.is_file()}
    # exactly one new file (the run manifest); every pre-existing byte intact
    assert len(after) == len(before) + 1
    for k, v in before.items():
        assert after[k] == v


def test_dashboard_renders_recipe_runs_and_stays_deterministic(env):
    d1 = env.forge.get_dashboard(env.model_id).model_dump(mode="json")
    d2 = env.forge.get_dashboard(env.model_id).model_dump(mode="json")
    assert d1 == d2 and d1["result_hash"] == d2["result_hash"]
    wf_records = d1["workflows"]["records"]
    recipe_runs = [r for r in wf_records if r.get("recipe_id")]
    assert recipe_runs and all(r["recipe_hash"] for r in recipe_runs)
    counts = d1["workflows"]["counts"]
    assert counts["completed"] == sum(1 for r in wf_records
                                      if r["status"] == "completed")


# =========================================================================== #
# HTTP API coverage (shared TestClient root; fresh per-test model setup)
# =========================================================================== #

import json as _json  # noqa: E402

MODELS = "/api/v1/models"
RECIPES = "/api/v1/workflows/recipes"
RUNS = "/api/v1/workflows/recipes/{rid}/runs"


def _http_env(api_client, tag: str):
    """Model + dataset + tokenizer over the real API (no training needed:
    recipe HTTP tests bind current state of the fresh model)."""
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", (f"{tag}.txt",
                                           _domain_bytes(TAIL_A, 120),
                                           "text/plain"))],
                         data={"name": f"api12-{tag}-ds"})
    ds = up.json()["dataset_id"]
    tok = api_client.post("/api/v1/tokenizers/train",
                          data={"config": _json.dumps(
                              {"name": f"api12-{tag}-tok",
                               "vocab_size": 600}),
                                "dataset_id": ds}).json()["tokenizer"]["id"]
    api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                    json={"tokenizer_id": tok})
    cfg = {"name": f"api12-{tag}-model", "vocab_size": 640,
           "context_length": 64, "hidden_size": 64, "n_layers": 2,
           "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128, "seed": 1}
    mid = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    api_client.post("/api/v1/probe-suites", json={
        "suite_id": f"api12-{tag}-suite",
        "probes": [{"dataset_id": ds, "split": "validation",
                    "tokenizer_id": tok, "batch_size": 8,
                    "max_seq_len": 32, "seed": s}
                   for s in (91001, 91002)]})
    return {"mid": mid, "suite": f"api12-{tag}-suite", "ds": ds, "tok": tok}


def _current_suite_stage(sid: str, suite_id: str) -> dict:
    return {"stage_id": sid, "type": "suite_run",
            "suite_run": {"suite_id": suite_id,
                          "state": {"state_kind": "current"}}}


def test_recipe_api_lifecycle_immutability_and_reuse(api_client):
    h = _http_env(api_client, "life")
    mid, suite = h["mid"], h["suite"]
    body = {"recipe_id": "api12-life", "description": "lifecycle recipe",
            "stages": [_current_suite_stage("eval_suite", suite)]}

    r = api_client.post(RECIPES, json=body)
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["recipe_id"] == "api12-life" and len(rec["config_hash"]) == 64

    # idempotent identical re-registration -> same definition, 201
    again = api_client.post(RECIPES, json=body)
    assert again.status_code == 201
    assert again.json()["config_hash"] == rec["config_hash"]
    assert len(api_client.get(RECIPES).json()) == 1

    # same id + different semantic content -> 409, original untouched
    clash = dict(body, stages=[_current_suite_stage("eval_suite", "other")])
    r = api_client.post(RECIPES, json=clash)
    assert r.status_code == 409 and "already exists" in r.text
    assert len(api_client.get(RECIPES).json()) == 1

    # reads: list / get / unknown 404
    got = api_client.get(f"{RECIPES}/api12-life")
    assert got.status_code == 200 and got.json()["config_hash"] == rec["config_hash"]
    assert api_client.get(f"{RECIPES}/no-such").status_code == 404

    # immutability: no update/delete endpoints exist (405)
    assert api_client.put(f"{RECIPES}/api12-life").status_code == 405
    assert api_client.delete(f"{RECIPES}/api12-life").status_code == 405
    assert api_client.put(RUNS.format(rid="api12-life")).status_code == 405

    # run against the explicit model -> completed + recipe provenance
    run = api_client.post(RUNS.format(rid="api12-life"),
                          json={"model_id": mid})
    assert run.status_code == 200, run.text
    w1 = run.json()
    assert w1["status"] == "completed"
    assert w1["recipe_id"] == "api12-life"
    assert w1["recipe_hash"] == rec["config_hash"]
    assert w1["plan"]["model_id"] == mid
    art = w1["stages"][0]["artifact"]
    assert art["kind"] == "suite_run" and art["artifact_id"]
    n_sr = len(api_client.get(f"/api/v1/models/{mid}/suite-runs").json())
    assert n_sr == 1
    n_ev = len(api_client.get(f"{MODELS}/{mid}/evaluations").json())

    # persisted run is readable with provenance intact
    got_wf = api_client.get(f"{MODELS}/{mid}/workflows/{w1['workflow_id']}")
    assert got_wf.json()["recipe_id"] == "api12-life"

    # repeat -> distinct run + suite run, M4 evidence reused (no new evals)
    run2 = api_client.post(RUNS.format(rid="api12-life"),
                           json={"model_id": mid})
    w2 = run2.json()
    assert run2.status_code == 200 and w2["workflow_id"] != w1["workflow_id"]
    assert w2["stages"][0]["artifact"]["artifact_id"] != \
        w1["stages"][0]["artifact"]["artifact_id"]
    assert len(api_client.get(f"/api/v1/models/{mid}/suite-runs").json()) == 2
    assert len(api_client.get(f"{MODELS}/{mid}/evaluations").json()) == n_ev
    # suite-run evidence hash deterministic across the two runs
    sr = api_client.get(f"/api/v1/models/{mid}/suite-runs").json()
    assert sr[0]["result_hash"] == sr[1]["result_hash"]

    # dashboard: deterministic and counts reflect recipe runs
    d1 = api_client.get(f"{MODELS}/{mid}/dashboard").json()
    d2 = api_client.get(f"{MODELS}/{mid}/dashboard").json()
    assert d1 == d2
    assert d1["workflows"]["counts"]["completed"] >= 2
    assert all(rr["recipe_id"] == "api12-life"
               for rr in d1["workflows"]["records"])


def test_recipe_api_failure_semantics(api_client):
    h = _http_env(api_client, "fail")
    mid, suite = h["mid"], h["suite"]
    good = {"recipe_id": "api12-fail-ok",
            "stages": [_current_suite_stage("eval_suite", suite)]}
    api_client.post(RECIPES, json=good)
    ghost = {"recipe_id": "api12-fail-ghost",
             "stages": [_current_suite_stage("eval_suite", "no-such-suite")]}
    api_client.post(RECIPES, json=ghost)

    # unknown recipe -> 404, nothing persisted
    n_wf0 = len(api_client.get(f"{MODELS}/{mid}/workflows").json())
    assert api_client.post(RUNS.format(rid="no-such-recipe"),
                           json={"model_id": mid}).status_code == 404
    assert len(api_client.get(f"{MODELS}/{mid}/workflows").json()) == n_wf0

    # unknown model -> 404, nothing persisted
    assert api_client.post(RUNS.format(rid="api12-fail-ok"),
                           json={"model_id": "no-such-model"}).status_code == 404
    assert len(api_client.get(f"{MODELS}/{mid}/workflows").json()) == n_wf0

    # missing suite at runtime -> 404, but the failed run IS recorded with
    # recipe provenance and no suite-run artifact is fabricated
    n_sr0 = len(api_client.get(f"/api/v1/models/{mid}/suite-runs").json())
    r = api_client.post(RUNS.format(rid="api12-fail-ghost"),
                        json={"model_id": mid})
    assert r.status_code == 404
    wfs = api_client.get(f"{MODELS}/{mid}/workflows").json()
    assert len(wfs) == n_wf0 + 1
    failed = wfs[-1]
    assert failed["status"] == "failed"
    assert failed["failed_stage_id"] == "eval_suite"
    assert failed["recipe_id"] == "api12-fail-ghost"
    assert failed["stages"][0]["artifact"] is None
    assert len(api_client.get(f"/api/v1/models/{mid}/suite-runs").json()) \
        == n_sr0


def test_recipe_api_schema_validation_errors(api_client):
    h = _http_env(api_client, "val")
    mid, suite = h["mid"], h["suite"]
    st = _current_suite_stage("s1", suite)
    # duplicate stage ids -> 422 (same shared rule as inline plans)
    r = api_client.post(RECIPES, json={"recipe_id": "api12-val",
                                       "stages": [st, dict(st)]})
    assert r.status_code == 422
    # empty stage list -> 422
    assert api_client.post(RECIPES,
                           json={"recipe_id": "api12-val2",
                                 "stages": []}).status_code == 422
    # bad recipe id pattern -> 422
    assert api_client.post(RECIPES,
                           json={"recipe_id": "bad id!",
                                 "stages": [st]}).status_code == 422
    # bad suite-run state (current + checkpoint_id) -> 422
    bad = _current_suite_stage("s1", suite)
    bad["suite_run"]["state"] = {"state_kind": "current",
                                 "checkpoint_id": "x" * 12}
    assert api_client.post(RECIPES,
                           json={"recipe_id": "api12-val3",
                                 "stages": [bad]}).status_code == 422
    # run body without model_id -> 422
    assert api_client.post(RUNS.format(rid="api12-val"),
                           json={}).status_code == 422
    assert api_client.post(RUNS.format(rid="api12-val"),
                           json={"model_id": mid,
                                 "extra": 1}).status_code == 422


# =========================================================================== #
# M13: recipe-run lineage (read-only, cross-model) + per-model dashboard
# suite-run section over HTTP
# =========================================================================== #

def test_recipe_lineage_cross_model_ordered_and_annotated(env):
    """Lineage is recipe-scoped: runs from every model that ever bound the
    recipe, merged in (created_at, workflow_id) order, each record annotated
    with its recorded model_id."""
    f = env.forge
    other = env.fresh_model("m13-lineage-b", seed=23)
    env.recipes.register(WorkflowRecipeCreateRequest(
        recipe_id="m13-lin",
        description="M13 lineage fixture recipe",
        stages=[_suite_stage("eval_suite", "m12-suite", current=True)]))
    expected_hash = env.recipes.get("m13-lin").config_hash

    r_main_1 = env.recipes.run("m13-lin", env.model_id)      # model A
    r_other = env.recipes.run("m13-lin", other)              # model B
    r_main_2 = env.recipes.run("m13-lin", env.model_id)      # model A again

    runs = env.recipes.runs("m13-lin")
    assert [r.workflow_id for r in runs] == \
        [r_main_1.workflow_id, r_other.workflow_id,
         r_main_2.workflow_id]
    # (created_at, workflow_id) ordering, never model grouping or FS order
    assert [r.workflow_id for r in runs] == [r.workflow_id for r in sorted(
        runs, key=lambda r: (r.created_at, r.workflow_id))]
    # cross-model annotation, provenance, terminal state
    assert [r.model_id for r in runs] == \
        [env.model_id, other, env.model_id]
    assert all(r.recipe_id == "m13-lin" for r in runs)
    assert all(r.recipe_hash == expected_hash for r in runs)
    assert all(r.status.value == "completed" for r in runs)
    # plan model binding follows the run-time binding, never guessed
    assert [r.plan.model_id for r in runs] == \
        [env.model_id, other, env.model_id]
    # every record resolves through the unchanged per-model read endpoints
    for r in runs:
        got = f.get_workflow(r.model_id, r.workflow_id)
        assert got.workflow_id == r.workflow_id
        assert got.recipe_id == "m13-lin" and got.recipe_hash == expected_hash
    # per-model listings keep their own scoping (recipe runs stay visible
    # under the model that owns them)
    own = [w for w in f.list_workflows(other) if w.recipe_id == "m13-lin"]
    assert [w.workflow_id for w in own] == [r_other.workflow_id]
    # engine and facade agree
    assert [r.workflow_id for r in f.list_recipe_runs("m13-lin")] == \
        [r.workflow_id for r in runs]


def test_recipe_lineage_unknown_recipe_and_empty_recipe(env):
    f = env.forge
    # unknown recipe -> FileNotFoundError (HTTP 404), never a [] guess
    with pytest.raises(FileNotFoundError, match="not found"):
        env.recipes.runs("m13-no-such-recipe")
    with pytest.raises(FileNotFoundError, match="not found"):
        f.list_recipe_runs("m13-no-such-recipe")
    # existing recipe with zero runs -> deterministic empty list
    env.recipes.register(WorkflowRecipeCreateRequest(
        recipe_id="m13-idle",
        stages=[_suite_stage("s1", "m12-suite", current=True)]))
    assert env.recipes.runs("m13-idle") == []
    assert f.list_recipe_runs("m13-idle") == []


def test_recipe_lineage_ignores_inline_runs(env):
    """Workflow runs without recipe provenance never appear in any
    recipe's lineage."""
    f = env.forge
    inline = f.run_workflow(WorkflowPlan(
        name="m13-inline-no-recipe", model_id=env.model_id,
        stages=[_suite_stage("s1", "m12-suite", current=True)]))
    assert inline.recipe_id is None and inline.recipe_hash is None
    ids = {r.workflow_id for r in env.recipes.runs("m13-lin")}
    assert inline.workflow_id not in ids
    # dashboard workflow records keep the null provenance visible as null
    dash = f.get_dashboard(env.model_id).model_dump(mode="json")
    inline_dash = [r for r in dash["workflows"]["records"]
                   if r["workflow_id"] == inline.workflow_id]
    assert len(inline_dash) == 1
    assert inline_dash[0]["recipe_id"] is None
    # no recipe ever claims it
    for rec in f.list_workflow_recipes():
        assert inline.workflow_id not in \
            {r.workflow_id for r in env.recipes.runs(rec.recipe_id)}


def test_recipe_lineage_reads_never_write_byte_audit(env):
    f = env.forge
    root = f.storage.root
    before = sorted(p.relative_to(root).as_posix()
                    for p in root.rglob("*") if p.is_file())
    blobs = {p: (root / p).read_bytes() for p in before}
    for _ in range(2):
        env.recipes.runs("m13-lin")
        env.recipes.runs("m13-idle")
        f.list_recipe_runs("m13-lin")
        f.list_workflows(env.model_id)
        f.list_workflow_recipes()
    after = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*") if p.is_file())
    assert after == before
    for p in before:
        assert (root / p).read_bytes() == blobs[p]
    assert [p for p in root.rglob("*") if ".tmp" in p.name] == []


def test_recipe_lineage_api_deterministic_empty_404_and_readonly(api_client):
    h = _http_env(api_client, "lin13")
    mid, suite = h["mid"], h["suite"]
    # second model over the same shared dataset/tokenizer/probe-suite
    cfg = {"name": "api13-lin-model-b", "vocab_size": 640,
           "context_length": 64, "hidden_size": 64, "n_layers": 2,
           "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128, "seed": 2}
    mid2 = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    body = {"recipe_id": "api13-lin", "description": "lineage over HTTP",
            "stages": [_current_suite_stage("eval_suite", suite)]}
    created = api_client.post(RECIPES, json=body)
    assert created.status_code == 201
    cfg_hash = created.json()["config_hash"]
    path = RUNS.format(rid="api13-lin")

    # existing recipe with no runs -> deterministic [] (a 200, not a 404)
    assert api_client.get(path).json() == []
    # unknown recipe -> 404
    assert api_client.get(RUNS.format(rid="api13-no-such")).status_code == 404

    # two explicit bindings -> cross-model lineage
    w1 = api_client.post(path, json={"model_id": mid})
    assert w1.status_code == 200 and w1.json()["status"] == "completed"
    w2 = api_client.post(path, json={"model_id": mid2})
    assert w2.status_code == 200 and w2.json()["status"] == "completed"
    got = api_client.get(path)
    assert got.status_code == 200
    runs = got.json()
    assert len(runs) == 2
    assert {r["model_id"] for r in runs} == {mid, mid2}
    assert {r["workflow_id"] for r in runs} == \
        {w1.json()["workflow_id"], w2.json()["workflow_id"]}
    for r in runs:
        assert r["recipe_id"] == "api13-lin"
        assert r["recipe_hash"] == cfg_hash
        assert r["status"] == "completed" and r["created_at"]
        assert r["plan"]["model_id"] == r["model_id"]
    assert [r["workflow_id"] for r in runs] == [r["workflow_id"]
                                                for r in sorted(
        runs, key=lambda r: (r["created_at"], r["workflow_id"]))]
    # byte-identical on every repeat
    assert api_client.get(path).text == got.text
    assert api_client.get(path).json() == runs

    # lineage is a read: repeated GETs add nothing to either model
    for mid_x in (mid, mid2):
        n_wf = len(api_client.get(f"{MODELS}/{mid_x}/workflows").json())
        n_sr = len(api_client.get(
            f"/api/v1/models/{mid_x}/suite-runs").json())
        api_client.get(path)
        assert len(api_client.get(f"{MODELS}/{mid_x}/workflows").json()) \
            == n_wf
        assert len(api_client.get(
            f"/api/v1/models/{mid_x}/suite-runs").json()) == n_sr
    # no method mutates the lineage path
    assert api_client.put(path).status_code == 405
    assert api_client.patch(path).status_code == 405
    assert api_client.delete(path).status_code == 405

    # the per-model dashboard suite-run section mirrors the same records
    for mid_x, want in ((mid, 1), (mid2, 1)):
        d1 = api_client.get(f"{MODELS}/{mid_x}/dashboard").json()
        d2 = api_client.get(f"{MODELS}/{mid_x}/dashboard").json()
        assert d1 == d2
        assert d1["suite_runs"]["counts"] == {"completed": want}
        assert len(d1["suite_runs"]["records"]) == want
        assert all(r["model_id"] == mid_x
                   for r in d1["suite_runs"]["records"])


# =========================================================================== #
# M14: composable workflow recipes (recipe stages, deterministic expansion,
# registration-time rejection, composition provenance)
# =========================================================================== #

import hashlib as _m14hash  # noqa: E402
import json as _m14json  # noqa: E402
from app.recipes import MAX_COMPOSITION_DEPTH as _M14_MAX_DEPTH  # noqa: E402
from app.schemas import (  # noqa: E402
    WorkflowRecipeCallStage as _CallStage,
    WorkflowRecipeRef,
)


def _call(sid: str, rid: str) -> WorkflowStage:
    """One M14 recipe-reference stage (registered recipes only)."""
    return WorkflowStage(stage_id=sid, type=StageType.RECIPE,
                         recipe=_CallStage(recipe_id=rid))


def _registered_suite_leaf(env, rid: str, current=True):
    """Register a tiny leaf recipe whose stage ids are predictable."""
    stage = _suite_stage("s1", "m12-suite", current=current)
    return env.recipes.register(_recipe(rid, [stage]))


def _seed_manifest(env, rid: str, refs) -> None:
    """Directly seed one recipe manifest (test-only registry manipulation for
    cycle coverage: through the public API a cycle can never be closed because
    registrations are immutable and may only reference existing recipes)."""
    d = env.forge.storage.root / "workflow-recipes" / rid
    d.mkdir(parents=True, exist_ok=False)
    (d / "manifest.json").write_text(_m14json.dumps({
        "recipe_id": rid, "description": "M14 test seed", "stages": refs,
        "composition": None, "config_hash": "0" * 64,
        "created_at": "2026-09-04T00:00:00Z", "schema_version": 1},
        sort_keys=True))


# --------------------------------------------------------------------------- #
# Schema shape: the recipe stage
# --------------------------------------------------------------------------- #

def test_recipe_stage_schema_shape(env):
    """A recipe stage carries exactly one ``recipe`` payload (recipe_id) and
    is legal ONLY inside registered recipe definitions — never in a plan."""
    stage = WorkflowStage(stage_id="leg", type=StageType.RECIPE,
                          recipe=_CallStage(recipe_id="m14-some-leaf"))
    assert stage.recipe.recipe_id == "m14-some-leaf"
    # declared recipe lists accept the stage...
    req = WorkflowRecipeCreateRequest(recipe_id="m14-shape",
                                      stages=[stage])
    assert req.stages[0].type == StageType.RECIPE
    # ...but a real WorkflowPlan rejects it (the engine never sees one)
    with pytest.raises(ValidationError) as ei:
        WorkflowPlan(name="inline", model_id=env.model_id, stages=[stage])
    assert "only allowed inside registered workflow recipes" in \
        ei.value.errors()[0]["msg"]
    # payload must match the type exactly (shared payload rule)
    with pytest.raises(ValidationError, match="requires the matching payload"):
        WorkflowStage(stage_id="x", type=StageType.TRAIN,
                      recipe=_CallStage(recipe_id="r"))
    with pytest.raises(ValidationError, match="exactly one payload"):
        WorkflowStage(stage_id="x", type=StageType.RECIPE)
    with pytest.raises(ValidationError):
        WorkflowStage(stage_id="x", type=StageType.RECIPE,
                      recipe=_CallStage(recipe_id="r"),
                      suite_run=WorkflowSuiteRunStage(
                          suite_id="m12-suite",
                          state=StageStateRef(state_kind=EvalStateKind.CURRENT)))
    # branches stay gate-only even on recipe stages
    with pytest.raises(ValidationError, match="gate-only"):
        WorkflowStage(stage_id="x", type=StageType.RECIPE,
                      recipe=_CallStage(recipe_id="r"), on_pass="later")


# --------------------------------------------------------------------------- #
# Registration: reference resolution, cycles, depth, immutable writes
# --------------------------------------------------------------------------- #

def test_composite_registration_provenance_and_manifest_shape(env):
    """Registration stores the DECLARED stages (call stage intact) plus a
    reference-oriented ``composition`` (recipe_id + config_hash of each
    direct dependency). Plain recipes keep composition null."""
    leaf = _registered_suite_leaf(env, "m14-mani-leaf")
    own = _suite_stage("own", "m12-suite", current=True)
    comp = env.recipes.register(_recipe("m14-mani-comp", [own,
                                                          _call("leg", leaf.recipe_id)]))
    assert comp.composition == [WorkflowRecipeRef(
        recipe_id=leaf.recipe_id, config_hash=leaf.config_hash)]
    assert [s.type for s in comp.stages] == [StageType.SUITE_RUN,
                                             StageType.RECIPE]
    assert comp.stages[1].recipe.recipe_id == leaf.recipe_id
    # the manifest on disk round-trips the same shape
    disk = env.recipes.get("m14-mani-comp")
    assert disk.config_hash == comp.config_hash
    assert [(r.recipe_id, r.config_hash) for r in disk.composition] == \
        [(leaf.recipe_id, leaf.config_hash)]
    # a plain recipe (no recipe stages) keeps null composition
    plain = _registered_suite_leaf(env, "m14-mani-plain")
    assert plain.composition is None
    assert plain.config_hash == recipe_config_hash(plain.stages)
    env.no_tmp()


def test_composite_hash_determinism_and_sensitivity(env):
    leaf = _registered_suite_leaf(env, "m14-hash-leaf")
    # an alternative leaf with DIFFERENT semantic content (same id not
    # possible — immutable — so a new id with different stages)
    alt = env.recipes.register(_recipe("m14-hash-leaf2",
                                       [_suite_stage("s1", "m12-suite",
                                                     current=True),
                                        _suite_stage("s1b", "m12-suite",
                                                     current=True)]))
    assert alt.config_hash != leaf.config_hash
    own = _suite_stage("own", "m12-suite", current=True)
    stages = [own, _call("leg", leaf.recipe_id)]
    a = env.recipes.register(_recipe("m14-hash-a", stages))
    # same semantics under a different composite id -> identical hash
    b = env.recipes.register(_recipe("m14-hash-b", stages))
    assert a.config_hash == b.config_hash
    # changed referenced recipe (its config hash differs) -> hash changes
    c = env.recipes.register(_recipe("m14-hash-c",
                                     [own, _call("leg", alt.recipe_id)]))
    assert c.config_hash != a.config_hash
    # changed declared stage -> hash changes
    d = env.recipes.register(_recipe("m14-hash-d",
                                     [_suite_stage("own", "m12-suite",
                                                   ckpt=env.ck_main),
                                      _call("leg", leaf.recipe_id)]))
    assert d.config_hash != a.config_hash
    env.no_tmp()


def test_composite_idempotent_reregistration_and_conflict_bytes(env):
    leaf = _registered_suite_leaf(env, "m14-idem-leaf")
    stages = [_suite_stage("own", "m12-suite", current=True),
              _call("leg", leaf.recipe_id)]
    r = env.recipes.register(_recipe("m14-idem-comp", stages))
    b0 = env.manifest_bytes("m14-idem-comp")
    n0 = len(list((env.forge.storage.root / "workflow-recipes").rglob("*.json")))
    again = env.recipes.register(_recipe("m14-idem-comp", stages))
    assert again.config_hash == r.config_hash
    assert env.manifest_bytes("m14-idem-comp") == b0
    assert len(list((env.forge.storage.root / "workflow-recipes")
                    .rglob("*.json"))) == n0
    # same id + any different content -> ValueError (409) and bytes untouched
    with pytest.raises(ValueError, match="already exists"):
        env.recipes.register(_recipe("m14-idem-comp",
                                     [_call("leg", leaf.recipe_id)]))
    assert env.manifest_bytes("m14-idem-comp") == b0
    env.no_tmp()


def test_unknown_dependency_rejected_no_manifest(env):
    n0 = len(list((env.forge.storage.root / "workflow-recipes").rglob("*.json")))
    with pytest.raises(ValueError, match="references unknown workflow recipe"
                       " 'm14-no-such'"):
        env.recipes.register(_recipe("m14-unknown-comp",
                                     [_call("leg", "m14-no-such")]))
    assert not (env.forge.storage.root / "workflow-recipes"
                / "m14-unknown-comp").exists()
    assert len(list((env.forge.storage.root / "workflow-recipes")
                    .rglob("*.json"))) == n0
    env.no_tmp()


def test_cycles_rejected_at_registration_no_manifest(env):
    """Self-cycles via an unknown self-reference; 2- and 3-node cycles are
    closed only by registry manipulation (public registrations are immutable
    and point at existing recipes only), which the engine still detects."""
    n0 = len(list((env.forge.storage.root / "workflow-recipes").rglob("*.json")))
    with pytest.raises(ValueError, match="unknown workflow recipe 'm14-self'"):
        env.recipes.register(_recipe("m14-self",
                                     [_call("x", "m14-self")]))
    assert not (env.forge.storage.root / "workflow-recipes"
                / "m14-self").exists()

    # 2-node: seed B -> A, then registering A -> B closes the cycle
    _seed_manifest(env, "m14-cycle-b",
                   [_call("y", "m14-cycle-a").model_dump(mode="json")])
    with pytest.raises(ValueError, match="composition cycle detected"):
        env.recipes.register(_recipe("m14-cycle-a",
                                     [_call("x", "m14-cycle-b")]))
    assert not (env.forge.storage.root / "workflow-recipes"
                / "m14-cycle-a").exists()

    # longer: seed C -> B and B -> A; registering A -> C closes A-B-C-A
    _seed_manifest(env, "m14-cy3-c",
                   [_call("z", "m14-cy3-b").model_dump(mode="json")])
    _seed_manifest(env, "m14-cy3-b",
                   [_call("y", "m14-cy3-a").model_dump(mode="json")])
    with pytest.raises(ValueError, match="composition cycle detected"):
        env.recipes.register(_recipe("m14-cy3-a",
                                     [_call("x", "m14-cy3-c")]))
    assert not (env.forge.storage.root / "workflow-recipes"
                / "m14-cy3-a").exists()
    assert len(list((env.forge.storage.root / "workflow-recipes")
                    .rglob("*.json"))) == n0 + 3  # only the three seeds
    env.no_tmp()


def test_composition_depth_32_accepted_33_rejected(env):
    assert _M14_MAX_DEPTH == 32
    # chain base(1) + 31 links = 32 recipes: accepted all the way
    prev = "m14-depth-leaf"
    _registered_suite_leaf(env, prev)
    for i in range(1, _M14_MAX_DEPTH):
        rid = f"m14-depth-{i}"
        env.recipes.register(_recipe(rid, [_call("d", prev)]))
        prev = rid
    # the deepest accepted chain executes: ONE run, fully qualified ids
    rec = env.recipes.run(prev, env.model_id)
    assert rec.status == WorkflowStatus.COMPLETED
    (deep_id,) = [s.stage_id for s in rec.plan.stages]
    assert deep_id.endswith("s1") and len(deep_id) <= 64
    # 33rd recipe on the same chain -> rejected, nothing persisted
    n0 = len(list((env.forge.storage.root / "workflow-recipes").rglob("*.json")))
    with pytest.raises(ValueError,
                       match="composition depth 33 exceeds the maximum"):
        env.recipes.register(_recipe("m14-depth-33",
                                     [_call("d", prev)]))
    assert not (env.forge.storage.root / "workflow-recipes"
                / "m14-depth-33").exists()
    assert len(list((env.forge.storage.root / "workflow-recipes")
                    .rglob("*.json"))) == n0
    env.no_tmp()


def test_invalid_expanded_plan_rejected_no_manifest(env):
    """A gate branch targeting a recipe-call stage passes the DECLARED-stage
    rules (the call stage exists and is later) but the call stage vanishes
    during expansion — the shared validator over the FULLY EXPANDED list must
    reject the composite at registration."""
    leaf = _registered_suite_leaf(env, "m14-branch-leaf")
    policy = GatePolicy(name="min-loss", model_id=env.model_id,
                        dataset_id=env.ds_a, split="validation",
                        tokenizer_id=env.tok_id, batch_size=8,
                        max_seq_len=32, seed=4,
                        baseline_type=GateBaselineType.MINIMUM_LOSS,
                        minimum_loss=1e9)
    gate = WorkflowStage(
        stage_id="g0", type=StageType.GATE,
        gate=WorkflowGateStage(policy=policy,
                               candidate=StageStateRef(
                                   state_kind=EvalStateKind.CURRENT)),
        on_pass="leg")
    n0 = len(list((env.forge.storage.root / "workflow-recipes").rglob("*.json")))
    with pytest.raises(ValueError,
                       match="stage 'g0' on_pass references unknown stage"
                       " 'leg'"):
        env.recipes.register(_recipe("m14-branch-comp",
                                     [gate, _call("leg", leaf.recipe_id)]))
    assert not (env.forge.storage.root / "workflow-recipes"
                / "m14-branch-comp").exists()
    assert len(list((env.forge.storage.root / "workflow-recipes")
                    .rglob("*.json"))) == n0

    # cross-boundary from_stage INTO a callee is rejected at schema level
    # (declared lists contain no qualified ids): nothing is ever written
    with pytest.raises(ValidationError, match="references unknown stage"
                       " 'leg.s1'"):
        env.recipes.register(_recipe("m14-cross-comp", [
            _call("leg", leaf.recipe_id),
            _suite_stage("late", "m12-suite", from_stage="leg.s1")]))
    assert not (env.forge.storage.root / "workflow-recipes"
                / "m14-cross-comp").exists()
    env.no_tmp()


def test_pinned_model_inside_referenced_recipe_conflict_rejected(env):
    """A model pinned inside a referenced recipe is never rewritten: running
    the composite against a DIFFERENT explicit model is rejected before any
    stage executes and persists nothing."""
    target = env.fresh_model("m14-pin-target", seed=31)
    other = env.fresh_model("m14-pin-other", seed=33)
    pinned_leaf = env.recipes.register(_recipe("m14-pin-leaf", [
        _train_stage("tr", target, env.ds_a, env.tok_id, epochs=1)]))
    env.recipes.register(_recipe("m14-pin-comp",
                                 [_call("leg", pinned_leaf.recipe_id)]))
    n_wf = len(env.forge.list_workflows(other))
    n_root = len(list(env.forge.storage.root.rglob("*.json")))
    with pytest.raises(ValueError,
                       match="stage 'leg.tr' training config targets model"):
        env.recipes.run("m14-pin-comp", other)
    assert len(env.forge.list_workflows(other)) == n_wf
    assert len(list(env.forge.storage.root.rglob("*.json"))) == n_root
    # the same composite against its own pinned model executes normally
    rec = env.recipes.run("m14-pin-comp", target)
    assert rec.status == WorkflowStatus.COMPLETED
    assert rec.recipe_id == "m14-pin-comp"
    env.no_tmp()


# --------------------------------------------------------------------------- #
# Expansion + execution semantics
# --------------------------------------------------------------------------- #

def test_expansion_order_parent_first_unique_qualified_ids(env):
    """A1, A2, B1, B2, A3 splice order: stages of the parent stay in place and
    every referenced recipe is spliced at its call position, its own ids
    qualified by the call-stage path (deterministic; identical leaf stages at
    different call positions never collide)."""
    fresh = env.fresh_model("m14-order-model", seed=41)
    leaf_p = env.recipes.register(_recipe("m14-order-p", [
        _suite_stage("p1", "m12-suite", current=True),
        _suite_stage("p2", "m12-suite", current=True)]))
    env.recipes.register(_recipe("m14-order-comp", [
        _suite_stage("ownA1", "m12-suite", current=True),
        _call("legA", leaf_p.recipe_id),
        _suite_stage("ownA2", "m12-suite", current=True),
        _call("legB", leaf_p.recipe_id),      # same leaf twice
        _suite_stage("ownA3", "m12-suite", current=True)]))
    rec = env.recipes.run("m14-order-comp", fresh)
    assert rec.status == WorkflowStatus.COMPLETED
    assert [s.stage_id for s in rec.plan.stages] == \
        ["ownA1", "legA.p1", "legA.p2", "ownA2", "legB.p1", "legB.p2",
         "ownA3"]
    assert [r.stage_id for r in rec.stages] == \
        ["ownA1", "legA.p1", "legA.p2", "ownA2", "legB.p1", "legB.p2",
         "ownA3"]
    # exactly ONE workflow record; seven suite-run manifests (one per suite
    # stage: 3 own + 2x2 from the leaf), M4 evidence created exactly once
    assert len(env.forge.list_workflows(fresh)) == 1
    assert len(env.forge.list_suite_runs(fresh)) == 7
    assert len(env.forge.list_evaluations(fresh)) == 2
    assert rec.composition is not None and len(rec.composition) == 2
    assert all(x.recipe_id == leaf_p.recipe_id for x in rec.composition)
    assert rec.recipe_id == "m14-order-comp"
    env.no_tmp()


def test_nested_composite_expansion_provenance_and_lineage(env):
    """A -> B -> C: nested calls expand recursively under ONE documented
    qualification convention (dot-joined call-stage path); the run record
    carries the full depth-first composition trace; lineage ownership stays
    with the TOP-LEVEL recipe only."""
    fresh = env.fresh_model("m14-nest-model", seed=43)
    c = _registered_suite_leaf(env, "m14-nest-c")
    b = env.recipes.register(_recipe("m14-nest-b", [
        _suite_stage("ownB", "m12-suite", current=True),
        _call("legC", c.recipe_id)]))
    a = env.recipes.register(_recipe("m14-nest-a", [
        _suite_stage("ownA", "m12-suite", current=True),
        _call("legB", b.recipe_id)]))
    rec = env.recipes.run("m14-nest-a", fresh)
    assert rec.status == WorkflowStatus.COMPLETED
    assert [s.stage_id for s in rec.plan.stages] == \
        ["ownA", "legB.ownB", "legB.legC.s1"]
    # provenance: depth-first trace [B, C] with their real config hashes
    assert [(x.recipe_id, x.config_hash) for x in rec.composition] == \
        [(b.recipe_id, b.config_hash), (c.recipe_id, c.config_hash)]
    assert rec.recipe_id == "m14-nest-a" and rec.recipe_hash == a.config_hash
    # persisted record round-trips the provenance
    disk = env.forge.get_workflow(fresh, rec.workflow_id)
    assert disk.recipe_id == "m14-nest-a"
    assert [x.recipe_id for x in disk.composition] == ["m14-nest-b",
                                                       "m14-nest-c"]
    # lineage ownership: ONLY the top-level recipe claims the run
    assert [r.workflow_id for r in env.recipes.runs("m14-nest-a")] == \
        [rec.workflow_id]
    assert env.recipes.runs("m14-nest-b") == []
    assert env.recipes.runs("m14-nest-c") == []
    # plain/leaf runs never carry composition provenance
    leaf_run = env.recipes.run("m14-nest-c", fresh)
    assert leaf_run.composition is None
    assert [r.workflow_id for r in env.recipes.runs("m14-nest-c")] == \
        [leaf_run.workflow_id]
    # deterministic repeat: identical result hash + composition, no new evals
    n_ev = len(env.forge.list_evaluations(fresh))
    n_sr = len(env.forge.list_suite_runs(fresh))
    rec2 = env.recipes.run("m14-nest-a", fresh)
    assert rec2.result_hash == rec.result_hash
    assert [x.recipe_id for x in rec2.composition] == ["m14-nest-b",
                                                       "m14-nest-c"]
    assert rec2.recipe_hash == rec.recipe_hash
    assert len(env.forge.list_evaluations(fresh)) == n_ev
    assert len(env.forge.list_suite_runs(fresh)) == n_sr + 3
    env.no_tmp()


def test_internal_references_rewritten_within_referenced_expansion(env):
    """A referenced recipe that trains and then evaluates / runs a suite /
    gates on its own earlier train stage must keep working after expansion:
    every internal from_stage reference is qualified to the SAME expansion
    (never an identically named sibling/parent stage)."""
    fresh = env.fresh_model("m14-inner-model", seed=47)
    policy = GatePolicy(name="min-loss", model_id=fresh,
                        dataset_id=env.ds_a, split="validation",
                        tokenizer_id=env.tok_id, batch_size=8,
                        max_seq_len=32, seed=4,
                        baseline_type=GateBaselineType.MINIMUM_LOSS,
                        minimum_loss=1e9)
    inner = env.recipes.register(_recipe("m14-inner-leaf", [
        _train_stage("tr", fresh, env.ds_a, env.tok_id, epochs=2),
        _eval_stage("e1", fresh, env.ds_a, env.tok_id, from_stage="tr"),
        _suite_stage("s1", "m12-suite", from_stage="tr"),
        WorkflowStage(
            stage_id="g1", type=StageType.GATE,
            gate=WorkflowGateStage(policy=policy,
                                   candidate=StageStateRef(
                                       state_kind=EvalStateKind.CHECKPOINT,
                                       from_stage="tr")))]))
    env.recipes.register(_recipe("m14-inner-comp",
                                 [_call("leg", inner.recipe_id)]))
    rec = env.recipes.run("m14-inner-comp", fresh)
    assert rec.status == WorkflowStatus.COMPLETED
    assert [s.stage_id for s in rec.plan.stages] == \
        ["leg.tr", "leg.e1", "leg.s1", "leg.g1"]
    kinds = [s.artifact.kind.value for s in rec.stages]
    assert kinds == ["training_report", "evaluation", "suite_run",
                     "gate_decision"]
    # every downstream stage measured the SAME in-expansion train checkpoint
    ck = rec.stages[0].artifact.checkpoint_id
    assert rec.stages[1].artifact.checkpoint_id == ck
    assert rec.stages[2].artifact.checkpoint_id == ck
    assert rec.stages[3].artifact.checkpoint_id == ck
    sr = env.forge.get_suite_run(fresh, rec.stages[2].artifact.artifact_id)
    assert sr.state.checkpoint_id == ck
    gate = env.forge.get_gate_decision(fresh,
                                       rec.stages[3].artifact.artifact_id)
    assert gate.decision.value == "passed"
    env.no_tmp()


# --------------------------------------------------------------------------- #
# HTTP API coverage (recipe stages over the wire)
# --------------------------------------------------------------------------- #

def test_recipe_api_composite_lifecycle(api_client):
    h = _http_env(api_client, "c14")
    mid, suite = h["mid"], h["suite"]
    call = {"stage_id": "leg", "type": "recipe",
            "recipe": {"recipe_id": "api14-leaf"}}
    leaf = {"recipe_id": "api14-leaf",
            "stages": [_current_suite_stage("s1", suite)]}
    comp_body = {"recipe_id": "api14-comp", "description": "composite",
                 "stages": [_current_suite_stage("own", suite), call]}

    n_rec0 = len(api_client.get(RECIPES).json())  # shared HTTP registry root
    r = api_client.post(RECIPES, json=leaf)
    assert r.status_code == 201, r.text
    leaf_hash = r.json()["config_hash"]
    r = api_client.post(RECIPES, json=comp_body)
    assert r.status_code == 201, r.text
    comp = r.json()
    assert comp["composition"] == [{"recipe_id": "api14-leaf",
                                    "config_hash": leaf_hash}]
    assert comp["stages"][1]["type"] == "recipe"
    assert comp["stages"][1]["recipe"]["recipe_id"] == "api14-leaf"
    # plain leaf recipe: no composition key content
    leaf_got = api_client.get(f"{RECIPES}/api14-leaf").json()
    assert leaf_got["composition"] is None
    assert len(api_client.get(RECIPES).json()) == n_rec0 + 2

    # idempotent identical re-registration; conflict -> 409 bytes untouched
    again = api_client.post(RECIPES, json=comp_body)
    assert again.status_code == 201
    assert again.json()["config_hash"] == comp["config_hash"]
    clash = dict(comp_body, stages=[call])
    r = api_client.post(RECIPES, json=clash)
    assert r.status_code == 409 and "already exists" in r.text
    assert api_client.get(f"{RECIPES}/api14-comp").json() == comp

    # unknown dependency -> 422, nothing registered
    bad = {"recipe_id": "api14-ghost",
           "stages": [{"stage_id": "x", "type": "recipe",
                       "recipe": {"recipe_id": "api14-no-such"}}]}
    r = api_client.post(RECIPES, json=bad)
    assert r.status_code == 422 and "unknown" in r.text
    assert len(api_client.get(RECIPES).json()) == n_rec0 + 2
    # schema-level: recipe stage inside an inline plan body is rejected 422
    assert api_client.post("/api/v1/workflows/run", json={
        "name": "inline", "model_id": mid, "stages": [call]}).status_code == 422

    # run the composite -> ONE completed run with qualified ids + provenance
    run = api_client.post(RUNS.format(rid="api14-comp"),
                          json={"model_id": mid})
    assert run.status_code == 200, run.text
    w1 = run.json()
    assert w1["status"] == "completed"
    assert [s["stage_id"] for s in w1["plan"]["stages"]] == ["own", "leg.s1"]
    assert w1["recipe_id"] == "api14-comp"
    assert w1["recipe_hash"] == comp["config_hash"]
    assert w1["composition"] == [{"recipe_id": "api14-leaf",
                                  "config_hash": leaf_hash}]
    # exactly one workflow record; suite-run evidence created once
    assert len(api_client.get(f"{MODELS}/{mid}/workflows").json()) == 1
    n_sr = len(api_client.get(f"/api/v1/models/{mid}/suite-runs").json())
    assert n_sr == 2
    n_ev = len(api_client.get(f"{MODELS}/{mid}/evaluations").json())
    assert n_ev == 2

    # repeat -> new run, same semantics, zero duplicate evaluation manifests
    w2 = api_client.post(RUNS.format(rid="api14-comp"),
                         json={"model_id": mid}).json()
    assert w2["workflow_id"] != w1["workflow_id"]
    assert w2["recipe_hash"] == w1["recipe_hash"]
    assert w2["composition"] == w1["composition"]
    assert len(api_client.get(f"{MODELS}/{mid}/evaluations").json()) == n_ev
    assert len(api_client.get(f"/api/v1/models/{mid}/suite-runs").json()) \
        == n_sr + 2

    # lineage: ONLY the top-level composite recipe owns the two runs
    lin1 = api_client.get(RUNS.format(rid="api14-comp"))
    runs = lin1.json()
    assert len(runs) == 2
    assert all(r["recipe_id"] == "api14-comp" and r["composition"]
               for r in runs)
    assert api_client.get(RUNS.format(rid="api14-leaf")).json() == []
    # lineage GET is deterministic (byte-identical repeats) and read-only
    assert api_client.get(RUNS.format(rid="api14-comp")).text == lin1.text
    assert api_client.get(RUNS.format(rid="api14-comp")).json() == runs
    for mid_x in (mid,):
        d1 = api_client.get(f"{MODELS}/{mid_x}/dashboard").json()
        d2 = api_client.get(f"{MODELS}/{mid_x}/dashboard").json()
        assert d1 == d2 and d1["result_hash"] == d2["result_hash"]
    assert d1["suite_runs"]["counts"]["completed"] == 4

    # OpenAPI exposes the new stage kind and the reference model
    spec = api_client.get("/openapi.json").json()
    stage_enums = [s.get("enum") for s in spec["components"]["schemas"].values()
                   if {"train", "evaluate", "compare", "gate", "suite_run"}
                   <= set(s.get("enum") or [])]
    assert len(stage_enums) == 1 and "recipe" in stage_enums[0]
    assert "WorkflowRecipeRef" in spec["components"]["schemas"]
    assert "WorkflowRecipeCallStage" in spec["components"]["schemas"]


def test_plain_recipe_hash_stable_across_m12_m14_schema(env):
    """The canonical stage serialization excludes the nullable ``recipe``
    payload field (M14), so an identical definition reproduces the exact
    M12 config hash — the hash never changes just because the schema gained
    a field. Regression: pre-M14 manifests keep their stored hashes."""
    st = _suite_stage("eval_suite", "m12-suite", ckpt=env.ck_main)
    registered = env.recipes.register(_recipe("m14-hashparity", [st]))
    # m12-era serialization = full dump minus the recipe key
    legacy = _m14json.dumps(
        [{k: v for k, v in s.model_dump(mode="json").items()
          if k != "recipe"} for s in [st]],
        sort_keys=True).encode("utf-8")
    assert _m14hash.sha256(legacy).hexdigest() == registered.config_hash
    assert registered.config_hash == recipe_config_hash([st])
    # identical content registered under a new id matches too
    again = env.recipes.register(_recipe("m14-hashparity-b", [st]))
    assert again.config_hash == registered.config_hash
    # and the pre-M14 production-style manifest (stored hash over the
    # legacy serialization) recomputes identically through the engine
    from app.schemas import WorkflowRecipe as _WR
    crafted = _WR(recipe_id="m14-hashparity-legacy", description=None,
                  stages=[st], composition=None,
                  config_hash=registered.config_hash,
                  created_at=env.recipes.get("m14-hashparity").created_at)
    assert recipe_config_hash(crafted.stages) == crafted.config_hash
    env.no_tmp()


# =========================================================================== #
# M35: read-only model-scoped by-recipe grouping of the workflow history
# =========================================================================== #

BY_RECIPE = "/api/v1/models/{mid}/workflows/by-recipe/{rid}"


def _m35_state(env):
    """M35 state on top of the shared module env (cached): two recipes
    (r1 run twice, r2 run once) over the shared m12-suite checkpoint
    stage, a registered-but-never-run recipe r3 (the natural valid-id
    empty case), one ad-hoc inline run (recipe_id None), and a fresh
    model with zero workflows (model-scoped isolation). Returns
    (w1a, w1b, w2, adhoc, fresh).
    """
    cached = getattr(env, "_m35", None)
    if cached is not None:
        return cached
    st = _suite_stage("eval_suite", "m12-suite", ckpt=env.ck_main)
    env.recipes.register(_recipe("m35-r1", [st]))
    env.recipes.register(_recipe("m35-r2", [st]))
    env.recipes.register(_recipe("m35-r3", [st]))      # never run
    w1a = env.recipes.run("m35-r1", env.model_id)
    w1b = env.recipes.run("m35-r1", env.model_id)
    w2 = env.recipes.run("m35-r2", env.model_id)
    adhoc = env.forge.run_workflow(WorkflowPlan(
        name="m35-inline", model_id=env.model_id, stages=[st]))
    assert adhoc.recipe_id is None
    fresh = env.fresh_model("m35-fresh")
    env._m35 = (w1a, w1b, w2, adhoc, fresh)
    return env._m35


def test_m35_engine_grouping_parity_order_verbatim(env):
    w1a, w1b, w2, adhoc, fresh = _m35_state(env)
    f = env.forge
    listing = f.list_workflows(env.model_id)
    for rid in ("m35-r1", "m35-r2"):
        got = f.list_workflows_for_recipe(env.model_id, rid)
        # parity with the authoritative M11 listing filtered by the
        # persisted recipe identity; deterministic (created_at,
        # workflow_id) order; membership from the persisted field only
        assert got == [w for w in listing if w.recipe_id == rid]
        keyed = [(x.created_at, x.workflow_id) for x in got]
        assert keyed == sorted(keyed)
        ids = [x.workflow_id for x in got]
        assert len(ids) == len(set(ids))
        assert all(x.recipe_id == rid and x.model_id == env.model_id
                   for x in got)
        # verbatim payload parity with the M11 single-record getter AND
        # verbatim recipe_hash provenance vs the registered definition
        definition = env.recipes.get(rid)
        for x in got:
            assert x == f.get_workflow(env.model_id, x.workflow_id)
            assert x.recipe_hash == definition.config_hash
    got1 = f.list_workflows_for_recipe(env.model_id, "m35-r1")
    assert {x.workflow_id for x in got1} == {w1a.workflow_id,
                                             w1b.workflow_id}
    assert [x.workflow_id for x in f.list_workflows_for_recipe(
        env.model_id, "m35-r2")] == [w2.workflow_id]
    # ad-hoc (None) runs belong to NO group but stay listed
    assert adhoc.recipe_id is None
    assert adhoc.workflow_id in {w.workflow_id for w in listing}
    # explicit partition: groups over EVERY non-null persisted recipe id
    # cover exactly the recipe-attributed runs (discovery, not fixed)
    attributed = [w for w in listing if w.recipe_id is not None]
    groups = {rid: {x.workflow_id for x in
                    f.list_workflows_for_recipe(env.model_id, rid)}
              for rid in {w.recipe_id for w in attributed}}
    flat = [i for g in groups.values() for i in g]
    assert set(flat) == {w.workflow_id for w in attributed}
    assert len(flat) == len(set(flat)) == len(attributed)
    assert adhoc.workflow_id not in set(flat)
    assert "m35-r1" in groups


def test_m35_engine_empty_404s_cross_model_read_only(env):
    w1a, w1b, w2, adhoc, fresh = _m35_state(env)
    f = env.forge
    # valid registered recipe with zero runs (globally fresh) -> []
    assert f.list_workflows_for_recipe(env.model_id, "m35-r3") == []
    # model-scoped empty: fresh model has an EMPTY workflow listing, so
    # every registered recipe returns [] under it (global recipes,
    # model-scoped history)
    assert f.list_workflows(fresh) == []
    for rid in ("m35-r1", "m35-r2", "m35-r3"):
        assert f.list_workflows_for_recipe(fresh, rid) == []
    # unknown model / unknown recipe -> FileNotFoundError (404 at API);
    # a valid recipe never makes an unknown model valid
    with pytest.raises(FileNotFoundError):
        f.list_workflows_for_recipe("ghost-model-35", "m35-r1")
    with pytest.raises(FileNotFoundError):
        f.list_workflows_for_recipe(env.model_id, "ghost-recipe-35")
    # cross-model isolation: the fresh model never sees main's runs
    fresh_ids = {w.workflow_id for w in f.list_workflows(fresh)}
    main_ids = {w.workflow_id for w in f.list_workflows(env.model_id)}
    assert fresh_ids.isdisjoint(main_ids)
    # read-only: neither workflow manifests nor recipe manifests change
    wf_files = {p.relative_to(env.forge.storage.root).as_posix()
                for p in (env.forge.storage.root / "workflows")
                .rglob("*") if p.is_file()} \
        if (env.forge.storage.root / "workflows").exists() else set()
    rc_bytes = {rid: env.manifest_bytes(rid)
                for rid in ("m35-r1", "m35-r2", "m35-r3")}
    f.list_workflows_for_recipe(env.model_id, "m35-r1")
    f.list_workflows_for_recipe(env.model_id, "m35-r3")
    f.list_workflows_for_recipe(fresh, "m35-r1")
    wf_after = {p.relative_to(env.forge.storage.root).as_posix()
                for p in (env.forge.storage.root / "workflows")
                .rglob("*") if p.is_file()} \
        if (env.forge.storage.root / "workflows").exists() else set()
    assert wf_after == wf_files
    assert {rid: env.manifest_bytes(rid)
            for rid in ("m35-r1", "m35-r2", "m35-r3")} == rc_bytes


def test_m35_engine_repeated_calls_identical(env):
    w1a, w1b, w2, adhoc, fresh = _m35_state(env)
    f = env.forge
    first = [w.model_dump(mode="json") for w in
             f.list_workflows_for_recipe(env.model_id, "m35-r1")]
    for _ in range(3):
        again = [w.model_dump(mode="json") for w in
                 f.list_workflows_for_recipe(env.model_id, "m35-r1")]
        assert again == first


def test_m35_api_by_recipe_grouping_partition_determinism(api_client):
    h = _http_env(api_client, "m35a")
    mid, suite = h["mid"], h["suite"]
    stage = _current_suite_stage("eval_suite", suite)
    for rid in ("api35-r1", "api35-r2", "api35-r3"):
        r = api_client.post(RECIPES, json={
            "recipe_id": rid, "description": f"m35 {rid}",
            "stages": [stage]})
        assert r.status_code == 201, r.text
    w1a = api_client.post(RUNS.format(rid="api35-r1"),
                          json={"model_id": mid}).json()
    w1b = api_client.post(RUNS.format(rid="api35-r1"),
                          json={"model_id": mid}).json()
    w2 = api_client.post(RUNS.format(rid="api35-r2"),
                         json={"model_id": mid}).json()
    adhoc = api_client.post("/api/v1/workflows/run", json={
        "name": "m35-inline", "model_id": mid,
        "stages": [stage]}).json()
    assert adhoc["recipe_id"] is None

    url = BY_RECIPE.format(mid=mid, rid="api35-r1")
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # authoritative-filter parity: exact subset of the M11 listing
    # whose persisted recipe_id matches, in the same order
    listing = api_client.get(f"/api/v1/models/{mid}/workflows").json()
    assert recs == [x for x in listing
                    if x["recipe_id"] == "api35-r1"]
    assert {x["workflow_id"] for x in recs} == {w1a["workflow_id"],
                                                w1b["workflow_id"]}
    keyed = [(x["created_at"], x["workflow_id"]) for x in recs]
    assert keyed == sorted(keyed)
    # verbatim: each element equals its POST payload and its
    # detail-getter payload (incl. recipe_hash provenance)
    by_id = {x["workflow_id"]: x for x in recs}
    assert by_id[w1a["workflow_id"]] == w1a
    assert by_id[w1b["workflow_id"]] == w1b
    definition = api_client.get(f"{RECIPES}/api35-r1").json()
    for x in recs:
        one = api_client.get(
            f"/api/v1/models/{mid}/workflows/{x['workflow_id']}")
        assert one.status_code == 200 and one.json() == x
        assert x["recipe_hash"] == definition["config_hash"]
    # deterministic: three repeats return identical raw bytes
    raws = {api_client.get(url).content for _ in range(3)}
    assert len(raws) == 1
    # partition: r2 holds exactly its own run, disjoint from r1's
    # group; ad-hoc (None) runs belong to NO group
    recs2 = api_client.get(
        BY_RECIPE.format(mid=mid, rid="api35-r2")).json()
    assert [x["workflow_id"] for x in recs2] == [w2["workflow_id"]]
    ids1 = {x["workflow_id"] for x in recs}
    ids2 = {x["workflow_id"] for x in recs2}
    assert ids1.isdisjoint(ids2)
    attributed = {x["workflow_id"] for x in listing
                  if x["recipe_id"] is not None}
    assert ids1 | ids2 == attributed
    assert adhoc["workflow_id"] not in attributed
    # valid registered recipe with zero runs for the model -> [] (200)
    empty = api_client.get(BY_RECIPE.format(mid=mid, rid="api35-r3"))
    assert empty.status_code == 200 and empty.json() == []
    # no side effects: the filter itself added no runs
    assert len(api_client.get(
        f"/api/v1/models/{mid}/workflows").json()) == 4


def test_m35_api_404s_isolation_regressions_openapi(api_client):
    h = _http_env(api_client, "m35b")
    mid, suite = h["mid"], h["suite"]
    h2 = _http_env(api_client, "m35c")     # second real model
    mid2 = h2["mid"]
    stage = _current_suite_stage("eval_suite", suite)
    assert api_client.post(RECIPES, json={
        "recipe_id": "api35-r4", "description": "d",
        "stages": [stage]}).status_code == 201
    w1 = api_client.post(RUNS.format(rid="api35-r4"),
                         json={"model_id": mid}).json()

    # 404s: unknown model / unknown recipe (two ghost forms); a valid
    # recipe never makes an unknown model valid
    assert api_client.get(BY_RECIPE.format(
        mid="ghost-model-35", rid="api35-r4")).status_code == 404
    assert api_client.get(BY_RECIPE.format(
        mid=mid, rid="ghost-recipe-35")).status_code == 404
    assert api_client.get(
        f"/api/v1/models/{mid}/workflows/by-recipe/"
        "m35--not-a-real-recipe-id").status_code == 404
    assert api_client.get(BY_RECIPE.format(
        mid="ghost-model-35", rid="ghost-recipe-35")).status_code == 404

    # cross-model isolation: recipes are global, but the other model's
    # group is empty (scoping from the model's own listing)
    iso = api_client.get(BY_RECIPE.format(mid=mid2, rid="api35-r4"))
    assert iso.status_code == 200 and iso.json() == []

    # M11 listing/getter intact; generic ghost workflow id 404 (no
    # route capture)
    listing = api_client.get(f"/api/v1/models/{mid}/workflows").json()
    assert [x["workflow_id"] for x in listing] == [w1["workflow_id"]]
    got = api_client.get(
        f"/api/v1/models/{mid}/workflows/{w1['workflow_id']}")
    assert got.status_code == 200 and got.json() == w1
    assert api_client.get(
        f"/api/v1/models/{mid}/workflows/ghost-wf-35").status_code == 404

    # M12 regression: the GLOBAL cross-model recipe-runs surface still
    # works unchanged (same record, its own model_id; unknown 404)
    global_runs = api_client.get(RUNS.format(rid="api35-r4")).json()
    assert [x["workflow_id"] for x in global_runs] == [w1["workflow_id"]]
    assert global_runs[0]["model_id"] == mid
    assert api_client.get(RUNS.format(rid="ghost-recipe-35")
                          ).status_code == 404
    # M34 regression: gate-decision by-comparison route intact
    assert api_client.get(
        f"/api/v1/models/{mid}/gates/decisions/by-comparison/"
        "ghost-comp-35").status_code == 404

    # OpenAPI: 67 paths, the new path exactly once, GET-only, tag
    # workflows, WorkflowRecord items; route order M11 listing <
    # by-recipe < generic workflow detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer) + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer) + 1 (M34 gate decisions
    # by-comparison) + 1 (M35 workflows by-recipe) = 67
    assert len(spec["paths"]) == 81
    path = "/api/v1/models/{model_id}/workflows/by-recipe/{recipe_id}"
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["workflows"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/WorkflowRecord"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/workflows") \
        < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/workflows/{workflow_id}")


# =========================================================================== #
# M42: read-only per-status grouping of the workflow history
# =========================================================================== #

def _wf_manifest_files(env, model_ids) -> set[str]:
    out = set()
    for mid in model_ids:
        root = env.forge.storage.model_dir(mid) / "workflows"
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                out.add(f"{mid}:{p.relative_to(root).as_posix()}")
    return out


def _m42_state(env):
    """M42 state on top of the shared module env (cached): ALL THREE
    statuses deterministically — one COMPLETED suite run, one FAILED
    run (a stage referencing an unknown suite; the failure IS
    persisted), one STOPPED run (a checkpoint-baseline gate whose
    regressed candidate fails with no on_fail branch) — plus a fresh
    model with ONE completed run (cross-model isolation partner) and
    a fresh model with NO workflows (the natural valid-empty case).
    Returns (w_completed, w_failed, w_stopped, fresh, w_fresh,
    empty_model).
    """
    cached = getattr(env, "_m42", None)
    if cached is not None:
        return cached
    f = env.forge
    w_completed = f.run_workflow(WorkflowPlan(
        name="m42-done", model_id=env.model_id,
        stages=[_suite_stage("s1", "m12-suite", ckpt=env.ck_main)]))
    with pytest.raises(FileNotFoundError):
        f.run_workflow(WorkflowPlan(
            name="m42-fail", model_id=env.model_id,
            stages=[_suite_stage("s1", "m42-no-such-suite", current=True)]))
    w_failed = f.list_workflows(env.model_id)[-1]
    w_stopped = f.run_workflow(WorkflowPlan(
        name="m42-stop", model_id=env.model_id,
        stages=[WorkflowStage(
            stage_id="g1", type=StageType.GATE,
            gate=WorkflowGateStage(
                policy=GatePolicy(
                    name="m42-gate", model_id=env.model_id,
                    dataset_id=env.ds_a, split="validation",
                    tokenizer_id=env.tok_id, batch_size=8,
                    max_seq_len=32, seed=4201, tolerance=1e-4,
                    baseline_type="checkpoint",
                    baseline_checkpoint_id=env.ck_late),
                candidate=StageStateRef(
                    state_kind=EvalStateKind.CHECKPOINT,
                    checkpoint_id=env.ck_main)))]))
    fresh = env.fresh_model("m42-fresh")
    w_fresh = f.run_workflow(WorkflowPlan(
        name="m42-fresh-run", model_id=fresh,
        stages=[_suite_stage("s1", "m12-suite", current=True)]))
    empty_model = env.fresh_model("m42-empty")
    assert w_completed.status == WorkflowStatus.COMPLETED
    assert w_failed.status == WorkflowStatus.FAILED
    assert w_stopped.status == WorkflowStatus.STOPPED
    assert w_fresh.status == WorkflowStatus.COMPLETED
    env._m42 = (w_completed, w_failed, w_stopped, fresh, w_fresh,
                empty_model)
    return env._m42


def test_m42_engine_filters_by_persisted_status_identity(env):
    w_completed, w_failed, w_stopped, fresh, w_fresh, empty_model = \
        _m42_state(env)
    f = env.forge
    listing = f.list_workflows(env.model_id)
    for status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED,
                   WorkflowStatus.STOPPED):
        got = f.list_workflows_for_status(env.model_id, status)
        # parity with the authoritative M11 listing filtered by the
        # persisted status; deterministic (created_at, workflow_id)
        # order; membership from the persisted field only
        assert got == [w for w in listing if w.status == status]
        keyed = [(x.created_at, x.workflow_id) for x in got]
        assert keyed == sorted(keyed)
        ids = [x.workflow_id for x in got]
        assert len(ids) == len(set(ids))
        assert all(x.status == status and x.model_id == env.model_id
                   for x in got)
        # verbatim payload parity with the M11 single-record getter
        for x in got:
            assert x == f.get_workflow(env.model_id, x.workflow_id)
    # the fixture's records land in their own groups with the
    # persisted status VERBATIM (earlier module tests may have added
    # other runs — membership derives from the listing)
    got_c = f.list_workflows_for_status(env.model_id,
                                        WorkflowStatus.COMPLETED)
    got_f = f.list_workflows_for_status(env.model_id,
                                        WorkflowStatus.FAILED)
    got_s = f.list_workflows_for_status(env.model_id,
                                        WorkflowStatus.STOPPED)
    assert w_completed.workflow_id in {x.workflow_id for x in got_c}
    assert w_failed.workflow_id in {x.workflow_id for x in got_f}
    assert w_stopped.workflow_id in {x.workflow_id for x in got_s}
    # explicit partition: pairwise-disjoint groups over ALL THREE enum
    # values whose union is the full listing
    ids_c = {x.workflow_id for x in got_c}
    ids_f = {x.workflow_id for x in got_f}
    ids_s = {x.workflow_id for x in got_s}
    assert ids_c and ids_f and ids_s
    assert ids_c.isdisjoint(ids_f) and ids_c.isdisjoint(ids_s)
    assert ids_f.isdisjoint(ids_s)
    assert ids_c | ids_f | ids_s == {w.workflow_id for w in listing}


def test_m42_engine_empty_404s_model_scoping_read_only(env):
    w_completed, w_failed, w_stopped, fresh, w_fresh, empty_model = \
        _m42_state(env)
    f = env.forge
    # a model whose ENTIRE M11 listing is empty -> [] for ALL THREE
    # statuses (the fixture created it with no runs — never 404)
    assert f.list_workflows(empty_model) == []
    for status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED,
                   WorkflowStatus.STOPPED):
        assert f.list_workflows_for_status(empty_model, status) == []
    # unknown model -> FileNotFoundError (404 at the API); the enum
    # itself needs NO registry lookup (unsupported values are 422 at
    # the API boundary and never reach the engine)
    with pytest.raises(FileNotFoundError):
        f.list_workflows_for_status("ghost-model-42",
                                    WorkflowStatus.COMPLETED)
    # cross-model isolation: both models hold runs, the listings are
    # disjoint, each group is a subset of its own model's listing,
    # and the fresh model's failed/stopped groups are the natural
    # valid-empty (it owns only its completed run)
    a_ids = {w.workflow_id for w in f.list_workflows(env.model_id)}
    b_ids = {w.workflow_id for w in f.list_workflows(fresh)}
    assert a_ids and b_ids and a_ids.isdisjoint(b_ids)
    for status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED,
                   WorkflowStatus.STOPPED):
        for mid, ids in ((env.model_id, a_ids), (fresh, b_ids)):
            group = {x.workflow_id for x in
                     f.list_workflows_for_status(mid, status)}
            assert group <= ids
    assert [x.workflow_id for x in f.list_workflows_for_status(
        fresh, WorkflowStatus.COMPLETED)] == [w_fresh.workflow_id]
    assert f.list_workflows_for_status(
        fresh, WorkflowStatus.FAILED) == []
    assert f.list_workflows_for_status(
        fresh, WorkflowStatus.STOPPED) == []
    # read-only: the filter never writes workflow manifests
    before = _wf_manifest_files(env, [env.model_id, fresh, empty_model])
    for status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED,
                   WorkflowStatus.STOPPED):
        f.list_workflows_for_status(env.model_id, status)
        f.list_workflows_for_status(fresh, status)
        f.list_workflows_for_status(empty_model, status)
    after = _wf_manifest_files(env, [env.model_id, fresh, empty_model])
    assert after == before


def test_m42_engine_repeated_calls_identical(env):
    w_completed, w_failed, w_stopped, fresh, w_fresh, empty_model = \
        _m42_state(env)
    f = env.forge
    first = [w.model_dump(mode="json") for w in
             f.list_workflows_for_status(env.model_id,
                                         WorkflowStatus.COMPLETED)]
    for _ in range(3):
        again = [w.model_dump(mode="json") for w in
                 f.list_workflows_for_status(env.model_id,
                                             WorkflowStatus.COMPLETED)]
        assert again == first


# --------------------------------------------------------------------------- #
# M42 API: workflow history by status (read-only grouping, enum contract)
# --------------------------------------------------------------------------- #

BY_STATUS = "/api/v1/models/{mid}/workflows/by-status/{status}"


def _m42_gate_stage_dict(mid, ds, tok) -> dict:
    """A minimum-loss gate over the model's CURRENT state with an
    impossible ceiling: the decision fails deterministically and the
    run (no on_fail branch) STOPs."""
    return {"stage_id": "g1", "type": "gate",
            "gate": {"policy": {
                "name": "api42-gate", "model_id": mid, "dataset_id": ds,
                "split": "validation", "tokenizer_id": tok,
                "batch_size": 8, "max_seq_len": 32, "seed": 4202,
                "tolerance": 1e-4, "baseline_type": "minimum_loss",
                "minimum_loss": 1e-9},
                "candidate": {"state_kind": "current"}}}


def test_m42_api_by_status_grouping_partition_determinism(api_client):
    h = _http_env(api_client, "m42a")
    mid, suite = h["mid"], h["suite"]
    # one run per status: completed (current-state suite run),
    # stopped (impossible minimum-loss ceiling, no on_fail), failed
    # (unknown suite at run time — the failure IS persisted)
    w_c = api_client.post("/api/v1/workflows/run", json={
        "name": "m42-done", "model_id": mid,
        "stages": [_current_suite_stage("s1", suite)]}).json()
    w_s = api_client.post("/api/v1/workflows/run", json={
        "name": "m42-stop", "model_id": mid,
        "stages": [_m42_gate_stage_dict(mid, h["ds"], h["tok"])]}).json()
    n_before = len(api_client.get(f"{MODELS}/{mid}/workflows").json())
    r = api_client.post("/api/v1/workflows/run", json={
        "name": "m42-fail", "model_id": mid,
        "stages": [_current_suite_stage("s1", "m42-no-such-suite")]})
    assert r.status_code == 404                     # missing input
    listing = api_client.get(f"{MODELS}/{mid}/workflows").json()
    assert len(listing) == n_before + 1
    w_f = listing[-1]
    assert w_c["status"] == "completed"
    assert w_s["status"] == "stopped"
    assert w_f["status"] == "failed" and w_f["failed_stage_id"] == "s1"

    # groups for ALL THREE statuses: authoritative-filter parity,
    # ordering, verbatim detail-getter payloads
    for status in ("completed", "failed", "stopped"):
        got = api_client.get(BY_STATUS.format(mid=mid, status=status))
        assert got.status_code == 200, got.text
        recs = got.json()
        assert recs == [x for x in listing if x["status"] == status]
        keyed = [(x["created_at"], x["workflow_id"]) for x in recs]
        assert keyed == sorted(keyed)
        for x in recs:
            one = api_client.get(
                f"{MODELS}/{mid}/workflows/{x['workflow_id']}")
            assert one.status_code == 200 and one.json() == x
    # deterministic: three repeats per status return identical bytes
    for status in ("completed", "failed", "stopped"):
        raws = {api_client.get(
            BY_STATUS.format(mid=mid, status=status)).content
                for _ in range(3)}
        assert len(raws) == 1
    # partition: the three groups are pairwise disjoint and cover the
    # full listing exactly
    ids = {st: {x["workflow_id"] for x in api_client.get(
        BY_STATUS.format(mid=mid, status=st)).json()}
        for st in ("completed", "failed", "stopped")}
    assert ids["completed"].isdisjoint(ids["failed"])
    assert ids["completed"].isdisjoint(ids["stopped"])
    assert ids["failed"].isdisjoint(ids["stopped"])
    assert ids["completed"] | ids["failed"] | ids["stopped"] == \
        {x["workflow_id"] for x in listing}
    assert w_c["workflow_id"] in ids["completed"]
    assert w_f["workflow_id"] in ids["failed"]
    assert w_s["workflow_id"] in ids["stopped"]
    # no execution side effects: the filter itself added no runs
    assert len(api_client.get(
        f"{MODELS}/{mid}/workflows").json()) == n_before + 1


def test_m42_api_by_status_404_422s_isolation_regressions_openapi(
        api_client):
    h = _http_env(api_client, "m42b")
    mid, suite = h["mid"], h["suite"]
    other = _http_env(api_client, "m42c")     # second real model
    stage = _current_suite_stage("s1", suite)
    w1 = api_client.post("/api/v1/workflows/run", json={
        "name": "m42-one", "model_id": mid,
        "stages": [stage]}).json()

    # 404: unknown model with a VALID status (exactly like the
    # sibling grouping)
    assert api_client.get(BY_STATUS.format(
        mid="ghost-model-42", status="completed")).status_code == 404
    # 422: unsupported status values are rejected by the schema enum
    # at the API boundary — before the handler, so the 422 wins even
    # for an UNKNOWN model (never a registry-style 404, never []; the
    # mid-flight 'running' value is deliberately NOT modelled)
    for bad in ("COMPLETED", "compl%20eted", "1", "running"):
        got = api_client.get(BY_STATUS.format(mid=mid, status=bad))
        assert got.status_code == 422, (bad, got.status_code)
    assert api_client.get(BY_STATUS.format(
        mid="ghost-model-42", status="running")).status_code == 422

    # cross-model isolation: the other model has NO workflows, so all
    # three groups are the natural valid empty
    for status in ("completed", "failed", "stopped"):
        iso = api_client.get(BY_STATUS.format(mid=other["mid"],
                                              status=status))
        assert iso.status_code == 200 and iso.json() == []

    # M11 listing/getter intact; the generic detail getter still 404s
    # ghost ids (no route capture by the new literal segment)
    listing = api_client.get(f"{MODELS}/{mid}/workflows").json()
    assert [x["workflow_id"] for x in listing] == [w1["workflow_id"]]
    assert api_client.get(
        f"{MODELS}/{mid}/workflows/{w1['workflow_id']}").json() == w1
    assert api_client.get(
        f"{MODELS}/{mid}/workflows/ghost-wf-42").status_code == 404

    # M35 by-recipe regression: register + run a recipe, the
    # by-recipe group is exactly the recipe-attributed runs with
    # listing parity (ad-hoc runs never appear)
    assert api_client.post(RECIPES, json={
        "recipe_id": "api42-r1", "description": "d",
        "stages": [stage]}).status_code == 201
    wr = api_client.post(RUNS.format(rid="api42-r1"),
                         json={"model_id": mid}).json()
    listing = api_client.get(f"{MODELS}/{mid}/workflows").json()
    byrec = api_client.get(
        f"{MODELS}/{mid}/workflows/by-recipe/api42-r1")
    assert byrec.status_code == 200
    assert byrec.json() == [x for x in listing
                            if x["recipe_id"] == "api42-r1"]
    assert [x["workflow_id"] for x in byrec.json()] == \
        [wr["workflow_id"]]
    # and the by-status view of the same listing stays coherent
    assert api_client.get(BY_STATUS.format(
        mid=mid, status="completed")).json() == \
        [x for x in listing if x["status"] == "completed"]

    # M41 gate-decision by-decision regression: this harness has no
    # gate decisions, so both groups are the natural valid empty with
    # listing parity
    gds = api_client.get(f"{MODELS}/{mid}/gates/decisions").json()
    assert gds == []
    for decision in ("passed", "failed"):
        g = api_client.get(
            f"{MODELS}/{mid}/gates/decisions/by-decision/{decision}")
        assert g.status_code == 200
        assert g.json() == [d for d in gds
                            if d["decision"] == decision]
    # M40 samples-by-strategy regression: natural valid empty too
    samples = api_client.get(f"{MODELS}/{mid}/samples").json()
    assert samples == []
    for strategy in ("greedy", "temperature"):
        g = api_client.get(
            f"{MODELS}/{mid}/samples/by-strategy/{strategy}")
        assert g.status_code == 200
        assert g.json() == [x for x in samples
                            if x["strategy"] == strategy]

    # OpenAPI: 77 paths, the new path exactly once, GET-only, tag
    # workflows, WorkflowRecord items, status $ref WorkflowStatus;
    # route order M35 by-recipe < by-status < generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind)
    # + 1 (M39 comparisons by-verdict)
    # + 1 (M40 samples by-strategy)
    # + 1 (M41 gate decisions by-decision)
    # + 1 (M42 workflows by-status)
    # + 1 (M43 gate decisions by-verdict)
    # + 1 (M44 comparisons by-state-kind)
    # + 1 (M45 gate decisions by-baseline-type)
    # + 1 (M46 checkpoints by-run)
    # + 1 (M47 evaluations by-truncated)
    # + 1 (M48 evaluations by-seed)
    # + 1 (M49 comparisons by-seed) = 81
    assert len(spec["paths"]) == 81
    path = "/api/v1/models/{model_id}/workflows/by-status/{status}"
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["workflows"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/WorkflowRecord"}
    status_param = [p for p in item["get"]["parameters"]
                    if p["name"] == "status"][0]
    assert status_param["schema"] == {
        "$ref": "#/components/schemas/WorkflowStatus"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/workflows/by-recipe/"
                      "{recipe_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/workflows/{workflow_id}")
