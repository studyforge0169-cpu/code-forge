"""Milestone 7 tests: workflow engine (ordered orchestration over M3–M6).

Reuses the M5/M6 two-domain recipe: a module-scoped environment with one
model trained on domain A (checkpoints improve on the A probe) then continued
on domain B (the final checkpoint regresses on the A probe). Workflow stages
that TRAIN use dedicated small models so the shared environment's measured
checkpoint deltas stay valid; evaluation/compare/gate stages never mutate
weights, so they run against the shared model.
"""
from __future__ import annotations

import json
import random

import pytest
from pydantic import ValidationError

from app.schemas import (
    EvalStateKind,
    EvaluationConfig,
    GateDecisionResult,
    GatePolicy,
    ModelCreateRequest,
    ProbeSuiteCreateRequest,
    StageStateRef,
    StageType,
    SuiteProbe,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
    WorkflowEvaluationStage,
    WorkflowGateStage,
    WorkflowPlan,
    WorkflowStage,
    WorkflowStatus,
    WorkflowSuiteRunStage,
)

HEADS = ("river mountain cloud forest desert ocean valley island meadow canyon "
         "table chair lamp desk shelf couch rug clock mirror vase").split()
TAIL_A = ("flows stands gleams rises falls drifts looms shines hides waits").split()
TAIL_B = ("sings dances jumps laughs weeps shouts whispers roars murmurs chants").split()


def _domain_bytes(tails, n: int) -> bytes:
    rng = random.Random(11)
    out = []
    for i in range(n):
        h1, h2 = rng.sample(HEADS, 2)
        out.append(f"{h1} is {rng.choice(tails)} near {h2} with number {i}")
    return ("\n\n".join(out) + "\n").encode("utf-8")


class Env:
    """One temp root with two domains, one tokenizer, one trained model."""

    def __init__(self, root):
        from app.engine import ModelForge

        self.forge = ModelForge(root=root)

    def prepare(self):
        f = self.forge
        up_a = f.upload_dataset([("a.txt", _domain_bytes(TAIL_A, 240))], name="m7-domA")
        up_b = f.upload_dataset([("b.txt", _domain_bytes(TAIL_B, 240))], name="m7-domB")
        self.ds_a, self.ds_b = up_a["dataset_id"], up_b["dataset_id"]
        tok = f.train_tokenizer(TokenizerConfig(name="m7-tok", vocab_size=600),
                                dataset_id=self.ds_a)
        self.tok_id = tok.id
        f.tokenize_dataset(self.ds_a, tok.id)
        f.tokenize_dataset(self.ds_b, tok.id)

        cfg = TransformerConfig(name="m7-main", vocab_size=640, context_length=64,
                                hidden_size=64, n_layers=2, n_heads=4,
                                n_kv_heads=2, intermediate_size=128, seed=1)
        self.model_id = f.create_model(ModelCreateRequest(config=cfg))[0].id
        self.tokens_a = f.datasets.tokenized_artifact(
            self.ds_a, 1, tok.id)[0].splits["train"].count
        self.tokens_b = f.datasets.tokenized_artifact(
            self.ds_b, 1, tok.id)[0].splits["train"].count

        r1 = f.run_training(self._train(self.ds_a, 30))
        ck = r1.checkpoints
        assert len(ck) == 15
        self.imp_early = f.get_checkpoint(self.model_id, ck[2]["checkpoint_id"])
        self.imp_final = f.get_checkpoint(self.model_id, ck[-1]["checkpoint_id"])
        r2 = f.run_training(self._train(self.ds_b, 40))
        self.reg_final = f.get_checkpoint(self.model_id,
                                          r2.checkpoints[-1]["checkpoint_id"])

    def _train(self, ds_id: str, epochs: int) -> TrainingConfig:
        tokens = self.tokens_a if ds_id == self.ds_a else self.tokens_b
        sp_epoch = max(1, (tokens // 32) // 8)
        return TrainingConfig(method="continued_pretraining",
                              model_id=self.model_id, dataset_id=ds_id,
                              tokenizer_id=self.tok_id, learning_rate=3e-3,
                              batch_size=8, max_seq_len=32, epochs=epochs,
                              eval_every_steps=sp_epoch * 2, keep_best=False,
                              seed=1)

    # ------------------------------------------------------------------ #
    # Stage/plan builders (explicit references, never guessed)
    # ------------------------------------------------------------------ #

    def train_stage(self, sid: str, model_id=None, ds=None, epochs=2,
                    eval_every=7, seed=1, method="continued_pretraining",
                    name="w-run") -> WorkflowStage:
        mid = model_id or self.model_id
        return WorkflowStage(
            stage_id=sid, type=StageType.TRAIN,
            training=TrainingConfig(
                name=name, method=method, model_id=mid,
                dataset_id=ds or self.ds_a, tokenizer_id=self.tok_id,
                learning_rate=3e-3, batch_size=8, max_seq_len=32,
                epochs=epochs, eval_every_steps=eval_every, keep_best=False,
                seed=seed))

    def evaluate_stage(self, sid: str, *, ckpt=None, from_stage=None,
                       seed=1, ds=None, split="validation", model_id=None,
                       **cfg_kw) -> WorkflowStage:
        mid = model_id or self.model_id
        cfg = dict(model_id=mid, dataset_id=ds or self.ds_a, split=split,
                   tokenizer_id=self.tok_id, batch_size=8, max_seq_len=32,
                   seed=seed)
        cfg.update(cfg_kw)
        if ckpt is not None:
            cfg["checkpoint_id"] = ckpt
        return WorkflowStage(
            stage_id=sid, type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=EvaluationConfig(**cfg),
                checkpoint_from_stage=from_stage))

    def compare_stage(self, sid: str, *, state_a, state_b, seed=1, ds=None,
                      split="validation", tolerance=1e-4, model_id=None,
                      **kw) -> WorkflowStage:
        from app.schemas import WorkflowComparisonStage

        return WorkflowStage(
            stage_id=sid, type=StageType.COMPARE,
            comparison=WorkflowComparisonStage(
                state_a=state_a, state_b=state_b,
                dataset_id=ds or self.ds_a, split=split,
                tokenizer_id=self.tok_id, batch_size=8, max_seq_len=32,
                seed=seed, tolerance=tolerance, **kw))

    def gate_stage(self, sid: str, *, candidate=None, ckpt=None, from_stage=None,
                   baseline_type="checkpoint", baseline_ckpt=None,
                   baseline_hash=None, minimum_loss=None,
                   max_regression_delta=None, tolerance=1e-4, seed=1,
                   on_pass=None, on_fail=None, model_id=None, **pol_kw) -> WorkflowStage:
        mid = model_id or self.model_id
        if candidate is None:
            candidate = StageStateRef(
                state_kind=EvalStateKind.CHECKPOINT,
                **({"checkpoint_id": ckpt} if ckpt is not None
                   else {"from_stage": from_stage}))
        pol = dict(name="w-gate", model_id=mid, dataset_id=self.ds_a,
                   split="validation", tokenizer_id=self.tok_id, batch_size=8,
                   max_seq_len=32, seed=seed, tolerance=tolerance,
                   baseline_type=baseline_type,
                   baseline_checkpoint_id=baseline_ckpt,
                   baseline_result_hash=baseline_hash,
                   minimum_loss=minimum_loss,
                   max_regression_delta=max_regression_delta)
        pol.update(pol_kw)
        for drop in ("baseline_checkpoint_id", "baseline_result_hash",
                     "minimum_loss", "max_regression_delta"):
            if pol[drop] is None:
                del pol[drop]
        return WorkflowStage(
            stage_id=sid, type=StageType.GATE,
            gate=WorkflowGateStage(policy=GatePolicy(**pol), candidate=candidate),
            on_pass=on_pass, on_fail=on_fail)

    def state(self, *, ckpt=None, from_stage=None, kind="checkpoint") -> StageStateRef:
        return StageStateRef(
            state_kind=EvalStateKind.CHECKPOINT if kind == "checkpoint"
            else EvalStateKind.CURRENT,
            **({"checkpoint_id": ckpt} if ckpt is not None
               else {"from_stage": from_stage} if from_stage is not None else {}))

    def plan(self, name: str, stages, model_id=None) -> WorkflowPlan:
        return WorkflowPlan(name=name, model_id=model_id or self.model_id,
                            stages=stages)

    def workflows_root(self, model_id=None):
        root = (self.forge.storage.model_dir(model_id or self.model_id)
                / "workflows")
        return root if root.exists() else []

    # ------------------------------------------------------------------ #
    # Tiny dedicated models (training stages / corruption tests)
    # ------------------------------------------------------------------ #

    def fresh_model(self, name="m7-fresh", seed=5) -> str:
        return self.forge.create_model(ModelCreateRequest(config=TransformerConfig(
            name=name, vocab_size=640, context_length=64, hidden_size=64,
            n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
            seed=seed)))[0].id

    def no_tmp(self, model_id=None):
        root = self.forge.storage.model_dir(model_id or self.model_id)
        leftovers = [p for p in root.rglob("*")
                     if p.is_file() and ".tmp" in p.name]
        assert leftovers == []


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m7-root"))
    e.prepare()
    return e


# =========================================================================== #
# Plan validation
# =========================================================================== #


def _snapshot(model_dir):
    return {p.relative_to(model_dir).as_posix(): p.read_bytes()
            for p in model_dir.rglob("*") if p.is_file()}


def _expected_new_files(f, model_id, rec, before):
    """Exact set of manifest files a completed/stopped run may add, relative
    to the model dir: its own run manifest, the stage artifacts it records,
    and any gate-internal candidate/baseline evaluations or comparison that
    did not already exist (identity reuse never rewrites pre-existing files)."""
    expected = {f"workflows/workflow-{rec.workflow_id}/manifest.json"}
    for st in rec.stages:
        art = st.artifact
        if art is None:
            continue
        kind, aid = art.kind.value, art.artifact_id
        if kind == "evaluation":
            expected.add(f"evaluations/eval-{aid}/manifest.json")
        elif kind == "comparison":
            expected.add(f"comparisons/comp-{aid}/manifest.json")
        elif kind == "gate_decision":
            expected.add(f"gates/gate-{aid}/manifest.json")
            dec = f.get_gate_decision(model_id, aid)
            expected.add(f"evaluations/eval-{dec.candidate.evaluation_id}"
                         f"/manifest.json")
            if dec.baseline is not None:
                expected.add(f"evaluations/eval-{dec.baseline.evaluation_id}"
                             f"/manifest.json")
            if dec.comparison_id:
                expected.add(f"comparisons/comp-{dec.comparison_id}"
                             f"/manifest.json")
    return {p for p in expected if p not in before}



def test_valid_linear_plan_constructs(env):
    plan = env.plan("linear", [env.evaluate_stage("s1", ckpt=env.imp_early.checkpoint_id),
                               env.gate_stage("s2", ckpt=env.imp_final.checkpoint_id,
                                              baseline_ckpt=env.imp_early.checkpoint_id)])
    assert len(plan.stages) == 2
    assert len(plan.plan_hash()) == 64
    assert plan.plan_hash() == plan.plan_hash()


def test_unknown_stage_reference_rejected(env):
    a = env.imp_final.checkpoint_id
    with pytest.raises(ValidationError):
        env.plan("x", [env.evaluate_stage("s1", ckpt=a),
                       env.evaluate_stage("s2", from_stage="ghost")])
    with pytest.raises(ValidationError):
        env.plan("x", [env.train_stage("s1"),
                       env.gate_stage("s2", from_stage="ghost")])
    with pytest.raises(ValidationError):          # on_fail -> unknown stage
        env.plan("x", [env.gate_stage("s1", ckpt=a, baseline_ckpt=a,
                                      on_fail="ghost"),
                       env.evaluate_stage("s2", ckpt=a)])
    with pytest.raises(ValidationError):          # on_pass -> unknown stage
        env.plan("x", [env.gate_stage("s1", ckpt=a, baseline_ckpt=a,
                                      on_pass="ghost"),
                       env.evaluate_stage("s2", ckpt=a)])


def test_duplicate_stage_ids_rejected(env):
    s = env.evaluate_stage("s1", ckpt=env.imp_final.checkpoint_id)
    with pytest.raises(ValidationError, match="duplicate stage ids"):
        env.plan("x", [s, s])


def test_invalid_stage_type_and_payload_rejected(env):
    a = env.imp_final.checkpoint_id
    train_payload = env.train_stage("s1").training
    with pytest.raises(ValidationError):          # wrong payload for type
        WorkflowStage(stage_id="s1", type=StageType.TRAIN,
                      evaluation=env.evaluate_stage("s1", ckpt=a).evaluation)
    with pytest.raises(ValidationError):          # two payloads
        WorkflowStage(stage_id="s1", type=StageType.TRAIN,
                      training=train_payload,
                      evaluation=env.evaluate_stage("s1", ckpt=a).evaluation)
    with pytest.raises(ValidationError):          # zero payloads
        WorkflowStage(stage_id="s1", type=StageType.TRAIN)
    with pytest.raises(ValidationError):          # on_* on non-gate
        WorkflowStage(stage_id="s1", type=StageType.EVALUATE,
                      evaluation=env.evaluate_stage("s1", ckpt=a).evaluation,
                      on_fail="s2")


def test_circular_and_backward_transitions_rejected(env):
    a = env.imp_final.checkpoint_id
    g1 = env.gate_stage("g1", ckpt=a, baseline_ckpt=a)
    g2 = env.gate_stage("g2", ckpt=a, baseline_ckpt=a)
    with pytest.raises(ValidationError):          # backward on_fail (cycle)
        env.plan("x", [g1, g2.model_copy(update={"on_fail": "g1"})])
    with pytest.raises(ValidationError):          # self-loop
        env.plan("x", [g1.model_copy(update={"on_fail": "g1"}), g2])
    with pytest.raises(ValidationError):          # backward from_stage
        env.plan("x", [env.train_stage("s1", ds=env.ds_a),
                       env.train_stage("s2", ds=env.ds_a),
                       env.evaluate_stage("s3", from_stage="s3")])
    with pytest.raises(ValidationError):          # from_stage not a train stage
        env.plan("x", [env.evaluate_stage("s1", ckpt=a),
                       env.evaluate_stage("s2", from_stage="s1")])
    # forward transitions ARE legal: this plan is valid
    env.plan("x", [g1.model_copy(update={"on_pass": "g2"}), g2,
                   env.evaluate_stage("s3", ckpt=a)])
    with pytest.raises(ValidationError):          # from_stage -> later stage
        env.plan("x", [env.train_stage("s1"),
                       env.evaluate_stage("s2", from_stage="s3"),
                       env.train_stage("s3")])


def test_missing_referenced_artifact_fails_cleanly(env):
    """A skipped train stage has no checkpoint: the consumer fails cleanly,
    the failure is recorded, and NO artifact id is fabricated."""
    f = env.forge
    mid = env.fresh_model("m7-miss-ref")
    # g1 fails (absolute threshold) -> on_fail s3; s2 (train) is SKIPPED; s3
    # evaluates s2's final checkpoint which never came into existence
    plan = env.plan("missref", [
        env.gate_stage("g1", baseline_type="minimum_loss", minimum_loss=1e-6,
                       model_id=mid, on_fail="s3",
                       candidate=env.state(kind="current")),
        env.train_stage("s2", model_id=mid, ds=env.ds_a),
        env.evaluate_stage("s3", from_stage="s2", model_id=mid)], model_id=mid)
    weights_before = f.storage.weights_path(mid).read_bytes()
    n_runs = len(f.list_workflows(mid))
    with pytest.raises(FileNotFoundError, match="produced no final checkpoint"):
        f.run_workflow(plan)
    runs = f.list_workflows(mid)
    assert len(runs) == n_runs + 1
    rec = runs[-1]
    assert rec.status == WorkflowStatus.FAILED
    assert rec.failed_stage_id == "s3"
    assert "produced no final checkpoint" in rec.terminal_reason
    by_id = {s.stage_id: s for s in rec.stages}
    assert by_id["g1"].executed and by_id["g1"].artifact is not None
    assert by_id["s2"].skipped and not by_id["s2"].executed
    assert by_id["s3"].executed and by_id["s3"].artifact is None
    assert "final checkpoint" in by_id["s3"].error
    # the failed gate decision remains referenced; nothing was rolled back
    assert f.storage.weights_path(mid).read_bytes() == weights_before
    env.no_tmp(mid)


def test_unknown_model_refused_before_any_run(env):
    plan = env.plan("ghost",
                    [env.evaluate_stage("s1", ckpt="x", model_id="ghost")],
                    model_id="ghost")
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_workflow(plan)


def test_config_model_mismatch_rejected(env):
    a = env.imp_final.checkpoint_id
    other = env.fresh_model("m7-other")
    with pytest.raises(ValidationError):          # training config other model
        env.plan("x", [env.train_stage("s1", model_id=other)], model_id=env.model_id)
    with pytest.raises(ValidationError):          # gate policy other model
        s = env.gate_stage("g1", ckpt=a, baseline_ckpt=a, model_id=other)
        env.plan("x", [s], model_id=env.model_id)
    with pytest.raises(ValidationError):          # evaluate stage other model
        env.plan("x", [env.evaluate_stage("s1", ckpt=a, model_id=other)],
                 model_id=env.model_id)
    with pytest.raises(ValidationError):          # ambiguous eval ref+literal
        env.plan("x", [env.train_stage("s1", model_id=env.model_id),
                       WorkflowStage(
                           stage_id="s2", type=StageType.EVALUATE,
                           evaluation=WorkflowEvaluationStage(
                               config=EvaluationConfig(
                                   model_id=env.model_id, dataset_id=env.ds_a,
                                   checkpoint_id=a, split="validation",
                                   tokenizer_id=env.tok_id),
                               checkpoint_from_stage="s1"))])


# =========================================================================== #
# Execution: success paths
# =========================================================================== #

def test_evaluate_gate_workflow_completes(env):
    f = env.forge
    mid = env.model_id
    plan = env.plan("eval-gate", [
        env.evaluate_stage("s1", ckpt=env.imp_final.checkpoint_id, seed=7701),
        env.gate_stage("s2", ckpt=env.imp_final.checkpoint_id, seed=7701,
                       baseline_type="current")])
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    assert rec.failed_stage_id is None
    assert [s.stage_id for s in rec.stages] == ["s1", "s2"]
    assert all(s.executed and not s.skipped for s in rec.stages)
    e, g = rec.stages[0].artifact, rec.stages[1].artifact
    assert e.kind.value == "evaluation" and g.kind.value == "gate_decision"
    assert g.gate_decision == GateDecisionResult.PASSED
    assert g.verdict.value == "improved"       # current(=B-final) worse than A-final
    # evidence chains to real manifests
    ev = f.get_evaluation(mid, e.artifact_id)
    assert ev.result_hash == e.result_hash and ev.loss_nats == e.loss_nats
    assert ev.state_hash == e.state_hash
    dec = f.get_gate_decision(mid, g.artifact_id)
    assert dec.result_hash == g.result_hash
    assert dec.decision == GateDecisionResult.PASSED
    assert dec.candidate.state_hash == e.state_hash   # gate used same state
    env.no_tmp(mid)


def test_train_evaluate_gate_workflow_completes(env):
    """Explicitly-requested training stages run through the real M3 engine."""
    f = env.forge
    mid = env.fresh_model("m7-train-eg")
    ck_before = len(f.list_checkpoints(mid))
    prov_before = len(f.get_model(mid).training_provenance)
    plan = env.plan("train-eval-gate", [
        env.train_stage("s1", model_id=mid, ds=env.ds_a, epochs=2),
        env.evaluate_stage("s2", from_stage="s1", model_id=mid, seed=7702),
        env.gate_stage("s3", from_stage="s1", model_id=mid, seed=7702,
                       baseline_type="current")], model_id=mid)
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    t, e, g = [s.artifact for s in rec.stages]
    assert t.kind.value == "training_report" and t.accepted is True
    assert t.checkpoint_id and t.state_hash
    # M3 really trained: new checkpoints + provenance appended
    assert len(f.list_checkpoints(mid)) > ck_before
    assert len(f.get_model(mid).training_provenance) == prov_before + 1
    prov = f.get_model(mid).training_provenance[-1]
    assert prov.run_id == t.artifact_id                     # report referenced
    assert prov.final_checkpoint_id == t.checkpoint_id
    final_ckpt = f.get_checkpoint(mid, t.checkpoint_id)
    assert final_ckpt.weights_sha256 == t.state_hash
    assert t.final_validation_loss == final_ckpt.validation_loss
    # evaluate stage consumed the train stage's final checkpoint
    ev = f.get_evaluation(mid, e.artifact_id)
    assert ev.checkpoint_id == t.checkpoint_id
    assert ev.state_hash == t.state_hash
    assert e.loss_nats == ev.loss_nats
    # gate passed against the live weights it just produced (unchanged)
    dec = f.get_gate_decision(mid, g.artifact_id)
    assert dec.decision == GateDecisionResult.PASSED
    env.no_tmp(mid)


def test_train_evaluate_compare_gate_workflow_completes(env):
    f = env.forge
    mid = env.fresh_model("m7-full")
    cur = env.state(kind="current")
    plan = env.plan("full", [
        env.train_stage("s1", model_id=mid, ds=env.ds_a, epochs=2),
        env.evaluate_stage("s2", from_stage="s1", model_id=mid, seed=7703),
        env.compare_stage("s3", state_a=env.state(from_stage="s1"),
                          state_b=cur, model_id=mid, seed=7703),
        env.gate_stage("s4", from_stage="s1", model_id=mid, seed=7703,
                       baseline_type="current")], model_id=mid)
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    kinds = [s.artifact.kind.value for s in rec.stages]
    assert kinds == ["training_report", "evaluation", "comparison",
                     "gate_decision"]
    cmp_art = rec.stages[2].artifact
    assert cmp_art.verdict.value == "unchanged"   # current == s1 final weights
    assert cmp_art.delta_loss_nats == 0.0
    cmp_rec = f.get_comparison(mid, cmp_art.artifact_id)
    assert cmp_rec.result_hash == cmp_art.result_hash
    assert cmp_rec.state_a.checkpoint_id == rec.stages[0].artifact.checkpoint_id
    assert cmp_rec.state_b.state_kind.value == "current"
    # evaluation + comparison + gate evidence reused the same probe/state
    dec = f.get_gate_decision(mid, rec.stages[3].artifact.artifact_id)
    assert dec.decision == GateDecisionResult.PASSED
    env.no_tmp(mid)


def test_multi_stage_with_branches_completes(env):
    """Executed order preserved; gate failure branches to on_fail; all
    executed stages are recorded in plan order."""
    f = env.forge
    mid = env.model_id
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    reg = env.reg_final.checkpoint_id
    plan = env.plan("multi", [
        env.evaluate_stage("s1", ckpt=b, seed=7704),
        env.gate_stage("s2", ckpt=b, seed=7704, baseline_ckpt=a),   # passed
        env.evaluate_stage("s3", ckpt=a, seed=7704),
        env.gate_stage("s4", ckpt=reg, seed=7704, baseline_ckpt=b,
                       on_fail="s5"),                                # failed
        env.evaluate_stage("s5", ckpt=reg, seed=7704)])
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    assert [s.stage_id for s in rec.stages if s.executed] == \
        ["s1", "s2", "s3", "s4", "s5"]
    assert all(not s.skipped for s in rec.stages)
    decisions = [(t.stage_id, t.decision, t.to_stage) for t in rec.transitions]
    assert decisions[-1] == ("s5", "next", None)
    assert ("s4", "failed", "s5") in decisions
    arts = {s.stage_id: s.artifact for s in rec.stages}
    assert arts["s4"].gate_decision == GateDecisionResult.FAILED
    assert arts["s5"].kind.value == "evaluation"
    # gate decision artifacts referenced real persisted decisions
    dec = f.get_gate_decision(mid, arts["s4"].artifact_id)
    assert dec.decision == GateDecisionResult.FAILED


# =========================================================================== #
# Conditional branching
# =========================================================================== #

def test_gate_passed_on_pass_branch_executes_and_skips_middle(env):
    f = env.forge
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    plan = env.plan("onpass", [
        env.gate_stage("g1", ckpt=b, baseline_ckpt=a, on_pass="s3"),  # passed
        env.evaluate_stage("s2", ckpt=b, seed=7705),    # skipped
        env.evaluate_stage("s3", ckpt=b, seed=7705)])
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    by_id = {s.stage_id: s for s in rec.stages}
    assert by_id["g1"].executed and not by_id["g1"].skipped
    assert not by_id["s2"].executed and by_id["s2"].skipped
    assert by_id["s3"].executed and by_id["s3"].artifact is not None
    assert rec.transitions[0].decision == "passed"
    assert rec.transitions[0].to_stage == "s3"
    # s3 evaluation executed (its artifact is a real evaluation record)
    f.get_evaluation(env.model_id, by_id["s3"].artifact.artifact_id)


def test_gate_failed_on_fail_branch_executes(env):
    f = env.forge
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    plan = env.plan("onfail", [
        env.gate_stage("g1", ckpt=a, baseline_ckpt=b, on_fail="s3"),  # regressed
        env.evaluate_stage("s2", ckpt=a, seed=7706),     # skipped
        env.evaluate_stage("s3", ckpt=a, seed=7706)])
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    by_id = {s.stage_id: s for s in rec.stages}
    assert by_id["g1"].artifact.gate_decision == GateDecisionResult.FAILED
    assert not by_id["s2"].executed and by_id["s2"].skipped
    assert by_id["s3"].executed
    assert rec.transitions[0].decision == "failed"
    assert rec.transitions[0].to_stage == "s3"


def test_failed_gate_without_branch_stops(env):
    f = env.forge
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    plan = env.plan("stop", [
        env.gate_stage("g1", ckpt=a, baseline_ckpt=b),   # regressed -> stop
        env.evaluate_stage("s2", ckpt=a, seed=7707)])
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.STOPPED
    assert "stopped" in rec.terminal_reason
    assert rec.failed_stage_id is None            # a decision, not an error
    assert rec.stages[0].artifact.gate_decision == GateDecisionResult.FAILED
    assert not rec.stages[1].executed and rec.stages[1].skipped
    assert [(t.stage_id, t.decision, t.to_stage)
            for t in rec.transitions] == [("g1", "failed", None)]
    f.get_gate_decision(env.model_id, rec.stages[0].artifact.artifact_id)


def test_failed_gate_records_rollback_suggestion_without_executing(env):
    """Failed checkpoint-baseline gate -> STOPPED with suggestion; weights,
    checkpoints and provenance stay byte-identical (rollback never runs)."""
    f = env.forge
    mid = env.model_id
    model_dir = f.storage.model_dir(mid)
    before = _snapshot(model_dir)
    weights_before = f.storage.weights_path(mid).read_bytes()
    prov_before = [p.model_dump(mode="json")
                   for p in f.get_model(mid).training_provenance]
    latest_before = f.get_model(mid).latest_checkpoint

    plan = env.plan("suggest", [
        env.gate_stage("g1", ckpt=env.reg_final.checkpoint_id,
                       baseline_ckpt=env.imp_final.checkpoint_id)])
    rec = f.run_workflow(plan)
    assert rec.status == WorkflowStatus.STOPPED
    assert rec.suggested_checkpoint_id == env.imp_final.checkpoint_id
    assert rec.hint == "rollback recommended"
    dec = f.get_gate_decision(mid, rec.stages[0].artifact.artifact_id)
    assert dec.suggested_checkpoint_id == env.imp_final.checkpoint_id
    assert dec.hint == "rollback recommended"

    after = _snapshot(model_dir)
    added = set(after) - set(before)
    # exactly the run manifest + the decision's new evidence (baseline
    # evaluation already existed and was reused by identity)
    assert added == _expected_new_files(f, mid, rec, before)
    for k in before:
        assert after[k] == before[k]
    assert f.storage.weights_path(mid).read_bytes() == weights_before
    assert f.get_model(mid).latest_checkpoint == latest_before
    assert [p.model_dump(mode="json")
            for p in f.get_model(mid).training_provenance] == prov_before
    env.no_tmp(mid)


# =========================================================================== #
# Evidence + reuse + determinism
# =========================================================================== #

def test_repeat_workflow_reuses_evidence_and_hash_matches(env):
    """Repeated identical workflows: no duplicate evaluations or comparisons,
    a fresh gate decision + run manifest each time, identical result_hash."""
    f = env.forge
    mid = env.model_id
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    plan = env.plan("repeat", [
        env.evaluate_stage("s1", ckpt=a, seed=7708),
        env.compare_stage("s2", state_a=env.state(ckpt=a),
                          state_b=env.state(ckpt=b), seed=7708),
        env.gate_stage("s3", ckpt=b, seed=7708, baseline_ckpt=a)])
    n_ev, n_cmp, n_gate, n_wf = (len(f.list_evaluations(mid)),
                                 len(f.list_comparisons(mid)),
                                 len(f.list_gate_decisions(mid)),
                                 len(f.list_workflows(mid)))
    r1 = f.run_workflow(plan)
    assert len(f.list_evaluations(mid)) == n_ev + 2
    assert len(f.list_comparisons(mid)) == n_cmp + 1
    assert len(f.list_gate_decisions(mid)) == n_gate + 1
    assert len(f.list_workflows(mid)) == n_wf + 1

    r2 = f.run_workflow(plan)
    # evidence reused by identity; one new gate decision + one new run
    assert len(f.list_evaluations(mid)) == n_ev + 2
    assert len(f.list_comparisons(mid)) == n_cmp + 1
    assert len(f.list_gate_decisions(mid)) == n_gate + 2
    assert len(f.list_workflows(mid)) == n_wf + 2
    for ridx in (0, 1):          # evaluation + comparison reused by identity
        assert r1.stages[ridx].artifact.artifact_id == \
            r2.stages[ridx].artifact.artifact_id
    assert r1.stages[2].artifact.gate_decision == r2.stages[2].artifact.gate_decision
    # ...but the gate decision itself is a fresh audit event each execution
    assert r1.stages[2].artifact.artifact_id != r2.stages[2].artifact.artifact_id
    assert r1.result_hash == r2.result_hash
    assert r1.workflow_id != r2.workflow_id


def test_result_hash_excludes_ids_timestamps(env):
    f = env.forge
    a = env.imp_final.checkpoint_id
    plan = env.plan("det", [
        env.evaluate_stage("s1", ckpt=a, seed=7709),
        env.gate_stage("s2", ckpt=a, seed=7709, baseline_type="current")])
    r1 = f.run_workflow(plan)
    r2 = f.run_workflow(plan)
    assert r1.result_hash == r2.result_hash
    assert r1.workflow_id != r2.workflow_id
    assert len(r1.result_hash) == 64
    # a semantic change (tolerance) changes the hash
    plan2 = env.plan("det2", [
        env.evaluate_stage("s1", ckpt=a, seed=7709),
        env.gate_stage("s2", ckpt=a, seed=7709, baseline_type="current",
                       tolerance=0.5)])
    r3 = f.run_workflow(plan2)
    assert r3.result_hash != r1.result_hash
    # plan identity differs too
    assert r3.plan_hash != r1.plan_hash


def test_threshold_workflow_reuses_evaluation(env):
    """Absolute-threshold gate stages share candidate evaluations (M4 reuse)."""
    f = env.forge
    mid = env.model_id
    a = env.imp_final.checkpoint_id
    plan_pass = env.plan("thr-pass", [
        env.gate_stage("g1", ckpt=a, baseline_type="minimum_loss",
                       minimum_loss=1e9, seed=7710)])
    plan_fail = env.plan("thr-fail", [
        env.gate_stage("g1", ckpt=a, baseline_type="minimum_loss",
                       minimum_loss=1e-6, seed=7710)])
    n_ev = len(f.list_evaluations(mid))
    p1 = f.run_workflow(plan_pass)
    assert len(f.list_evaluations(mid)) == n_ev + 1
    p2 = f.run_workflow(plan_fail)
    assert len(f.list_evaluations(mid)) == n_ev + 1     # evaluation reused
    assert p1.status == WorkflowStatus.COMPLETED
    assert p2.status == WorkflowStatus.STOPPED
    # each run is its own audit event, but both decisions cite the SAME
    # candidate evaluation artifact (M4 identity reuse)
    d1 = f.get_gate_decision(mid, p1.stages[0].artifact.artifact_id)
    d2 = f.get_gate_decision(mid, p2.stages[0].artifact.artifact_id)
    assert d1.candidate.evaluation_id == d2.candidate.evaluation_id
    assert p1.stages[0].artifact.artifact_id != p2.stages[0].artifact.artifact_id


# =========================================================================== #
# Failure recording
# =========================================================================== #

def test_training_failure_recorded_and_earlier_artifacts_kept(env):
    f = env.forge
    mid = env.model_id
    a = env.imp_final.checkpoint_id
    # s1 succeeds (creates an evaluation), s2's training references a missing
    # dataset -> M3 raises before training anything
    plan = env.plan("trainfail", [
        env.evaluate_stage("s1", ckpt=a, seed=7711),
        WorkflowStage(stage_id="s2", type=StageType.TRAIN,
                      training=env.train_stage("s2", ds=env.ds_a).training
                      .model_copy(update={"dataset_id": "ghost"}))])
    n_wf = len(f.list_workflows(mid))
    with pytest.raises(FileNotFoundError, match="dataset"):
        f.run_workflow(plan)
    rec = f.list_workflows(mid)[-1]
    assert len(f.list_workflows(mid)) == n_wf + 1
    assert rec.status == WorkflowStatus.FAILED
    assert rec.failed_stage_id == "s2"
    assert "failed" in rec.terminal_reason
    assert rec.stages[0].executed and rec.stages[0].artifact is not None
    assert rec.stages[1].executed and rec.stages[1].artifact is None
    assert rec.stages[1].error
    # the successful evaluation manifest still exists and is referenced
    ev = f.get_evaluation(mid, rec.stages[0].artifact.artifact_id)
    assert ev.result_hash == rec.stages[0].artifact.result_hash
    env.no_tmp(mid)


def test_evaluation_failure_recorded_cleanly(env):
    f = env.forge
    mid = env.model_id
    n_wf = len(f.list_workflows(mid))
    n_ev = len(f.list_evaluations(mid))
    plan = env.plan("evfail", [
        env.evaluate_stage("s1", ckpt="ghost-ck", seed=7712),
        env.evaluate_stage("s2", ckpt=env.imp_final.checkpoint_id, seed=7712)])
    with pytest.raises(FileNotFoundError, match="not found"):
        f.run_workflow(plan)
    rec = f.list_workflows(mid)[-1]
    assert len(f.list_workflows(mid)) == n_wf + 1
    assert rec.status == WorkflowStatus.FAILED
    assert rec.failed_stage_id == "s1"
    assert rec.stages[0].artifact is None and rec.stages[0].error
    assert not rec.stages[1].executed and rec.stages[1].skipped
    assert len(f.list_evaluations(mid)) == n_ev     # nothing evaluated
    assert "completed" not in rec.terminal_reason


def test_gate_engine_error_recorded_and_no_decision_fabricated(env):
    f = env.forge
    mid = env.model_id
    a = env.imp_final.checkpoint_id
    n_gate = len(f.list_gate_decisions(mid))
    # gate baseline evaluation-result-hash that does not exist -> M6 404 path
    plan = env.plan("gatefail", [
        env.gate_stage("g1", ckpt=a, baseline_type="evaluation_result_hash",
                       baseline_hash="f" * 64, seed=7713)])
    with pytest.raises(FileNotFoundError, match="no evaluation"):
        f.run_workflow(plan)
    rec = f.list_workflows(mid)[-1]
    assert rec.status == WorkflowStatus.FAILED
    assert rec.failed_stage_id == "g1"
    assert len(f.list_gate_decisions(mid)) == n_gate   # nothing fabricated
    assert rec.stages[0].artifact is None


# =========================================================================== #
# Persistence: one immutable manifest per run
# =========================================================================== #

def test_one_manifest_per_run_and_immutability(env):
    f = env.forge
    mid = env.model_id
    a = env.imp_final.checkpoint_id
    root = f.storage.model_dir(mid) / "workflows"
    dirs_before = {d.name for d in root.iterdir() if d.is_dir()}
    rec = f.run_workflow(env.plan("persist", [
        env.evaluate_stage("s1", ckpt=a, seed=7714)]))
    dirs = {d.name for d in root.iterdir() if d.is_dir()}
    assert dirs - dirs_before == {f"workflow-{rec.workflow_id}"}
    assert len(dirs) == len(dirs_before) + 1
    mpath = root / f"workflow-{rec.workflow_id}" / "manifest.json"
    raw = json.loads(mpath.read_text())               # valid JSON
    assert raw["workflow_id"] == rec.workflow_id
    assert raw["result_hash"] == rec.result_hash
    assert raw["status"] == "completed"
    assert raw["plan"]["name"] == "persist"
    assert raw == rec.model_dump(mode="json")         # immutable byte-equality
    # run is complete: the file was written once and never touched after
    mpath_bytes = mpath.read_bytes()
    f.run_workflow(env.plan("persist", [
        env.evaluate_stage("s1", ckpt=a, seed=7714)]))
    assert mpath.read_bytes() == mpath_bytes
    env.no_tmp(mid)


def test_list_get_semantics(env):
    f = env.forge
    mid = env.model_id
    lst = f.list_workflows(mid)
    assert lst == sorted(lst, key=lambda r: (r.created_at, r.workflow_id))
    assert len(lst) >= 8
    one = f.get_workflow(mid, lst[0].workflow_id)
    assert one.workflow_id == lst[0].workflow_id
    with pytest.raises(FileNotFoundError):
        f.get_workflow(mid, "ghost")                  # unknown workflow
    with pytest.raises(FileNotFoundError):
        f.list_workflows("ghost")                     # unknown model
    with pytest.raises(FileNotFoundError):
        f.get_workflow("ghost", "ghost")
    fresh = env.fresh_model("m7-no-wf")
    assert f.list_workflows(fresh) == []              # known model, no runs
    env.no_tmp(mid)


def test_readonly_audit_exact_new_file_set(env):
    """An evaluate/compare/gate workflow adds exactly its own evidence —
    run manifest, stage artifacts, and gate-internal evaluations/comparison
    that were genuinely new — and never mutates pre-existing files
    (byte-level audit over the whole model directory)."""
    f = env.forge
    mid = env.model_id
    model_dir = f.storage.model_dir(mid)
    before = _snapshot(model_dir)
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    rec = f.run_workflow(env.plan("ro", [
        env.evaluate_stage("s1", ckpt=a, seed=7715),
        env.compare_stage("s2", state_a=env.state(ckpt=a),
                          state_b=env.state(ckpt=b), seed=7715),
        env.gate_stage("s3", ckpt=b, seed=7715, baseline_ckpt=a)]))
    after = _snapshot(model_dir)
    added = set(after) - set(before)
    assert added == _expected_new_files(f, mid, rec, before)
    for k in before:
        assert after[k] == before[k]
    env.no_tmp(mid)


# =========================================================================== #
# M11 — suite_run stages inside workflows (M10 engine adapter)
# =========================================================================== #

def _register_suite(env, suite_id: str, seeds, ds=None) -> str:
    """Register one probe suite on the shared tokenized domain."""
    probes = [SuiteProbe(dataset_id=ds or env.ds_a, split="validation",
                         tokenizer_id=env.tok_id, batch_size=8,
                         max_seq_len=32, seed=s) for s in seeds]
    env.forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id=suite_id, probes=probes))
    return suite_id


def _suite_stage(sid: str, suite_id: str, *, ckpt=None, from_stage=None,
                 current=False) -> WorkflowStage:
    """Explicit suite-run stage: ONE suite + ONE state source, never guessed."""
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


def _suite_evals_of(env, suite_run_id):
    rec = env.forge.get_suite_run(env.model_id, suite_run_id)
    return [r.evaluation_id for r in rec.results]


# --------------------------------------------------------------------------- #
# Schema: valid + malformed suite_run stages
# --------------------------------------------------------------------------- #

def test_suite_run_stage_schema_valid(env):
    st = _suite_stage("sr", "m11-s", ckpt=env.imp_final.checkpoint_id)
    assert st.type == StageType.SUITE_RUN
    assert st.suite_run.suite_id == "m11-s"
    assert st.suite_run.state.checkpoint_id == env.imp_final.checkpoint_id
    assert st.training is None and st.gate is None and st.evaluation is None
    cur = _suite_stage("sr2", "m11-s", current=True)
    assert cur.suite_run.state.state_kind == EvalStateKind.CURRENT
    # existing stage types remain valid
    env.evaluate_stage("e1", ckpt=env.imp_final.checkpoint_id)
    env.gate_stage("g1", ckpt=env.imp_final.checkpoint_id,
                   baseline_type="minimum_loss", minimum_loss=1e9)


def test_suite_run_stage_schema_rejections(env):
    # missing suite id
    with pytest.raises(ValidationError):
        WorkflowSuiteRunStage(
            suite_id="", state=StageStateRef(
                state_kind=EvalStateKind.CHECKPOINT,
                checkpoint_id=env.imp_final.checkpoint_id))
    # missing state entirely
    with pytest.raises(ValidationError):
        WorkflowSuiteRunStage(suite_id="m11-s")
    # invalid state: checkpoint with neither id nor from_stage
    with pytest.raises(ValidationError):
        WorkflowSuiteRunStage(
            suite_id="m11-s", state=StageStateRef(
                state_kind=EvalStateKind.CHECKPOINT))
    with pytest.raises(ValidationError):
        WorkflowStage(stage_id="x", type=StageType.SUITE_RUN,
                      suite_run=WorkflowSuiteRunStage(
                          suite_id="m11-s", state=StageStateRef(
                              state_kind=EvalStateKind.CHECKPOINT)))
    # current state must not carry a checkpoint/from_stage
    with pytest.raises(ValidationError):
        WorkflowSuiteRunStage(
            suite_id="m11-s", state=StageStateRef(
                state_kind=EvalStateKind.CURRENT,
                checkpoint_id=env.imp_final.checkpoint_id))
    # suite stage must not carry another payload or branches
    with pytest.raises(ValidationError):
        WorkflowStage(stage_id="x", type=StageType.SUITE_RUN,
                      training=env.train_stage("t").training)
    with pytest.raises(ValidationError):
        WorkflowStage(
            stage_id="x", type=StageType.SUITE_RUN,
            suite_run=WorkflowSuiteRunStage(
                suite_id="m11-s", state=StageStateRef(
                    state_kind=EvalStateKind.CURRENT)),
            on_fail="later")                    # branches are gate-only


def test_suite_run_stage_dependency_validation(env):
    """Explicit refs only: from_stage must name an EARLIER TRAIN stage."""
    ck = env.imp_final.checkpoint_id
    good = _suite_stage("sr0", "m11-s", ckpt=ck)
    # unknown from_stage target
    bad1 = _suite_stage("sr1", "m11-s", from_stage="no-such-stage")
    with pytest.raises(ValueError, match="unknown stage"):
        env.plan("m11-dep-1", [good, bad1])
    # from_stage pointing at ITSELF is not "an EARLIER stage"
    with pytest.raises(ValueError, match="EARLIER"):
        env.plan("m11-dep-2", [_suite_stage("sr0", "m11-s",
                                            from_stage="sr0")])
    # from_stage pointing at an earlier NON-TRAIN stage (evals produce no ckpt)
    bad2 = _suite_stage("sr1", "m11-s", from_stage="sr0")
    with pytest.raises(ValueError, match="not a train stage"):
        env.plan("m11-dep-3", [good, bad2])
    # from_stage pointing at a non-train earlier stage (evals produce no ckpt)
    eval_st = env.evaluate_stage("e1", ckpt=ck)
    bad3 = _suite_stage("sr", "m11-s", from_stage="e1")
    with pytest.raises(ValueError, match="not a train stage"):
        env.plan("m11-dep-3", [eval_st, bad3])


# --------------------------------------------------------------------------- #
# Engine: execution + provenance + reuse
# --------------------------------------------------------------------------- #

def test_suite_run_stage_executes_checkpoint_state(env):
    _register_suite(env, "m11-exec", [83001, 83002])
    ck = env.imp_final.checkpoint_id
    rec = env.forge.run_workflow(env.plan("m11-exec-wf",
                                          [_suite_stage("sr", "m11-exec",
                                                        ckpt=ck)]))
    assert rec.status == WorkflowStatus.COMPLETED
    st = rec.stages[0]
    assert st.executed and st.skipped is False and st.error is None
    art = st.artifact
    assert art.kind.value == "suite_run" and art.artifact_id
    assert art.result_hash and art.checkpoint_id == ck
    sr = env.forge.get_suite_run(env.model_id, art.artifact_id)
    assert sr.status.value == "completed"
    assert sr.suite_id == "m11-exec" and sr.state.checkpoint_id == ck
    assert sr.suite_run_id == art.artifact_id
    assert [r.outcome for r in sr.results] == ["created", "created"]
    # every suite probe mapped to an exact M4 evaluation of that checkpoint
    for r in sr.results:
        ev = env.forge.get_evaluation(env.model_id, r.evaluation_id)
        assert ev.checkpoint_id == ck and ev.seed == r.probe.seed


def test_suite_run_stage_uses_earlier_train_stage_output(env):
    """from_stage provenance: the suite evaluates THAT stage's final ckpt."""
    mid = env.fresh_model("m11-train-suite")
    env.forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m11-ts", probes=[
            SuiteProbe(dataset_id=env.ds_a, split="validation",
                       tokenizer_id=env.tok_id, batch_size=8, max_seq_len=32,
                       seed=83011)]))
    plan = env.plan("m11-train-suite-wf", [
        env.train_stage("tr", model_id=mid, epochs=2, eval_every=2),
        _suite_stage("sr", "m11-ts", from_stage="tr")], model_id=mid)
    rec = env.forge.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    train_ck = rec.stages[0].artifact.checkpoint_id
    sr_id = rec.stages[1].artifact.artifact_id
    assert rec.stages[1].artifact.checkpoint_id == train_ck
    sr = env.forge.get_suite_run(mid, sr_id)
    assert sr.state.checkpoint_id == train_ck           # provenance is exact
    ev = env.forge.get_evaluation(mid, sr.results[0].evaluation_id)
    assert ev.checkpoint_id == train_ck and ev.state_hash == sr.state_hash


def test_suite_run_stage_current_state(env):
    _register_suite(env, "m11-cur", [83021])
    rec = env.forge.run_workflow(env.plan("m11-cur-wf",
                                          [_suite_stage("sr", "m11-cur",
                                                        current=True)]))
    sr = env.forge.get_suite_run(env.model_id,
                                 rec.stages[0].artifact.artifact_id)
    assert sr.state.state_kind == EvalStateKind.CURRENT
    live = env.forge.comparison.verified_state_hash(
        env.model_id, env.state(kind="current"))
    assert sr.state_hash == live
    ev = env.forge.get_evaluation(env.model_id, sr.results[0].evaluation_id)
    assert ev.state_kind == EvalStateKind.CURRENT


def test_repeated_workflows_reuse_suite_evidence(env):
    """Two workflows, same state+suite: distinct immutable run records,
    identical underlying evaluations, no duplicate M4 evidence."""
    _register_suite(env, "m11-reuse", [83031, 83032])
    ck = env.imp_final.checkpoint_id
    evals_dir = env.forge.storage.model_dir(env.model_id) / "evaluations"
    n_eval_dirs = len([p for p in evals_dir.iterdir() if p.is_dir()])
    suite_runs_dir = env.forge.storage.root / "suite-runs"
    n_sr = len([p for p in suite_runs_dir.iterdir() if p.is_dir()])
    r1 = env.forge.run_workflow(env.plan("m11-reuse-wf1",
                                         [_suite_stage("sr", "m11-reuse",
                                                       ckpt=ck)]))
    n_eval_dirs += 2                                    # two fresh evaluations
    r2 = env.forge.run_workflow(env.plan("m11-reuse-wf2",
                                         [_suite_stage("sr", "m11-reuse",
                                                       ckpt=ck)]))
    sr1 = env.forge.get_suite_run(env.model_id, r1.stages[0].artifact.artifact_id)
    sr2 = env.forge.get_suite_run(env.model_id, r2.stages[0].artifact.artifact_id)
    assert sr1.suite_run_id != sr2.suite_run_id         # distinct invocations
    assert [x.outcome for x in sr1.results] == ["created", "created"]
    assert [x.outcome for x in sr2.results] == ["reused", "reused"]
    assert [x.evaluation_id for x in sr2.results] == \
        [x.evaluation_id for x in sr1.results]
    assert len([p for p in evals_dir.iterdir() if p.is_dir()]) == n_eval_dirs
    assert len([p for p in suite_runs_dir.iterdir() if p.is_dir()]) == n_sr + 2


def test_suite_run_stage_records_are_immutable_and_deterministic(env):
    _register_suite(env, "m11-det", [83041])
    ck = env.imp_final.checkpoint_id
    r = env.forge.run_workflow(env.plan("m11-det-wf",
                                        [_suite_stage("sr", "m11-det",
                                                      ckpt=ck)]))
    wf = env.forge.get_workflow(env.model_id, r.workflow_id)
    assert wf.model_dump() == r.model_dump()            # read back identical
    # suite-run manifest is a single immutable file
    d = env.forge.storage.root / "suite-runs" / \
        r.stages[0].artifact.artifact_id
    assert [p.name for p in d.iterdir()] == ["manifest.json"]
    # deterministic suite-run result_hash on repeat
    r2 = env.forge.run_workflow(env.plan("m11-det-wf-2",
                                         [_suite_stage("sr", "m11-det",
                                                       ckpt=ck)]))
    sr1 = env.forge.get_suite_run(env.model_id, r.stages[0].artifact.artifact_id)
    sr2 = env.forge.get_suite_run(env.model_id, r2.stages[0].artifact.artifact_id)
    assert sr1.result_hash == sr2.result_hash


# --------------------------------------------------------------------------- #
# Failure semantics
# --------------------------------------------------------------------------- #

def test_unknown_suite_fails_cleanly_no_fabricated_artifact(env):
    ck = env.imp_final.checkpoint_id
    n_sr_before = len([p for p in (env.forge.storage.root / "suite-runs")
                       .iterdir() if p.is_dir()]) \
        if (env.forge.storage.root / "suite-runs").exists() else 0
    with pytest.raises(FileNotFoundError, match="probe suite"):
        env.forge.run_workflow(env.plan("m11-unknown-suite-wf",
                                        [_suite_stage("sr", "m11-ghost",
                                                      ckpt=ck)]))
    runs = env.forge.list_workflows(env.model_id)
    failed = [w for w in runs if w.name == "m11-unknown-suite-wf"]
    assert len(failed) == 1 and failed[0].status == WorkflowStatus.FAILED
    assert failed[0].failed_stage_id == "sr"
    assert failed[0].stages[0].artifact is None          # nothing fabricated
    assert failed[0].stages[0].error and "m11-ghost" in failed[0].stages[0].error
    n_sr_after = len([p for p in (env.forge.storage.root / "suite-runs")
                      .iterdir() if p.is_dir()])
    assert n_sr_after == n_sr_before                     # no suite-run record


def test_invalid_state_fails_cleanly(env):
    _register_suite(env, "m11-st", [83051])
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_workflow(env.plan("m11-bad-state-wf",
                                        [_suite_stage("sr", "m11-st",
                                                      ckpt="ghost-ck")]))
    runs = env.forge.list_workflows(env.model_id)
    failed = [w for w in runs if w.name == "m11-bad-state-wf"]
    assert len(failed) == 1 and failed[0].status == WorkflowStatus.FAILED
    assert failed[0].stages[0].artifact is None


def test_per_probe_failed_suite_run_referenced_not_hidden(env):
    """Underlying M10 run with per-probe failures is a REAL persisted record;
    the workflow references it and continues — never fake success."""
    env.forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m11-broken", probes=[
            SuiteProbe(dataset_id="ghost-ds", split="validation",
                       tokenizer_id=env.tok_id, batch_size=8,
                       max_seq_len=32, seed=83061)]))
    _register_suite(env, "m11-ok", [83062])
    rec = env.forge.run_workflow(env.plan("m11-partial-wf", [
        _suite_stage("sr1", "m11-broken", current=True),
        _suite_stage("sr2", "m11-ok", current=True)]))
    # both suite stages EXECUTED (adapter level), workflow completed
    assert rec.status == WorkflowStatus.COMPLETED
    assert rec.stages[0].artifact.kind.value == "suite_run"
    sr_bad = env.forge.get_suite_run(env.model_id,
                                     rec.stages[0].artifact.artifact_id)
    assert sr_bad.status.value == "failed"               # real, not converted
    assert sr_bad.results[0].outcome == "failed" and \
        sr_bad.results[0].evaluation_id is None
    sr_ok = env.forge.get_suite_run(env.model_id,
                                    rec.stages[1].artifact.artifact_id)
    assert sr_ok.status.value == "completed"             # execution continued


def test_suite_run_stage_participates_in_normal_flow(env):
    """A suite stage is a non-branching stage: next-stage transition follows,
    and a later gate still branches on its real M6 decision."""
    _register_suite(env, "m11-flow", [83071])
    ck = env.imp_final.checkpoint_id
    rec = env.forge.run_workflow(env.plan("m11-flow-wf", [
        _suite_stage("sr", "m11-flow", ckpt=ck),
        env.gate_stage("g1", ckpt=ck, baseline_type="minimum_loss",
                       minimum_loss=1e9)]))
    assert rec.status == WorkflowStatus.COMPLETED
    transitions = rec.transitions
    assert transitions[0].stage_id == "sr"
    assert transitions[0].decision == "next" and transitions[0].to_stage == "g1"
    assert transitions[1].stage_id == "g1"
    assert transitions[1].decision in ("passed", "failed")
    assert transitions[1].to_stage is None


def test_dashboard_renders_suite_run_workflows_deterministically(env):
    """M8 keeps working: workflows with suite-run stages render (suite runs
    are M10 artifacts the dashboard references, not new metrics), no
    diagnostics, stable result hash."""
    mid = env.fresh_model("m11-dash")
    env.forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m11-dash-s", probes=[
            SuiteProbe(dataset_id=env.ds_a, split="validation",
                       tokenizer_id=env.tok_id, batch_size=8, max_seq_len=32,
                       seed=83081)]))
    plan = env.plan("m11-dash-wf", [
        env.train_stage("tr", model_id=mid, epochs=2, eval_every=2),
        _suite_stage("sr", "m11-dash-s", from_stage="tr")], model_id=mid)
    rec = env.forge.run_workflow(plan)
    assert rec.status == WorkflowStatus.COMPLETED
    j1 = env.forge.get_dashboard(mid).model_dump(mode="json")
    assert j1["diagnostics"] == []
    wf_rows = [w for w in j1["workflows"]["records"]
               if w["workflow_id"] == rec.workflow_id]
    assert len(wf_rows) == 1
    stage = wf_rows[0]["stages"][1]
    assert stage["artifact"]["kind"] == "suite_run"
    assert stage["artifact"]["artifact_id"]
    # underlying evaluations visible in the dashboard series as usual
    ev_ids = {r["eval_id"] for g in j1["evaluations"] for r in g["records"]}
    sr = env.forge.get_suite_run(mid, stage["artifact"]["artifact_id"])
    assert {x.evaluation_id for x in sr.results} <= ev_ids
    # deterministic repeat
    j2 = env.forge.get_dashboard(mid).model_dump(mode="json")
    assert j2 == j1
    assert env.forge.get_dashboard(mid).result_hash == \
        env.forge.get_dashboard(mid).result_hash
