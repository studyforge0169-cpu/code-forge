"""Milestone 73 tests: read-only FIRST-LEVEL deletion impact preview.

The M73 invariants under test:

* CURRENT-STATE EQUIVALENCE — for every deletable family (all 14),
  the preview's current section is the family's EXISTING retention
  view verbatim (deletable / integrity / files / bytes / a 1:1
  blocker projection) — never a second blocker engine.
* DEPENDENTS — the immediate dependents are exactly the artifacts
  whose CURRENT blocker lists contain the selected artifact (its
  outbound persisted edges, verified per dependent); reported for
  blocked artifacts too (current-state facts).
* FIRST LEVEL ONLY — the becomes-deletable set contains exactly the
  dependents whose blocker lists become empty after THIS ONE
  deletion (the minimal in-memory shadow over the SAME canonical
  filters); dependents of dependents are deliberately absent.
* REAL-DELETION ORACLE — on a disposable fixture, really deleting
  the artifact through the existing verified guards makes the
  prediction TRUE: the predicted becomes-deletable artifacts are
  deletable, the others stay blocked, and the M72 project
  reclaimable after the deletion equals the predicted after —
  including the model/record overlap rule (a deletable model
  contributes its WHOLE directory; a record of a deletable model
  contributes only its own bytes).
* GUARD SEMANTICS — a blocked or integrity-failed artifact has NO
  executable deletion: empty becomes-deletable set, zero delta.
* READ-ONLY — repeated previews mutate nothing; byte-identical
  repeats; unknown artifacts -> 404.
"""
from __future__ import annotations

import hashlib
import json
import random
import zlib
from pathlib import Path

import pytest

from app.engine import ModelForge
from app.schemas import (
    ComparisonState,
    EvalStateKind,
    EvaluationConfig,
    GatePolicy,
    GateRequest,
    ModelCreateRequest,
    ProbeSuiteCreateRequest,
    SampleGenerateRequest,
    SampleStrategy,
    StageStateRef,
    StageType,
    SuiteProbe,
    SuiteRunRequest,
    TokenizerConfig,
    TransformerConfig,
    TrainingConfig,
    WorkflowEvaluationStage,
    WorkflowGateStage,
    WorkflowPlan,
    WorkflowRecipeCallStage,
    WorkflowRecipeCreateRequest,
    WorkflowStage,
    PolicyCreateRequest,
)

_WORDS = ("river mountain cloud forest desert ocean valley island meadow "
          "canyon table chair lamp desk shelf couch rug clock mirror "
          "vase").split()


def _corpus(n: int, tag: str) -> bytes:
    rng = random.Random(zlib.crc32(tag.encode()))
    lines = []
    for i in range(n):
        k = rng.randint(10, 22)
        lines.append(" ".join(rng.choices(_WORDS, k=k)) + f" number {i}")
    return ("\n\n".join(lines) + "\n").encode("utf-8")


def _inventory(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


def _train_cfg(model, ds, tok, seed, steps=8):
    return TrainingConfig(
        method="continued_pretraining", model_id=model, dataset_id=ds,
        tokenizer_id=tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=seed,
        steps=steps)


def _policy(model, ds, tok, name, seed, ckpt, tolerance=1.0):
    return GatePolicy(
        name=name, model_id=model, dataset_id=ds, tokenizer_id=tok,
        split="validation", batch_size=8, max_seq_len=32, seed=seed,
        baseline_type="checkpoint", baseline_checkpoint_id=ckpt,
        tolerance=tolerance)


class Env:
    pass


def _build(root: Path) -> Env:
    """The multi-family fixture: every deletable family plus the
    reference chains W2 -> G_W -> C_G -> evals (the first-level
    boundary), a suite run, sample+quality, a recipe run, a policy
    blocking model B and a DELETABLE model D with records (the
    overlap-rule case)."""
    e = Env()
    e.root = root
    forge = ModelForge(root=root)
    e.forge = forge
    up = forge.upload_dataset([("m73.txt", _corpus(190, "m73"))],
                              name="m73-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m73-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)

    # model A: the reference-rich model
    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m73-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    forge.run_training(_train_cfg(e.model, e.ds, e.tok, 1))
    e.ck = sorted(c.checkpoint_id for c in
                  forge.list_checkpoints(e.model))
    e.direct_eval = forge.run_evaluation(EvaluationConfig(
        model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=e.ck[0]))
    # W2: train + INLINE gate stage -> the chain
    # W2 (leaf) -> G_W -> C_G -> side evaluations
    e.w2 = forge.run_workflow(WorkflowPlan(
        name="m73-w2", model_id=e.model, description="M73 chain",
        stages=[
            WorkflowStage(stage_id="tr", type=StageType.TRAIN,
                          training=_train_cfg(e.model, e.ds, e.tok, 3,
                                              steps=4)),
            WorkflowStage(
                stage_id="gt", type=StageType.GATE,
                gate=WorkflowGateStage(
                    policy=_policy(e.model, e.ds, e.tok, "m73-w2-gate", 4,
                                   e.ck[0]),
                    candidate=StageStateRef(
                        state_kind=EvalStateKind.CHECKPOINT,
                        checkpoint_id=e.ck[1])))]))
    e.gw = next(st.artifact.artifact_id for st in e.w2.stages
                if st.artifact is not None
                and st.artifact.kind.value == "gate_decision")
    e.cg = forge.gates.get_decision(e.model, e.gw).comparison_id
    # suite + suite run (probe evaluation + model external reference)
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m73-suite", description="M73 suite",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))
    e.sr = forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m73-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    e.probe_eval = next(r.evaluation_id for r in e.sr.results
                        if r.evaluation_id)
    # sample + sample-quality measurement
    e.sample = forge.generate_sample(SampleGenerateRequest(
        model_id=e.model, checkpoint_id=e.ck[0], tokenizer_id=e.tok,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    e.quality = forge.evaluate_sample(e.model, e.sample.sample_id)
    # recipe + ONE run with recipe provenance
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m73-recipe", description="M73 recipe",
        stages=[WorkflowStage(
            stage_id="ev", type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=EvaluationConfig(
                    model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
                    split="validation", batch_size=8, max_seq_len=32,
                    seed=5, checkpoint_id=e.ck[0])))]))
    e.w1 = forge.run_workflow_recipe("m73-recipe", e.model)
    # composite recipe: a composition blocker on the base recipe
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m73-comp", description="M73 composite",
        stages=[WorkflowStage(
            stage_id="call", type=StageType.RECIPE,
            recipe=WorkflowRecipeCallStage(recipe_id="m73-recipe"))]))
    e.comp = "m73-comp"

    # model B: blocked by the registry policies
    e.model_b = forge.create_model(ModelCreateRequest(
        config=TransformerConfig(
            name="m73-model-b", vocab_size=640, context_length=64,
            hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
            intermediate_size=128, seed=2)))[0].id
    forge.run_training(_train_cfg(e.model_b, e.ds, e.tok, 6))
    ck_b = sorted(c.checkpoint_id for c in
                  forge.list_checkpoints(e.model_b))
    e.policy = forge.register_policy(PolicyCreateRequest(
        policy_id="m73-policy", description="M73 policy",
        policy=_policy(e.model_b, e.ds, e.tok, "m73-policy", 7,
                       ck_b[0]))).policy_id
    # a second registry policy + ONE gate decision from it (the
    # policy's OWN blocker, and the gate's outbound definition ref)
    e.policy2 = forge.register_policy(PolicyCreateRequest(
        policy_id="m73-policy-2", description="M73 policy 2",
        policy=_policy(e.model_b, e.ds, e.tok, "m73-policy-2", 12,
                       ck_b[0]))).policy_id
    e.g_b = forge.run_gate(GateRequest(
        model_id=e.model_b, policy_id=e.policy2,
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=ck_b[1])))

    # model D: DELETABLE (no external references) WITH records — the
    # overlap-rule case; G_D2 (leaf) blocks its own comparison C_D
    e.model_d = forge.create_model(ModelCreateRequest(
        config=TransformerConfig(
            name="m73-model-d", vocab_size=640, context_length=64,
            hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
            intermediate_size=128, seed=3)))[0].id
    forge.run_training(_train_cfg(e.model_d, e.ds, e.tok, 8))
    ck_d = sorted(c.checkpoint_id for c in
                  forge.list_checkpoints(e.model_d))
    e.w_d = forge.run_workflow(WorkflowPlan(
        name="m73-w-d", model_id=e.model_d,
        stages=[
            WorkflowStage(stage_id="tr", type=StageType.TRAIN,
                          training=_train_cfg(e.model_d, e.ds, e.tok, 9,
                                              steps=4)),
            WorkflowStage(
                stage_id="ev", type=StageType.EVALUATE,
                evaluation=WorkflowEvaluationStage(
                    config=EvaluationConfig(
                        model_id=e.model_d, dataset_id=e.ds,
                        tokenizer_id=e.tok, split="validation",
                        batch_size=8, max_seq_len=32, seed=10,
                        checkpoint_id=ck_d[0])))]))
    e.g_d2 = forge.run_gate(GateRequest(
        model_id=e.model_d,
        policy=_policy(e.model_d, e.ds, e.tok, "m73-g-d2", 11, ck_d[1]),
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=ck_d[0])))
    e.c_d = forge.gates.get_decision(e.model_d,
                                     e.g_d2.decision_id).comparison_id
    return e


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("m73-env"))


@pytest.fixture(scope="module")
def denv(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("m73-denv"))


def _impact(env, family, artifact_id, model_id=None):
    return env.forge.deletion_impact_preview(family, artifact_id,
                                             model_id)


def _deps(impact) -> set[tuple[str, str, str]]:
    return {(d.family, d.artifact_id, d.reference_category)
            for d in impact.immediate_dependents}


def _becomes(impact) -> set[tuple[str, str]]:
    return {(b.family, b.artifact_id) for b in impact.becomes_deletable}


# --------------------------------------------------------------------------- #
# Test 1: current-state equivalence for EVERY deletable family
# --------------------------------------------------------------------------- #

def test_m73_current_state_matches_retention_views(env):
    forge = env.forge
    cases = []
    v = forge.model_retention_overview(env.model)
    cases.append(("model", env.model, None, v.deletable,
                  v.integrity_verified, len(v.files), v.size_bytes,
                  [(b.category, b.reference_id, b.detail)
                   for b in v.blockers]))
    v = forge.dataset_retention_overview(env.ds)
    cases.append(("dataset", env.ds, None, v.deletable,
                  v.integrity_verified, len(v.files), v.size_bytes,
                  [(b.reason, None, b.detail) for b in v.blockers]))
    v = forge.tokenizer_retention_overview(env.tok)
    cases.append(("tokenizer", env.tok, None, v.deletable,
                  v.integrity_verified, len(v.files), v.size_bytes,
                  [(b.reason, None, b.detail) for b in v.blockers]))
    for family, did in (("workflow_recipe", "m73-recipe"),
                        ("gate_policy", env.policy),
                        ("probe_suite", "m73-suite")):
        v = forge.definition_retention_overview(family, did)
        cases.append((family, did, None, v.deletable,
                      v.integrity_verified, len(v.files), v.size_bytes,
                      [(b.category, b.reference_id, b.detail)
                       for b in v.blockers]))
    ck = forge.checkpoint_retention_overview(env.model)
    entry = next(c for c in ck.checkpoints
                 if c.checkpoint_id == env.ck[0])
    cases.append(("checkpoint", env.ck[0], env.model, entry.deletable,
                  entry.integrity_verified, entry.files,
                  entry.size_bytes,
                  [(b.reason, None, b.detail) for b in entry.blockers]))
    for family, rid, rec in (
            ("workflow", env.w2.workflow_id, env.w2),
            ("evaluation", env.direct_eval.eval_id, env.direct_eval),
            ("comparison", env.cg, None),
            ("gate", env.gw, None)):
        v = forge.model_record_retention_overview(env.model, family, rid)
        cases.append((family, rid, env.model, v.deletable,
                      v.integrity_verified, len(v.files), v.size_bytes,
                      [(b.category, b.reference_id, b.detail)
                       for b in v.blockers]))
    v = forge.suite_run_retention_overview(env.model,
                                           env.sr.suite_run_id)
    cases.append(("suite_run", env.sr.suite_run_id, env.model,
                  v.deletable, v.integrity_verified, len(v.files),
                  v.size_bytes,
                  [(b.reason, None, b.detail) for b in v.blockers]))
    v = forge.sample_retention_overview(env.model, env.sample.sample_id)
    cases.append(("sample", env.sample.sample_id, env.model, v.deletable,
                  v.integrity_verified, len(v.files), v.size_bytes,
                  [(b.reason, None, b.detail) for b in v.blockers]))
    v = forge.sample_evaluation_retention_overview(
        env.model, env.quality.evaluation_id)
    cases.append(("sample_quality", env.quality.evaluation_id, env.model,
                  v.deletable, v.integrity_verified, len(v.files),
                  v.size_bytes,
                  [(b.reason, None, b.detail) for b in v.blockers]))

    assert len(cases) == 14
    for family, aid, mid, deletable, integrity, files, size, blockers \
            in cases:
        imp = _impact(env, family, aid, mid)
        assert imp.family == family and imp.artifact_id == aid
        assert imp.model_id == mid
        assert imp.deletion_supported is True
        assert imp.deletable == deletable, (family, aid)
        assert imp.integrity_verified == integrity, (family, aid)
        assert imp.files == files and imp.size_bytes == size, (family, aid)
        assert imp.immediate_files == files
        assert imp.immediate_bytes == size
        assert imp.executable == deletable
        assert [(b.category, b.reference_id, b.detail)
                for b in imp.blockers] == blockers, (family, aid)
        assert imp.project_reclaimable_delta == (
            imp.project_reclaimable_after - imp.project_reclaimable_before)
        assert imp.project_reclaimable_before == \
            forge.project_retention_overview().reclaimable_bytes


# --------------------------------------------------------------------------- #
# Test 2: immediate dependents for every family
# --------------------------------------------------------------------------- #

def test_m73_dependents_per_family(env):
    # workflow W2 (leaf): blocks its gate stage artifact, its held
    # checkpoints, its plan dataset/tokenizer
    d = _deps(_impact(env, "workflow", env.w2.workflow_id, env.model))
    assert ("gate", env.gw, "workflow") in d
    assert ("dataset", env.ds, "workflow") in d
    assert ("tokenizer", env.tok, "workflow") in d
    assert all(cat == "workflow" for _, _, cat in d)
    assert all(fam in ("gate", "checkpoint", "dataset", "tokenizer")
               for fam, _, _ in d)
    # the gate G_W (blocked by W2 — dependents are still factual):
    # its comparison, side evaluations and side checkpoints (the
    # gate's data refs flow through its comparison + evaluations —
    # no gate category exists in the M64 dataset/tokenizer usage)
    d = _deps(_impact(env, "gate", env.gw, env.model))
    g = env.forge.gates.get_decision(env.model, env.gw)
    exp = {("comparison", env.cg, "gate")}
    for side in (g.candidate, g.baseline):
        if side is None:
            continue
        exp.add(("evaluation", side.evaluation_id, "gate"))
        if side.state_kind == EvalStateKind.CHECKPOINT \
                and side.checkpoint_id:
            exp.add(("checkpoint", side.checkpoint_id, "gate"))
    if g.suggested_checkpoint_id:
        exp.add(("checkpoint", g.suggested_checkpoint_id, "gate"))
    assert d == exp
    # the gate's comparison: its side evaluations + data refs
    d = _deps(_impact(env, "comparison", env.cg, env.model))
    comp = env.forge.comparison.get_comparison(env.model, env.cg)
    assert {("evaluation", comp.state_a.evaluation_id, "comparison"),
            ("evaluation", comp.state_b.evaluation_id, "comparison")} <= d
    assert ("dataset", env.ds, "comparison") in d
    assert ("tokenizer", env.tok, "comparison") in d
    # a direct evaluation: the checkpoint it measured + data refs
    d = _deps(_impact(env, "evaluation", env.direct_eval.eval_id,
                      env.model))
    assert d == {("checkpoint", env.ck[0], "evaluation"),
                 ("dataset", env.ds, "evaluation"),
                 ("tokenizer", env.tok, "evaluation")}
    # suite run: its model, the suite, its probe evaluation, data refs
    d = _deps(_impact(env, "suite_run", env.sr.suite_run_id, env.model))
    assert d == {("model", env.model, "suite_run"),
                 ("probe_suite", "m73-suite", "suite_run"),
                 ("evaluation", env.probe_eval, "suite_run"),
                 ("dataset", env.ds, "suite_run"),
                 ("tokenizer", env.tok, "suite_run")}
    # sample: its model, checkpoint, tokenizer
    d = _deps(_impact(env, "sample", env.sample.sample_id, env.model))
    assert d == {("model", env.model, "sample"),
                 ("checkpoint", env.sample.checkpoint_id, "sample"),
                 ("tokenizer", env.tok, "sample")}
    # sample-quality: its model, the sample, checkpoint, tokenizer
    d = _deps(_impact(env, "sample_quality", env.quality.evaluation_id,
                      env.model))
    assert d == {("model", env.model, "sample_quality"),
                 ("sample", env.sample.sample_id, "sample_quality"),
                 ("checkpoint", env.quality.checkpoint_id,
                  "sample_quality"),
                 ("tokenizer", env.tok, "sample_quality")}
    # recipe: the models its stage configs name; the COMPOSITE names
    # its composition targets (and a composite itself names NO model)
    d = _deps(_impact(env, "workflow_recipe", "m73-recipe"))
    assert d == {("model", env.model, "workflow_recipe")}
    d = _deps(_impact(env, "workflow_recipe", env.comp))
    assert d == {("workflow_recipe", "m73-recipe",
                  "workflow_recipe")}
    # policy: the model its policy targets (the gates that RAN with
    # it are the POLICY's own blockers — never its dependents)
    d = _deps(_impact(env, "gate_policy", env.policy))
    assert d == {("model", env.model_b, "policy")}
    d = _deps(_impact(env, "gate_policy", env.policy2))
    assert d == {("model", env.model_b, "policy")}
    # the registry-policy gate: its comparison + sides + the POLICY
    # itself (the gate is in the policy's blocker list)
    d = _deps(_impact(env, "gate", env.g_b.decision_id, env.model_b))
    c_b = env.forge.gates.get_decision(
        env.model_b, env.g_b.decision_id).comparison_id
    assert ("comparison", c_b, "gate") in d
    assert ("gate_policy", env.policy2, "gate") in d
    # model A (blocked): datasets/tokenizers its stuff references
    d = _deps(_impact(env, "model", env.model))
    assert ("dataset", env.ds, "training_run") in d
    assert ("tokenizer", env.tok, "training_run") in d
    assert all(fam in ("dataset", "tokenizer") for fam, _, _ in d)
    # dataset: the tokenizers its tokenized versions belong to
    d = _deps(_impact(env, "dataset", env.ds))
    assert d == {("tokenizer", env.tok, "tokenized_dataset")}
    # tokenizer / probe_suite / checkpoint: NOTHING depends on them
    assert _deps(_impact(env, "tokenizer", env.tok)) == set()
    assert _deps(_impact(env, "probe_suite", "m73-suite")) == set()
    assert _deps(_impact(env, "checkpoint", env.ck[0],
                         env.model)) == set()


# --------------------------------------------------------------------------- #
# Test 3: the FIRST-LEVEL boundary (A -> B -> C stops at B)
# --------------------------------------------------------------------------- #

def test_m73_first_level_boundary(env):
    # W2 (deletable) -> G_W -> C_G -> side evals: deleting W2 makes
    # G_W deletable but NOT C_G and NOT the gate's side evaluations
    imp = _impact(env, "workflow", env.w2.workflow_id, env.model)
    assert imp.executable is True
    b = _becomes(imp)
    assert ("gate", env.gw) in b
    assert ("comparison", env.cg) not in b
    g = env.forge.gates.get_decision(env.model, env.gw)
    side_evals = {g.candidate.evaluation_id}
    if g.baseline is not None:
        side_evals.add(g.baseline.evaluation_id)
    assert not any(("evaluation", eid) in b for eid in side_evals)
    # G_D2 (model D's leaf gate): deleting it unblocks EXACTLY its
    # comparison — the side evaluations stay blocked by the comparison
    imp = _impact(env, "gate", env.g_d2.decision_id, env.model_d)
    assert imp.executable is True
    assert _becomes(imp) == {("comparison", env.c_d)}
    # the registry-policy gate: deleting it unblocks its comparison
    # AND the policy (whose ONLY blocker is this gate); model B stays
    # blocked (the policy still targets it after the deletion)
    imp = _impact(env, "gate", env.g_b.decision_id, env.model_b)
    assert imp.executable is True
    c_b = env.forge.gates.get_decision(
        env.model_b, env.g_b.decision_id).comparison_id
    assert _becomes(imp) == {("comparison", c_b),
                             ("gate_policy", env.policy2)}
    # the composite: a DEPENDENT that does NOT become deletable —
    # the base recipe is ALSO blocked by its provenance run W1
    imp = _impact(env, "workflow_recipe", env.comp)
    assert imp.executable is True
    assert _deps(imp) == {("workflow_recipe", "m73-recipe",
                           "workflow_recipe")}
    assert imp.becomes_deletable == []
    # blocked artifacts have NO executable impact (the guard refuses)
    for family, aid, mid in (
            ("workflow_recipe", "m73-recipe", None),   # blocked by W1
            ("probe_suite", "m73-suite", None),        # blocked by SR
            ("sample", env.sample.sample_id, env.model),  # by quality
            ("model", env.model, None),                # by SR + recipe
            ("dataset", env.ds, None),
            ("tokenizer", env.tok, None)):
        imp = _impact(env, family, aid, mid)
        assert imp.executable is False and imp.deletable is False
        assert imp.becomes_deletable == []
        assert imp.project_reclaimable_delta == 0
        assert imp.project_reclaimable_after == \
            imp.project_reclaimable_before


# --------------------------------------------------------------------------- #
# Tests 4 + 5: the REAL-DELETION oracle + reclaimable correctness
# --------------------------------------------------------------------------- #

def test_m73_real_deletion_oracle_and_reclaim(denv):
    forge = denv.forge

    # ---- W2: the chain head — predict, delete, verify --------------- #
    pred = _impact(denv, "workflow", denv.w2.workflow_id, denv.model)
    assert ("gate", denv.gw) in _becomes(pred)
    assert ("comparison", denv.cg) not in _becomes(pred)
    forge.delete_model_record(denv.model, "workflow",
                              denv.w2.workflow_id)
    assert forge.model_record_retention_overview(
        denv.model, "gate", denv.gw).deletable is True
    assert forge.model_record_retention_overview(
        denv.model, "comparison", denv.cg).deletable is False
    ov = forge.project_retention_overview()
    assert ov.reclaimable_bytes == pred.project_reclaimable_after
    assert (pred.project_reclaimable_delta ==
            pred.project_reclaimable_after
            - pred.project_reclaimable_before)

    # ---- the registry-policy gate: unblocks its comparison AND the
    #      policy itself (a DEFINITION entering the becomes set) ---- #
    c_b = denv.forge.gates.get_decision(
        denv.model_b, denv.g_b.decision_id).comparison_id
    pred = _impact(denv, "gate", denv.g_b.decision_id, denv.model_b)
    assert _becomes(pred) == {("comparison", c_b),
                              ("gate_policy", denv.policy2)}
    forge.delete_model_record(denv.model_b, "gate",
                              denv.g_b.decision_id)
    assert forge.model_record_retention_overview(
        denv.model_b, "comparison", c_b).deletable is True
    assert forge.definition_retention_overview(
        "gate_policy", denv.policy2).deletable is True
    assert forge.model_retention_overview(denv.model_b).deletable \
        is False  # both policies still target model B
    assert forge.project_retention_overview().reclaimable_bytes == \
        pred.project_reclaimable_after
    # really delete the unblocked policy (M71) so the SECOND policy
    # becomes model B's only remaining blocker
    forge.delete_definition("gate_policy", denv.policy2)

    # ---- the policy: deleting it unblocks model B (whole dir) ------- #
    pred = _impact(denv, "gate_policy", denv.policy)
    assert _becomes(pred) == {("model", denv.model_b)}
    b_before = forge.model_retention_overview(denv.model_b)
    assert pred.becomes_deletable[0].size_bytes == b_before.size_bytes
    forge.delete_definition("gate_policy", denv.policy)
    assert forge.model_retention_overview(denv.model_b).deletable is True
    assert forge.project_retention_overview().reclaimable_bytes == \
        pred.project_reclaimable_after

    # ---- the suite run: unblocks its probe eval AND the suite ------- #
    pred = _impact(denv, "suite_run", denv.sr.suite_run_id, denv.model)
    assert _becomes(pred) == {("evaluation", denv.probe_eval),
                              ("probe_suite", "m73-suite")}
    forge.delete_suite_run(denv.model, denv.sr.suite_run_id)
    assert forge.model_record_retention_overview(
        denv.model, "evaluation", denv.probe_eval).deletable is True
    assert forge.definition_retention_overview(
        "probe_suite", "m73-suite").deletable is True
    assert forge.project_retention_overview().reclaimable_bytes == \
        pred.project_reclaimable_after

    # ---- the sample-quality measurement: unblocks the sample -------- #
    pred = _impact(denv, "sample_quality", denv.quality.evaluation_id,
                   denv.model)
    assert _becomes(pred) == {("sample", denv.sample.sample_id)}
    forge.delete_sample_evaluation(denv.model,
                                   denv.quality.evaluation_id)
    assert forge.sample_retention_overview(
        denv.model, denv.sample.sample_id).deletable is True
    assert forge.project_retention_overview().reclaimable_bytes == \
        pred.project_reclaimable_after

    # ---- G_D2: exactly its comparison (model D stays deletable) ----- #
    pred = _impact(denv, "gate", denv.g_d2.decision_id, denv.model_d)
    assert _becomes(pred) == {("comparison", denv.c_d)}
    forge.delete_model_record(denv.model_d, "gate",
                              denv.g_d2.decision_id)
    assert forge.model_record_retention_overview(
        denv.model_d, "comparison", denv.c_d).deletable is True
    assert forge.project_retention_overview().reclaimable_bytes == \
        pred.project_reclaimable_after

    # ---- W_D: the OVERLAP RULE — a record of a DELETABLE model:
    #      deleting it shrinks the reclaimable pool by EXACTLY its
    #      own bytes (the model's whole-dir contribution shrinks;
    #      the newly-deletable held checkpoint / stage evaluation
    #      add NOTHING — subsumed by the whole-dir rule, no double
    #      counting) -------------------------------------------------- #
    pred = _impact(denv, "workflow", denv.w_d.workflow_id, denv.model_d)
    w_d_view = forge.model_record_retention_overview(
        denv.model_d, "workflow", denv.w_d.workflow_id)
    assert pred.immediate_bytes == w_d_view.size_bytes
    assert pred.project_reclaimable_delta == -w_d_view.size_bytes
    forge.delete_model_record(denv.model_d, "workflow",
                              denv.w_d.workflow_id)
    assert forge.project_retention_overview().reclaimable_bytes == \
        pred.project_reclaimable_after

    # ---- model D itself (deletable, with records): the whole dir
    #      leaves the reclaimable pool (its records were subsumed) -- #
    pred = _impact(denv, "model", denv.model_d)
    assert pred.executable is True
    d_view = forge.model_retention_overview(denv.model_d)
    assert pred.project_reclaimable_delta == -d_view.size_bytes
    forge.delete_model(denv.model_d)
    assert forge.project_retention_overview().reclaimable_bytes == \
        pred.project_reclaimable_after


# --------------------------------------------------------------------------- #
# Tests 6 + 7: no mutation + deterministic repeats
# --------------------------------------------------------------------------- #

def test_m73_no_mutation_and_determinism(env):
    forge = env.forge
    before = _inventory(env.root)
    probes = [
        ("model", env.model, None),
        ("dataset", env.ds, None),
        ("tokenizer", env.tok, None),
        ("workflow_recipe", "m73-recipe", None),
        ("gate_policy", env.policy, None),
        ("probe_suite", "m73-suite", None),
        ("checkpoint", env.ck[0], env.model),
        ("workflow", env.w2.workflow_id, env.model),
        ("evaluation", env.direct_eval.eval_id, env.model),
        ("comparison", env.cg, env.model),
        ("gate", env.gw, env.model),
        ("suite_run", env.sr.suite_run_id, env.model),
        ("sample", env.sample.sample_id, env.model),
        ("sample_quality", env.quality.evaluation_id, env.model),
    ]
    first = None
    for family, aid, mid in probes:
        one = forge.deletion_impact_preview(family, aid, mid)
        two = forge.deletion_impact_preview(family, aid, mid)
        assert one.model_dump(mode="json") == two.model_dump(
            mode="json"), (family, aid)
        if first is None:
            first = (family, aid, one.model_dump(mode="json"))
    # a third call of the first probe is still byte-identical
    again = forge.deletion_impact_preview(first[0], first[1])
    assert again.model_dump(mode="json") == first[2]
    assert _inventory(env.root) == before
    assert not list((env.root / "tmp").iterdir())


# --------------------------------------------------------------------------- #
# Test 8: integrity refusal (tampered artifact — disposable root)
# --------------------------------------------------------------------------- #

def test_m73_integrity_refusal(tmp_path):
    e = _build(tmp_path)
    forge = e.forge
    mpath = (e.root / "models" / e.model / "workflows" /
             f"workflow-{e.w2.workflow_id}" / "manifest.json")
    original = mpath.read_bytes()
    tampered = json.loads(original)
    tampered["result_hash"] = "0" * 64
    mpath.write_text(json.dumps(tampered, indent=2, sort_keys=True))
    imp = forge.deletion_impact_preview("workflow", e.w2.workflow_id,
                                        e.model)
    assert imp.integrity_verified is False
    assert imp.deletable is False and imp.executable is False
    assert imp.becomes_deletable == []
    assert imp.project_reclaimable_delta == 0
    assert imp.project_reclaimable_after == \
        imp.project_reclaimable_before
    # the dependents are still factual (current-state edges)
    assert ("gate", e.gw, "workflow") in _deps(imp)
    # restore -> integrity passes again
    mpath.write_bytes(original)
    imp = forge.deletion_impact_preview("workflow", e.w2.workflow_id,
                                        e.model)
    assert imp.integrity_verified is True and imp.executable is True


# --------------------------------------------------------------------------- #
# Tests 9 + G: the API surface — 404s, OpenAPI, determinism
# --------------------------------------------------------------------------- #

def test_m73_api_lifecycle_and_openapi(api_client):
    c = api_client
    import json as _json
    up = c.post("/api/v1/datasets/upload", files=[
        ("files", ("m73api.txt", _corpus(80, "m73api"), "text/plain"))],
        data={"name": "m73api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = c.post("/api/v1/tokenizers/train", data={
        "config": _json.dumps({"name": "m73api-tok",
                               "vocab_size": 320}),
        "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert c.post(f"/api/v1/datasets/{ds}/tokenize",
                  json={"tokenizer_id": tok}).status_code == 200
    m = c.post("/api/v1/models", json={
        "config": {"name": "m73api-model", "vocab_size": 640,
                   "context_length": 64, "hidden_size": 64, "n_layers": 2,
                   "n_heads": 4, "n_kv_heads": 2,
                   "intermediate_size": 128, "seed": 1}})
    assert m.status_code == 201, m.text
    mid = m.json()["model"]["id"]
    run = c.post("/api/v1/training/run", json={
        "method": "continued_pretraining", "model_id": mid,
        "dataset_id": ds, "tokenizer_id": tok, "learning_rate": 3e-3,
        "batch_size": 8, "max_seq_len": 32, "eval_every_steps": 4,
        "keep_best": False, "seed": 1, "steps": 4})
    assert run.status_code == 200, run.text
    ck = sorted(cp["checkpoint_id"] for cp in c.get(
        f"/api/v1/models/{mid}/checkpoints").json())
    ev = c.post("/api/v1/evaluations/run", json={
        "model_id": mid, "dataset_id": ds, "tokenizer_id": tok,
        "split": "validation", "batch_size": 8, "max_seq_len": 32,
        "seed": 2, "checkpoint_id": ck[0]})
    assert ev.status_code == 200, ev.text
    eid = ev.json()["eval_id"]

    # the evaluation's impact: current state == its retention view
    r1 = c.get(f"/api/v1/models/{mid}/evaluations/{eid}"
               "/retention/impact")
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert body["family"] == "evaluation" and body["model_id"] == mid
    assert body["deletable"] is True and body["executable"] is True
    assert {(d["family"], d["artifact_id"], d["reference_category"])
            for d in body["immediate_dependents"]} == {
        ("checkpoint", ck[0], "evaluation"),
        ("dataset", ds, "evaluation"),
        ("tokenizer", tok, "evaluation")}
    r2 = c.get(f"/api/v1/models/{mid}/evaluations/{eid}"
               "/retention/impact")
    assert r1.content == r2.content  # deterministic byte-identical

    # unknown artifacts -> 404 on ALL 14 impact routes
    paths = [
        "/api/v1/models/no-such/retention/impact",
        "/api/v1/datasets/no-such/retention/impact",
        "/api/v1/tokenizers/no-such/retention/impact",
        "/api/v1/workflows/recipes/no-such/retention/impact",
        "/api/v1/policies/no-such/retention/impact",
        "/api/v1/probe-suites/no-such/retention/impact",
        "/api/v1/models/no-such/checkpoints/no-such/retention/impact",
        "/api/v1/models/no-such/workflows/no-such/retention/impact",
        "/api/v1/models/no-such/evaluations/no-such/retention/impact",
        "/api/v1/models/no-such/comparisons/no-such/retention/impact",
        "/api/v1/models/no-such/gates/decisions/no-such/retention/impact",
        "/api/v1/models/no-such/suite-runs/no-such/retention/impact",
        "/api/v1/models/no-such/samples/no-such/retention/impact",
        "/api/v1/models/no-such/sample-quality/no-such/retention/impact",
    ]
    for p in paths:
        assert c.get(p).status_code == 404, p

    # OpenAPI: 119 paths / 14 deletes / 14 GET-only impact routes
    spec = c.get("/openapi.json").json()
    assert len(spec["paths"]) == 119
    deletes = [p for p, ops in spec["paths"].items() if "delete" in ops]
    assert len(deletes) == 14
    impacts = sorted(p for p in spec["paths"]
                     if p.endswith("/retention/impact"))
    assert len(impacts) == 14
    for p in impacts:
        assert set(spec["paths"][p].keys()) == {"get"}, p
    for s in ("DeletionImpactPreview", "ImpactBlocker",
              "ImpactDependent", "ImpactBecomesDeletable"):
        assert s in spec["components"]["schemas"], s


# --------------------------------------------------------------------------- #
# Test 13: the independent RAW-MANIFEST oracle (no app analysis)
# --------------------------------------------------------------------------- #

def _raw_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_m73_independent_raw_manifest_oracle(env):
    root = env.root
    mdir = root / "models" / env.model

    # ---- oracle for W2's dependents + becomes, from raw manifests -- #
    w2 = _raw_json(mdir / "workflows" /
                   f"workflow-{env.w2.workflow_id}" / "manifest.json")
    # stage artifacts: the gate decision id
    gate_ids = {st["artifact"]["artifact_id"] for st in w2["stages"]
                if st.get("artifact")
                and st["artifact"]["kind"] == "gate_decision"}
    assert gate_ids == {env.gw}
    # plan data refs (raw scan of the plan configs)
    plan_ds = set()
    plan_tok = set()
    for st in w2["plan"]["stages"]:
        cfg = st.get("training")
        if not cfg and st.get("evaluation"):
            cfg = st["evaluation"].get("config")
        if cfg:
            if cfg.get("dataset_id"):
                plan_ds.add(cfg["dataset_id"])
            if cfg.get("tokenizer_id"):
                plan_tok.add(cfg["tokenizer_id"])
        if st.get("gate"):
            pol = st["gate"].get("policy") or {}
            if pol.get("dataset_id"):
                plan_ds.add(pol["dataset_id"])
            if pol.get("tokenizer_id"):
                plan_tok.add(pol["tokenizer_id"])
    assert plan_ds == {env.ds} and plan_tok == {env.tok}

    # who references G_W? raw scan of every workflow record manifest
    refs_to_gw = set()
    for wf in (mdir / "workflows").glob("*/manifest.json"):
        rec = _raw_json(wf)
        for st in rec.get("stages", []):
            art = st.get("artifact")
            if art and art.get("artifact_id") == env.gw:
                refs_to_gw.add(("workflow", rec["workflow_id"]))
    assert refs_to_gw == {("workflow", env.w2.workflow_id)}

    imp = _impact(env, "workflow", env.w2.workflow_id, env.model)
    d = _deps(imp)
    assert ("gate", env.gw, "workflow") in d
    assert ("dataset", env.ds, "workflow") in d
    assert ("tokenizer", env.tok, "workflow") in d
    # G_W becomes deletable (its ONLY raw blocker is W2)
    assert ("gate", env.gw) in _becomes(imp)
    # C_G does NOT (the raw gate manifest still references it)
    gate = _raw_json(mdir / "gates" / f"gate-{env.gw}" / "manifest.json")
    assert gate["comparison_id"] == env.cg
    assert ("comparison", env.cg) not in _becomes(imp)

    # ---- oracle for the suite run's dependents + becomes ----------- #
    sr = _raw_json(root / "suite-runs" /
                   f"{env.sr.suite_run_id}" / "manifest.json")
    assert sr["model_id"] == env.model and sr["suite_id"] == "m73-suite"
    probe_evals = {r["evaluation_id"] for r in sr["results"]
                   if r.get("evaluation_id")}
    assert probe_evals == {env.probe_eval}
    # who references the suite? raw scan of suite runs
    suite_refs = {r["suite_id"] for r in
                  (_raw_json(p / "manifest.json")
                   for p in (root / "suite-runs").iterdir()
                   if (p / "manifest.json").exists())}
    assert suite_refs == {"m73-suite"}
    imp = _impact(env, "suite_run", env.sr.suite_run_id, env.model)
    d = _deps(imp)
    assert d == {("model", env.model, "suite_run"),
                 ("probe_suite", "m73-suite", "suite_run"),
                 ("evaluation", env.probe_eval, "suite_run"),
                 ("dataset", env.ds, "suite_run"),
                 ("tokenizer", env.tok, "suite_run")}
    b = _becomes(imp)
    # the probe eval's raw blockers: who references it? (only SR)
    refs_to_eval = set()
    for rec_p in (mdir / "comparisons").glob("*/manifest.json"):
        rec = _raw_json(rec_p)
        for side in (rec["state_a"], rec["state_b"]):
            if side.get("evaluation_id") == env.probe_eval:
                refs_to_eval.add(rec["comparison_id"])
    for rec_p in (mdir / "gates").glob("*/manifest.json"):
        rec = _raw_json(rec_p)
        for side_name in ("candidate", "baseline"):
            side = rec.get(side_name)
            if side and side.get("evaluation_id") == env.probe_eval:
                refs_to_eval.add(rec["decision_id"])
    assert not refs_to_eval  # nothing else references the probe eval
    assert ("evaluation", env.probe_eval) in b
    assert ("probe_suite", "m73-suite") in b
    assert ("model", env.model) not in b  # A stays blocked (recipe)
