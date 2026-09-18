"""Milestone 71 tests: explicit definition retention.

The M71 invariants under test (the M70 pattern, lifted to the three
root-level DEFINITION families — workflow recipes, gate policies,
probe suites):

* ONE BLOCKER SOURCE — the guard's blockers are EXACTLY the
  persisted records that reference the definition: workflow runs
  with recipe provenance, COMPOSITE recipes' composition references,
  gate decisions with policy provenance, suite runs with the suite
  id, and the STRUCTURAL directions (recipes whose gate stage names
  the policy / whose suite-run stage names the suite). Proven
  against an independent RAW-MANIFEST oracle; the retention view
  shows the SAME list.
* INTEGRITY FIRST — scope (unknown/registry-invisible -> 404,
  nothing deleted) -> the definition's persisted content hash must
  reproduce (tampered -> RuntimeError -> 409, never deletable) ->
  blockers -> ONE atomic removal with deterministic files/bytes.
* THE MODEL-UNBLOCK INVARIANT (§7) — a recipe/policy IS an M66
  external model reference; deleting the definition removes exactly
  that reference (no other reference vanishes), and the LAST
  external reference flips the UNTOUCHED M67 model retention state.
* THE UNBLOCK CHAIN (§8) — deleting a dependent record (M70/M68)
  unblocks the definition live; structural recipe references
  unblock through recipe deletion. No guard code is bypassed.
* NO CASCADE — a blocked deletion changes nothing on disk; a
  successful deletion removes ONLY the definition's own directory
  (and changes NO dataset/tokenizer blocker surface: suites are not
  M64 categories — suite RUNS are).
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
    StageStateRef,
    StageType,
    SuiteProbe,
    SuiteRunRequest,
    TokenizerConfig,
    TransformerConfig,
    TrainingConfig,
    WorkflowEvaluationStage,
    WorkflowGateStage,
    WorkflowRecipeCallStage,
    WorkflowRecipeCreateRequest,
    WorkflowStage,
    WorkflowSuiteRunStage,
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


def _walk(directory: Path) -> tuple[list[str], int]:
    files = sorted(p.relative_to(directory).as_posix()
                   for p in directory.rglob("*") if p.is_file())
    return files, sum(p.stat().st_size for p in directory.rglob("*")
                      if p.is_file())


def _pairs(blockers) -> list[tuple[str, str]]:
    return [(b.category, b.reference_id) for b in blockers]


def _ds_pairs(blockers) -> list[tuple[str, str]]:
    """(reason, detail) pairs of the M65 artifact blockers (a DIFFERENT
    schema than the M71 definition blockers — reason + joined ids)."""
    return [(b.reason, b.detail) for b in blockers]


# --------------------------------------------------------------------------- #
# Independent oracle: raw manifest parse of EVERY definition reference
# --------------------------------------------------------------------------- #

def _oracle(root: Path) -> dict[tuple[str, str], set[tuple[str, str]]]:
    """(family, definition_id) -> {(category, reference_id)} — every
    persisted reference TO a recipe/policy/suite, derived from raw
    manifests (never the app's analysis)."""
    refs: dict = {}

    def add(family, did, category, rid):
        refs.setdefault((family, did), set()).add((category, rid))

    # workflows -> recipes (recipe_id provenance, every model)
    for wdir in (root / "models").glob("*/workflows/*"):
        if not (wdir / "manifest.json").exists():
            continue
        rec = json.loads((wdir / "manifest.json").read_text())
        if rec.get("recipe_id"):
            add("workflow_recipe", rec["recipe_id"], "workflow",
                rec["workflow_id"])
    # recipes -> recipes (composition) and recipes -> policies/suites
    for rdir in (root / "workflow-recipes").glob("*/manifest.json"):
        rec = json.loads(rdir.read_text())
        rid = rec["recipe_id"]
        for ref in rec.get("composition") or []:
            add("workflow_recipe", ref["recipe_id"], "workflow_recipe", rid)
        for st in rec.get("stages", []):
            if st.get("type") == "gate" and st.get("gate", {}).get(
                    "policy_id"):
                add("gate_policy", st["gate"]["policy_id"],
                    "workflow_recipe", rid)
            if st.get("type") == "suite_run" and st.get(
                    "suite_run", {}).get("suite_id"):
                add("probe_suite", st["suite_run"]["suite_id"],
                    "workflow_recipe", rid)
    # gates -> policies (registry provenance, every model)
    for gdir in (root / "models").glob("*/gates/*"):
        if not (gdir / "manifest.json").exists():
            continue
        rec = json.loads((gdir / "manifest.json").read_text())
        if rec.get("policy_id"):
            add("gate_policy", rec["policy_id"], "gate",
                rec["decision_id"])
    # suite runs -> suites (suite id, every model)
    for sdir in (root / "suite-runs").glob("*"):
        if not (sdir / "manifest.json").exists():
            continue
        rec = json.loads((sdir / "manifest.json").read_text())
        if rec.get("suite_id"):
            add("probe_suite", rec["suite_id"], "suite_run",
                rec["suite_run_id"])
    return refs


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

class Env:
    pass


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env()
    e.root = tmp_path_factory.mktemp("m71-root")
    e.forge = ModelForge(root=e.root)
    forge = e.forge
    up = forge.upload_dataset([("m71.txt", _corpus(190, "m71"))],
                              name="m71-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m71-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)

    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m71-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.model, dataset_id=e.ds,
        tokenizer_id=e.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1,
        steps=8))
    e.ck = sorted(c.checkpoint_id for c in
                  forge.list_checkpoints(e.model))

    # --- recipes -------------------------------------------------------- #
    # base recipe (evaluate one checkpoint) + ONE run -> workflow blocker
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-recipe", description="M71 base",
        stages=[WorkflowStage(
            stage_id="ev", type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=EvaluationConfig(
                    model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
                    split="validation", batch_size=8, max_seq_len=32,
                    seed=6, checkpoint_id=e.ck[0])))]))
    e.recipe_run = forge.run_workflow_recipe("m71-recipe", e.model)
    # leaf recipe (registered, never run, nothing references it)
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-leaf-recipe", description="M71 leaf",
        stages=[WorkflowStage(
            stage_id="ev", type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=EvaluationConfig(
                    model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
                    split="validation", batch_size=8, max_seq_len=32,
                    seed=7, checkpoint_id=e.ck[0])))]))
    # composite recipe -> composition blocker on the base recipe
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-comp", description="M71 composite",
        stages=[WorkflowStage(
            stage_id="call", type=StageType.RECIPE,
            recipe=WorkflowRecipeCallStage(recipe_id="m71-recipe"))]))

    # --- policies ------------------------------------------------------- #
    e.policy = forge.register_policy(PolicyCreateRequest(
        policy_id="m71-policy", description="M71 policy",
        policy=GatePolicy(
            name="m71-policy", model_id=e.model, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=8, baseline_type="checkpoint",
            baseline_checkpoint_id=e.ck[1],
            tolerance=1.0))).policy_id
    # ONE gate decision from the registry policy -> gate blocker
    e.gate = forge.run_gate(GateRequest(
        model_id=e.model, policy_id=e.policy,
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=e.ck[0])))
    # a recipe whose gate stage names the policy -> STRUCTURAL blocker
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-recipe-gated", description="M71 gated",
        stages=[WorkflowStage(
            stage_id="gt", type=StageType.GATE,
            gate=WorkflowGateStage(
                policy_id=e.policy,
                candidate=StageStateRef(
                    state_kind=EvalStateKind.CHECKPOINT,
                    checkpoint_id=e.ck[0])))]))

    # --- probe suites --------------------------------------------------- #
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m71-suite", description="M71 suite",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))
    # suite runs on TWO models -> cross-model blocker scan
    e.suite_run_a = forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m71-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=e.ck[0])))
    e.model_b = forge.create_model(ModelCreateRequest(
        config=TransformerConfig(
            name="m71-model-b", vocab_size=640, context_length=64,
            hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
            intermediate_size=128, seed=2)))[0].id
    e.suite_run_b = forge.run_suite(SuiteRunRequest(
        model_id=e.model_b, suite_id="m71-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    # a recipe whose suite-run stage names the suite -> STRUCTURAL blocker
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-recipe-suited", description="M71 suited",
        stages=[WorkflowStage(
            stage_id="sr", type=StageType.SUITE_RUN,
            suite_run=WorkflowSuiteRunStage(
                suite_id="m71-suite",
                state=StageStateRef(
                    state_kind=EvalStateKind.CHECKPOINT,
                    checkpoint_id=e.ck[0])))]))
    # leaf suite (registered, never run)
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m71-leaf-suite", description="M71 leaf suite",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=43)]))

    # --- the §7 final-ref model: untrained, ONLY external ref = a policy
    e.model_d = forge.create_model(ModelCreateRequest(
        config=TransformerConfig(
            name="m71-model-d", vocab_size=640, context_length=64,
            hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
            intermediate_size=128, seed=3)))[0].id
    e.policy_d = forge.register_policy(PolicyCreateRequest(
        policy_id="m71-policy-d", description="M71 final-ref",
        policy=GatePolicy(
            name="m71-policy-d", model_id=e.model_d, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=9, baseline_type="checkpoint",
            baseline_checkpoint_id=e.ck[1],
            tolerance=1.0))).policy_id
    return e


# --------------------------------------------------------------------------- #
# Engine: one blocker source (raw-manifest oracle), views, zero mutation
# --------------------------------------------------------------------------- #

def test_m71_blockers_match_oracle_and_views(env):
    forge, root = env.forge, env.root
    oracle = _oracle(root)
    before = _inventory(root)

    cases = [
        ("workflow_recipe", "m71-recipe",
         [("workflow", env.recipe_run.workflow_id),
          ("workflow_recipe", "m71-comp")]),
        ("workflow_recipe", "m71-comp", []),
        ("workflow_recipe", "m71-leaf-recipe", []),
        ("gate_policy", "m71-policy",
         [("gate", env.gate.decision_id),
          ("workflow_recipe", "m71-recipe-gated")]),
        ("probe_suite", "m71-suite",
         sorted([("suite_run", env.suite_run_a.suite_run_id),
                 ("suite_run", env.suite_run_b.suite_run_id)]) +
         [("workflow_recipe", "m71-recipe-suited")]),
        ("probe_suite", "m71-leaf-suite", []),
    ]
    for family, did, expected in cases:
        # the guard's ordered blockers == the expected canonical list
        blockers = forge.definition_deletion_blockers(family, did)
        assert _pairs(blockers) == expected, (family, did)
        # ... == the independent raw-manifest oracle (set equality)
        assert {(b.category, b.reference_id) for b in blockers} == \
            oracle.get((family, did), set()), (family, did)
        # the retention view shows the SAME list and agrees on
        # deletability
        view = forge.definition_retention_overview(family, did)
        assert _pairs(view.blockers) == expected, (family, did)
        assert view.deletable == (not expected), (family, did)
        assert view.integrity_verified is True
        assert view.family == family and view.definition_id == did
        # files/bytes == an independent walk of the definition's dir
        dirs = {"workflow_recipe": root / "workflow-recipes" / did,
                "gate_policy": root / "policies" / did,
                "probe_suite": root / "probe-suites" / did}
        files, nbytes = _walk(dirs[family])
        assert view.files == files and view.size_bytes == nbytes
        assert view.files and view.size_bytes > 0

    # model ids per family (the ONE M66 analysis reused)
    assert forge.definition_model_ids(
        "workflow_recipe", "m71-recipe") == [env.model]
    assert forge.definition_model_ids("gate_policy", "m71-policy") == \
        [env.model]
    assert forge.definition_model_ids("probe_suite", "m71-suite") == []
    # out-of-scope family -> ValueError; unknown -> FileNotFoundError
    with pytest.raises(ValueError):
        forge.definition_retention_overview("model", "x")
    with pytest.raises(FileNotFoundError):
        forge.definition_retention_overview("gate_policy", "no-such")

    # blocked engine DELETEs refuse (ValueError) and leaf deletes are
    # deferred to the chain test; EVERYTHING above mutated nothing
    with pytest.raises(ValueError, match="m71-recipe"):
        forge.delete_definition("workflow_recipe", "m71-recipe")
    with pytest.raises(ValueError, match="m71-policy"):
        forge.delete_definition("gate_policy", "m71-policy")
    with pytest.raises(ValueError, match="m71-suite"):
        forge.delete_definition("probe_suite", "m71-suite")
    assert _inventory(root) == before


# --------------------------------------------------------------------------- #
# Engine: §7 model-unblock invariant + §8 dependency chains
# --------------------------------------------------------------------------- #

def test_m71_model_unblock_and_dependent_chains(env):
    forge, root = env.forge, env.root

    # ---- §7: the recipe IS an M66 external model reference ------------ #
    usage = forge.model_usage_overview(env.model)
    cats = {c.category: c.references for c in usage.categories}
    # every recipe whose stage configs name model A (the base, leaf,
    # gated and suited recipes); the policy names model A too
    recipes_before = sorted(cats["workflow_recipe"])
    assert "m71-recipe" in recipes_before
    assert cats["policy"] == ["m71-policy"]

    # ---- §8 recipe chain: workflow run -> composite -> base recipe ---- #
    # blocked by the run AND the composite
    assert _pairs(forge.definition_deletion_blockers(
        "workflow_recipe", "m71-recipe")) == [
        ("workflow", env.recipe_run.workflow_id),
        ("workflow_recipe", "m71-comp")]
    with pytest.raises(ValueError):
        forge.delete_definition("workflow_recipe", "m71-recipe")
    # deleting the composite (a leaf) removes the structural blocker
    comp_view = forge.definition_retention_overview(
        "workflow_recipe", "m71-comp")
    res = forge.delete_definition("workflow_recipe", "m71-comp")
    assert res.family == "workflow_recipe"
    assert res.definition_id == "m71-comp"
    assert res.files_removed == len(comp_view.files)
    assert res.bytes_reclaimed == comp_view.size_bytes
    assert not (root / "workflow-recipes" / "m71-comp").exists()
    assert _pairs(forge.definition_deletion_blockers(
        "workflow_recipe", "m71-recipe")) == [
        ("workflow", env.recipe_run.workflow_id)]
    # deleting the workflow run (M70) unblocks the recipe
    forge.delete_model_record(env.model, "workflow",
                              env.recipe_run.workflow_id)
    view = forge.definition_retention_overview(
        "workflow_recipe", "m71-recipe")
    assert view.deletable is True and view.blockers == []
    # §7 snapshot: every OTHER category, taken right before the
    # definition deletion (the M70 record deletion above is the
    # test's own unblock step, not part of the §7 invariant)
    other_before = {
        k: list(v) for k, v in
        {c.category: c.references
         for c in forge.model_usage_overview(env.model).categories}.items()
        if k not in ("workflow_recipe", "policy")}
    # the M66 external reference is STILL there before the deletion
    # (the composite never referenced the model directly — its single
    # recipe-call stage names no model_id)
    assert {c.category: c.references for c in
            forge.model_usage_overview(env.model).categories}[
                "workflow_recipe"] == recipes_before
    res = forge.delete_definition("workflow_recipe", "m71-recipe")
    assert res.files_removed == len(view.files)
    assert res.bytes_reclaimed == view.size_bytes
    assert not (root / "workflow-recipes" / "m71-recipe").exists()
    # §7: EXACTLY that reference vanished — nothing else changed
    cats_after = {c.category: c.references
                  for c in forge.model_usage_overview(env.model).categories}
    assert cats_after["workflow_recipe"] == [
        r for r in recipes_before if r != "m71-recipe"]
    assert cats_after["policy"] == ["m71-policy"]
    assert {k: list(v) for k, v in cats_after.items()
            if k not in ("workflow_recipe", "policy")} == other_before
    # ... and the recipe is now unknown (404 at the API)
    with pytest.raises(FileNotFoundError):
        forge.definition_retention_overview("workflow_recipe", "m71-recipe")

    # ---- §8 policy chain: gate decision + structural recipe ------------ #
    assert _pairs(forge.definition_deletion_blockers(
        "gate_policy", "m71-policy")) == [
        ("gate", env.gate.decision_id),
        ("workflow_recipe", "m71-recipe-gated")]
    # removing ONLY the gate decision leaves the structural blocker
    forge.delete_model_record(env.model, "gate", env.gate.decision_id)
    assert _pairs(forge.definition_deletion_blockers(
        "gate_policy", "m71-policy")) == [
        ("workflow_recipe", "m71-recipe-gated")]
    with pytest.raises(ValueError):
        forge.delete_definition("gate_policy", "m71-policy")
    # deleting the referencing recipe unblocks the policy
    forge.delete_definition("workflow_recipe", "m71-recipe-gated")
    view = forge.definition_retention_overview("gate_policy", "m71-policy")
    assert view.deletable is True
    res = forge.delete_definition("gate_policy", "m71-policy")
    assert res.family == "gate_policy"
    assert not (root / "policies" / "m71-policy").exists()
    # §7: the policy was model A's M66 external reference — now gone
    # (the gated recipe referencing it was deleted just before, so the
    # workflow_recipe tail holds only the never-deleted leaf/suited
    # recipes — the suited one goes with the suite chain below)
    cats_after = {c.category: c.references
                  for c in forge.model_usage_overview(env.model).categories}
    assert cats_after["policy"] == []
    assert "m71-recipe-gated" not in cats_after["workflow_recipe"]

    # ---- §7 final-ref: deleting the LAST external reference flips the
    #      UNTOUCHED M67 model retention state --------------------------- #
    m67_before = forge.model_retention_overview(env.model_d)
    assert m67_before.deletable is False
    assert _pairs(m67_before.blockers) == [("policy", "m71-policy-d")]
    res = forge.delete_definition("gate_policy", "m71-policy-d")
    assert res.definition_id == "m71-policy-d"
    m67_after = forge.model_retention_overview(env.model_d)
    assert m67_after.deletable is True
    assert m67_after.blockers == []
    # the model itself is untouched (no cascade): still registered
    assert any(m.id == env.model_d
               for m in forge.list_models())

    # ---- §8 suite chain: cross-model suite runs + structural recipe ---- #
    assert _pairs(forge.definition_deletion_blockers(
        "probe_suite", "m71-suite")) == sorted([
        ("suite_run", env.suite_run_a.suite_run_id),
        ("suite_run", env.suite_run_b.suite_run_id)]) + [
        ("workflow_recipe", "m71-recipe-suited")]
    # delete both suite runs (M68) — the cross-model scan surface
    forge.delete_suite_run(env.model, env.suite_run_a.suite_run_id)
    assert _pairs(forge.definition_deletion_blockers(
        "probe_suite", "m71-suite")) == [
        ("suite_run", env.suite_run_b.suite_run_id),
        ("workflow_recipe", "m71-recipe-suited")]  # a's run removed
    forge.delete_suite_run(env.model_b, env.suite_run_b.suite_run_id)
    # still blocked by the structural recipe reference
    with pytest.raises(ValueError):
        forge.delete_definition("probe_suite", "m71-suite")
    forge.delete_definition("workflow_recipe", "m71-recipe-suited")
    view = forge.definition_retention_overview("probe_suite", "m71-suite")
    assert view.deletable is True and view.model_ids == []
    # the dataset's M65 blockers RIGHT BEFORE the suite deletion (the
    # deleted suite RUNS already left the surface via M68; suite
    # DEFINITIONS were never M64 categories)
    pre = _ds_pairs(forge.dataset_deletion_blockers(env.ds))
    assert not any(cat == "suite_run" for cat, _ in pre)
    res = forge.delete_definition("probe_suite", "m71-suite")
    assert res.family == "probe_suite"
    assert res.files_removed == len(view.files)
    assert res.bytes_reclaimed == view.size_bytes
    assert not (root / "probe-suites" / "m71-suite").exists()
    # the suite deletion itself changed NOTHING on the dataset's
    # blocker surface (no M64/M65 coupling — only suite RUNS block)
    assert _ds_pairs(forge.dataset_deletion_blockers(env.ds)) == pre
    assert len(pre) >= 1  # other blockers (training, evals) remain

    # ---- leaf definitions delete cleanly (empty blocker set) ----------- #
    for family, did, directory in (
            ("workflow_recipe", "m71-leaf-recipe",
             root / "workflow-recipes" / "m71-leaf-recipe"),
            ("probe_suite", "m71-leaf-suite",
             root / "probe-suites" / "m71-leaf-suite")):
        view = forge.definition_retention_overview(family, did)
        assert view.deletable is True and view.blockers == []
        res = forge.delete_definition(family, did)
        assert res.definition_id == did
        assert res.files_removed == len(view.files)
        assert res.bytes_reclaimed == view.size_bytes
        assert not directory.exists()

    # ---- no cascade: the model, dataset and tokenizer survive ----------- #
    assert any(m.id == env.model for m in forge.list_models())
    assert forge.datasets.load_meta(env.ds) is not None
    assert forge.tokenizers.load(env.tok) is not None


# --------------------------------------------------------------------------- #
# Engine: integrity first, scope, atomicity (disposable root)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tamper_env(tmp_path):
    e = Env()
    e.root = tmp_path
    e.forge = ModelForge(root=tmp_path)
    forge = e.forge
    up = forge.upload_dataset([("m71t.txt", _corpus(60, "m71t"))],
                              name="m71t-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m71t-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, tok.id)
    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m71t-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.model, dataset_id=e.ds,
        tokenizer_id=e.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1,
        steps=4))
    e.ck = sorted(c.checkpoint_id for c in
                  forge.list_checkpoints(e.model))
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71t-recipe", stages=[WorkflowStage(
            stage_id="ev", type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=EvaluationConfig(
                    model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
                    split="validation", batch_size=8, max_seq_len=32,
                    seed=2, checkpoint_id=e.ck[0])))]))
    forge.register_policy(PolicyCreateRequest(
        policy_id="m71t-policy",
        policy=GatePolicy(
            name="m71t-policy", model_id=e.model, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=3, baseline_type="checkpoint",
            baseline_checkpoint_id=e.ck[0], tolerance=1.0)))
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m71t-suite",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=5)]))
    return e


def _tamper(directory: Path, mutate) -> None:
    mpath = directory / "manifest.json"
    rec = json.loads(mpath.read_text())
    mutate(rec)
    mpath.write_text(json.dumps(rec, indent=2, sort_keys=True))


def test_m71_integrity_first_scope_and_atomicity(tamper_env):
    forge, root = tamper_env.forge, tamper_env.root

    # ---- scope: unknown -> FileNotFoundError, nothing deleted ---------- #
    before = _inventory(root)
    for family in ("workflow_recipe", "gate_policy", "probe_suite"):
        with pytest.raises(FileNotFoundError):
            forge.delete_definition(family, "no-such")

    # ---- integrity FIRST: tampered payload -> never deletable ---------- #
    # recipe: tamper the evaluate stage's seed (config no longer hashes)
    _tamper(root / "workflow-recipes" / "m71t-recipe",
            lambda r: r["stages"][0]["evaluation"]["config"].update(
                seed=99))
    view = forge.definition_retention_overview(
        "workflow_recipe", "m71t-recipe")
    assert view.integrity_verified is False
    assert view.deletable is False          # integrity gates deletability
    assert view.blockers == []              # even with NO dependents
    with pytest.raises(RuntimeError, match="integrity"):
        forge.delete_definition("workflow_recipe", "m71t-recipe")
    assert (root / "workflow-recipes" / "m71t-recipe" /
            "manifest.json").exists()

    # policy: tamper the tolerance inside the persisted policy
    _tamper(root / "policies" / "m71t-policy",
            lambda r: r["policy"].update(tolerance=0.123))
    view = forge.definition_retention_overview("gate_policy",
                                               "m71t-policy")
    assert view.integrity_verified is False and view.deletable is False
    with pytest.raises(RuntimeError, match="integrity"):
        forge.delete_definition("gate_policy", "m71t-policy")
    assert (root / "policies" / "m71t-policy" / "manifest.json").exists()

    # suite: tamper the probe set (probes no longer hash)
    _tamper(root / "probe-suites" / "m71t-suite",
            lambda r: r["probes"].append(
                dict(r["probes"][0], seed=77)))
    view = forge.definition_retention_overview("probe_suite",
                                               "m71t-suite")
    assert view.integrity_verified is False and view.deletable is False
    with pytest.raises(RuntimeError, match="integrity"):
        forge.delete_definition("probe_suite", "m71t-suite")
    assert (root / "probe-suites" / "m71t-suite" /
            "manifest.json").exists()

    # integrity refusals changed nothing on disk
    tampered = _inventory(root)
    for path, digest in tampered.items():
        if path.startswith(("workflow-recipes/m71t-recipe/",
                            "policies/m71t-policy/",
                            "probe-suites/m71t-suite/")):
            continue
        assert before[path] == digest, path

    # ---- scope: registry-invisible (unparseable manifest) -> 404 ------- #
    mpath = root / "workflow-recipes" / "m71t-recipe" / "manifest.json"
    keep = mpath.read_text()
    mpath.write_text("{not json")
    with pytest.raises(FileNotFoundError):
        forge.delete_definition("workflow_recipe", "m71t-recipe")
    with pytest.raises(FileNotFoundError):
        forge.definition_retention_overview("workflow_recipe",
                                             "m71t-recipe")
    mpath.write_text(keep)

    # ---- atomicity: success removes EXACTLY the definition's files ----- #
    # the three tampered definitions stay permanently protected (the
    # honest outcome — integrity is never bypassed); a fresh leaf
    # definition proves atomicity
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m71t-leaf",
        probes=[SuiteProbe(dataset_id=tamper_env.ds, split="validation",
                           tokenizer_id=tamper_env.tok, batch_size=8,
                           max_seq_len=32, seed=6)]))
    files, nbytes = _walk(root / "probe-suites" / "m71t-leaf")
    before_delete = _inventory(root)
    res = forge.delete_definition("probe_suite", "m71t-leaf")
    assert res.files_removed == len(files)
    assert res.bytes_reclaimed == nbytes
    after_delete = _inventory(root)
    removed = set(before_delete) - set(after_delete)
    assert removed == {f"probe-suites/m71t-leaf/{f}" for f in files}
    assert set(after_delete) == set(before_delete) - removed
    # every remaining file is byte-identical (no partial writes)
    for path, digest in after_delete.items():
        assert before_delete[path] == digest, path


# --------------------------------------------------------------------------- #
# API: lifecycle, structured 409s, OpenAPI surface
# --------------------------------------------------------------------------- #

def test_m71_api_lifecycle_and_openapi(api_client):
    c = api_client
    import json as _json

    # build a model with a checkpoint (API level)
    up = c.post("/api/v1/datasets/upload", files=[
        ("files", ("m71api.txt", _corpus(80, "m71api"), "text/plain"))],
        data={"name": "m71api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = c.post("/api/v1/tokenizers/train", data={
        "config": _json.dumps({"name": "m71api-tok",
                               "vocab_size": 320}),
        "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert c.post(f"/api/v1/datasets/{ds}/tokenize",
                  json={"tokenizer_id": tok}).status_code == 200
    m = c.post("/api/v1/models", json={
        "config": {"name": "m71api-model", "vocab_size": 640,
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

    # register a recipe (API) and run it once
    reg = c.post("/api/v1/workflows/recipes", json={
        "recipe_id": "m71api-recipe",
        "description": "M71 API",
        "stages": [{"stage_id": "ev", "type": "evaluate",
                    "evaluation": {"config": {
                        "model_id": mid, "dataset_id": ds,
                        "tokenizer_id": tok, "split": "validation",
                        "batch_size": 8, "max_seq_len": 32, "seed": 2,
                        "checkpoint_id": ck[0]}}}]})
    assert reg.status_code == 201, reg.text
    ran = c.post("/api/v1/workflows/recipes/m71api-recipe/runs",
                 json={"model_id": mid})
    assert ran.status_code == 200, ran.text
    wid = ran.json()["workflow_id"]

    # retention view: blocked by the workflow run
    view = c.get("/api/v1/workflows/recipes/m71api-recipe/retention")
    assert view.status_code == 200
    body = view.json()
    assert body["family"] == "workflow_recipe"
    assert body["deletable"] is False
    assert body["integrity_verified"] is True
    assert [(b["category"], b["reference_id"]) for b in
            body["blockers"]] == [("workflow", wid)]
    assert body["model_ids"] == [mid]
    assert body["files"] and body["size_bytes"] > 0

    # DELETE -> structured 409 with the SAME blocker list
    blocked = c.delete("/api/v1/workflows/recipes/m71api-recipe")
    assert blocked.status_code == 409
    detail = blocked.json()["detail"]
    assert detail["family"] == "workflow_recipe"
    assert detail["definition_id"] == "m71api-recipe"
    assert detail["protected"] is True
    assert "m71api-recipe" in detail["message"]
    assert [(b["category"], b["reference_id"]) for b in
            detail["blockers"]] == [("workflow", wid)]

    # unblock through M70 (delete the workflow record) -> DELETE 200
    assert c.delete(
        f"/api/v1/models/{mid}/workflows/{wid}").status_code == 200
    view = c.get("/api/v1/workflows/recipes/m71api-recipe/retention")
    assert view.json()["deletable"] is True
    gone = c.delete("/api/v1/workflows/recipes/m71api-recipe")
    assert gone.status_code == 200, gone.text
    result = gone.json()
    assert result["family"] == "workflow_recipe"
    assert result["definition_id"] == "m71api-recipe"
    assert result["files_removed"] >= 1
    assert result["bytes_reclaimed"] > 0
    # the definition is gone: GET 404, DELETE 404, retention 404
    assert c.get("/api/v1/workflows/recipes/m71api-recipe"
                 ).status_code == 404
    assert c.delete(
        "/api/v1/workflows/recipes/m71api-recipe").status_code == 404
    assert c.get("/api/v1/workflows/recipes/m71api-recipe/retention"
                 ).status_code == 404

    # policies: register (leaf) -> retention -> DELETE 200 -> 404
    pol = c.post("/api/v1/policies", json={
        "policy_id": "m71api-policy",
        "description": "M71 API policy",
        "policy": {"name": "m71api-policy", "model_id": mid,
                   "dataset_id": ds, "tokenizer_id": tok,
                   "split": "validation", "batch_size": 8,
                   "max_seq_len": 32, "seed": 3,
                   "baseline_type": "checkpoint",
                   "baseline_checkpoint_id": ck[0], "tolerance": 1.0}})
    assert pol.status_code == 201, pol.text
    view = c.get("/api/v1/policies/m71api-policy/retention")
    assert view.status_code == 200
    assert view.json()["deletable"] is True
    assert view.json()["model_ids"] == [mid]
    gone = c.delete("/api/v1/policies/m71api-policy")
    assert gone.status_code == 200
    assert gone.json()["files_removed"] >= 1
    assert c.delete("/api/v1/policies/m71api-policy").status_code == 404
    assert c.get("/api/v1/policies/m71api-policy/retention"
                 ).status_code == 404

    # probe suites: register (leaf) -> retention -> DELETE 200 -> 404
    suite = c.post("/api/v1/probe-suites", json={
        "suite_id": "m71api-suite",
        "description": "M71 API suite",
        "probes": [{"dataset_id": ds, "split": "validation",
                    "tokenizer_id": tok, "batch_size": 8,
                    "max_seq_len": 32, "seed": 4}]})
    assert suite.status_code == 201, suite.text
    view = c.get("/api/v1/probe-suites/m71api-suite/retention")
    assert view.status_code == 200
    assert view.json()["deletable"] is True
    assert view.json()["model_ids"] == []
    gone = c.delete("/api/v1/probe-suites/m71api-suite")
    assert gone.status_code == 200
    assert c.delete("/api/v1/probe-suites/m71api-suite").status_code == 404

    # integrity refusal at the API: tamper a fresh recipe's manifest
    # through the forge root bound to the app (the singleton's root)
    reg = c.post("/api/v1/workflows/recipes", json={
        "recipe_id": "m71api-tampered",
        "stages": [{"stage_id": "ev", "type": "evaluate",
                    "evaluation": {"config": {
                        "model_id": mid, "dataset_id": ds,
                        "tokenizer_id": tok, "split": "validation",
                        "batch_size": 8, "max_seq_len": 32, "seed": 5,
                        "checkpoint_id": ck[0]}}}]})
    assert reg.status_code == 201, reg.text
    import app.engine as engine_module
    forge = engine_module.get_forge()
    mpath = (forge.storage.root / "workflow-recipes" /
             "m71api-tampered" / "manifest.json")
    rec = json.loads(mpath.read_text())
    original = mpath.read_text()
    rec["stages"][0]["evaluation"]["config"]["seed"] = 99
    mpath.write_text(json.dumps(rec))
    assert c.get("/api/v1/workflows/recipes/m71api-tampered"
                 "/retention").json()["deletable"] is False
    refused = c.delete("/api/v1/workflows/recipes/m71api-tampered")
    assert refused.status_code == 409
    assert "integrity" in refused.json()["detail"]
    # restore integrity and remove the recipe: zero residue in the
    # shared session root (the tamper proved the refusal; the API
    # test must not leak definitions into later tests' listings)
    mpath.write_text(original)
    assert c.delete(
        "/api/v1/workflows/recipes/m71api-tampered").status_code == 200
    assert c.get("/api/v1/workflows/recipes"
                 ).json().count("m71api-tampered") == 0

    # OpenAPI: 105 paths (M72 adds /project/retention); three NEW GET-only
    # retention paths; the
    # three DELETEs are new OPERATIONS on the EXISTING GET-one paths;
    # the delete-operation set is exactly 14
    spec = c.get("/openapi.json").json()
    assert len(spec["paths"]) == 105
    for path in ("/api/v1/workflows/recipes/{recipe_id}",
                 "/api/v1/policies/{policy_id}",
                 "/api/v1/probe-suites/{suite_id}"):
        assert set(spec["paths"][path].keys()) == {"get", "delete"}, path
        assert set(spec["paths"][path + "/retention"].keys()) == {"get"}
    deletes = sorted(p for p, ops in spec["paths"].items()
                     if "delete" in ops)
    assert len(deletes) == 14
    for s in ("DefinitionDeletionResult", "DefinitionDeletionBlocked",
              "DefinitionDeletionBlocker", "DefinitionRetentionOverview"):
        assert s in spec["components"]["schemas"], s
    assert spec["paths"]["/api/v1/workflows/recipes/{recipe_id}"][
        "delete"]["responses"]["409"]["content"][
        "application/json"]["schema"]["$ref"] == \
        "#/components/schemas/DefinitionDeletionBlocked"
