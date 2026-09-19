"""Milestone 72 tests: the read-only PROJECT retention inventory.

The M72 invariants under test:

* ONE AGGREGATION — the project view introduces NO second scanner:
  every per-family number is the SUM of the EXISTING per-artifact
  retention views (M62 checkpoints, M65 datasets/tokenizers, M67
  models, M68 root-level records, M70 records, M71 definitions) and
  the totals come from the ONE M63 physical storage walk — proven
  against independent oracles (per-item view sums AND raw
  filesystem walks).
* TRUE TOTALS — ``total_files``/``total_size_bytes`` are the actual
  project storage (family sizes overlap by ownership — the model
  family INCLUDES its owned records — and never sum into them).
* EXACT RECLAIMABLE — ``reclaimable_files``/``bytes`` are the exact
  result of deleting every currently-deletable artifact: a deletable
  MODEL contributes its WHOLE directory (subsuming its records); a
  blocked model contributes only its own deletable records.
* ZERO MUTATION — the overview is read-only, deterministic and
  byte-identical over unchanged state; deleting ONE artifact moves
  the inventory by EXACTLY that artifact's numbers.
"""
from __future__ import annotations

import hashlib

import random
import zlib
from pathlib import Path

import pytest

from app.engine import ModelForge
from app.schemas import (
    ComparisonRequest,
    ComparisonState,
    EvalStateKind,
    EvaluationConfig,
    GatePolicy,
    GateRequest,
    ModelCreateRequest,
    ProbeSuiteCreateRequest,
    SampleGenerateRequest,
    SampleStrategy,

    StageType,
    SuiteProbe,
    SuiteRunRequest,
    TokenizerConfig,
    TransformerConfig,
    TrainingConfig,
    WorkflowEvaluationStage,
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


def _walk(directory: Path) -> tuple[int, int]:
    files = [p for p in directory.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def _sum_walks(walks) -> tuple[int, int]:
    total = (0, 0)
    for files, size in walks:
        total = (total[0] + files, total[1] + size)
    return total


def _train_cfg(model, ds, tok, seed, steps=8):
    return TrainingConfig(
        method="continued_pretraining", model_id=model, dataset_id=ds,
        tokenizer_id=tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=seed,
        steps=steps)


def _eval_cfg(model, ds, tok, seed, ckpt):
    return EvaluationConfig(
        model_id=model, dataset_id=ds, tokenizer_id=tok,
        split="validation", batch_size=8, max_seq_len=32, seed=seed,
        checkpoint_id=ckpt)


def _by_family(overview):
    return {f.family: f for f in overview.families}


# --------------------------------------------------------------------------- #
# Fixture: one rich project touching EVERY family
# --------------------------------------------------------------------------- #

class Env:
    pass


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env()
    e.root = tmp_path_factory.mktemp("m72-root")
    e.forge = ModelForge(root=e.root)
    forge = e.forge
    up = forge.upload_dataset([("m72.txt", _corpus(190, "m72"))],
                              name="m72-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m72-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)

    # model A: records + definitions that reference it (BLOCKED)
    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m72-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    forge.run_training(_train_cfg(e.model, e.ds, e.tok, 1))
    e.ck = sorted(c.checkpoint_id for c in
                  forge.list_checkpoints(e.model))
    e.direct_eval = forge.run_evaluation(
        _eval_cfg(e.model, e.ds, e.tok, 2, e.ck[0]))
    e.comp = forge.run_comparison(ComparisonRequest(
        model_id=e.model,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=e.ck[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=e.ck[1]),
        dataset_id=e.ds, split="validation", tokenizer_id=e.tok,
        batch_size=8, max_seq_len=32, seed=3, tolerance=1e-4))
    e.gate = forge.run_gate(GateRequest(
        model_id=e.model,
        policy=GatePolicy(
            name="m72-gate", model_id=e.model, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=4, baseline_type="checkpoint",
            baseline_checkpoint_id=e.ck[1], tolerance=1.0),
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=e.ck[0])))
    e.workflow = forge.run_workflow(_plan(e.model, e.ds, e.tok))
    # definitions referencing model A
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m72-recipe", stages=[WorkflowStage(
            stage_id="ev", type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=_eval_cfg(e.model, e.ds, e.tok, 6, e.ck[0])))]))
    e.recipe_run = forge.run_workflow_recipe("m72-recipe", e.model)
    forge.register_policy(PolicyCreateRequest(
        policy_id="m72-policy",
        policy=GatePolicy(
            name="m72-policy", model_id=e.model, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=8, baseline_type="checkpoint",
            baseline_checkpoint_id=e.ck[1], tolerance=1.0)))
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m72-suite",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))
    e.suite_run = forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m72-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=e.ck[0])))

    # model B: trained + sample + quality (BLOCKED by external refs)
    e.model_b = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m72-model-b", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=2)))[0].id
    forge.run_training(_train_cfg(e.model_b, e.ds, e.tok, 2))
    best = forge.select_best_checkpoint(e.model_b).checkpoint.checkpoint_id
    e.sample = forge.generate_sample(SampleGenerateRequest(
        model_id=e.model_b, checkpoint_id=best, tokenizer_id=e.tok,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    e.quality = forge.evaluate_sample(e.model_b, e.sample.sample_id)

    # model C: trained + ONE leaf record, NO external references —
    # the DELETABLE model with a deletable record (the overlap case)
    e.model_c = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m72-model-c", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=3)))[0].id
    forge.run_training(_train_cfg(e.model_c, e.ds, e.tok, 3))
    e.ck_c = sorted(c.checkpoint_id for c in
                    forge.list_checkpoints(e.model_c))
    e.leaf_eval_c = forge.run_evaluation(
        _eval_cfg(e.model_c, e.ds, e.tok, 5, e.ck_c[0]))
    return e


def _plan(model, ds, tok):
    from app.schemas import WorkflowPlan
    return WorkflowPlan(
        name="m72-workflow", model_id=model, description="M72 fixture",
        stages=[WorkflowStage(
            stage_id="ev", type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=EvaluationConfig(
                    model_id=model, dataset_id=ds, tokenizer_id=tok,
                    split="validation", batch_size=8, max_seq_len=32)))])


# --------------------------------------------------------------------------- #
# The inventory: aggregation, oracles, totals, zero mutation
# --------------------------------------------------------------------------- #

def test_m72_inventory_matches_oracles(env):
    forge, root = env.forge, env.root
    before = _inventory(root)
    ov = forge.project_retention_overview()
    fam = _by_family(ov)

    # canonical family order (15 families)
    assert [f.family for f in ov.families] == [
        "model", "dataset", "tokenizer", "workflow_recipe",
        "gate_policy", "probe_suite", "training_run", "checkpoint",
        "workflow", "evaluation", "comparison", "gate", "suite_run",
        "sample", "sample_quality"]
    assert all(f.deletion_supported for f in ov.families
               if f.family != "training_run")
    assert fam["training_run"].deletion_supported is False

    # ---- oracle 1: per-item view sums (the aggregation source) ------ #
    expected = {f.family: dict(count=0, files=0, size=0, deletable=0,
                               rfiles=0, rbytes=0)
                for f in ov.families}

    def acc(f, v):
        d = expected[f]
        d["count"] += 1
        d["files"] += len(v.files)
        d["size"] += v.size_bytes
        if v.deletable:
            d["deletable"] += 1
            d["rfiles"] += len(v.files)
            d["rbytes"] += v.size_bytes

    for d in forge.datasets.list():
        acc("dataset", forge.dataset_retention_overview(d.id))
    for t in forge.tokenizers.list():
        acc("tokenizer", forge.tokenizer_retention_overview(t.id))
    for r in forge.recipes.list():
        acc("workflow_recipe", forge.definition_retention_overview(
            "workflow_recipe", r.recipe_id))
    for pol in forge.policies.list_policies():
        acc("gate_policy", forge.definition_retention_overview(
            "gate_policy", pol.policy_id))
    for s in forge.policies.list_suites():
        acc("probe_suite", forge.definition_retention_overview(
            "probe_suite", s.suite_id))
    for m in forge.list_models():
        acc("model", forge.model_retention_overview(m.id))
        usage = forge.model_records_usage_overview(m.id)
        cats = {c.category: c.records for c in usage.categories}
        expected["training_run"]["count"] += len(
            cats.get("training_run", []))
        # checkpoints: the ONE M62 registry overview (aggregates +
        # per-entry sums — exactly what the engine reads)
        ck = forge.checkpoint_retention_overview(m.id)
        d = expected["checkpoint"]
        d["count"] += ck.total_checkpoints
        d["files"] += sum(x.files for x in ck.checkpoints)
        d["size"] += ck.total_checkpoint_bytes
        d["deletable"] += ck.deletable_checkpoints
        d["rbytes"] += ck.reclaimable_checkpoint_bytes
        d["rfiles"] += sum(x.files for x in ck.checkpoints if x.deletable)
        for category in ("workflow", "evaluation", "comparison", "gate"):
            for r in cats.get(category, []):
                acc(category, forge.model_record_retention_overview(
                    m.id, category, r.record_id))
        for sr in forge.list_suite_runs(m.id):
            acc("suite_run", forge.suite_run_retention_overview(
                m.id, sr.suite_run_id))
        for s in forge.list_samples(m.id):
            acc("sample", forge.sample_retention_overview(
                m.id, s.sample_id))
        for q in forge.list_sample_evaluations(m.id):
            acc("sample_quality",
                forge.sample_evaluation_retention_overview(
                    m.id, q.evaluation_id))

    for f in ov.families:
        d = expected[f.family]
        if f.family == "training_run":
            assert f.count == d["count"] and f.files == 0 \
                and f.size_bytes == 0 and f.deletable_count == 0 \
                and f.blocked_count == 0 and f.reclaimable_files == 0 \
                and f.reclaimable_bytes == 0
            continue
        assert f.count == d["count"], f.family
        assert f.files == d["files"], f.family
        assert f.size_bytes == d["size"], f.family
        assert f.deletable_count == d["deletable"], f.family
        assert f.blocked_count == d["count"] - d["deletable"], f.family
        assert f.reclaimable_files == d["rfiles"], f.family
        assert f.reclaimable_bytes == d["rbytes"], f.family

    # ---- oracle 2: raw filesystem walks (simple-path families) ------ #
    fs = {
        "model": _sum_walks(_walk(root / "models" / m.id)
                            for m in forge.list_models()),
        "dataset": _walk(root / "datasets" / env.ds),
        "tokenizer": _walk(root / "tokenizers" / env.tok),
        "workflow_recipe": _walk(root / "workflow-recipes" / "m72-recipe"),
        "gate_policy": _walk(root / "policies" / "m72-policy"),
        "probe_suite": _walk(root / "probe-suites" / "m72-suite"),
        "suite_run": _walk(root / "suite-runs" / env.suite_run.suite_run_id),
        "sample": _walk(root / "samples" / env.model_b
                        / f"sample-{env.sample.sample_id}"),
        "checkpoint": _sum_walks(
            _walk(root / "models" / m.id / "checkpoints" / c)
            for m in forge.list_models()
            for c in sorted(x.checkpoint_id for x in
                            forge.list_checkpoints(m.id))),
    }
    for family, (files, size) in fs.items():
        assert fam[family].files == files, family
        assert fam[family].size_bytes == size, family

    # ---- totals: the ONE M63 physical walk, counts sum families ----- #
    storage = forge.project_storage_overview()
    assert ov.total_files == storage.total_files
    assert ov.total_size_bytes == storage.total_bytes
    assert ov.total_count == sum(f.count for f in ov.families)
    assert ov.total_deletable == sum(f.deletable_count
                                     for f in ov.families)
    assert ov.total_blocked == sum(f.blocked_count for f in ov.families)
    # the family sizes OVERLAP (model includes records) — they must
    # NOT sum into the true total
    assert sum(f.size_bytes for f in ov.families) > ov.total_size_bytes

    # ---- EXACT reclaimable (the model/record overlap rule) ---------- #
    exp_rf = sum(f.reclaimable_files for f in ov.families
                 if f.family not in ("model", "checkpoint", "workflow",
                                     "evaluation", "comparison", "gate"))
    exp_rb = sum(f.reclaimable_bytes for f in ov.families
                 if f.family not in ("model", "checkpoint", "workflow",
                                     "evaluation", "comparison", "gate"))
    for m in forge.list_models():
        mv = forge.model_retention_overview(m.id)
        if mv.deletable:
            exp_rf += len(mv.files)
            exp_rb += mv.size_bytes
        else:
            usage = forge.model_records_usage_overview(m.id)
            cats = {c.category: c.records for c in usage.categories}
            ck = forge.checkpoint_retention_overview(m.id)
            exp_rb += ck.reclaimable_checkpoint_bytes
            exp_rf += sum(x.files for x in ck.checkpoints if x.deletable)
            for category in ("workflow", "evaluation", "comparison",
                             "gate"):
                for r in cats.get(category, []):
                    rv = forge.model_record_retention_overview(
                        m.id, category, r.record_id)
                    if rv.deletable:
                        exp_rf += len(rv.files)
                        exp_rb += rv.size_bytes
    assert ov.reclaimable_files == exp_rf
    assert ov.reclaimable_bytes == exp_rb
    # the fixture's overlap case: model C is deletable WITH a
    # deletable record — the per-family sums DOUBLE-COUNT it, the
    # project total does not
    assert forge.model_retention_overview(env.model_c).deletable is True
    assert forge.model_record_retention_overview(
        env.model_c, "evaluation",
        env.leaf_eval_c.eval_id).deletable is True
    assert ov.reclaimable_bytes < sum(f.reclaimable_bytes
                                      for f in ov.families)
    assert ov.reclaimable_bytes == exp_rb  # (already asserted; restated)

    # ---- zero mutation + determinism --------------------------------- #
    assert _inventory(root) == before
    ov2 = forge.project_retention_overview()
    assert ov2.model_dump(mode="json") == ov.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Live deltas: deletions move the inventory by EXACTLY their numbers
# --------------------------------------------------------------------------- #

def test_m72_deletions_move_inventory_exactly(tmp_path):
    forge = ModelForge(root=tmp_path)
    up = forge.upload_dataset([("m72d.txt", _corpus(60, "m72d"))],
                              name="m72d-ds")
    ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m72d-tok", vocab_size=320), dataset_id=ds)
    forge.tokenize_dataset(ds, tok.id)
    model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m72d-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    forge.run_training(_train_cfg(model, ds, tok.id, 1))
    ck = sorted(c.checkpoint_id for c in forge.list_checkpoints(model))
    leaf = forge.run_evaluation(_eval_cfg(model, ds, tok.id, 2, ck[0]))

    leaf_view = forge.model_record_retention_overview(
        model, "evaluation", leaf.eval_id)
    ov = forge.project_retention_overview()
    fam = _by_family(ov)
    storage_before = ov.total_size_bytes
    # the model is DELETABLE (no external refs) WITH a deletable
    # record — reclaimable == model size + checkpoints' deletable
    mv = forge.model_retention_overview(model)
    assert mv.deletable is True
    ev = fam["evaluation"]
    assert ev.count == 1 and ev.deletable_count == 1

    # deleting the leaf evaluation moves the inventory EXACTLY
    forge.delete_model_record(model, "evaluation", leaf.eval_id)
    ov2 = forge.project_retention_overview()
    fam2 = _by_family(ov2)
    assert fam2["evaluation"].count == 0
    assert fam2["evaluation"].reclaimable_bytes == 0
    assert ov2.total_size_bytes == storage_before - leaf_view.size_bytes
    assert ov2.total_count == ov.total_count - 1

    # deleting the whole model reclaims its ENTIRE (current)
    # directory — the inventory's reclaimable is EXACTLY the model's
    # post-eval-deletion size (the overlap rule: the deletable model
    # subsumes its remaining records; the deleted eval already left)
    assert ov2.reclaimable_bytes == mv.size_bytes - leaf_view.size_bytes
    forge.delete_model(model)
    ov3 = forge.project_retention_overview()
    fam3 = _by_family(ov3)
    assert fam3["model"].count == 0 and fam3["checkpoint"].count == 0
    assert fam3["model"].size_bytes == 0
    assert ov3.total_size_bytes == storage_before - mv.size_bytes
    assert ov3.total_count == ov2.total_count - (
        ov2.families[0].count + sum(
            _by_family(ov2)[f].count for f in
            ("training_run", "checkpoint", "workflow", "evaluation",
             "comparison", "gate", "suite_run", "sample",
             "sample_quality")))
    # the dataset/tokenizer survive (blocked, protected)
    assert fam3["dataset"].count == 1 and fam3["tokenizer"].count == 1
    assert ov3.reclaimable_bytes == 0  # nothing deletible remains


# --------------------------------------------------------------------------- #
# API + OpenAPI
# --------------------------------------------------------------------------- #

def test_m72_api_and_openapi(api_client):
    c = api_client
    import app.engine as engine_module

    r1 = c.get("/api/v1/project/retention")
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert set(body) == {"families", "total_count", "total_files",
                         "total_size_bytes", "total_deletable",
                         "total_blocked", "reclaimable_files",
                         "reclaimable_bytes"}
    assert [f["family"] for f in body["families"]] == [
        "model", "dataset", "tokenizer", "workflow_recipe",
        "gate_policy", "probe_suite", "training_run", "checkpoint",
        "workflow", "evaluation", "comparison", "gate", "suite_run",
        "sample", "sample_quality"]
    # the API view == the engine view on the SAME root
    forge = engine_module.get_forge()
    assert body == forge.project_retention_overview().model_dump(
        mode="json")
    # deterministic byte-identical repeat
    assert c.get("/api/v1/project/retention").content == r1.content

    # OpenAPI: 119 paths; the new route is GET-only; schemas present
    spec = c.get("/openapi.json").json()
    assert len(spec["paths"]) == 119
    assert set(spec["paths"]["/api/v1/project/retention"].keys()) == \
        {"get"}
    for s in ("ProjectFamilyRetention", "ProjectRetentionOverview"):
        assert s in spec["components"]["schemas"], s


# --------------------------------------------------------------------------- #
# Empty project
# --------------------------------------------------------------------------- #

def test_m72_empty_project(tmp_path):
    forge = ModelForge(root=tmp_path)
    ov = forge.project_retention_overview()
    assert all(f.count == 0 and f.files == 0 and f.size_bytes == 0
               and f.deletable_count == 0 and f.blocked_count == 0
               and f.reclaimable_files == 0 and f.reclaimable_bytes == 0
               for f in ov.families)
    assert ov.total_count == 0 and ov.total_deletable == 0 \
        and ov.total_blocked == 0 and ov.reclaimable_files == 0 \
        and ov.reclaimable_bytes == 0
    # the only file on a fresh root is the project manifest
    assert ov.total_files == 1
    assert ov.total_size_bytes == (tmp_path / "project.json").stat(
        ).st_size

