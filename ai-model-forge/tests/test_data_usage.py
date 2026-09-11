"""Milestone 64 tests: read-only DATASET & TOKENIZER usage overviews.

The M64 invariants under test:

* REUSE — every reference comes from the ONE existing cross-reference
  filters (``..._for_dataset`` / ``..._for_tokenizer``), the ONE
  tokenizer-training scan the dataset deletion guard refuses on, the
  authoritative registries and the ONE M2 tokenized layout; parity is
  verified per category against INDEPENDENT recomputation.
* READ-ONLY — zero mutation (byte-level sha256 inventories before and
  after), deterministic byte-identical repeats, live recompute.
* HONEST VISIBILITY — the usage view explains the EXISTING dataset
  deletion guard exactly (the tokenizers it refuses on are the
  ``tokenizer_training`` references), unknown ids -> the family 404,
  unreferenced artifacts -> 200 with empty categories.
"""
from __future__ import annotations

import hashlib
import random
import zlib
from pathlib import Path

import pytest

from app.dataset import _referencing_tokenizers
from app.engine import ModelForge
from app.schemas import (
    ComparisonRequest,
    ComparisonState,
    EvalStateKind,
    EvaluationConfig,
    ModelCreateRequest,
    ProbeSuiteCreateRequest,
    SampleGenerateRequest,
    SampleStrategy,
    SuiteProbe,
    SuiteRunRequest,
    TokenizerConfig,
    TransformerConfig,
    TrainingConfig,
    WorkflowEvaluationStage,
    WorkflowPlan,
    WorkflowStage,
    StageType,
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
    """rel-posix -> sha256 (the zero-mutation proof)."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


def _tokenized_walk(root: Path, dataset_id: str) -> list[str]:
    """Independent walk of the M2 tokenized layout: f"v{v}/{tok}" ids."""
    out = []
    droot = root / "datasets" / dataset_id
    for vdir in sorted(droot.iterdir()):
        if not vdir.is_dir() or not vdir.name.startswith("v"):
            continue
        troot = vdir / "tokenized"
        if not troot.exists():
            continue
        for tdir in sorted(troot.iterdir()):
            if tdir.is_dir():
                out.append(f"{vdir.name}/{tdir.name}")
    return sorted(out)


def _plan_refs(plan) -> set[tuple[str, str]]:
    """Independent scan of a workflow plan's stage configs (dataset,
    tokenizer) pairs — mirrors the documented taxonomy, not the engine."""
    refs = set()
    for stage in plan.stages:
        if stage.training is not None:
            refs.add((stage.training.dataset_id, stage.training.tokenizer_id))
        if stage.evaluation is not None:
            refs.add((stage.evaluation.config.dataset_id,
                      stage.evaluation.config.tokenizer_id))
        if stage.comparison is not None:
            refs.add((stage.comparison.dataset_id,
                      stage.comparison.tokenizer_id))
    return refs


# --------------------------------------------------------------------------- #
# Multi-reference fixture
# --------------------------------------------------------------------------- #

DS_A = "m64-alpha-ds"       # referenced by everything
DS_B = "m64-beta-ds"        # unreferenced


class Env:
    def __init__(self, root: Path):
        self.forge = ModelForge(root=root)
        self.root = root
        self.ds_a = None
        self.ds_b = None
        self.tok_a = None      # referenced by everything
        self.tok_b = None      # unreferenced (trained on ds_a, never used)
        self.model = None


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m64-root"))
    up = e.forge.upload_dataset([("a.txt", _corpus(190, "m64a"))],
                                name=DS_A)
    e.ds_a = up["dataset_id"]
    tok = e.forge.train_tokenizer(
        TokenizerConfig(name="m64-tok-a", vocab_size=320), dataset_id=e.ds_a)
    e.tok_a = tok.id
    e.forge.tokenize_dataset(e.ds_a, e.tok_a)
    tok2 = e.forge.train_tokenizer(
        TokenizerConfig(name="m64-tok-b", vocab_size=320), dataset_id=e.ds_a)
    e.tok_b = tok2.id
    upb = e.forge.upload_dataset([("b.txt", _corpus(60, "m64b"))],
                                 name=DS_B)
    e.ds_b = upb["dataset_id"]

    e.model = e.forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m64-model", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
        seed=1)))[0].id
    # (1) a direct training run -> training_run reference
    e.forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.model, dataset_id=e.ds_a,
        tokenizer_id=e.tok_a, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1, steps=8))
    ckpts = [c.checkpoint_id for c in e.forge.list_checkpoints(e.model)]
    # (2) a direct evaluation -> evaluation reference
    e.forge.run_evaluation(EvaluationConfig(
        model_id=e.model, dataset_id=e.ds_a, tokenizer_id=e.tok_a,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=ckpts[0]))
    # (3) a direct comparison -> comparison reference
    e.forge.run_comparison(ComparisonRequest(
        model_id=e.model,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ckpts[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ckpts[1]),
        dataset_id=e.ds_a, split="validation", tokenizer_id=e.tok_a,
        batch_size=8, max_seq_len=32, seed=3, tolerance=1e-4))
    # (4) a workflow (TRAIN + EVALUATE stages) -> workflow reference
    #     (+ its own training run and evaluation)
    e.forge.run_workflow(WorkflowPlan(
        name="m64-usage-workflow", model_id=e.model,
        description="M64 usage-overview reference fixture",
        stages=[
            WorkflowStage(stage_id="tr", type=StageType.TRAIN,
                          training=TrainingConfig(
                              method="continued_pretraining",
                              model_id=e.model, dataset_id=e.ds_a,
                              tokenizer_id=e.tok_a, learning_rate=3e-3,
                              batch_size=8, max_seq_len=32,
                              eval_every_steps=4, keep_best=False, seed=5,
                              steps=4)),
            WorkflowStage(stage_id="ev", type=StageType.EVALUATE,
                          evaluation=WorkflowEvaluationStage(
                              config=EvaluationConfig(
                                  model_id=e.model, dataset_id=e.ds_a,
                                  tokenizer_id=e.tok_a, split="validation",
                                  batch_size=8, max_seq_len=32),
                              checkpoint_from_best=True)),
        ]))
    # (5) a suite run -> suite_run reference (+ its probe evaluation)
    e.forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m64-suite",
        description="M64 usage fixture",
        probes=[SuiteProbe(dataset_id=e.ds_a, split="validation",
                           tokenizer_id=e.tok_a, batch_size=8,
                           max_seq_len=32, seed=42)]))
    e.forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m64-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=ckpts[0])))
    # (6) a sample + sample-quality measurement -> sample references
    best = e.forge.select_best_checkpoint(e.model).checkpoint.checkpoint_id
    sample = e.forge.generate_sample(SampleGenerateRequest(
        model_id=e.model, checkpoint_id=best, tokenizer_id=e.tok_a,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    e.forge.evaluate_sample(e.model, sample.sample_id)
    return e


# --------------------------------------------------------------------------- #
# Dataset usage overview (engine)
# --------------------------------------------------------------------------- #

def test_m64_dataset_usage_overview(env):
    forge, root = env.forge, env.root
    before = _inventory(root)
    o1 = forge.dataset_usage_overview(env.ds_a)
    o2 = forge.dataset_usage_overview(env.ds_a)
    after = _inventory(root)

    # identity + determinism + zero writes
    meta = forge.datasets.load_meta(env.ds_a)
    assert o1.dataset_id == env.ds_a
    assert o1.name == meta.name == DS_A
    assert o1.created_at == meta.created_at
    assert o1.version_count == len(meta.versions) == 1
    assert o1.latest_version == meta.latest_version
    assert o1.model_dump(mode="json") == o2.model_dump(mode="json")
    assert before == after

    # canonical category order
    cats = {c.category: c.references for c in o1.categories}
    assert [c.category for c in o1.categories] == \
        list(ModelForge.DATASET_USAGE_CATEGORIES)

    # ---- per-category parity vs INDEPENDENT recomputation ----
    m = forge.get_model(env.model)
    assert cats["training_run"] == sorted(
        f"{env.model}/{p.run_id}" for p in m.training_provenance
        if p.dataset_id == env.ds_a)
    assert len(cats["training_run"]) == 2          # direct run + workflow run
    assert cats["workflow"] == sorted(
        f"{env.model}/{w.workflow_id}" for w in forge.list_workflows(env.model)
        if any(ds == env.ds_a for ds, _t in _plan_refs(w.plan)))
    assert len(cats["workflow"]) == 1
    assert cats["evaluation"] == sorted(
        f"{env.model}/{e.eval_id}"
        for e in forge.list_evaluations_for_dataset(env.model, env.ds_a))
    assert len(cats["evaluation"]) == 5
    # direct eval + the comparison's TWO side evaluations + the workflow's
    # EVALUATE stage + the suite probe's evaluation
    assert cats["comparison"] == sorted(
        f"{env.model}/{c.comparison_id}"
        for c in forge.list_comparisons_for_dataset(env.model, env.ds_a))
    assert len(cats["comparison"]) == 1
    assert cats["suite_run"] == sorted(
        f"{env.model}/{r.suite_run_id}"
        for r in forge.list_suite_runs(env.model)
        if any(res.probe.dataset_id == env.ds_a for res in r.results))
    assert len(cats["suite_run"]) == 1
    # the ONE guard scan — the usage view explains the deletion guard
    assert cats["tokenizer_training"] == sorted(
        _referencing_tokenizers(forge.storage, env.ds_a))
    assert sorted(cats["tokenizer_training"]) == sorted([env.tok_a, env.tok_b])
    assert cats["tokenized_version"] == _tokenized_walk(root, env.ds_a)
    assert cats["tokenized_version"] == [f"v1/{env.tok_a}"]

    # aggregates
    assert o1.total_references == sum(len(r) for r in cats.values()) == 13
    assert o1.referenced is True

    # THE guard invariant: deleting the dataset is refused with exactly
    # the tokenizer_training references (and nothing is written)
    with pytest.raises(ValueError) as exc:
        forge.delete_dataset(env.ds_a)
    assert env.tok_a in str(exc.value) and env.tok_b in str(exc.value)
    assert _inventory(root) == before               # the refusal wrote nothing

    # unknown dataset -> FileNotFoundError (the family's 404)
    with pytest.raises(FileNotFoundError):
        forge.dataset_usage_overview("no-such-m64-dataset")

    # unreferenced dataset: 200-equivalent, empty categories, zero totals
    ob = forge.dataset_usage_overview(env.ds_b)
    assert ob.referenced is False and ob.total_references == 0
    assert ob.model_dump(mode="json") == \
        forge.dataset_usage_overview(env.ds_b).model_dump(mode="json")
    for c in ob.categories:
        assert c.references == []


# --------------------------------------------------------------------------- #
# Tokenizer usage overview (engine)
# --------------------------------------------------------------------------- #

def test_m64_tokenizer_usage_overview(env):
    forge, root = env.forge, env.root
    before = _inventory(root)
    o1 = forge.tokenizer_usage_overview(env.tok_a)
    after = _inventory(root)

    # identity (trained_on_dataset_id is provenance, not a reference)
    rec = forge.tokenizers.load(env.tok_a)
    assert o1.tokenizer_id == env.tok_a
    assert o1.name == rec.name == "m64-tok-a"
    assert o1.created_at == rec.created_at
    assert o1.requested_vocab_size == rec.requested_vocab_size
    assert o1.actual_vocab_size == rec.actual_vocab_size
    assert o1.trained_on_dataset_id == env.ds_a
    assert before == after

    cats = {c.category: c.references for c in o1.categories}
    assert [c.category for c in o1.categories] == \
        list(ModelForge.TOKENIZER_USAGE_CATEGORIES)

    # ---- per-category parity vs INDEPENDENT recomputation ----
    m = forge.get_model(env.model)
    assert cats["training_run"] == sorted(
        f"{env.model}/{p.run_id}" for p in m.training_provenance
        if p.tokenizer_id == env.tok_a)
    assert cats["workflow"] == sorted(
        f"{env.model}/{w.workflow_id}" for w in forge.list_workflows(env.model)
        if any(t == env.tok_a for _d, t in _plan_refs(w.plan)))
    assert cats["evaluation"] == sorted(
        f"{env.model}/{e.eval_id}" for e in
        forge.list_evaluations_for_tokenizer(env.model, env.tok_a))
    assert cats["comparison"] == sorted(
        f"{env.model}/{c.comparison_id}" for c in
        forge.list_comparisons_for_tokenizer(env.model, env.tok_a))
    assert cats["suite_run"] == sorted(
        f"{env.model}/{r.suite_run_id}"
        for r in forge.list_suite_runs(env.model)
        if any(res.probe.tokenizer_id == env.tok_a for res in r.results))
    assert cats["sample"] == sorted(
        f"{env.model}/{s.sample_id}" for s in
        forge.list_samples_for_tokenizer(env.model, env.tok_a))
    assert cats["sample_quality"] == sorted(
        f"{env.model}/{sq.evaluation_id}" for sq in
        forge.list_sample_evaluations_for_tokenizer(env.model, env.tok_a))
    # the reverse M2 layout walk: which dataset versions it tokenized
    assert cats["tokenized_dataset"] == [f"{env.ds_a}/v1"]

    assert o1.total_references == sum(len(r) for r in cats.values()) == 13
    assert o1.referenced is True

    # live recompute (no cache): a NEW evaluation with this tokenizer
    ck = forge.get_model(env.model).latest_checkpoint
    forge.run_evaluation(EvaluationConfig(
        model_id=env.model, dataset_id=env.ds_a, tokenizer_id=env.tok_a,
        split="validation", batch_size=8, max_seq_len=32, seed=77,
        checkpoint_id=ck))
    o2 = forge.tokenizer_usage_overview(env.tok_a)
    cats2 = {c.category: c.references for c in o2.categories}
    assert len(cats2["evaluation"]) == len(cats["evaluation"]) + 1
    assert o2.total_references == o1.total_references + 1
    # ... and the dataset view grew the same reference (project-coherent)
    od = forge.dataset_usage_overview(env.ds_a)
    od_cats = {c.category: c.references for c in od.categories}
    assert od_cats["evaluation"] == cats2["evaluation"]
    assert od.total_references == 14

    # unknown tokenizer -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        forge.tokenizer_usage_overview("no-such-m64-tokenizer")

    # unreferenced tokenizer (trained on ds_a but never used)
    ob = forge.tokenizer_usage_overview(env.tok_b)
    assert ob.referenced is False and ob.total_references == 0
    assert ob.trained_on_dataset_id == env.ds_a
    for c in ob.categories:
        assert c.references == []


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m64_api_usage_overviews(api_client):
    import json as _json

    # fixture through the public routes only
    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m64api.txt", _corpus(170, "m64api"), "text/plain"))],
        data={"name": "m64api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m64api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m64api-model", "vocab_size": 640,
                   "context_length": 64, "hidden_size": 64, "n_layers": 2,
                   "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128}})
    assert m.status_code == 201, m.text
    mid = m.json()["model"]["id"]
    r = api_client.post("/api/v1/training/run", json={
        "method": "continued_pretraining", "model_id": mid, "dataset_id": ds,
        "tokenizer_id": tok, "learning_rate": 3e-3, "batch_size": 8,
        "max_seq_len": 32, "eval_every_steps": 4, "keep_best": False,
        "seed": 1, "steps": 8})
    assert r.status_code == 200, r.text
    ck = api_client.get(f"/api/v1/models/{mid}/checkpoints").json()[0][
        "checkpoint_id"]
    r = api_client.post("/api/v1/evaluations/run", json={
        "model_id": mid, "dataset_id": ds, "tokenizer_id": tok,
        "split": "validation", "batch_size": 8, "max_seq_len": 32,
        "seed": 2, "checkpoint_id": ck})
    assert r.status_code == 200, r.text

    storage_root = Path(api_client.get("/api/v1/project").json()["storage_root"])

    # dataset usage over HTTP
    r = api_client.get(f"/api/v1/datasets/{ds}/usage")
    assert r.status_code == 200, r.text
    body_ds = r.content
    ov = r.json()
    assert set(ov) == {"dataset_id", "name", "created_at", "version_count",
                       "latest_version", "referenced", "total_references",
                       "categories"}
    assert ov["dataset_id"] == ds and ov["referenced"] is True
    cats = {c["category"]: c["references"] for c in ov["categories"]}
    assert set(cats) == set(ModelForge.DATASET_USAGE_CATEGORIES)
    for c in ov["categories"]:
        assert set(c) == {"category", "references"}
    # parity with the ONE public cross-reference route
    evals = api_client.get(
        f"/api/v1/models/{mid}/evaluations/by-dataset/{ds}").json()
    assert cats["evaluation"] == sorted(
        f"{mid}/{e['eval_id']}" for e in evals)
    assert all(r.startswith(f"{mid}/") for r in cats["training_run"])
    assert len(cats["training_run"]) == 1
    assert cats["tokenizer_training"] == [tok]
    assert cats["tokenized_version"] == [f"v1/{tok}"]
    assert ov["total_references"] == sum(len(v) for v in cats.values())

    # tokenizer usage over HTTP
    r = api_client.get(f"/api/v1/tokenizers/{tok}/usage")
    assert r.status_code == 200, r.text
    body_tok = r.content
    tv = r.json()
    assert set(tv) == {"tokenizer_id", "name", "created_at",
                       "requested_vocab_size", "actual_vocab_size",
                       "trained_on_dataset_id", "referenced",
                       "total_references", "categories"}
    assert tv["tokenizer_id"] == tok and tv["referenced"] is True
    assert tv["trained_on_dataset_id"] == ds
    tcats = {c["category"]: c["references"] for c in tv["categories"]}
    assert set(tcats) == set(ModelForge.TOKENIZER_USAGE_CATEGORIES)
    evals_t = api_client.get(
        f"/api/v1/models/{mid}/evaluations/by-tokenizer/{tok}").json()
    assert tcats["evaluation"] == sorted(
        f"{mid}/{e['eval_id']}" for e in evals_t)
    assert tcats["tokenized_dataset"] == [f"{ds}/v1"]
    assert tcats["training_run"] == cats["training_run"]
    assert tv["total_references"] == sum(len(v) for v in tcats.values())

    # 404s; unreferenced artifacts -> 200 with empty categories
    assert api_client.get("/api/v1/datasets/no-m64/usage").status_code == 404
    assert api_client.get("/api/v1/tokenizers/no-m64/usage").status_code == 404
    up2 = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m64free.txt", _corpus(30, "m64free"),
                          "text/plain"))],
        data={"name": "m64-free-ds"})
    assert up2.status_code == 201, up2.text
    r = api_client.get(f"/api/v1/datasets/{up2.json()['dataset_id']}/usage")
    assert r.status_code == 200, r.text
    free = r.json()
    assert free["referenced"] is False and free["total_references"] == 0
    assert all(c["references"] == [] for c in free["categories"])

    # determinism over HTTP + zero mutation (every call after this point
    # is a pure read: the usage routes write nothing)
    before = _inventory(storage_root)
    assert api_client.get(f"/api/v1/datasets/{ds}/usage").content == body_ds
    assert api_client.get(f"/api/v1/tokenizers/{tok}/usage").content == body_tok
    assert api_client.get(f"/api/v1/datasets/{ds}/usage").status_code == 200
    assert _inventory(storage_root) == before

    # OpenAPI: exactly two new paths (87 -> 89), only GET, schemas exposed
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 89
    for path in ("/api/v1/datasets/{dataset_id}/usage",
                 "/api/v1/tokenizers/{tokenizer_id}/usage"):
        assert set(spec["paths"][path].keys()) == {"get"}
    for s in ("DatasetUsageOverview", "TokenizerUsageOverview",
              "ArtifactUsageCategory"):
        assert s in spec["components"]["schemas"]
