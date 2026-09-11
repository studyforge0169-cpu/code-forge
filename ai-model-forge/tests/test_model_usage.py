"""Milestone 66 tests: read-only MODEL USAGE overview (engine + HTTP).

The M66 invariants under test:

* REUSE — every category comes from the ONE authoritative listings
  (the model manifest's provenance, the family listings, the
  model-scoped root-family listings) and the recipe scan follows the
  M64 plan-scan pattern; parity is verified per category against
  INDEPENDENT manifest parsing (the oracle never calls the app's
  analysis).
* INTERNAL vs EXTERNAL — the six model-scoped families are ownership
  (inside models/<id>/); suite runs, samples, sample-quality
  measurements and workflow recipes persist the model id OUTSIDE the
  model directory (the future-guard surface), counted in
  ``external_references``.
* READ-ONLY — zero mutation (byte-level sha256 inventories),
  deterministic byte-identical repeats, live recompute.
* HONEST SCOPE — unknown model -> the family's 404; a malformed
  manifest is registry-invisible (404); a fresh model -> all-empty
  categories; references of OTHER models never leak in.
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
    SuiteProbe,
    SuiteRunRequest,
    TokenizerConfig,
    TransformerConfig,
    TrainingConfig,
    WorkflowRecipeCreateRequest,
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


# --------------------------------------------------------------------------- #
# INDEPENDENT oracle: raw manifest parsing (never the app's analysis)
# --------------------------------------------------------------------------- #

INTERNAL_FAMILIES = {  # category -> (directory under models/<id>/, id field)
    "checkpoint": ("checkpoints", "checkpoint_id"),
    "workflow": ("workflows", "workflow_id"),
    "evaluation": ("evaluations", "eval_id"),
    "comparison": ("comparisons", "comparison_id"),
    "gate": ("gates", "decision_id"),
}


def _oracle(root: Path, model_id: str) -> dict[str, list[str]]:
    """category -> sorted reference ids, derived from raw manifests."""
    out: dict[str, list[str]] = {c: [] for c in ModelForge.MODEL_USAGE_CATEGORIES}
    mdir = root / "models" / model_id
    man = json.loads((mdir / "manifest.json").read_text())
    out["training_run"] = sorted(
        p["run_id"] for p in man.get("training_provenance", []))
    for cat, (fam, idfield) in INTERNAL_FAMILIES.items():
        froot = mdir / fam
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if not d.is_dir():
                    continue
                try:
                    rec = json.loads((d / "manifest.json").read_text())
                except Exception:
                    continue
                if rec.get("model_id", model_id) == model_id:
                    out[cat].append(rec[idfield])
    # root-level external families
    sroot = root / "suite-runs"
    if sroot.exists():
        for d in sorted(sroot.iterdir()):
            if not d.is_dir():
                continue
            try:
                rec = json.loads((d / "manifest.json").read_text())
            except Exception:
                continue
            if rec.get("model_id") == model_id:
                out["suite_run"].append(rec["suite_run_id"])
    for cat, fam, idfield in (("sample", "samples", "sample_id"),
                              ("sample_quality", "sample-evaluations",
                               "evaluation_id")):
        froot = root / fam / model_id
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if not d.is_dir():
                    continue
                try:
                    rec = json.loads((d / "manifest.json").read_text())
                except Exception:
                    continue
                if rec.get("model_id", model_id) == model_id:
                    out[cat].append(rec[idfield])
    # recipes: scan stage configs for the model id (train/evaluate/gate)
    rroot = root / "workflow-recipes"
    if rroot.exists():
        for d in sorted(rroot.iterdir()):
            if not d.is_dir():
                continue
            try:
                rec = json.loads((d / "manifest.json").read_text())
            except Exception:
                continue
            refs = set()

            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "model_id":
                            refs.add(v)
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)

            walk(rec.get("stages", []))
            if model_id in refs:
                out["workflow_recipe"].append(rec["recipe_id"])
    for c in out:
        out[c] = sorted(out[c])
    return out


# --------------------------------------------------------------------------- #
# Multi-reference fixture
# --------------------------------------------------------------------------- #

class Env:
    def __init__(self, root: Path):
        self.forge = ModelForge(root=root)
        self.root = root
        self.ds = None
        self.tok = None
        self.model = None       # rich: every category
        self.other = None       # a second model with its own training
        self.fresh = None       # never trained


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m66-root"))
    forge = e.forge
    up = forge.upload_dataset([("m66.txt", _corpus(190, "m66"))],
                              name="m66-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m66-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)

    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m66-rich-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    # (1) direct training run -> training_run + checkpoints
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.model, dataset_id=e.ds,
        tokenizer_id=e.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1, steps=8))
    ckpts = [c.checkpoint_id for c in forge.list_checkpoints(e.model)]
    # (2) direct evaluation
    forge.run_evaluation(EvaluationConfig(
        model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=ckpts[0]))
    # (3) comparison (adds its two side evaluations)
    forge.run_comparison(ComparisonRequest(
        model_id=e.model,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ckpts[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ckpts[1]),
        dataset_id=e.ds, split="validation", tokenizer_id=e.tok,
        batch_size=8, max_seq_len=32, seed=3, tolerance=1e-4))
    # (4) gate decision (independent persisted record)
    forge.run_gate(GateRequest(
        model_id=e.model,
        policy=GatePolicy(
            name="m66-gate", model_id=e.model, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=4, baseline_type="checkpoint",
            baseline_checkpoint_id=ckpts[1], tolerance=1.0),
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=ckpts[0])))
    # (5) a workflow (TRAIN + EVALUATE) -> its own training run + eval
    from app.schemas import WorkflowEvaluationStage, WorkflowPlan
    forge.run_workflow(WorkflowPlan(
        name="m66-usage-workflow", model_id=e.model,
        description="M66 usage fixture",
        stages=[
            WorkflowStage(stage_id="tr", type=StageType.TRAIN,
                          training=TrainingConfig(
                              method="continued_pretraining",
                              model_id=e.model, dataset_id=e.ds,
                              tokenizer_id=e.tok, learning_rate=3e-3,
                              batch_size=8, max_seq_len=32,
                              eval_every_steps=4, keep_best=False, seed=5,
                              steps=4)),
            WorkflowStage(stage_id="ev", type=StageType.EVALUATE,
                          evaluation=WorkflowEvaluationStage(
                              config=EvaluationConfig(
                                  model_id=e.model, dataset_id=e.ds,
                                  tokenizer_id=e.tok, split="validation",
                                  batch_size=8, max_seq_len=32),
                              checkpoint_from_best=True)),
        ]))
    # (6) a suite run (root-level suite-runs/) with its probe evaluation
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m66-suite", description="M66 usage fixture",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))
    forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m66-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=ckpts[0])))
    # (7) a sample + sample-quality measurement (root-level families)
    best = forge.select_best_checkpoint(e.model).checkpoint.checkpoint_id
    sample = forge.generate_sample(SampleGenerateRequest(
        model_id=e.model, checkpoint_id=best, tokenizer_id=e.tok,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    forge.evaluate_sample(e.model, sample.sample_id)
    # (8) a workflow recipe naming this model (root-level definitions)
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m66-model-recipe", description="M66 usage fixture",
        stages=[WorkflowStage(
            stage_id="tr", type=StageType.TRAIN,
            training=TrainingConfig(
                method="continued_pretraining", model_id=e.model,
                dataset_id=e.ds, tokenizer_id=e.tok, learning_rate=3e-3,
                batch_size=8, max_seq_len=32, eval_every_steps=4,
                keep_best=False, seed=9, steps=4))]))

    # a second model with its OWN training (isolation) and a fresh one
    e.other = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m66-other-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=2)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.other, dataset_id=e.ds,
        tokenizer_id=e.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=7, steps=4))
    e.fresh = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m66-fresh-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=3)))[0].id
    return e


# --------------------------------------------------------------------------- #
# Engine: category coverage, oracle parity, determinism, zero writes
# --------------------------------------------------------------------------- #

def test_m66_model_usage_overview(env):
    forge, root = env.forge, env.root
    before = _inventory(root)
    o1 = forge.model_usage_overview(env.model)
    o2 = forge.model_usage_overview(env.model)
    after = _inventory(root)

    # determinism + zero mutation
    assert o1.model_dump(mode="json") == o2.model_dump(mode="json")
    assert before == after

    # identity
    rec = forge.get_model(env.model)
    assert o1.model_id == env.model
    assert o1.name == rec.name == "m66-rich-model"
    assert o1.created_at == rec.created_at
    assert o1.architecture == rec.architecture.value
    assert o1.parameter_count == rec.parameter_count

    # canonical category order + the oracle (independent manifest parse)
    assert [c.category for c in o1.categories] == \
        list(ModelForge.MODEL_USAGE_CATEGORIES)
    oracle = _oracle(root, env.model)
    cats = {c.category: c.references for c in o1.categories}
    for cat, refs in oracle.items():
        assert cats[cat] == refs, cat
    # no duplicates anywhere
    for cat, refs in cats.items():
        assert len(refs) == len(set(refs))

    # per-category parity with the ONE authoritative listings
    assert cats["training_run"] == sorted(
        p.run_id for p in rec.training_provenance)
    assert cats["checkpoint"] == sorted(
        c.checkpoint_id for c in forge.list_checkpoints(env.model))
    assert cats["workflow"] == sorted(
        w.workflow_id for w in forge.list_workflows(env.model))
    assert cats["evaluation"] == sorted(
        e.eval_id for e in forge.list_evaluations(env.model))
    assert cats["comparison"] == sorted(
        c.comparison_id for c in forge.list_comparisons(env.model))
    assert cats["gate"] == sorted(
        g.decision_id for g in forge.list_gate_decisions(env.model))
    assert cats["suite_run"] == sorted(
        r.suite_run_id for r in forge.list_suite_runs(env.model))
    assert cats["sample"] == sorted(
        s.sample_id for s in forge.list_samples(env.model))
    assert cats["sample_quality"] == sorted(
        sq.evaluation_id
        for sq in forge.list_sample_evaluations(env.model))
    assert cats["workflow_recipe"] == ["m66-model-recipe"]

    # full coverage of the fixture: every category non-empty
    assert all(cats[c] for c in ModelForge.MODEL_USAGE_CATEGORIES)

    # internal/external split (the future-guard surface)
    internal = sum(len(cats[c]) for c in
                   ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES)
    external = sum(len(cats[c]) for c in ModelForge.MODEL_USAGE_CATEGORIES
                   if c not in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES)
    assert o1.internal_references == internal
    # suite_run(1) + sample(1) + sample_quality(1) + workflow_recipe(1)
    assert o1.external_references == external == 4
    assert o1.total_references == internal + external
    assert o1.referenced is True and o1.externally_referenced is True
    # the internal families live INSIDE the model directory; the
    # external ones are root-level (spot-check the physical layout)
    assert (root / "models" / env.model / "checkpoints").exists()
    assert (root / "suite-runs").exists()
    assert (root / "samples" / env.model).exists()
    assert (root / "sample-evaluations" / env.model).exists()
    assert (root / "workflow-recipes" / "m66-model-recipe").exists()


def test_m66_scope_isolation_and_empty(env):
    forge, root = env.forge, env.root

    # a second model: only its OWN references, never the rich model's
    oo = forge.model_usage_overview(env.other)
    ocats = {c.category: c.references for c in oo.categories}
    assert ocats["training_run"] and len(ocats["training_run"]) == 1
    assert len(ocats["checkpoint"]) == 1
    assert ocats["workflow"] == [] and ocats["evaluation"] == []
    assert ocats["comparison"] == [] and ocats["gate"] == []
    assert ocats["suite_run"] == [] and ocats["sample"] == []
    assert ocats["sample_quality"] == []
    assert ocats["workflow_recipe"] == []   # the recipe names the OTHER model
    assert oo.external_references == 0 and oo.externally_referenced is False
    assert oo.internal_references == 2 and oo.referenced is True
    # the rich model's references never leak
    rich = forge.model_usage_overview(env.model)
    rcats = {c.category: c.references for c in rich.categories}
    assert not (set(ocats["training_run"]) & set(rcats["training_run"]))
    assert not (set(ocats["checkpoint"]) & set(rcats["checkpoint"]))

    # a fresh model: ALL categories empty (the collection convention)
    fo = forge.model_usage_overview(env.fresh)
    fcats = {c.category: c.references for c in fo.categories}
    assert all(v == [] for v in fcats.values())
    assert fo.referenced is False and fo.externally_referenced is False
    assert fo.total_references == 0 and fo.internal_references == 0 \
        and fo.external_references == 0

    # unknown model -> the family's 404
    with pytest.raises(FileNotFoundError):
        forge.model_usage_overview("no-such-m66-model")

    # malformed manifest -> registry-invisible (the M61/M65 convention)
    mpath = root / "models" / env.fresh / "manifest.json"
    original = mpath.read_bytes()
    mpath.write_bytes(b"not json at all")
    try:
        with pytest.raises(FileNotFoundError):
            forge.model_usage_overview(env.fresh)
    finally:
        mpath.write_bytes(original)


def test_m66_live_recompute(env):
    forge = env.forge
    o1 = forge.model_usage_overview(env.model)
    # a NEW external reference (a suite run against the fresh model's
    # checkpoint? fresh has none) — use the other model: run a suite
    other_before = forge.model_usage_overview(env.other)
    assert other_before.external_references == 0
    assert other_before.internal_references == 2   # one run + one checkpoint
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m66-suite-2", description="M66 recompute fixture",
        probes=[SuiteProbe(dataset_id=env.ds, split="validation",
                           tokenizer_id=env.tok, batch_size=8,
                           max_seq_len=32, seed=43)]))
    ck = forge.list_checkpoints(env.other)[0].checkpoint_id
    forge.run_suite(SuiteRunRequest(
        model_id=env.other, suite_id="m66-suite-2",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=ck)))
    o2 = forge.model_usage_overview(env.other)
    cats2 = {c.category: c.references for c in o2.categories}
    assert len(cats2["suite_run"]) == 1
    assert o2.external_references == 1 and o2.externally_referenced is True
    # the suite run adds TWO references: itself (external) + the M4
    # evaluation its probe executes (internal)
    assert o2.total_references == other_before.total_references + 2
    assert o2.internal_references == other_before.internal_references + 1
    cats2_full = {c.category: len(c.references) for c in o2.categories}
    assert cats2_full["evaluation"] == 1
    # and the rich model is unchanged (no cache, no cross-talk)
    assert forge.model_usage_overview(env.model).model_dump(
        mode="json") == o1.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m66_api_model_usage(api_client):
    import json as _json

    # fixture through the public routes only
    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m66api.txt", _corpus(170, "m66api"), "text/plain"))],
        data={"name": "m66api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m66api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m66api-model", "vocab_size": 640,
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

    r = api_client.get(f"/api/v1/models/{mid}/usage")
    assert r.status_code == 200, r.text
    body = r.content
    ov = r.json()
    assert set(ov) == {"model_id", "name", "created_at", "architecture",
                       "parameter_count", "referenced",
                       "externally_referenced", "total_references",
                       "internal_references", "external_references",
                       "categories"}
    assert ov["model_id"] == mid and ov["referenced"] is True
    assert ov["externally_referenced"] is False   # nothing root-level yet
    for c in ov["categories"]:
        assert set(c) == {"category", "references"}

    cats = {c["category"]: c["references"] for c in ov["categories"]}
    # parity with the public listing routes (the ONE listings)
    assert cats["checkpoint"] == sorted(
        c["checkpoint_id"] for c in
        api_client.get(f"/api/v1/models/{mid}/checkpoints").json())
    assert cats["training_run"] == sorted(
        p["run_id"] for p in
        api_client.get(f"/api/v1/models/{mid}").json()["training_provenance"])
    assert cats["evaluation"] == sorted(
        e["eval_id"] for e in
        api_client.get(f"/api/v1/models/{mid}/evaluations").json())
    assert cats["workflow"] == sorted(
        w["workflow_id"] for w in
        api_client.get(f"/api/v1/models/{mid}/workflows").json())
    assert cats["comparison"] == []
    assert cats["gate"] == []
    assert cats["suite_run"] == sorted(
        s["suite_run_id"] for s in
        api_client.get(f"/api/v1/models/{mid}/suite-runs").json())
    assert cats["sample"] == sorted(
        s["sample_id"] for s in
        api_client.get(f"/api/v1/models/{mid}/samples").json())
    assert cats["sample_quality"] == sorted(
        s["evaluation_id"] for s in
        api_client.get(f"/api/v1/models/{mid}/sample-quality").json())
    assert cats["workflow_recipe"] == sorted(
        r["recipe_id"] for r in
        api_client.get("/api/v1/workflows/recipes").json()
        if mid in _json.dumps(r["stages"]))   # independent recipe scan

    # independent oracle over the session root
    oracle = _oracle(storage_root, mid)
    assert {c["category"]: c["references"] for c in ov["categories"]} == oracle
    assert ov["internal_references"] == sum(
        len(oracle[c]) for c in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES)
    assert ov["total_references"] == sum(len(v) for v in oracle.values())

    # unknown -> 404; malformed -> 404 (registry-invisible)
    assert api_client.get("/api/v1/models/no-m66/usage").status_code == 404
    mpath = storage_root / "models" / mid / "manifest.json"
    original = mpath.read_bytes()
    mpath.write_bytes(b"not json")
    assert api_client.get(f"/api/v1/models/{mid}/usage").status_code == 404
    mpath.write_bytes(original)

    # determinism + zero mutation
    before = _inventory(storage_root)
    assert api_client.get(f"/api/v1/models/{mid}/usage").content == body
    assert api_client.get(f"/api/v1/models/{mid}/usage").status_code == 200
    assert _inventory(storage_root) == before

    # OpenAPI: exactly one new path (91 -> 92)
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 92
    NEW = "/api/v1/models/{model_id}/usage"
    assert set(spec["paths"][NEW].keys()) == {"get"}
    for s in ("ModelUsageOverview", "ModelUsageCategory"):
        assert s in spec["components"]["schemas"], s
    # M66 adds ZERO delete operations: the new usage path is GET-only,
    # and the spec's total delete count is unchanged (the pre-existing
    # model DELETE route — the M67 subject — stays exactly as it was)
    assert "delete" not in spec["paths"][NEW]
    deletes = [p for p, ops in spec["paths"].items() if "delete" in ops]
    assert sorted(deletes) == [
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/models/{model_id}",
        "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}",
        "/api/v1/tokenizers/{tokenizer_id}",
    ]
