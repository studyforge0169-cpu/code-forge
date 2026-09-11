"""Milestone 65 tests: EXPLICIT VERIFIED dataset & tokenizer retention.

The M65 invariants under test:

* THE headline invariant (both directions): the deletion guard refuses
  on EXACTLY what the M64 usage overview reports — every blocker
  category appears in the overview with the same reference ids, and
  every non-empty overview category is a blocker, in the SAME
  canonical order. Nothing protected that is not shown; nothing shown
  that is not protected.
* Refusals write nothing (byte-level sha256 inventories).
* Successful deletions remove ONLY the artifact's own directory,
  ATOMICALLY (no ``.tmp-delete-*`` residue), with a deterministic
  files/bytes result measured before removal; registries shrink;
  unknown ids -> the family's 404; the guard recomputes live.
* No cascade: deleting a referencing artifact (an unreferenced
  tokenizer a dataset's tokenizer_training names) shrinks the
  reference list and may unblock the dataset — the user drives every
  step explicitly.
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


def _dir_stats(path: Path) -> tuple[int, int]:
    """INDEPENDENT walk of one artifact directory (files, bytes)."""
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def _no_delete_residue(root: Path) -> bool:
    for family in ("datasets", "tokenizers"):
        d = root / family
        if d.exists() and any(p.name.startswith(".") for p in d.iterdir()):
            return False
    return True


# --------------------------------------------------------------------------- #
# Multi-reference fixture (the M64 shape: every category non-empty)
# --------------------------------------------------------------------------- #

DS_A = "m65-alpha-ds"       # referenced by everything
DS_B = "m65-beta-ds"        # unreferenced


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
    e = Env(tmp_path_factory.mktemp("m65-root"))
    forge = e.forge
    up = forge.upload_dataset([("a.txt", _corpus(190, "m65a"))], name=DS_A)
    e.ds_a = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m65-tok-a", vocab_size=320), dataset_id=e.ds_a)
    e.tok_a = tok.id
    forge.tokenize_dataset(e.ds_a, e.tok_a)
    tok2 = forge.train_tokenizer(
        TokenizerConfig(name="m65-tok-b", vocab_size=320), dataset_id=e.ds_a)
    e.tok_b = tok2.id
    upb = forge.upload_dataset([("b.txt", _corpus(60, "m65b"))],
                               name=DS_B)
    e.ds_b = upb["dataset_id"]

    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m65-model", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
        seed=1)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.model, dataset_id=e.ds_a,
        tokenizer_id=e.tok_a, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1, steps=8))
    ckpts = [c.checkpoint_id for c in forge.list_checkpoints(e.model)]
    forge.run_evaluation(EvaluationConfig(
        model_id=e.model, dataset_id=e.ds_a, tokenizer_id=e.tok_a,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=ckpts[0]))
    forge.run_comparison(ComparisonRequest(
        model_id=e.model,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ckpts[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ckpts[1]),
        dataset_id=e.ds_a, split="validation", tokenizer_id=e.tok_a,
        batch_size=8, max_seq_len=32, seed=3, tolerance=1e-4))
    forge.run_workflow(WorkflowPlan(
        name="m65-usage-workflow", model_id=e.model,
        description="M65 retention fixture",
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
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m65-suite", description="M65 retention fixture",
        probes=[SuiteProbe(dataset_id=e.ds_a, split="validation",
                           tokenizer_id=e.tok_a, batch_size=8,
                           max_seq_len=32, seed=42)]))
    forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m65-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=ckpts[0])))
    best = forge.select_best_checkpoint(e.model).checkpoint.checkpoint_id
    sample = forge.generate_sample(SampleGenerateRequest(
        model_id=e.model, checkpoint_id=best, tokenizer_id=e.tok_a,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    forge.evaluate_sample(e.model, sample.sample_id)
    return e


# --------------------------------------------------------------------------- #
# The headline invariant, both directions
# --------------------------------------------------------------------------- #

def test_m65_dataset_guard_is_m64(env):
    forge, root = env.forge, env.root
    before = _inventory(root)

    blockers = forge.dataset_deletion_blockers(env.ds_a)
    ov = forge.dataset_usage_overview(env.ds_a)

    # canonical order: blockers are the non-empty categories, in the
    # overview's own order
    expected = [(c.category, c.references)
                for c in ov.categories if c.references]
    assert [b.reason for b in blockers] == [c for c, _r in expected]
    for b, (cat, refs) in zip(blockers, expected):
        assert b.detail == ", ".join(refs)

    # every category of the fixture is non-empty (full coverage)
    assert [c for c, _r in expected] == \
        list(ModelForge.DATASET_USAGE_CATEGORIES)

    # the deletion is refused and writes nothing
    with pytest.raises(ValueError) as exc:
        forge.delete_dataset(env.ds_a)
    assert "tokenizer_training" in str(exc.value)
    assert env.tok_a in str(exc.value) and env.tok_b in str(exc.value)
    assert _inventory(root) == before
    assert _no_delete_residue(root)

    # unknown dataset -> the family's 404
    with pytest.raises(FileNotFoundError):
        forge.delete_dataset("no-such-m65-dataset")


def test_m65_tokenizer_guard_is_m64(env):
    forge, root = env.forge, env.root
    before = _inventory(root)

    blockers = forge.tokenizer_deletion_blockers(env.tok_a)
    ov = forge.tokenizer_usage_overview(env.tok_a)
    expected = [(c.category, c.references)
                for c in ov.categories if c.references]
    assert [b.reason for b in blockers] == [c for c, _r in expected]
    for b, (cat, refs) in zip(blockers, expected):
        assert b.detail == ", ".join(refs)
    assert [c for c, _r in expected] == \
        list(ModelForge.TOKENIZER_USAGE_CATEGORIES)

    with pytest.raises(ValueError) as exc:
        forge.delete_tokenizer(env.tok_a)
    assert "tokenized_dataset" in str(exc.value)   # blocks its own artifacts
    assert _inventory(root) == before
    assert _no_delete_residue(root)

    with pytest.raises(FileNotFoundError):
        forge.delete_tokenizer("no-such-m65-tokenizer")


# --------------------------------------------------------------------------- #
# Successful deletions: exact stats, atomic, live recompute
# --------------------------------------------------------------------------- #

def test_m65_unreferenced_deletion_and_live_recompute(env):
    forge, root = env.forge, env.root

    # ---- an unreferenced tokenizer deletes with EXACT stats -------- #
    stats = _dir_stats(root / "tokenizers" / env.tok_b)
    before = _inventory(root)
    res = forge.delete_tokenizer(env.tok_b)
    assert res.model_dump() == {"tokenizer_id": env.tok_b,
                                "files_removed": stats[0],
                                "bytes_reclaimed": stats[1]}
    # only the tokenizer's own files are gone — nothing else moved
    after = _inventory(root)
    assert set(before) - set(after) == {
        f"tokenizers/{env.tok_b}/manifest.json",
        f"tokenizers/{env.tok_b}/tokenizer.json"}
    assert {k: v for k, v in after.items() if k in before} == \
        {k: v for k, v in before.items() if k in after}
    # registries shrink; repeated deletion -> the family's 404
    assert env.tok_b not in {t.id for t in forge.tokenizers.list()}
    with pytest.raises(FileNotFoundError):
        forge.delete_tokenizer(env.tok_b)
    assert _no_delete_residue(root)               # atomic: no residue

    # ---- live recompute: the dataset's tokenizer_training shrank ---- #
    ov = forge.dataset_usage_overview(env.ds_a)
    tt = next(c for c in ov.categories if c.category == "tokenizer_training")
    assert tt.references == [env.tok_a]
    # ...and the dataset is STILL protected by its other references
    blockers = forge.dataset_deletion_blockers(env.ds_a)
    assert "tokenizer_training" in [b.reason for b in blockers]

    # ---- an unreferenced dataset deletes with EXACT stats ---------- #
    stats = _dir_stats(root / "datasets" / env.ds_b)
    before = _inventory(root)
    res = forge.delete_dataset(env.ds_b)
    assert res.dataset_id == env.ds_b
    assert (res.files_removed, res.bytes_reclaimed) == stats
    after = _inventory(root)
    assert all(not k.startswith(f"datasets/{env.ds_b}/")
               for k in after)                    # own tree fully gone
    assert {k: v for k, v in after.items() if k in before} == \
        {k: v for k, v in before.items() if k in after}
    with pytest.raises(FileNotFoundError):
        forge.datasets.load_meta(env.ds_b)
    assert _no_delete_residue(root)

    # ---- the full unblock chain: fresh dataset + tokenizer trained --
    # -- on it; the dataset is blocked ONLY by tokenizer_training, ----
    # -- the tokenizer is unreferenced -> deleting it unblocks both -- #
    up = forge.upload_dataset([("c.txt", _corpus(40, "m65c"))],
                              name="m65-chain-ds")
    chain_ds = up["dataset_id"]
    chain_tok = forge.train_tokenizer(
        TokenizerConfig(name="m65-chain-tok", vocab_size=300),
        dataset_id=chain_ds).id
    blockers = forge.dataset_deletion_blockers(chain_ds)
    assert [b.reason for b in blockers] == ["tokenizer_training"]
    assert blockers[0].detail == chain_tok
    with pytest.raises(ValueError):
        forge.delete_dataset(chain_ds)            # still protected
    res = forge.delete_tokenizer(chain_tok)       # the explicit unblock
    assert res.files_removed == 2
    res = forge.delete_dataset(chain_ds)          # NOW allowed
    assert res.files_removed > 0
    with pytest.raises(FileNotFoundError):
        forge.datasets.load_meta(chain_ds)


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m65_api_retention(api_client):
    import json as _json

    # fixture through the public routes: a referenced dataset+tokenizer
    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m65api.txt", _corpus(170, "m65api"), "text/plain"))],
        data={"name": "m65api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m65api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m65api-model", "vocab_size": 640,
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

    storage_root = Path(api_client.get("/api/v1/project").json()["storage_root"])

    # ---- referenced: 409 with the SAME blockers the overview shows -- #
    ov = api_client.get(f"/api/v1/datasets/{ds}/usage").json()
    ov_blockers = [{"reason": c["category"], "detail": ", ".join(c["references"])}
                   for c in ov["categories"] if c["references"]]
    r = api_client.delete(f"/api/v1/datasets/{ds}")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert set(detail) == {"message", "dataset_id", "protected", "blockers"}
    assert detail["dataset_id"] == ds and detail["protected"] is True
    assert detail["blockers"] == ov_blockers          # both directions

    tov = api_client.get(f"/api/v1/tokenizers/{tok}/usage").json()
    tov_blockers = [{"reason": c["category"],
                     "detail": ", ".join(c["references"])}
                    for c in tov["categories"] if c["references"]]
    r = api_client.delete(f"/api/v1/tokenizers/{tok}")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert set(detail) == {"message", "tokenizer_id", "protected", "blockers"}
    assert detail["tokenizer_id"] == tok and detail["protected"] is True
    assert detail["blockers"] == tov_blockers

    # refusals wrote nothing
    def inv(root):
        return {p.relative_to(root).as_posix():
                hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file() and p.relative_to(root).parts[0] != "tmp"}
    before = inv(storage_root)
    assert api_client.delete(f"/api/v1/datasets/{ds}").status_code == 409
    assert api_client.delete(f"/api/v1/tokenizers/{tok}").status_code == 409
    assert inv(storage_root) == before

    # ---- unknown ids -> 404 ----------------------------------------- #
    assert api_client.delete("/api/v1/datasets/no-m65").status_code == 404
    assert api_client.delete("/api/v1/tokenizers/no-m65").status_code == 404

    # ---- unreferenced artifacts delete over HTTP --------------------- #
    up2 = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m65free.txt", _corpus(30, "m65free"),
                          "text/plain"))],
        data={"name": "m65-free-ds"})
    free_ds = up2.json()["dataset_id"]
    tr2 = api_client.post(
        "/api/v1/tokenizers/train",
        files=[("files", ("m65free-tok.txt", _corpus(120, "m65freetok"),
                          "text/plain"))],
        data={"config": _json.dumps({"name": "m65-free-tok",
                                     "vocab_size": 300})})
    assert tr2.status_code == 201, tr2.text
    free_tok = tr2.json()["tokenizer"]["id"]

    expected_ds = _dir_stats(storage_root / "datasets" / free_ds)
    r = api_client.delete(f"/api/v1/datasets/{free_ds}")
    assert r.status_code == 200, r.text
    assert r.json() == {"dataset_id": free_ds,
                        "files_removed": expected_ds[0],
                        "bytes_reclaimed": expected_ds[1]}
    assert api_client.get(f"/api/v1/datasets/{free_ds}").status_code == 404

    expected_tok = _dir_stats(storage_root / "tokenizers" / free_tok)
    r = api_client.delete(f"/api/v1/tokenizers/{free_tok}")
    assert r.status_code == 200, r.text
    assert r.json() == {"tokenizer_id": free_tok,
                        "files_removed": expected_tok[0],
                        "bytes_reclaimed": expected_tok[1]}
    assert api_client.get(f"/api/v1/tokenizers/{free_tok}").status_code == 404

    # the guarded artifacts survive everything
    assert api_client.get(f"/api/v1/datasets/{ds}").status_code == 200
    assert api_client.get(f"/api/v1/tokenizers/{tok}").status_code == 200

    # ---- OpenAPI: NO new paths; schemas exposed --------------------- #
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 89
    d = spec["paths"]["/api/v1/datasets/{dataset_id}"]
    assert set(d.keys()) == {"get", "delete"}
    t = spec["paths"]["/api/v1/tokenizers/{tokenizer_id}"]
    assert set(t.keys()) == {"get", "delete"}
    for s in ("DatasetDeletionResult", "TokenizerDeletionResult",
              "ArtifactDeletionBlocker", "DatasetDeletionBlocked",
              "TokenizerDeletionBlocked"):
        assert s in spec["components"]["schemas"], s
