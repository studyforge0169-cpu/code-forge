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
    assert len(spec["paths"]) == 91
    d = spec["paths"]["/api/v1/datasets/{dataset_id}"]
    assert set(d.keys()) == {"get", "delete"}
    t = spec["paths"]["/api/v1/tokenizers/{tokenizer_id}"]
    assert set(t.keys()) == {"get", "delete"}
    for s in ("DatasetDeletionResult", "TokenizerDeletionResult",
              "ArtifactDeletionBlocker", "DatasetDeletionBlocked",
              "TokenizerDeletionBlocked"):
        assert s in spec["components"]["schemas"], s


# --------------------------------------------------------------------------- #
# M65 spec compliance: retention views + integrity-first guards
# --------------------------------------------------------------------------- #

def _walk_artifact(directory: Path) -> tuple[list[str], int]:
    """INDEPENDENT ordered walk of one artifact directory."""
    files, total = [], 0
    for q in sorted(directory.rglob("*")):
        rel = q.relative_to(directory)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if q.is_file():
            files.append(rel.as_posix())
            total += q.stat().st_size
    return files, total


def test_m65_retention_views_and_integrity(env):
    forge, root = env.forge, env.root

    # ---- healthy unreferenced artifacts: deletable, exact stats ------ #
    # rv_free: NOTHING references it (no tokenizer trained on it)
    up = forge.upload_dataset([("m65free.txt", _corpus(50, "m65free"))],
                              name="m65-rv-free-ds")
    rv_free = up["dataset_id"]
    ov = forge.dataset_retention_overview(rv_free)
    ov2 = forge.dataset_retention_overview(rv_free)
    files, nbytes = _walk_artifact(root / "datasets" / rv_free)
    assert ov.model_dump(mode="json") == ov2.model_dump(mode="json")
    assert ov.dataset_id == rv_free and ov.version_count == 1
    assert ov.files == files and ov.size_bytes == nbytes
    assert ov.integrity_verified is True
    assert ov.deletable is True and ov.blockers == []

    # the pair: rv_tok is trained on rv_ds (unreferenced itself), so
    # rv_ds carries exactly ONE blocker and rv_tok none
    up = forge.upload_dataset([("m65rv.txt", _corpus(50, "m65rv"))],
                              name="m65-rv-ds")
    rv_ds = up["dataset_id"]
    rv_tok = forge.train_tokenizer(
        TokenizerConfig(name="m65-rv-tok", vocab_size=300),
        dataset_id=rv_ds).id
    dov = forge.dataset_retention_overview(rv_ds)
    assert [b.reason for b in dov.blockers] == ["tokenizer_training"]
    assert dov.deletable is False and dov.integrity_verified is True

    tv = forge.tokenizer_retention_overview(rv_tok)
    tfiles, tbytes = _walk_artifact(root / "tokenizers" / rv_tok)
    assert tv.files == tfiles == ["manifest.json", "tokenizer.json"]
    assert tv.size_bytes == tbytes
    assert tv.integrity_verified is True
    assert tv.deletable is True and tv.blockers == []
    assert tv.trained_on_dataset_id == rv_ds

    # ---- referenced artifacts: the view is the guard (both directions) #
    dref = forge.dataset_retention_overview(env.ds_a)
    usage = forge.dataset_usage_overview(env.ds_a)
    guard = forge.dataset_deletion_blockers(env.ds_a)
    ret_reasons = [b.reason for b in dref.blockers]
    usage_nonempty = [c.category for c in usage.categories if c.references]
    assert ret_reasons == usage_nonempty == [b.reason for b in guard]
    assert dref.integrity_verified is True and dref.deletable is False
    # THE §10 invariant, per category, both directions
    for c in usage.categories:
        if c.references:
            assert c.category in ret_reasons
        else:
            assert c.category not in ret_reasons
    # the dataset's own files include the tokenized artifacts
    assert any(f.startswith("v1/tokenized/") for f in dref.files)

    tref = forge.tokenizer_retention_overview(env.tok_a)
    tusage = forge.tokenizer_usage_overview(env.tok_a)
    tguard = forge.tokenizer_deletion_blockers(env.tok_a)
    assert [b.reason for b in tref.blockers] == \
        [c.category for c in tusage.categories if c.references] == \
        [b.reason for b in tguard]
    assert tref.deletable is False and tref.integrity_verified is True

    # ---- corrupt dataset: never deletable, deletion refused ---------- #
    rec = root / "datasets" / rv_ds / "v1" / "records.jsonl.gz"
    original = rec.read_bytes()
    rec.write_bytes(original + b"x")            # tamper (hash mismatch)
    cov = forge.dataset_retention_overview(rv_ds)
    assert cov.integrity_verified is False and cov.deletable is False
    before = _inventory(root)
    # INTEGRITY FIRST: rv_ds is also reference-protected, but the M61
    # ordering refuses on the integrity failure (RuntimeError), never
    # on the blockers — deletion never bypasses integrity validation
    with pytest.raises(RuntimeError):
        forge.delete_dataset(rv_ds)
    assert _inventory(root) == before           # the refusal wrote nothing
    rec.write_bytes(original)                   # restore -> integrity flips
    restored = forge.dataset_retention_overview(rv_ds)
    assert restored.integrity_verified is True
    assert [b.reason for b in restored.blockers] == ["tokenizer_training"]
    assert restored.deletable is False         # still pair-protected

    # ---- corrupt tokenizer: content-hash mismatch, refused ----------- #
    tj = root / "tokenizers" / rv_tok / "tokenizer.json"
    torig = tj.read_bytes()
    flipped = bytearray(torig)
    flipped[len(flipped) // 2] ^= 0xFF
    tj.write_bytes(bytes(flipped))
    ctv = forge.tokenizer_retention_overview(rv_tok)
    assert ctv.integrity_verified is False and ctv.deletable is False
    before = _inventory(root)
    with pytest.raises(RuntimeError):
        forge.delete_tokenizer(rv_tok)
    assert _inventory(root) == before
    tj.write_bytes(torig)
    assert forge.tokenizer_retention_overview(rv_tok).deletable is True

    # ---- malformed manifests: registry-invisible (the M61 404) ------- #
    dmeta = root / "datasets" / rv_ds / "dataset.json"
    dorig = dmeta.read_bytes()
    dmeta.write_bytes(b"not json at all")
    with pytest.raises(FileNotFoundError):
        forge.dataset_retention_overview(rv_ds)
    with pytest.raises(FileNotFoundError):
        forge.delete_dataset(rv_ds)             # refused, nothing deleted
    dmeta.write_bytes(dorig)

    tmeta = root / "tokenizers" / rv_tok / "manifest.json"
    tmeta_orig = tmeta.read_bytes()
    tmeta.write_bytes(b"not json at all")
    with pytest.raises(FileNotFoundError):
        forge.tokenizer_retention_overview(rv_tok)
    with pytest.raises(FileNotFoundError):
        forge.delete_tokenizer(rv_tok)
    tmeta.write_bytes(tmeta_orig)
    assert forge.tokenizer_retention_overview(rv_tok).deletable is True

    # cleanup: rv_tok is unreferenced -> safe deletion; rv_ds + rv_free
    assert forge.delete_tokenizer(rv_tok).files_removed == 2
    assert forge.delete_dataset(rv_ds).files_removed == len(dov.files)
    assert forge.delete_dataset(rv_free).files_removed == len(files)


def test_m65_api_retention_views(api_client):
    import json as _json

    # a referenced dataset/tokenizer pair through the public routes
    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m65rv.txt", _corpus(170, "m65rvapi"), "text/plain"))],
        data={"name": "m65rvapi-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m65rvapi-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200

    storage_root = Path(api_client.get("/api/v1/project").json()["storage_root"])

    def inv(root):
        return {p.relative_to(root).as_posix():
                hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file() and p.relative_to(root).parts[0] != "tmp"}

    # ---- the retention views over HTTP -------------------------------- #
    r = api_client.get(f"/api/v1/datasets/{ds}/retention")
    assert r.status_code == 200, r.text
    body_ds = r.content
    ov = r.json()
    assert set(ov) == {"dataset_id", "name", "created_at", "version_count",
                       "latest_version", "files", "size_bytes",
                       "integrity_verified", "deletable", "blockers"}
    files, nbytes = _walk_artifact(storage_root / "datasets" / ds)
    assert ov["files"] == files and ov["size_bytes"] == nbytes
    assert ov["integrity_verified"] is True and ov["deletable"] is False
    usage = api_client.get(f"/api/v1/datasets/{ds}/usage").json()
    usage_blockers = [{"reason": c["category"],
                       "detail": ", ".join(c["references"])}
                      for c in usage["categories"] if c["references"]]
    assert ov["blockers"] == usage_blockers        # the view is the guard

    r = api_client.get(f"/api/v1/tokenizers/{tok}/retention")
    assert r.status_code == 200, r.text
    body_tok = r.content
    tv = r.json()
    assert set(tv) == {"tokenizer_id", "name", "created_at",
                       "requested_vocab_size", "actual_vocab_size",
                       "trained_on_dataset_id", "files", "size_bytes",
                       "integrity_verified", "deletable", "blockers"}
    assert tv["files"] == ["manifest.json", "tokenizer.json"]
    assert tv["integrity_verified"] is True and tv["deletable"] is False
    tusage = api_client.get(f"/api/v1/tokenizers/{tok}/usage").json()
    assert tv["blockers"] == [{"reason": c["category"],
                               "detail": ", ".join(c["references"])}
                              for c in tusage["categories"]
                              if c["references"]]

    # ---- integrity refusals over HTTP --------------------------------- #
    before = inv(storage_root)
    rec = storage_root / "datasets" / ds / "v1" / "records.jsonl.gz"
    original = rec.read_bytes()
    rec.write_bytes(original + b"x")
    r = api_client.get(f"/api/v1/datasets/{ds}/retention")
    assert r.status_code == 200, r.text
    assert r.json()["integrity_verified"] is False
    assert r.json()["deletable"] is False         # corrupt: never deletable
    r = api_client.delete(f"/api/v1/datasets/{ds}")
    assert r.status_code == 409, r.text           # integrity refusal (409)
    assert "integrity" in r.json()["detail"]
    rec.write_bytes(original)
    assert api_client.get(
        f"/api/v1/datasets/{ds}/retention").json()["deletable"] is False
    # (still protected by references — integrity passed again)

    # malformed manifest -> registry-invisible -> 404 (view AND delete)
    dmeta = storage_root / "datasets" / ds / "dataset.json"
    dorig = dmeta.read_bytes()
    dmeta.write_bytes(b"not json")
    assert api_client.get(
        f"/api/v1/datasets/{ds}/retention").status_code == 404
    r = api_client.delete(f"/api/v1/datasets/{ds}")
    assert r.status_code == 404
    dmeta.write_bytes(dorig)
    assert api_client.get(
        f"/api/v1/datasets/{ds}/retention").status_code == 200

    # unknown ids -> 404
    assert api_client.get("/api/v1/datasets/no-m65/retention").status_code == 404
    assert api_client.get(
        "/api/v1/tokenizers/no-m65/retention").status_code == 404

    # ---- determinism + zero mutation ---------------------------------- #
    assert api_client.get(
        f"/api/v1/datasets/{ds}/retention").content == body_ds
    assert api_client.get(
        f"/api/v1/tokenizers/{tok}/retention").content == body_tok
    assert inv(storage_root) == before

    # ---- OpenAPI: exactly two new paths (89 -> 91) -------------------- #
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 91
    for pth in ("/api/v1/datasets/{dataset_id}/retention",
                "/api/v1/tokenizers/{tokenizer_id}/retention"):
        assert set(spec["paths"][pth].keys()) == {"get"}
    for s in ("DatasetRetentionOverview", "TokenizerRetentionOverview"):
        assert s in spec["components"]["schemas"], s
    assert set(spec["paths"]["/api/v1/datasets/{dataset_id}"].keys()) == \
        {"get", "delete"}
    assert set(spec["paths"]["/api/v1/tokenizers/{tokenizer_id}"].keys()) == \
        {"get", "delete"}
