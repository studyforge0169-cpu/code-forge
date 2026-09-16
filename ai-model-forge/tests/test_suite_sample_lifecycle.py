"""Milestone 68 tests: explicit suite-run & sample lifecycle.

The M68 invariants under test:

* VERIFIED DELETION — the M61/M65/M67 pattern applied to the root-level
  runtime records: scope (unknown/registry-invisible -> 404, nothing
  deleted) -> INTEGRITY FIRST (the record's persisted result_hash must
  reproduce; tampered -> refuse, never a silent rmtree) -> blockers
  (samples only: sample-quality measurements referencing the sample) ->
  ONE atomic removal with deterministic files/bytes stats.
* NO CASCADE — a suite run's probe EVALUATIONS are model-owned and are
  never touched by suite-run deletion; a blocked sample deletion
  changes nothing on disk.
* THE UNBLOCK CHAIN (the M68 payoff) — deleting the referencing
  runtime records makes a previously blocked model deletable, live:
  the M66 usage view, the M67 retention view and the model DELETE all
  agree at every step (the M67 guard code is untouched; only its live
  input set shrinks).
* INDEPENDENT ORACLE — stats and blocker ids are cross-checked against
  raw manifest parsing, never the app's own analysis.
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
    ModelCreateRequest,
    ProbeSuiteCreateRequest,
    SampleGenerateRequest,
    SampleStrategy,
    SuiteProbe,
    SuiteRunRequest,
    TokenizerConfig,
    TransformerConfig,
    TrainingConfig,
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
    """rel-posix -> sha256 (the zero-mutation / isolation proof)."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


def _walk_stats(directory: Path) -> tuple[int, int]:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return len(files), sum(p.stat().st_size for p in files)


def _oracle_quality_ids(root: Path, model_id: str, sample_id: str) -> list[str]:
    """Independent: measurement ids referencing the sample, from raw
    manifests under sample-evaluations/<model_id>/ (never the app)."""
    out = []
    froot = root / "sample-evaluations" / model_id
    if froot.exists():
        for d in sorted(froot.iterdir()):
            if not d.is_dir():
                continue
            try:
                rec = json.loads((d / "manifest.json").read_text())
            except Exception:
                continue
            if rec.get("model_id") == model_id and \
                    rec.get("sample_id") == sample_id:
                out.append(rec["evaluation_id"])
    return sorted(out)


def _make_model(forge, name, seed):
    return forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name=name, vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
        seed=seed)))[0].id


def _train(forge, model_id, ds, tok, seed, steps=4):
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=model_id, dataset_id=ds,
        tokenizer_id=tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=2, keep_best=False, seed=seed,
        steps=steps))


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

class Env:
    def __init__(self, root: Path):
        self.forge = ModelForge(root=root)
        self.root = root
        self.ds = None
        self.tok = None
        self.suite_model = None    # blocked ONLY by a suite run
        self.suite_run = None
        self.sample_model = None   # blocked by sample + sample_quality
        self.sample = None
        self.quality = None


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m68-root"))
    forge = e.forge
    up = forge.upload_dataset([("m68.txt", _corpus(180, "m68"))],
                              name="m68-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m68-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m68-suite", description="M68 fixture",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))

    # model A: trained + one suite run (CURRENT state) -> blocked by
    # suite_run only (its probe evaluation is internal ownership)
    e.suite_model = _make_model(forge, "m68-suite-model", 1)
    _train(forge, e.suite_model, e.ds, e.tok, seed=1)
    e.suite_run = forge.run_suite(SuiteRunRequest(
        model_id=e.suite_model, suite_id="m68-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))

    # model B: trained + one sample + one quality measurement ->
    # blocked by sample + sample_quality
    e.sample_model = _make_model(forge, "m68-sample-model", 2)
    _train(forge, e.sample_model, e.ds, e.tok, seed=2)
    best = forge.select_best_checkpoint(e.sample_model).checkpoint.checkpoint_id
    e.sample = forge.generate_sample(SampleGenerateRequest(
        model_id=e.sample_model, checkpoint_id=best, tokenizer_id=e.tok,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    e.quality = forge.evaluate_sample(e.sample_model, e.sample.sample_id)
    return e


# --------------------------------------------------------------------------- #
# Suite-run deletion
# --------------------------------------------------------------------------- #

def test_m68_suite_run_deletion(env):
    forge, root = env.forge, env.root
    run_id = env.suite_run.suite_run_id

    # the run references the model (M66 surface) and the model's probe
    # evaluations exist (internal ownership)
    usage = forge.model_usage_overview(env.suite_model)
    ucats = {c.category: c.references for c in usage.categories}
    assert ucats["suite_run"] == [run_id]
    evals_before = len(ucats["evaluation"])

    run_dir = root / "suite-runs" / run_id
    exp_files, exp_bytes = _walk_stats(run_dir)
    before = _inventory(root)

    result = forge.delete_suite_run(env.suite_model, run_id)

    # deterministic stats == independent walk; the run's manifest is
    # the single file
    assert result.model_id == env.suite_model
    assert result.suite_run_id == run_id
    assert result.files_removed == exp_files == 1
    assert result.bytes_reclaimed == exp_bytes

    # ONLY the run's directory is gone; probe evaluations are NOT
    # cascaded (model-owned); M66 usage recomputes live
    after = _inventory(root)
    assert set(after) == set(before) - {
        f"suite-runs/{run_id}/manifest.json"}
    assert not run_dir.exists()
    assert len(forge.list_evaluations(env.suite_model)) == evals_before
    usage2 = forge.model_usage_overview(env.suite_model)
    assert {c.category: c.references for c in usage2.categories}[
        "suite_run"] == []

    # repeat -> missing-resource semantics; unknown -> 404 semantics
    with pytest.raises(FileNotFoundError):
        forge.delete_suite_run(env.suite_model, run_id)
    with pytest.raises(FileNotFoundError):
        forge.delete_suite_run(env.suite_model, "no-such-m68-run")
    with pytest.raises(FileNotFoundError):
        forge.delete_suite_run("no-such-m68-model", "x")

    # ---- integrity: a SECOND run, tampered result_hash -> refuse ----- #
    run2 = forge.run_suite(SuiteRunRequest(
        model_id=env.suite_model, suite_id="m68-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    mpath = root / "suite-runs" / run2.suite_run_id / "manifest.json"
    original = mpath.read_bytes()
    rec = json.loads(original)
    rec["result_hash"] = "0" * 64
    mpath.write_text(json.dumps(rec))
    before = _inventory(root)
    with pytest.raises(RuntimeError):
        forge.delete_suite_run(env.suite_model, run2.suite_run_id)
    assert _inventory(root) == before

    # corrupt manifest -> registry-invisible -> 404 semantics
    mpath.write_bytes(b"not json at all")
    before = _inventory(root)
    with pytest.raises(FileNotFoundError):
        forge.delete_suite_run(env.suite_model, run2.suite_run_id)
    assert _inventory(root) == before

    # restored -> integrity passes -> deletable again (and the first
    # tamper proves integrity, not just scope, is what refused)
    mpath.write_bytes(original)
    r2 = forge.delete_suite_run(env.suite_model, run2.suite_run_id)
    assert r2.suite_run_id == run2.suite_run_id and r2.files_removed == 1


# --------------------------------------------------------------------------- #
# Sample + sample-quality deletion (blocker semantics)
# --------------------------------------------------------------------------- #

def test_m68_sample_deletion_and_quality(env):
    forge, root = env.forge, env.root
    model_id, sample_id = env.sample_model, env.sample.sample_id
    quality_id = env.quality.evaluation_id

    usage = forge.model_usage_overview(model_id)
    ucats = {c.category: c.references for c in usage.categories}
    assert ucats["sample"] == [sample_id]
    assert ucats["sample_quality"] == [quality_id]

    sdir = root / "samples" / model_id / f"sample-{sample_id}"
    qdir = (root / "sample-evaluations" / model_id
            / f"evaluation-{quality_id}")

    # (1) the sample is BLOCKED by its quality measurement
    before = _inventory(root)
    with pytest.raises(ValueError):
        forge.delete_sample(model_id, sample_id)
    assert _inventory(root) == before
    blockers = forge.sample_deletion_blockers(model_id, sample_id)
    assert len(blockers) == 1
    assert blockers[0].reason == "sample_quality"
    assert quality_id in blockers[0].detail
    # the independent oracle agrees on the blocking measurement ids
    assert _oracle_quality_ids(root, model_id, sample_id) == [quality_id]

    # (2) delete the measurement (leaf): exact stats, isolation
    exp_q = _walk_stats(qdir)
    rq = forge.delete_sample_evaluation(model_id, quality_id)
    assert rq.model_id == model_id and rq.evaluation_id == quality_id
    assert rq.sample_id == sample_id
    assert (rq.files_removed, rq.bytes_reclaimed) == exp_q
    after_q = _inventory(root)
    assert set(after_q) == set(before) - {
        f"sample-evaluations/{model_id}/evaluation-{quality_id}/"
        f"manifest.json"}
    # the sample itself survives
    assert forge.get_sample(model_id, sample_id).sample_id == sample_id
    # the blocker set is now EMPTY (live recompute)
    assert forge.sample_deletion_blockers(model_id, sample_id) == []
    # ... but the MODEL is still blocked by the sample itself
    m_usage = forge.model_usage_overview(model_id)
    m_cats = {c.category: c.references
              for c in m_usage.categories}
    assert m_cats["sample"] == [sample_id]
    assert m_cats["sample_quality"] == []

    # (3) delete the sample: exact stats, isolation
    exp_s = _walk_stats(sdir)
    rs = forge.delete_sample(model_id, sample_id)
    assert rs.model_id == model_id and rs.sample_id == sample_id
    assert (rs.files_removed, rs.bytes_reclaimed) == exp_s
    after_s = _inventory(root)
    assert set(after_s) == set(after_q) - {
        f"samples/{model_id}/sample-{sample_id}/manifest.json"}
    with pytest.raises(FileNotFoundError):
        forge.get_sample(model_id, sample_id)
    # the model's usage drains live
    m_cats = {c.category: c.references
              for c in forge.model_usage_overview(model_id).categories}
    assert m_cats["sample"] == [] and m_cats["sample_quality"] == []

    # (4) unknowns + registry-invisible + tampered hash
    with pytest.raises(FileNotFoundError):
        forge.delete_sample(model_id, "no-such-m68-sample")
    with pytest.raises(FileNotFoundError):
        forge.delete_sample("no-such-m68-model", "x")
    with pytest.raises(FileNotFoundError):
        forge.delete_sample_evaluation(model_id, "no-such-m68-eval")

    # a fresh sample: tampered result_hash -> integrity refusal
    best = forge.select_best_checkpoint(model_id).checkpoint.checkpoint_id
    s2 = forge.generate_sample(SampleGenerateRequest(
        model_id=model_id, checkpoint_id=best, tokenizer_id=env.tok,
        prompt="cloud forest", strategy=SampleStrategy.GREEDY,
        max_new_tokens=4))
    mpath = (root / "samples" / model_id
             / f"sample-{s2.sample_id}" / "manifest.json")
    original = mpath.read_bytes()
    rec = json.loads(original)
    rec["result_hash"] = "0" * 64
    mpath.write_text(json.dumps(rec))
    before = _inventory(root)
    with pytest.raises(RuntimeError):
        forge.delete_sample(model_id, s2.sample_id)
    assert _inventory(root) == before
    mpath.write_bytes(b"not json")
    before = _inventory(root)
    with pytest.raises(FileNotFoundError):
        forge.delete_sample(model_id, s2.sample_id)
    assert _inventory(root) == before
    mpath.write_bytes(original)
    assert forge.delete_sample(model_id, s2.sample_id).files_removed == 1

    # a fresh measurement: tampered hash -> integrity refusal too
    s3 = forge.generate_sample(SampleGenerateRequest(
        model_id=model_id, checkpoint_id=best, tokenizer_id=env.tok,
        prompt="desert island", strategy=SampleStrategy.GREEDY,
        max_new_tokens=4))
    q3 = forge.evaluate_sample(model_id, s3.sample_id)
    qpath = (root / "sample-evaluations" / model_id
             / f"evaluation-{q3.evaluation_id}" / "manifest.json")
    qoriginal = qpath.read_bytes()
    rec = json.loads(qoriginal)
    rec["result_hash"] = "0" * 64
    qpath.write_text(json.dumps(rec))
    before = _inventory(root)
    with pytest.raises(RuntimeError):
        forge.delete_sample_evaluation(model_id, q3.evaluation_id)
    assert _inventory(root) == before
    qpath.write_bytes(qoriginal)
    # clean up the fixture tail (sample + measurement), proving the
    # normal path still works after the integrity round-trip
    assert forge.delete_sample_evaluation(
        model_id, q3.evaluation_id).files_removed == 1
    assert forge.delete_sample(model_id, s3.sample_id).files_removed == 1


# --------------------------------------------------------------------------- #
# The unblock chains (M66/M67/M68 agreement, live)
# --------------------------------------------------------------------------- #

def test_m68_unblock_chains(env):
    """Self-contained: each chain builds its own model, watches the
    M66 usage / M67 retention views and the model DELETE agree at
    every step, drains the external references through the NEW M68
    deletions and finally deletes the model (the M68 payoff)."""
    forge, root = env.forge, env.root

    # ---- chain A: model blocked ONLY by a suite run ------------------- #
    mid = _make_model(forge, "m68-chain-a", 11)
    _train(forge, mid, env.ds, env.tok, seed=11)
    run = forge.run_suite(SuiteRunRequest(
        model_id=mid, suite_id="m68-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    usage = forge.model_usage_overview(mid)
    assert usage.external_references == 1       # the suite run only
    ret = forge.model_retention_overview(mid)
    assert ret.deletable is False
    assert [b.category for b in ret.blockers] == ["suite_run"]
    with pytest.raises(ValueError):
        forge.delete_model(mid)                  # M67 guard refuses
    # drain: delete the suite run through the NEW M68 deletion
    assert forge.delete_suite_run(mid, run.suite_run_id).files_removed == 1
    # the model is NOW unblocked — live, guard code untouched
    usage = forge.model_usage_overview(mid)
    assert usage.external_references == 0
    ret = forge.model_retention_overview(mid)
    assert ret.deletable is True and ret.blockers == []
    model_dir = root / "models" / mid
    exp_files, exp_bytes = _walk_stats(model_dir)
    assert (len(ret.files), ret.size_bytes) == (exp_files, exp_bytes)
    res = forge.delete_model(mid)
    assert res.model_id == mid
    assert (res.files_removed, res.bytes_reclaimed) == \
        (exp_files, exp_bytes)
    assert not model_dir.exists()
    with pytest.raises(FileNotFoundError):
        forge.delete_model(mid)
    # the shared dataset/tokenizer survive untouched
    assert (root / "datasets" / env.ds).exists()
    assert (root / "tokenizers" / env.tok).exists()

    # ---- chain B: model blocked by sample + sample_quality ------------ #
    mid = _make_model(forge, "m68-chain-b", 12)
    _train(forge, mid, env.ds, env.tok, seed=12)
    best = forge.select_best_checkpoint(mid).checkpoint.checkpoint_id
    smp = forge.generate_sample(SampleGenerateRequest(
        model_id=mid, checkpoint_id=best, tokenizer_id=env.tok,
        prompt="cloud forest", strategy=SampleStrategy.GREEDY,
        max_new_tokens=4))
    sq = forge.evaluate_sample(mid, smp.sample_id)
    ret = forge.model_retention_overview(mid)
    assert ret.deletable is False
    # canonical M66 order: sample before sample_quality
    assert [(b.category, b.reference_id) for b in ret.blockers] == [
        ("sample", smp.sample_id), ("sample_quality", sq.evaluation_id)]
    # step 1: delete the measurement -> still blocked by the sample
    assert forge.delete_sample_evaluation(
        mid, sq.evaluation_id).files_removed == 1
    ret = forge.model_retention_overview(mid)
    assert ret.deletable is False
    assert [(b.category, b.reference_id) for b in ret.blockers] == [
        ("sample", smp.sample_id)]
    # step 2: delete the sample -> unblocked
    assert forge.delete_sample(mid, smp.sample_id).files_removed == 1
    ret = forge.model_retention_overview(mid)
    assert ret.deletable is True and ret.blockers == []
    res = forge.delete_model(mid)
    assert res.files_removed == len(ret.files)
    assert res.bytes_reclaimed == ret.size_bytes
    assert not (root / "models" / mid).exists()


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m68_api_lifecycle(api_client):
    import json as _json

    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m68api.txt", _corpus(170, "m68api"),
                          "text/plain"))],
        data={"name": "m68api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m68api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m68api-model", "vocab_size": 640,
                   "context_length": 64, "hidden_size": 64, "n_layers": 2,
                   "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128}})
    assert m.status_code == 201, m.text
    mid = m.json()["model"]["id"]
    r = api_client.post("/api/v1/training/run", json={
        "method": "continued_pretraining", "model_id": mid, "dataset_id": ds,
        "tokenizer_id": tok, "learning_rate": 3e-3, "batch_size": 8,
        "max_seq_len": 32, "eval_every_steps": 2, "keep_best": False,
        "seed": 1, "steps": 4})
    assert r.status_code == 200, r.text
    ck = api_client.get(f"/api/v1/models/{mid}/checkpoints").json()[0][
        "checkpoint_id"]

    # the three runtime records over HTTP
    r = api_client.post("/api/v1/probe-suites", json={
        "suite_id": "m68api-suite", "description": "M68 api fixture",
        "probes": [{"dataset_id": ds, "split": "validation",
                    "tokenizer_id": tok, "batch_size": 8,
                    "max_seq_len": 32, "seed": 42}]})
    assert r.status_code == 201, r.text
    r = api_client.post("/api/v1/suite-runs", json={
        "model_id": mid, "suite_id": "m68api-suite",
        "state": {"state_kind": "current"}})
    assert r.status_code == 200, r.text
    run_id = r.json()["suite_run_id"]
    r = api_client.post("/api/v1/samples/generate", json={
        "model_id": mid, "checkpoint_id": ck, "tokenizer_id": tok,
        "prompt": "river mountain", "strategy": "greedy",
        "max_new_tokens": 4})
    assert r.status_code == 200, r.text
    sample_id = r.json()["sample_id"]
    r = api_client.post(
        f"/api/v1/models/{mid}/samples/{sample_id}/quality")
    assert r.status_code == 200, r.text
    quality_id = r.json()["evaluation_id"]

    storage_root = Path(api_client.get("/api/v1/project").json()[
        "storage_root"])

    # the model is blocked by all THREE external references
    usage = api_client.get(f"/api/v1/models/{mid}/usage").json()
    ucats = {c["category"]: c["references"] for c in usage["categories"]}
    assert ucats["suite_run"] == [run_id]
    assert ucats["sample"] == [sample_id]
    assert ucats["sample_quality"] == [quality_id]

    # the sample DELETE is refused with the typed structured 409
    before = _inventory(storage_root)
    d = api_client.delete(f"/api/v1/models/{mid}/samples/{sample_id}")
    assert d.status_code == 409, d.text
    detail = d.json()["detail"]
    assert set(detail) == {"message", "model_id", "sample_id",
                           "protected", "blockers"}
    assert detail["model_id"] == mid and detail["sample_id"] == sample_id
    assert detail["protected"] is True
    assert detail["blockers"] == [
        {"reason": "sample_quality",
         "detail": f"sample-quality measurement(s) '{quality_id}'"}]
    assert _inventory(storage_root) == before

    # the model DELETE is still refused (M67 surface, untouched guard)
    d = api_client.delete(f"/api/v1/models/{mid}")
    assert d.status_code == 409, d.text
    mdetail = d.json()["detail"]
    assert [(b["category"], b["reference_id"])
            for b in mdetail["blockers"]] == [
        ("suite_run", run_id), ("sample", sample_id),
        ("sample_quality", quality_id)]
    assert _inventory(storage_root) == before

    # unblock over HTTP: measurement -> sample -> suite run -> model
    d = api_client.delete(
        f"/api/v1/models/{mid}/sample-quality/{quality_id}")
    assert d.status_code == 200, d.text
    assert d.json() == {"model_id": mid, "evaluation_id": quality_id,
                        "sample_id": sample_id, "files_removed": 1,
                        "bytes_reclaimed": d.json()["bytes_reclaimed"]}
    assert d.json()["bytes_reclaimed"] > 0

    d = api_client.delete(f"/api/v1/models/{mid}/samples/{sample_id}")
    assert d.status_code == 200, d.text
    body = d.json()
    assert set(body) == {"model_id", "sample_id", "files_removed",
                         "bytes_reclaimed"}
    assert body["files_removed"] == 1 and body["bytes_reclaimed"] > 0

    d = api_client.delete(
        f"/api/v1/models/{mid}/suite-runs/{run_id}")
    assert d.status_code == 200, d.text
    body = d.json()
    assert set(body) == {"model_id", "suite_run_id", "files_removed",
                         "bytes_reclaimed"}
    assert body["suite_run_id"] == run_id

    # the model is now deletable — the full M68 payoff over HTTP
    ret = api_client.get(f"/api/v1/models/{mid}/retention").json()
    assert ret["deletable"] is True and ret["blockers"] == []
    d = api_client.delete(f"/api/v1/models/{mid}")
    assert d.status_code == 200, d.text
    assert d.json()["model_id"] == mid
    assert api_client.get(f"/api/v1/models/{mid}").status_code == 404

    # 404s: unknown ids on all three routes + repeats; determinism
    for path in (f"/api/v1/models/{mid}/suite-runs/{run_id}",
                 f"/api/v1/models/{mid}/samples/{sample_id}",
                 f"/api/v1/models/{mid}/sample-quality/{quality_id}",
                 "/api/v1/models/no-m68/suite-runs/x",
                 "/api/v1/models/no-m68/samples/x",
                 "/api/v1/models/no-m68/sample-quality/x",
                 f"/api/v1/models/{mid}/suite-runs/no-such",
                 f"/api/v1/models/{mid}/samples/no-such",
                 f"/api/v1/models/{mid}/sample-quality/no-such"):
        r1 = api_client.delete(path)
        r2 = api_client.delete(path)
        assert r1.status_code == 404, path
        assert r1.content == r2.content    # deterministic repeats

    # OpenAPI: 93 paths (the three DELETEs are new OPERATIONS on the
    # EXISTING resource paths — zero new paths); delete set == 7
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 93
    for path, get_and_delete in (
            ("/api/v1/models/{model_id}/suite-runs/{suite_run_id}", True),
            ("/api/v1/models/{model_id}/samples/{sample_id}", True),
            ("/api/v1/models/{model_id}/sample-quality/{evaluation_id}",
             True)):
        ops = set(spec["paths"][path].keys())
        if get_and_delete:
            assert ops == {"get", "delete"}, path
    deletes = sorted(p for p, ops in spec["paths"].items()
                     if "delete" in ops)
    assert deletes == [
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/models/{model_id}",
        "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}",
        "/api/v1/models/{model_id}/sample-quality/{evaluation_id}",
        "/api/v1/models/{model_id}/samples/{sample_id}",
        "/api/v1/models/{model_id}/suite-runs/{suite_run_id}",
        "/api/v1/tokenizers/{tokenizer_id}",
    ]
    for s in ("SuiteRunDeletionResult", "SampleDeletionResult",
              "SampleEvaluationDeletionResult", "SampleDeletionBlocked"):
        assert s in spec["components"]["schemas"], s
    # the 409 response model is documented on the sample DELETE
    assert spec["paths"]["/api/v1/models/{model_id}/samples/{sample_id}"][
        "delete"]["responses"]["409"]["content"]["application/json"][
        "schema"]["$ref"] == "#/components/schemas/SampleDeletionBlocked"
