"""Milestone 63 tests: read-only PROJECT STORAGE overview (engine + HTTP).

The M63 invariants under test:

* PHYSICAL accounting — one filesystem walk, every project file in
  EXACTLY ONE category (never double-counted, never silently
  discarded; unknown layouts land in the explicit ``unclassified``
  category); the categories sum to the project totals and an
  INDEPENDENT filesystem oracle reconciles both totals and per-category
  numbers.
* RETENTION reuse — every per-model row carries that model's M62
  ``checkpoint_retention_overview`` aggregates VERBATIM (one retention
  analysis, never a second); project aggregates are deterministic sums
  over the rows; ``reclaimable`` counts ONLY currently-deletable
  checkpoint artifact sets.
* READ-ONLY — zero mutation (byte-level sha256 inventories before and
  after), zero tmp residue, deterministic byte-identical repeats.
* HONEST corruption accounting — integrity-failed checkpoints are
  never reclaimable, listing-invisible checkpoint directories stay in
  the physical category totals, and a model with an unparseable
  manifest drops its row while its files stay counted.
"""
from __future__ import annotations

import hashlib
import random
import zlib
from pathlib import Path

import pytest

from app.engine import ModelForge
from app.schemas import (
    EvaluationConfig,
    ModelCreateRequest,
    SampleGenerateRequest,
    SampleStrategy,
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


# --------------------------------------------------------------------------- #
# INDEPENDENT oracles (deliberately NOT the engine's code paths)
# --------------------------------------------------------------------------- #

_ORACLE_EVIDENCE = {"evaluations", "comparisons", "gates", "workflows"}
_ORACLE_ROOT_FAMILIES = {
    "datasets", "tokenizers", "suite-runs", "samples",
    "sample-evaluations", "policies", "probe-suites", "workflow-recipes",
}


def _walk(root: Path) -> dict[str, int]:
    """Independent physical walk: rel-posix -> size (the M63 boundary:
    everything under root except tmp/ scratch and hidden entries)."""
    out: dict[str, int] = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if rel.parts[0] == "tmp":
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        if p.is_file():
            out[rel.as_posix()] = p.stat().st_size
    return out


def _classify(rel: str) -> str:
    """Independent category classifier (mirrors the documented taxonomy,
    not the engine's implementation)."""
    parts = rel.split("/")
    if parts[0] == "models":
        if len(parts) == 3:
            return "models"
        if len(parts) >= 4:
            if parts[2] == "checkpoints":
                return "checkpoints"
            if parts[2] in _ORACLE_EVIDENCE:
                return "model_records"
        return "unclassified"
    if rel == "project.json":
        return "project"
    if parts[0] in _ORACLE_ROOT_FAMILIES:
        return parts[0].replace("-", "_")
    return "unclassified"


def _oracle(root: Path) -> tuple[int, int, dict[str, list[int]]]:
    files = _walk(root)
    cats: dict[str, list[int]] = {}
    for rel, size in files.items():
        cats.setdefault(_classify(rel), [0, 0])
        cats[_classify(rel)][0] += 1
        cats[_classify(rel)][1] += size
    return len(files), sum(files.values()), cats


def _inventory(root: Path) -> dict[str, str]:
    """rel-posix -> sha256 (the zero-mutation proof)."""
    return {
        rel: hashlib.sha256(p.read_bytes()).hexdigest()
        for rel, p in (
            (r.as_posix(), r) for r in root.rglob("*")
            if r.is_file() and r.relative_to(root).parts[0] != "tmp"
        )
    }


# --------------------------------------------------------------------------- #
# Multi-model fixture: two models, different storage + retention states
# --------------------------------------------------------------------------- #

ALPHA = "m63-alpha"      # 3 checkpoints + an evaluation reference on latest
BETA = "m63-beta"        # 2 checkpoints + a sample + sample-quality record


class Store:
    def __init__(self, root: Path):
        self.forge = ModelForge(root=root)
        self.root = root
        self.ds = None
        self.tok = None
        self.alpha = None
        self.beta = None

    def train(self, model_id: str, steps: int, seed: int):
        return self.forge.run_training(TrainingConfig(
            method="continued_pretraining", model_id=model_id,
            dataset_id=self.ds, tokenizer_id=self.tok,
            learning_rate=3e-3, batch_size=8, max_seq_len=32,
            eval_every_steps=4, keep_best=False, seed=seed, steps=steps))


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("m63-root"))
    up = s.forge.upload_dataset([("m63.txt", _corpus(190, "m63"))],
                                name="m63-ds")
    s.ds = up["dataset_id"]
    tok = s.forge.train_tokenizer(
        TokenizerConfig(name="m63-tok", vocab_size=320), dataset_id=s.ds)
    s.tok = tok.id
    s.forge.tokenize_dataset(s.ds, s.tok)

    s.alpha = s.forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name=ALPHA, vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
        seed=1)))[0].id
    s.train(s.alpha, steps=12, seed=1)
    latest_a = s.forge.get_model(s.alpha).latest_checkpoint
    s.forge.run_evaluation(EvaluationConfig(
        model_id=s.alpha, dataset_id=s.ds, tokenizer_id=s.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=latest_a))

    # BETA: a different architecture (different weights/checkpoint bytes)
    # and a different retention state (sample + sample-quality references
    # instead of an evaluation reference)
    s.beta = s.forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name=BETA, vocab_size=640, context_length=64, hidden_size=96,
        n_layers=3, n_heads=6, n_kv_heads=3, intermediate_size=192,
        seed=7)))[0].id
    s.train(s.beta, steps=8, seed=3)
    best_b = s.forge.select_best_checkpoint(s.beta).checkpoint.checkpoint_id
    sample = s.forge.generate_sample(SampleGenerateRequest(
        model_id=s.beta, checkpoint_id=best_b, tokenizer_id=s.tok,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    s.forge.evaluate_sample(s.beta, sample.sample_id)
    return s


# --------------------------------------------------------------------------- #
# Engine: oracle reconciliation, M62 parity, determinism, zero writes
# --------------------------------------------------------------------------- #

def test_m63_reconciles_with_oracle_and_m62(store):
    root = store.root
    before = _inventory(root)
    o1 = store.forge.project_storage_overview()
    o2 = store.forge.project_storage_overview()
    after = _inventory(root)

    # determinism + zero mutation
    assert o1.model_dump(mode="json") == o2.model_dump(mode="json")
    assert before == after
    assert not list((root / "tmp").iterdir())

    # physical oracle: totals, per-category, partition
    n_files, n_bytes, cats = _oracle(root)
    assert o1.total_files == n_files
    assert o1.total_bytes == n_bytes
    assert [c.name for c in o1.categories] == \
        list(ModelForge.PROJECT_STORAGE_CATEGORIES)
    for c in o1.categories:
        assert [c.files, c.bytes] == cats.get(c.name, [0, 0])
    assert sum(c.files for c in o1.categories) == o1.total_files
    assert sum(c.bytes for c in o1.categories) == o1.total_bytes
    assert cats.get("unclassified", [0, 0]) == [0, 0]

    # registry rows: order, count, M62 parity per model
    registry = store.forge.list_models()
    assert [r.model_id for r in o1.models] == [r.id for r in registry]
    assert o1.model_count == len(registry) == 2
    assert [r.name for r in o1.models] == [ALPHA, BETA]
    for row in o1.models:
        m = store.forge.checkpoint_retention_overview(row.model_id)
        assert (row.checkpoint_count, row.deletable_checkpoints,
                row.protected_checkpoints, row.total_checkpoint_bytes,
                row.reclaimable_checkpoint_bytes) == \
            (m.total_checkpoints, m.deletable_checkpoints,
             m.protected_checkpoints, m.total_checkpoint_bytes,
             m.reclaimable_checkpoint_bytes)
        assert row.protected_checkpoint_bytes == sum(
            e.size_bytes for e in m.checkpoints if not e.deletable)
        assert row.total_model_bytes == (row.model_bytes + row.records_bytes
                                         + row.total_checkpoint_bytes)
        # per-model physical attribution straight from the oracle walk
        files = _walk(root)
        assert row.model_bytes == sum(
            s for r, s in files.items()
            if r.split("/")[:2] == ["models", row.model_id]
            and len(r.split("/")) == 3)
        assert row.records_bytes == sum(
            s for r, s in files.items()
            if r.split("/")[:2] == ["models", row.model_id]
            and len(r.split("/")) >= 4
            and r.split("/")[2] in _ORACLE_EVIDENCE)

    # project aggregates are exact sums over the rows
    for field in ("checkpoint_count", "deletable_checkpoints",
                  "protected_checkpoints", "total_checkpoint_bytes",
                  "reclaimable_checkpoint_bytes", "protected_checkpoint_bytes"):
        assert getattr(o1, field) == sum(getattr(r, field) for r in o1.models)

    # healthy storage: the physical checkpoints category equals the
    # registry-visible M62 sum (no orphaned checkpoint directories)
    by_cat = {c.name: c for c in o1.categories}
    assert by_cat["checkpoints"].bytes == o1.total_checkpoint_bytes
    assert by_cat["checkpoints"].files == 2 * o1.checkpoint_count

    # the two models really differ (multi-model fixture is not degenerate)
    rows = {r.name: r for r in o1.models}
    assert rows[ALPHA].checkpoint_count == 3
    assert rows[BETA].checkpoint_count == 2
    assert rows[ALPHA].total_checkpoint_bytes != rows[BETA].total_checkpoint_bytes


def test_m63_multi_model_aggregation_and_shared_storage(store):
    o = store.forge.project_storage_overview()
    rows = {r.model_id: r for r in o.models}
    a, b = rows[store.alpha], rows[store.beta]
    n_files, n_bytes, cats = _oracle(store.root)

    # retention states differ: ALPHA is protected by best/published/eval
    # reference; BETA's sampled checkpoint carries sample references
    m_a = store.forge.checkpoint_retention_overview(store.alpha)
    m_b = store.forge.checkpoint_retention_overview(store.beta)
    best_a = store.forge.select_best_checkpoint(store.alpha).checkpoint.checkpoint_id
    latest_a = store.forge.get_model(store.alpha).latest_checkpoint
    a_entry = {e.checkpoint_id: e for e in m_a.checkpoints}
    assert not a_entry[best_a].deletable
    assert not a_entry[latest_a].deletable
    best_b = store.forge.select_best_checkpoint(store.beta).checkpoint.checkpoint_id
    sampled = [e for e in m_b.checkpoints
               if any(bl.reason == "sample_reference" for bl in e.blockers)]
    assert [e.checkpoint_id for e in sampled] == [best_b]
    assert not sampled[0].deletable

    # project totals are the exact sum of the per-model M62 views
    assert o.checkpoint_count == m_a.total_checkpoints + m_b.total_checkpoints
    assert o.deletable_checkpoints == (m_a.deletable_checkpoints
                                       + m_b.deletable_checkpoints)
    assert o.protected_checkpoints == (m_a.protected_checkpoints
                                       + m_b.protected_checkpoints)
    assert o.total_checkpoint_bytes == (m_a.total_checkpoint_bytes
                                        + m_b.total_checkpoint_bytes)
    assert o.reclaimable_checkpoint_bytes == (m_a.reclaimable_checkpoint_bytes
                                              + m_b.reclaimable_checkpoint_bytes)

    # §13 ownership identity: project bytes == model-scoped bytes of BOTH
    # models + shared project storage (dataset/tokenizer/recipe/project)
    shared = sum(cats.get(k, [0, 0])[1] for k in
                 ("datasets", "tokenizers", "workflow_recipes", "project",
                  "suite_runs", "samples", "sample_evaluations", "policies",
                  "probe_suites", "unclassified"))
    assert o.total_bytes == a.total_model_bytes + b.total_model_bytes + shared

    # shared physical artifacts are counted EXACTLY ONCE: the dataset and
    # tokenizer categories are the single physical trees BOTH models use
    by_cat = {c.name: c for c in o.categories}
    ds_files = [q for q in (store.root / "datasets").rglob("*") if q.is_file()]
    assert by_cat["datasets"].files == len(ds_files) == cats["datasets"][0]
    assert by_cat["datasets"].bytes == cats["datasets"][1]
    assert by_cat["tokenizers"].bytes == cats["tokenizers"][1]
    # samples + sample-quality evidence are root-level families (M15/M16)
    assert by_cat["samples"].files == 1
    assert by_cat["sample_evaluations"].files == 1


def test_m63_empty_project(forge):
    ov = forge.project_storage_overview()
    ov2 = forge.project_storage_overview()
    n_files, n_bytes, cats = _oracle(forge.storage.root)
    assert ov.model_dump(mode="json") == ov2.model_dump(mode="json")
    assert ov.model_count == 0 and ov.models == []
    assert ov.checkpoint_count == 0
    assert ov.total_files == n_files == 1        # just project.json
    assert ov.total_bytes == n_bytes
    assert ov.total_checkpoint_bytes == 0
    assert ov.reclaimable_checkpoint_bytes == 0
    assert ov.protected_checkpoint_bytes == 0
    for c in ov.categories:
        expect = cats.get(c.name, [0, 0])
        assert [c.files, c.bytes] == expect
    assert cats["project"][0] == 1               # never manufactured storage


def test_m63_unclassified_and_boundary(forge):
    root = forge.storage.root
    (root / "stray.txt").write_bytes(b"unaccounted")
    (root / "models" / "m-abc").mkdir(parents=True, exist_ok=True)
    (root / "models" / "m-abc" / "manifest.json").write_bytes(b"{}")
    (root / "models" / "m-abc" / "notes").mkdir()
    (root / "models" / "m-abc" / "notes" / "x.txt").write_bytes(b"deep")
    (root / "tmp" / "scratch.bin").write_bytes(b"excluded")
    (root / ".hidden").write_bytes(b"excluded")

    ov = forge.project_storage_overview()
    n_files, n_bytes, cats = _oracle(root)
    assert ov.total_files == n_files and ov.total_bytes == n_bytes
    by_cat = {c.name: c for c in ov.categories}
    # the stray root file + the unknown nested dir are explicit, never silent
    assert [by_cat["unclassified"].files, by_cat["unclassified"].bytes] == \
        cats["unclassified"]
    assert by_cat["unclassified"].files == 2
    # a dir that LOOKS like a model but is not in the registry: physical
    # bytes counted, no row manufactured
    assert by_cat["models"].bytes == cats["models"][1]
    assert ov.model_count == 0
    # tmp/ scratch and hidden entries are OUTSIDE the boundary
    assert ov.total_bytes == n_bytes
    assert "scratch.bin" not in _walk(root) and ".hidden" not in _walk(root)
    # categories still partition the totals exactly
    assert sum(c.files for c in ov.categories) == ov.total_files
    assert sum(c.bytes for c in ov.categories) == ov.total_bytes


def test_m63_corruption_accounting(store):
    """Honest accounting under corruption — and a full restore."""
    forge, root = store.forge, store.root
    baseline = forge.project_storage_overview().model_dump(mode="json")
    base_files = _walk(root)
    base_a = {r["model_id"]: r for r in baseline["models"]}[store.alpha]
    base_b = {r["model_id"]: r for r in baseline["models"]}[store.beta]

    # (1) integrity failure: overwrite a deletable ALPHA checkpoint's
    # weights with same-length garbage (torch.load fails -> the M3
    # verifier refuses; physical totals must not move)
    m_a = forge.checkpoint_retention_overview(store.alpha)
    victim = next(e for e in m_a.checkpoints if e.deletable)
    vdir = root / "models" / store.alpha / "checkpoints" / victim.checkpoint_id
    original_weights = (vdir / "weights.pt").read_bytes()
    (vdir / "weights.pt").write_bytes(b"\x00" * len(original_weights))

    o1 = forge.project_storage_overview()
    row1 = {r.model_id: r for r in o1.models}[store.alpha]
    m_a2 = forge.checkpoint_retention_overview(store.alpha)
    v2 = {e.checkpoint_id: e for e in m_a2.checkpoints}[victim.checkpoint_id]
    # M61/M62 semantics: an integrity-failed checkpoint is NEVER deletable
    assert v2.integrity_verified is False and v2.deletable is False
    # reclaimable drops by EXACTLY the victim's size, protected grows by it
    assert row1.reclaimable_checkpoint_bytes == \
        base_a["reclaimable_checkpoint_bytes"] - victim.size_bytes
    assert row1.protected_checkpoint_bytes == \
        base_a["protected_checkpoint_bytes"] + victim.size_bytes
    assert row1.checkpoint_count == base_a["checkpoint_count"]
    # physical accounting unchanged (same-size corruption)
    assert o1.total_files == len(base_files)
    assert o1.total_bytes == sum(base_files.values())
    cat1 = {c.name: c for c in o1.categories}
    assert cat1["checkpoints"].bytes == baseline["total_checkpoint_bytes"]

    # (2) listing-invisible checkpoint: remove BETA's one deletable
    # checkpoint's manifest — M62 no longer lists it, but the physical
    # walk still counts its weights (never silently discarded)
    m_b = forge.checkpoint_retention_overview(store.beta)
    victim2 = next(e for e in m_b.checkpoints if e.deletable)
    vdir2 = root / "models" / store.beta / "checkpoints" / victim2.checkpoint_id
    original_manifest = (vdir2 / "manifest.json").read_bytes()
    manifest_size = len(original_manifest)
    (vdir2 / "manifest.json").unlink()

    o2 = forge.project_storage_overview()
    row2 = {r.model_id: r for r in o2.models}[store.beta]
    m_b2 = forge.checkpoint_retention_overview(store.beta)
    assert row2.checkpoint_count == m_b2.total_checkpoints
    assert row2.checkpoint_count == base_b["checkpoint_count"] - 1
    assert victim2.checkpoint_id not in {e.checkpoint_id
                                         for e in m_b2.checkpoints}
    # registry-visible bytes dropped by the FULL artifact set; the
    # physical category only lost the manifest file — the orphan weights
    # remain counted (the honest divergence, visible not hidden)
    assert row2.total_checkpoint_bytes == \
        base_b["total_checkpoint_bytes"] - victim2.size_bytes
    cat2 = {c.name: c for c in o2.categories}
    assert cat2["checkpoints"].bytes == \
        cat1["checkpoints"].bytes - manifest_size
    assert o2.total_files == len(base_files) - 1
    assert o2.total_bytes == sum(base_files.values()) - manifest_size
    n_files_o, n_bytes_o, cats_o = _oracle(root)
    assert o2.total_files == n_files_o and o2.total_bytes == n_bytes_o

    # (3) unparseable MODEL manifest: the row disappears (the registry
    # convention) but every physical byte stays counted
    bm = root / "models" / store.beta / "manifest.json"
    beta_manifest = bm.read_bytes()
    bm.write_bytes(b"not json at all")
    o3 = forge.project_storage_overview()
    assert o3.model_count == 1
    assert [r.model_id for r in o3.models] == [store.alpha]
    n_files_o, n_bytes_o, cats_o = _oracle(root)
    assert o3.total_files == n_files_o and o3.total_bytes == n_bytes_o
    for c in o3.categories:
        assert [c.files, c.bytes] == cats_o.get(c.name, [0, 0])

    # full restore -> byte-identical recompute of the baseline overview
    (vdir / "weights.pt").write_bytes(original_weights)
    (vdir2 / "manifest.json").write_bytes(original_manifest)
    bm.write_bytes(beta_manifest)
    o4 = forge.project_storage_overview()
    assert o4.model_dump(mode="json") == baseline


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m63_api_project_storage(api_client):
    import json as _json

    # a model with checkpoints + an evaluation reference, through the
    # public routes only
    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m63api.txt", _corpus(170, "m63api"), "text/plain"))],
        data={"name": "m63api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m63api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m63api-model", "vocab_size": 640,
                   "context_length": 64, "hidden_size": 64, "n_layers": 2,
                   "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128},
        "description": "m63 api test"})
    assert m.status_code == 201, m.text
    mid = m.json()["model"]["id"]
    r = api_client.post("/api/v1/training/run", json={
        "method": "continued_pretraining", "model_id": mid, "dataset_id": ds,
        "tokenizer_id": tok, "learning_rate": 3e-3, "batch_size": 8,
        "max_seq_len": 32, "eval_every_steps": 4, "keep_best": False,
        "seed": 1, "steps": 12})
    assert r.status_code == 200, r.text
    latest = api_client.get(f"/api/v1/models/{mid}").json()["latest_checkpoint"]
    r = api_client.post("/api/v1/evaluations/run", json={
        "model_id": mid, "dataset_id": ds, "tokenizer_id": tok,
        "split": "validation", "batch_size": 8, "max_seq_len": 32,
        "seed": 2, "checkpoint_id": latest})
    assert r.status_code == 200, r.text

    storage_root = Path(api_client.get("/api/v1/project").json()["storage_root"])
    before = _inventory(storage_root)

    r = api_client.get("/api/v1/project/storage")
    assert r.status_code == 200, r.text
    body = r.content
    ov = r.json()
    assert set(ov) == {"total_files", "total_bytes", "model_count",
                       "checkpoint_count", "deletable_checkpoints",
                       "protected_checkpoints", "total_checkpoint_bytes",
                       "reclaimable_checkpoint_bytes",
                       "protected_checkpoint_bytes", "categories", "models"}

    # physical oracle over the LIVE session root
    n_files, n_bytes, cats = _oracle(storage_root)
    assert ov["total_files"] == n_files
    assert ov["total_bytes"] == n_bytes
    for c in ov["categories"]:
        assert [c["files"], c["bytes"]] == cats.get(c["name"], [0, 0])
        assert set(c) == {"name", "files", "bytes"}
    assert sum(c["files"] for c in ov["categories"]) == ov["total_files"]
    assert sum(c["bytes"] for c in ov["categories"]) == ov["total_bytes"]
    assert cats.get("unclassified", [0, 0]) == [0, 0]

    # rows: registry order/identity through the public listing route
    listing = api_client.get("/api/v1/models").json()
    assert ov["model_count"] == len(listing)
    assert [r["model_id"] for r in ov["models"]] == \
        [m["id"] for m in listing]
    for row in ov["models"]:
        assert set(row) == {"model_id", "name", "created_at", "model_bytes",
                            "records_bytes", "checkpoint_count",
                            "deletable_checkpoints", "protected_checkpoints",
                            "total_checkpoint_bytes",
                            "reclaimable_checkpoint_bytes",
                            "protected_checkpoint_bytes", "total_model_bytes"}
        # THE aggregation invariant: the row is EXACTLY that model's own
        # M62 overview (queried through the public route)
        m62 = api_client.get(
            f"/api/v1/models/{row['model_id']}/checkpoints/retention").json()
        assert (row["checkpoint_count"], row["deletable_checkpoints"],
                row["protected_checkpoints"], row["total_checkpoint_bytes"],
                row["reclaimable_checkpoint_bytes"]) == \
            (m62["total_checkpoints"], m62["deletable_checkpoints"],
             m62["protected_checkpoints"], m62["total_checkpoint_bytes"],
             m62["reclaimable_checkpoint_bytes"])
        assert row["protected_checkpoint_bytes"] == sum(
            e["size_bytes"] for e in m62["checkpoints"] if not e["deletable"])
        assert row["total_model_bytes"] == (row["model_bytes"]
                                            + row["records_bytes"]
                                            + row["total_checkpoint_bytes"])

    # project aggregates are exact sums over the rows
    for field in ("checkpoint_count", "deletable_checkpoints",
                  "protected_checkpoints", "total_checkpoint_bytes",
                  "reclaimable_checkpoint_bytes", "protected_checkpoint_bytes"):
        assert ov[field] == sum(r[field] for r in ov["models"])

    # reclaimable NEVER includes non-checkpoint storage
    assert ov["reclaimable_checkpoint_bytes"] <= ov["total_checkpoint_bytes"]
    assert ov["reclaimable_checkpoint_bytes"] < ov["total_bytes"]

    # M52/M60 consistency for the freshly trained model: the M52 best,
    # the M60 published state and the evaluation-referenced checkpoint
    # are all non-deletable, whatever their overlap
    my_row = next(r for r in ov["models"] if r["model_id"] == mid)
    m62 = api_client.get(
        f"/api/v1/models/{mid}/checkpoints/retention").json()
    best = api_client.get(
        f"/api/v1/models/{mid}/checkpoints/best").json()["checkpoint"]["checkpoint_id"]
    by_id = {e["checkpoint_id"]: e for e in m62["checkpoints"]}
    assert not by_id[best]["deletable"]
    assert not by_id[latest]["deletable"]
    expected_protected = {e["checkpoint_id"] for e in m62["checkpoints"]
                          if not e["deletable"]}
    assert {best, latest} <= expected_protected
    assert my_row["protected_checkpoints"] == len(expected_protected)
    assert my_row["deletable_checkpoints"] == \
        len(m62["checkpoints"]) - len(expected_protected)

    # determinism over HTTP + zero mutation
    r2 = api_client.get("/api/v1/project/storage")
    assert r2.status_code == 200 and r2.content == body
    assert _inventory(storage_root) == before
    assert not list((storage_root / "tmp").iterdir())

    # OpenAPI: exactly one new path (86 -> 87), only GET, schemas exposed
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 87
    NEW = "/api/v1/project/storage"
    assert set(spec["paths"][NEW].keys()) == {"get"}
    assert spec["paths"][NEW]["get"]["tags"] == ["meta"]
    for s in ("ProjectStorageOverview", "ProjectModelStorageSummary",
              "ProjectStorageCategory"):
        assert s in spec["components"]["schemas"]
    keys = list(spec["paths"].keys())
    assert keys.index(NEW) > keys.index("/api/v1/project")
    assert keys.index(NEW) < keys.index("/api/v1/system")
