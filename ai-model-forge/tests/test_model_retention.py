"""Milestone 67 tests: explicit VERIFIED model retention.

The M67 invariants under test:

* HEADLINE INVARIANT — the deletion guard refuses on EXACTLY the
  EXTERNAL (root-level) references the M66 usage overview reports:
  same categories, same persisted reference ids, same canonical
  order. Nothing protected that is not shown; nothing shown that is
  not protected. Verified bidirectionally against an INDEPENDENT
  manifest-parsing oracle (never the app's own analysis).
* OWNERSHIP — the INTERNAL model-scoped families (training runs,
  checkpoints, workflows, evaluations, comparisons, gate decisions)
  live inside models/<id>/ and are removed atomically WITH the model:
  they never block, and a model with rich internal history but zero
  external references IS deletable.
* INTEGRITY FIRST — scope (unknown or registry-invisible -> 404,
  nothing deleted) -> verification (corrupt / missing weights / hash
  mismatch -> refuse, never a silent rmtree) -> blockers -> ONE
  atomic removal with deterministic file/byte stats.
* NO CASCADE — a protected DELETE changes nothing on disk; a
  successful DELETE removes ONLY the model's own directory.
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
    PolicyCreateRequest,
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
    """rel-posix -> sha256 (the zero-mutation / isolation proof)."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


# --------------------------------------------------------------------------- #
# INDEPENDENT oracle: raw manifest parsing of the EXTERNAL families
# (the blocking subset; never the app's own analysis)
# --------------------------------------------------------------------------- #

EXTERNAL_FAMILIES = {  # category -> (root dir, id field, model-id path)
    "suite_run": ("suite-runs", "suite_run_id", ("model_id",)),
    "sample": ("samples", "sample_id", ("model_id",)),
    "sample_quality": ("sample-evaluations", "evaluation_id",
                       ("model_id",)),
    "workflow_recipe": ("workflow-recipes", "recipe_id", None),  # stage scan
    "policy": ("policies", "policy_id", ("policy", "model_id")),
}


def _oracle_external(root: Path, model_id: str) -> dict[str, list[str]]:
    """category -> sorted reference ids, derived from raw manifests."""
    out: dict[str, list[str]] = {c: [] for c in EXTERNAL_FAMILIES}
    for cat, (fam, idfield, midpath) in EXTERNAL_FAMILIES.items():
        # samples/ and sample-evaluations/ are model-scoped one level
        # deeper: <root>/<family>/<model_id>/<record>/manifest.json
        if cat in ("sample", "sample_quality"):
            candidates = [d for d in sorted((root / fam / model_id)
                                            .iterdir())] if (
                root / fam / model_id).exists() else []
        else:
            candidates = [d for d in sorted((root / fam).iterdir())] if (
                root / fam).exists() else []
        for d in candidates:
            if not d.is_dir():
                continue
            try:
                rec = json.loads((d / "manifest.json").read_text())
            except Exception:
                continue
            if midpath is None:  # recipe: scan stage configs for the id
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
                hit = model_id in refs
            else:
                val = rec
                for part in midpath:
                    val = val.get(part, {})
                hit = val == model_id
            if hit:
                out[cat].append(rec[idfield])
    return {c: sorted(v) for c, v in out.items()}


def _blocker_pairs(blockers) -> list[tuple[str, str]]:
    return [(b.category, b.reference_id) for b in blockers]


# --------------------------------------------------------------------------- #
# Multi-reference fixture
# --------------------------------------------------------------------------- #

class Env:
    def __init__(self, root: Path):
        self.forge = ModelForge(root=root)
        self.root = root
        self.ds = None
        self.tok = None
        self.rich = None       # all ELEVEN categories -> blocked
        self.internal = None   # all six INTERNAL families, no external
        self.suite_only = None   # one suite run
        self.recipe_only = None  # one model-bound recipe
        self.sample_only = None  # one sample + its quality measurement
        self.policy_only = None  # one model-bound gate policy
        self.fresh = None        # nothing at all


def _train(forge, model_id, ds, tok, seed, steps=8):
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=model_id, dataset_id=ds,
        tokenizer_id=tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=seed,
        steps=steps))


def _make_model(forge, name, seed):
    return forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name=name, vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
        seed=seed)))[0].id


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m67-root"))
    forge = e.forge
    up = forge.upload_dataset([("m67.txt", _corpus(190, "m67"))],
                              name="m67-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m67-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)

    # ---- rich model: every category, internal AND external ---------- #
    e.rich = _make_model(forge, "m67-rich-model", 1)
    _train(forge, e.rich, e.ds, e.tok, seed=1)
    ck = [c.checkpoint_id for c in forge.list_checkpoints(e.rich)]
    forge.run_evaluation(EvaluationConfig(
        model_id=e.rich, dataset_id=e.ds, tokenizer_id=e.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=ck[0]))
    forge.run_comparison(ComparisonRequest(
        model_id=e.rich,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ck[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ck[1]),
        dataset_id=e.ds, split="validation", tokenizer_id=e.tok,
        batch_size=8, max_seq_len=32, seed=3, tolerance=1e-4))
    forge.run_gate(GateRequest(
        model_id=e.rich,
        policy=GatePolicy(
            name="m67-gate", model_id=e.rich, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=4, baseline_type="checkpoint",
            baseline_checkpoint_id=ck[1], tolerance=1.0),
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=ck[0])))
    forge.run_workflow(WorkflowPlan(
        name="m67-rich-workflow", model_id=e.rich,
        description="M67 fixture",
        stages=[
            WorkflowStage(stage_id="tr", type=StageType.TRAIN,
                          training=TrainingConfig(
                              method="continued_pretraining",
                              model_id=e.rich, dataset_id=e.ds,
                              tokenizer_id=e.tok, learning_rate=3e-3,
                              batch_size=8, max_seq_len=32,
                              eval_every_steps=4, keep_best=False, seed=5,
                              steps=4)),
            WorkflowStage(stage_id="ev", type=StageType.EVALUATE,
                          evaluation=WorkflowEvaluationStage(
                              config=EvaluationConfig(
                                  model_id=e.rich, dataset_id=e.ds,
                                  tokenizer_id=e.tok, split="validation",
                                  batch_size=8, max_seq_len=32),
                              checkpoint_from_best=True)),
        ]))
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m67-suite", description="M67 fixture",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))
    forge.run_suite(SuiteRunRequest(
        model_id=e.rich, suite_id="m67-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=ck[0])))
    best = forge.select_best_checkpoint(e.rich).checkpoint.checkpoint_id
    sample = forge.generate_sample(SampleGenerateRequest(
        model_id=e.rich, checkpoint_id=best, tokenizer_id=e.tok,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    forge.evaluate_sample(e.rich, sample.sample_id)
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m67-rich-recipe", description="M67 fixture",
        stages=[WorkflowStage(
            stage_id="tr", type=StageType.TRAIN,
            training=TrainingConfig(
                method="continued_pretraining", model_id=e.rich,
                dataset_id=e.ds, tokenizer_id=e.tok, learning_rate=3e-3,
                batch_size=8, max_seq_len=32, eval_every_steps=4,
                keep_best=False, seed=9, steps=4))]))
    forge.register_policy(PolicyCreateRequest(
        policy_id="m67-rich-policy", description="M67 fixture",
        policy=GatePolicy(
            name="m67-rich-policy", model_id=e.rich, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=10, baseline_type="current",
            tolerance=1.0)))

    # ---- internal-only model: ALL SIX internal families, no external - #
    e.internal = _make_model(forge, "m67-internal-model", 2)
    _train(forge, e.internal, e.ds, e.tok, seed=11, steps=8)
    ick = [c.checkpoint_id for c in forge.list_checkpoints(e.internal)]
    forge.run_evaluation(EvaluationConfig(
        model_id=e.internal, dataset_id=e.ds, tokenizer_id=e.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=12,
        checkpoint_id=ick[0]))
    forge.run_comparison(ComparisonRequest(
        model_id=e.internal,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ick[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ick[1]),
        dataset_id=e.ds, split="validation", tokenizer_id=e.tok,
        batch_size=8, max_seq_len=32, seed=13, tolerance=1e-4))
    forge.run_gate(GateRequest(
        model_id=e.internal,
        policy=GatePolicy(
            name="m67-internal-gate", model_id=e.internal, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=14, baseline_type="checkpoint",
            baseline_checkpoint_id=ick[1], tolerance=1.0),
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=ick[0])))
    forge.run_workflow(WorkflowPlan(
        name="m67-internal-workflow", model_id=e.internal,
        description="M67 fixture",
        stages=[WorkflowStage(
            stage_id="tr", type=StageType.TRAIN,
            training=TrainingConfig(
                method="continued_pretraining", model_id=e.internal,
                dataset_id=e.ds, tokenizer_id=e.tok, learning_rate=3e-3,
                batch_size=8, max_seq_len=32, eval_every_steps=4,
                keep_best=False, seed=15, steps=4))]))

    # ---- one-blocker models ----------------------------------------- #
    e.suite_only = _make_model(forge, "m67-suite-model", 3)
    forge.run_suite(SuiteRunRequest(
        model_id=e.suite_only, suite_id="m67-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    e.recipe_only = _make_model(forge, "m67-recipe-model", 4)
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m67-recipe-only", description="M67 fixture",
        stages=[WorkflowStage(
            stage_id="tr", type=StageType.TRAIN,
            training=TrainingConfig(
                method="continued_pretraining", model_id=e.recipe_only,
                dataset_id=e.ds, tokenizer_id=e.tok, learning_rate=3e-3,
                batch_size=8, max_seq_len=32, eval_every_steps=4,
                keep_best=False, seed=16, steps=4))]))
    e.sample_only = _make_model(forge, "m67-sample-model", 5)
    _train(forge, e.sample_only, e.ds, e.tok, seed=17, steps=4)
    sbest = forge.select_best_checkpoint(e.sample_only).checkpoint.checkpoint_id
    smp = forge.generate_sample(SampleGenerateRequest(
        model_id=e.sample_only, checkpoint_id=sbest, tokenizer_id=e.tok,
        prompt="cloud forest", strategy=SampleStrategy.GREEDY,
        max_new_tokens=4))
    forge.evaluate_sample(e.sample_only, smp.sample_id)
    e.policy_only = _make_model(forge, "m67-policy-model", 6)
    forge.register_policy(PolicyCreateRequest(
        policy_id="m67-policy-only", description="M67 fixture",
        policy=GatePolicy(
            name="m67-policy", model_id=e.policy_only, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=18, baseline_type="current",
            tolerance=1.0)))

    # ---- fresh model: nothing at all --------------------------------- #
    e.fresh = _make_model(forge, "m67-fresh-model", 7)
    return e


# --------------------------------------------------------------------------- #
# The headline invariant (engine)
# --------------------------------------------------------------------------- #

def test_m67_headline_invariant(env):
    forge, root = env.forge, env.root
    before = _inventory(root)
    usage = forge.model_usage_overview(env.rich)
    retention = forge.model_retention_overview(env.rich)
    blockers = forge.model_deletion_blockers(env.rich)

    # determinism + zero mutation of the read-only surface
    assert forge.model_usage_overview(env.rich).model_dump(
        mode="json") == usage.model_dump(mode="json")
    assert forge.model_retention_overview(env.rich).model_dump(
        mode="json") == retention.model_dump(mode="json")
    assert _blocker_pairs(forge.model_deletion_blockers(env.rich)) == \
        _blocker_pairs(blockers)
    assert _inventory(root) == before

    # the rich model has EVERY internal category non-empty ...
    ucats = {c.category: c.references for c in usage.categories}
    for cat in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES:
        assert ucats[cat], cat
    # ... and every external category non-empty
    ext = [c for c in ModelForge.MODEL_USAGE_CATEGORIES
           if c not in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES]
    for cat in ext:
        assert ucats[cat], cat

    # INVARIANT (both directions): blockers == EXACTLY the M66-visible
    # external references, in canonical category order, sorted ids
    expected = [(cat, ref) for cat in ext for ref in ucats[cat]]
    assert _blocker_pairs(blockers) == expected
    # every blocker corresponds to an M66-visible reference ...
    for b in blockers:
        assert b.reference_id in ucats[b.category]
        assert b.category not in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES
        assert b.detail  # short authoritative identifying info present
    # ... and every M66-visible blocking reference appears in blockers
    assert len(_blocker_pairs(blockers)) == len(expected) == \
        len(set(expected))
    # NO internal reference ever blocks (ownership, not dependency)
    internal_ids = {ref for cat in
                    ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES
                    for ref in ucats[cat]}
    assert not (internal_ids &
                {b.reference_id for b in blockers})

    # the INDEPENDENT oracle agrees — on the usage surface (external
    # subset) AND on the blockers
    oracle = _oracle_external(root, env.rich)
    for cat in ext:
        assert ucats[cat] == oracle[cat], cat
    oracle_blockers = [(cat, ref) for cat in ext for ref in oracle[cat]]
    assert _blocker_pairs(blockers) == oracle_blockers

    # the retention view exposes the same decision
    assert retention.model_id == env.rich
    assert retention.integrity_verified is True
    assert retention.deletable is False
    assert retention.blockers == blockers
    mdir = root / "models" / env.rich
    walk = sorted(p.relative_to(mdir).as_posix()
                  for p in mdir.rglob("*")
                  if p.is_file() and not any(
                      part.startswith(".") for part in
                      p.relative_to(mdir).parts))
    assert retention.files == walk
    assert retention.size_bytes == sum(
        p.stat().st_size for p in mdir.rglob("*") if p.is_file())

    # the guard refuses (engine) — and nothing changed
    with pytest.raises(ValueError):
        forge.delete_model(env.rich)
    assert _inventory(root) == before


# --------------------------------------------------------------------------- #
# Per-external-category guards + ownership non-blocking + live recompute
# --------------------------------------------------------------------------- #

def test_m67_per_category_guards(env):
    forge, root = env.forge, env.root
    before = _inventory(root)

    cases = [
        (env.suite_only, [("suite_run", None)]),
        (env.recipe_only, [("workflow_recipe", ["m67-recipe-only"])]),
        (env.sample_only, [("sample", None), ("sample_quality", None)]),
        (env.policy_only, [("policy", ["m67-policy-only"])]),
    ]
    for model_id, expected_shape in cases:
        usage = forge.model_usage_overview(model_id)
        ucats = {c.category: c.references for c in usage.categories}
        blockers = forge.model_deletion_blockers(model_id)
        pairs = _blocker_pairs(blockers)
        # exactly the expected external categories, canonical order
        assert [c for c, _ in pairs] == [c for c, _ in expected_shape]
        for cat, ids in expected_shape:
            got = [r for c, r in pairs if c == cat]
            assert got == sorted(ucats[cat]) and got, (model_id, cat)
            if ids is not None:
                assert got == ids
        # every OTHER external category is empty for this model
        ext = [c for c in ModelForge.MODEL_USAGE_CATEGORIES
               if c not in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES]
        for cat in ext:
            if cat not in [c for c, _ in expected_shape]:
                assert ucats[cat] == [], (model_id, cat)
        # not deletable; the guard refuses; nothing changed
        assert forge.model_retention_overview(
            model_id).deletable is False
        with pytest.raises(ValueError):
            forge.delete_model(model_id)
        # internal references exist for some (suite run -> evaluation;
        # sample model -> training/checkpoint) but never block
        internal_ids = {ref for c in
                        ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES
                        for ref in ucats[c]}
        assert not internal_ids & {b.reference_id for b in blockers}
    assert _inventory(root) == before

    # live recompute: the fresh model is deletable, then a NEW external
    # reference (a policy) flips it to blocked — no cache anywhere, and
    # the rich model's decision is untouched (no cross-talk)
    fresh_ret = forge.model_retention_overview(env.fresh)
    assert fresh_ret.deletable is True and fresh_ret.blockers == []
    rich_before = forge.model_retention_overview(env.rich).model_dump(
        mode="json")
    forge.register_policy(PolicyCreateRequest(
        policy_id="m67-fresh-policy", description="M67 live recompute",
        policy=GatePolicy(
            name="m67-fresh-policy", model_id=env.fresh, dataset_id=env.ds,
            tokenizer_id=env.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=19, baseline_type="current",
            tolerance=1.0)))
    after = forge.model_retention_overview(env.fresh)
    assert after.deletable is False
    assert _blocker_pairs(after.blockers) == [("policy",
                                               "m67-fresh-policy")]
    # ... and the rich model's decision is untouched (no cross-talk)
    assert forge.model_retention_overview(env.rich).model_dump(
        mode="json") == rich_before


def test_m67_internal_only_model_is_deletable(env):
    forge, root = env.forge, env.root
    model_id = env.internal

    # rich INTERNAL history: all six families non-empty
    usage = forge.model_usage_overview(model_id)
    ucats = {c.category: c.references for c in usage.categories}
    for cat in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES:
        assert ucats[cat], cat
    for cat in ModelForge.MODEL_USAGE_CATEGORIES:
        if cat not in ModelForge.MODEL_USAGE_INTERNAL_CATEGORIES:
            assert ucats[cat] == [], cat
    assert usage.external_references == 0
    assert usage.externally_referenced is False

    # retention: verified integrity, zero blockers, deletable
    ret = forge.model_retention_overview(model_id)
    assert ret.integrity_verified is True
    assert ret.blockers == [] and ret.deletable is True

    # independent walk of the model's own directory (the exact stats)
    mdir = root / "models" / model_id
    expected_files = sorted(
        p.relative_to(mdir).as_posix() for p in mdir.rglob("*")
        if p.is_file() and not any(
            part.startswith(".") for part in p.relative_to(mdir).parts))
    expected_bytes = sum(p.stat().st_size for p in mdir.rglob("*")
                         if p.is_file())

    before = _inventory(root)
    result = forge.delete_model(model_id)

    # deterministic stats == the independent walk
    assert result.model_id == model_id
    assert result.files_removed == len(expected_files)
    assert result.bytes_reclaimed == expected_bytes
    assert result.files_removed == len(ret.files)
    assert result.bytes_reclaimed == ret.size_bytes

    # ONLY the model's own directory is gone; everything else
    # byte-identical; no hidden residue; registries shrink
    after = _inventory(root)
    assert set(after) == set(before) - {
        f"models/{model_id}/{rel}" for rel in expected_files}
    assert all(after[f] == before[f] for f in after)
    assert not mdir.exists()
    assert not any(p.name.startswith(".tmp-delete")
                   for p in (root / "models").iterdir())
    assert model_id not in forge.storage.model_ids()
    assert model_id not in [r.id for r in forge.list_models()]
    # the root-level families were NOT cascaded (rich model untouched)
    assert forge.model_retention_overview(env.rich).deletable is False

    # repeated deletion -> missing-resource semantics
    with pytest.raises(FileNotFoundError):
        forge.delete_model(model_id)


# --------------------------------------------------------------------------- #
# Corruption / integrity: refuse, never a silent rmtree
# --------------------------------------------------------------------------- #

def test_m67_integrity_refusals(env):
    forge, root = env.forge, env.root

    def tampered_model(name, seed, tamper):
        mid = _make_model(forge, name, seed)
        mdir = root / "models" / mid
        originals = {p: p.read_bytes() for p in mdir.rglob("*")
                     if p.is_file()}
        tamper(mdir)
        return mid, originals

    def restore(mdir, originals):
        for p, data in originals.items():
            p.write_bytes(data)

    # (1) malformed manifest -> registry-invisible -> 404 semantics
    mid, originals = tampered_model("m67-bad-manifest", 31,
                                    lambda d: (d / "manifest.json")
                                    .write_bytes(b"not json at all"))
    mdir = root / "models" / mid
    try:
        before = _inventory(root)
        with pytest.raises(FileNotFoundError):
            forge.delete_model(mid)
        assert _inventory(root) == before
        assert mdir.exists()  # nothing deleted
    finally:
        restore(mdir, originals)

    # (2) missing manifest -> same registry-invisible semantics
    mid, originals = tampered_model("m67-no-manifest", 32,
                                    lambda d: (d / "manifest.json").unlink())
    mdir = root / "models" / mid
    try:
        before = _inventory(root)
        with pytest.raises(FileNotFoundError):
            forge.delete_model(mid)
        assert _inventory(root) == before
    finally:
        restore(mdir, originals)

    # (3) missing weights -> integrity refusal (RuntimeError -> 409)
    mid, originals = tampered_model("m67-no-weights", 33,
                                    lambda d: (d / "weights.pt").unlink())
    mdir = root / "models" / mid
    try:
        before = _inventory(root)
        with pytest.raises(RuntimeError):
            forge.delete_model(mid)
        assert _inventory(root) == before
        assert mdir.exists()
        assert forge.model_retention_overview(
            mid).integrity_verified is False
        assert forge.model_retention_overview(mid).deletable is False
    finally:
        restore(mdir, originals)

    # (4) corrupted weights -> integrity refusal
    mid, originals = tampered_model(
        "m67-corrupt-weights", 34,
        lambda d: (d / "weights.pt").write_bytes(b"garbage"))
    mdir = root / "models" / mid
    try:
        before = _inventory(root)
        with pytest.raises(RuntimeError):
            forge.delete_model(mid)
        assert _inventory(root) == before
    finally:
        restore(mdir, originals)

    # (5) hash mismatch (sidecar disagrees with the weights) -> refusal
    def bad_sidecar(d):
        (d / "weights.sha256").write_text(json.dumps({"sha256": "0" * 64}))
    mid, originals = tampered_model("m67-hash-mismatch", 35, bad_sidecar)
    mdir = root / "models" / mid
    try:
        before = _inventory(root)
        with pytest.raises(RuntimeError):
            forge.delete_model(mid)
        assert _inventory(root) == before
    finally:
        restore(mdir, originals)

    # (6) invalid model metadata (parseable JSON, schema-invalid) ->
    #     registry-invisible -> 404 semantics, nothing deleted
    def bad_meta(d):
        man = json.loads((d / "manifest.json").read_text())
        man["config"] = {"utterly": "invalid"}
        (d / "manifest.json").write_text(json.dumps(man))
    mid, originals = tampered_model("m67-bad-metadata", 36, bad_meta)
    mdir = root / "models" / mid
    try:
        before = _inventory(root)
        with pytest.raises(FileNotFoundError):
            forge.delete_model(mid)
        assert _inventory(root) == before
    finally:
        restore(mdir, originals)


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m67_api_model_retention(api_client):
    import json as _json

    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m67api.txt", _corpus(170, "m67api"),
                          "text/plain"))],
        data={"name": "m67api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m67api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m67api-model", "vocab_size": 640,
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
    # an external reference: a registered policy naming the model
    r = api_client.post("/api/v1/policies", json={
        "policy_id": "m67api-policy",
        "description": "M67 api fixture",
        "policy": {"name": "m67api-policy", "model_id": mid,
                   "dataset_id": ds, "tokenizer_id": tok,
                   "split": "validation", "batch_size": 8,
                   "max_seq_len": 32, "seed": 7,
                   "baseline_type": "current", "tolerance": 1.0}})
    assert r.status_code == 201, r.text

    storage_root = Path(api_client.get("/api/v1/project").json()[
        "storage_root"])

    # usage (M66) and retention (M67) agree; retention is deterministic
    usage = api_client.get(f"/api/v1/models/{mid}/usage").json()
    ret1 = api_client.get(f"/api/v1/models/{mid}/retention")
    assert ret1.status_code == 200, ret1.text
    ret_body = ret1.content
    ret = ret1.json()
    assert set(ret) == {"model_id", "name", "created_at", "architecture",
                        "parameter_count", "files", "size_bytes",
                        "integrity_verified", "deletable", "blockers"}
    assert ret["model_id"] == mid and ret["integrity_verified"] is True
    assert ret["deletable"] is False
    assert ret["files"] and ret["size_bytes"] > 0
    ucats = {c["category"]: c["references"] for c in usage["categories"]}
    assert [(b["category"], b["reference_id"]) for b in ret["blockers"]] \
        == [("policy", "m67api-policy")]
    assert ucats["policy"] == ["m67api-policy"]
    # internal references exist (training + checkpoints) but never block
    assert ucats["training_run"] and ucats["checkpoint"]
    assert all(b["category"] == "policy" for b in ret["blockers"])
    assert api_client.get(
        f"/api/v1/models/{mid}/retention").content == ret_body

    # protected DELETE -> structured 409, zero filesystem mutation
    before = _inventory(storage_root)
    d = api_client.delete(f"/api/v1/models/{mid}")
    assert d.status_code == 409, d.text
    detail = d.json()["detail"]
    assert set(detail) == {"message", "model_id", "protected", "blockers"}
    assert detail["model_id"] == mid and detail["protected"] is True
    assert detail["blockers"] == [
        {"category": "policy", "reference_id": "m67api-policy",
         "detail": "gate policy 'm67api-policy'"}]
    # the 409 blockers == the M66 usage external references EXACTLY
    assert [(b["category"], b["reference_id"]) for b in detail["blockers"]] \
        == [(c, r) for c in ("suite_run", "sample", "sample_quality",
                             "workflow_recipe", "policy")
            for r in ucats[c]]
    assert _inventory(storage_root) == before

    # unknown model -> 404 on both endpoints; nothing mutated
    assert api_client.delete("/api/v1/models/no-m67").status_code == 404
    assert api_client.get(
        "/api/v1/models/no-m67/retention").status_code == 404
    assert _inventory(storage_root) == before

    # a fresh model (internal-free) deletes successfully over HTTP
    m2 = api_client.post("/api/v1/models", json={
        "config": {"name": "m67api-fresh", "vocab_size": 640,
                   "context_length": 64, "hidden_size": 64, "n_layers": 2,
                   "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128}})
    assert m2.status_code == 201, m2.text
    mid2 = m2.json()["model"]["id"]
    pre_walk = sorted(
        p.relative_to(storage_root / "models" / mid2).as_posix()
        for p in (storage_root / "models" / mid2).rglob("*")
        if p.is_file())
    d2 = api_client.delete(f"/api/v1/models/{mid2}")
    assert d2.status_code == 200, d2.text
    res = d2.json()
    assert set(res) == {"model_id", "files_removed", "bytes_reclaimed"}
    assert res["model_id"] == mid2
    assert res["files_removed"] == len(pre_walk)
    assert api_client.get(f"/api/v1/models/{mid2}").status_code == 404
    # repeated deletion -> 404
    assert api_client.delete(f"/api/v1/models/{mid2}").status_code == 404

    # OpenAPI: 93 paths; the retention route is GET-only; NO new
    # DELETE paths (the model DELETE is the SAME pre-existing route);
    # the delete-operation set is exactly the four pre-existing ones
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 93
    assert set(spec["paths"]["/api/v1/models/{model_id}/retention"]
               .keys()) == {"get"}
    assert set(spec["paths"]["/api/v1/models/{model_id}"].keys()) == \
        {"get", "delete"}
    deletes = sorted(p for p, ops in spec["paths"].items()
                     if "delete" in ops)
    assert deletes == [
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/models/{model_id}",
        "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}",
        "/api/v1/tokenizers/{tokenizer_id}",
    ]
    for s in ("ModelDeletionResult", "ModelDeletionBlocked",
              "ModelDeletionBlocker", "ModelRetentionOverview"):
        assert s in spec["components"]["schemas"], s
    # the 409 response model is documented on the DELETE route
    assert spec["paths"]["/api/v1/models/{model_id}"]["delete"][
        "responses"]["409"]["content"]["application/json"]["schema"][
        "$ref"] == "#/components/schemas/ModelDeletionBlocked"
