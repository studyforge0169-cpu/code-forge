"""Milestone 69 tests: model-owned record usage overview (read-only).

The M69 invariants under test:

* TOPOLOGY — every persisted reference to each model-owned record
  (training runs, checkpoints, workflows, evaluations, comparisons,
  gate decisions) is reported: internal references (model manifest
  pointers; checkpoint parent + run-provenance lineage — the M61
  informational edges; workflow stage artifacts; evaluation /
  comparison / gate evidence records) and EXTERNAL root-level
  references (suite-run states + probe results, samples,
  sample-quality measurements — the future per-record blocker
  surface). Verified against an INDEPENDENT raw-manifest oracle.
* CONSISTENCY — record sets == the ONE public listings (M66/M69
  agreement); lineage-only checkpoints have EMPTY M61 blocker lists
  (the M61 classification preserved); evaluation-referenced
  checkpoints are M61-blocked.
* READ-ONLY — zero mutation (byte-level inventories), deterministic
  byte-identical repeats, no phantom records, no repair.
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
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


# --------------------------------------------------------------------------- #
# INDEPENDENT oracle: raw manifest parsing (never the app's analysis)
# --------------------------------------------------------------------------- #

RECORD_CATS = ["training_run", "checkpoint", "workflow", "evaluation",
               "comparison", "gate"]
REF_CATS = ["model", "checkpoint", "run_provenance", "workflow",
            "evaluation", "comparison", "gate", "suite_run", "sample",
            "sample_quality"]
EXTERNAL_CATS = {"suite_run", "sample", "sample_quality"}
LINEAGE_EDGES = {("checkpoint", "checkpoint"), ("run_provenance",
                                                "checkpoint")}
KIND_TO_FAMILY = {"training_report": "training_run",
                  "evaluation": "evaluation",
                  "comparison": "comparison",
                  "gate_decision": "gate"}


def _oracle(root: Path, model_id: str) -> dict:
    """(family, record_id) -> sorted [(category, reference_id)] and
    the family record-id lists, derived from raw manifests."""
    refs: dict[tuple[str, str], set] = {}

    def add(family, rid, category, ref_id):
        refs.setdefault((family, rid), set()).add((category, ref_id))

    mdir = root / "models" / model_id
    man = json.loads((mdir / "manifest.json").read_text())
    if man.get("latest_checkpoint"):
        add("checkpoint", man["latest_checkpoint"], "model", model_id)
    if man.get("best_checkpoint"):
        add("checkpoint", man["best_checkpoint"], "model", model_id)
    for p in man.get("training_provenance", []):
        for key in ("parent_checkpoint_id", "initial_checkpoint_id",
                    "final_checkpoint_id", "rolled_back_to"):
            if p.get(key):
                add("checkpoint", p[key], "run_provenance", model_id)

    def family_records(fam):
        froot = mdir / fam
        out = []
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if d.is_dir():
                    out.append(json.loads(
                        (d / "manifest.json").read_text()))
        return out

    for c in family_records("checkpoints"):
        add("training_run", c["run_id"], "checkpoint",
            c["checkpoint_id"])
        if c.get("parent_checkpoint_id"):
            add("checkpoint", c["parent_checkpoint_id"], "checkpoint",
                c["checkpoint_id"])
    for e in family_records("evaluations"):
        if e.get("checkpoint_id"):
            add("checkpoint", e["checkpoint_id"], "evaluation",
                e["eval_id"])
    for c in family_records("comparisons"):
        for side in (c["state_a"], c["state_b"]):
            if side.get("checkpoint_id"):
                add("checkpoint", side["checkpoint_id"], "comparison",
                    c["comparison_id"])
            add("evaluation", side["evaluation_id"], "comparison",
                c["comparison_id"])
    for g in family_records("gates"):
        for side in (g.get("candidate"), g.get("baseline")):
            if side:
                if side.get("checkpoint_id"):
                    add("checkpoint", side["checkpoint_id"], "gate",
                        g["decision_id"])
                add("evaluation", side["evaluation_id"], "gate",
                    g["decision_id"])
        if g.get("comparison_id"):
            add("comparison", g["comparison_id"], "gate",
                g["decision_id"])
    for w in family_records("workflows"):
        for st in w.get("stages", []):
            art = st.get("artifact")
            if not art:
                continue
            fam = KIND_TO_FAMILY.get(art.get("kind"))
            if fam:
                add(fam, art["artifact_id"], "workflow",
                    w["workflow_id"])
            if art.get("checkpoint_id"):
                add("checkpoint", art["checkpoint_id"], "workflow",
                    w["workflow_id"])
        if w.get("suggested_checkpoint_id"):
            add("checkpoint", w["suggested_checkpoint_id"], "workflow",
                w["workflow_id"])
    # root-level external families (model-scoped listings by layout)
    sroot = root / "suite-runs"
    if sroot.exists():
        for d in sorted(sroot.iterdir()):
            if not d.is_dir():
                continue
            r = json.loads((d / "manifest.json").read_text())
            if r.get("model_id") != model_id:
                continue
            if r.get("state", {}).get("checkpoint_id"):
                add("checkpoint", r["state"]["checkpoint_id"],
                    "suite_run", r["suite_run_id"])
            for pr in r.get("results", []):
                if pr.get("evaluation_id"):
                    add("evaluation", pr["evaluation_id"], "suite_run",
                        r["suite_run_id"])
    for fam, cat in (("samples", "sample"),
                     ("sample-evaluations", "sample_quality")):
        froot = root / fam / model_id
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if not d.is_dir():
                    continue
                rec = json.loads((d / "manifest.json").read_text())
                add("checkpoint", rec["checkpoint_id"], cat,
                    rec["sample_id"] if fam == "samples"
                    else rec["evaluation_id"])

    families = {
        "training_run": sorted(p["run_id"] for p in
                               man.get("training_provenance", [])),
        "checkpoint": sorted(c["checkpoint_id"] for c in
                             family_records("checkpoints")),
        "workflow": sorted(w["workflow_id"] for w in
                           family_records("workflows")),
        "evaluation": sorted(e["eval_id"] for e in
                             family_records("evaluations")),
        "comparison": sorted(c["comparison_id"] for c in
                             family_records("comparisons")),
        "gate": sorted(g["decision_id"] for g in
                       family_records("gates")),
    }
    order = {c: i for i, c in enumerate(REF_CATS)}
    return {"refs": {k: sorted(v, key=lambda cr: (order[cr[0]], cr[1]))
                     for k, v in refs.items()},
            "families": families}


# --------------------------------------------------------------------------- #
# Rich fixture
# --------------------------------------------------------------------------- #

class Env:
    def __init__(self, root: Path):
        self.forge = ModelForge(root=root)
        self.root = root
        self.ds = None
        self.tok = None
        self.model = None      # rich: every family + external refs
        self.internal = None   # internal-only model
        self.fresh = None      # empty model


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m69-root"))
    forge = e.forge
    up = forge.upload_dataset([("m69.txt", _corpus(190, "m69"))],
                              name="m69-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m69-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)

    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m69-rich-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.model, dataset_id=e.ds,
        tokenizer_id=e.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1,
        steps=8))
    ck = [c.checkpoint_id for c in forge.list_checkpoints(e.model)]
    forge.run_evaluation(EvaluationConfig(
        model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=ck[0]))
    forge.run_comparison(ComparisonRequest(
        model_id=e.model,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ck[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=ck[1]),
        dataset_id=e.ds, split="validation", tokenizer_id=e.tok,
        batch_size=8, max_seq_len=32, seed=3, tolerance=1e-4))
    forge.run_gate(GateRequest(
        model_id=e.model,
        policy=GatePolicy(
            name="m69-gate", model_id=e.model, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=4, baseline_type="checkpoint",
            baseline_checkpoint_id=ck[1], tolerance=1.0),
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=ck[0])))
    forge.run_workflow(WorkflowPlan(
        name="m69-workflow", model_id=e.model, description="M69 fixture",
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
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m69-suite", description="M69 fixture",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))
    e.suite_run = forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m69-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=ck[0])))
    best = forge.select_best_checkpoint(e.model).checkpoint.checkpoint_id
    e.sample = forge.generate_sample(SampleGenerateRequest(
        model_id=e.model, checkpoint_id=best, tokenizer_id=e.tok,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    e.quality = forge.evaluate_sample(e.model, e.sample.sample_id)

    # an internal-only model: training + evaluation, nothing external
    e.internal = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m69-internal-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=2)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.internal, dataset_id=e.ds,
        tokenizer_id=e.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=7,
        steps=8))
    ick = [c.checkpoint_id for c in forge.list_checkpoints(e.internal)]
    forge.run_evaluation(EvaluationConfig(
        model_id=e.internal, dataset_id=e.ds, tokenizer_id=e.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=8,
        checkpoint_id=ick[0]))

    # a fresh model: no records at all
    e.fresh = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m69-fresh-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=3)))[0].id
    return e


# --------------------------------------------------------------------------- #
# Engine: topology, oracle parity, consistency, determinism
# --------------------------------------------------------------------------- #

def test_m69_records_usage_overview(env):
    forge, root = env.forge, env.root
    before = _inventory(root)
    o1 = forge.model_records_usage_overview(env.model)
    o2 = forge.model_records_usage_overview(env.model)
    assert o1.model_dump(mode="json") == o2.model_dump(mode="json")
    assert _inventory(root) == before        # read-only

    # identity + canonical category order
    assert o1.model_id == env.model and o1.name == "m69-rich-model"
    assert [c.category for c in o1.categories] == \
        list(ModelForge.MODEL_RECORD_CATEGORIES)

    # record sets == the ONE public listings (the §6 invariant)
    cats = {c.category: c.records for c in o1.categories}
    assert [r.record_id for r in cats["training_run"]] == sorted(
        p.run_id for p in forge.get_model(env.model).training_provenance)
    assert [r.record_id for r in cats["checkpoint"]] == sorted(
        c.checkpoint_id for c in forge.list_checkpoints(env.model))
    assert [r.record_id for r in cats["workflow"]] == sorted(
        w.workflow_id for w in forge.list_workflows(env.model))
    assert [r.record_id for r in cats["evaluation"]] == sorted(
        e.eval_id for e in forge.list_evaluations(env.model))
    assert [r.record_id for r in cats["comparison"]] == sorted(
        c.comparison_id for c in forge.list_comparisons(env.model))
    assert [r.record_id for r in cats["gate"]] == sorted(
        g.decision_id for g in forge.list_gate_decisions(env.model))

    # every family non-empty in the rich fixture
    assert all(cats[c] for c in ModelForge.MODEL_RECORD_CATEGORIES)

    # ---- the INDEPENDENT ORACLE (raw manifest parse) ------------------ #
    ora = _oracle(root, env.model)
    assert ora["families"] == {c: [r.record_id for r in cats[c]]
                               for c in cats}
    for category in o1.categories:
        for r in category.records:
            expected = ora["refs"].get(
                (category.category, r.record_id), [])
            got = [(x.category, x.reference_id)
                   for x in r.references]
            assert got == expected, (category.category, r.record_id)
            # counts are exact sums; external flags correct
            assert r.total_references == len(r.references)
            assert r.external_references == sum(
                1 for x in r.references if x.external)
            assert r.internal_references == \
                r.total_references - r.external_references
            for x in r.references:
                assert x.external == (x.category in EXTERNAL_CATS)

    # aggregates are exact sums over the categories
    assert o1.total_records == sum(len(c.records) for c in o1.categories)
    assert o1.total_references == sum(
        r.total_references for c in o1.categories for r in c.records)
    assert o1.internal_references == sum(
        r.internal_references for c in o1.categories for r in c.records)
    assert o1.external_references == sum(
        r.external_references for c in o1.categories for r in c.records)

    # ---- exact topology spot checks (from the fixture's known shape) -- #
    flat = {(c.category, r.record_id): r for c in o1.categories
            for r in c.records}
    # the suite run references its state checkpoint + its probe eval
    ck_ids = {c.checkpoint_id for c in forge.list_checkpoints(env.model)}
    state_ck = env.suite_run.state.checkpoint_id
    assert ("checkpoint", state_ck) in flat
    assert ("suite_run", env.suite_run.suite_run_id) in [
        (x.category, x.reference_id)
        for x in flat[("checkpoint", state_ck)].references]
    probe_evals = [pr.evaluation_id for pr in env.suite_run.results
                   if pr.evaluation_id]
    for pe in probe_evals:
        assert ("suite_run", env.suite_run.suite_run_id) in [
            (x.category, x.reference_id)
            for x in flat[("evaluation", pe)].references]
    # the sample + measurement reference the best checkpoint (EXTERNAL)
    best = forge.select_best_checkpoint(env.model).checkpoint.checkpoint_id
    sam_refs = [(x.category, x.reference_id)
                for x in flat[("checkpoint", best)].references]
    assert ("sample", env.sample.sample_id) in sam_refs
    assert ("sample_quality", env.quality.evaluation_id) in sam_refs
    # the model manifest's own pointers are reported (internal)
    rec = forge.get_model(env.model)
    if rec.latest_checkpoint:
        assert ("model", env.model) in [
            (x.category, x.reference_id) for x in
            flat[("checkpoint", rec.latest_checkpoint)].references]
    # training runs are referenced by their checkpoints + the workflow
    wf = forge.list_workflows(env.model)[0]
    run_ids = {p.run_id for p in rec.training_provenance}
    wf_run_refs = [(x.category, x.reference_id)
                   for rid in run_ids
                   for x in flat[("training_run", rid)].references]
    assert all(cat in ("checkpoint", "workflow")
               for cat, _ in wf_run_refs)
    assert ("workflow", wf.workflow_id) in wf_run_refs
    # comparisons are referenced by the gate (its comparison_id)
    gate = forge.list_gate_decisions(env.model)[0]
    if gate.comparison_id:
        assert ("gate", gate.decision_id) in [
            (x.category, x.reference_id) for x in
            flat[("comparison", gate.comparison_id)].references]
    # evaluations are referenced by comparison/gate sides + workflow
    comp = forge.list_comparisons(env.model)[0]
    for side in (comp.state_a, comp.state_b):
        refs = [(x.category, x.reference_id)
                for x in flat[("evaluation", side.evaluation_id)].references]
        assert ("comparison", comp.comparison_id) in refs
    # workflows and gates are referenced by NOTHING (leaves)
    assert all(r.total_references == 0 for r in cats["workflow"]) or \
        all(("workflow", w.workflow_id) not in
            [(x.category, x.reference_id)
             for c2 in o1.categories for r2 in c2.records
             for x in r2.references]
            for w in forge.list_workflows(env.model))
    assert all(r.total_references == 0 for r in cats["gate"])

    # ---- M61 consistency: lineage never blocks ----------------------- #
    # any checkpoint whose references are ALL lineage edges has an
    # EMPTY M61 blocker list (the M61 classification preserved)
    for rid in ck_ids:
        refs = flat[("checkpoint", rid)].references
        if refs and all(
                (x.category, "checkpoint") in LINEAGE_EDGES
                for x in refs):
            assert forge.checkpoint_blockers(env.model, rid) == []
    # a checkpoint referenced by an evaluation IS M61-blocked
    eval_ck = next(e.checkpoint_id for e in
                   forge.list_evaluations(env.model)
                   if e.checkpoint_id)
    assert forge.checkpoint_blockers(env.model, eval_ck) != []

    # ---- M66 cross-check ---------------------------------------------- #
    usage = forge.model_usage_overview(env.model)
    ucats = {c.category: c.references for c in usage.categories}
    for fam, m66cat in (("training_run", "training_run"),
                        ("checkpoint", "checkpoint"),
                        ("workflow", "workflow"),
                        ("evaluation", "evaluation"),
                        ("comparison", "comparison"),
                        ("gate", "gate")):
        assert sorted(r.record_id for r in cats[fam]) == \
            sorted(ucats[m66cat])
    # the M66 external record ids appear as external referencing ids;
    # external_references counts ENTRIES (the suite run externally
    # references TWO records: its state checkpoint + its probe eval)
    ext_ids = {r for c in ("suite_run", "sample", "sample_quality")
               for r in ucats[c]}
    m69_ext_ids = {x.reference_id for c2 in o1.categories
                   for r2 in c2.records for x in r2.references
                   if x.external}
    assert ext_ids <= m69_ext_ids
    assert o1.external_references == sum(
        r2.external_references for c2 in o1.categories
        for r2 in c2.records)
    # the fixture's exact external entries: suite-run state checkpoint
    # + suite-run probe evaluation + the sample + its measurement
    assert o1.external_references == 4

    # the rich model has BOTH internal and external references
    assert o1.internal_references > 0 and o1.external_references > 0


def test_m69_scope_isolation_and_edges(env):
    forge, root = env.forge, env.root

    # internal-only model: references, but zero external
    io = forge.model_records_usage_overview(env.internal)
    assert io.total_records > 0 and io.total_references > 0
    assert io.external_references == 0
    assert io.internal_references == io.total_references
    ora = _oracle(root, env.internal)
    for category in io.categories:
        for r in category.records:
            assert [(x.category, x.reference_id) for x in r.references] == \
                ora["refs"].get((category.category, r.record_id), [])

    # fresh model: all six categories present, all empty, zero totals
    fo = forge.model_records_usage_overview(env.fresh)
    assert [c.category for c in fo.categories] == \
        list(ModelForge.MODEL_RECORD_CATEGORIES)
    assert all(c.records == [] for c in fo.categories)
    assert fo.total_records == 0 and fo.total_references == 0
    assert fo.internal_references == 0 and fo.external_references == 0

    # unknown model -> the family's 404
    with pytest.raises(FileNotFoundError):
        forge.model_records_usage_overview("no-such-m69-model")

    # malformed manifest -> registry-invisible (the M61/M65 convention)
    mpath = root / "models" / env.fresh / "manifest.json"
    original = mpath.read_bytes()
    mpath.write_bytes(b"not json at all")
    try:
        with pytest.raises(FileNotFoundError):
            forge.model_records_usage_overview(env.fresh)
    finally:
        mpath.write_bytes(original)

    # dangling reference: an evaluation pointing at a bogus checkpoint
    # id produces NO phantom record (the target does not exist) and
    # never mutates anything
    edir = root / "models" / env.internal / "evaluations"
    ed = sorted(edir.iterdir())[0]
    eorig = (ed / "manifest.json").read_bytes()
    eman = json.loads(eorig)
    eman["checkpoint_id"] = "bogus-checkpoint-id"
    (ed / "manifest.json").write_text(json.dumps(eman))
    try:
        before = _inventory(root)
        o = forge.model_records_usage_overview(env.internal)
        all_records = {r.record_id for c in o.categories
                       for r in c.records}
        assert "bogus-checkpoint-id" not in all_records   # no phantom
        assert o.total_references > 0                     # still sane
        assert forge.model_records_usage_overview(
            env.internal).model_dump(mode="json") == \
            o.model_dump(mode="json")                     # deterministic
        assert _inventory(root) == before                 # read-only
    finally:
        (ed / "manifest.json").write_bytes(eorig)
    assert (ed / "manifest.json").read_bytes() == eorig  # restored

    # corrupt-but-parseable record (tampered result_hash): the usage
    # view stays deterministic and read-only (no repair, no crash)
    gdir = root / "models" / env.model / "gates"
    gd = sorted(gdir.iterdir())[0]
    gorig = (gd / "manifest.json").read_bytes()
    gman = json.loads(gorig)
    gman["result_hash"] = "0" * 64
    (gd / "manifest.json").write_text(json.dumps(gman))
    try:
        before = _inventory(root)
        o = forge.model_records_usage_overview(env.model)
        assert forge.model_records_usage_overview(
            env.model).model_dump(mode="json") == \
            o.model_dump(mode="json")
        assert _inventory(root) == before
    finally:
        (gd / "manifest.json").write_bytes(gorig)

    # duplicate reference ids: a comparison with BOTH sides on the
    # same checkpoint references it ONCE (unique (category, id) pairs)
    cks = [c.checkpoint_id for c in forge.list_checkpoints(env.model)]
    comp = forge.run_comparison(ComparisonRequest(
        model_id=env.model,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=cks[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=cks[0]),
        dataset_id=env.ds, split="validation", tokenizer_id=env.tok,
        batch_size=8, max_seq_len=32, seed=11, tolerance=1e-4))
    o = forge.model_records_usage_overview(env.model)
    flat = {(c.category, r.record_id): r for c in o.categories
            for r in c.records}
    refs = [x.reference_id for x in
            flat[("checkpoint", cks[0])].references
            if x.category == "comparison"]
    assert refs.count(comp.comparison_id) == 1
    # oracle still agrees after the new record
    ora = _oracle(root, env.model)
    for category in o.categories:
        for r in category.records:
            assert [(x.category, x.reference_id) for x in r.references] \
                == ora["refs"].get((category.category, r.record_id), [])


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m69_api_records_usage(api_client):
    import json as _json

    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m69api.txt", _corpus(180, "m69api"),
                          "text/plain"))],
        data={"name": "m69api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m69api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m69api-model", "vocab_size": 640,
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
    ck = sorted(c["checkpoint_id"] for c in
                api_client.get(f"/api/v1/models/{mid}/checkpoints").json())
    r = api_client.post("/api/v1/evaluations/run", json={
        "model_id": mid, "dataset_id": ds, "tokenizer_id": tok,
        "split": "validation", "batch_size": 8, "max_seq_len": 32,
        "seed": 2, "checkpoint_id": ck[0]})
    assert r.status_code == 200, r.text
    r = api_client.post("/api/v1/suite-runs", json={
        "model_id": mid, "suite_id": "m69api-suite",
        "state": {"state_kind": "checkpoint", "checkpoint_id": ck[0]}})
    # (unknown suite -> 404 is fine; register it first)
    if r.status_code == 404:
        r2 = api_client.post("/api/v1/probe-suites", json={
            "suite_id": "m69api-suite", "description": "M69 api fixture",
            "probes": [{"dataset_id": ds, "split": "validation",
                        "tokenizer_id": tok, "batch_size": 8,
                        "max_seq_len": 32, "seed": 42}]})
        assert r2.status_code == 201, r2.text
        r = api_client.post("/api/v1/suite-runs", json={
            "model_id": mid, "suite_id": "m69api-suite",
            "state": {"state_kind": "checkpoint", "checkpoint_id": ck[0]}})
    assert r.status_code == 200, r.text
    run_id = r.json()["suite_run_id"]
    r = api_client.post("/api/v1/samples/generate", json={
        "model_id": mid, "checkpoint_id": ck[1], "tokenizer_id": tok,
        "prompt": "river mountain", "strategy": "greedy",
        "max_new_tokens": 4})
    assert r.status_code == 200, r.text
    sample_id = r.json()["sample_id"]
    r = api_client.post(
        f"/api/v1/models/{mid}/samples/{sample_id}/quality")
    assert r.status_code == 200, r.text

    storage_root = Path(api_client.get("/api/v1/project").json()[
        "storage_root"])

    r = api_client.get(f"/api/v1/models/{mid}/records/usage")
    assert r.status_code == 200, r.text
    body = r.content
    ov = r.json()
    assert set(ov) == {"model_id", "name", "total_records",
                       "total_references", "internal_references",
                       "external_references", "categories"}
    assert ov["model_id"] == mid
    assert [c["category"] for c in ov["categories"]] == RECORD_CATS
    for c in ov["categories"]:
        for rec in c["records"]:
            assert set(rec) == {"record_id", "references",
                                "total_references",
                                "internal_references",
                                "external_references"}
            for x in rec["references"]:
                assert set(x) == {"category", "reference_id", "external"}

    # aggregate == sums
    assert ov["total_records"] == sum(len(c["records"])
                                      for c in ov["categories"])
    assert ov["total_references"] == sum(
        r2["total_references"] for c in ov["categories"]
        for r2 in c["records"])

    # oracle parity over the session root
    ora = _oracle(storage_root, mid)
    for c in ov["categories"]:
        assert [r2["record_id"] for r2 in c["records"]] == \
            ora["families"][c["category"]]
        for rec in c["records"]:
            assert [(x["category"], x["reference_id"])
                    for x in rec["references"]] == \
                ora["refs"].get((c["category"], rec["record_id"]), [])

    # the external references are exactly the M66 external record ids
    usage = api_client.get(f"/api/v1/models/{mid}/usage").json()
    ucats = {c["category"]: c["references"] for c in usage["categories"]}
    m69_ext = {x["reference_id"] for c in ov["categories"]
               for rec in c["records"] for x in rec["references"]
               if x["external"]}
    assert m69_ext == {run_id, sample_id} | \
        set(ucats["sample_quality"])
    # external_references counts ENTRIES (the suite run references
    # its state checkpoint AND its probe evaluation)
    assert ov["external_references"] == sum(
        rec["external_references"] for c in ov["categories"]
        for rec in c["records"])
    # the suite run's state checkpoint + probe eval are discovered
    flat = {(c["category"], rec["record_id"]): rec
            for c in ov["categories"] for rec in c["records"]}
    assert ("suite_run", run_id) in [
        (x["category"], x["reference_id"])
        for x in flat[("checkpoint", ck[0])]["references"]]

    # determinism + zero mutation
    before = _inventory(storage_root)
    assert api_client.get(
        f"/api/v1/models/{mid}/records/usage").content == body
    assert _inventory(storage_root) == before

    # unknown model -> 404
    assert api_client.get(
        "/api/v1/models/no-m69/records/usage").status_code == 404

    # OpenAPI: 119 paths, GET-only, delete set grew to 14 with M71
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 119
    NEW = "/api/v1/models/{model_id}/records/usage"
    assert set(spec["paths"][NEW].keys()) == {"get"}
    deletes = sorted(p for p, ops in spec["paths"].items()
                     if "delete" in ops)
    assert len(deletes) == 14
    for s in ("RecordReference", "ModelOwnedRecordUsage",
              "ModelOwnedRecordCategory", "ModelRecordsUsageOverview"):
        assert s in spec["components"]["schemas"], s
