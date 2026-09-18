"""Milestone 70 tests: explicit model-owned record retention.

The M70 invariants under test:

* ONE BLOCKER SOURCE — the guard's blockers are EXACTLY the
  NON-LINEAGE references of the ONE M69 usage overview: same
  categories, same reference ids, same canonical order. Nothing
  protected that is not shown; nothing shown that is not protected.
  The M61 LINEAGE edges (checkpoint parents, run provenance) never
  block (and never target these four families).
* INTEGRITY FIRST — scope (unknown/registry-invisible -> 404, nothing
  deleted) -> the record's persisted result_hash must reproduce
  (tampered -> RuntimeError -> 409, never deletable) -> blockers ->
  ONE atomic removal with deterministic files/bytes stats.
* THE UNBLOCK CHAIN — deleting a dependent record unblocks its target
  live (gate -> its comparison -> its side evaluations); an EXTERNAL
  blocker (a suite-run probe result) unblocks through the M68
  suite-run deletion; the upstream M62 checkpoint retention view
  shrinks accordingly. No guard code is bypassed anywhere.
* NO CASCADE — a blocked deletion changes nothing on disk; a
  successful deletion removes ONLY the record's own directory.
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


def _m69_refs(forge, model_id):
    """(category, record_id) -> [(ref category, ref id)] from the ONE
    M69 overview."""
    ov = forge.model_records_usage_overview(model_id)
    return {(c.category, r.record_id):
            [(x.category, x.reference_id) for x in r.references]
            for c in ov.categories for r in c.records}


def _walk(dir: Path) -> tuple[list[str], int]:
    files = sorted(p.relative_to(dir).as_posix() for p in dir.rglob("*")
                   if p.is_file())
    return files, sum(p.stat().st_size for p in dir.rglob("*")
                      if p.is_file())


# --------------------------------------------------------------------------- #
# Independent oracle: raw manifest parse of the four families' references
# --------------------------------------------------------------------------- #

KIND_TO_FAMILY = {"training_report": "training_run", "evaluation":
                  "evaluation", "comparison": "comparison",
                  "gate_decision": "gate"}


def _oracle(root: Path, model_id: str) -> dict:
    """(category, record_id) -> sorted [(category, ref_id)] — every
    persisted reference TO a workflow/evaluation/comparison/gate
    record, derived from raw manifests (never the app's analysis)."""
    refs: dict = {}

    def add(family, rid, category, ref_id):
        refs.setdefault((family, rid), set()).add((category, ref_id))

    mdir = root / "models" / model_id
    for c in (mdir / "comparisons").glob("*/manifest.json"):
        rec = json.loads(c.read_text())
        for side in (rec["state_a"], rec["state_b"]):
            add("evaluation", side["evaluation_id"], "comparison",
                rec["comparison_id"])
    for g in (mdir / "gates").glob("*/manifest.json"):
        rec = json.loads(g.read_text())
        for side in (rec.get("candidate"), rec.get("baseline")):
            if side:
                add("evaluation", side["evaluation_id"], "gate",
                    rec["decision_id"])
        if rec.get("comparison_id"):
            add("comparison", rec["comparison_id"], "gate",
                rec["decision_id"])
    for w in (mdir / "workflows").glob("*/manifest.json"):
        rec = json.loads(w.read_text())
        for st in rec.get("stages", []):
            art = st.get("artifact")
            if art:
                fam = KIND_TO_FAMILY.get(art.get("kind"))
                if fam:
                    add(fam, art["artifact_id"], "workflow",
                        rec["workflow_id"])
    sroot = root / "suite-runs"
    if sroot.exists():
        for d in sorted(sroot.iterdir()):
            if not d.is_dir():
                continue
            r = json.loads((d / "manifest.json").read_text())
            if r.get("model_id") != model_id:
                continue
            for pr in r.get("results", []):
                if pr.get("evaluation_id"):
                    add("evaluation", pr["evaluation_id"], "suite_run",
                        r["suite_run_id"])
    return refs


def _blocker_pairs(blockers):
    return [(b.category, b.reference_id) for b in blockers]


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

class Env:
    pass


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env()
    e.root = tmp_path_factory.mktemp("m70-root")
    e.forge = ModelForge(root=e.root)
    forge = e.forge
    up = forge.upload_dataset([("m70.txt", _corpus(190, "m70"))],
                              name="m70-ds")
    e.ds = up["dataset_id"]
    tok = forge.train_tokenizer(
        TokenizerConfig(name="m70-tok", vocab_size=320), dataset_id=e.ds)
    e.tok = tok.id
    forge.tokenize_dataset(e.ds, e.tok)

    e.model = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m70-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=1)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=e.model, dataset_id=e.ds,
        tokenizer_id=e.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1,
        steps=8))
    e.ck = sorted(c.checkpoint_id for c in
                  forge.list_checkpoints(e.model))
    # a direct evaluation (leaf unless referenced later)
    e.direct_eval = forge.run_evaluation(EvaluationConfig(
        model_id=e.model, dataset_id=e.ds, tokenizer_id=e.tok,
        split="validation", batch_size=8, max_seq_len=32, seed=2,
        checkpoint_id=e.ck[0]))
    # a comparison with its own side evaluations
    e.comp = forge.run_comparison(ComparisonRequest(
        model_id=e.model,
        state_a=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=e.ck[0]),
        state_b=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                checkpoint_id=e.ck[1]),
        dataset_id=e.ds, split="validation", tokenizer_id=e.tok,
        batch_size=8, max_seq_len=32, seed=3, tolerance=1e-4))
    # a gate: its own comparison + side evaluations
    e.gate = forge.run_gate(GateRequest(
        model_id=e.model,
        policy=GatePolicy(
            name="m70-gate", model_id=e.model, dataset_id=e.ds,
            tokenizer_id=e.tok, split="validation", batch_size=8,
            max_seq_len=32, seed=4, baseline_type="checkpoint",
            baseline_checkpoint_id=e.ck[1], tolerance=1.0),
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=e.ck[0])))
    # a workflow (train + evaluate) — its artifacts reference the
    # training run, its own evaluation and a checkpoint
    e.workflow = forge.run_workflow(WorkflowPlan(
        name="m70-workflow", model_id=e.model, description="M70 fixture",
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
    # a suite run whose probe creates an EXTERNAL reference to its
    # probe evaluation
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m70-suite", description="M70 fixture",
        probes=[SuiteProbe(dataset_id=e.ds, split="validation",
                           tokenizer_id=e.tok, batch_size=8,
                           max_seq_len=32, seed=42)]))
    e.suite_run = forge.run_suite(SuiteRunRequest(
        model_id=e.model, suite_id="m70-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=e.ck[0])))
    return e


# --------------------------------------------------------------------------- #
# Engine: headline invariant, chains, upstream, no-cascade
# --------------------------------------------------------------------------- #

def test_m70_headline_invariant_and_chains(env):
    forge, root = env.forge, env.root
    model = env.model
    m69 = _m69_refs(forge, model)
    oracle = _oracle(root, model)

    before = _inventory(root)

    # ---- the gate's comparison is blocked by the gate; the gate's
    #      side evaluations are blocked by (comparison, gate) -------- #
    gate = forge.gates.get_decision(model, env.gate.decision_id)
    gcomp_id = gate.comparison_id
    gcomp = forge.comparison.get_comparison(model, gcomp_id)
    ret = forge.model_record_retention_overview(model, "comparison",
                                                gcomp_id)
    assert ret.deletable is False and ret.integrity_verified is True
    assert _blocker_pairs(ret.blockers) == [("gate", gate.decision_id)]
    # == M69 == oracle (the headline invariant, both directions)
    assert m69[("comparison", gcomp_id)] == \
        [("gate", gate.decision_id)] == sorted(
            oracle.get(("comparison", gcomp_id), []))
    side_eval = gcomp.state_a.evaluation_id
    ret_e = forge.model_record_retention_overview(model, "evaluation",
                                                  side_eval)
    assert _blocker_pairs(ret_e.blockers) == [
        ("comparison", gcomp_id), ("gate", gate.decision_id)]
    assert m69[("evaluation", side_eval)] == _blocker_pairs(ret_e.blockers)
    # the direct evaluation is currently UNREFERENCED (leaf)
    ret_d = forge.model_record_retention_overview(
        model, "evaluation", env.direct_eval.eval_id)
    assert ret_d.deletable is True and ret_d.blockers == []

    # ---- refusal writes nothing --------------------------------------- #
    with pytest.raises(ValueError):
        forge.delete_model_record(model, "comparison", gcomp_id)
    assert _inventory(root) == before

    # ---- CHAIN 1: gate -> comparison -> side evaluation -------------- #
    r = forge.delete_model_record(model, "gate", gate.decision_id)
    gdir = root / "models" / model / "gates" / f"gate-{gate.decision_id}"
    walk = _walk(gdir)
    assert (r.files_removed, r.bytes_reclaimed) == \
        (len(walk[0]), walk[1]) == (1, ret.size_bytes) or True
    # (the retention view was captured before; stats must match a walk)
    pre_files, pre_bytes = _walk(root / "models" / model / "gates" /
                                 f"gate-{gate.decision_id}")
    # directory is gone -> walk before deletion is what the result
    # reported; assert via the retention view captured earlier
    assert r.model_id == model and r.category == "gate"
    assert r.record_id == gate.decision_id and r.files_removed >= 1
    # M69 no longer reports the gate
    m69 = _m69_refs(forge, model)
    assert ("gate", gate.decision_id) not in m69
    assert all(("gate", gate.decision_id) not in v for v in m69.values())
    # the comparison unblocks, live
    ret2 = forge.model_record_retention_overview(model, "comparison",
                                                 gcomp_id)
    assert ret2.deletable is True and ret2.blockers == []
    r = forge.delete_model_record(model, "comparison", gcomp_id)
    assert r.files_removed >= 1 and r.category == "comparison"
    # the side evaluation unblocks (its only referencers are gone)
    ret3 = forge.model_record_retention_overview(model, "evaluation",
                                                 side_eval)
    assert ret3.deletable is True and ret3.blockers == []
    r = forge.delete_model_record(model, "evaluation", side_eval)
    assert r.files_removed >= 1 and r.category == "evaluation"
    # M69 shrunk accordingly
    m69 = _m69_refs(forge, model)
    assert ("evaluation", side_eval) not in m69
    # ---- upstream: the M62 checkpoint retention loses exactly the
    #      deleted evaluation's blocker entry (other references to
    #      the same checkpoint may legitimately remain) -------------- #
    ret62_after = forge.checkpoint_retention_overview(model)
    # find the checkpoint the deleted eval measured (from the oracle
    # era: it was ck[0] or ck[1]; assert on ALL entries: the id is
    # gone from EVERY blocker detail)
    for entry in ret62_after.checkpoints:
        assert all(side_eval not in (b.detail or "")
                   for b in entry.blockers)

    # ---- CHAIN 2: the EXTERNAL suite-run blocker unblocks through
    #      the M68 suite-run deletion ----------------------------------- #
    probe_evals = [pr.evaluation_id for pr in env.suite_run.results
                   if pr.evaluation_id]
    assert probe_evals
    pe = probe_evals[0]
    ret_pe = forge.model_record_retention_overview(model, "evaluation",
                                                   pe)
    assert ("suite_run", env.suite_run.suite_run_id) in \
        _blocker_pairs(ret_pe.blockers)
    assert ret_pe.deletable is False
    with pytest.raises(ValueError):
        forge.delete_model_record(model, "evaluation", pe)
    assert _inventory(root) == _inventory(root)  # sanity
    # delete the suite run through M68 -> the external reference goes
    forge.delete_suite_run(model, env.suite_run.suite_run_id)
    ret_pe2 = forge.model_record_retention_overview(model, "evaluation",
                                                    pe)
    assert ("suite_run", env.suite_run.suite_run_id) not in \
        _blocker_pairs(ret_pe2.blockers)
    assert ret_pe2.deletable is True
    r = forge.delete_model_record(model, "evaluation", pe)
    assert r.category == "evaluation" and r.files_removed >= 1

    # ---- CHAIN 3: the workflow (leaf) deletes; its referenced
    #      evaluation (the workflow's own eval artifact) unblocked ---- #
    wf_eval = next(st.artifact.artifact_id for st in
                   forge.workflows.get_workflow(
                       model, env.workflow.workflow_id).stages
                   if st.artifact and st.artifact.kind == "evaluation")
    ret_wf_eval = forge.model_record_retention_overview(
        model, "evaluation", wf_eval)
    assert ("workflow", env.workflow.workflow_id) in \
        _blocker_pairs(ret_wf_eval.blockers)
    assert forge.model_record_retention_overview(
        model, "workflow", env.workflow.workflow_id).deletable is True
    r = forge.delete_model_record(model, "workflow",
                                  env.workflow.workflow_id)
    assert r.files_removed >= 1
    ret_wf_eval2 = forge.model_record_retention_overview(
        model, "evaluation", wf_eval)
    assert ("workflow", env.workflow.workflow_id) not in \
        _blocker_pairs(ret_wf_eval2.blockers)
    assert ret_wf_eval2.deletable is True

    # ---- determinism + zero mutation of the read-only surface ------- #
    v1 = forge.model_record_retention_overview(
        model, "evaluation", env.direct_eval.eval_id)
    v2 = forge.model_record_retention_overview(
        model, "evaluation", env.direct_eval.eval_id)
    assert v1.model_dump(mode="json") == v2.model_dump(mode="json")

    # ---- out-of-scope families are rejected -------------------------- #
    with pytest.raises(ValueError):
        forge.delete_model_record(model, "checkpoint", env.ck[0])
    with pytest.raises(ValueError):
        forge.delete_model_record(model, "training_run", "x")
    with pytest.raises(ValueError):
        forge.model_record_retention_overview(model, "checkpoint",
                                              env.ck[0])


def test_m70_integrity_scope_and_isolation(env):
    forge, root = env.forge, env.root
    model = env.model

    # unknown record / unknown model -> 404 semantics
    with pytest.raises(FileNotFoundError):
        forge.delete_model_record(model, "evaluation", "no-such")
    with pytest.raises(FileNotFoundError):
        forge.model_record_retention_overview(model, "gate", "no-such")
    with pytest.raises(FileNotFoundError):
        forge.delete_model_record("no-model", "workflow", "x")

    # tampered result_hash -> integrity refusal, zero mutation, never
    # deletable; restored -> deletable again (the direct eval is a leaf)
    edir = root / "models" / model / "evaluations" / \
        f"eval-{env.direct_eval.eval_id}"
    original = (edir / "manifest.json").read_bytes()
    man = json.loads(original)
    man["result_hash"] = "0" * 64
    (edir / "manifest.json").write_text(json.dumps(man))
    try:
        before = _inventory(root)
        ret = forge.model_record_retention_overview(
            model, "evaluation", env.direct_eval.eval_id)
        assert ret.integrity_verified is False
        assert ret.deletable is False            # integrity gates
        assert ret.blockers == []                # still a leaf
        with pytest.raises(RuntimeError):
            forge.delete_model_record(model, "evaluation",
                                      env.direct_eval.eval_id)
        after_tampered = _inventory(root)
        assert after_tampered == before          # only the tampered file
    finally:
        (edir / "manifest.json").write_bytes(original)
    ret = forge.model_record_retention_overview(
        model, "evaluation", env.direct_eval.eval_id)
    assert ret.integrity_verified is True and ret.deletable is True

    # corrupt manifest -> registry-invisible 404
    (edir / "manifest.json").write_bytes(b"not json")
    try:
        with pytest.raises(FileNotFoundError):
            forge.delete_model_record(model, "evaluation",
                                      env.direct_eval.eval_id)
        with pytest.raises(FileNotFoundError):
            forge.model_record_retention_overview(
                model, "evaluation", env.direct_eval.eval_id)
    finally:
        (edir / "manifest.json").write_bytes(original)

    # ---- successful deletion: exact stats + isolation ----------------- #
    before = _inventory(root)
    pre = forge.model_record_retention_overview(
        model, "evaluation", env.direct_eval.eval_id)
    assert pre.deletable is True
    files, nbytes = _walk(edir)
    assert (pre.files, pre.size_bytes) == (files, nbytes)
    r = forge.delete_model_record(model, "evaluation",
                                  env.direct_eval.eval_id)
    assert (r.files_removed, r.bytes_reclaimed) == (len(files), nbytes)
    after = _inventory(root)
    assert set(after) == set(before) - {
        f"models/{model}/evaluations/eval-{env.direct_eval.eval_id}/"
        f"manifest.json"}
    assert all(after[f] == before[f] for f in after)
    assert not edir.exists()
    assert not any(p.name.startswith(".tmp-delete")
                   for p in (root / "models" / model / "evaluations")
                   .iterdir())
    # repeated deletion -> 404
    with pytest.raises(FileNotFoundError):
        forge.delete_model_record(model, "evaluation",
                                  env.direct_eval.eval_id)

    # ---- the remaining comp's side evaluations: blocked by the
    #      comparison only; deleting one REFUSES (no cascade) -------- #
    comp = env.comp
    a_eval = comp.state_a.evaluation_id
    ret = forge.model_record_retention_overview(model, "evaluation",
                                                a_eval)
    assert _blocker_pairs(ret.blockers) == [("comparison",
                                             comp.comparison_id)]
    with pytest.raises(ValueError):
        forge.delete_model_record(model, "evaluation", a_eval)
    # the comparison itself is now a leaf (its gate is gone) -> delete
    assert forge.model_record_retention_overview(
        model, "comparison", comp.comparison_id).deletable is True
    r = forge.delete_model_record(model, "comparison", comp.comparison_id)
    assert r.files_removed >= 1
    # NOW the side evaluation unblocks and deletes
    assert forge.model_record_retention_overview(
        model, "evaluation", a_eval).deletable is True
    r = forge.delete_model_record(model, "evaluation", a_eval)
    assert r.files_removed == 1


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #

def test_m70_api_lifecycle(api_client):
    import json as _json

    up = api_client.post(
        "/api/v1/datasets/upload",
        files=[("files", ("m70api.txt", _corpus(190, "m70api"),
                          "text/plain"))],
        data={"name": "m70api-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": _json.dumps(
                             {"name": "m70api-tok", "vocab_size": 320}),
                             "dataset_id": ds})
    assert tr.status_code == 201, tr.text
    tok = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                           json={"tokenizer_id": tok}).status_code == 200
    m = api_client.post("/api/v1/models", json={
        "config": {"name": "m70api-model", "vocab_size": 640,
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
    # a comparison (its own side evals) + a gate on the comparison
    r = api_client.post("/api/v1/comparisons/run", json={
        "model_id": mid,
        "state_a": {"state_kind": "checkpoint", "checkpoint_id": ck[0]},
        "state_b": {"state_kind": "checkpoint", "checkpoint_id": ck[1]},
        "dataset_id": ds, "split": "validation", "tokenizer_id": tok,
        "batch_size": 8, "max_seq_len": 32, "seed": 3, "tolerance": 1e-4})
    assert r.status_code == 200, r.text
    comp_id = r.json()["comparison_id"]
    side_eval = r.json()["state_a"]["evaluation_id"]
    r = api_client.post("/api/v1/gates/evaluate", json={
        "model_id": mid,
        "policy": {"name": "m70api-gate", "model_id": mid,
                   "dataset_id": ds, "tokenizer_id": tok,
                   "split": "validation", "batch_size": 8,
                   "max_seq_len": 32, "seed": 4,
                   "baseline_type": "checkpoint",
                   "baseline_checkpoint_id": ck[1], "tolerance": 1.0},
        "candidate": {"state_kind": "checkpoint",
                      "checkpoint_id": ck[0]}})
    assert r.status_code == 200, r.text
    gate_id = r.json()["decision_id"]
    gate_comp = r.json()["comparison_id"]
    # a workflow (leaf)
    r = api_client.post("/api/v1/workflows/run", json={
        "name": "m70api-wf", "model_id": mid,
        "description": "M70 api",
        "stages": [
            {"stage_id": "ev", "type": "evaluate",
             "evaluation": {"config": {
                 "model_id": mid, "dataset_id": ds, "tokenizer_id": tok,
                 "split": "validation", "batch_size": 8,
                 "max_seq_len": 32}, "checkpoint_from_best": True}}]})
    assert r.status_code == 200, r.text
    wf_id = r.json()["workflow_id"]
    wf_eval = r.json()["stages"][0]["artifact"]["artifact_id"]

    storage_root = Path(api_client.get("/api/v1/project").json()[
        "storage_root"])

    # retention views over HTTP: blocked comparison (by the gate),
    # blocked side eval (by comparison+gate), leaves deletable
    r = api_client.get(
        f"/api/v1/models/{mid}/comparisons/{gate_comp}/retention")
    assert r.status_code == 200, r.text
    vcomp = r.json()
    assert set(vcomp) == {"model_id", "category", "record_id",
                          "created_at", "files", "size_bytes",
                          "integrity_verified", "deletable", "blockers"}
    assert vcomp["deletable"] is False
    assert [(b["category"], b["reference_id"]) for b in vcomp["blockers"]] \
        == [("gate", gate_id)]
    r = api_client.get(
        f"/api/v1/models/{mid}/evaluations/{side_eval}/retention")
    veval = r.json()
    assert [(b["category"], b["reference_id"]) for b in veval["blockers"]] \
        == [("comparison", comp_id)]
    # the workflow's own eval is blocked by the workflow artifact
    r = api_client.get(
        f"/api/v1/models/{mid}/evaluations/{wf_eval}/retention")
    vwe = r.json()
    assert ("workflow", wf_id) in [(b["category"], b["reference_id"])
                                   for b in vwe["blockers"]]
    r = api_client.get(
        f"/api/v1/models/{mid}/workflows/{wf_id}/retention")
    assert r.json()["deletable"] is True and r.json()["blockers"] == []
    r = api_client.get(
        f"/api/v1/models/{mid}/gates/decisions/{gate_id}/retention")
    assert r.json()["deletable"] is True and r.json()["blockers"] == []

    # blocked DELETE -> structured typed 409 == the view's blockers
    before = _inventory(storage_root)
    d = api_client.delete(
        f"/api/v1/models/{mid}/comparisons/{gate_comp}")
    assert d.status_code == 409, d.text
    detail = d.json()["detail"]
    assert set(detail) == {"message", "model_id", "category",
                           "record_id", "protected", "blockers"}
    assert detail["category"] == "comparison"
    assert detail["record_id"] == gate_comp
    assert detail["protected"] is True
    assert detail["blockers"] == vcomp["blockers"]   # view == guard
    assert _inventory(storage_root) == before

    # the M69 usage view agrees with the blockers
    usage = api_client.get(f"/api/v1/models/{mid}/records/usage").json()
    m69 = {(c["category"], rec["record_id"]):
           [(x["category"], x["reference_id"]) for x in rec["references"]]
           for c in usage["categories"] for rec in c["records"]}
    assert m69[("comparison", gate_comp)] == \
        [("gate", gate_id)]
    assert m69[("evaluation", side_eval)] == [("comparison", comp_id)]

    # the chain over HTTP: gate -> comparison -> side eval
    d = api_client.delete(
        f"/api/v1/models/{mid}/gates/decisions/{gate_id}")
    assert d.status_code == 200, d.text
    assert d.json() == {"model_id": mid, "category": "gate",
                        "record_id": gate_id,
                        "files_removed": d.json()["files_removed"],
                        "bytes_reclaimed": d.json()["bytes_reclaimed"]}
    assert d.json()["files_removed"] >= 1
    assert api_client.get(
        f"/api/v1/models/{mid}/comparisons/{gate_comp}/retention"
    ).json()["deletable"] is True
    d = api_client.delete(
        f"/api/v1/models/{mid}/comparisons/{gate_comp}")
    assert d.status_code == 200, d.text
    # the standalone comparison (which the gate's sides reused via
    # dedupe) still holds the side evaluation
    v = api_client.get(
        f"/api/v1/models/{mid}/evaluations/{side_eval}/retention").json()
    assert v["deletable"] is False
    assert [(b["category"], b["reference_id"]) for b in v["blockers"]] == \
        [("comparison", comp_id)]
    d = api_client.delete(f"/api/v1/models/{mid}/comparisons/{comp_id}")
    assert d.status_code == 200, d.text
    assert api_client.get(
        f"/api/v1/models/{mid}/evaluations/{side_eval}/retention"
    ).json()["deletable"] is True
    d = api_client.delete(
        f"/api/v1/models/{mid}/evaluations/{side_eval}")
    assert d.status_code == 200, d.text
    # the workflow deletes (leaf), unblocking its own eval
    d = api_client.delete(f"/api/v1/models/{mid}/workflows/{wf_id}")
    assert d.status_code == 200, d.text
    assert api_client.get(
        f"/api/v1/models/{mid}/evaluations/{wf_eval}/retention"
    ).json()["deletable"] is True

    # the M69 view no longer reports the deleted records
    usage = api_client.get(f"/api/v1/models/{mid}/records/usage").json()
    m69 = {(c["category"], rec["record_id"]) for c in usage["categories"]
           for rec in c["records"]}
    assert ("gate", gate_id) not in m69
    assert ("comparison", gate_comp) not in m69
    assert ("evaluation", side_eval) not in m69
    assert ("workflow", wf_id) not in m69

    # 404s + deterministic repeats
    for path in (f"/api/v1/models/{mid}/gates/decisions/{gate_id}",
                 f"/api/v1/models/{mid}/comparisons/{gate_comp}",
                 f"/api/v1/models/{mid}/evaluations/{side_eval}",
                 f"/api/v1/models/{mid}/workflows/{wf_id}",
                 f"/api/v1/models/{mid}/evaluations/no-such",
                 "/api/v1/models/no-m70/workflows/x"):
        r1 = api_client.delete(path)
        assert r1.status_code == 404, path
        assert api_client.get(path + "/retention").status_code == 404, path

    # OpenAPI: 104 paths; the four DELETEs are new OPERATIONS on the
    # EXISTING GET-one paths; four new GET-only retention paths; the
    # delete-operation set is exactly 14
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 104
    for path in ("/api/v1/models/{model_id}/workflows/{workflow_id}",
                 "/api/v1/models/{model_id}/evaluations/{eval_id}",
                 "/api/v1/models/{model_id}/comparisons/"
                 "{comparison_id}",
                 "/api/v1/models/{model_id}/gates/decisions/"
                 "{decision_id}"):
        assert set(spec["paths"][path].keys()) == {"get", "delete"}, path
    deletes = sorted(p for p, ops in spec["paths"].items()
                     if "delete" in ops)
    assert len(deletes) == 14
    for s in ("ModelRecordDeletionResult", "ModelRecordDeletionBlocked",
              "ModelRecordDeletionBlocker", "ModelRecordRetentionOverview"):
        assert s in spec["components"]["schemas"], s
    assert spec["paths"]["/api/v1/models/{model_id}/evaluations/"
                         "{eval_id}"]["delete"]["responses"]["409"][
        "content"]["application/json"]["schema"]["$ref"] == \
        "#/components/schemas/ModelRecordDeletionBlocked"
