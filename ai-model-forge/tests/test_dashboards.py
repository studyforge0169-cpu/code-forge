"""Milestone 8 tests: read-only dashboard engine over immutable histories.

Module-scoped environment: one domain, one tokenizer, one model trained in
two runs (real M3 checkpoints + provenance), then populated with real
evaluations, comparisons, gate decisions and workflow runs through the
existing engines — the dashboard then aggregates exactly that evidence.
Pass/fail thresholds of gates are derived from losses actually measured by
the engines in the same root, so every expected verdict is deterministic.

Coverage: empty history; per-family series/grouping/ordering semantics;
artifact reference graph; cross-family id consistency; determinism
(hash-stable, filesystem-order independent); corruption resilience with
deterministic diagnostics; byte-level read-only audit; unknown-model 404 and
corrupt-model-manifest behaviour.
"""
from __future__ import annotations

import json
import random

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
    StageStateRef,
    StageType,
    SuiteProbe,
    SuiteRunRequest,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
    WorkflowEvaluationStage,
    WorkflowGateStage,
    WorkflowPlan,
    WorkflowRecipeCreateRequest,
    WorkflowStage,
    WorkflowSuiteRunStage,
)

HEADS = ("river mountain cloud forest desert ocean valley island meadow canyon "
         "table chair lamp desk shelf couch rug clock mirror vase").split()
TAIL_A = ("flows stands gleams rises falls drifts looms shines hides waits").split()


def _domain_bytes(tails, n: int = 240) -> bytes:
    rng = random.Random(11)
    out = []
    for i in range(n):
        h1, h2 = rng.sample(HEADS, 2)
        out.append(f"{h1} is {rng.choice(tails)} near {h2} with number {i}")
    return ("\n\n".join(out) + "\n").encode("utf-8")


def _probe(ds: str, tok: str, seed: int) -> dict:
    return dict(dataset_id=ds, split="validation", tokenizer_id=tok,
                batch_size=8, max_seq_len=32, seed=seed)


def _sp_epochs_per_step(forge, ds: str, tok: str) -> int:
    tokens = forge.datasets.tokenized_artifact(ds, 1, tok)[0] \
        .splits["train"].count
    return max(1, (tokens // 32) // 8)


class Env:
    """One temp root with a model that owns a real M3–M7 history."""

    def __init__(self, root):
        self.forge = ModelForge(root=root)

    def prepare(self):
        f = self.forge
        up = f.upload_dataset([("a.txt", _domain_bytes(TAIL_A))], name="m8-dom")
        self.ds = up["dataset_id"]
        tok = f.train_tokenizer(TokenizerConfig(name="m8-tok", vocab_size=600),
                                dataset_id=self.ds)
        self.tok = tok.id
        f.tokenize_dataset(self.ds, tok.id)
        cfg = TransformerConfig(name="m8-main", vocab_size=640, context_length=64,
                                hidden_size=64, n_layers=2, n_heads=4,
                                n_kv_heads=2, intermediate_size=128, seed=1)
        self.model_id = f.create_model(ModelCreateRequest(config=cfg))[0].id
        every = _sp_epochs_per_step(f, self.ds, self.tok) // 2
        self.every = max(1, every)

        def train(epochs: int, seed: int):
            return f.run_training(TrainingConfig(
                method="continued_pretraining", model_id=self.model_id,
                dataset_id=self.ds, tokenizer_id=self.tok, learning_rate=3e-3,
                batch_size=8, max_seq_len=32, epochs=epochs,
                eval_every_steps=self.every, keep_best=False, seed=seed))

        r1 = train(4, seed=1)               # run 1 -> lineage chain
        r2 = train(5, seed=1)               # run 2 -> longer chain
        self.run1_final = r1.checkpoints[-1]["checkpoint_id"]
        self.last_ck = r2.checkpoints[-1]["checkpoint_id"]

        # ---- explicit evaluations (variety + duplicate probe + measured) --
        def evl(ckpt, seed: int):
            cfg = dict(model_id=self.model_id, checkpoint_id=ckpt,
                       **_probe(self.ds, self.tok, seed))
            return f.run_evaluation(EvaluationConfig(**cfg)).loss_nats

        evl(self.last_ck, 111)               # first of a duplicate pair
        evl(self.last_ck, 111)               # second duplicate record
        evl(self.run1_final, 112)            # earlier-state probe (comp 121)
        f.run_evaluation(EvaluationConfig(model_id=self.model_id,
                                          **_probe(self.ds, self.tok, 113)))
        # measured losses feed deterministic gate/workflow thresholds
        Ls = {seed: evl(self.last_ck, seed) for seed in (131, 132, 141, 142,
                                                         143)}
        assert all(v > 1e-6 for v in Ls.values()), \
            f"unexpected near-zero loss {Ls} (fixture thresholds invalid)"

        # ---- comparisons: changed pair / current-vs-ckpt / identical pair --
        f.run_comparison(ComparisonRequest(
            model_id=self.model_id, state_a=self.cstate(self.run1_final),
            state_b=self.cstate(self.last_ck), tolerance=1e-4,
            **_probe(self.ds, self.tok, 112)))
        f.run_comparison(ComparisonRequest(
            model_id=self.model_id, state_a=self.cstate(current=True),
            state_b=self.cstate(self.last_ck), tolerance=1e-4,
            **_probe(self.ds, self.tok, 113)))
        f.run_comparison(ComparisonRequest(
            model_id=self.model_id, state_a=self.cstate(self.last_ck),
            state_b=self.cstate(self.last_ck), tolerance=0.0,
            **_probe(self.ds, self.tok, 121)))

        # ---- gate decisions: pass / fail / checkpoint-baseline pass --------
        def gate(name: str, seed: int, minimum_loss=None, baseline_ck=None,
                 max_regression_delta=None, candidate_ck=None):
            pol = dict(name=name, model_id=self.model_id,
                       baseline_type=("minimum_loss" if baseline_ck is None
                                      else "checkpoint"),
                       **_probe(self.ds, self.tok, seed))
            if minimum_loss is not None:
                pol["minimum_loss"] = minimum_loss
            if baseline_ck is not None:
                pol["baseline_checkpoint_id"] = baseline_ck
            if max_regression_delta is not None:
                pol["max_regression_delta"] = max_regression_delta
            return f.run_gate(GateRequest(
                model_id=self.model_id, policy=GatePolicy(**pol),
                candidate=self.cstate(candidate_ck or self.last_ck)))

        gate("m8-gate-pass", 131, minimum_loss=Ls[131] + 1.0)
        gate("m8-gate-fail", 132, minimum_loss=Ls[132] * 0.5)
        gate("m8-gate-baseline", 133, baseline_ck=self.run1_final,
             max_regression_delta=1e12, candidate_ck=self.last_ck)

        # ---- workflow runs: completed / stopped / branched / failed --------
        f.run_workflow(self._wf_plan("m8-wf1", [
            self.eval_stage("s1", ckpt=self.last_ck, seed=141),
            self.gate_stage("g1", ckpt=self.last_ck, seed=141,
                            minimum_loss=Ls[141] + 1.0)]))
        f.run_workflow(self._wf_plan("m8-wf2", [
            self.gate_stage("g1", ckpt=self.last_ck, seed=142,
                            minimum_loss=Ls[142] * 0.5)]))
        f.run_workflow(self._wf_plan("m8-wf3", [
            self.gate_stage("g1", ckpt=self.last_ck, seed=143,
                            minimum_loss=Ls[143] * 0.5, on_fail="s3"),
            self.eval_stage("s3", ckpt=self.last_ck, seed=143)]))
        bad = self._wf_plan("m8-wf4", [self.eval_stage("s1", ckpt="ghost-ck")])
        try:
            f.run_workflow(bad)
        except Exception:                                 # recorded + re-raised
            pass
        failed = [r for r in f.list_workflows(self.model_id)
                  if r.status.value == "failed"]
        assert len(failed) == 1 and failed[0].failed_stage_id == "s1", \
            "expected the ghost-checkpoint workflow to persist a failed run"
        self.wf_counts = {"completed": 2, "stopped": 1, "failed": 1}

        # ---- M16 sample-quality history (real samples + measurements) -----
        def sample(prompt: str, n: int) -> str:
            return f.samples.run(SampleGenerateRequest(
                model_id=self.model_id, checkpoint_id=self.last_ck,
                tokenizer_id=self.tok, prompt=prompt, strategy="greedy",
                max_new_tokens=n)).sample_id

        self.sample_a = sample("river mountain cloud forest ocean", 6)
        self.sample_b = sample("desert valley island meadow canyon lamp", 4)
        # two measurements of sample_a, one of sample_b -> 3 records
        self.sq_eval_ids = []
        for sid in (self.sample_a, self.sample_a, self.sample_b):
            self.sq_eval_ids.append(
                f.evaluate_sample(self.model_id, sid).evaluation_id)
        assert len(f.list_sample_evaluations(self.model_id)) == 3

    # ---- workflow plan helpers ------------------------------------------

    def _wf_plan(self, name: str, stages, model_id=None) -> WorkflowPlan:
        return WorkflowPlan(name=name,
                            model_id=model_id or self.model_id, stages=stages)

    def eval_stage(self, sid: str, *, ckpt=None, seed: int = 1,
                   from_stage: str | None = None,
                   model_id: str | None = None) -> WorkflowStage:
        mid = model_id or self.model_id
        cfg = dict(model_id=mid, **_probe(self.ds, self.tok, seed))
        if ckpt is not None:
            cfg["checkpoint_id"] = ckpt
        return WorkflowStage(
            stage_id=sid, type=StageType.EVALUATE,
            evaluation=WorkflowEvaluationStage(
                config=EvaluationConfig(**cfg),
                checkpoint_from_stage=from_stage))

    def gate_stage(self, sid: str, *, ckpt=None, seed: int = 1,
                   minimum_loss: float = None, on_fail: str | None = None,
                   on_pass: str | None = None,
                   model_id: str | None = None) -> WorkflowStage:
        mid = model_id or self.model_id
        pol = dict(name=f"wf-gate-{sid}", model_id=mid,
                   baseline_type="minimum_loss",
                   **_probe(self.ds, self.tok, seed))
        if minimum_loss is not None:
            pol["minimum_loss"] = minimum_loss
        st = WorkflowStage(
            stage_id=sid, type=StageType.GATE,
            gate=WorkflowGateStage(
                policy=GatePolicy(**pol),
                candidate=StageStateRef(state_kind=EvalStateKind.CHECKPOINT,
                                        checkpoint_id=ckpt)))
        if on_pass:
            st.on_pass = on_pass
        if on_fail:
            st.on_fail = on_fail
        return st

    def fresh_model(self, name: str = "m8-fresh", seed: int = 5) -> str:
        return self.forge.create_model(ModelCreateRequest(config=TransformerConfig(
            name=name, vocab_size=640, context_length=64, hidden_size=64,
            n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
            seed=seed)))[0].id

    def cstate(self, ckpt=None, current=False) -> ComparisonState:
        if current:
            return ComparisonState(state_kind=EvalStateKind.CURRENT)
        return ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                               checkpoint_id=ckpt)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m8-root"))
    e.prepare()
    return e


# =========================================================================== #
# Empty history / unknown model
# =========================================================================== #

def test_empty_history_is_valid_and_deterministic(env):
    mid = env.fresh_model("m8-empty")
    d1 = env.forge.get_dashboard(mid)
    j = d1.model_dump(mode="json")
    assert j["model_id"] == mid and j["schema_version"] == 1
    assert j["summary"]["latest_checkpoint"] is None
    assert j["checkpoints"] == [] and j["training_runs"] == []
    assert j["evaluations"] == [] and j["comparisons"] == []
    assert j["gate_decisions"] == [] and j["workflows"]["counts"] == {}
    assert j["workflows"]["records"] == []
    assert j["artifact_graph"]["nodes"] == []
    assert j["artifact_graph"]["edges"] == []
    assert j["diagnostics"] == []
    assert len(d1.result_hash) == 64
    d2 = env.forge.get_dashboard(mid)
    assert d2.model_dump(mode="json") == j
    assert d1.result_hash == d2.result_hash


def test_unknown_model_raises_not_found(env):
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.get_dashboard("ghost-model")


# =========================================================================== #
# Sections over the populated history (Env.prepare)
# =========================================================================== #

def test_dashboard_sections_present(env):
    f, mid = env.forge, env.model_id
    j = f.get_dashboard(mid).model_dump(mode="json")
    assert j["model_id"] == mid and j["config_hash"]
    s = j["summary"]
    assert s["id"] == mid and s["training_run_count"] == 2
    assert s["latest_checkpoint"] == env.last_ck
    assert len(j["checkpoints"]) >= 8
    assert len(j["training_runs"]) == 2
    assert j["evaluations"] and j["comparisons"] and j["gate_decisions"]
    assert j["workflows"]["counts"] == env.wf_counts
    assert len(j["workflows"]["records"]) == sum(env.wf_counts.values())
    assert j["diagnostics"] == []
    assert len(j["artifact_graph"]["nodes"]) > 0


def test_checkpoint_lineage_order_matches_engine(env):
    """Dashboard checkpoints are chronological (created_at, id) — the true
    chain order — every parent precedes its child, ids match the engine."""
    f, mid = env.forge, env.model_id
    dash = f.get_dashboard(mid)
    engine = f.list_checkpoints(mid)
    engine_ids = {c.checkpoint_id for c in engine}
    got = [c.checkpoint_id for c in dash.checkpoints]
    assert set(got) == engine_ids
    by_id = {c.checkpoint_id: c for c in engine}
    # order: (created_at, checkpoint_id), never directory order
    expected = [c.checkpoint_id for c in sorted(
        engine, key=lambda c: (c.created_at, c.checkpoint_id))]
    assert got == expected
    # parent always precedes its child in the lineage
    pos = {cid: i for i, cid in enumerate(got)}
    for cid in got:
        parent = by_id[cid].parent_checkpoint_id
        if parent is not None:
            assert pos[parent] < pos[cid]
    prov = f.get_model(mid).training_provenance
    assert [r.run_id for r in dash.training_runs] == [r.run_id for r in prov]
    # every recorded parent relationship is a graph edge
    edges = {(e.source.artifact_id, e.target.artifact_id)
             for e in dash.artifact_graph.edges if e.role == "parent"}
    run_ids = {p.run_id for p in prov}
    for c in engine:
        assert c.run_id in run_ids
        if c.parent_checkpoint_id is not None:
            assert (c.checkpoint_id, c.parent_checkpoint_id) in edges


def test_evaluation_series_group_by_exact_probe(env):
    f, mid = env.forge, env.model_id
    j = f.get_dashboard(mid).model_dump(mode="json")
    series = list(j["evaluations"])
    assert sum(g["count"] for g in series) == len(f.list_evaluations(mid))
    # the two explicit repeated evaluations share ONE series (same probe)
    dup = [g for g in series if g["identity"]["seed"] == 111]
    assert len(dup) == 1 and dup[0]["count"] == 2
    assert dup[0]["records"][0]["eval_id"] != dup[0]["records"][1]["eval_id"]
    # series order + inner record order are deterministic
    assert [g["key"] for g in series] == sorted(g["key"] for g in series)
    for g in series:
        times = [r["created_at"] for r in g["records"]]
        assert times == sorted(times)
        # no incompatible probes mixed inside one series
        rec0 = g["records"][0]
        cfg0 = rec0["config"]
        for other in g["records"][1:]:
            for k in ("state_hash", "dataset_id", "dataset_version", "split",
                      "tokenizer_id", "seed"):
                assert other[k] == rec0[k]
            cfg = other["config"]
            for k in ("max_seq_len", "max_eval_tokens", "batch_size"):
                assert cfg.get(k) == cfg0.get(k)
    # identity mirrors the exact M4 reuse fields of every member record
    for g in series:
        ident, rec = g["identity"], g["records"][0]
        assert ident["state_kind"] == rec["state_kind"]
        assert ident["checkpoint_id"] == rec["checkpoint_id"]
        assert ident["state_hash"] == rec["state_hash"]
        assert ident["dataset_id"] == rec["dataset_id"]
        assert ident["dataset_version"] == rec["dataset_version"]
        assert ident["split"] == rec["split"]
        assert ident["tokenizer_id"] == rec["tokenizer_id"]
        assert ident["seed"] == rec["seed"]
        cfg = rec["config"]
        assert ident["max_seq_len"] == cfg.get("max_seq_len")
        assert ident["max_eval_tokens"] == cfg.get("max_eval_tokens")
        assert ident["batch_size"] == cfg.get("batch_size")


def test_comparison_series_semantics(env):
    f, mid = env.forge, env.model_id
    j = f.get_dashboard(mid).model_dump(mode="json")
    series = list(j["comparisons"])
    assert sum(g["count"] for g in series) == len(f.list_comparisons(mid))
    total = [r for g in series for r in g["records"]]
    # M5 semantics preserved verbatim: delta_loss = loss_B - loss_A
    for r in total:
        assert abs(r["delta_loss_nats"] - (r["loss_b"] - r["loss_a"])) < 1e-9
        assert r["verdict"] in ("improved", "regressed", "unchanged")
        assert r["result_hash"] and r["created_at"]
    # an identical-state comparison is recorded unchanged with delta 0
    u = [r for r in total
         if r["state_a"]["checkpoint_id"] == r["state_b"]["checkpoint_id"]
         and r["state_a"]["state_hash"] == r["state_b"]["state_hash"]]
    assert u and all(abs(x["delta_loss_nats"]) < 1e-9
                     and x["verdict"] == "unchanged" for x in u)
    # identity exposes the ordered pair + probe + tolerance of the group
    for g in series:
        ident, rec = g["identity"], g["records"][0]
        assert ident["state_a"]["state_hash"] == rec["state_a"]["state_hash"]
        assert ident["state_b"]["state_hash"] == rec["state_b"]["state_hash"]
        assert ident["tolerance"] == rec["tolerance"]
        assert ident["seed"] == rec["seed"]
        assert (ident["state_a"]["state_kind"], ident["state_b"]["state_kind"]) \
            == (rec["state_a"]["state_kind"], rec["state_b"]["state_kind"])


def test_gate_series_grouped_by_recorded_policy(env):
    f, mid = env.forge, env.model_id
    j = f.get_dashboard(mid).model_dump(mode="json")
    groups = list(j["gate_decisions"])
    assert sum(sum(g["statistics"].values()) for g in groups) == \
        len(f.list_gate_decisions(mid))
    by_name = {g["policy"]["name"]: g for g in groups}
    # three explicitly named gates (fixture): pass / fail / pass
    assert by_name["m8-gate-pass"]["statistics"] == {"passed": 1}
    assert by_name["m8-gate-fail"]["statistics"] == {"failed": 1}
    assert by_name["m8-gate-baseline"]["statistics"] == {"passed": 1}
    for g in groups:
        # statistics are descriptive counts, never a quality score
        assert set(g["statistics"]) <= {"passed", "failed"}
        for r in g["records"]:
            assert r["decision"] in ("passed", "failed")
            assert r["policy"]["model_id"] == mid
            assert r["policy"]["name"] == g["policy"]["name"]
        # every decision in one group shares the same recorded policy
        base = json.dumps(g["policy"], sort_keys=True, default=str)
        for r in g["records"][1:]:
            assert json.dumps(r["policy"], sort_keys=True,
                              default=str) == base


def test_workflow_history_records_and_counts(env):
    f, mid = env.forge, env.model_id
    j = f.get_dashboard(mid).model_dump(mode="json")
    wf = j["workflows"]
    assert wf["counts"] == env.wf_counts
    engine_ids = [w.workflow_id for w in f.list_workflows(mid)]
    assert [r["workflow_id"] for r in wf["records"]] == engine_ids
    for r in wf["records"]:
        assert r["plan"]["model_id"] == mid and r["plan_hash"]
        assert [s["stage_id"] for s in r["stages"]] == \
            [s["stage_id"] for s in r["plan"]["stages"]]
    statuses = [r["status"] for r in wf["records"]]
    assert sorted(statuses) == ["completed", "completed", "failed", "stopped"]
    stopped = [r for r in wf["records"] if r["status"] == "stopped"]
    assert len(stopped) == 1
    assert stopped[0]["failed_stage_id"] is None
    assert "stopped" in stopped[0]["terminal_reason"]
    assert stopped[0]["transitions"][-1]["decision"] == "failed"
    assert stopped[0]["transitions"][-1]["to_stage"] is None
    branched = [r for r in wf["records"]
                if r["status"] == "completed"
                and any(t["decision"] == "failed" for t in r["transitions"])]
    assert len(branched) == 1          # gate failed, on_fail branch executed
    assert any(t["to_stage"] == "s3" for t in branched[0]["transitions"])
    failed = [r for r in wf["records"] if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["failed_stage_id"] == "s1"
    assert "failed" in failed[0]["terminal_reason"]
    assert failed[0]["stages"][0]["artifact"] is None
    assert failed[0]["stages"][0]["error"]


# =========================================================================== #
# Artifact graph + cross-family consistency
# =========================================================================== #

def test_artifact_graph_edges_roles(env):
    f, mid = env.forge, env.model_id
    dash = f.get_dashboard(mid)
    g = dash.artifact_graph
    roles = {e.role for e in g.edges}
    assert {"parent", "produced", "final_state"} <= roles      # runs/ckpts
    assert {"state", "state_a", "state_b"} <= roles            # evals/comps
    assert {"candidate", "evidence"} <= roles                  # gates
    assert "stage_artifact" in roles                           # workflows
    assert dash.diagnostics == []                              # healthy store
    # source and target of every edge are real nodes of the graph
    node_ids = {(n.family, n.artifact_id) for n in g.nodes}
    for e in g.edges:
        assert (e.source.family, e.source.artifact_id) in node_ids
        assert (e.target.family, e.target.artifact_id) in node_ids
    # nodes sorted canonically
    fams = [n.family for n in g.nodes]
    assert fams == sorted(fams)


def test_cross_family_id_consistency(env):
    f, mid = env.forge, env.model_id
    j = f.get_dashboard(mid).model_dump(mode="json")
    engine_lists = {
        "checkpoint": {c.checkpoint_id for c in f.list_checkpoints(mid)},
        "evaluation": {e.eval_id for e in f.list_evaluations(mid)},
        "comparison": {c.comparison_id for c in f.list_comparisons(mid)},
        "gate_decision": {d.decision_id for d in f.list_gate_decisions(mid)},
        "workflow": {w.workflow_id for w in f.list_workflows(mid)},
    }
    dashboard_ids = {
        "checkpoint": {c["checkpoint_id"] for c in j["checkpoints"]},
        "evaluation": {r["eval_id"] for g in j["evaluations"]
                       for r in g["records"]},
        "comparison": {r["comparison_id"] for g in j["comparisons"]
                       for r in g["records"]},
        "gate_decision": {r["decision_id"] for g in j["gate_decisions"]
                          for r in g["records"]},
        "workflow": {r["workflow_id"] for r in j["workflows"]["records"]},
    }
    for family in engine_lists:
        assert dashboard_ids[family] == engine_lists[family]
    # graph nodes = all visible artifacts + recorded training runs; nothing else
    node_ids = {(n["family"], n["artifact_id"])
                for n in j["artifact_graph"]["nodes"]}
    run_ids = {(p["run_id"]) for p in j["training_runs"]}
    assert node_ids == {(fam, i)
                        for fam, ids in dashboard_ids.items() for i in ids} | \
        {("training_run", rid) for rid in run_ids}
    # every exposed id resolves through the existing engine getters
    for cid in dashboard_ids["checkpoint"]:
        assert f.get_checkpoint(mid, cid).checkpoint_id == cid
    for eid in dashboard_ids["evaluation"]:
        assert f.get_evaluation(mid, eid).eval_id == eid
    for cid in dashboard_ids["comparison"]:
        assert f.get_comparison(mid, cid).comparison_id == cid
    for did in dashboard_ids["gate_decision"]:
        assert f.get_gate_decision(mid, did).decision_id == did
    for wid in dashboard_ids["workflow"]:
        assert f.get_workflow(mid, wid).workflow_id == wid


# =========================================================================== #
# Determinism + ordering
# =========================================================================== #

def test_deterministic_across_calls_and_hash_stable(env):
    f, mid = env.forge, env.model_id
    d1, d2 = f.get_dashboard(mid), f.get_dashboard(mid)
    assert d1.model_dump(mode="json") == d2.model_dump(mode="json")
    assert d1.result_hash == d2.result_hash and len(d1.result_hash) == 64
    # other read-only traffic does not perturb anything
    f.list_evaluations(mid)
    assert f.get_dashboard(mid).model_dump(mode="json") == \
        d1.model_dump(mode="json")


def test_result_hash_tracks_storage_state(env):
    """New immutable evidence changes the dashboard hash and content."""
    f = env.forge
    mid = env.fresh_model("m8-hash")
    h0 = f.get_dashboard(mid).result_hash
    f.run_evaluation(EvaluationConfig(model_id=mid, **_probe(env.ds,
                                                             env.tok, 555)))
    d1 = f.get_dashboard(mid)
    assert d1.result_hash != h0
    assert len(d1.evaluations) == 1
    assert f.get_dashboard(mid).result_hash == d1.result_hash


def test_ordering_ignores_filesystem_directory_order(env):
    """Renaming artifact directories cannot reorder dashboard output."""
    f = env.forge
    mid = env.fresh_model("m8-fsorder")
    p = _probe(env.ds, env.tok, 501)
    f.run_evaluation(EvaluationConfig(model_id=mid, **p))
    f.run_evaluation(EvaluationConfig(model_id=mid, **p))   # duplicate probe
    expected = [e.eval_id for e in f.list_evaluations(mid)]  # persisted order
    assert len(expected) == 2
    assert len({r.eval_id for g in
                f.get_dashboard(mid).evaluations for r in g.records}) == 2
    evals_dir = f.storage.model_dir(mid) / "evaluations"
    # give the EARLIER record a lexically LATER directory and vice versa
    (evals_dir / f"eval-{expected[0]}").rename(evals_dir / "eval-zzz")
    (evals_dir / f"eval-{expected[1]}").rename(evals_dir / "eval-aaa")
    names = sorted(d.name for d in evals_dir.iterdir() if d.is_dir())
    assert names == ["eval-aaa", "eval-zzz"]               # fs order REVERSED
    j = f.get_dashboard(mid).model_dump(mode="json")
    series = [g for g in j["evaluations"] if g["count"] == 2]
    assert len(series) == 1
    got = [r["eval_id"] for r in series[0]["records"]]
    assert got == expected                                 # created_at wins


# =========================================================================== #
# Corruption resilience
# =========================================================================== #

def test_corruption_resilience_with_deterministic_diagnostics(env):
    f = env.forge
    mid = env.fresh_model("m8-corrupt")
    # minimal real history on the dedicated model: one run + eval + compare
    # + gate + two workflow runs
    f.run_training(TrainingConfig(
        method="continued_pretraining", model_id=mid, dataset_id=env.ds,
        tokenizer_id=env.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, epochs=2, eval_every_steps=env.every,
        keep_best=False, seed=1))
    all_ckpts = [c.checkpoint_id for c in f.list_checkpoints(mid)]
    ck_last = all_ckpts[-1]
    ev_ok = f.run_evaluation(EvaluationConfig(
        model_id=mid, checkpoint_id=ck_last,
        **_probe(env.ds, env.tok, 601))).eval_id
    comp = f.run_comparison(ComparisonRequest(
        model_id=mid, state_a=env.cstate(current=True),
        state_b=env.cstate(ck_last), tolerance=1e-4,
        **_probe(env.ds, env.tok, 602)))
    ev_cb = comp.state_b.evaluation_id                 # eval the comp cites
    gate = f.run_gate(GateRequest(
        model_id=mid, candidate=env.cstate(ck_last),
        policy=GatePolicy(name="corr-g", model_id=mid,
                          baseline_type="minimum_loss", minimum_loss=1e9,
                          **_probe(env.ds, env.tok, 603))))
    wf1 = f.run_workflow(env._wf_plan("m8-corrupt-wf1",
                                      [env.gate_stage("g1", ckpt=ck_last,
                                                      seed=604,
                                                      minimum_loss=1e9,
                                                      model_id=mid)],
                                      model_id=mid))
    wf2 = f.run_workflow(env._wf_plan("m8-corrupt-wf2",
                                      [env.gate_stage("g1", ckpt=ck_last,
                                                      seed=605,
                                                      minimum_loss=1e9,
                                                      model_id=mid)],
                                      model_id=mid))
    wf1_gate_id = wf1.stages[0].artifact.artifact_id   # decision wf1 cites
    md = f.storage.model_dir(mid)

    # ---- introduce corruption of several flavours --------------------------
    (md / "checkpoints" / ck_last / "manifest.json").write_bytes(
        b"{corrupt json!!!")                            # checkpoint manifest
    (md / "evaluations" / f"eval-{ev_cb}" / "manifest.json").write_bytes(
        b"{corrupt json!!!")                            # eval cited by comp
    (md / "evaluations" / "eval-zzzz00000000").mkdir(exist_ok=True)
    #                                                         missing manifest
    decoy = md / "evaluations" / "eval-deadbeef0000"
    decoy.mkdir(exist_ok=True)
    decoy.joinpath("manifest.json").write_text(
        json.dumps({"not": "an evaluation"}))           # schema mismatch
    (md / "gates" / f"gate-{gate.decision_id}" / "manifest.json") \
        .write_bytes(b"not json at all")                # gate decision
    (md / "gates" / f"gate-{wf1_gate_id}" / "manifest.json") \
        .write_bytes(b"not json at all")                # decision wf1 cites
    (md / "workflows" / f"workflow-{wf2.workflow_id}" / "manifest.json") \
        .write_bytes(b"not json at all")                # one workflow run

    j1 = f.get_dashboard(mid).model_dump(mode="json")
    diags = {(d["family"], d["artifact_id"], d["issue"])
             for d in j1["diagnostics"]}
    visible_evals = {r["eval_id"] for g in j1["evaluations"]
                     for r in g["records"]}
    visible_ckpts = {c["checkpoint_id"] for c in j1["checkpoints"]}
    # corrupt artifacts are skipped, valid ones remain visible
    assert ev_cb not in visible_evals and ev_ok in visible_evals
    assert ck_last not in visible_ckpts
    assert set(all_ckpts) - {ck_last} <= visible_ckpts
    assert comp.comparison_id in {r["comparison_id"]
                                  for g in j1["comparisons"]
                                  for r in g["records"]}
    assert visible_evals == \
        {e.eval_id for e in f.list_evaluations(mid)}   # == engine view
    # workflow counts reflect the surviving record only
    assert j1["workflows"]["counts"] == {"completed": 1}
    assert {r["workflow_id"] for r in j1["workflows"]["records"]} == \
        {wf1.workflow_id}
    # unreadable manifests produce deterministic diagnostics...
    assert any(d[0] == "evaluation" and d[1] == ev_cb
               and "JSONDecodeError" in d[2] for d in diags)
    assert any(d[0] == "evaluation" and d[1] == "deadbeef0000"
               and "ValidationError" in d[2] for d in diags)
    assert any(d[0] == "evaluation" and d[1] == "zzzz00000000"
               and d[2] == "manifest.json missing" for d in diags)
    assert any(d[0] == "checkpoint" and d[1] == ck_last
               and "JSONDecodeError" in d[2] for d in diags)
    assert any(d[0] == "gate_decision" and d[1] == gate.decision_id
               and "JSONDecodeError" in d[2] for d in diags)
    assert any(d[0] == "workflow" and d[1] == wf2.workflow_id
               and "JSONDecodeError" in d[2] for d in diags)
    # ...and internal references to skipped artifacts are surfaced as
    # diagnostics (never as fabricated graph edges)
    assert any(d[0] == "comparison" and d[1] == comp.comparison_id
               and f"references missing evaluation '{ev_cb}'" in d[2]
               for d in diags)
    assert any(d[0] == "workflow" and d[1] == wf1.workflow_id
               and f"references missing gate_decision '{wf1_gate_id}'"
               in d[2] for d in diags)
    assert len(j1["artifact_graph"]["nodes"]) > 0
    # deterministic across repeated calls (diagnostics included)
    j2 = f.get_dashboard(mid).model_dump(mode="json")
    assert j2 == j1
    assert f.get_dashboard(mid).result_hash == f.get_dashboard(mid).result_hash


def test_corrupt_model_manifest_reported(env):
    f = env.forge
    mid = env.fresh_model("m8-badmanifest")
    (f.storage.model_dir(mid) / "manifest.json").write_bytes(b"{broken")
    with pytest.raises(RuntimeError, match="corrupt"):
        f.get_dashboard(mid)


# =========================================================================== #
# Read-only guarantee
# =========================================================================== #

def test_dashboard_is_read_only_byte_audit(env):
    f, mid = env.forge, env.model_id
    root = f.storage.model_dir(mid)
    before_files = sorted(p.relative_to(root).as_posix()
                          for p in root.rglob("*") if p.is_file())
    before = {p: (root / p).read_bytes() for p in before_files}
    other = env.fresh_model("m8-ro-extra")
    for _ in range(2):
        f.get_dashboard(mid)
        f.get_dashboard(other)                            # other model too
    after_files = sorted(p.relative_to(root).as_posix()
                         for p in root.rglob("*") if p.is_file())
    assert after_files == before_files                     # zero new files
    for p in before_files:
        assert (root / p).read_bytes() == before[p]        # byte-identical
    tmp = [p for p in root.rglob("*") if ".tmp" in p.name]
    assert tmp == []


def test_dashboard_readonly_when_history_large(env):
    """Repeated reads across all five families never add artifacts."""
    f, mid = env.forge, env.model_id
    root = f.storage.model_dir(mid)
    dirs = {sub: len([p for p in (root / sub).iterdir() if p.is_dir()])
            for sub in ("checkpoints", "evaluations", "comparisons", "gates",
                        "workflows")}
    assert all(v >= 1 for v in dirs.values())
    for _ in range(3):
        assert f.get_dashboard(mid).diagnostics == []
    assert {sub: len([p for p in (root / sub).iterdir() if p.is_dir()])
            for sub in dirs} == dirs


# =========================================================================== #
# M13: dashboard suite-run section + suite_run graph support (read-only)
#
# SuiteEnv builds three models with no training at all (suite stages bind
# the CURRENT persisted weights): model A owns 4 suite runs (two standalone,
# one inline workflow suite stage, one M12 recipe-bound run), model B owns
# one standalone run, model C owns none. The dashboard reads the
# storage-root ``suite-runs/`` family and filters by the persisted
# ``model_id`` — exactly the M10 records, never a copy under model dirs.
# =========================================================================== #

class SuiteEnv:
    def __init__(self, root):
        self.forge = ModelForge(root=root)

    def prepare(self):
        f = self.forge
        up = f.upload_dataset([("a.txt", _domain_bytes(TAIL_A))],
                              name="m13-dom")
        self.ds = up["dataset_id"]
        tok = f.train_tokenizer(TokenizerConfig(name="m13-tok",
                                                vocab_size=600),
                                dataset_id=self.ds)
        self.tok = tok.id
        f.tokenize_dataset(self.ds, tok.id)

        def model(name: str, seed: int) -> str:
            return f.create_model(ModelCreateRequest(config=TransformerConfig(
                name=name, vocab_size=640, context_length=64, hidden_size=64,
                n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
                seed=seed)))[0].id

        self.model_a = model("m13-main", 1)
        self.model_b = model("m13-b", 2)
        self.model_c = model("m13-c", 3)
        self.suite_id = "m13-suite"
        f.register_probe_suite(ProbeSuiteCreateRequest(
            suite_id=self.suite_id,
            probes=[SuiteProbe(dataset_id=self.ds, split="validation",
                               tokenizer_id=self.tok, batch_size=8,
                               max_seq_len=32, seed=s) for s in (93001, 93002)]))

        cur = ComparisonState(state_kind=EvalStateKind.CURRENT)
        # 1 + 2: standalone suite runs — first creates the two exact M4
        # evaluations, second reuses them
        self.sr_created = f.run_suite(SuiteRunRequest(
            model_id=self.model_a, suite_id=self.suite_id, state=cur))
        self.sr_reused = f.run_suite(SuiteRunRequest(
            model_id=self.model_a, suite_id=self.suite_id, state=cur))
        # 3: inline workflow whose only stage is a suite_run (no recipe
        # provenance — plan is inline, provenance fields stay null)
        stage = WorkflowStage(
            stage_id="s1", type=StageType.SUITE_RUN,
            suite_run=WorkflowSuiteRunStage(
                suite_id=self.suite_id,
                state=StageStateRef(state_kind=EvalStateKind.CURRENT)))
        self.wf_inline = f.run_workflow(WorkflowPlan(
            name="m13-inline", model_id=self.model_a, stages=[stage]))
        # 4: the same stage shape executed through a registered recipe
        # (M12 provenance: recipe_id + recipe_hash recorded on the run)
        f.register_workflow_recipe(WorkflowRecipeCreateRequest(
            recipe_id="m13-rec", stages=[stage]))
        self.wf_recipe = f.run_workflow_recipe("m13-rec", self.model_a)
        # model B: one standalone run; model C: none at all
        f.run_suite(SuiteRunRequest(model_id=self.model_b,
                                    suite_id=self.suite_id, state=cur))

    def suite_root(self):
        return self.forge.storage.root / "suite-runs"

    def suite_manifest(self, suite_run_id: str) -> dict:
        import json as _json
        p = self.suite_root() / suite_run_id / "manifest.json"
        return _json.loads(p.read_text())


@pytest.fixture(scope="module")
def sr_env(tmp_path_factory):
    e = SuiteEnv(tmp_path_factory.mktemp("m13-dash-root"))
    e.prepare()
    return e


def test_suite_runs_section_matches_engine_records_exactly(sr_env):
    f, mid = sr_env.forge, sr_env.model_a
    j = f.get_dashboard(mid).model_dump(mode="json")
    sec = j["suite_runs"]
    # additive section, never null: counts + records, like workflows
    assert list(sec.keys()) == ["counts", "records"]
    assert sec["counts"] == {"completed": 4}
    engine = sorted(f.list_suite_runs(mid),
                    key=lambda r: (r.created_at, r.suite_run_id))
    assert [r["suite_run_id"] for r in sec["records"]] ==         [r.suite_run_id for r in engine]
    for rec in sec["records"]:
        assert rec["model_id"] == mid and rec["suite_id"] == "m13-suite"
        assert rec["status"] == "completed" and rec["probe_count"] == 2
        assert rec["completed_count"] == 2
        assert rec["reused_count"] + rec["failed_count"] >= 0
        assert len(rec["results"]) == 2
        for p in rec["results"]:
            assert p["outcome"] in ("created", "reused")
            assert p["evaluation_id"] and p["error"] is None
            assert p["probe"]["seed"] in (93001, 93002)
    # records are the persisted manifests verbatim (per-probe fields intact)
    for rec in sec["records"]:
        engine_rec = f.get_suite_run(mid, rec["suite_run_id"])
        assert rec == engine_rec.model_dump(mode="json")
    # execution bookkeeping preserved exactly as recorded
    by_id = {r["suite_run_id"]: r for r in sec["records"]}
    created = by_id[sr_env.sr_created.suite_run_id]
    reused = by_id[sr_env.sr_reused.suite_run_id]
    assert [p["outcome"] for p in created["results"]] ==         ["created", "created"]
    assert [p["outcome"] for p in reused["results"]] == ["reused", "reused"]
    assert created["results"][0]["evaluation_id"] == \
        reused["results"][0]["evaluation_id"]
    # no invented score / quality interpretation anywhere in the section
    blob = json.dumps(sec)
    for banned in ("score", "quality", "rank", "benchmark", "leaderboard",
                   "pass_rate", "accuracy"):
        assert banned not in blob
    # architecture: suite runs live at the storage root only — never copied
    # under a model dir
    assert not (f.storage.model_dir(mid) / "suite-runs").exists()
    sr_dirs = sorted(p.name for p in sr_env.suite_root().iterdir()
                     if p.is_dir())
    assert len(sr_dirs) == 5                      # 4 x A + 1 x B


def test_suite_runs_model_filtering_and_empty_section(sr_env):
    f = sr_env.forge
    a_ids = {r.suite_run_id
             for r in f.get_dashboard(sr_env.model_a).suite_runs.records}
    b = f.get_dashboard(sr_env.model_b).model_dump(mode="json")
    c = f.get_dashboard(sr_env.model_c).model_dump(mode="json")
    # model B: only its own run; nothing from A leaks in
    b_sec = b["suite_runs"]
    assert b_sec["counts"] == {"completed": 1}
    assert len(b_sec["records"]) == 1
    assert b_sec["records"][0]["model_id"] == sr_env.model_b
    assert b_sec["records"][0]["suite_run_id"] not in a_ids
    # model C: deterministic empty section, never null/omitted
    assert c["suite_runs"] == {"counts": {}, "records": []}
    c2 = f.get_dashboard(sr_env.model_c).model_dump(mode="json")
    assert c2 == c
    # suite-run family absent from a suite-less model's graph
    assert all(n["family"] != "suite_run"
               for n in c["artifact_graph"]["nodes"])
    assert c["diagnostics"] == []
    # engine parity for both models
    for mid, want in ((sr_env.model_b, 1), (sr_env.model_c, 0)):
        engine = f.list_suite_runs(mid)
        assert len(engine) == want
        dash_ids = [r.suite_run_id
                    for r in f.get_dashboard(mid).suite_runs.records]
        assert dash_ids == [r.suite_run_id for r in sorted(
            engine, key=lambda r: (r.created_at, r.suite_run_id))]


def test_suite_runs_ordering_and_full_json_determinism(sr_env):
    f, mid = sr_env.forge, sr_env.model_a
    d1 = f.get_dashboard(mid).model_dump(mode="json")
    d2 = f.get_dashboard(mid).model_dump(mode="json")
    assert d1 == d2 and d1["result_hash"] == d2["result_hash"]
    engine = f.list_suite_runs(mid)
    ids = [r["suite_run_id"] for r in d1["suite_runs"]["records"]]
    assert ids == [r.suite_run_id for r in sorted(
        engine, key=lambda r: (r.created_at, r.suite_run_id))]
    assert ids == [r.suite_run_id for r in sorted(
        engine, key=lambda r: (r.created_at.isoformat(),
                               r.suite_run_id))]
    # existing M8 sections keep their shape next to the new one
    assert set(d1["workflows"].keys()) == {"counts", "records"}
    assert "suite_runs" in d1 and "artifact_graph" in d1


def test_suite_runs_ordering_ignores_filesystem_order(sr_env):
    f, mid = sr_env.forge, sr_env.model_a
    # rename nothing: instead add valid suite-run manifests whose directory
    # names sort *before/after* every existing one (zzz/aaa), proving record
    # order comes from (created_at, id), never directory order
    root = sr_env.suite_root()
    for decoy, source in (("aaa-decoy-1", sr_env.sr_reused.suite_run_id),
                          ("zzz-decoy-9", sr_env.sr_created.suite_run_id)):
        man = sr_env.suite_manifest(source)
        man["suite_run_id"] = decoy
        d = root / decoy
        d.mkdir(exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(man))
    engine = sorted(f.list_suite_runs(mid),
                    key=lambda r: (r.created_at, r.suite_run_id))
    ids = [r.suite_run_id for r in f.get_dashboard(mid)
           .suite_runs.records]
    assert ids == [r.suite_run_id for r in engine]
    assert ids != sorted(p.name for p in root.iterdir() if p.is_dir())
    d1 = f.get_dashboard(mid).model_dump(mode="json")
    assert d1["suite_runs"]["counts"] == {"completed": 6}
    assert f.get_dashboard(mid).model_dump(mode="json") == d1


def test_suite_section_hash_tracks_storage_state_change_only_for_owner(sr_env):
    f = sr_env.forge
    h_a1 = f.get_dashboard(sr_env.model_a).result_hash
    h_b = f.get_dashboard(sr_env.model_b).result_hash
    h_c = f.get_dashboard(sr_env.model_c).result_hash
    # a new suite run on A changes only A's dashboard hash
    new_run = f.run_suite(SuiteRunRequest(
        model_id=sr_env.model_a, suite_id=sr_env.suite_id,
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    assert f.get_dashboard(sr_env.model_a).result_hash != h_a1
    assert f.get_dashboard(sr_env.model_b).result_hash == h_b
    assert f.get_dashboard(sr_env.model_c).result_hash == h_c
    assert len(f.get_dashboard(sr_env.model_a).suite_runs.records) == 7
    assert new_run.status.value == "completed"
    # repeat reads of the changed state are stable again
    assert f.get_dashboard(sr_env.model_a).result_hash == \
        f.get_dashboard(sr_env.model_a).result_hash


def test_graph_suite_run_nodes_and_edges(sr_env):
    f, mid = sr_env.forge, sr_env.model_a
    dash = f.get_dashboard(mid)
    g = dash.artifact_graph
    node_ids = {(n.family, n.artifact_id) for n in g.nodes}
    edge_ids = {(e.source.family, e.source.artifact_id,
                 e.target.family, e.target.artifact_id, e.role)
                for e in g.edges}
    # every suite run owned by the model is a real node of the graph
    sr_ids = [r.suite_run_id for r in f.list_suite_runs(mid)]
    for rid in sr_ids:
        assert ("suite_run", rid) in node_ids
    # workflow -> suite_run edges for the recorded stage artifacts only
    wf_artifacts = set()
    for w in f.list_workflows(mid):
        for st in w.stages:
            if st.artifact and st.artifact.kind.value == "suite_run":
                wf_artifacts.add((w.workflow_id, st.artifact.artifact_id))
    assert len(wf_artifacts) == 2                      # inline + recipe run
    for wid, rid in wf_artifacts:
        assert ("workflow", wid, "suite_run", rid, "stage_artifact") \
            in edge_ids
    # suite_run -> evaluation edges for every real per-probe reference
    probe_edges = {(r.suite_run_id, p.evaluation_id)
                   for r in f.list_suite_runs(mid)
                   for p in r.results if p.evaluation_id}
    for rid, ev in probe_edges:
        assert ("suite_run", rid, "evaluation", ev, "probe") in edge_ids
        assert ("evaluation", ev) in node_ids
    # every edge's endpoints are nodes (global graph consistency)
    for e in g.edges:
        assert (e.source.family, e.source.artifact_id) in node_ids
        assert (e.target.family, e.target.artifact_id) in node_ids
    # healthy store: no diagnostics anywhere
    assert dash.diagnostics == []


def test_dashboard_suite_reads_never_write_byte_audit(sr_env):
    f = sr_env.forge
    root = f.storage.root
    before = sorted(p.relative_to(root).as_posix()
                    for p in root.rglob("*") if p.is_file())
    blob = {p: (root / p).read_bytes() for p in before}
    for _ in range(2):
        for mid in (sr_env.model_a, sr_env.model_b, sr_env.model_c):
            f.get_dashboard(mid)
            f.list_suite_runs(mid)
    after = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*") if p.is_file())
    assert after == before
    for p in before:
        assert (root / p).read_bytes() == blob[p]
    assert [p for p in root.rglob("*") if ".tmp" in p.name] == []


def test_suite_run_corruption_skipped_and_ghost_reference_diagnosed(sr_env):
    f, mid = sr_env.forge, sr_env.model_a
    root = sr_env.suite_root()
    engine_before = f.list_suite_runs(mid)
    n_before = len(engine_before)
    # corrupt manifest (bad json) -> skipped with deterministic diagnostic
    (root / "bad-json-corrupt").mkdir(exist_ok=True)
    (root / "bad-json-corrupt" / "manifest.json").write_text("{not json")
    # missing manifest -> skipped with its own deterministic diagnostic
    (root / "missing-manifest").mkdir(exist_ok=True)
    # valid manifest whose per-probe reference points at a missing
    # evaluation -> record stays visible, reference becomes a diagnostic
    # and never a fabricated node/edge
    ghost_id = "ghost-eval-reference"
    man = sr_env.suite_manifest(sr_env.sr_created.suite_run_id)
    man["suite_run_id"] = ghost_id
    man["results"][0]["evaluation_id"] = "ghost-eval-00000000"
    (root / ghost_id).mkdir(exist_ok=True)
    (root / ghost_id / "manifest.json").write_text(json.dumps(man))

    j1 = f.get_dashboard(mid).model_dump(mode="json")
    sec = j1["suite_runs"]
    diags = {(d["family"], d["artifact_id"], d["issue"])
             for d in j1["diagnostics"]}
    # corrupt runs skipped; every valid record (incl. the ghost-reference
    # one) remains visible and counted exactly by its recorded status
    assert len(sec["records"]) == n_before + 1
    assert sec["counts"] == {"completed": n_before + 1}
    assert {r["suite_run_id"] for r in sec["records"]} == \
        {r.suite_run_id for r in f.list_suite_runs(mid)}
    assert "bad-json-corrupt" not in {r["suite_run_id"]
                                      for r in sec["records"]}
    assert "missing-manifest" not in {r["suite_run_id"]
                                      for r in sec["records"]}
    assert any(d[0] == "suite_run" and d[1] == "bad-json-corrupt"
               and "JSONDecodeError" in d[2] for d in diags)
    assert any(d[0] == "suite_run" and d[1] == "missing-manifest"
               and d[2] == "manifest.json missing" for d in diags)
    assert any(d[0] == "suite_run" and d[1] == ghost_id
               and "references missing evaluation 'ghost-eval-00000000'"
               and "role 'probe'" in d[2] for d in diags)
    # the ghost evaluation is neither a node nor an edge target
    nodes = {(n["family"], n["artifact_id"])
             for n in j1["artifact_graph"]["nodes"]}
    assert ("evaluation", "ghost-eval-00000000") not in nodes
    for e in j1["artifact_graph"]["edges"]:
        assert not (e["target"]["family"] == "evaluation"
                    and e["target"]["artifact_id"] == "ghost-eval-00000000")
    # real suite_run -> evaluation edges unaffected by the corruption; only
    # references that resolve to an existing evaluation become edges (the
    # ghost one does not)
    real_edges = [e for e in j1["artifact_graph"]["edges"]
                  if e["source"]["family"] == "suite_run"
                  and e["role"] == "probe"]
    resolvable = sum(
        1 for r in f.list_suite_runs(mid)
        for p in r.results
        if p.evaluation_id
        and ("evaluation", p.evaluation_id) in nodes)
    assert len(real_edges) == resolvable
    # repeated reads stay byte-deterministic, diagnostics included
    j2 = f.get_dashboard(mid).model_dump(mode="json")
    assert j2 == j1
    assert f.get_dashboard(mid).result_hash == \
        f.get_dashboard(mid).result_hash
    # reading never rewrote or added anything under suite-runs/
    listing_before = sorted(p.relative_to(root).as_posix()
                            for p in root.rglob("*") if p.is_file())
    f.get_dashboard(mid)
    listing_after = sorted(p.relative_to(root).as_posix()
                           for p in root.rglob("*") if p.is_file())
    assert listing_after == listing_before


# =========================================================================== #
# M17: dashboard sample-quality observability over the M16 history
#
# The dashboard gains a deterministic read-only ``sample_quality`` section
# derived LIVE from the immutable sample-evaluations/<model_id>/ manifests
# (root-level M16 family; per-model segment). Section = history structure
# only: total_count, per-sample references (ordered by sample_id) and a
# newest-first evaluation reference list ((created_at, evaluation_id)
# DESCENDING). No metric values, no aggregation, no quality judgment. The
# module ``env`` model owns REAL M16 samples + measurements; crafted
# manifests on dedicated fresh models cover ordering/isolation/corruption.
# =========================================================================== #

def _sq_record(**over) -> dict:
    """Minimal schema-valid SampleEvaluationRecord JSON (references only)."""
    base = dict(
        evaluation_id="e00000000001", model_id="m", sample_id="s00000000001",
        sample_result_hash="a" * 64, token_sequence_sha256="b" * 64,
        checkpoint_id="c00000000001", checkpoint_weights_sha256="c" * 64,
        tokenizer_id="t00000000001", tokenizer_hash="d" * 64,
        prompt_token_count=8, generated_token_count=6,
        evaluated_token_count=6, context_length=64, window_token_count=14,
        window_rule="single_window", loss_nats=5.5, perplexity=244.69,
        result_hash="0" * 64,
        created_at="2026-09-01T00:00:00+00:00", schema_version=1)
    base.update(over)
    return base


def _sq_write(root, model_id: str, evaluation_id: str,
              record: dict | None) -> None:
    d = root / "sample-evaluations" / model_id / f"evaluation-{evaluation_id}"
    d.mkdir(parents=True, exist_ok=True)
    if record is not None:
        (d / "manifest.json").write_text(json.dumps(record))


def _ensure_sample_dir(root, model_id: str, sample_id: str) -> None:
    """The dashboard resolves sample references by directory existence under
    samples/<model_id>/ (it never parses sample manifests)."""
    (root / "samples" / model_id / f"sample-{sample_id}").mkdir(
        parents=True, exist_ok=True)


def test_sample_quality_empty_section_deterministic(env):
    mid = env.fresh_model("m17-empty")
    d1 = env.forge.get_dashboard(mid)
    sec = d1.model_dump(mode="json")["sample_quality"]
    assert sec == {"total_count": 0, "by_sample": [], "latest": []}
    assert len(d1.result_hash) == 64
    d2 = env.forge.get_dashboard(mid)
    assert d2.model_dump(mode="json")["sample_quality"] == sec
    assert d2.result_hash == d1.result_hash
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.get_dashboard("m17-ghost-model")


def test_sample_quality_section_matches_m16_engine_history(env):
    """Real M16 records on the module model: section == engine listing,
    grouped per sample, references only (never metric values)."""
    f, mid = env.forge, env.model_id
    engine = f.list_sample_evaluations(mid)          # (created_at, id) order
    assert len(engine) == 3
    sec = f.get_dashboard(mid).sample_quality
    assert sec.total_count == 3 == len(engine)
    # by_sample: sample_a has 2 records, sample_b 1; ordered by sample_id
    assert [b.sample_id for b in sec.by_sample] == \
        sorted([env.sample_a, env.sample_b])
    by_sid = {b.sample_id: b for b in sec.by_sample}
    a = by_sid[env.sample_a]
    assert a.evaluation_count == 2 and by_sid[env.sample_b].evaluation_count == 1
    # latest per sample = max by (created_at, evaluation_id) over its records
    for b in sec.by_sample:
        recs = [r for r in engine if r.sample_id == b.sample_id]
        newest = max(recs, key=lambda r: (r.created_at, r.evaluation_id))
        assert b.latest_evaluation_id == newest.evaluation_id
        assert b.latest_evaluated_at == newest.created_at
        assert b.sample_result_hash == newest.sample_result_hash
    # latest list = every record newest-first, references only
    assert [(x.evaluation_id, x.sample_id) for x in sec.latest] == \
        [(r.evaluation_id, r.sample_id)
         for r in sorted(engine,
                         key=lambda r: (r.created_at, r.evaluation_id),
                         reverse=True)]
    # references resolve through the M16 getter
    for x in sec.latest:
        rec = f.get_sample_evaluation(mid, x.evaluation_id)
        assert rec.evaluation_id == x.evaluation_id
        assert rec.sample_id == x.sample_id
        assert rec.created_at == x.created_at
    # repeated reads byte-deterministic, hash stable
    d1 = f.get_dashboard(mid).model_dump(mode="json")
    assert f.get_dashboard(mid).model_dump(mode="json") == d1
    assert f.get_dashboard(mid).result_hash == d1["result_hash"]
    assert d1["diagnostics"] == []


def test_sample_quality_grouping_and_latest_with_crafted_history(env):
    """Interleaved records across samples: per-sample counts/latest follow
    (created_at, evaluation_id), overall latest is newest-first."""
    f = env.forge
    mid = env.fresh_model("m17-group")
    root = f.storage.root
    sa, sb, sc = "sa0000000001", "sb0000000001", "sc0000000001"
    for s in (sa, sb, sc):
        _ensure_sample_dir(root, mid, s)
    # interleaved creation order across the three samples
    rows = [("e-t1-sa", sa, "2026-09-01T00:00:01+00:00"),
            ("e-t2-sb", sb, "2026-09-01T00:00:02+00:00"),
            ("e-t3-sc", sc, "2026-09-01T00:00:03+00:00"),
            ("e-t4-sa", sa, "2026-09-01T00:00:04+00:00"),
            ("e-t5-sc", sc, "2026-09-01T00:00:05+00:00")]
    for eid, sid, t in rows:
        _sq_write(root, mid, eid,
                  _sq_record(evaluation_id=eid, sample_id=sid,
                             created_at=t, model_id=mid))
    sec = f.get_dashboard(mid).sample_quality
    assert sec.total_count == 5
    assert [b.sample_id for b in sec.by_sample] == [sa, sb, sc]
    by_sid = {b.sample_id: b for b in sec.by_sample}
    assert by_sid[sa].evaluation_count == 2
    assert by_sid[sa].latest_evaluation_id == "e-t4-sa"
    assert by_sid[sb].evaluation_count == 1
    assert by_sid[sb].latest_evaluation_id == "e-t2-sb"
    assert by_sid[sc].evaluation_count == 2
    assert by_sid[sc].latest_evaluation_id == "e-t5-sc"
    # newest-first over (created_at, evaluation_id) — deterministic and
    # independent of the directory iteration order
    assert [x.evaluation_id for x in sec.latest] == \
        ["e-t5-sc", "e-t4-sa", "e-t3-sc", "e-t2-sb", "e-t1-sa"]
    # identical second read
    assert f.get_dashboard(mid).model_dump(mode="json") == \
        f.get_dashboard(mid).model_dump(mode="json")


def test_sample_quality_ordering_ignores_directory_order(env):
    """Directory names that sort opposite to creation times never leak into
    the section ordering (records order by persisted fields only)."""
    f = env.forge
    mid = env.fresh_model("m17-ordering")
    root = f.storage.root
    _ensure_sample_dir(root, mid, "sample-ord-1")
    _ensure_sample_dir(root, mid, "sample-ord-2")
    # zzz-* dirs hold the OLDEST records; aaa-* dirs hold the NEWEST
    _sq_write(root, mid, "zzz-old-0001",
              _sq_record(evaluation_id="zzz-old-0001", model_id=mid,
                         sample_id="sample-ord-1",
                         created_at="2026-08-01T00:00:00+00:00"))
    _sq_write(root, mid, "zzz-old-0002",
              _sq_record(evaluation_id="zzz-old-0002", model_id=mid,
                         sample_id="sample-ord-2",
                         created_at="2026-08-01T00:00:01+00:00"))
    _sq_write(root, mid, "aaa-new-0001",
              _sq_record(evaluation_id="aaa-new-0001", model_id=mid,
                         sample_id="sample-ord-2",
                         created_at="2026-09-01T00:00:00+00:00"))
    _sq_write(root, mid, "aaa-new-0002",
              _sq_record(evaluation_id="aaa-new-0002", model_id=mid,
                         sample_id="sample-ord-1",
                         created_at="2026-09-01T00:00:01+00:00"))
    sec = f.get_dashboard(mid).sample_quality
    assert sec.total_count == 4
    # newest-first overall: aaa-new-0002 > aaa-new-0001 > zzz-old-0002 ...
    assert [x.evaluation_id for x in sec.latest] == \
        ["aaa-new-0002", "aaa-new-0001", "zzz-old-0002", "zzz-old-0001"]
    # per-sample latest: sample-ord-1 -> aaa-new-0002; sample-ord-2 ->
    # aaa-new-0001; by_sample ordering is by sample_id (deterministic)
    by_sid = {b.sample_id: b for b in sec.by_sample}
    assert [b.sample_id for b in sec.by_sample] == \
        sorted([b.sample_id for b in sec.by_sample])
    assert by_sid["sample-ord-1"].latest_evaluation_id == "aaa-new-0002"
    assert by_sid["sample-ord-2"].latest_evaluation_id == "aaa-new-0001"
    assert f.get_dashboard(mid).result_hash == \
        f.get_dashboard(mid).result_hash


def test_sample_quality_model_isolation(env):
    """Only sample-evaluations/<model>/ manifests contribute; a manifest
    under the wrong model's directory with a mismatching persisted model_id
    is silently skipped (the platform convention for model-scoped records);
    nothing ever aggregates across models."""
    f = env.forge
    ma = env.fresh_model("m17-iso-a")
    mb = env.fresh_model("m17-iso-b")
    root = f.storage.root
    _ensure_sample_dir(root, ma, "sample-a-0000001")
    _ensure_sample_dir(root, mb, "sample-b-0000001")
    _sq_write(root, ma, "e-a1", _sq_record(
        evaluation_id="e-a1", model_id=ma, sample_id="sample-a-0000001",
        created_at="2026-09-01T00:00:01+00:00"))
    _sq_write(root, ma, "e-a2", _sq_record(
        evaluation_id="e-a2", model_id=ma, sample_id="sample-a-0000001",
        created_at="2026-09-01T00:00:02+00:00"))
    _sq_write(root, mb, "e-b1", _sq_record(
        evaluation_id="e-b1", model_id=mb, sample_id="sample-b-0000001",
        created_at="2026-09-01T00:00:01+00:00"))
    # stray manifests under the WRONG model directory (persisted model_id
    # disagrees with the directory): skipped silently, never aggregated
    _sq_write(root, ma, "e-stray-b", _sq_record(
        evaluation_id="e-stray-b", model_id=mb, sample_id="sample-b-0000001",
        created_at="2026-09-01T00:00:03+00:00"))
    _sq_write(root, mb, "e-stray-a", _sq_record(
        evaluation_id="e-stray-a", model_id=ma, sample_id="sample-a-0000001",
        created_at="2026-09-01T00:00:03+00:00"))

    ja = f.get_dashboard(ma).model_dump(mode="json")
    jb = f.get_dashboard(mb).model_dump(mode="json")
    assert ja["sample_quality"]["total_count"] == 2
    assert {x["evaluation_id"] for x in ja["sample_quality"]["latest"]} == \
        {"e-a1", "e-a2"}
    assert {x["sample_id"] for x in ja["sample_quality"]["latest"]} == \
        {"sample-a-0000001"}
    assert jb["sample_quality"]["total_count"] == 1
    assert [x["evaluation_id"] for x in jb["sample_quality"]["latest"]] == \
        ["e-b1"]
    assert "e-stray-b" not in {x["evaluation_id"]
                               for x in ja["sample_quality"]["latest"]}
    assert "e-stray-a" not in {x["evaluation_id"]
                               for x in jb["sample_quality"]["latest"]}
    assert ja["diagnostics"] == [] and jb["diagnostics"] == []
    # the fresh empty model stays empty while A and B carry records
    mc = env.fresh_model("m17-iso-c")
    assert f.get_dashboard(mc).model_dump(mode="json")["sample_quality"] == \
        {"total_count": 0, "by_sample": [], "latest": []}


def test_sample_quality_corrupt_manifests_skipped_with_diagnostics(env):
    """Malformed/unreadable/schema-invalid evaluation manifests are skipped
    with deterministic diagnostics; valid records (including one referencing
    a missing sample) stay visible; the dashboard never crashes or writes."""
    f = env.forge
    mid = env.fresh_model("m17-corrupt-sq")
    root = f.storage.root
    _ensure_sample_dir(root, mid, "sample-good-0001")
    _ensure_sample_dir(root, mid, "sample-good-0002")
    _sq_write(root, mid, "good-1", _sq_record(
        evaluation_id="good-1", model_id=mid, sample_id="sample-good-0001",
        created_at="2026-09-01T00:00:01+00:00"))
    _sq_write(root, mid, "good-2", _sq_record(
        evaluation_id="good-2", model_id=mid, sample_id="sample-good-0002",
        created_at="2026-09-01T00:00:02+00:00"))
    # ghost: schema-valid record whose sample does not exist under samples/
    _sq_write(root, mid, "ghost-sample", _sq_record(
        evaluation_id="ghost-sample", model_id=mid,
        sample_id="sample-ghost-001",
        created_at="2026-09-01T00:00:03+00:00"))
    # bad json manifest
    _sq_write(root, mid, "bad-json", None)
    p = root / "sample-evaluations" / mid / "evaluation-bad-json" / \
        "manifest.json"
    p.write_text("{not json")
    # missing manifest
    (root / "sample-evaluations" / mid / "evaluation-missing-manifest"
     ).mkdir(parents=True)
    # schema-invalid manifest
    d = root / "sample-evaluations" / mid / "evaluation-not-a-record"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"not": "a record"}))

    j1 = f.get_dashboard(mid).model_dump(mode="json")
    sec = j1["sample_quality"]
    diags = {(x["family"], x["artifact_id"], x["issue"])
             for x in j1["diagnostics"]}
    # corrupt records skipped; 2 valid + 1 ghost still counted
    assert sec["total_count"] == 3
    assert {b["sample_id"] for b in sec["by_sample"]} == \
        {"sample-good-0001", "sample-good-0002", "sample-ghost-001"}
    assert {x["evaluation_id"] for x in sec["latest"]} == \
        {"good-1", "good-2", "ghost-sample"}
    # visible records == the M16 engine listing (which skips the corrupt
    # ones the same way)
    engine_ids = {r.evaluation_id
                  for r in f.list_sample_evaluations(mid)}
    assert engine_ids == {x["evaluation_id"] for x in sec["latest"]}
    # deterministic diagnostics per corrupt artifact
    assert any(x[0] == "sample_evaluation" and x[1] == "bad-json"
               and "JSONDecodeError" in x[2] for x in diags)
    assert any(x[0] == "sample_evaluation" and x[1] == "missing-manifest"
               and x[2] == "manifest.json missing" for x in diags)
    assert any(x[0] == "sample_evaluation" and x[1] == "not-a-record"
               and "ValidationError" in x[2] for x in diags)
    # internal reference to a missing sample surfaces as a diagnostic
    assert any(x[0] == "sample_evaluation" and x[1] == "ghost-sample"
               and "references missing sample 'sample-ghost-001'" in x[2]
               for x in diags)
    # corrupt artifacts never appear in by_sample/latest payloads
    assert all(b["sample_id"] != "bad-json" for b in sec["by_sample"])
    # repeated reads deterministic (diagnostics included), never writes
    j2 = f.get_dashboard(mid).model_dump(mode="json")
    assert j2 == j1
    assert f.get_dashboard(mid).result_hash == \
        f.get_dashboard(mid).result_hash
    sq_root = root / "sample-evaluations" / mid
    listing = sorted(p.relative_to(sq_root).as_posix()
                     for p in sq_root.rglob("*") if p.is_file())
    f.get_dashboard(mid)
    assert sorted(p.relative_to(sq_root).as_posix()
                  for p in sq_root.rglob("*") if p.is_file()) == listing


def test_sample_quality_section_never_exposes_metric_values(env):
    """The section is references + counts: loss/perplexity and any
    aggregate/rank vocabulary never appear in its payload."""
    f, mid = env.forge, env.model_id
    sec = f.get_dashboard(mid).sample_quality.model_dump(mode="json")
    assert sec["total_count"] >= 3        # module env owns real M16 history
    blob = json.dumps(sec).lower()
    for banned in ("loss", "perplexity", "score", "rank", "best", "worst",
                   "average", "mean", "min_", "max_", "verdict",
                   "aggregate", "summary_metric"):
        assert banned not in blob, banned
    # every payload value is a reference or a count
    for b in sec["by_sample"]:
        assert set(b) == {"sample_id", "sample_result_hash",
                          "evaluation_count", "latest_evaluation_id",
                          "latest_evaluated_at"}
        assert isinstance(b["evaluation_count"], int)
    for x in sec["latest"]:
        assert set(x) == {"evaluation_id", "sample_id", "created_at"}


def test_sample_quality_hash_tracks_new_measurement_only_for_owner(env):
    f = env.forge
    ma = env.fresh_model("m17-hash-a")
    mb = env.fresh_model("m17-hash-b")
    root = f.storage.root
    for m in (ma, mb):
        _ensure_sample_dir(root, m, "sample-h-00000001")
    _sq_write(root, ma, "e-1", _sq_record(
        evaluation_id="e-1", model_id=ma, sample_id="sample-h-00000001",
        created_at="2026-09-01T00:00:01+00:00"))
    _sq_write(root, mb, "e-1", _sq_record(
        evaluation_id="e-1", model_id=mb, sample_id="sample-h-00000001",
        created_at="2026-09-01T00:00:01+00:00"))
    h_a0 = f.get_dashboard(ma).result_hash
    h_b0 = f.get_dashboard(mb).result_hash
    # a new measurement on A changes only A's dashboard hash
    _sq_write(root, ma, "e-2", _sq_record(
        evaluation_id="e-2", model_id=ma, sample_id="sample-h-00000001",
        created_at="2026-09-01T00:00:02+00:00"))
    assert f.get_dashboard(ma).result_hash != h_a0
    assert f.get_dashboard(ma).sample_quality.total_count == 2
    assert f.get_dashboard(mb).result_hash == h_b0
    assert f.get_dashboard(mb).sample_quality.total_count == 1
    # repeat reads of the changed state are stable again
    assert f.get_dashboard(ma).result_hash == \
        f.get_dashboard(ma).result_hash


def test_sample_quality_additive_section_existing_semantics_intact(env):
    """The new section is additive: existing sections keep their exact
    shape; the artifact graph gains no sample/sample_evaluation nodes; a
    healthy store reports no diagnostics."""
    f, mid = env.forge, env.model_id
    j = f.get_dashboard(mid).model_dump(mode="json")
    # existing sections unchanged in shape
    assert set(j["workflows"].keys()) == {"counts", "records"}
    assert set(j["suite_runs"].keys()) == {"counts", "records"}
    assert j["workflows"]["counts"] == env.wf_counts
    assert set(j["evaluations"][0].keys()) >= {"identity", "key", "count",
                                               "records"}
    assert set(j["gate_decisions"][0].keys()) >= {"policy_key", "policy",
                                                  "statistics", "records"}
    assert "comparisons" in j and "artifact_graph" in j
    assert j["diagnostics"] == []
    # the graph families are exactly the pre-M17 set (no redesign)
    fams = {n["family"] for n in j["artifact_graph"]["nodes"]}
    assert fams == {"checkpoint", "evaluation", "comparison",
                    "gate_decision", "workflow", "training_run"}
    # the M16 records are visible through the M16 engine and nowhere else
    assert j["sample_quality"]["total_count"] == \
        len(f.list_sample_evaluations(mid))
    # dashboard still deterministic end-to-end
    assert f.get_dashboard(mid).model_dump(mode="json") == j


def test_dashboard_read_only_with_sample_quality_history(env):
    """Byte audit around repeated dashboard reads of a model with real M16
    sample-evaluations: zero writes, zero new files, zero .tmp, zero
    modification of the sample-evaluation manifests (audit scoped to the
    model dir + its sample-evaluations so shared-env growth from other
    tests never interferes)."""
    f, mid = env.forge, env.model_id
    model_root = f.storage.model_dir(mid)
    sq_dir = f.storage.root / "sample-evaluations" / mid
    assert sq_dir.exists()
    before_files = sorted(p.relative_to(sq_dir).as_posix()
                          for p in sq_dir.rglob("*") if p.is_file())
    before_sq = {p: (sq_dir / p).read_bytes() for p in before_files}
    model_files = sorted(p.relative_to(model_root).as_posix()
                         for p in model_root.rglob("*") if p.is_file())
    before_model = {p: (model_root / p).read_bytes()
                    for p in model_files}
    for _ in range(3):
        d = f.get_dashboard(mid).model_dump(mode="json")
        assert d["sample_quality"]["total_count"] >= 3
        assert f.get_dashboard(mid).result_hash == d["result_hash"]
    after_sq = sorted(p.relative_to(sq_dir).as_posix()
                      for p in sq_dir.rglob("*") if p.is_file())
    assert after_sq == before_files                    # zero new files
    for p in before_files:
        assert (sq_dir / p).read_bytes() == before_sq[p]
    for p in model_files:
        assert (model_root / p).read_bytes() == before_model[p]
    assert [p for p in sq_dir.rglob("*") if ".tmp" in p.name] == []
    assert [p for p in model_root.rglob("*") if ".tmp" in p.name] == []


def test_dashboard_sample_quality_openapi_and_http(api_client):
    """HTTP: dashboard exposes the section for a model with no history and
    keeps 404/405 behavior; OpenAPI documents the schema; no new endpoints."""
    cfg = {"name": "m17-api", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4,
           "n_kv_heads": 2, "intermediate_size": 128, "seed": 1}
    r = api_client.post("/api/v1/models", json={"config": cfg})
    assert r.status_code == 201, r.text
    mid = r.json()["model"]["id"]
    d1 = api_client.get(f"/api/v1/models/{mid}/dashboard")
    assert d1.status_code == 200
    assert d1.json()["sample_quality"] == \
        {"total_count": 0, "by_sample": [], "latest": []}
    d2 = api_client.get(f"/api/v1/models/{mid}/dashboard")
    assert d2.text == d1.text and d2.json()["result_hash"] == \
        d1.json()["result_hash"]
    assert api_client.get("/api/v1/models/ghost-m17/dashboard").status_code \
        == 404
    spec = api_client.get("/openapi.json").json()
    dash_schema = spec["components"]["schemas"]["ModelDashboard"]
    assert "sample_quality" in dash_schema["properties"]
    assert "DashboardSampleQuality" in spec["components"]["schemas"]
    assert "DashboardSampleQualityBySample" in \
        spec["components"]["schemas"]
    assert "DashboardSampleQualityLatest" in \
        spec["components"]["schemas"]
    assert "/api/v1/models/{model_id}/dashboard" in spec["paths"]
    # M17 added no endpoints; M18–M22 each added exactly one
    # documented route, so the surface count is 54 (assertion stays
    # exact)
    assert len(spec["paths"]) == 54
