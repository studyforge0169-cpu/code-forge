"""Milestone 10 tests: explicit multi-probe evaluation batches over suites.

Engine tests (module-scoped real history: domain -> tokenizer -> model ->
training -> checkpoints) prove M10 is a thin orchestration layer over M4:

  * every suite probe executes as an ordinary exact M4 evaluation (or is
    reused when identical evidence exists) — no second identity, no
    duplicate evidence from repeated runs
  * canonical suite order is respected; per-probe results expose each
    evaluation independently; NO aggregation of any kind exists
  * deterministic failure semantics: model/state/suite preflight errors
    persist nothing; per-probe execution failures are recorded cleanly with
    deterministic text and a ``failed`` run status
  * one immutable manifest per run; deterministic result_hash over the
    semantic execution (bookkeeping labels excluded) so logically identical
    runs reproduce it
  * dashboard keeps rendering the underlying M4 evaluations normally
  * API coverage: POST /suite-runs + model-scoped list/get with the
    established 404/409/422 mapping and 405 for update/delete
"""
from __future__ import annotations

import json
import random
import time

import pytest
from pydantic import ValidationError

from app.engine import ModelForge
from app.schemas import (
    ComparisonState,
    EvalStateKind,
    EvaluationConfig,
    ModelCreateRequest,
    ProbeSuiteCreateRequest,
    SuiteRunRecord,
    SuiteRunRequest,
    SuiteProbe,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
)

HEADS = ("river mountain cloud forest desert ocean valley island meadow canyon "
         "table chair lamp desk shelf couch rug clock mirror vase").split()
TAIL = ("flows stands gleams rises falls drifts looms shines hides waits").split()


def _domain_bytes(n: int = 240) -> bytes:
    rng = random.Random(11)
    out = []
    for i in range(n):
        h1, h2 = rng.sample(HEADS, 2)
        out.append(f"{h1} is {rng.choice(TAIL)} near {h2} with number {i}")
    return ("\n\n".join(out) + "\n").encode("utf-8")


class Env:
    """One temp root: model with real training history + tokenized domain."""

    def __init__(self, root):
        self.forge = ModelForge(root=root)

    def prepare(self):
        f = self.forge
        up = f.upload_dataset([("a.txt", _domain_bytes())], name="m10-dom")
        self.ds = up["dataset_id"]
        tok = f.train_tokenizer(TokenizerConfig(name="m10-tok", vocab_size=600),
                                dataset_id=self.ds)
        self.tok = tok.id
        f.tokenize_dataset(self.ds, tok.id)
        cfg = TransformerConfig(name="m10-main", vocab_size=640,
                                context_length=64, hidden_size=64,
                                n_layers=2, n_heads=4, n_kv_heads=2,
                                intermediate_size=128, seed=1)
        self.model_id = f.create_model(ModelCreateRequest(config=cfg))[0].id
        tokens = f.datasets.tokenized_artifact(self.ds, 1, self.tok)[0] \
            .splits["train"].count
        sp_epoch = max(1, (tokens // 32) // 8)
        f.run_training(TrainingConfig(
            method="continued_pretraining", model_id=self.model_id,
            dataset_id=self.ds, tokenizer_id=self.tok, learning_rate=3e-3,
            batch_size=8, max_seq_len=32, epochs=5,
            eval_every_steps=max(1, sp_epoch // 2), keep_best=False, seed=1))
        ckpts = f.list_checkpoints(self.model_id)
        self.ck_a = ckpts[len(ckpts) // 2].checkpoint_id
        self.ck_b = ckpts[-1].checkpoint_id

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def fresh_model(self, name: str = "m10-extra", seed: int = 31) -> str:
        return self.forge.create_model(ModelCreateRequest(config=TransformerConfig(
            name=name, vocab_size=640, context_length=64, hidden_size=64,
            n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
            seed=seed)))[0].id

    def probe(self, seed: int, dsid: str | None = None, split: str = "validation",
              seq: int = 32, tok: str | None = None) -> SuiteProbe:
        return SuiteProbe(dataset_id=dsid or self.ds, split=split,
                          tokenizer_id=tok or self.tok, batch_size=8,
                          max_seq_len=seq, seed=seed)

    def register_suite(self, suite_id: str, probes) -> None:
        self.forge.register_probe_suite(ProbeSuiteCreateRequest(
            suite_id=suite_id, probes=probes))

    def run(self, suite_id: str, state: ComparisonState,
            model_id: str | None = None) -> SuiteRunRecord:
        return self.forge.run_suite(SuiteRunRequest(
            model_id=model_id or self.model_id, suite_id=suite_id, state=state))

    def eval_count(self, model_id: str | None = None) -> int:
        return len(self.forge.list_evaluations(model_id or self.model_id))

    def run_files(self) -> set[str]:
        root = self.forge.storage.root / "suite-runs"
        if not root.exists():
            return set()
        return {f"suite-runs/{d.name}/manifest.json"
                for d in root.iterdir() if d.is_dir()}


def cstate(ckpt=None, current=False) -> ComparisonState:
    if current:
        return ComparisonState(state_kind=EvalStateKind.CURRENT)
    return ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                           checkpoint_id=ckpt)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m10-root"))
    e.prepare()
    return e


# =========================================================================== #
# Execution: one-probe / multi-probe / current & checkpoint states
# =========================================================================== #

def test_one_probe_suite_on_checkpoint_state(env):
    f = env.forge
    env.register_suite("m10-one", [env.probe(seed=10001)])
    rec = env.run("m10-one", cstate(env.ck_b))
    assert rec.status.value == "completed"
    assert rec.probe_count == 1 and rec.completed_count == 1
    assert rec.reused_count == 0 and rec.failed_count == 0
    assert rec.suite_id == "m10-one"
    assert rec.state == cstate(env.ck_b)
    res = rec.results[0]
    assert res.outcome == "created" and res.evaluation_id and res.error is None
    assert res.probe.dataset_version == 1        # version resolved at run time
    # the result IS an ordinary M4 evaluation with the exact expected identity
    ev = f.get_evaluation(env.model_id, res.evaluation_id)
    assert ev.state_kind.value == "checkpoint" and ev.checkpoint_id == env.ck_b
    assert ev.dataset_id == env.ds and ev.seed == 10001
    assert ev.state_hash == rec.state_hash
    assert ev.config.get("batch_size") == 8 and ev.config.get("max_seq_len") == 32
    assert ev.result_hash and ev.created_at


def test_multi_probe_suite_canonical_order_current_state(env):
    f = env.forge
    # register deliberately out of canonical order: seeds [10013, 10011, 10012]
    env.register_suite("m10-multi", [env.probe(seed=10013),
                                     env.probe(seed=10011),
                                     env.probe(seed=10012)])
    rec = env.run("m10-multi", cstate(current=True))
    assert rec.status.value == "completed"
    assert [r.probe.seed for r in rec.results] == [10011, 10012, 10013]
    assert [r.outcome for r in rec.results] == ["created"] * 3
    assert rec.completed_count == 3 and rec.reused_count == 0
    # each probe maps to an independent M4 evaluation of the CURRENT state
    evs = [f.get_evaluation(env.model_id, r.evaluation_id)
           for r in rec.results]
    assert all(e.state_kind.value == "current" and e.checkpoint_id is None
               for e in evs)
    assert len({e.eval_id for e in evs}) == 3
    # recorded state hash equals the live verified canonical hash
    live_hash = f.comparison.verified_state_hash(
        env.model_id, ComparisonState(state_kind=EvalStateKind.CURRENT))
    assert rec.state_hash == live_hash


def test_suite_run_on_second_model_is_model_scoped(env):
    f = env.forge
    mid2 = env.fresh_model("m10-scope")
    env.register_suite("m10-scope-s", [env.probe(seed=10021)])
    rec = env.run("m10-scope-s", cstate(current=True), model_id=mid2)
    assert rec.model_id == mid2 and rec.status.value == "completed"
    # the run and its evaluations belong to model 2, never to model 1
    assert all(e.model_id == mid2 for e in f.list_evaluations(mid2))
    assert env.eval_count(mid2) == 1
    with pytest.raises(FileNotFoundError):
        f.get_suite_run(env.model_id, rec.suite_run_id)   # wrong model -> 404


# =========================================================================== #
# Evidence reuse: exact M4 identity, no duplicate evaluations
# =========================================================================== #

def test_repeated_run_reuses_evidence_no_duplicates(env):
    f = env.forge
    env.register_suite("m10-reuse", [env.probe(seed=10031),
                                     env.probe(seed=10032)])
    runs0 = env.run_files()
    evals0 = env.eval_count()
    r1 = env.run("m10-reuse", cstate(env.ck_b))
    assert env.eval_count() == evals0 + 2
    r2 = env.run("m10-reuse", cstate(env.ck_b))
    # identical evidence reused: same evaluations, no duplicate manifests
    assert [r.outcome for r in r2.results] == ["reused", "reused"]
    assert [r.evaluation_id for r in r2.results] == \
        [r.evaluation_id for r in r1.results]
    assert env.eval_count() == evals0 + 2
    assert env.run_files() == runs0 | {f"suite-runs/{r1.suite_run_id}/manifest.json",
                                       f"suite-runs/{r2.suite_run_id}/manifest.json"}
    # storage grew by exactly the two immutable run manifests
    root = f.storage.root
    before = {p.relative_to(root).as_posix(): p.read_bytes()
              for p in root.rglob("*") if p.is_file()}
    r3 = env.run("m10-reuse", cstate(env.ck_b))
    after = {p.relative_to(root).as_posix(): p.read_bytes()
             for p in root.rglob("*") if p.is_file()}
    new = set(after) - set(before)
    assert new == {f"suite-runs/{r3.suite_run_id}/manifest.json"}
    for path in before:
        assert after[path] == before[path]                 # byte-identical
    assert [r.outcome for r in r3.results] == ["reused", "reused"]


def test_suite_probe_identity_is_exact_m4_identity(env):
    f = env.forge
    # direct M4 evaluation with identical fields
    direct = f.run_evaluation(EvaluationConfig(
        model_id=env.model_id, checkpoint_id=env.ck_b, dataset_id=env.ds,
        split="validation", tokenizer_id=env.tok, batch_size=8,
        max_seq_len=32, seed=10041))
    env.register_suite("m10-id", [env.probe(seed=10041)])
    rec = env.run("m10-id", cstate(env.ck_b))
    res = rec.results[0]
    # exact identity -> the suite run REUSED the direct evaluation
    assert res.outcome == "reused" and res.evaluation_id == direct.eval_id
    # a different probe (other seed) is a different M4 evaluation
    env.register_suite("m10-id2", [env.probe(seed=10042)])
    rec2 = env.run("m10-id2", cstate(env.ck_b))
    ev2 = f.get_evaluation(env.model_id, rec2.results[0].evaluation_id)
    assert rec2.results[0].outcome == "created"
    assert ev2.eval_id != direct.eval_id and ev2.result_hash != direct.result_hash
    # result_hash of the run is semantic: probe set differs -> differs
    assert rec2.result_hash != rec.result_hash


# =========================================================================== #
# Failure semantics
# =========================================================================== #

def test_unknown_suite_or_state_persists_nothing(env):
    f = env.forge
    runs_before = env.run_files()
    evals_before = env.eval_count()
    with pytest.raises(FileNotFoundError, match="probe suite"):
        env.run("m10-ghost-suite", cstate(env.ck_b))
    # corrupt suite manifest -> RuntimeError (409 class), nothing persisted
    env.register_suite("m10-corrupt-s", [env.probe(seed=10051)])
    (f.storage.root / "probe-suites" / "m10-corrupt-s" / "manifest.json") \
        .write_bytes(b"{broken")
    with pytest.raises(RuntimeError, match="corrupt"):
        env.run("m10-corrupt-s", cstate(env.ck_b))
    # invalid checkpoint state -> FileNotFoundError before any probe runs
    with pytest.raises(FileNotFoundError, match="not found"):
        env.run("m10-corrupt-s", cstate("ghost-ckpt"))   # suite also corrupt...
    env.register_suite("m10-ok-s", [env.probe(seed=10052)])
    with pytest.raises(FileNotFoundError, match="not found"):
        env.run("m10-ok-s", cstate("ghost-ckpt"))
    with pytest.raises(FileNotFoundError, match="model"):
        env.forge.run_suite(SuiteRunRequest(model_id="ghost-model",
                                            suite_id="m10-ok-s",
                                            state=cstate(env.ck_b)))
    assert env.run_files() == runs_before               # nothing persisted
    assert env.eval_count() == evals_before


def test_missing_dataset_probe_recorded_failure(env):
    evals_before = env.eval_count()
    env.register_suite("m10-missing-ds",
                       [env.probe(seed=10061, dsid="ghost-dataset")])
    rec = env.run("m10-missing-ds", cstate(env.ck_b))
    assert rec.status.value == "failed"
    assert rec.probe_count == 1 and rec.failed_count == 1
    assert rec.completed_count == 0
    res = rec.results[0]
    assert res.outcome == "failed" and res.evaluation_id is None
    assert "FileNotFoundError" in res.error and "ghost-dataset" in res.error
    assert env.eval_count() == evals_before             # nothing fabricated
    # deterministic failure text -> identical rerun reproduces the hash
    rec2 = env.run("m10-missing-ds", cstate(env.ck_b))
    assert rec2.status.value == "failed"
    assert rec2.result_hash == rec.result_hash
    assert rec2.results[0].error == rec.results[0].error


def test_partial_failure_keeps_valid_probes_and_records_failure(env):
    f = env.forge
    evals_before = env.eval_count()
    env.register_suite("m10-partial",
                       [env.probe(seed=10071),
                        env.probe(seed=10072, tok="ghost-tokenizer")])
    rec = env.run("m10-partial", cstate(env.ck_b))
    assert rec.status.value == "failed"
    assert rec.completed_count == 1 and rec.failed_count == 1
    by_seed = {r.probe.seed: r for r in rec.results}
    ok, bad = by_seed[10071], by_seed[10072]
    assert ok.outcome == "created" and ok.evaluation_id is not None
    assert bad.outcome == "failed" and bad.evaluation_id is None
    assert bad.error and "ghost-tokenizer" in bad.error
    assert env.eval_count() == evals_before + 1         # only the valid probe ran
    assert len(f.list_suite_runs(env.model_id)) >= 1    # failed run persisted
    # remaining probes still executed deterministically (order preserved)
    assert [r.probe.seed for r in rec.results] == [10071, 10072]


def test_corrupt_existing_evaluation_manifest_becomes_fresh_run(env):
    """Unreadable evidence is treated as unavailable: a fresh exact M4 run
    replaces it (recorded as created) — never a fabricated result."""
    f = env.forge
    evals_before = env.eval_count()
    md = f.storage.model_dir(env.model_id) / "evaluations"
    dirs_before = len([p for p in md.iterdir() if p.is_dir()])
    env.register_suite("m10-evl-corrupt", [env.probe(seed=10081)])
    r1 = env.run("m10-evl-corrupt", cstate(env.ck_b))
    evid = r1.results[0].evaluation_id
    mpath = md / f"eval-{evid}" / "manifest.json"
    mpath.write_bytes(b"{corrupt json")
    r2 = env.run("m10-evl-corrupt", cstate(env.ck_b))
    res = r2.results[0]
    assert res.outcome == "created" and res.evaluation_id != evid
    assert r2.status.value == "completed"
    # corrupt record is invisible to reuse; a fresh manifest was written
    assert env.eval_count() == evals_before + 1         # corrupt one hidden
    assert len([p for p in md.iterdir() if p.is_dir()]) == dirs_before + 2


# =========================================================================== #
# Persistence, listing, hashing
# =========================================================================== #

def test_persistence_one_manifest_and_deterministic_listing(env):
    f = env.forge
    env.register_suite("m10-persist", [env.probe(seed=10091)])
    before = [r.suite_run_id for r in f.list_suite_runs(env.model_id)]
    a = env.run("m10-persist", cstate(env.ck_a))
    time.sleep(0.01)
    b = env.run("m10-persist", cstate(env.ck_a))
    d = f.storage.root / "suite-runs" / a.suite_run_id
    assert [p.name for p in d.iterdir()] == ["manifest.json"]
    # manifest on disk round-trips to the returned record
    disk = SuiteRunRecord(**json.loads(
        (d / "manifest.json").read_text()))
    assert disk.model_dump() == a.model_dump()
    # deterministic list: oldest first, new runs appended in order
    ids = [r.suite_run_id for r in f.list_suite_runs(env.model_id)]
    assert ids == before + [a.suite_run_id, b.suite_run_id]
    assert f.list_suite_runs(env.model_id) == f.list_suite_runs(env.model_id)
    assert f.get_suite_run(env.model_id, a.suite_run_id).result_hash == \
        a.result_hash
    with pytest.raises(FileNotFoundError, match="not found"):
        f.get_suite_run(env.model_id, "ghost-run")
    with pytest.raises(FileNotFoundError):
        f.list_suite_runs("ghost-model")


def test_result_hash_semantics_and_bookkeeping_fields(env):
    f = env.forge
    env.register_suite("m10-hash", [env.probe(seed=10101)])
    st_ck, st_cur = cstate(env.ck_b), cstate(current=True)
    r_ck1 = env.run("m10-hash", st_ck)
    r_ck2 = env.run("m10-hash", st_ck)          # reuse path
    r_cur = env.run("m10-hash", st_cur)         # different state evidence
    assert r_ck1.result_hash == r_ck2.result_hash
    assert r_cur.result_hash != r_ck1.result_hash
    # ids/timestamps/durations do not influence the semantic hash
    mutant = r_ck1.model_copy(update={
        "suite_run_id": "different-id",
        "created_at": r_ck1.created_at.replace(year=1999),
        "duration_seconds": 1234.5})
    assert f.suite_runs.result_hash(mutant) == r_ck1.result_hash
    # counts are plain execution bookkeeping; arithmetic holds
    for r in (r_ck1, r_cur):
        assert r.completed_count + r.failed_count == r.probe_count
        assert r.reused_count <= r.completed_count
        assert set(r.model_dump(mode="json")) >= \
            {"probe_count", "completed_count", "reused_count", "failed_count"}
    # no aggregated metric exists anywhere in the record
    j = r_ck1.model_dump(mode="json")
    blob = json.dumps(j)
    assert not any(k in blob for k in
                   ("average", "avg_loss", "total_loss", "score", "ranking"))


def test_listing_skips_corrupt_run_manifests_and_foreign_runs(env):
    f = env.forge
    env.register_suite("m10-list", [env.probe(seed=10111)])
    r = env.run("m10-list", cstate(env.ck_a))
    # corrupt one run manifest -> list/get semantics
    mpath = f.storage.root / "suite-runs" / r.suite_run_id / "manifest.json"
    backup = mpath.read_bytes()
    mpath.write_bytes(b"{broken")
    with pytest.raises(Exception):
        f.get_suite_run(env.model_id, r.suite_run_id)   # corrupt -> parse error
    mpath.write_bytes(backup)                            # restore
    assert f.get_suite_run(env.model_id, r.suite_run_id).suite_run_id == \
        r.suite_run_id


# =========================================================================== #
# Request schema validation
# =========================================================================== #

def test_suite_run_request_validation(env):
    with pytest.raises(ValidationError):
        SuiteRunRequest(model_id=env.model_id, suite_id="x",
                        state=ComparisonState(state_kind=EvalStateKind.CURRENT),
                        **{"bogus": 1})                 # extra field forbidden
    with pytest.raises(ValidationError):
        SuiteRunRequest(model_id=env.model_id, suite_id="x",
                        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT))
    # checkpoint state without id is rejected at the state level
    with pytest.raises(ValidationError):
        ComparisonState(state_kind=EvalStateKind.CHECKPOINT, checkpoint_id=None)
    assert cstate(env.ck_b).checkpoint_id == env.ck_b


# =========================================================================== #
# HTTP API coverage
# =========================================================================== #

SUITE_RUNS = "/api/v1/suite-runs"
MODEL_SUITE_RUNS = "/api/v1/models/{mid}/suite-runs"
MODELS = "/api/v1/models"
TRAIN_RUN = "/api/v1/training/run"


def _http_env(api_client, tag: str, train: bool):
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", (f"{tag}.txt",
                                           _domain_bytes(120),
                                           "text/plain"))],
                         data={"name": f"api10-{tag}-ds"})
    ds = up.json()["dataset_id"]
    tok = api_client.post("/api/v1/tokenizers/train",
                          data={"config": json.dumps(
                              {"name": f"api10-{tag}-tok",
                               "vocab_size": 600}),
                                "dataset_id": ds}).json()["tokenizer"]["id"]
    api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                    json={"tokenizer_id": tok})
    cfg = {"name": f"api10-{tag}-model", "vocab_size": 640,
           "context_length": 64, "hidden_size": 64, "n_layers": 2,
           "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128}
    mid = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    ck_last = None
    if train:
        run = api_client.post(TRAIN_RUN, json={
            "method": "continued_pretraining", "model_id": mid,
            "dataset_id": ds, "tokenizer_id": tok, "learning_rate": 3e-3,
            "batch_size": 8, "max_seq_len": 32, "epochs": 3,
            "eval_every_steps": 3, "keep_best": False, "seed": 1})
        assert run.status_code == 200, run.text
        ck_last = api_client.get(f"{MODELS}/{mid}/checkpoints") \
            .json()[-1]["checkpoint_id"]
    return {"mid": mid, "ds": ds, "tok": tok, "ck": ck_last}


def test_suite_runs_api_lifecycle_reuse(api_client):
    h = _http_env(api_client, "life", train=True)
    mid, ds, tok, ck = h["mid"], h["ds"], h["tok"], h["ck"]
    probes = [{"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
               "batch_size": 8, "max_seq_len": 32, "seed": 5001},
              {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
               "batch_size": 8, "max_seq_len": 32, "seed": 5002}]
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api10-life-suite", "probes": probes})
    body = {"model_id": mid, "suite_id": "api10-life-suite",
            "state": {"state_kind": "checkpoint", "checkpoint_id": ck}}
    n_eval0 = len(api_client.get(f"{MODELS}/{mid}/evaluations").json())
    r = api_client.post(SUITE_RUNS, json=body)
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["status"] == "completed" and rec["probe_count"] == 2
    assert rec["reused_count"] == 0 and rec["failed_count"] == 0
    assert len(rec["results"]) == 2 and len(rec["result_hash"]) == 64
    assert all(x["outcome"] == "created" for x in rec["results"])
    assert n_eval0 + 2 == len(api_client.get(
        f"{MODELS}/{mid}/evaluations").json())
    # second identical run reuses; no duplicate evaluations
    r2 = api_client.post(SUITE_RUNS, json=body)
    assert r2.status_code == 200
    rec2 = r2.json()
    assert [x["outcome"] for x in rec2["results"]] == ["reused", "reused"]
    assert rec2["result_hash"] == rec["result_hash"]
    assert [x["evaluation_id"] for x in rec2["results"]] == \
        [x["evaluation_id"] for x in rec["results"]]
    assert len(api_client.get(f"{MODELS}/{mid}/evaluations").json()) == n_eval0 + 2
    # list + detail
    listing = api_client.get(MODEL_SUITE_RUNS.format(mid=mid)).json()
    assert [x["suite_run_id"] for x in listing] == [rec["suite_run_id"],
                                                    rec2["suite_run_id"]]
    got = api_client.get(f"{MODEL_SUITE_RUNS.format(mid=mid)}/"
                         f"{rec['suite_run_id']}")
    assert got.status_code == 200 and got.json()["result_hash"] == \
        rec["result_hash"]
    # dashboard stays valid + deterministic (evals render normally)
    dash = api_client.get(f"{MODELS}/{mid}/dashboard").json()
    assert dash["diagnostics"] == []
    evals_in_dash = {r["eval_id"] for g in dash["evaluations"]
                     for r in g["records"]}
    assert len(evals_in_dash) == n_eval0 + 2
    assert dash["result_hash"] == api_client.get(
        f"{MODELS}/{mid}/dashboard").json()["result_hash"]


def test_suite_runs_api_errors_and_immutability(api_client):
    h = _http_env(api_client, "err", train=False)
    mid, ds, tok = h["mid"], h["ds"], h["tok"]
    probe = {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": 5011}
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api10-err-suite", "probes": [probe]})
    body = {"model_id": mid, "suite_id": "api10-err-suite",
            "state": {"state_kind": "current"}}
    # valid run on the current state (fresh model weights exist)
    r = api_client.post(SUITE_RUNS, json=body)
    assert r.status_code == 200 and r.json()["status"] == "completed"
    run_id = r.json()["suite_run_id"]
    # unknown suite -> 404 (corrupt suite -> 409 is covered at engine level)
    bad_suite = dict(body, suite_id="api10-ghost-suite")
    assert api_client.post(SUITE_RUNS, json=bad_suite).status_code == 404
    # invalid checkpoint state -> 404, malformed request -> 422
    ghost_state = dict(body, state={"state_kind": "checkpoint",
                                    "checkpoint_id": "ghost-ck"})
    assert api_client.post(SUITE_RUNS, json=ghost_state).status_code == 404
    missing_ck = dict(body, state={"state_kind": "checkpoint"})
    assert api_client.post(SUITE_RUNS, json=missing_ck).status_code == 422
    unknown_model = dict(body, model_id="ghost-model")
    assert api_client.post(SUITE_RUNS, json=unknown_model).status_code == 404
    # list/detail: unknown model 404, unknown run 404, wrong model 404
    assert api_client.get(MODEL_SUITE_RUNS.format(mid="ghost-model")) \
        .status_code == 404
    assert api_client.get(f"{MODEL_SUITE_RUNS.format(mid=mid)}/ghost-run") \
        .status_code == 404
    # immutable artifacts: no update/delete/patch endpoints
    assert api_client.put(SUITE_RUNS).status_code == 405
    assert api_client.delete(SUITE_RUNS).status_code == 405
    assert api_client.patch(f"{MODEL_SUITE_RUNS.format(mid=mid)}/{run_id}") \
        .status_code == 405
    # repeated identical run stays deterministic
    r2 = api_client.post(SUITE_RUNS, json=body)
    assert r2.json()["result_hash"] == r.json()["result_hash"]


# =========================================================================== #
# M21: read-only per-suite grouping of the immutable suite-run history
# =========================================================================== #

BY_SUITE = "/api/v1/models/{mid}/suite-runs/by-suite/{suite}"


def _m21_env(env):
    """Two models, two registered suites, runs spread across both models
    (model_a: suite-x twice + suite-y once; model_b: suite-x once).
    Built once per module env and cached — every M21 engine test sees the
    exact same immutable history. Returns (a, b, suite_x, suite_y,
    a's suite-x runs, a's suite-y run, b's suite-x run).
    """
    cached = getattr(env, "_m21_state", None)
    if cached is not None:
        return cached
    a, b = env.model_id, env.fresh_model("m21-b")
    env.register_suite("m21-suite-x", [env.probe(seed=21001)])
    env.register_suite("m21-suite-y", [env.probe(seed=21002)])
    ra1 = env.run("m21-suite-x", cstate(env.ck_a), model_id=a)
    ra2 = env.run("m21-suite-x", cstate(env.ck_b), model_id=a)
    ra3 = env.run("m21-suite-y", cstate(env.ck_a), model_id=a)
    rb1 = env.run("m21-suite-x", cstate(current=True), model_id=b)
    env._m21_state = (a, b, "m21-suite-x", "m21-suite-y",
                      [ra1, ra2], [ra3], rb1)
    return env._m21_state


def test_m21_engine_filters_by_persisted_suite_and_model(env):
    a, b, sx, sy, ra, ra_y, rb1 = _m21_env(env)
    f = env.forge
    got = f.list_suite_runs_for_suite(a, sx)
    # every record belongs to the requested model AND suite; ordering is
    # the exact M10 convention ((created_at, suite_run_id) ascending)
    assert [r.suite_run_id for r in got] == [r.suite_run_id for r in ra]
    assert all(r.model_id == a and r.suite_id == sx for r in got)
    assert [(r.created_at, r.suite_run_id) for r in got] == \
        sorted((r.created_at, r.suite_run_id) for r in got)
    # a second suite of the same model returns only its own run
    got_y = f.list_suite_runs_for_suite(a, sy)
    assert [r.suite_run_id for r in got_y] == \
        [r.suite_run_id for r in ra_y]
    # payload parity: verbatim SuiteRunRecord equality with both the
    # authoritative M10 listing and the M10 single-record getter
    listing = {r.suite_run_id: r for r in f.list_suite_runs(a)}
    for r in got:
        assert r == listing[r.suite_run_id]
        assert r == f.get_suite_run(a, r.suite_run_id)
    # the other model's suite-x run never appears under this model
    assert rb1.suite_run_id not in {r.suite_run_id for r in got}
    assert all(r.model_id == a for r in got)


def test_m21_engine_valid_suite_without_runs_and_404s(env):
    a, b, sx, sy, ra, ra_y, rb1 = _m21_env(env)
    f = env.forge
    # valid suite registered, but model b never ran suite-y -> []
    assert f.list_suite_runs_for_suite(b, sy) == []
    # model b did run suite-x: exactly its own run, never model a's
    got_b = f.list_suite_runs_for_suite(b, sx)
    assert [r.suite_run_id for r in got_b] == [rb1.suite_run_id]
    assert {r.suite_run_id for r in got_b}.isdisjoint(
        {r.suite_run_id for r in ra})
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_suite_runs_for_suite("ghost-model-21", sx)
    # unknown suite -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_suite_runs_for_suite(a, "m21-ghost-suite")
    # read-only: the filter never writes run manifests
    before = env.run_files()
    f.list_suite_runs_for_suite(a, sx)
    f.list_suite_runs_for_suite(b, sy)
    assert env.run_files() == before


def test_m21_engine_repeated_calls_identical(env):
    a, _, sx, _, _, _, _ = _m21_env(env)
    f = env.forge
    first = [r.model_dump(mode="json")
             for r in f.list_suite_runs_for_suite(a, sx)]
    for _ in range(3):
        again = [r.model_dump(mode="json")
                 for r in f.list_suite_runs_for_suite(a, sx)]
        assert again == first


def test_m21_api_by_suite_grouping_and_parity(api_client):
    h = _http_env(api_client, "m21api", train=True)
    mid, ds, tok, ck = h["mid"], h["ds"], h["tok"], h["ck"]
    probes = [{"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
               "batch_size": 8, "max_seq_len": 32, "seed": 5201},
              {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
               "batch_size": 8, "max_seq_len": 32, "seed": 5202}]
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api21-suite-a", "probes": probes})
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api21-suite-b",
                          "probes": probes[:1]})
    body = {"model_id": mid, "state": {"state_kind": "checkpoint",
                                       "checkpoint_id": ck}}
    r1 = api_client.post(SUITE_RUNS,
                         json=dict(body, suite_id="api21-suite-a")).json()
    r2 = api_client.post(SUITE_RUNS,
                         json=dict(body, suite_id="api21-suite-a")).json()
    r3 = api_client.post(SUITE_RUNS,
                         json=dict(body, suite_id="api21-suite-b")).json()
    url = BY_SUITE.format(mid=mid, suite="api21-suite-a")
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly the two suite-a runs, deterministic M10 order, verbatim
    # payloads equal to the POST responses (existing representation)
    assert [x["suite_run_id"] for x in recs] == \
        [r1["suite_run_id"], r2["suite_run_id"]]
    assert all(x["suite_id"] == "api21-suite-a" and x["model_id"] == mid
               for x in recs)
    assert [(x["created_at"], x["suite_run_id"]) for x in recs] == \
        sorted((x["created_at"], x["suite_run_id"]) for x in recs)
    by_id = {x["suite_run_id"]: x for x in recs}
    assert by_id[r1["suite_run_id"]] == r1
    assert by_id[r2["suite_run_id"]] == r2
    # parity with the existing M10 listing filtered by persisted suite_id
    listing = api_client.get(MODEL_SUITE_RUNS.format(mid=mid)).json()
    assert recs == [x for x in listing if x["suite_id"] == "api21-suite-a"]
    # the suite-b run stays outside the suite-a grouping
    assert r3["suite_run_id"] not in {x["suite_run_id"] for x in recs}
    # repeated GET returns identical JSON (and identical raw bytes)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    assert raw1 == raw2 and json.loads(raw1) == recs
    # valid suite with no runs for a second model -> 200 + []
    cfg = {"name": "api21-b-model", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    empty = api_client.get(BY_SUITE.format(mid=mid_b, suite="api21-suite-a"))
    assert empty.status_code == 200 and empty.json() == []


def test_m21_api_404s_and_cross_model_isolation(api_client):
    h = _http_env(api_client, "m21err", train=True)
    mid, ds, tok, ck = h["mid"], h["ds"], h["tok"], h["ck"]
    probe = {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": 5211}
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api21-iso-suite", "probes": [probe]})
    run = api_client.post(SUITE_RUNS, json={
        "model_id": mid, "suite_id": "api21-iso-suite",
        "state": {"state_kind": "checkpoint", "checkpoint_id": ck}}).json()
    # unknown model -> 404 (even with a real suite id)
    assert api_client.get(BY_SUITE.format(mid="ghost-model-21",
                                          suite="api21-iso-suite")) \
        .status_code == 404
    # unknown suite -> 404 (even with a real model id)
    assert api_client.get(BY_SUITE.format(mid=mid,
                                          suite="api21-ghost-suite")) \
        .status_code == 404
    # cross-model isolation: another real model + this real suite -> the
    # other model's run ids are never exposed (valid suite, no runs -> [])
    cfg = {"name": "api21-iso-b", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    leak = api_client.get(BY_SUITE.format(mid=mid_b,
                                          suite="api21-iso-suite"))
    assert leak.status_code == 200 and leak.json() == []
    assert run["suite_run_id"] not in {x["suite_run_id"] for x in leak.json()}
    # detail getter keeps rejecting the other model's run id (unchanged)
    assert api_client.get(f"{MODEL_SUITE_RUNS.format(mid=mid_b)}/"
                          f"{run['suite_run_id']}").status_code == 404
    # by-suite never shadows the single-record getter for a real run id
    one = api_client.get(f"{MODEL_SUITE_RUNS.format(mid=mid)}/"
                         f"{run['suite_run_id']}")
    assert one.status_code == 200 and one.json() == run


def test_m21_api_existing_surfaces_and_openapi(api_client):
    h = _http_env(api_client, "m21surf", train=True)
    mid, ds, tok, ck = h["mid"], h["ds"], h["tok"], h["ck"]
    probe = {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": 5221}
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api21-surf-suite", "probes": [probe]})
    rec = api_client.post(SUITE_RUNS, json={
        "model_id": mid, "suite_id": "api21-surf-suite",
        "state": {"state_kind": "checkpoint", "checkpoint_id": ck}}).json()
    # existing M10 suite-run APIs unchanged: listing + detail + 404s
    listing = api_client.get(MODEL_SUITE_RUNS.format(mid=mid)).json()
    assert [x["suite_run_id"] for x in listing] == [rec["suite_run_id"]]
    assert api_client.get(
        f"{MODEL_SUITE_RUNS.format(mid=mid)}/{rec['suite_run_id']}") \
        .json() == rec
    assert api_client.get(MODEL_SUITE_RUNS.format(mid="ghost-model-21")) \
        .status_code == 404
    # by-suite agrees with the listing for this single-run model
    assert api_client.get(BY_SUITE.format(mid=mid,
                                          suite="api21-surf-suite")) \
        .json() == listing
    # M16–M20 sample-quality surface intact
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality").status_code == 200
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/records").status_code == 200
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/by-sample/no-such-sample") \
        .status_code == 404
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/by-checkpoint/no-such-ck") \
        .status_code == 404
    # new route documented correctly in OpenAPI (GET, right tag, schema)
    spec = api_client.get("/openapi.json").json()
    path = "/api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}"
    assert path in spec["paths"]
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["suite-runs"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/SuiteRunRecord"}
    assert "SuiteRunRecord" in spec["components"]["schemas"]
    # surface: 46 (M15 era) + 3 (M16) + 1 (M18) + 1 (M19) + 1 (M20)
    # + 1 (M21) + 1 (M22 summary) + 1 (M23 gates by-policy)
    # + 1 (M24 evaluations by-checkpoint)
    # + 1 (M25 suite-runs by-checkpoint)
    # + 1 (M26 comparisons by-checkpoint)
    # + 1 (M27 samples by-checkpoint)
    # + 1 (M28 evaluations by-dataset)
    # + 1 (M29 comparisons by-dataset)
    # + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe) = 67
    assert len(spec["paths"]) == 67


# =========================================================================== #
# M22: read-only per-suite bookkeeping summary over the M21 grouping
# =========================================================================== #

SUMMARY = "/api/v1/models/{mid}/suite-runs/by-suite/{suite}/summary"


def test_m22_engine_summary_fields_order_and_parity(env):
    a, b, sx, sy, ra, ra_y, rb1 = _m21_env(env)
    f = env.forge
    listing = f.list_suite_runs_for_suite(a, sx)
    s = f.list_suite_run_summary_for_suite(a, sx)
    # only identity/counting bookkeeping fields
    assert set(s.model_dump()) == {"model_id", "suite_id", "total_count",
                                   "run_ids", "earliest_created_at",
                                   "latest_created_at"}
    assert s.model_id == a and s.suite_id == sx
    assert s.total_count == 2
    # run ids are exactly the M21 listing, same deterministic order
    assert s.run_ids == [r.suite_run_id for r in listing]
    assert s.run_ids == [r.suite_run_id for r in ra]
    # earliest/latest are the summarized records' min/max created_at
    assert s.earliest_created_at == listing[0].created_at
    assert s.latest_created_at == listing[-1].created_at
    assert s.earliest_created_at == min(r.created_at for r in listing)
    assert s.latest_created_at == max(r.created_at for r in listing)
    # a second suite of the same model summarizes only its own run
    sy_sum = f.list_suite_run_summary_for_suite(a, sy)
    assert sy_sum.total_count == 1
    assert sy_sum.run_ids == [r.suite_run_id for r in ra_y]


def test_m22_engine_zero_summary_404s_and_read_only(env):
    a, b, sx, sy, ra, ra_y, rb1 = _m21_env(env)
    f = env.forge
    # valid registered suite, model b has no runs of it -> ZERO summary
    zero = f.list_suite_run_summary_for_suite(b, sy)
    assert zero.model_id == b and zero.suite_id == sy
    assert zero.total_count == 0 and zero.run_ids == []
    assert zero.earliest_created_at is None
    assert zero.latest_created_at is None
    # model b's own suite-x history stays separate (exactly its one run)
    b_sum = f.list_suite_run_summary_for_suite(b, sx)
    assert b_sum.total_count == 1 and b_sum.run_ids == [rb1.suite_run_id]
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_suite_run_summary_for_suite("ghost-model-22", sx)
    # unknown suite -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_suite_run_summary_for_suite(a, "m22-ghost-suite")
    # read-only: summaries never write run manifests
    before = env.run_files()
    f.list_suite_run_summary_for_suite(a, sx)
    f.list_suite_run_summary_for_suite(b, sy)
    assert env.run_files() == before


def test_m22_engine_repeated_calls_identical(env):
    a, _, sx, _, _, _, _ = _m21_env(env)
    f = env.forge
    first = f.list_suite_run_summary_for_suite(a, sx).model_dump(mode="json")
    for _ in range(3):
        again = f.list_suite_run_summary_for_suite(a, sx)
        assert again.model_dump(mode="json") == first


def test_m22_api_summary_parity_and_determinism(api_client):
    h = _http_env(api_client, "m22api", train=True)
    mid, ds, tok, ck = h["mid"], h["ds"], h["tok"], h["ck"]
    probes = [{"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
               "batch_size": 8, "max_seq_len": 32, "seed": 5301},
              {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
               "batch_size": 8, "max_seq_len": 32, "seed": 5302}]
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api22-suite-a", "probes": probes})
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api22-suite-b",
                          "probes": probes[:1]})
    body = {"model_id": mid, "state": {"state_kind": "checkpoint",
                                       "checkpoint_id": ck}}
    r1 = api_client.post(SUITE_RUNS,
                         json=dict(body, suite_id="api22-suite-a")).json()
    r2 = api_client.post(SUITE_RUNS,
                         json=dict(body, suite_id="api22-suite-a")).json()
    r3 = api_client.post(SUITE_RUNS,
                         json=dict(body, suite_id="api22-suite-b")).json()
    url = SUMMARY.format(mid=mid, suite="api22-suite-a")
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    s = got.json()
    # parity with the M21 by-suite listing of the same model + suite
    listing = api_client.get(
        BY_SUITE.format(mid=mid, suite="api22-suite-a")).json()
    assert s["model_id"] == mid and s["suite_id"] == "api22-suite-a"
    assert s["total_count"] == 2 == len(listing)
    assert s["run_ids"] == [x["suite_run_id"] for x in listing]
    assert s["run_ids"] == [r1["suite_run_id"], r2["suite_run_id"]]
    assert s["earliest_created_at"] == listing[0]["created_at"] == \
        min(x["created_at"] for x in listing)
    assert s["latest_created_at"] == listing[-1]["created_at"] == \
        max(x["created_at"] for x in listing)
    # the suite-b run stays out of the suite-a summary
    assert r3["suite_run_id"] not in s["run_ids"]
    # repeated GET returns identical JSON (and identical raw bytes)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    assert raw1 == raw2 and json.loads(raw1) == s
    # valid suite with no runs for a second model -> ZERO summary
    cfg = {"name": "api22-b-model", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    zero = api_client.get(SUMMARY.format(mid=mid_b, suite="api22-suite-a"))
    assert zero.status_code == 200
    assert zero.json() == {"model_id": mid_b, "suite_id": "api22-suite-a",
                           "total_count": 0, "run_ids": [],
                           "earliest_created_at": None,
                           "latest_created_at": None}


def test_m22_api_404s_isolation_and_prior_surfaces(api_client):
    h = _http_env(api_client, "m22err", train=True)
    mid, ds, tok, ck = h["mid"], h["ds"], h["tok"], h["ck"]
    probe = {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": 5311}
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api22-iso-suite", "probes": [probe]})
    run = api_client.post(SUITE_RUNS, json={
        "model_id": mid, "suite_id": "api22-iso-suite",
        "state": {"state_kind": "checkpoint", "checkpoint_id": ck}}).json()
    # unknown model -> 404 (even with a real suite id)
    assert api_client.get(SUMMARY.format(mid="ghost-model-22",
                                         suite="api22-iso-suite")) \
        .status_code == 404
    # unknown suite -> 404 (even with a real model id)
    assert api_client.get(SUMMARY.format(mid=mid,
                                         suite="api22-ghost-suite")) \
        .status_code == 404
    # cross-model isolation: another real model + this real suite -> a
    # ZERO summary; the first model's run ids are never counted/leaked
    cfg = {"name": "api22-iso-b", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    leak = api_client.get(SUMMARY.format(mid=mid_b,
                                         suite="api22-iso-suite"))
    assert leak.status_code == 200 and leak.json()["total_count"] == 0
    assert run["suite_run_id"] not in leak.json()["run_ids"]
    # M21 by-suite unchanged: still the full record list (not a summary)
    recs = api_client.get(BY_SUITE.format(mid=mid,
                                          suite="api22-iso-suite")).json()
    assert [x["suite_run_id"] for x in recs] == [run["suite_run_id"]]
    assert api_client.get(BY_SUITE.format(mid=mid_b,
                                          suite="api22-iso-suite")) \
        .json() == []
    # M10 listing/detail unchanged
    listing = api_client.get(MODEL_SUITE_RUNS.format(mid=mid)).json()
    assert run["suite_run_id"] in {x["suite_run_id"] for x in listing}
    assert api_client.get(
        f"{MODEL_SUITE_RUNS.format(mid=mid)}/{run['suite_run_id']}") \
        .json() == run
    # M16-M20 sample-quality surface intact
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality").status_code == 200
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/records").status_code == 200
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/by-sample/no-such-sample") \
        .status_code == 404
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/by-checkpoint/no-such-ck") \
        .status_code == 404


def test_m22_api_openapi_documented(api_client):
    spec = api_client.get("/openapi.json").json()
    path = ("/api/v1/models/{model_id}/suite-runs/by-suite/{suite_id}"
            "/summary")
    assert path in spec["paths"]
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["suite-runs"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema == {"$ref": "#/components/schemas/SuiteRunSummary"}
    assert "SuiteRunSummary" in spec["components"]["schemas"]
    # surface: 46 (M15 era) + 3 (M16) + 1 (M18) + 1 (M19) + 1 (M20)
    # + 1 (M21) + 1 (M22) + 1 (M23 gates by-policy)
    # + 1 (M24 evaluations by-checkpoint)
    # + 1 (M25 suite-runs by-checkpoint)
    # + 1 (M26 comparisons by-checkpoint)
    # + 1 (M27 samples by-checkpoint)
    # + 1 (M28 evaluations by-dataset)
    # + 1 (M29 comparisons by-dataset)
    # + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe) = 67
    assert len(spec["paths"]) == 67


# =========================================================================== #
# M25: read-only per-checkpoint grouping of the suite-run history
# =========================================================================== #

BY_CHECKPOINT = "/api/v1/models/{mid}/suite-runs/by-checkpoint/{ck}"


def _m25_env(env):
    """M25 state on top of the shared module env (cached): one more
    checkpoint-state run on ck_b, one CURRENT-state run on model a, and
    a fresh tiny trained model whose checkpoint has zero runs. Returns
    (a, m21_b, ck_a, ck_b, r_ck_b, r_cur, empty_model, empty_ck).
    """
    cached = getattr(env, "_m25_state", None)
    if cached is not None:
        return cached
    a, m21_b, sx, sy, ra, ra_y, rb1 = _m21_env(env)
    f = env.forge
    r_ck_b = env.run(sx, cstate(env.ck_b))                # checkpoint state
    r_cur = env.run(sx, cstate(current=True))             # current state
    empty = env.fresh_model("m25-empty", seed=250)
    f.run_training(TrainingConfig(
        method="continued_pretraining", model_id=empty, dataset_id=env.ds,
        tokenizer_id=env.tok, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, steps=10, eval_every_steps=5, keep_best=False,
        seed=25))
    empty_ck = f.list_checkpoints(empty)[0].checkpoint_id
    env._m25_state = (a, m21_b, env.ck_a, env.ck_b, r_ck_b, r_cur,
                      empty, empty_ck)
    return env._m25_state


def test_m25_engine_filters_by_persisted_state_and_model(env):
    a, m21_b, ck_a, ck_b, r_ck_b, r_cur, empty, empty_ck = _m25_env(env)
    f = env.forge
    listing = f.list_suite_runs(a)
    for ck in (ck_a, ck_b):
        got = f.list_suite_runs_for_checkpoint(a, ck)
        expected = [r for r in listing
                    if r.state.state_kind == EvalStateKind.CHECKPOINT
                    and r.state.checkpoint_id == ck]
        # parity with the authoritative M10 listing filtered by the
        # persisted run state; deterministic (created_at, suite_run_id)
        # order; every record belongs to the model AND checkpoint state
        assert got == expected
        assert all(r.model_id == a
                   and r.state.state_kind == EvalStateKind.CHECKPOINT
                   and r.state.checkpoint_id == ck for r in got)
        assert [(r.created_at, r.suite_run_id) for r in got] == \
            sorted((r.created_at, r.suite_run_id) for r in got)
        # verbatim payload parity with the M10 single-record getter
        for r in got:
            assert r == f.get_suite_run(a, r.suite_run_id)
    # the new ck_b run is grouped under ck_b, never under ck_a
    got_a = f.list_suite_runs_for_checkpoint(a, ck_a)
    got_b = f.list_suite_runs_for_checkpoint(a, ck_b)
    assert r_ck_b.suite_run_id in {r.suite_run_id for r in got_b}
    # checkpoint histories are disjoint (one run, one checkpoint state)
    assert {r.suite_run_id for r in got_a}.isdisjoint(
        {r.suite_run_id for r in got_b})
    # the CURRENT-state run (state.checkpoint_id None) never appears
    assert r_cur.state.state_kind == EvalStateKind.CURRENT
    assert r_cur.state.checkpoint_id is None
    assert r_cur.suite_run_id not in {r.suite_run_id for r in got_a}
    assert r_cur.suite_run_id not in {r.suite_run_id for r in got_b}


def test_m25_engine_empty_404s_cross_model_and_read_only(env):
    a, m21_b, ck_a, ck_b, r_ck_b, r_cur, empty, empty_ck = _m25_env(env)
    f = env.forge
    # valid registered checkpoint with zero suite runs -> []
    assert f.list_suite_runs_for_checkpoint(empty, empty_ck) == []
    # cross-model: the other model (no checkpoints of its own) cannot
    # resolve model a's checkpoint id, and vice versa
    with pytest.raises(FileNotFoundError):
        f.list_suite_runs_for_checkpoint(m21_b, ck_a)
    with pytest.raises(FileNotFoundError):
        f.list_suite_runs_for_checkpoint(a, empty_ck)
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_suite_runs_for_checkpoint("ghost-model-25", ck_a)
    # unknown checkpoint -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_suite_runs_for_checkpoint(a, "ghost-ck-25")
    # read-only: the filter never writes run manifests
    before = env.run_files()
    f.list_suite_runs_for_checkpoint(a, ck_a)
    f.list_suite_runs_for_checkpoint(empty, empty_ck)
    assert env.run_files() == before


def test_m25_engine_repeated_calls_identical(env):
    a, _, ck_a, _, _, _, _, _ = _m25_env(env)
    f = env.forge
    first = [r.model_dump(mode="json")
             for r in f.list_suite_runs_for_checkpoint(a, ck_a)]
    for _ in range(3):
        again = [r.model_dump(mode="json")
                 for r in f.list_suite_runs_for_checkpoint(a, ck_a)]
        assert again == first


def test_m25_api_by_checkpoint_grouping_parity_and_determinism(api_client):
    h = _http_env(api_client, "m25api", train=True)
    mid, ds, tok = h["mid"], h["ds"], h["tok"]
    ckpts = api_client.get(f"{MODELS}/{mid}/checkpoints").json()
    ck_a = ckpts[0]["checkpoint_id"]
    ck_z = ckpts[-1]["checkpoint_id"]
    probe = {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": 5501}
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api25-suite", "probes": [probe]})
    body_ck = {"model_id": mid, "suite_id": "api25-suite",
               "state": {"state_kind": "checkpoint", "checkpoint_id": ck_a}}
    r1 = api_client.post(SUITE_RUNS, json=body_ck).json()
    r2 = api_client.post(SUITE_RUNS, json=body_ck).json()
    r_cur = api_client.post(SUITE_RUNS, json={
        "model_id": mid, "suite_id": "api25-suite",
        "state": {"state_kind": "current"}}).json()
    r_z = None
    if ck_z != ck_a:
        r_z = api_client.post(SUITE_RUNS, json=dict(
            body_ck, state={"state_kind": "checkpoint",
                            "checkpoint_id": ck_z})).json()
    url = BY_CHECKPOINT.format(mid=mid, ck=ck_a)
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly the two ck_a runs, deterministic M10 order, verbatim
    # payloads equal to the POST responses (existing representation)
    assert [x["suite_run_id"] for x in recs] == \
        [r1["suite_run_id"], r2["suite_run_id"]]
    assert all(x["model_id"] == mid and x["state"]["state_kind"]
               == "checkpoint" and x["state"]["checkpoint_id"] == ck_a
               for x in recs)
    assert [(x["created_at"], x["suite_run_id"]) for x in recs] == \
        sorted((x["created_at"], x["suite_run_id"]) for x in recs)
    by_id = {x["suite_run_id"]: x for x in recs}
    assert by_id[r1["suite_run_id"]] == r1
    assert by_id[r2["suite_run_id"]] == r2
    # parity with the existing M10 listing filtered by persisted state
    listing = api_client.get(MODEL_SUITE_RUNS.format(mid=mid)).json()
    assert recs == [x for x in listing
                    if x["state"]["state_kind"] == "checkpoint"
                    and x["state"]["checkpoint_id"] == ck_a]
    # the current-state run (and the other checkpoint's run) stay outside
    assert r_cur["suite_run_id"] not in {x["suite_run_id"] for x in recs}
    if r_z is not None:
        assert r_z["suite_run_id"] not in {x["suite_run_id"] for x in recs}
        other = api_client.get(BY_CHECKPOINT.format(mid=mid, ck=ck_z))
        assert other.status_code == 200
        assert [x["suite_run_id"] for x in other.json()] == \
            [r_z["suite_run_id"]]
        assert {x["suite_run_id"] for x in other.json()}.isdisjoint(
            {x["suite_run_id"] for x in recs})
    # repeated GET returns identical JSON (and identical raw bytes)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    assert raw1 == raw2 and json.loads(raw1) == recs
    # valid checkpoint with no suite runs -> 200 + [] (a second trained
    # model whose checkpoints were never suite-run)
    h2 = _http_env(api_client, "m25empty", train=True)
    empty = api_client.get(BY_CHECKPOINT.format(mid=h2["mid"],
                                                ck=h2["ck"]))
    assert empty.status_code == 200 and empty.json() == []


def test_m25_api_404s_isolation_and_prior_surfaces(api_client):
    h = _http_env(api_client, "m25err", train=True)
    mid, ds, tok, ck = h["mid"], h["ds"], h["tok"], h["ck"]
    probe = {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": 5511}
    api_client.post("/api/v1/probe-suites",
                    json={"suite_id": "api25-iso-suite", "probes": [probe]})
    run = api_client.post(SUITE_RUNS, json={
        "model_id": mid, "suite_id": "api25-iso-suite",
        "state": {"state_kind": "checkpoint", "checkpoint_id": ck}}).json()
    # unknown model -> 404 (even with a real checkpoint id)
    assert api_client.get(BY_CHECKPOINT.format(mid="ghost-model-25",
                                               ck=ck)).status_code == 404
    # unknown checkpoint -> 404 (even with a real model id)
    assert api_client.get(BY_CHECKPOINT.format(mid=mid,
                                               ck="ghost-ck-25")) \
        .status_code == 404
    # cross-model isolation: another real model + this real checkpoint id
    # -> 404 (checkpoint ids are model-scoped through the M3 registry)
    cfg = {"name": "api25-iso-b", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    assert api_client.get(BY_CHECKPOINT.format(mid=mid_b, ck=ck)) \
        .status_code == 404
    # M10 listing/detail unchanged; by-checkpoint never shadows the
    # detail getter
    listing = api_client.get(MODEL_SUITE_RUNS.format(mid=mid)).json()
    assert run["suite_run_id"] in {x["suite_run_id"] for x in listing}
    assert api_client.get(
        f"{MODEL_SUITE_RUNS.format(mid=mid)}/{run['suite_run_id']}") \
        .json() == run
    assert api_client.get(f"{MODEL_SUITE_RUNS.format(mid=mid)}/ghost-run") \
        .status_code == 404
    # M21/M22 unchanged: by-suite grouping + summary of the same suite
    by_suite = api_client.get(
        BY_SUITE.format(mid=mid, suite="api25-iso-suite")).json()
    assert [x["suite_run_id"] for x in by_suite] == [run["suite_run_id"]]
    summary = api_client.get(
        f"{BY_SUITE.format(mid=mid, suite='api25-iso-suite')}/summary") \
        .json()
    assert summary["total_count"] == 1
    assert summary["run_ids"] == [run["suite_run_id"]]
    # M24 unchanged: evaluations by-checkpoint (the suite run created
    # checkpoint evaluations); M23 ghost policy 404; M18-M20 surface
    evs = api_client.get(f"{MODELS}/{mid}/evaluations/by-checkpoint/{ck}")
    assert evs.status_code == 200 and len(evs.json()) >= 1
    assert all(x["checkpoint_id"] == ck for x in evs.json())
    assert api_client.get(
        f"{MODELS}/{mid}/gates/decisions/by-policy/ghost-pol-25") \
        .status_code == 404
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality").status_code == 200
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/records").status_code == 200
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/by-sample/no-such-sample") \
        .status_code == 404
    assert api_client.get(
        f"{MODELS}/{mid}/sample-quality/by-checkpoint/no-such-ck") \
        .status_code == 404
    # new route documented correctly in OpenAPI (GET, right tag, schema)
    spec = api_client.get("/openapi.json").json()
    path = ("/api/v1/models/{model_id}/suite-runs/by-checkpoint/"
            "{checkpoint_id}")
    assert path in spec["paths"]
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["suite-runs"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/SuiteRunRecord"}
    assert "SuiteRunRecord" in spec["components"]["schemas"]
    # surface: 46 (M15 era) + 3 (M16) + 1 (M18) + 1 (M19) + 1 (M20)
    # + 1 (M21) + 1 (M22) + 1 (M23) + 1 (M24) + 1 (M25)
    # + 1 (M26 comparisons by-checkpoint)
    # + 1 (M27 samples by-checkpoint)
    # + 1 (M28 evaluations by-dataset)
    # + 1 (M29 comparisons by-dataset)
    # + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe) = 67
    assert len(spec["paths"]) == 67
