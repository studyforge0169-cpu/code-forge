"""Milestone 6 tests: stage-gate REST API (evaluate / list / get + error map).

HTTP mirror of the engine suite using the M5 two-domain recipe: domain-A
training improves across checkpoints; continued domain-B training regresses
the same model on the A probe.
"""
from __future__ import annotations

import json
import pathlib
import random

import torch

UPLOAD = "/api/v1/datasets/upload"
DATASETS = "/api/v1/datasets"
TOKENIZERS = "/api/v1/tokenizers"
MODELS = "/api/v1/models"
EVAL_RUN = "/api/v1/evaluations/run"
TRAIN_RUN = "/api/v1/training/run"
GATES = "/api/v1/gates/evaluate"
GATE_DECISIONS = "/api/v1/models/{model_id}/gates/decisions"

HEADS = ("river mountain cloud forest desert ocean valley island meadow canyon "
         "table chair lamp desk shelf couch rug clock mirror vase").split()
TAIL_A = ("flows stands gleams rises falls drifts looms shines hides waits").split()
TAIL_B = ("sings dances jumps laughs weeps shouts whispers roars murmurs chants").split()


def _domain_bytes(tails, n: int) -> bytes:
    rng = random.Random(11)
    out = []
    for i in range(n):
        h1, h2 = rng.sample(HEADS, 2)
        out.append(f"{h1} is {rng.choice(tails)} near {h2} with number {i}")
    return ("\n\n".join(out) + "\n").encode("utf-8")


def _upload_domain(api_client, tag: str, tails) -> tuple[str, str]:
    up = api_client.post(UPLOAD,
                         files=[("files", (f"{tag}.txt", _domain_bytes(tails, 240),
                                           "text/plain"))],
                         data={"name": f"api6-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds_id = up.json()["dataset_id"]

    tr = api_client.post(TOKENIZERS + "/train",
                         data={"config": json.dumps({"name": f"api6-{tag}-tok",
                                                     "vocab_size": 600}),
                               "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    tok_id = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={"tokenizer_id": tok_id}).status_code == 200
    return ds_id, tok_id


def _train(api_client, tag: str, model_id: str, ds_id: str, tok_id: str,
           epochs: int, eval_every: int) -> list[dict]:
    run = api_client.post(TRAIN_RUN, json={
        "name": f"api6-{tag}-run", "method": "continued_pretraining",
        "model_id": model_id, "dataset_id": ds_id, "tokenizer_id": tok_id,
        "learning_rate": 3e-3, "batch_size": 8, "max_seq_len": 32,
        "epochs": epochs, "eval_every_steps": eval_every, "keep_best": False,
        "lr_schedule": "constant", "seed": 1})
    assert run.status_code == 200, run.text
    return api_client.get(f"{MODELS}/{model_id}/checkpoints").json()


def _make_model(api_client, tag: str, tails=TAIL_A, epochs: int = 30,
                eval_every: int = 14) -> dict:
    """Domain + tokenizer + model + one training run on that domain."""
    ds_id, tok_id = _upload_domain(api_client, tag, tails)
    cfg = {"name": f"api6-{tag}-model", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    m = api_client.post(MODELS, json={"config": cfg})
    assert m.status_code == 201, m.text
    model_id = m.json()["model"]["id"]
    ckpts = _train(api_client, tag, model_id, ds_id, tok_id, epochs, eval_every)
    assert len(ckpts) >= 2
    return {"model_id": model_id, "ds_id": ds_id, "tok_id": tok_id,
            "ckpts": ckpts}


def _policy(model_id: str, ds_id: str, tok_id: str, *, baseline_type="checkpoint",
            baseline_ckpt=None, baseline_hash=None, minimum_loss=None,
            max_regression_delta=None, tolerance=1e-4, seed=1, name="api6-gate",
            split="validation", window=32, **overrides) -> dict:
    pol = {"name": name, "model_id": model_id, "dataset_id": ds_id,
           "split": split, "tokenizer_id": tok_id, "batch_size": 8,
           "max_seq_len": window, "seed": seed, "tolerance": tolerance,
           "baseline_type": baseline_type}
    if baseline_ckpt is not None:
        pol["baseline_checkpoint_id"] = baseline_ckpt
    if baseline_hash is not None:
        pol["baseline_result_hash"] = baseline_hash
    if minimum_loss is not None:
        pol["minimum_loss"] = minimum_loss
    if max_regression_delta is not None:
        pol["max_regression_delta"] = max_regression_delta
    pol.update(overrides)
    return pol


def _gate_body(model_id: str, ds_id: str, tok_id: str, *, baseline_type="checkpoint",
               baseline_ckpt=None, candidate=None, **policy_kwargs) -> dict:
    pol = _policy(model_id, ds_id, tok_id, baseline_type=baseline_type,
                  baseline_ckpt=baseline_ckpt, **policy_kwargs)
    cand = candidate or {"state_kind": "checkpoint"}
    return {"model_id": model_id, "policy": pol, "candidate": cand}


# --------------------------------------------------------------------------- #
# Full lifecycle: pass, fail + rollback suggestion, list/get, immutability
# --------------------------------------------------------------------------- #

def test_gate_lifecycle_pass_fail_suggest_api(api_client):
    env = _make_model(api_client, "flow", epochs=30)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    a_ck = env["ckpts"]
    a_final = a_ck[-1]["checkpoint_id"]
    a_early = a_ck[2]["checkpoint_id"]

    # continued training on domain B (tokenized with the A-trained tokenizer,
    # exactly like the engine recipe) regresses the model on the A probe
    up_b = api_client.post(UPLOAD,
                           files=[("files", ("flowb.txt",
                                             _domain_bytes(TAIL_B, 240),
                                             "text/plain"))],
                           data={"name": "api6-flowb-ds"})
    assert up_b.status_code == 201, up_b.text
    ds_b = up_b.json()["dataset_id"]
    tok_resp = api_client.post(f"{DATASETS}/{ds_b}/tokenize",
                               json={"tokenizer_id": tok})
    assert tok_resp.status_code == 200, tok_resp.text
    count_b = tok_resp.json()["splits"]["train"]["count"]
    sp_epoch = max(1, (count_b // 32) // 8)
    b_ck = _train(api_client, "flowb", mid, ds_b, tok, epochs=40,
                  eval_every=sp_epoch * 2)
    b_final = b_ck[-1]["checkpoint_id"]
    weights_before = api_client.get(f"{MODELS}/{mid}/weights").content

    # PASS: improvement candidate (A checkpoint vs later A checkpoint)
    r1 = api_client.post(GATES, json=_gate_body(mid, ds, tok,
                                                baseline_ckpt=a_early,
                                                candidate={"state_kind": "checkpoint",
                                                           "checkpoint_id": a_final}))
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert d1["decision"] == "passed"
    assert d1["verdict"] == "improved"
    assert d1["delta_loss_nats"] < -1.0
    assert d1["comparison_id"] and d1["comparison_result_hash"]
    assert d1["baseline"]["checkpoint_id"] == a_early
    assert d1["candidate"]["checkpoint_id"] == a_final
    assert d1["candidate"]["state_hash"] == a_ck[-1]["weights_sha256"]
    assert len(d1["result_hash"]) == 64

    # FAIL: regression vs the domain-A final checkpoint; rollback suggested
    r2 = api_client.post(GATES, json=_gate_body(mid, ds, tok,
                                                baseline_ckpt=a_final,
                                                candidate={"state_kind": "checkpoint",
                                                           "checkpoint_id": b_final}))
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["decision"] == "failed"
    assert d2["verdict"] == "regressed"
    assert d2["delta_loss_nats"] > 0.15
    assert d2["suggested_checkpoint_id"] == a_final
    assert d2["hint"] == "rollback recommended"

    # a bounded-regression policy PASSES the same candidate (different decision);
    # the bound rides just above the measured regression delta
    d_meas = d2["delta_loss_nats"]
    r3 = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_ckpt=a_final, max_regression_delta=d_meas + 0.5,
        candidate={"state_kind": "checkpoint", "checkpoint_id": b_final}))
    assert r3.status_code == 200
    assert r3.json()["decision"] == "passed"
    assert r3.json()["delta_loss_nats"] == d_meas     # evidence reused
    # ...and FAILS when the bound sits just below it
    r4 = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_ckpt=a_final,
        max_regression_delta=max(1e-6, d_meas - 0.5),
        candidate={"state_kind": "checkpoint", "checkpoint_id": b_final}))
    assert r4.status_code == 200
    assert r4.json()["decision"] == "failed"

    # list / get semantics + persisted manifest
    lst = api_client.get(GATE_DECISIONS.format(model_id=mid))
    assert lst.status_code == 200
    body = lst.json()
    assert len(body) == 4
    got = api_client.get(f"{MODELS}/{mid}/gates/decisions/{d2['decision_id']}")
    assert got.status_code == 200 and got.json() == d2
    assert api_client.get(f"{MODELS}/{mid}/gates/decisions/nope").status_code == 404

    # evidence chain: referenced evaluation records exist and match exactly
    ev = api_client.get(f"{MODELS}/{mid}/evaluations/"
                        f"{d2['candidate']['evaluation_id']}").json()
    assert ev["loss_nats"] == d2["candidate_loss"]
    assert ev["result_hash"] == d2["candidate"]["evaluation_result_hash"]
    cmp_rec = api_client.get(f"{MODELS}/{mid}/comparisons/"
                             f"{d2['comparison_id']}").json()
    assert cmp_rec["loss_a"] == d2["baseline_loss"]
    assert cmp_rec["loss_b"] == d2["candidate_loss"]
    assert cmp_rec["result_hash"] == d2["comparison_result_hash"]

    # the gate NEVER executed a rollback: weights + latest checkpoint intact
    assert api_client.get(f"{MODELS}/{mid}/weights").content == weights_before
    model = api_client.get(f"{MODELS}/{mid}").json()
    assert model["latest_checkpoint"] == b_final


# --------------------------------------------------------------------------- #
# Current-state baseline and absolute threshold mode
# --------------------------------------------------------------------------- #

def test_gate_current_baseline_and_threshold_api(api_client):
    env = _make_model(api_client, "cur", epochs=10, eval_every=10)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    ck = env["ckpts"][-1]["checkpoint_id"]

    # baseline = live weights (== last checkpoint after keep_best=False)
    r = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="current",
        candidate={"state_kind": "checkpoint", "checkpoint_id": ck}))
    assert r.status_code == 200, r.text
    dec = r.json()
    assert dec["decision"] == "passed" and dec["verdict"] == "unchanged"
    assert dec["baseline"]["state_kind"] == "current"
    assert dec["baseline"]["checkpoint_id"] is None
    assert dec["candidate"]["state_kind"] == "checkpoint"

    # learn the candidate loss on the gate probe, then gate against thresholds
    ev = api_client.post(EVAL_RUN, json={
        "model_id": mid, "checkpoint_id": ck, "dataset_id": ds,
        "split": "validation", "tokenizer_id": tok, "batch_size": 8,
        "max_seq_len": 32, "seed": 700}).json()
    loss = ev["loss_nats"]
    assert loss > 0

    passed = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="minimum_loss", minimum_loss=loss * 1.5,
        candidate={"state_kind": "checkpoint", "checkpoint_id": ck}, seed=700))
    assert passed.status_code == 200
    p = passed.json()
    assert p["decision"] == "passed"
    assert p["verdict"] is None and p["comparison_id"] is None
    assert p["baseline"] is None and p["baseline_loss"] is None
    assert p["candidate_loss"] == loss           # candidate eval reused

    failed = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="minimum_loss", minimum_loss=loss * 0.9,
        candidate={"state_kind": "checkpoint", "checkpoint_id": ck}, seed=700))
    assert failed.status_code == 200
    f = failed.json()
    assert f["decision"] == "failed"
    assert "threshold" in f["reason"]
    assert f["suggested_checkpoint_id"] is None   # no state baseline: no target


# --------------------------------------------------------------------------- #
# Evaluation-result-hash baseline (type C) over HTTP
# --------------------------------------------------------------------------- #

def test_gate_evaluation_result_hash_baseline_api(api_client):
    # 30 epochs so checkpoint order vs loss is monotone and robust
    env = _make_model(api_client, "hash", epochs=30)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    ck_last = env["ckpts"][-1]
    ck_first = env["ckpts"][0]["checkpoint_id"]

    ev = api_client.post(EVAL_RUN, json={
        "model_id": mid, "checkpoint_id": ck_last["checkpoint_id"],
        "dataset_id": ds, "split": "validation", "tokenizer_id": tok,
        "batch_size": 8, "max_seq_len": 32, "seed": 9000}).json()
    h = ev["result_hash"]

    # identical candidate vs its own recorded result -> unchanged -> passed
    ok = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="evaluation_result_hash", baseline_hash=h,
        candidate={"state_kind": "checkpoint",
                   "checkpoint_id": ck_last["checkpoint_id"]}, seed=9000))
    assert ok.status_code == 200, ok.text
    o = ok.json()
    assert o["decision"] == "passed" and o["verdict"] == "unchanged"
    assert o["baseline"]["evaluation_id"] == ev["eval_id"]
    assert o["baseline"]["evaluation_result_hash"] == h

    # earlier checkpoint vs the recorded result -> regressed -> failed
    bad = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="evaluation_result_hash", baseline_hash=h,
        candidate={"state_kind": "checkpoint", "checkpoint_id": ck_first},
        seed=9000))
    assert bad.status_code == 200
    assert bad.json()["decision"] == "failed"
    assert bad.json()["verdict"] == "regressed"
    assert bad.json()["delta_loss_nats"] > 0

    # probe mismatch: same hash, different seed -> 422, nothing persisted
    mismatch = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="evaluation_result_hash", baseline_hash=h,
        candidate={"state_kind": "checkpoint",
                   "checkpoint_id": ck_last["checkpoint_id"]}, seed=9001))
    assert mismatch.status_code == 422, mismatch.text
    assert "unrelated evaluation" in mismatch.json()["detail"]
    # missing hash -> 404
    ghost = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="evaluation_result_hash", baseline_hash="f" * 64,
        candidate={"state_kind": "checkpoint",
                   "checkpoint_id": ck_last["checkpoint_id"]}, seed=9000))
    assert ghost.status_code == 404
    # only the two valid decisions exist
    lst = api_client.get(GATE_DECISIONS.format(model_id=mid)).json()
    assert len(lst) == 2


# --------------------------------------------------------------------------- #
# Error mapping: 404 / 422 / 409
# --------------------------------------------------------------------------- #

def test_gate_errors_404_422_api(api_client):
    env = _make_model(api_client, "rej", epochs=10, eval_every=10)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    ck = env["ckpts"][-1]["checkpoint_id"]
    base = _gate_body(mid, ds, tok, baseline_ckpt=ck,
                      candidate={"state_kind": "checkpoint", "checkpoint_id": ck})

    # 404: missing model / dataset / tokenizer / version / checkpoint
    for payload in ({**base, "policy": {**base["policy"], "model_id": "ghost"},
                     "model_id": "ghost"},
                    {**base, "policy": {**base["policy"], "dataset_id": "ghost"}},
                    {**base, "policy": {**base["policy"], "tokenizer_id": "ghost"}},
                    {**base, "policy": {**base["policy"], "dataset_version": 99}},
                    {**base, "policy": {**base["policy"],
                                        "baseline_checkpoint_id": "ghost"}},
                    {**base, "candidate": {"state_kind": "checkpoint",
                                           "checkpoint_id": "ghost"}}):
        resp = api_client.post(GATES, json=payload)
        assert resp.status_code == 404, (payload, resp.text)

    # 422: malformed policies / candidates / probes
    cases = [
        {**base, "policy": {**base["policy"], "model_id": "someone-else"}},
        {**base, "policy": {**base["policy"], "baseline_type": "current",
                            "baseline_checkpoint_id": ck}},
        {**base, "policy": {**base["policy"], "baseline_type": "checkpoint",
                            "baseline_checkpoint_id": None}},
        {**base, "candidate": {"state_kind": "current", "checkpoint_id": "x"}},
        {**base, "candidate": {"state_kind": "checkpoint"}},
        {**base, "policy": {**base["policy"], "tolerance": -0.1}},
        {**base, "policy": {**base["policy"], "batch_size": 0}},
        {**base, "policy": {**base["policy"], "max_seq_len": 128}},
        {**base, "policy": {**base["policy"], "baseline_type": "minimum_loss"}},
        {**base, "policy": {**base["policy"], "baseline_type": "minimum_loss",
                            "minimum_loss": 1.0, "max_regression_delta": 0.5}},
        {**base, "policy": {**base["policy"], "baseline_type": "checkpoint",
                            "baseline_checkpoint_id": ck, "max_regression_delta": 0.0,
                            "tolerance": 1e-2}},
        {**base, "policy": {**base["policy"], "baseline_type": "zombie"}},
        {**base, "extra_field": 1},
    ]
    for payload in cases:
        resp = api_client.post(GATES, json=payload)
        assert resp.status_code == 422, (payload, resp.text)

    # refused runs created no decisions and no evaluations
    assert api_client.get(GATE_DECISIONS.format(model_id=mid)).json() == []
    assert api_client.get(f"{MODELS}/{mid}/evaluations").json() == []


def test_gate_corrupt_checkpoint_409_api(api_client):
    env = _make_model(api_client, "crp", epochs=10, eval_every=10)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    victim = env["ckpts"][-1]

    root = api_client.get("/api/v1/project").json()["storage_root"]
    wpath = (f"{root}/models/{mid}/checkpoints/"
             f"{victim['checkpoint_id']}/weights.pt")
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    tampered = {k: v.clone() for k, v in state.items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    resp = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_ckpt=victim["checkpoint_id"],
        candidate={"state_kind": "current"}))
    assert resp.status_code == 409, resp.text
    assert "integrity" in resp.json()["detail"]
    # refused before anything was persisted
    assert api_client.get(GATE_DECISIONS.format(model_id=mid)).json() == []
    assert api_client.get(f"{MODELS}/{mid}/evaluations").json() == []
    dl = api_client.get(f"{MODELS}/{mid}/weights")
    assert dl.status_code == 200
    manifest_dir = pathlib.Path(root) / "models" / mid / "gates"
    assert not manifest_dir.exists() or not list(manifest_dir.iterdir())


# --------------------------------------------------------------------------- #
# Reuse: repeated requests append decisions but duplicate no evidence
# --------------------------------------------------------------------------- #

def test_gate_reuse_and_append_api(api_client):
    env = _make_model(api_client, "reuse", epochs=30)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    a, b = env["ckpts"][2]["checkpoint_id"], env["ckpts"][-1]["checkpoint_id"]
    body = _gate_body(mid, ds, tok, baseline_ckpt=a, seed=800,
                      candidate={"state_kind": "checkpoint", "checkpoint_id": b})
    evals_url = f"{MODELS}/{mid}/evaluations"
    comps_url = f"{MODELS}/{mid}/comparisons"

    n_ev0 = len(api_client.get(evals_url).json())
    r1 = api_client.post(GATES, json=body).json()
    assert api_client.get(evals_url).json().__len__() == n_ev0 + 2
    r2 = api_client.post(GATES, json=body).json()
    evals = api_client.get(evals_url).json()
    assert len(evals) == n_ev0 + 2                 # exact reuse, no duplicates
    assert r1["candidate"]["evaluation_id"] == r2["candidate"]["evaluation_id"]
    assert r1["baseline"]["evaluation_id"] == r2["baseline"]["evaluation_id"]
    # comparison evidence reused by identity: only one comparison manifest
    comps = api_client.get(comps_url).json()
    assert [c["comparison_id"] for c in comps].count(r1["comparison_id"]) == 1
    assert r1["comparison_id"] == r2["comparison_id"]
    # decisions appended: deterministic hash, new ids
    assert r1["result_hash"] == r2["result_hash"]
    assert r1["decision_id"] != r2["decision_id"]
    decisions = api_client.get(GATE_DECISIONS.format(model_id=mid)).json()
    assert len(decisions) == 2
    got = api_client.get(f"{MODELS}/{mid}/gates/decisions/"
                         f"{r2['decision_id']}").json()
    assert got == r2


# --------------------------------------------------------------------------- #
# OpenAPI surface + empty/404 list semantics
# --------------------------------------------------------------------------- #

def test_gate_openapi_and_empty_model_api(api_client):
    spec = api_client.get("/openapi.json").json()["paths"]
    for p in ("/api/v1/gates/evaluate",
              "/api/v1/models/{model_id}/gates/decisions",
              "/api/v1/models/{model_id}/gates/decisions/{decision_id}"):
        assert p in spec, p

    m = api_client.post(MODELS, json={"config": {"name": "api6-empty-model",
                                                 "vocab_size": 64,
                                                 "context_length": 32,
                                                 "hidden_size": 32,
                                                 "n_layers": 2, "n_heads": 4,
                                                 "n_kv_heads": 2,
                                                 "intermediate_size": 48}})
    assert m.status_code == 201
    model_id = m.json()["model"]["id"]
    assert api_client.get(GATE_DECISIONS.format(model_id=model_id)).json() == []
    assert api_client.get(f"{MODELS}/{model_id}/gates/decisions/nope").status_code == 404
    assert api_client.get(f"{MODELS}/ghost/gates/decisions").status_code == 404
    assert api_client.get(f"{MODELS}/ghost/gates/decisions/nope").status_code == 404


# =========================================================================== #
# M23: read-only per-policy grouping of the gate-decision history (API)
# =========================================================================== #

POLICIES = "/api/v1/policies"
BY_POLICY = "/api/v1/models/{mid}/gates/decisions/by-policy/{pid}"


def _m23_cfg(tag: str) -> dict:
    return {"name": f"api23-{tag}-model", "vocab_size": 640,
            "context_length": 64, "hidden_size": 64, "n_layers": 2,
            "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128}


def test_m23_api_by_policy_grouping_parity_and_determinism(api_client):
    env = _make_model(api_client, "m23a", epochs=10, eval_every=10)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    ck_e = env["ckpts"][2]["checkpoint_id"]
    ck_f = env["ckpts"][-1]["checkpoint_id"]
    pol = _policy(mid, ds, tok, baseline_type="checkpoint",
                  baseline_ckpt=ck_e, seed=5401, name="api23-pol")
    assert api_client.post(POLICIES, json={
        "policy_id": "api23-pol-a", "description": "d",
        "policy": pol}).status_code == 201
    assert api_client.post(POLICIES, json={
        "policy_id": "api23-pol-b", "description": "d",
        "policy": dict(pol, seed=5402, name="api23-pol-b")}).status_code == 201
    body_a = {"model_id": mid, "policy_id": "api23-pol-a",
              "candidate": {"state_kind": "checkpoint",
                            "checkpoint_id": ck_f}}
    d1 = api_client.post(GATES, json=body_a).json()
    d2 = api_client.post(GATES, json=body_a).json()
    d3 = api_client.post(GATES, json=dict(body_a,
                                          policy_id="api23-pol-b")).json()
    d4 = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="checkpoint", baseline_ckpt=ck_e,
        seed=5403, candidate={"state_kind": "checkpoint",
                              "checkpoint_id": ck_f})).json()   # inline
    url = BY_POLICY.format(mid=mid, pid="api23-pol-a")
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly the two pol-a decisions, deterministic M6 order, verbatim
    # payloads equal to the evaluate responses (existing representation)
    assert [x["decision_id"] for x in recs] == \
        [d1["decision_id"], d2["decision_id"]]
    assert all(x["model_id"] == mid and x["policy_id"] == "api23-pol-a"
               for x in recs)
    assert [(x["created_at"], x["decision_id"]) for x in recs] == \
        sorted((x["created_at"], x["decision_id"]) for x in recs)
    by_id = {x["decision_id"]: x for x in recs}
    assert by_id[d1["decision_id"]] == d1
    assert by_id[d2["decision_id"]] == d2
    # parity with the existing M6 listing filtered by persisted policy_id
    listing = api_client.get(GATE_DECISIONS.format(model_id=mid)).json()
    assert recs == [x for x in listing
                    if x.get("policy_id") == "api23-pol-a"]
    # the pol-b decision and the inline decision stay outside
    assert d3["decision_id"] not in {x["decision_id"] for x in recs}
    assert d4["policy_id"] is None
    assert d4["decision_id"] not in {x["decision_id"] for x in recs}
    # repeated GET returns identical JSON (and identical raw bytes)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    assert raw1 == raw2 and json.loads(raw1) == recs


def test_m23_api_404s_isolation_and_prior_surfaces(api_client):
    env = _make_model(api_client, "m23e", epochs=10, eval_every=10)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    ck_f = env["ckpts"][-1]["checkpoint_id"]
    pol = _policy(mid, ds, tok, baseline_type="current", seed=5411,
                  name="api23-iso-pol")
    assert api_client.post(POLICIES, json={
        "policy_id": "api23-iso-pol", "description": "d",
        "policy": pol}).status_code == 201
    dec = api_client.post(GATES, json={
        "model_id": mid, "policy_id": "api23-iso-pol",
        "candidate": {"state_kind": "checkpoint",
                      "checkpoint_id": ck_f}}).json()
    # unknown model -> 404 (even with a real policy id)
    assert api_client.get(BY_POLICY.format(mid="ghost-model-23",
                                           pid="api23-iso-pol")) \
        .status_code == 404
    # unknown policy -> 404 (even with a real model id)
    assert api_client.get(BY_POLICY.format(mid=mid,
                                           pid="api23-ghost-pol")) \
        .status_code == 404
    # cross-model isolation: another real model + this real policy -> the
    # valid empty case (200 + []); no foreign decision leaks
    mid_b = api_client.post(MODELS, json={"config": _m23_cfg("iso-b")}) \
        .json()["model"]["id"]
    leak = api_client.get(BY_POLICY.format(mid=mid_b, pid="api23-iso-pol"))
    assert leak.status_code == 200 and leak.json() == []
    assert dec["decision_id"] not in {x["decision_id"] for x in leak.json()}
    # M6 listing/detail unchanged; by-policy never shadows the detail
    # getter
    listing = api_client.get(GATE_DECISIONS.format(model_id=mid)).json()
    assert dec["decision_id"] in {x["decision_id"] for x in listing}
    assert api_client.get(f"{GATE_DECISIONS.format(model_id=mid)}/"
                          f"{dec['decision_id']}").json() == dec
    assert api_client.get(f"{GATE_DECISIONS.format(model_id=mid)}/"
                          "ghost-dec").status_code == 404
    # M9 policy registry unchanged
    got_pol = api_client.get(f"{POLICIES}/api23-iso-pol")
    assert got_pol.status_code == 200
    assert got_pol.json()["policy_id"] == "api23-iso-pol"
    # M18-M20 sample-quality surface intact
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
    # M21/M22 suite-run surfaces intact (registered suite, no runs)
    probe = {"dataset_id": ds, "split": "validation", "tokenizer_id": tok,
             "batch_size": 8, "max_seq_len": 32, "seed": 5421}
    assert api_client.post("/api/v1/probe-suites",
                           json={"suite_id": "api23-suite",
                                 "probes": [probe]}).status_code == 201
    assert api_client.get(
        f"{MODELS}/{mid}/suite-runs/by-suite/api23-suite").json() == []
    summary = api_client.get(
        f"{MODELS}/{mid}/suite-runs/by-suite/api23-suite/summary").json()
    assert summary["total_count"] == 0 and summary["run_ids"] == []
    # new route documented correctly in OpenAPI (GET, right tag, schema)
    spec = api_client.get("/openapi.json").json()
    path = "/api/v1/models/{model_id}/gates/decisions/by-policy/{policy_id}"
    assert path in spec["paths"]
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["gates"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/GateDecision"}
    assert "GateDecision" in spec["components"]["schemas"]
    # surface: 46 (M15 era) + 3 (M16) + 1 (M18) + 1 (M19) + 1 (M20)
    # + 1 (M21) + 1 (M22) + 1 (M23) = 55 (pre-M24 ladder); + 1 (M24) + 1 (M25) + 1 (M26)
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


# --------------------------------------------------------------------------- #
# M34: gate-decision history by comparison (read-only grouping)
# --------------------------------------------------------------------------- #

BY_COMP = "/api/v1/models/{mid}/gates/decisions/by-comparison/{cid}"
_COMP_RUN = "/api/v1/comparisons/run"


def _comp34(mid, ds, tok, a_ck, b_ck, seed) -> dict:
    return {"model_id": mid,
            "state_a": {"state_kind": "checkpoint", "checkpoint_id": a_ck},
            "state_b": {"state_kind": "checkpoint", "checkpoint_id": b_ck},
            "dataset_id": ds, "split": "validation",
            "tokenizer_id": tok, "batch_size": 8, "max_seq_len": 32,
            "seed": seed, "tolerance": 1e-4}


def test_m34_api_by_comparison_grouping_determinism(api_client):
    env = _make_model(api_client, "m34a", epochs=24, eval_every=8)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    ck_e = env["ckpts"][2]["checkpoint_id"]
    ck_f = env["ckpts"][-1]["checkpoint_id"]

    # identical inline gates REUSE the same M5 evidence -> ONE
    # comparison with TWO decisions; a different seed -> its own
    # comparison; a threshold-only gate -> comparison_id null
    cand = {"state_kind": "checkpoint", "checkpoint_id": ck_f}
    g1a = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_ckpt=ck_e, candidate=cand,
        seed=6401)).json()
    g1b = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_ckpt=ck_e, candidate=cand,
        seed=6401)).json()
    g2 = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_ckpt=ck_e, candidate=cand,
        seed=6402)).json()
    g_none = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_type="minimum_loss", minimum_loss=1e9,
        candidate=cand, seed=6403)).json()
    assert g1a["comparison_id"] == g1b["comparison_id"]
    assert g1a["comparison_id"] != g2["comparison_id"]
    assert g_none["comparison_id"] is None

    url = BY_COMP.format(mid=mid, cid=g1a["comparison_id"])
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # authoritative-filter parity: exact subset of the M6 listing whose
    # persisted comparison_id matches, in the same order
    listing = api_client.get(GATE_DECISIONS.format(model_id=mid)).json()
    assert recs == [x for x in listing
                    if x["comparison_id"] == g1a["comparison_id"]]
    assert {x["decision_id"] for x in recs} == {g1a["decision_id"],
                                                g1b["decision_id"]}
    keyed = [(x["created_at"], x["decision_id"]) for x in recs]
    assert keyed == sorted(keyed)
    # verbatim: each element equals its POST payload and its
    # detail-getter payload (verdict/decision/losses/delta/reason)
    by_id = {x["decision_id"]: x for x in recs}
    assert by_id[g1a["decision_id"]] == g1a
    assert by_id[g1b["decision_id"]] == g1b
    for x in recs:
        one = api_client.get(
            f"{GATE_DECISIONS.format(model_id=mid)}/{x['decision_id']}")
        assert one.status_code == 200 and one.json() == x
    # deterministic: three repeats return identical raw bytes
    raws = {api_client.get(url).content for _ in range(3)}
    assert len(raws) == 1
    # partition: g2's group holds exactly its own decision, disjoint
    # from g1's group; null-comparison records belong to NO group
    recs2 = api_client.get(
        BY_COMP.format(mid=mid, cid=g2["comparison_id"])).json()
    assert [x["decision_id"] for x in recs2] == [g2["decision_id"]]
    ids1 = {x["decision_id"] for x in recs}
    ids2 = {x["decision_id"] for x in recs2}
    assert ids1.isdisjoint(ids2)
    non_null = {x["decision_id"] for x in listing
                if x["comparison_id"] is not None}
    assert ids1 | ids2 == non_null
    assert g_none["decision_id"] not in non_null
    # valid comparison with zero decisions -> [] (200): fresh M5 runs
    for seed in (6498, 6499):
        fresh = api_client.post(_COMP_RUN, json=_comp34(mid, ds, tok,
                                                        ck_e, ck_f, seed))
        assert fresh.status_code == 200, fresh.text
        empty = api_client.get(BY_COMP.format(
            mid=mid, cid=fresh.json()["comparison_id"]))
        assert empty.status_code == 200 and empty.json() == []
    # no side effects: the filter itself added no decisions
    assert len(api_client.get(
        GATE_DECISIONS.format(model_id=mid)).json()) == 4


def test_m34_api_404s_isolation_regressions_openapi(api_client):
    env = _make_model(api_client, "m34b", epochs=10, eval_every=10)
    mid, ds, tok = env["model_id"], env["ds_id"], env["tok_id"]
    ck_e = env["ckpts"][2]["checkpoint_id"]
    ck_f = env["ckpts"][-1]["checkpoint_id"]
    g1 = api_client.post(GATES, json=_gate_body(
        mid, ds, tok, baseline_ckpt=ck_e,
        candidate={"state_kind": "checkpoint", "checkpoint_id": ck_f},
        seed=6404)).json()
    # a second REAL model with no comparisons of its own
    mid_b = api_client.post(MODELS, json={"config": _m23_cfg("iso-34")}) \
        .json()["model"]["id"]

    # 404s: unknown model / unknown comparison (two ghost forms)
    assert api_client.get(BY_COMP.format(
        mid="ghost-model-34", cid=g1["comparison_id"])).status_code == 404
    assert api_client.get(BY_COMP.format(
        mid=mid, cid="ghost-comp-34")).status_code == 404
    assert api_client.get(
        f"{GATE_DECISIONS.format(model_id=mid)}/by-comparison/"
        "m34--not-a-real-comparison-id").status_code == 404
    assert api_client.get(BY_COMP.format(
        mid="ghost-model-34", cid="ghost-comp-34")).status_code == 404

    # cross-model isolation: this model's comparison id does not
    # resolve under the other model (comparisons are model-scoped
    # through the M5 registry)
    assert api_client.get(BY_COMP.format(
        mid=mid_b, cid=g1["comparison_id"])).status_code == 404

    # M6 listing/getter intact
    listing = api_client.get(GATE_DECISIONS.format(model_id=mid)).json()
    assert [x["decision_id"] for x in listing] == [g1["decision_id"]]
    got = api_client.get(
        f"{GATE_DECISIONS.format(model_id=mid)}/{g1['decision_id']}")
    assert got.status_code == 200 and got.json() == g1

    # M23 by-policy regression: registered policy grouping still works
    pol = _policy(mid, ds, tok, baseline_type="checkpoint",
                  baseline_ckpt=ck_e, seed=6405, name="api34-pol")
    assert api_client.post(POLICIES, json={
        "policy_id": "api34-pol-a", "description": "d",
        "policy": pol}).status_code == 201
    dp = api_client.post(GATES, json={
        "model_id": mid, "policy_id": "api34-pol-a",
        "candidate": {"state_kind": "checkpoint",
                      "checkpoint_id": ck_f}}).json()
    bypol = api_client.get(BY_POLICY.format(mid=mid, pid="api34-pol-a"))
    assert bypol.status_code == 200
    assert [x["decision_id"] for x in bypol.json()] == [dp["decision_id"]]
    assert api_client.get(BY_POLICY.format(mid=mid,
                                           pid="ghost-pol-34")
                          ).status_code == 404
    # ...and the policy decision lands in ITS comparison's group too
    grp = api_client.get(BY_COMP.format(mid=mid,
                                        cid=dp["comparison_id"])).json()
    assert [x["decision_id"] for x in grp] == [dp["decision_id"]]

    # generic detail getter still 404s ghost ids (no route capture)
    assert api_client.get(
        f"{GATE_DECISIONS.format(model_id=mid)}/ghost-dec-34"
    ).status_code == 404

    # M31 comparisons-by-tokenizer regression: the gated comparisons
    # group correctly under the tokenizer they measured with
    comps = api_client.get(f"{MODELS}/{mid}/comparisons").json()
    cmpt = api_client.get(f"{MODELS}/{mid}/comparisons/by-tokenizer/"
                          f"{tok}").json()
    assert cmpt == [c for c in comps if c["tokenizer_id"] == tok]

    # OpenAPI: 67 paths, the new path exactly once, GET-only, tag
    # gates, GateDecision items; route order M23 by-policy <
    # by-comparison < generic decision detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer) + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe) = 67
    assert len(spec["paths"]) == 67
    path = ("/api/v1/models/{model_id}/gates/decisions/by-comparison"
            "/{comparison_id}")
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["gates"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/GateDecision"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/gates/decisions/"
                      "by-policy/{policy_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/gates/decisions/{decision_id}")
