"""Milestone 5 tests: comparison REST API (run / list / get + error mapping)."""
from __future__ import annotations

import json
import random

import torch

UPLOAD = "/api/v1/datasets/upload"
DATASETS = "/api/v1/datasets"
TOKENIZERS = "/api/v1/tokenizers"
MODELS = "/api/v1/models"
EVAL_RUN = "/api/v1/evaluations/run"
COMP_RUN = "/api/v1/comparisons/run"

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


def _prepare(api_client, tag: str, tails, epochs: int, eval_every: int = 14):
    """Upload one domain, tokenize, create a model, train it.

    Returns (ds_id, tok_id, model_id, checkpoints).
    """
    up = api_client.post(UPLOAD,
                         files=[("files", (f"{tag}.txt", _domain_bytes(tails, 240),
                                           "text/plain"))],
                         data={"name": f"api5-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds_id = up.json()["dataset_id"]

    tr = api_client.post(TOKENIZERS + "/train",
                         data={"config": json.dumps({"name": f"api5-{tag}-tok",
                                                     "vocab_size": 600}),
                               "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    tok_id = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={"tokenizer_id": tok_id}).status_code == 200

    cfg = {"name": f"api5-{tag}-model", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    m = api_client.post(MODELS, json={"config": cfg})
    assert m.status_code == 201, m.text
    model_id = m.json()["model"]["id"]

    run = api_client.post("/api/v1/training/run", json={
        "name": f"api5-{tag}-run", "method": "continued_pretraining",
        "model_id": model_id, "dataset_id": ds_id, "tokenizer_id": tok_id,
        "learning_rate": 3e-3, "batch_size": 8, "max_seq_len": 32,
        "epochs": epochs, "eval_every_steps": eval_every, "keep_best": False,
        "lr_schedule": "constant", "seed": 1})
    assert run.status_code == 200, run.text
    ckpts = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(ckpts) >= 2
    return ds_id, tok_id, model_id, ckpts


def _comp(model_id: str, ds_id: str, tok_id: str, a_id: str, b_id: str,
          **overrides) -> dict:
    body = {"model_id": model_id,
            "state_a": {"state_kind": "checkpoint", "checkpoint_id": a_id},
            "state_b": {"state_kind": "checkpoint", "checkpoint_id": b_id},
            "dataset_id": ds_id, "split": "validation",
            "tokenizer_id": tok_id, "batch_size": 8, "max_seq_len": 32,
            "seed": 1, "tolerance": 1e-4}
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# Success path: run / list / get + improvement verdict
# --------------------------------------------------------------------------- #

def test_comparison_run_list_get(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "flow", TAIL_A,
                                              epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]

    resp = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                                early, final))
    assert resp.status_code == 200, resp.text
    rec = resp.json()
    assert rec["verdict"] == "improved"
    assert rec["delta_loss_nats"] < -1.0
    assert rec["state_a"]["checkpoint_id"] == early
    assert rec["state_b"]["checkpoint_id"] == final
    assert rec["state_a"]["state_hash"] == ckpts[2]["weights_sha256"]
    assert rec["state_b"]["state_hash"] == ckpts[-1]["weights_sha256"]
    assert len(rec["result_hash"]) == 64
    # evidence: referenced evaluations exist and match the losses exactly
    ev = api_client.get(f"{MODELS}/{model_id}/evaluations/"
                        f"{rec['state_b']['evaluation_id']}").json()
    assert ev["loss_nats"] == rec["loss_b"]
    assert ev["result_hash"] == rec["state_b"]["evaluation_result_hash"]

    lst = api_client.get(f"{MODELS}/{model_id}/comparisons")
    assert lst.status_code == 200
    body = lst.json()
    assert len(body) == 1 and body[0]["comparison_id"] == rec["comparison_id"]
    got = api_client.get(f"{MODELS}/{model_id}/comparisons/{rec['comparison_id']}")
    assert got.status_code == 200
    assert got.json() == rec                     # persisted immutable record


# --------------------------------------------------------------------------- #
# Evaluation reuse through the API
# --------------------------------------------------------------------------- #

def test_comparison_evaluation_reuse_via_api(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "reuse", TAIL_A,
                                              epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    body = _comp(model_id, ds_id, tok_id, early, final)
    evals_url = f"{MODELS}/{model_id}/evaluations"

    n0 = len(api_client.get(evals_url).json())
    r1 = api_client.post(COMP_RUN, json=body).json()
    n1 = len(api_client.get(evals_url).json())
    assert n1 == n0 + 2                          # both evaluations created
    r2 = api_client.post(COMP_RUN, json=body).json()
    n2 = len(api_client.get(evals_url).json())
    assert n2 == n1                              # both reused, none duplicated
    assert r1["state_a"]["evaluation_id"] == r2["state_a"]["evaluation_id"]
    assert r1["state_b"]["evaluation_id"] == r2["state_b"]["evaluation_id"]
    assert r1["result_hash"] == r2["result_hash"]
    assert r1["comparison_id"] != r2["comparison_id"]  # auditable history
    assert len(api_client.get(f"{MODELS}/{model_id}/comparisons").json()) == 2


# --------------------------------------------------------------------------- #
# List/get semantics
# --------------------------------------------------------------------------- #

def test_comparison_list_404_and_empty(api_client):
    assert api_client.get(f"{MODELS}/ghost/comparisons").status_code == 404
    assert api_client.get(f"{MODELS}/ghost/comparisons/nope").status_code == 404
    m = api_client.post(MODELS, json={"config": {"name": "api5-empty-model",
                                                 "vocab_size": 64,
                                                 "context_length": 32,
                                                 "hidden_size": 32,
                                                 "n_layers": 2, "n_heads": 4,
                                                 "n_kv_heads": 2,
                                                 "intermediate_size": 48}})
    assert m.status_code == 201
    model_id = m.json()["model"]["id"]
    assert api_client.get(f"{MODELS}/{model_id}/comparisons").json() == []
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #

def test_comparison_rejections(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "reject", TAIL_A,
                                              epochs=10, eval_every=10)
    ck = ckpts[-1]["checkpoint_id"]
    base = _comp(model_id, ds_id, tok_id, ck, ck)

    # 404 missing artifacts
    for payload in ({**base, "model_id": "ghost"},
                    {**base, "dataset_id": "ghost"},
                    {**base, "tokenizer_id": "ghost"},
                    {**base, "dataset_version": 99},
                    {**base, "state_a": {"state_kind": "checkpoint",
                                         "checkpoint_id": "ghost"}}):
        resp = api_client.post(COMP_RUN, json=payload)
        assert resp.status_code == 404, (payload, resp.text)

    # 422 invalid request shapes
    cases = [
        {**base, "state_a": {"state_kind": "current",
                             "checkpoint_id": "x"}},          # contradictory
        {**base, "state_b": {"state_kind": "checkpoint"}},    # missing id
        {**base, "split": "trainx"},
        {**base, "tolerance": -0.1},
        {**base, "batch_size": 0},
        {**base, "max_seq_len": 128},                         # > model context
        {**base, "unknown_extra": 1},
    ]
    for payload in cases:
        resp = api_client.post(COMP_RUN, json=payload)
        assert resp.status_code == 422, (payload, resp.text)

    # refused runs created no comparison or evaluation artifacts
    assert api_client.get(f"{MODELS}/{model_id}/comparisons").json() == []
    assert api_client.get(f"{MODELS}/{model_id}/evaluations").json() == []


def test_corrupt_checkpoint_comparison_409(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "corrupt", TAIL_A,
                                              epochs=10, eval_every=10)
    target = ckpts[-1]
    body = _comp(model_id, ds_id, tok_id, target["checkpoint_id"],
                 target["checkpoint_id"])

    # corrupt the checkpoint's weights file (valid archive, changed values)
    root = api_client.get("/api/v1/project").json()["storage_root"]
    wpath = (f"{root}/models/{model_id}/checkpoints/"
             f"{target['checkpoint_id']}/weights.pt")
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    tampered = {k: v.clone() for k, v in state.items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    resp = api_client.post(COMP_RUN, json=body)
    assert resp.status_code == 409
    assert "integrity" in resp.json()["detail"]
    # refused before any artifact: no comparisons, no evaluations, weights fine
    assert api_client.get(f"{MODELS}/{model_id}/comparisons").json() == []
    assert api_client.get(f"{MODELS}/{model_id}/evaluations").json() == []
    dl = api_client.get(f"{MODELS}/{model_id}/weights")
    assert dl.status_code == 200


# =========================================================================== #
# M26: read-only per-checkpoint grouping of the comparison history (API)
# =========================================================================== #

BY_CHECKPOINT = "/api/v1/models/{mid}/comparisons/by-checkpoint/{ck}"


def test_m26_api_by_checkpoint_grouping_dedup_and_determinism(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m26a", TAIL_A,
                                              epochs=30)
    ck1 = ckpts[2]["checkpoint_id"]
    ck2 = ckpts[-1]["checkpoint_id"]
    # cross-checkpoint A/B, same-checkpoint A=B, current-vs-checkpoint and
    # current-vs-current comparisons
    r1 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                              ck1, ck2, seed=2651)).json()
    r2 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                              ck1, ck1, seed=2652)).json()
    r3 = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, ck2, ck2, seed=2653,
        state_a={"state_kind": "current"})).json()
    r4 = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, ck2, ck2, seed=2654,
        state_a={"state_kind": "current"},
        state_b={"state_kind": "current"})).json()
    url = BY_CHECKPOINT.format(mid=model_id, ck=ck1)
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly r1 + r2 under ck1 — the A=B record appears ONCE — in the
    # deterministic M5 order, verbatim payloads equal to the run
    # responses (existing representation)
    assert [x["comparison_id"] for x in recs] == \
        [r1["comparison_id"], r2["comparison_id"]]
    assert len({x["comparison_id"] for x in recs}) == len(recs)
    assert [(x["created_at"], x["comparison_id"]) for x in recs] == \
        sorted((x["created_at"], x["comparison_id"]) for x in recs)
    by_id = {x["comparison_id"]: x for x in recs}
    assert by_id[r1["comparison_id"]] == r1
    assert by_id[r2["comparison_id"]] == r2
    # parity with the existing M5 listing filtered by the persisted sides
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert recs == [x for x in listing
                    if any(s["state_kind"] == "checkpoint"
                           and s["checkpoint_id"] == ck1
                           for s in (x["state_a"], x["state_b"]))]
    # ck2 history: r1 (side B) + r3 (side B) exactly once each; the
    # same-checkpoint r2 stays under ck1 only; the current-vs-current r4
    # and the current side of r3 never match anything
    got2 = api_client.get(BY_CHECKPOINT.format(mid=model_id, ck=ck2))
    assert got2.status_code == 200
    assert [x["comparison_id"] for x in got2.json()] == \
        [r1["comparison_id"], r3["comparison_id"]]
    assert all(r4["comparison_id"] not in
               {x["comparison_id"] for x in body} for body in
               (recs, got2.json()))
    # repeated GET returns identical JSON (and identical raw bytes)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    assert raw1 == raw2 and json.loads(raw1) == recs
    # valid checkpoint with no comparisons -> 200 + [] (a fresh model's
    # checkpoint)
    ds2, tok2, mid2, ck2b = _prepare(api_client, "m26empty", TAIL_B,
                                     epochs=8)
    empty = api_client.get(BY_CHECKPOINT.format(mid=mid2,
                                                ck=ck2b[-1]["checkpoint_id"]))
    assert empty.status_code == 200 and empty.json() == []


def test_m26_api_404s_isolation_and_prior_surfaces(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m26e", TAIL_A,
                                              epochs=8)
    ck = ckpts[-1]["checkpoint_id"]
    rec = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                               ckpts[0]["checkpoint_id"],
                                               ck, seed=2661)).json()
    # unknown model -> 404 (even with a real checkpoint id)
    assert api_client.get(BY_CHECKPOINT.format(mid="ghost-model-26",
                                               ck=ck)).status_code == 404
    # unknown checkpoint -> 404 (even with a real model id)
    assert api_client.get(BY_CHECKPOINT.format(mid=model_id,
                                               ck="ghost-ck-26")) \
        .status_code == 404
    # cross-model isolation: another real model + this real checkpoint id
    # -> 404 (checkpoint ids are model-scoped through the M3 registry)
    cfg = {"name": "api26-iso-b", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    assert api_client.get(BY_CHECKPOINT.format(mid=mid_b, ck=ck)) \
        .status_code == 404
    # M5 listing/detail unchanged; by-checkpoint never shadows the detail
    # getter
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert rec["comparison_id"] in {x["comparison_id"] for x in listing}
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/{rec['comparison_id']}") \
        .json() == rec
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/ghost-comp") \
        .status_code == 404
    # M6 gate surface intact (decisions listing + by-policy ghost 404)
    assert api_client.get(
        f"{MODELS}/{model_id}/gates/decisions").status_code == 200
    assert api_client.get(
        f"{MODELS}/{model_id}/gates/decisions/by-policy/ghost-pol-26") \
        .status_code == 404
    # M20/M24/M25 checkpoint-history surfaces intact on the same model
    assert api_client.get(
        f"{MODELS}/{model_id}/sample-quality/by-checkpoint/{ck}") \
        .status_code == 200
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/by-checkpoint/{ck}") \
        .status_code == 200
    assert api_client.get(
        f"{MODELS}/{model_id}/suite-runs/by-checkpoint/{ck}") \
        .status_code == 200
    # M21/M22 intact (registered suite, no runs)
    probe = {"dataset_id": ds_id, "split": "validation",
             "tokenizer_id": tok_id, "batch_size": 8, "max_seq_len": 32,
             "seed": 2671}
    assert api_client.post("/api/v1/probe-suites",
                           json={"suite_id": "api26-suite",
                                 "probes": [probe]}).status_code == 201
    assert api_client.get(
        f"{MODELS}/{model_id}/suite-runs/by-suite/api26-suite").json() == []
    assert api_client.get(
        f"{MODELS}/{model_id}/suite-runs/by-suite/api26-suite/summary") \
        .json()["total_count"] == 0
    # new route documented correctly in OpenAPI (GET, right tag, schema)
    spec = api_client.get("/openapi.json").json()
    path = ("/api/v1/models/{model_id}/comparisons/by-checkpoint/"
            "{checkpoint_id}")
    assert path in spec["paths"]
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["comparison"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/ComparisonRecord"}
    assert "ComparisonRecord" in spec["components"]["schemas"]
    # surface: 46 (M15 era) + 3 (M16) + 1 (M18) + 1 (M19) + 1 (M20)
    # + 1 (M21) + 1 (M22) + 1 (M23) + 1 (M24) + 1 (M25)
    # + 1 (M26 comparisons by-checkpoint) = 58
    assert len(spec["paths"]) == 58
