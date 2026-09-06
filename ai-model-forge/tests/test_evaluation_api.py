"""Milestone 4 tests: evaluation REST API (run / list / get + error mapping)."""
from __future__ import annotations

import io
import json
import random

import torch

from app.model_builder import content_hash

UPLOAD = "/api/v1/datasets/upload"
DATASETS = "/api/v1/datasets"
TOKENIZERS = "/api/v1/tokenizers"
MODELS = "/api/v1/models"
EVAL_RUN = "/api/v1/evaluations/run"

_WORDS = ("river mountain cloud forest desert ocean valley island meadow canyon "
          "table chair lamp desk shelf couch rug clock mirror vase").split()


def _files(name: str, content: bytes):
    return [("files", (name, content, "text/plain"))]


def _corpus(n: int, tag: str) -> bytes:
    rng = random.Random(hash(tag) % (2**32))
    lines = []
    for i in range(n):
        k = rng.randint(10, 22)
        lines.append(" ".join(rng.choices(_WORDS, k=k)) + f" number {i}")
    return ("\n\n".join(lines) + "\n").encode("utf-8")


def _weights_sha(client, model_id: str) -> str:
    dl = client.get(f"{MODELS}/{model_id}/weights")
    assert dl.status_code == 200
    return content_hash(torch.load(io.BytesIO(dl.content), map_location="cpu",
                                   weights_only=True))


def _prepare(api_client, tag: str) -> tuple[str, str, str, dict]:
    """Dataset + tokenizer + tokenized bins + trained model (keep_best=False).

    Returns (ds_id, tok_id, model_id, run_report).
    """
    up = api_client.post(UPLOAD, files=_files(f"{tag}.txt", _corpus(220, tag)),
                         data={"name": f"api4-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds_id = up.json()["dataset_id"]

    tr = api_client.post(TOKENIZERS + "/train",
                         data={"config": json.dumps({"name": f"api4-{tag}-tok",
                                                     "vocab_size": 320}),
                               "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    tok_id = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={"tokenizer_id": tok_id}).status_code == 200

    model_cfg = {"name": f"api4-{tag}-model", "vocab_size": 640,
                 "context_length": 64, "hidden_size": 64, "n_layers": 2,
                 "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128}
    m = api_client.post(MODELS, json={"config": model_cfg})
    assert m.status_code == 201, m.text
    model_id = m.json()["model"]["id"]

    run = api_client.post("/api/v1/training/run", json={
        "name": f"api4-{tag}-run", "method": "continued_pretraining",
        "model_id": model_id, "dataset_id": ds_id, "tokenizer_id": tok_id,
        "learning_rate": 3e-3, "batch_size": 8, "max_seq_len": 32,
        "steps": 14, "eval_every_steps": 7, "keep_best": False,
        "lr_schedule": "constant", "seed": 3})
    assert run.status_code == 200, run.text
    return ds_id, tok_id, model_id, run.json()


def _eval_cfg(model_id: str, ds_id: str, tok_id: str, **overrides) -> dict:
    cfg = {"model_id": model_id, "dataset_id": ds_id, "split": "validation",
           "tokenizer_id": tok_id, "batch_size": 8, "max_seq_len": 32,
           "seed": 11}
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #

def test_evaluation_run_list_get(api_client):
    ds_id, tok_id, model_id, run = _prepare(api_client, "flow")
    ckpts = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    final_val = ckpts[-1]["validation_loss"]

    resp = api_client.post(EVAL_RUN, json=_eval_cfg(model_id, ds_id, tok_id))
    assert resp.status_code == 200, resp.text
    rec = resp.json()
    assert rec["model_id"] == model_id and rec["state_kind"] == "current"
    assert rec["checkpoint_id"] is None
    assert rec["split"] == "validation" and rec["dataset_version"] == 1
    assert abs(rec["loss_nats"] - final_val) < 1e-4      # M3-consistent
    assert rec["token_count"] > 0 and rec["truncated"] is False
    assert rec["records_covered"] is not None
    assert len(rec["result_hash"]) == 64
    assert rec["state_hash"] == _weights_sha(api_client, model_id)

    # checkpoint state through the API (same schema, correct state hash)
    resp = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, checkpoint_id=ckpts[0]["checkpoint_id"]))
    assert resp.status_code == 200, resp.text
    ck_rec = resp.json()
    assert ck_rec["state_kind"] == "checkpoint"
    assert ck_rec["checkpoint_id"] == ckpts[0]["checkpoint_id"]
    assert ck_rec["state_hash"] == ckpts[0]["weights_sha256"]
    assert abs(ck_rec["loss_nats"] - ckpts[0]["validation_loss"]) < 1e-4

    # list: both records present, deterministic order; get returns them
    lst = api_client.get(f"{MODELS}/{model_id}/evaluations")
    assert lst.status_code == 200
    body = lst.json()
    assert len(body) == 2
    assert {r["state_kind"] for r in body} == {"current", "checkpoint"}
    assert api_client.get(f"{MODELS}/{model_id}/evaluations").json() == body
    assert body == sorted(body, key=lambda r: (r["created_at"], r["eval_id"]))
    got = api_client.get(f"{MODELS}/{model_id}/evaluations/{body[0]['eval_id']}")
    assert got.status_code == 200
    assert got.json() == body[0]              # persisted record is immutable
    assert got.json()["result_hash"] == body[0]["result_hash"]


# --------------------------------------------------------------------------- #
# List/get semantics
# --------------------------------------------------------------------------- #

def test_evaluation_list_404_and_empty(api_client):
    assert api_client.get(f"{MODELS}/ghost/evaluations").status_code == 404
    assert api_client.get(f"{MODELS}/ghost/evaluations/nope").status_code == 404
    body = {"config": {"name": "api4-norec-model", "vocab_size": 64,
                       "context_length": 32, "hidden_size": 32, "n_layers": 2,
                       "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 48}}
    m = api_client.post(MODELS, json=body)
    assert m.status_code == 201
    model_id = m.json()["model"]["id"]
    assert api_client.get(f"{MODELS}/{model_id}/evaluations").status_code == 200
    assert api_client.get(f"{MODELS}/{model_id}/evaluations").json() == []
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Rejections + integrity
# --------------------------------------------------------------------------- #

def test_evaluation_rejections(api_client):
    ds_id, tok_id, model_id, _ = _prepare(api_client, "reject")
    base = _eval_cfg(model_id, ds_id, tok_id)

    cases_404 = [
        {**base, "model_id": "ghost"},
        {**base, "dataset_id": "ghost"},
        {**base, "tokenizer_id": "ghost"},
        {**base, "dataset_version": 99},
        {**base, "checkpoint_id": "ghost"},
    ]
    for payload in cases_404:
        assert api_client.post(EVAL_RUN, json=payload).status_code == 404, payload

    cases_422 = [
        {**base, "split": "trainx"},
        {**base, "split": "val"},
        {**base, "split": "validation2"},
        {**base, "max_seq_len": 128},          # > model context_length
        {**base, "batch_size": 0},
        {**base, "max_eval_tokens": 0},
        {**base, "unknown_field": 1},
    ]
    for payload in cases_422:
        resp = api_client.post(EVAL_RUN, json=payload)
        assert resp.status_code == 422, (payload, resp.text)

    # dataset without a tokenized artifact -> 404
    up = api_client.post(UPLOAD, files=_files("raw.txt", _corpus(40, "untok")),
                         data={"name": "api4-untok-ds"})
    assert up.status_code == 201
    resp = api_client.post(EVAL_RUN, json={**base, "dataset_id": up.json()["dataset_id"]})
    assert resp.status_code == 404
    assert "tokenized artifact" in resp.json()["detail"]

    # refused runs create no evaluation records
    lst = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    assert lst == []


def test_corrupt_checkpoint_eval_409_and_read_only(api_client):
    ds_id, tok_id, model_id, run = _prepare(api_client, "corrupt")
    ckpts = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    target = ckpts[0]
    weights_before = _weights_sha(api_client, model_id)

    # corrupt the checkpoint's weights file (valid archive, changed values)
    root = api_client.get("/api/v1/project").json()["storage_root"]
    wpath = (f"{root}/models/{model_id}/checkpoints/"
             f"{target['checkpoint_id']}/weights.pt")
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    tampered = {k: v.clone() for k, v in state.items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    resp = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, checkpoint_id=target["checkpoint_id"]))
    assert resp.status_code == 409
    assert "integrity" in resp.json()["detail"]
    # refused before any artifact: no new evaluation manifest, weights intact
    assert api_client.get(f"{MODELS}/{model_id}/evaluations").json() == []
    assert _weights_sha(api_client, model_id) == weights_before
    # current-state evaluation still works (untouched)
    ok = api_client.post(EVAL_RUN, json=_eval_cfg(model_id, ds_id, tok_id))
    assert ok.status_code == 200


def test_evaluation_schema_validation_422(api_client):
    ds_id, tok_id, model_id, _ = _prepare(api_client, "schema")
    base = _eval_cfg(model_id, ds_id, tok_id)
    resp = api_client.post(EVAL_RUN, json={**base, "model_id": ""})
    assert resp.status_code == 422
    resp = api_client.post(EVAL_RUN, json={**base, "checkpoint_id": ""})
    assert resp.status_code == 422
    resp = api_client.post(EVAL_RUN, json={**base, "split": "test"})
    assert resp.status_code == 200            # enum valid values accepted


# =========================================================================== #
# M24: read-only per-checkpoint grouping of the evaluation history (API)
# =========================================================================== #

BY_CHECKPOINT = "/api/v1/models/{mid}/evaluations/by-checkpoint/{ck}"


def test_m24_api_by_checkpoint_grouping_parity_and_determinism(api_client):
    ds_id, tok_id, model_id, _ = _prepare(api_client, "m24a")
    ckpts = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    ck_a, ck_b = ckpts[0]["checkpoint_id"], ckpts[1]["checkpoint_id"]
    # two checkpoint evals on ck_a (different seeds), one on ck_b, one
    # current-state eval
    r1 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, checkpoint_id=ck_a, seed=2451)).json()
    r2 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, checkpoint_id=ck_a, seed=2452)).json()
    r3 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, checkpoint_id=ck_b, seed=2453)).json()
    r4 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, seed=2454)).json()      # current state
    assert r4["checkpoint_id"] is None and r4["state_kind"] == "current"
    url = BY_CHECKPOINT.format(mid=model_id, ck=ck_a)
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly the two ck_a evaluations, deterministic M4 order, verbatim
    # payloads equal to the run responses (existing representation)
    assert [x["eval_id"] for x in recs] == [r1["eval_id"], r2["eval_id"]]
    assert all(x["model_id"] == model_id and x["state_kind"] == "checkpoint"
               and x["checkpoint_id"] == ck_a for x in recs)
    assert [(x["created_at"], x["eval_id"]) for x in recs] == \
        sorted((x["created_at"], x["eval_id"]) for x in recs)
    by_id = {x["eval_id"]: x for x in recs}
    assert by_id[r1["eval_id"]] == r1
    assert by_id[r2["eval_id"]] == r2
    # parity with the existing M4 listing filtered by persisted
    # checkpoint identity
    listing = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    assert recs == [x for x in listing if x["checkpoint_id"] == ck_a]
    # the ck_b evaluation and the current-state evaluation stay outside
    assert r3["eval_id"] not in {x["eval_id"] for x in recs}
    assert r4["eval_id"] not in {x["eval_id"] for x in recs}
    # repeated GET returns identical JSON (and identical raw bytes)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    assert raw1 == raw2 and json.loads(raw1) == recs
    # valid checkpoint with no evaluations -> 200 + [] (later checkpoints
    # of this fresh model were never evaluated)
    empty_ck = ckpts[-1]["checkpoint_id"] if len(ckpts) > 2 else ck_b
    if empty_ck != ck_b:
        empty = api_client.get(BY_CHECKPOINT.format(mid=model_id,
                                                    ck=empty_ck))
        assert empty.status_code == 200 and empty.json() == []


def test_m24_api_404s_isolation_and_prior_surfaces(api_client):
    ds_id, tok_id, model_id, _ = _prepare(api_client, "m24e")
    ckpts = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    ck_a = ckpts[0]["checkpoint_id"]
    rec = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, checkpoint_id=ck_a, seed=2461)).json()
    # a second real model (own weights, no checkpoints of its own needed)
    cfg = {"name": "api24-iso-b", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    # unknown model -> 404 (even with a real checkpoint id)
    assert api_client.get(BY_CHECKPOINT.format(mid="ghost-model-24",
                                               ck=ck_a)).status_code == 404
    # unknown checkpoint -> 404 (even with a real model id)
    assert api_client.get(BY_CHECKPOINT.format(mid=model_id,
                                               ck="ghost-ck-24")) \
        .status_code == 404
    # cross-model isolation: another real model + this real checkpoint id
    # -> 404 (checkpoint ids are model-scoped through the M3 registry)
    assert api_client.get(BY_CHECKPOINT.format(mid=mid_b, ck=ck_a)) \
        .status_code == 404
    # M4 listing/get unchanged; by-checkpoint never shadows the detail
    # getter
    listing = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    assert rec["eval_id"] in {x["eval_id"] for x in listing}
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/{rec['eval_id']}").json() == rec
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/ghost-eval").status_code == 404
    # M3 checkpoint registry unchanged
    one = api_client.get(f"{MODELS}/{model_id}/checkpoints/{ck_a}")
    assert one.status_code == 200 and one.json()["checkpoint_id"] == ck_a
    # M20/M23 regression: sample-quality by-checkpoint and gates
    # by-policy semantics unchanged
    assert api_client.get(
        f"{MODELS}/{mid_b}/sample-quality").json() == []
    assert api_client.get(
        f"{MODELS}/{mid_b}/sample-quality/by-checkpoint/{ck_a}") \
        .status_code == 404
    assert api_client.get(
        f"{MODELS}/{mid_b}/gates/decisions/by-policy/ghost-pol-24") \
        .status_code == 404
    # new route documented correctly in OpenAPI (GET, right tag, schema)
    spec = api_client.get("/openapi.json").json()
    path = "/api/v1/models/{model_id}/evaluations/by-checkpoint/{checkpoint_id}"
    assert path in spec["paths"]
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["evaluation"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/EvaluationRecord"}
    assert "EvaluationRecord" in spec["components"]["schemas"]
    # surface: 46 (M15 era) + 3 (M16) + 1 (M18) + 1 (M19) + 1 (M20)
    # + 1 (M21) + 1 (M22) + 1 (M23) + 1 (M24) + 1 (M25)
    # + 1 (M26 comparisons by-checkpoint)
    # + 1 (M27 samples by-checkpoint)
    # + 1 (M28 evaluations by-dataset)
    # + 1 (M29 comparisons by-dataset)
    # + 1 (M30 evaluations by-tokenizer) = 62
    assert len(spec["paths"]) == 62


# =========================================================================== #
# M28: read-only per-dataset grouping of the evaluation history (API)
# =========================================================================== #

BY_DATASET = "/api/v1/models/{mid}/evaluations/by-dataset/{ds}"


def _m28_second_dataset(api_client, tag: str, tok_id: str) -> str:
    """A second valid tokenized dataset for cross-dataset coverage."""
    up = api_client.post(UPLOAD, files=_files(f"{tag}.txt",
                                              _corpus(180, tag)),
                         data={"name": f"api28-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds2 = up.json()["dataset_id"]
    assert api_client.post(f"{DATASETS}/{ds2}/tokenize",
                           json={"tokenizer_id": tok_id}).status_code == 200
    return ds2


def test_m28_api_by_dataset_grouping_versions_determinism(api_client):
    ds_id, tok_id, model_id, _ = _prepare(api_client, "m28a")
    ds2 = _m28_second_dataset(api_client, "m28b", tok_id)
    # two evaluations on ds (distinct seeds), one on the second dataset
    e1 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, seed=2811)).json()
    e2 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, seed=2812)).json()
    e3 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds2, tok_id, seed=2813)).json()
    url = BY_DATASET.format(mid=model_id, ds=ds_id)
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly e1 + e2 under ds, in the deterministic M4 order, verbatim
    # equal to the run responses (existing representation)
    assert [x["eval_id"] for x in recs] == [e1["eval_id"], e2["eval_id"]]
    keyed = [(x["created_at"], x["eval_id"]) for x in recs]
    assert keyed == sorted(keyed)
    by_id = {x["eval_id"]: x for x in recs}
    assert by_id[e1["eval_id"]] == e1
    assert by_id[e2["eval_id"]] == e2
    # persisted dataset identity + version travel VERBATIM (no
    # normalization, no version rewriting)
    assert all(x["dataset_id"] == ds_id and x["dataset_version"]
               == e1["dataset_version"] for x in recs)
    # parity with the M4 listing filtered locally by the persisted
    # dataset identity — no missing, no extra, no duplicates
    listing = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    assert recs == [x for x in listing if x["dataset_id"] == ds_id]
    # the second dataset holds exactly e3; no cross-dataset leakage
    got2 = api_client.get(BY_DATASET.format(mid=model_id, ds=ds2))
    assert got2.status_code == 200
    assert [x["eval_id"] for x in got2.json()] == [e3["eval_id"]]
    # repeated GET returns identical raw bytes (3 repeats)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    raw3 = api_client.get(url).content
    assert raw1 == raw2 == raw3 and json.loads(raw1) == recs


def test_m28_api_404s_scoping_regressions_openapi(api_client):
    ds_id, tok_id, model_id, rep = _prepare(api_client, "m28e")
    ck = rep["checkpoints"][0]["checkpoint_id"]
    # unknown model / unknown dataset -> 404 (well-formed + malformed)
    assert api_client.get(BY_DATASET.format(mid="ghost-model-28",
                                            ds=ds_id)).status_code == 404
    assert api_client.get(BY_DATASET.format(mid=model_id,
                                            ds="ghost-ds-28")).status_code \
        == 404
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/by-dataset/ds%20id%2028!!") \
        .status_code == 404
    # model scoping: datasets are GLOBAL, so another real model + this
    # valid dataset is the natural model-scoped EMPTY case (200 + []),
    # and no evaluation id of model_id leaks into it
    ds2_id, tok2_id, model2_id, _ = _prepare(api_client, "m28iso")
    scoped = api_client.get(BY_DATASET.format(mid=model2_id, ds=ds_id))
    assert scoped.status_code == 200 and scoped.json() == []
    # M4 regression: run/list/get + validation unchanged
    e = api_client.post(EVAL_RUN, json=_eval_cfg(model_id, ds_id, tok_id,
                                                 seed=2814)).json()
    assert e["dataset_id"] == ds_id
    lst = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    assert e in lst
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/{e['eval_id']}").json() == e
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/ghost-eval-28").status_code == 404
    # by-dataset does not shadow the generic detail getter NOR the M24
    # by-checkpoint route (a different grouping of the same listing)
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/by-checkpoint/{ck}").json() == []
    # M27 samples-by-checkpoint + M26 comparisons-by-checkpoint intact
    assert api_client.get(
        f"{MODELS}/{model_id}/samples/by-checkpoint/{ck}").json() == []
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/by-checkpoint/{ck}").json() == []
    # OpenAPI: 60 paths, the new path exactly once, GET-only,
    # evaluation tag, array of EvaluationRecord, before the generic
    # evaluation route
    spec = api_client.get("/openapi.json").json()
    path = ("/api/v1/models/{model_id}/evaluations/by-dataset/"
            "{dataset_id}")
    generic = "/api/v1/models/{model_id}/evaluations/{eval_id}"
    assert len(spec["paths"]) == 62
    assert list(spec["paths"]).count(path) == 1
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["evaluation"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/EvaluationRecord"}
    assert "EvaluationRecord" in spec["components"]["schemas"]
    assert list(spec["paths"]).index(path) < list(spec["paths"]) \
        .index(generic)


# =========================================================================== #
# M30: read-only per-tokenizer grouping of the evaluation history (API)
# =========================================================================== #

BY_TOKENIZER = "/api/v1/models/{mid}/evaluations/by-tokenizer/{tok}"


def _m30_second_tokenizer(api_client, tag: str, ds_id: str) -> str:
    """A second valid tokenizer, trained on the same dataset, with the
    dataset tokenized for it (so evaluations with it can run)."""
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": json.dumps(
                             {"name": f"api30-{tag}-tok",
                              "vocab_size": 320}),
                               "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    tok2 = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={"tokenizer_id": tok2}).status_code == 200
    return tok2


def test_m30_api_by_tokenizer_grouping_partition_determinism(api_client):
    ds_id, tok_id, model_id, _ = _prepare(api_client, "m30a")
    tok2 = _m30_second_tokenizer(api_client, "m30b", ds_id)
    # two evaluations with tok (distinct seeds), one with tok2
    e1 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, seed=3011)).json()
    e2 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok_id, seed=3012)).json()
    e3 = api_client.post(EVAL_RUN, json=_eval_cfg(
        model_id, ds_id, tok2, seed=3013)).json()
    assert e3["tokenizer_id"] == tok2
    url = BY_TOKENIZER.format(mid=model_id, tok=tok_id)
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly e1 + e2 under tok, in the deterministic M4 order,
    # verbatim equal to the run responses (existing representation)
    assert [x["eval_id"] for x in recs] == [e1["eval_id"], e2["eval_id"]]
    keyed = [(x["created_at"], x["eval_id"]) for x in recs]
    assert keyed == sorted(keyed)
    by_id = {x["eval_id"]: x for x in recs}
    assert by_id[e1["eval_id"]] == e1
    assert by_id[e2["eval_id"]] == e2
    # persisted tokenizer identity travels VERBATIM
    assert all(x["tokenizer_id"] == tok_id for x in recs)
    # parity with the M4 listing filtered locally by the persisted
    # tokenizer identity — no missing, no extra, no duplicates
    listing = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    assert recs == [x for x in listing if x["tokenizer_id"] == tok_id]
    # partition: tok2 holds exactly e3; no cross-tokenizer leakage
    got2 = api_client.get(BY_TOKENIZER.format(mid=model_id, tok=tok2))
    assert got2.status_code == 200
    assert [x["eval_id"] for x in got2.json()] == [e3["eval_id"]]
    # repeated GET returns identical raw bytes (3 repeats)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    raw3 = api_client.get(url).content
    assert raw1 == raw2 == raw3 and json.loads(raw1) == recs


def test_m30_api_404s_scoping_regressions_openapi(api_client):
    ds_id, tok_id, model_id, rep = _prepare(api_client, "m30e")
    ck = rep["checkpoints"][0]["checkpoint_id"]
    # unknown model / unknown tokenizer -> 404 (well-formed + malformed)
    assert api_client.get(BY_TOKENIZER.format(mid="ghost-model-30",
                                              tok=tok_id)).status_code == 404
    assert api_client.get(BY_TOKENIZER.format(mid=model_id,
                                              tok="ghost-tok-30")).status_code \
        == 404
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/by-tokenizer/tok%20id%2030!!") \
        .status_code == 404
    # tokenizers are GLOBAL: another real model + this valid tokenizer
    # is the natural model-scoped EMPTY case (200 + []), and no
    # evaluation id of model_id leaks into it
    ds2_id, tok2_id, model2_id, _ = _prepare(api_client, "m30iso")
    scoped = api_client.get(BY_TOKENIZER.format(mid=model2_id, tok=tok_id))
    assert scoped.status_code == 200 and scoped.json() == []
    # M4 regression: run/list/get + validation unchanged
    e = api_client.post(EVAL_RUN, json=_eval_cfg(model_id, ds_id, tok_id,
                                                 seed=3014)).json()
    lst = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    assert e in lst
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/{e['eval_id']}").json() == e
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/ghost-eval-30").status_code == 404
    # by-tokenizer does not shadow the M24 by-checkpoint route, the
    # M28 by-dataset route, nor the generic detail getter
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/by-checkpoint/{ck}").json() == []
    assert api_client.get(
        f"{MODELS}/{model_id}/evaluations/by-dataset/{ds_id}").json() \
        == [x for x in lst if x["dataset_id"] == ds_id]
    # M29 comparisons-by-dataset intact
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/by-dataset/{ds_id}").json() == []
    # existing tokenizer registry behavior unchanged (get + unknown 404)
    got_tok = api_client.get(f"/api/v1/tokenizers/{tok_id}")
    assert got_tok.status_code == 200 \
        and got_tok.json()["id"] == tok_id
    assert api_client.get("/api/v1/tokenizers/ghost-tok-30").status_code \
        == 404
    # OpenAPI: 62 paths, the new path exactly once, GET-only,
    # evaluation tag, array of EvaluationRecord, after the M28
    # by-dataset route and before the generic evaluation route
    spec = api_client.get("/openapi.json").json()
    path = ("/api/v1/models/{model_id}/evaluations/by-tokenizer/"
            "{tokenizer_id}")
    generic = "/api/v1/models/{model_id}/evaluations/{eval_id}"
    m28 = ("/api/v1/models/{model_id}/evaluations/by-dataset/"
           "{dataset_id}")
    assert len(spec["paths"]) == 62
    assert list(spec["paths"]).count(path) == 1
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["evaluation"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/EvaluationRecord"}
    assert "EvaluationRecord" in spec["components"]["schemas"]
    assert list(spec["paths"]).index(m28) < list(spec["paths"]).index(path) \
        < list(spec["paths"]).index(generic)
