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
    # + 1 (M26 comparisons by-checkpoint)
    # + 1 (M27 samples by-checkpoint)
    # + 1 (M28 evaluations by-dataset)
    # + 1 (M29 comparisons by-dataset)
    # + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind)
    # + 1 (M39 comparisons by-verdict)
    # + 1 (M40 samples by-strategy)
    # + 1 (M41 gate decisions by-decision)
    # + 1 (M42 workflows by-status)
    # + 1 (M43 gate decisions by-verdict)
    # + 1 (M44 comparisons by-state-kind)
    # + 1 (M45 gate decisions by-baseline-type)
    # + 1 (M46 checkpoints by-run)
    # + 1 (M47 evaluations by-truncated)
    # + 1 (M48 evaluations by-seed)
    # + 1 (M49 comparisons by-seed) = 81
    assert len(spec["paths"]) == 81


# =========================================================================== #
# M29: read-only per-dataset grouping of the comparison history (API)
# =========================================================================== #

BY_DATASET = "/api/v1/models/{mid}/comparisons/by-dataset/{ds}"


def _m29_second_dataset(api_client, tag: str, tok_id: str) -> str:
    """A second valid tokenized dataset for cross-dataset coverage."""
    up = api_client.post(UPLOAD,
                         files=[("files", (f"{tag}.txt",
                                           _domain_bytes(TAIL_B, 200),
                                           "text/plain"))],
                         data={"name": f"api29-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds2 = up.json()["dataset_id"]
    assert api_client.post(f"{DATASETS}/{ds2}/tokenize",
                           json={"tokenizer_id": tok_id}).status_code == 200
    return ds2


def test_m29_api_by_dataset_grouping_versions_determinism(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m29a", TAIL_A,
                                              epochs=30)
    ds2 = _m29_second_dataset(api_client, "m29b", tok_id)
    ck1, ck2 = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    # cross-checkpoint A/B on ds, same-checkpoint A=B on ds (BOTH sides
    # measure the requested dataset), and one comparison on ds2
    r1 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                              ck1, ck2, seed=2951)).json()
    r2 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                              ck1, ck1, seed=2952)).json()
    r3 = api_client.post(COMP_RUN, json=_comp(model_id, ds2, tok_id,
                                              ck1, ck2, seed=2953)).json()
    url = BY_DATASET.format(mid=model_id, ds=ds_id)
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly r1 + r2 under ds — the A=B record appears EXACTLY ONCE —
    # in the deterministic M5 order, verbatim equal to the run
    # responses (existing representation)
    assert [x["comparison_id"] for x in recs] == \
        [r1["comparison_id"], r2["comparison_id"]]
    keyed = [(x["created_at"], x["comparison_id"]) for x in recs]
    assert keyed == sorted(keyed)
    by_id = {x["comparison_id"]: x for x in recs}
    assert by_id[r1["comparison_id"]] == r1
    assert by_id[r2["comparison_id"]] == r2
    # persisted dataset identity + version travel VERBATIM
    assert all(x["dataset_id"] == ds_id
               and x["dataset_version"] == r1["dataset_version"]
               for x in recs)
    # parity with the M5 listing filtered locally by the persisted
    # dataset identity — no missing, no extra, no duplicates
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert recs == [x for x in listing if x["dataset_id"] == ds_id]
    # the second dataset holds exactly r3; no cross-dataset leakage
    got2 = api_client.get(BY_DATASET.format(mid=model_id, ds=ds2))
    assert got2.status_code == 200
    assert [x["comparison_id"] for x in got2.json()] == [r3["comparison_id"]]
    # repeated GET returns identical raw bytes (3 repeats)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    raw3 = api_client.get(url).content
    assert raw1 == raw2 == raw3 and json.loads(raw1) == recs
    # the generic comparison detail getter is not shadowed
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/{r1['comparison_id']}") \
        .json() == r1


def test_m29_api_404s_isolation_regressions_openapi(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m29e", TAIL_A,
                                              epochs=8)
    ck = ckpts[-1]["checkpoint_id"]
    rec = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                               ckpts[0]["checkpoint_id"],
                                               ck, seed=2961)).json()
    # unknown model / unknown dataset -> 404 (well-formed + malformed)
    assert api_client.get(BY_DATASET.format(mid="ghost-model-29",
                                            ds=ds_id)).status_code == 404
    assert api_client.get(BY_DATASET.format(mid=model_id,
                                            ds="ghost-ds-29")).status_code \
        == 404
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/by-dataset/ds%20id%2029!!") \
        .status_code == 404
    # model scoping: datasets are GLOBAL, so another real model + this
    # valid dataset is the natural model-scoped EMPTY case (200 + []),
    # and no comparison id leaks into it
    cfg = {"name": "api29-iso-b", "vocab_size": 640, "context_length": 64,
           "hidden_size": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
           "intermediate_size": 128}
    mid_b = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    scoped = api_client.get(BY_DATASET.format(mid=mid_b, ds=ds_id))
    assert scoped.status_code == 200 and scoped.json() == []
    # M5 regression: run/list/get + validation unchanged
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert rec["comparison_id"] in {x["comparison_id"] for x in listing}
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/{rec['comparison_id']}") \
        .json() == rec
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/ghost-comp-29").status_code == 404
    # M26 by-checkpoint unchanged (different grouping of the same
    # listing): the record appears under BOTH its checkpoints
    for cki in (ckpts[0]["checkpoint_id"], ck):
        got = api_client.get(
            f"{MODELS}/{model_id}/comparisons/by-checkpoint/{cki}")
        assert got.status_code == 200
        assert [x["comparison_id"] for x in got.json()] == \
            [rec["comparison_id"]]
    # M28 evaluations-by-dataset unchanged (parity with the filtered
    # M4 listing of this model)
    evs = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    evd = api_client.get(f"{MODELS}/{model_id}/evaluations/"
                         f"by-dataset/{ds_id}")
    assert evd.status_code == 200
    assert evd.json() == [x for x in evs if x["dataset_id"] == ds_id]
    # OpenAPI: 61 paths, the new path exactly once, GET-only,
    # comparison tag, array of ComparisonRecord, after the M26
    # by-checkpoint route and before the generic comparison route
    spec = api_client.get("/openapi.json").json()
    path = ("/api/v1/models/{model_id}/comparisons/by-dataset/"
            "{dataset_id}")
    generic = "/api/v1/models/{model_id}/comparisons/{comparison_id}"
    m26 = ("/api/v1/models/{model_id}/comparisons/by-checkpoint/"
           "{checkpoint_id}")
    assert len(spec["paths"]) == 81
    assert list(spec["paths"]).count(path) == 1
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["comparison"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/ComparisonRecord"}
    assert "ComparisonRecord" in spec["components"]["schemas"]
    assert list(spec["paths"]).index(m26) < list(spec["paths"]).index(path) \
        < list(spec["paths"]).index(generic)


# --------------------------------------------------------------------------- #
# M31: comparison history by tokenizer (read-only grouping)
# --------------------------------------------------------------------------- #

def _train_extra_tokenizer(api_client, tag: str, ds_id: str,
                           vocab: int) -> str:
    tr = api_client.post(TOKENIZERS + "/train",
                         data={"config": json.dumps(
                             {"name": f"api5-{tag}-tok", "vocab_size": vocab}),
                             "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    tok_id = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={"tokenizer_id": tok_id}).status_code == 200
    return tok_id


def test_m31_by_tokenizer_grouping_partition_determinism(api_client):
    ds_id, tok1, model_id, ckpts = _prepare(api_client, "m31a", TAIL_A,
                                            epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    tok2 = _train_extra_tokenizer(api_client, "m31a2", ds_id, 500)
    tok3 = _train_extra_tokenizer(api_client, "m31a3", ds_id, 450)

    url = f"{MODELS}/{model_id}/comparisons/by-tokenizer"
    c1 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok1,
                                              early, final)).json()
    c2 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok1,
                                              early, final,
                                              seed=7)).json()
    c3 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok2,
                                              early, final,
                                              seed=11)).json()
    # same-checkpoint A=B record still measured the shared probe of
    # tok2 -> must appear EXACTLY once in the tok2 group
    c4 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok2,
                                              final, final,
                                              seed=13)).json()

    g1 = api_client.get(f"{url}/{tok1}")
    assert g1.status_code == 200, g1.text
    body1 = g1.json()
    # authoritative-filter parity: exact subset of the M5 listing whose
    # persisted top-level tokenizer_id matches, in the same order
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert body1 == [r for r in listing if r["tokenizer_id"] == tok1]
    assert [r["comparison_id"] for r in body1] == [c1["comparison_id"],
                                                   c2["comparison_id"]]
    # verbatim: each element is byte-equal to its detail-getter payload
    for r in body1:
        got = api_client.get(
            f"{MODELS}/{model_id}/comparisons/{r['comparison_id']}")
        assert got.status_code == 200 and got.json() == r
    # deterministic: three repeats return identical raw bytes
    raws = {api_client.get(f"{url}/{tok1}").content for _ in range(3)}
    assert len(raws) == 1
    # partition: groups are disjoint, tok2 holds exactly its own runs
    g2 = api_client.get(f"{url}/{tok2}").json()
    assert [r["comparison_id"] for r in g2] == [c3["comparison_id"],
                                                c4["comparison_id"]]
    ids1 = {r["comparison_id"] for r in body1}
    ids2 = {r["comparison_id"] for r in g2}
    assert ids1.isdisjoint(ids2)
    assert ids1 | ids2 == {r["comparison_id"] for r in listing}
    # valid tokenizer with zero comparisons for the model -> [] (200)
    g3 = api_client.get(f"{url}/{tok3}")
    assert g3.status_code == 200 and g3.json() == []
    # no execution side effects: the filter itself added no records
    assert len(api_client.get(
        f"{MODELS}/{model_id}/comparisons").json()) == 4


def test_m31_by_tokenizer_404s_isolation_regressions_openapi(api_client):
    ds_id, tok1, model_id, ckpts = _prepare(api_client, "m31b", TAIL_A,
                                            epochs=24)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    _, _, other_model, _ = _prepare(api_client, "m31c", TAIL_B,
                                    epochs=24)
    url = f"{MODELS}/{model_id}/comparisons/by-tokenizer/"

    c = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok1,
                                             early, final)).json()

    # 404s: unknown model / unknown tokenizer (two ghost forms); the
    # unknown MODEL 404 wins even for an unknown tokenizer
    assert api_client.get(
        f"{MODELS}/ghost-model-31/comparisons/by-tokenizer/{tok1}"
    ).status_code == 404
    assert api_client.get(url + "ghost-tok-31").status_code == 404
    assert api_client.get(
        url + "m31--not-a-real-tokenizer-id").status_code == 404
    assert api_client.get(
        f"{MODELS}/ghost-model-31/comparisons/by-tokenizer/ghost-tok-31"
    ).status_code == 404

    # cross-model isolation: tokenizers are global, but the other
    # model's group is empty (scoping from the model's own listing)
    iso = api_client.get(f"{MODELS}/{other_model}/comparisons/by-tokenizer/"
                         f"{tok1}")
    assert iso.status_code == 200 and iso.json() == []

    # M5 listing/getter intact
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert [r["comparison_id"] for r in listing] == [c["comparison_id"]]
    got = api_client.get(f"{MODELS}/{model_id}/comparisons/"
                         f"{c['comparison_id']}")
    assert got.status_code == 200 and got.json() == c

    # M26 by-checkpoint / M29 by-dataset / M30 evaluations-by-tokenizer
    # regressions: same listing, other groupings, unconfused
    byck = api_client.get(f"{MODELS}/{model_id}/comparisons/by-checkpoint/"
                          f"{early}").json()
    assert [r["comparison_id"] for r in byck] == [c["comparison_id"]]
    byds = api_client.get(f"{MODELS}/{model_id}/comparisons/by-dataset/"
                          f"{ds_id}").json()
    assert [r["comparison_id"] for r in byds] == [c["comparison_id"]]
    evs = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    evt = api_client.get(f"{MODELS}/{model_id}/evaluations/by-tokenizer/"
                         f"{tok1}").json()
    assert evt == [e for e in evs if e["tokenizer_id"] == tok1]
    # generic detail getter still 404s ghost ids (no route capture)
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/ghost-comp-31").status_code == 404

    # OpenAPI: 63 paths, the new path exactly once, GET-only, tag
    # comparison, ComparisonRecord items; route order M29 by-dataset <
    # by-tokenizer < generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind)
    # + 1 (M39 comparisons by-verdict)
    # + 1 (M40 samples by-strategy)
    # + 1 (M41 gate decisions by-decision)
    # + 1 (M42 workflows by-status)
    # + 1 (M43 gate decisions by-verdict)
    # + 1 (M44 comparisons by-state-kind)
    # + 1 (M45 gate decisions by-baseline-type)
    # + 1 (M46 checkpoints by-run)
    # + 1 (M47 evaluations by-truncated)
    # + 1 (M48 evaluations by-seed)
    # + 1 (M49 comparisons by-seed) = 81
    assert len(spec["paths"]) == 81
    path = ("/api/v1/models/{model_id}/comparisons/by-tokenizer"
            "/{tokenizer_id}")
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["comparison"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/ComparisonRecord"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/comparisons/"
                      "by-dataset/{dataset_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/comparisons/{comparison_id}")


# --------------------------------------------------------------------------- #
# M37: comparison history by split (read-only grouping, enum contract)
# --------------------------------------------------------------------------- #

def test_m37_by_split_grouping_partition_determinism(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m37a", TAIL_A,
                                              epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]

    url = f"{MODELS}/{model_id}/comparisons/by-split"
    c_val = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                                 early, final)).json()
    c_trn = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                                 early, final,
                                                 split="train",
                                                 seed=5)).json()

    g = api_client.get(f"{url}/validation")
    assert g.status_code == 200, g.text
    body = g.json()
    # authoritative-filter parity: exact subset of the M5 listing whose
    # persisted top-level split matches, in the same order
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert body == [r for r in listing if r["split"] == "validation"]
    assert [r["comparison_id"] for r in body] == [c_val["comparison_id"]]
    # verbatim: each element is byte-equal to its detail-getter payload
    for r in body:
        got = api_client.get(
            f"{MODELS}/{model_id}/comparisons/{r['comparison_id']}")
        assert got.status_code == 200 and got.json() == r
    # deterministic: three repeats return identical raw bytes
    raws = {api_client.get(f"{url}/validation").content for _ in range(3)}
    assert len(raws) == 1
    # partition: disjoint groups over ALL enum splits whose union is
    # the full listing; `test` is the natural valid-enum empty case
    ids_by = {sp: {r["comparison_id"] for r in
                   api_client.get(f"{url}/{sp}").json()}
              for sp in ("train", "validation", "test")}
    assert ids_by["train"] == {c_trn["comparison_id"]}
    assert ids_by["validation"] == {c_val["comparison_id"]}
    assert ids_by["test"] == set()
    assert ids_by["train"].isdisjoint(ids_by["validation"])
    assert ids_by["train"] | ids_by["validation"] == \
        {r["comparison_id"] for r in listing}
    for sp in ("train", "test"):
        got = api_client.get(f"{url}/{sp}")
        assert got.status_code == 200
    assert api_client.get(f"{url}/test").json() == []
    # no execution side effects: the filter itself added no records
    assert len(api_client.get(
        f"{MODELS}/{model_id}/comparisons").json()) == 2


def test_m37_by_split_404_422s_isolation_regressions_openapi(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m37b", TAIL_A,
                                              epochs=24)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    _, _, other_model, _ = _prepare(api_client, "m37c", TAIL_B,
                                    epochs=24)
    url = f"{MODELS}/{model_id}/comparisons/by-split/"

    c = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                             early, final)).json()

    # 404: unknown model with a VALID split (exactly like the sibling
    # groupings)
    assert api_client.get(
        f"{MODELS}/ghost-model-37/comparisons/by-split/validation"
    ).status_code == 404
    # 422: unsupported split values are rejected by the schema enum at
    # the API boundary — before the handler, so the 422 wins even for
    # an UNKNOWN model (never a registry-style 404, never [])
    for bad in ("VALIDATION", "valid%20ation", "3", "banana-split"):
        got = api_client.get(url + bad)
        assert got.status_code == 422, (bad, got.status_code)
    assert api_client.get(
        f"{MODELS}/ghost-model-37/comparisons/by-split/banana-split"
    ).status_code == 422

    # cross-model isolation: the other model's group is empty under
    # EVERY valid split (scoping from the model's own listing)
    for sp in ("train", "validation", "test"):
        iso = api_client.get(
            f"{MODELS}/{other_model}/comparisons/by-split/{sp}")
        assert iso.status_code == 200 and iso.json() == []

    # M5 listing/getter intact; generic detail getter still 404s ghost
    # ids (no route capture by the new literal segment)
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert [r["comparison_id"] for r in listing] == [c["comparison_id"]]
    got = api_client.get(
        f"{MODELS}/{model_id}/comparisons/{c['comparison_id']}")
    assert got.status_code == 200 and got.json() == c
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/ghost-comp-37").status_code == 404

    # M26 by-checkpoint / M29 by-dataset / M31 by-tokenizer
    # regressions: same listing, other groupings, unconfused
    for ep, arg in (("by-checkpoint", early), ("by-dataset", ds_id),
                    ("by-tokenizer", tok_id)):
        g = api_client.get(f"{MODELS}/{model_id}/comparisons/{ep}/{arg}")
        assert g.status_code == 200
        assert [r["comparison_id"] for r in g.json()] == \
            [c["comparison_id"]]
    # M36 evaluations-by-split regression: the evaluation twin keeps
    # its own contract on the same model
    evs = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    evs_by = api_client.get(
        f"{MODELS}/{model_id}/evaluations/by-split/validation")
    assert evs_by.status_code == 200
    assert evs_by.json() == [e for e in evs if e["split"] == "validation"]

    # OpenAPI: 77 paths, the new path exactly once, GET-only, tag
    # comparison, ComparisonRecord items, split $ref EvaluationSplit;
    # route order M31 by-tokenizer < by-split < generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind)
    # + 1 (M39 comparisons by-verdict)
    # + 1 (M40 samples by-strategy)
    # + 1 (M41 gate decisions by-decision)
    # + 1 (M42 workflows by-status)
    # + 1 (M43 gate decisions by-verdict)
    # + 1 (M44 comparisons by-state-kind)
    # + 1 (M45 gate decisions by-baseline-type)
    # + 1 (M46 checkpoints by-run)
    # + 1 (M47 evaluations by-truncated)
    # + 1 (M48 evaluations by-seed)
    # + 1 (M49 comparisons by-seed) = 81
    assert len(spec["paths"]) == 81
    path = ("/api/v1/models/{model_id}/comparisons/by-split/{split}")
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["comparison"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/ComparisonRecord"}
    split_param = [p for p in item["get"]["parameters"]
                   if p["name"] == "split"][0]
    assert split_param["schema"] == {
        "$ref": "#/components/schemas/EvaluationSplit"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/comparisons/"
                      "by-tokenizer/{tokenizer_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/comparisons/{comparison_id}")


# --------------------------------------------------------------------------- #
# M39: comparison history by verdict (read-only grouping, enum contract)
# --------------------------------------------------------------------------- #

def test_m39_by_verdict_grouping_partition_determinism(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m39a", TAIL_A,
                                              epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]

    url = f"{MODELS}/{model_id}/comparisons/by-verdict"
    # A=B -> unchanged by construction; the two directions of the
    # same pair under a tiny tolerance -> opposite non-unchanged
    # verdicts (improved/regressed)
    c_un = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                                final, final)).json()
    c_ab = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                                early, final,
                                                tolerance=1e-9)).json()
    c_ba = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                                final, early,
                                                tolerance=1e-9)).json()
    assert c_un["verdict"] == "unchanged"
    assert {c_ab["verdict"], c_ba["verdict"]} == {"improved",
                                                  "regressed"}

    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert len(listing) == 3
    # authoritative-filter parity for ALL THREE verdict groups; each
    # is a singleton holding exactly its own record; deterministic
    # (created_at, comparison_id) order
    ids_by = {}
    for v in ("improved", "regressed", "unchanged"):
        got = api_client.get(f"{url}/{v}")
        assert got.status_code == 200, got.text
        body = got.json()
        assert body == [r for r in listing if r["verdict"] == v]
        assert len(body) == 1
        ids_by[v] = {r["comparison_id"] for r in body}
        keyed = [(r["created_at"], r["comparison_id"]) for r in body]
        assert keyed == sorted(keyed)
        # verbatim: the element equals its detail-getter payload
        one = api_client.get(
            f"{MODELS}/{model_id}/comparisons/"
            f"{body[0]['comparison_id']}")
        assert one.status_code == 200 and one.json() == body[0]
    assert ids_by["unchanged"] == {c_un["comparison_id"]}
    assert ids_by[c_ab["verdict"]] == {c_ab["comparison_id"]}
    assert ids_by[c_ba["verdict"]] == {c_ba["comparison_id"]}
    # partition: disjoint groups whose union is the full listing
    all_ids = {r["comparison_id"] for r in listing}
    assert set().union(*ids_by.values()) == all_ids
    # deterministic: three repeats return identical raw bytes
    for v in ("improved", "regressed", "unchanged"):
        raws = {api_client.get(f"{url}/{v}").content
                for _ in range(3)}
        assert len(raws) == 1
    # no execution side effects: the filter itself added no records
    assert len(api_client.get(
        f"{MODELS}/{model_id}/comparisons").json()) == 3


def test_m39_by_verdict_404_422s_isolation_regressions_openapi(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m39b", TAIL_A,
                                              epochs=24)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    _, _, other_model, _ = _prepare(api_client, "m39c", TAIL_B,
                                    epochs=24)
    url = f"{MODELS}/{model_id}/comparisons/by-verdict/"

    c = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                             early, final)).json()

    # 404: unknown model with a VALID verdict (exactly like the
    # sibling groupings)
    assert api_client.get(
        f"{MODELS}/ghost-model-39/comparisons/by-verdict/improved"
    ).status_code == 404
    # 422: unsupported verdict values are rejected by the schema enum
    # at the API boundary — before the handler, so the 422 wins even
    # for an UNKNOWN model (never a registry-style 404, never [])
    for bad in ("IMPROVED", "im%20proved", "1", "better"):
        got = api_client.get(url + bad)
        assert got.status_code == 422, (bad, got.status_code)
    assert api_client.get(
        f"{MODELS}/ghost-model-39/comparisons/by-verdict/better"
    ).status_code == 422

    # cross-model isolation: the other model's group is empty under
    # EVERY valid verdict (scoping from the model's own listing)
    for v in ("improved", "regressed", "unchanged"):
        iso = api_client.get(
            f"{MODELS}/{other_model}/comparisons/by-verdict/{v}")
        assert iso.status_code == 200 and iso.json() == []

    # M5 listing/getter intact; generic detail getter still 404s ghost
    # ids (no route capture by the new literal segment)
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert [r["comparison_id"] for r in listing] == [c["comparison_id"]]
    got = api_client.get(
        f"{MODELS}/{model_id}/comparisons/{c['comparison_id']}")
    assert got.status_code == 200 and got.json() == c
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/ghost-comp-39").status_code == 404

    # M26 by-checkpoint / M29 by-dataset / M31 by-tokenizer / M37
    # by-split regressions: same listing, other groupings, unconfused
    for ep, arg in (("by-checkpoint", early), ("by-dataset", ds_id),
                    ("by-tokenizer", tok_id),
                    ("by-split", "validation")):
        g = api_client.get(f"{MODELS}/{model_id}/comparisons/{ep}/{arg}")
        assert g.status_code == 200
        assert [r["comparison_id"] for r in g.json()] == \
            [c["comparison_id"]]
    # M38 evaluations-by-state-kind regression: both groups keep
    # listing parity on the same model
    evs = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    for kind in ("current", "checkpoint"):
        g = api_client.get(
            f"{MODELS}/{model_id}/evaluations/by-state-kind/{kind}")
        assert g.status_code == 200
        assert g.json() == [e for e in evs
                            if e["state_kind"] == kind]

    # OpenAPI: 77 paths, the new path exactly once, GET-only, tag
    # comparison, ComparisonRecord items, verdict $ref
    # ComparisonVerdict; route order M37 by-split < by-verdict <
    # generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind)
    # + 1 (M39 comparisons by-verdict)
    # + 1 (M40 samples by-strategy)
    # + 1 (M41 gate decisions by-decision)
    # + 1 (M42 workflows by-status)
    # + 1 (M43 gate decisions by-verdict)
    # + 1 (M44 comparisons by-state-kind)
    # + 1 (M45 gate decisions by-baseline-type)
    # + 1 (M46 checkpoints by-run)
    # + 1 (M47 evaluations by-truncated)
    # + 1 (M48 evaluations by-seed)
    # + 1 (M49 comparisons by-seed) = 81
    assert len(spec["paths"]) == 81
    path = ("/api/v1/models/{model_id}/comparisons/by-verdict/"
            "{verdict}")
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["comparison"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/ComparisonRecord"}
    verdict_param = [p for p in item["get"]["parameters"]
                     if p["name"] == "verdict"][0]
    assert verdict_param["schema"] == {
        "$ref": "#/components/schemas/ComparisonVerdict"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/comparisons/"
                      "by-split/{split}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/comparisons/{comparison_id}")


# --------------------------------------------------------------------------- #
# M44: comparison history by state kind (either-side, enum contract)
# --------------------------------------------------------------------------- #

BY_STATE_KIND = ("/api/v1/models/{mid}/comparisons/by-state-kind/"
                 "{kind}")


def test_m44_by_state_kind_grouping_either_side_determinism(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m44a", TAIL_A,
                                              epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    # the M26 shapes: cross-checkpoint A/B, same-checkpoint A=B,
    # current-vs-checkpoint and current-vs-current comparisons
    r1 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                              early, final, seed=2661)).json()
    r2 = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                              early, early, seed=2662)).json()
    r3 = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, final, final, seed=2663,
        state_a={"state_kind": "current"})).json()
    r4 = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, final, final, seed=2664,
        state_a={"state_kind": "current"},
        state_b={"state_kind": "current"})).json()

    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert len(listing) == 4
    url = BY_STATE_KIND.format(mid=model_id, kind="")
    for kind in ("checkpoint", "current"):
        got = api_client.get(url + kind)
        assert got.status_code == 200, got.text
        recs = got.json()
        # M26 either-side parity with the authoritative M5 listing
        assert recs == [x for x in listing
                        if any(s["state_kind"] == kind
                               for s in (x["state_a"], x["state_b"]))]
        keyed = [(x["created_at"], x["comparison_id"]) for x in recs]
        assert keyed == sorted(keyed)
        # a both-sides match appears EXACTLY ONCE
        ids = [x["comparison_id"] for x in recs]
        assert len(ids) == len(set(ids))
        # verbatim: each element equals its detail-getter payload
        for x in recs:
            one = api_client.get(
                f"{MODELS}/{model_id}/comparisons/"
                f"{x['comparison_id']}")
            assert one.status_code == 200 and one.json() == x
    # either-side pins: r3 (current A / checkpoint B) in BOTH groups;
    # r1/r2 checkpoint-only; r4 current-only
    ids_ck = {x["comparison_id"] for x in api_client.get(
        url + "checkpoint").json()}
    ids_cu = {x["comparison_id"] for x in api_client.get(
        url + "current").json()}
    assert r3["comparison_id"] in ids_ck and r3["comparison_id"] in ids_cu
    assert r1["comparison_id"] in ids_ck and r1["comparison_id"] not in ids_cu
    assert r2["comparison_id"] in ids_ck
    assert r4["comparison_id"] in ids_cu
    assert r4["comparison_id"] not in ids_ck
    # the two groups COVER the full listing (overlap by design)
    assert ids_ck | ids_cu == {x["comparison_id"] for x in listing}
    # deterministic: three repeats per kind return identical bytes
    for kind in ("checkpoint", "current"):
        raws = {api_client.get(url + kind).content for _ in range(3)}
        assert len(raws) == 1
    # no execution side effects: the filter itself added no records
    assert len(api_client.get(
        f"{MODELS}/{model_id}/comparisons").json()) == 4


def test_m44_by_state_kind_404_422s_isolation_regressions_openapi(
        api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m44b", TAIL_A,
                                              epochs=24)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    _, _, other_model, _ = _prepare(api_client, "m44c", TAIL_B,
                                    epochs=24)
    url = BY_STATE_KIND.format(mid=model_id, kind="")

    c = api_client.post(COMP_RUN, json=_comp(model_id, ds_id, tok_id,
                                             early, final)).json()

    # 404: unknown model with a VALID state kind (exactly like the
    # sibling groupings)
    assert api_client.get(
        f"{MODELS}/ghost-model-44/comparisons/by-state-kind/checkpoint"
    ).status_code == 404
    # 422: unsupported state-kind values are rejected by the schema
    # enum at the API boundary — before the handler, so the 422 wins
    # even for an UNKNOWN model (never a registry-style 404, never [])
    for bad in ("CHECKPOINT", "checkp%20oint", "1", "weights"):
        got = api_client.get(url + bad)
        assert got.status_code == 422, (bad, got.status_code)
    assert api_client.get(
        f"{MODELS}/ghost-model-44/comparisons/by-state-kind/weights"
    ).status_code == 422

    # cross-model isolation: the other model has NO comparisons, so
    # both groups are the natural valid empty
    for kind in ("checkpoint", "current"):
        iso = api_client.get(BY_STATE_KIND.format(mid=other_model,
                                                  kind=kind))
        assert iso.status_code == 200 and iso.json() == []

    # M5 listing/getter intact; generic detail getter still 404s ghost
    # ids (no route capture by the new literal segment)
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert [x["comparison_id"] for x in listing] == [c["comparison_id"]]
    got = api_client.get(
        f"{MODELS}/{model_id}/comparisons/{c['comparison_id']}")
    assert got.status_code == 200 and got.json() == c
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/ghost-comp-44").status_code == 404

    # M26 by-checkpoint / M29 by-dataset / M31 by-tokenizer / M37
    # by-split / M39 by-verdict regressions: same listing, other
    # groupings, unconfused
    for ep, arg in (("by-checkpoint", early), ("by-dataset", ds_id),
                    ("by-tokenizer", tok_id),
                    ("by-split", "validation")):
        g = api_client.get(f"{MODELS}/{model_id}/comparisons/{ep}/{arg}")
        assert g.status_code == 200
        assert [x["comparison_id"] for x in g.json()] == \
            [c["comparison_id"]]
    for v in ("improved", "regressed", "unchanged"):
        g = api_client.get(
            f"{MODELS}/{model_id}/comparisons/by-verdict/{v}")
        assert g.status_code == 200
        assert g.json() == [x for x in listing
                            if x["verdict"] == v]
    # M38 evaluations-by-state-kind regression: both groups keep
    # listing parity on the same model (the enum-contract twin)
    evs = api_client.get(f"{MODELS}/{model_id}/evaluations").json()
    for kind in ("current", "checkpoint"):
        g = api_client.get(
            f"{MODELS}/{model_id}/evaluations/by-state-kind/{kind}")
        assert g.status_code == 200
        assert g.json() == [e for e in evs
                            if e["state_kind"] == kind]

    # OpenAPI: 77 paths, the new path exactly once, GET-only, tag
    # comparison, ComparisonRecord items, state_kind $ref
    # EvalStateKind; route order M39 by-verdict < by-state-kind <
    # generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer)
    # + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind)
    # + 1 (M39 comparisons by-verdict)
    # + 1 (M40 samples by-strategy)
    # + 1 (M41 gate decisions by-decision)
    # + 1 (M42 workflows by-status)
    # + 1 (M43 gate decisions by-verdict)
    # + 1 (M44 comparisons by-state-kind)
    # + 1 (M45 gate decisions by-baseline-type)
    # + 1 (M46 checkpoints by-run)
    # + 1 (M47 evaluations by-truncated)
    # + 1 (M48 evaluations by-seed)
    # + 1 (M49 comparisons by-seed) = 81
    assert len(spec["paths"]) == 81
    path = ("/api/v1/models/{model_id}/comparisons/by-state-kind/"
            "{state_kind}")
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["comparison"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/ComparisonRecord"}
    kind_param = [p for p in item["get"]["parameters"]
                  if p["name"] == "state_kind"][0]
    assert kind_param["schema"] == {
        "$ref": "#/components/schemas/EvalStateKind"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/comparisons/by-verdict/"
                      "{verdict}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/comparisons/{comparison_id}")


# --------------------------------------------------------------------------- #
# M49: comparison history by seed (read-only integer value grouping)
# --------------------------------------------------------------------------- #

BY_SEED = "/api/v1/models/{mid}/comparisons/by-seed/{seed}"


def test_m49_by_seed_grouping_partition_determinism(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m49a", TAIL_A,
                                              epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    # THREE seeds with different cardinalities, all through the REAL
    # engine: 4900 x2 (an A=B and a cross pair), 4901 x1
    a1 = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, final, final, seed=4900)).json()
    a2 = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, early, final, tolerance=1e-9,
        seed=4900)).json()
    b1 = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, early, final, seed=4901)).json()
    assert {r["seed"] for r in (a1, a2, b1)} == {4900, 4901}

    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    keyed = [(r["created_at"], r["comparison_id"]) for r in listing]
    assert keyed == sorted(keyed)
    groups = {}
    for seed, expected_ids in ((4900, {a1["comparison_id"],
                                       a2["comparison_id"]}),
                               (4901, {b1["comparison_id"]})):
        got = api_client.get(BY_SEED.format(mid=model_id, seed=seed))
        assert got.status_code == 200, got.text
        groups[seed] = got.json()
        # authoritative-filter parity: exact subset of the M5
        # listing whose persisted seed matches, in the same order
        assert groups[seed] == [x for x in listing if x["seed"] == seed]
        assert {x["comparison_id"] for x in groups[seed]} == expected_ids
        assert all(x["seed"] == seed for x in groups[seed])
        gk = [(x["created_at"], x["comparison_id"]) for x in groups[seed]]
        assert gk == sorted(gk)
    # known seed groups remain disjoint; union is the full listing
    ids0 = {x["comparison_id"] for x in groups[4900]}
    ids1 = {x["comparison_id"] for x in groups[4901]}
    assert ids0.isdisjoint(ids1)
    assert ids0 | ids1 == {x["comparison_id"] for x in listing}
    # verbatim: each element equals its detail-getter payload
    for x in groups[4900] + groups[4901]:
        one = api_client.get(
            f"{MODELS}/{model_id}/comparisons/{x['comparison_id']}")
        assert one.status_code == 200 and one.json() == x
    # determinism: byte-identical repeats
    again = api_client.get(BY_SEED.format(mid=model_id, seed=4900))
    assert again.content == api_client.get(
        BY_SEED.format(mid=model_id, seed=4900)).content


def test_m49_by_seed_404_422s_isolation_regressions_openapi(api_client):
    ds_id, tok_id, model_id, ckpts = _prepare(api_client, "m49b", TAIL_A,
                                              epochs=30)
    o_ds, o_tok, other_model, o_ckpts = _prepare(
        api_client, "m49c", TAIL_B, epochs=30)
    early, final = ckpts[2]["checkpoint_id"], ckpts[-1]["checkpoint_id"]
    o_early = o_ckpts[-1]["checkpoint_id"]
    mine = api_client.post(COMP_RUN, json=_comp(
        model_id, ds_id, tok_id, early, final, seed=4921)).json()
    other = api_client.post(COMP_RUN, json=_comp(
        other_model, o_ds, o_tok, o_early, o_early, seed=4922)).json()

    # unmatched valid integer -> 200 []
    empty = api_client.get(BY_SEED.format(mid=model_id, seed=987654))
    assert empty.status_code == 200 and empty.json() == []
    # unknown model + valid integer -> 404
    assert api_client.get(BY_SEED.format(
        mid="ghost", seed=4921)).status_code == 404
    # non-integer spellings -> 422 (schema-level validation at the
    # API boundary, pre-handler — integers are never silently
    # reinterpreted), on the known model AND on an unknown model
    for bad in ("abc", "1.5", "12x"):
        assert api_client.get(
            f"{MODELS}/{model_id}/comparisons/by-seed/{bad}"
        ).status_code == 422
        assert api_client.get(
            f"{MODELS}/ghost/comparisons/by-seed/{bad}"
        ).status_code == 422
    # cross-model isolation: the other model's comparison never leaks
    got_o = api_client.get(BY_SEED.format(mid=other_model, seed=4922))
    assert got_o.status_code == 200 and len(got_o.json()) == 1
    assert other["comparison_id"] not in {
        x["comparison_id"] for x in api_client.get(
            BY_SEED.format(mid=model_id, seed=4921)).json()}
    assert mine["comparison_id"] not in {
        x["comparison_id"] for x in got_o.json()}

    # M5 regressions: listing + detail unchanged; the M26/M39/M44
    # groupings keep listing parity
    listing = api_client.get(f"{MODELS}/{model_id}/comparisons").json()
    assert mine["comparison_id"] in {x["comparison_id"] for x in listing}
    one = api_client.get(
        f"{MODELS}/{model_id}/comparisons/{mine['comparison_id']}")
    assert one.status_code == 200 and one.json() == mine
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/by-checkpoint/{early}"
    ).json() == [x for x in listing
                 if early in (x["state_a"]["checkpoint_id"],
                              x["state_b"]["checkpoint_id"])]
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/by-verdict/{mine['verdict']}"
    ).json() == [x for x in listing if x["verdict"] == mine["verdict"]]
    assert api_client.get(
        f"{MODELS}/{model_id}/comparisons/by-state-kind/checkpoint"
    ).json() == [x for x in listing
                  if "checkpoint" in (x["state_a"]["state_kind"],
                                      x["state_b"]["state_kind"])]

    # OpenAPI: 81 paths, the new path exactly once, GET-only, tag
    # comparison, ComparisonRecord items, integer parameter; route
    # order by-state-kind < by-seed < generic comparison detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30) + 1 (M31) + 1 (M32) + 1 (M33)
    # + 1 (M34) + 1 (M35) + 1 (M36) + 1 (M37) + 1 (M38) + 1 (M39)
    # + 1 (M40) + 1 (M41) + 1 (M42) + 1 (M43) + 1 (M44)
    # + 1 (M45) + 1 (M46) + 1 (M47) + 1 (M48)
    # + 1 (M49 comparisons by-seed) = 81
    assert len(spec["paths"]) == 81
    path = "/api/v1/models/{model_id}/comparisons/by-seed/{seed}"
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["comparison"]
    params = {p["name"]: p for p in item["get"]["parameters"]}
    assert set(params) == {"model_id", "seed"}
    assert params["seed"]["schema"]["type"] == "integer"
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/ComparisonRecord"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/comparisons/"
                      "by-state-kind/{state_kind}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/comparisons/{comparison_id}")
    # the generic detail route and the M48 route are still present
    assert keys.count("/api/v1/models/{model_id}/comparisons/"
                      "{comparison_id}") == 1
    assert keys.count("/api/v1/models/{model_id}/evaluations/"
                      "by-seed/{seed}") == 1
