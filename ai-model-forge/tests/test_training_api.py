"""Milestone 3 tests: training REST API (run / checkpoints / rollback)."""
from __future__ import annotations

import io
import json
import random
import zlib

import torch

from app.model_builder import content_hash

UPLOAD = "/api/v1/datasets/upload"
DATASETS = "/api/v1/datasets"
TOKENIZERS = "/api/v1/tokenizers"
MODELS = "/api/v1/models"
TRAIN_RUN = "/api/v1/training/run"

_WORDS = ("river mountain cloud forest desert ocean valley island meadow canyon "
          "table chair lamp desk shelf couch rug clock mirror vase").split()


def _files(name: str, content: bytes):
    return [("files", (name, content, "text/plain"))]


def _corpus(n: int, tag: str) -> bytes:
    rng = random.Random(zlib.crc32(tag.encode()))
    lines = []
    for i in range(n):
        k = rng.randint(10, 22)
        lines.append(" ".join(rng.choices(_WORDS, k=k)) + f" number {i}")
    return ("\n\n".join(lines) + "\n").encode("utf-8")


def _prepare(api_client, tag: str, n: int = 170) -> tuple[str, str, str]:
    """Upload a dataset, train a tokenizer on it, tokenize, create a model."""
    up = api_client.post(UPLOAD, files=_files(f"{tag}.txt", _corpus(n, tag)),
                         data={"name": f"api3-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds_id = up.json()["dataset_id"]

    tr = api_client.post(TOKENIZERS + "/train",
                         data={"config": json.dumps({"name": f"api3-{tag}-tok",
                                                     "vocab_size": 320}),
                               "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    tok_id = tr.json()["tokenizer"]["id"]
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={"tokenizer_id": tok_id}).status_code == 200

    model_cfg = {"name": f"api3-{tag}-model", "vocab_size": 640,
                 "context_length": 64, "hidden_size": 64, "n_layers": 2,
                 "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128}
    m = api_client.post(MODELS, json={"config": model_cfg,
                                      "description": "milestone-3 api test"})
    assert m.status_code == 201, m.text
    return ds_id, tok_id, m.json()["model"]["id"]


def _run_cfg(model_id: str, ds_id: str, tok_id: str, **overrides) -> dict:
    cfg = {
        "name": "api3-run", "method": "continued_pretraining",
        "model_id": model_id, "dataset_id": ds_id, "tokenizer_id": tok_id,
        "learning_rate": 3e-3, "batch_size": 8, "steps": 8, "max_seq_len": 32,
        "warmup_steps": 0, "weight_decay": 0.01, "adam_beta1": 0.9,
        "adam_beta2": 0.999, "lr_schedule": "cosine",
        "eval_every_steps": 4, "keep_best": True, "seed": 3,
    }
    cfg.update(overrides)
    return cfg


def _weights_sha(client, model_id: str) -> str:
    dl = client.get(f"{MODELS}/{model_id}/weights")
    assert dl.status_code == 200
    return content_hash(torch.load(io.BytesIO(dl.content), map_location="cpu",
                                   weights_only=True))


# --------------------------------------------------------------------------- #
# Success path
# --------------------------------------------------------------------------- #

def test_training_run_end_to_end(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "e2e")
    before = _weights_sha(api_client, model_id)

    resp = api_client.post(TRAIN_RUN, json=_run_cfg(model_id, ds_id, tok_id))
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["model_id"] == model_id
    assert report["optimizer_steps"] == 8
    assert report["actual_batch_size"] == 8
    assert len(report["checkpoints"]) == 2  # eval_every 4 + final
    assert report["checkpoints"][-1]["step"] == 8
    assert report["accepted"] is True
    assert report["final_train_loss"] < report["initial_train_loss"]

    # checkpoints listable + readable
    ckpts = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert [c["checkpoint_id"] for c in ckpts] == \
        [c["checkpoint_id"] for c in report["checkpoints"]]
    one = api_client.get(f"{MODELS}/{model_id}/checkpoints/{ckpts[0]['checkpoint_id']}")
    assert one.status_code == 200
    assert one.json()["weights_sha256"] == ckpts[0]["weights_sha256"]

    # model manifest updated (lineage) and weights changed
    got = api_client.get(f"{MODELS}/{model_id}").json()
    assert got["latest_checkpoint"] == ckpts[-1]["checkpoint_id"]
    assert got["best_checkpoint"] is not None
    assert len(got["training_provenance"]) == 1
    assert _weights_sha(api_client, model_id) != before


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #

def test_training_run_rejections(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "reject")
    base = _run_cfg(model_id, ds_id, tok_id)

    resp = api_client.post(TRAIN_RUN, json={**base, "model_id": "ghost"})
    assert resp.status_code == 404 and "not found" in resp.json()["detail"]
    resp = api_client.post(TRAIN_RUN, json={**base, "dataset_id": "ghost"})
    assert resp.status_code == 404 and "not found" in resp.json()["detail"]
    resp = api_client.post(TRAIN_RUN, json={**base, "tokenizer_id": "ghost"})
    assert resp.status_code == 404
    resp = api_client.post(TRAIN_RUN, json={**base, "dataset_version": 99})
    assert resp.status_code == 404 and "version 99" in resp.json()["detail"]

    # schema-level invalid configs -> 422
    resp = api_client.post(TRAIN_RUN, json={**base, "epochs": 1, "steps": 1})
    assert resp.status_code == 422
    resp = api_client.post(TRAIN_RUN, json={**base, "learning_rate": 5.0})
    assert resp.status_code == 422
    resp = api_client.post(TRAIN_RUN, json={**base, "max_seq_len": 128})
    assert resp.status_code == 422 and "context_length" in resp.json()["detail"]

    # nothing was written by the refused runs
    got = api_client.get(f"{MODELS}/{model_id}").json()
    assert got["training_provenance"] == []


def test_missing_tokenization_refused_404(api_client):
    up = api_client.post(UPLOAD, files=_files("raw.txt", _corpus(120, "untok")),
                         data={"name": "api3-untok-ds"})
    ds_id = up.json()["dataset_id"]
    _, tok_id, model_id = _prepare(api_client, "untokmod")  # separate model+tokenizer
    before = _weights_sha(api_client, model_id)

    cfg = _run_cfg(model_id, ds_id, tok_id)
    resp = api_client.post(TRAIN_RUN, json=cfg)
    assert resp.status_code == 404
    assert "tokenized artifact" in resp.json()["detail"]

    # weights untouched, no provenance entry
    assert _weights_sha(api_client, model_id) == before
    got = api_client.get(f"{MODELS}/{model_id}").json()
    assert got["training_provenance"] == []


def test_checkpoint_endpoints_404(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "ck404")
    assert api_client.get(f"{MODELS}/ghost/checkpoints").status_code == 404
    assert api_client.get(f"{MODELS}/{model_id}/checkpoints").status_code == 200

    api_client.post(TRAIN_RUN, json=_run_cfg(model_id, ds_id, tok_id))
    assert api_client.get(f"{MODELS}/{model_id}/checkpoints/ghost").status_code == 404
    assert api_client.post(f"{MODELS}/{model_id}/rollback",
                           json={"checkpoint_id": "ghost"}).status_code == 404
    assert api_client.post(f"{MODELS}/ghost/rollback",
                           json={"checkpoint_id": "ghost"}).status_code == 404


# --------------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------------- #

def test_rollback_and_corrupt_checkpoint_via_api(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "rollback")
    cfg = _run_cfg(model_id, ds_id, tok_id, steps=14, eval_every_steps=7)
    assert api_client.post(TRAIN_RUN, json=cfg).status_code == 200
    ckpts = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(ckpts) == 2
    target = ckpts[0]

    # rollback to the earlier checkpoint -> its weights become current
    resp = api_client.post(f"{MODELS}/{model_id}/rollback",
                           json={"checkpoint_id": target["checkpoint_id"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["latest_checkpoint"] == target["checkpoint_id"]
    assert _weights_sha(api_client, model_id) == target["weights_sha256"]

    # corrupt the second checkpoint's weights on disk -> integrity refusal (409)
    storage_root = api_client.get("/api/v1/project").json()["storage_root"]
    ckpt_dir = f"{storage_root}/models/{model_id}/checkpoints/{ckpts[1]['checkpoint_id']}"
    wpath = f"{ckpt_dir}/weights.pt"
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    tampered = {k: v.clone() for k, v in state.items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    resp = api_client.post(f"{MODELS}/{model_id}/rollback",
                           json={"checkpoint_id": ckpts[1]["checkpoint_id"]})
    assert resp.status_code == 409
    assert "integrity" in resp.json()["detail"]
    # current weights are untouched by the refused rollback
    assert _weights_sha(api_client, model_id) == target["weights_sha256"]
    # the manifest was not repointed either
    got = api_client.get(f"{MODELS}/{model_id}").json()
    assert got["latest_checkpoint"] == target["checkpoint_id"]


def test_config_validation_422_via_api(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "val422")
    base = _run_cfg(model_id, ds_id, tok_id)
    bad = {**base, "steps": None, "epochs": None}
    resp = api_client.post(TRAIN_RUN, json=bad)
    assert resp.status_code == 422
    resp = api_client.post(TRAIN_RUN, json={**base, "unknown_extra": 1})
    assert resp.status_code == 422
