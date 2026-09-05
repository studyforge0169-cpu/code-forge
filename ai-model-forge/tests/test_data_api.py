"""Milestone 2 tests: dataset + tokenizer REST API."""
from __future__ import annotations

import json

UPLOAD = "/api/v1/datasets/upload"
DATASETS = "/api/v1/datasets"
TOKENIZERS = "/api/v1/tokenizers"


def _files(name: str, content: bytes = b"plain text record\n"):
    return [("files", (name, content, "text/plain"))]


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def test_upload_list_get_verify_delete(api_client):
    resp = api_client.post(UPLOAD, files=_files("api.txt", b"api record alpha\n"),
                           data={"name": "api-ds"})
    assert resp.status_code == 201, resp.text
    up = resp.json()
    ds_id = up["dataset_id"]
    assert up["version"] == 1
    assert up["counts"]["unique_record_count"] == 1
    assert up["files"][0]["sha256"]

    listing = api_client.get(DATASETS).json()
    assert any(d["id"] == ds_id for d in listing)

    got = api_client.get(f"{DATASETS}/{ds_id}").json()
    assert got["dataset"]["name"] == "api-ds"
    assert got["versions"][0]["counts"]["unique_record_count"] == 1

    assert api_client.get(f"{DATASETS}/{ds_id}/verify").json()["status"] == "ok"
    assert api_client.delete(f"{DATASETS}/{ds_id}").json()["deleted"] == ds_id
    assert api_client.get(f"{DATASETS}/{ds_id}").status_code == 404


def test_mixed_upload_per_file_report(api_client):
    resp = api_client.post(
        UPLOAD,
        files=[("files", ("ok.txt", b"good record\n", "text/plain")),
               ("files", ("bad.pdf", b"%PDF-1.4 nope", "application/pdf")),
               ("files", ("broken.json", b"{not json", "application/json"))],
        data={"name": "api-mixed"},
    )
    assert resp.status_code == 201, resp.text
    files = {f["filename"]: f for f in resp.json()["files"]}
    assert files["ok.txt"]["status"] == "ok"
    assert files["bad.pdf"]["status"] == "error"
    assert "unsupported extension" in files["bad.pdf"]["error"]
    assert files["broken.json"]["status"] == "error"
    assert "malformed JSON" in files["broken.json"]["error"]
    assert resp.json()["counts"]["unique_record_count"] == 1
    ds_id = resp.json()["dataset_id"]
    assert api_client.delete(f"{DATASETS}/{ds_id}").status_code == 200


def test_all_files_fail_and_duplicate_name_conflict(api_client):
    assert api_client.post(UPLOAD, files=_files("x.pdf", b"%PDF"),
                           data={"name": "api-allbad"}).status_code == 422
    assert api_client.post(UPLOAD, files=_files("a.txt", b"r1\n"),
                           data={"name": "api-conflict"}).status_code == 201
    resp = api_client.post(UPLOAD, files=_files("a.txt", b"r2\n"),
                           data={"name": "api-conflict"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


def test_append_version_roundtrip(api_client):
    r1 = api_client.post(UPLOAD, files=_files("a.txt", b"version one record\n"),
                         data={"name": "api-versions"})
    ds_id = r1.json()["dataset_id"]
    r2 = api_client.post(UPLOAD, files=_files("b.txt", b"version two record\n"),
                         data={"dataset_id": ds_id})
    assert r2.status_code == 201
    assert r2.json()["version"] == 2
    got = api_client.get(f"{DATASETS}/{ds_id}").json()
    assert got["dataset"]["versions"] == [1, 2]
    assert [v["version"] for v in got["versions"]] == [1, 2]
    assert api_client.get(f"{DATASETS}/{ds_id}/verify").json()["status"] == "ok"
    api_client.delete(f"{DATASETS}/{ds_id}")


def test_404s_and_missing_dataset_id(api_client):
    assert api_client.get(f"{DATASETS}/nope").status_code == 404
    assert api_client.get(f"{DATASETS}/nope/verify").status_code == 404
    assert api_client.post(f"{DATASETS}/nope/tokenize",
                           json={"tokenizer_id": "x"}).status_code == 404
    assert api_client.delete(f"{DATASETS}/nope").status_code == 404
    # appending to an unknown dataset id
    assert api_client.post(UPLOAD, files=_files("a.txt", b"x\n"),
                           data={"dataset_id": "nope"}).status_code == 404


# --------------------------------------------------------------------------- #
# Tokenizers (API)
# --------------------------------------------------------------------------- #


def _tok_config(name: str, **extra) -> str:
    return json.dumps({"name": name, "vocab_size": 320, **extra})


def _corpus(n: int, prefix: str = "sentence") -> bytes:
    return ("\n\n".join(
        f"{prefix} corpus sentence number {i} with enough words for tokenizer variety"
        for i in range(n)
    )).encode("utf-8")


def test_train_list_get_delete_tokenizer(api_client):
    resp = api_client.post(
        TOKENIZERS + "/train",
        data={"config": _tok_config("api-tok")},
        files=[("files", ("corpus.txt", _corpus(200), "text/plain"))],
    )
    assert resp.status_code == 201, resp.text
    rec = resp.json()["tokenizer"]
    assert rec["actual_vocab_size"] == 320
    assert rec["requested_vocab_size"] == 320
    assert rec["type"] == "byte_bpe"

    assert any(t["id"] == rec["id"] for t in api_client.get(TOKENIZERS).json())
    got = api_client.get(f"{TOKENIZERS}/{rec['id']}").json()
    assert got["tokenizer_hash"] == rec["tokenizer_hash"]
    assert api_client.delete(f"{TOKENIZERS}/{rec['id']}").json()["deleted"] == rec["id"]
    assert api_client.get(f"{TOKENIZERS}/{rec['id']}").status_code == 404
    assert api_client.delete(f"{TOKENIZERS}/{rec['id']}").status_code == 404


def test_tokenizer_config_validation_via_api(api_client):
    for vocab in (100, 70000):
        resp = api_client.post(TOKENIZERS + "/train",
                               data={"config": _tok_config("api-bad", vocab_size=vocab)},
                               files=_files("c.txt", b"x\n"))
        assert resp.status_code == 422
    # malformed config JSON
    resp = api_client.post(TOKENIZERS + "/train", data={"config": "{oops"},
                           files=_files("c.txt", b"x\n"))
    assert resp.status_code == 422
    # training source missing entirely
    resp = api_client.post(TOKENIZERS + "/train", data={"config": _tok_config("api-nosrc")})
    assert resp.status_code == 422
    # nonexistent dataset to train on
    resp = api_client.post(TOKENIZERS + "/train",
                           data={"config": _tok_config("api-nods"), "dataset_id": "nope"})
    assert resp.status_code == 404


def test_dataset_delete_conflict_409(api_client):
    up = api_client.post(UPLOAD, files=_files("ds.txt", _corpus(150, "conflict")),
                         data={"name": "api-409"})
    assert up.status_code == 201
    ds_id = up.json()["dataset_id"]
    tr = api_client.post(TOKENIZERS + "/train",
                         data={"config": _tok_config("api-409-tok"), "dataset_id": ds_id})
    assert tr.status_code == 201
    tok_id = tr.json()["tokenizer"]["id"]

    assert api_client.delete(f"{DATASETS}/{ds_id}").status_code == 409
    assert api_client.delete(f"{TOKENIZERS}/{tok_id}").status_code == 200
    assert api_client.delete(f"{DATASETS}/{ds_id}").status_code == 200
    assert api_client.get(f"{DATASETS}/{ds_id}").status_code == 404


def test_full_pipeline_tokenize_via_api(api_client):
    up = api_client.post(UPLOAD, files=_files("pipe.txt", _corpus(300, "pipeline")),
                         data={"name": "api-pipe"})
    assert up.status_code == 201
    ds_id = up.json()["dataset_id"]

    tr = api_client.post(TOKENIZERS + "/train",
                         data={"config": _tok_config("api-pipe-tok"), "dataset_id": ds_id})
    assert tr.status_code == 201
    tok_id = tr.json()["tokenizer"]["id"]
    assert tr.json()["tokenizer"]["trained_on_dataset_id"] == ds_id

    # dataset is protected while the tokenizer references it
    assert api_client.delete(f"{DATASETS}/{ds_id}").status_code == 409

    tok = api_client.post(f"{DATASETS}/{ds_id}/tokenize", json={"tokenizer_id": tok_id})
    assert tok.status_code == 200, tok.text
    body = tok.json()
    assert body["dtype"] == "uint16"
    assert set(body["splits"]) == {"train", "validation", "test"}
    assert body["splits"]["train"]["count"] > 0
    assert body["splits"]["validation"]["count"] > 0

    # missing tokenizer_id in the body
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={}).status_code == 422
    # unknown tokenizer
    assert api_client.post(f"{DATASETS}/{ds_id}/tokenize",
                           json={"tokenizer_id": "ghost"}).status_code == 404

    # verify sees the tokenized artifacts as healthy
    report = api_client.get(f"{DATASETS}/{ds_id}/verify").json()
    assert report["status"] == "ok"
    assert report["versions"][0]["tokenized"][0]["tokenizer_id"] == tok_id

    # cleanup: tokenizer first, then dataset
    assert api_client.delete(f"{TOKENIZERS}/{tok_id}").status_code == 200
    assert api_client.delete(f"{DATASETS}/{ds_id}").status_code == 200


def test_project_counts_include_new_artifact_kinds(api_client):
    info = api_client.get("/api/v1/project").json()
    assert {"model_count", "dataset_count", "tokenizer_count"} <= set(info)
    up = api_client.post(UPLOAD, files=_files("p.txt", b"project counts\n"),
                         data={"name": "api-projcounts"})
    assert up.status_code == 201
    api_client.delete(f"{DATASETS}/{up.json()['dataset_id']}")
