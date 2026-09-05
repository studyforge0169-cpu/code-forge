"""REST API smoke tests (health, project, model registry endpoints)."""
from __future__ import annotations

from app.schemas import TransformerConfig


def payload(name: str = "api-test-model") -> dict:
    cfg = TransformerConfig.tiny().model_dump(mode="json")
    cfg["name"] = name
    return {"config": cfg, "description": "created by api smoke test", "metadata": {"purpose": "testing"}}


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "ai-model-forge"


def test_project_and_system(api_client):
    project = api_client.get("/api/v1/project").json()
    assert project["model_count"] == 0
    assert project["storage_root"]
    system = api_client.get("/api/v1/system").json()
    assert system["hardware"]["device"] in ("cpu", "cuda")
    assert system["storage"]["models"] == 0


def test_create_and_read_model(api_client):
    resp = api_client.post("/api/v1/models", json=payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    model = body["model"]
    assert model["name"] == "api-test-model"
    assert model["architecture"] == "decoder_only_transformer"
    assert model["config"]["vocab_size"] == 64
    assert model["parameter_count"] == 19_616
    assert model["state_hash"]
    assert body["weights_bytes_on_disk"] > 0
    model_id = model["id"]

    # list contains it
    listing = api_client.get("/api/v1/models").json()
    assert [m["id"] for m in listing] == [model_id]

    # get returns the full record
    got = api_client.get(f"/api/v1/models/{model_id}").json()
    assert got["config"] == model["config"]
    assert got["description"] == "created by api smoke test"
    assert got["metadata"] == {"purpose": "testing"}

    # verify + weights download
    assert api_client.get(f"/api/v1/models/{model_id}/verify").json()["integrity"] == "ok"
    dl = api_client.get(f"/api/v1/models/{model_id}/weights")
    assert dl.status_code == 200
    assert dl.content[:2] == b"PK"  # torch 2.x zipfile container for state dicts
    assert dl.headers["content-type"] == "application/octet-stream"

    # delete and 404 afterwards
    assert api_client.delete(f"/api/v1/models/{model_id}").json()["deleted"] == model_id
    assert api_client.get(f"/api/v1/models/{model_id}").status_code == 404


def test_create_with_bad_config_rejected(api_client):
    body = payload()
    body["config"]["hidden_size"] = 33  # not divisible by n_heads=4
    resp = api_client.post("/api/v1/models", json=body)
    assert resp.status_code == 422

    body = payload()
    body["config"]["n_kv_heads"] = 3
    resp = api_client.post("/api/v1/models", json=body)
    assert resp.status_code == 422


def test_validate_endpoint_reports_cleanly(api_client):
    ok = api_client.post("/api/v1/models/validate", json=TransformerConfig.tiny().model_dump(mode="json"))
    assert ok.status_code == 200
    assert ok.json() == {"valid": True, "errors": []}

    bad = TransformerConfig.tiny().model_dump(mode="json")
    bad["n_heads"] = 4
    bad["n_kv_heads"] = 3
    report = api_client.post("/api/v1/models/validate", json=bad)
    assert report.status_code == 200
    body = report.json()
    assert body["valid"] is False
    assert any("n_kv_heads" in e for e in body["errors"])


def test_duplicate_name_conflict(api_client):
    body = payload(name="dup-model")
    assert api_client.post("/api/v1/models", json=body).status_code == 201
    resp = api_client.post("/api/v1/models", json=body)
    assert resp.status_code == 422
    assert "already exists" in resp.json()["detail"]


def test_unknown_model_404(api_client):
    assert api_client.get("/api/v1/models/nope").status_code == 404
    assert api_client.delete("/api/v1/models/nope").status_code == 404
    assert api_client.get("/api/v1/models/nope/verify").status_code == 404


def test_flash_attention_on_cpu_is_rejected(api_client):
    body = payload(name="flash-cpu-model")
    body["config"]["attention_impl"] = "flash_attn"
    resp = api_client.post("/api/v1/models", json=body)
    assert resp.status_code == 422
    assert "CUDA" in resp.json()["detail"]
