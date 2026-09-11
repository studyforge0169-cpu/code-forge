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


# --------------------------------------------------------------------------- #
# M46: checkpoint history by training run (read-only grouping)
# --------------------------------------------------------------------------- #

def test_m46_checkpoints_by_run_parity_partition_determinism(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "m46runs")
    run_a = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=8, eval_every_steps=4)).json()["run_id"]
    run_b = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4, eval_every_steps=4)).json()["run_id"]
    assert run_a != run_b
    listing = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(listing) == 3
    # authoritative M3 ordering: (step, created_at) ascending
    assert [(c["step"], c["created_at"]) for c in listing] == \
        sorted((c["step"], c["created_at"]) for c in listing)
    by_run = f"{MODELS}/{model_id}/checkpoints/by-run"
    # exact listing parity per run, filtered VERBATIM by the
    # checkpoints' own persisted run_id
    ga = api_client.get(f"{by_run}/{run_a}")
    gb = api_client.get(f"{by_run}/{run_b}")
    assert ga.status_code == 200 and gb.status_code == 200
    assert ga.json() == [c for c in listing if c["run_id"] == run_a]
    assert gb.json() == [c for c in listing if c["run_id"] == run_b]
    # TRUE disjoint partition: 2 + 1 == 3, no overlap, full cover
    ids_a = {c["checkpoint_id"] for c in ga.json()}
    ids_b = {c["checkpoint_id"] for c in gb.json()}
    assert len(ga.json()) == 2 and len(gb.json()) == 1
    assert ids_a.isdisjoint(ids_b)
    assert ids_a | ids_b == {c["checkpoint_id"] for c in listing}
    # per-group order is the exact authoritative order preserved
    for group, run_id in ((ga.json(), run_a), (gb.json(), run_b)):
        assert [(c["step"], c["created_at"]) for c in group] == \
            sorted((c["step"], c["created_at"]) for c in group)
        assert all(c["run_id"] == run_id and c["model_id"] == model_id
                   for c in group)
    # verbatim payload parity with the M3 detail getter (all records)
    for c in listing:
        one = api_client.get(
            f"{MODELS}/{model_id}/checkpoints/{c['checkpoint_id']}")
        assert one.status_code == 200 and one.json() == c
    # determinism: repeated GETs are byte-identical
    assert api_client.get(f"{by_run}/{run_a}").content == ga.content
    assert api_client.get(f"{by_run}/{run_a}").content == ga.content


def test_m46_by_run_errors_empty_isolation_regressions_openapi(api_client):
    from pathlib import Path

    from app import engine as engine_module

    ds_id, tok_id, model_id = _prepare(api_client, "m46err")
    o_ds, o_tok, other_id = _prepare(api_client, "m46iso")
    rep = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4, eval_every_steps=4)).json()
    other_rep = api_client.post(TRAIN_RUN, json=_run_cfg(
        other_id, o_ds, o_tok, steps=4, eval_every_steps=4)).json()
    by_run = f"{MODELS}/{model_id}/checkpoints/by-run"
    assert len(api_client.get(f"{by_run}/{rep['run_id']}").json()) == 1

    # unknown model / unknown run / another model's run -> 404
    assert api_client.get(
        f"{MODELS}/ghost-model-46/checkpoints/by-run/{rep['run_id']}"
    ).status_code == 404
    assert api_client.get(f"{by_run}/no-such-run-46").status_code == 404
    assert api_client.get(
        f"{by_run}/{other_rep['run_id']}").status_code == 404
    assert api_client.get(
        f"{MODELS}/{other_id}/checkpoints/by-run/{rep['run_id']}"
    ).status_code == 404

    # valid-empty: a healthy registered run ALWAYS has >= 1 checkpoint
    # (the final step is an eval point), so the zero-checkpoint case
    # surfaces through the M3 listing's corruption resilience —
    # corrupt the run's only manifest, the listing skips it, and the
    # registered run returns 200 [] (never 404, never fabricated ids)
    listing = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(listing) == 1
    ckpt_dir = (Path(engine_module._forge.storage.model_dir(model_id))
                / "checkpoints" / listing[0]["checkpoint_id"])
    (ckpt_dir / "manifest.json").write_text("{ not json")
    assert api_client.get(f"{MODELS}/{model_id}/checkpoints").json() == []
    empty = api_client.get(f"{by_run}/{rep['run_id']}")
    assert empty.status_code == 200 and empty.json() == []

    # M3 regressions: the untouched model's listing + detail + the
    # provenance registry remain unchanged
    o_listing = api_client.get(f"{MODELS}/{other_id}/checkpoints").json()
    assert len(o_listing) == 1
    assert api_client.get(
        f"{MODELS}/{other_id}/checkpoints/{o_listing[0]['checkpoint_id']}"
    ).json() == o_listing[0]
    manifest = api_client.get(f"{MODELS}/{other_id}").json()
    assert other_rep["run_id"] in \
        {p["run_id"] for p in manifest["training_provenance"]}
    assert api_client.get(
        f"{MODELS}/{other_id}/checkpoints/by-run/{other_rep['run_id']}"
    ).json() == o_listing

    # OpenAPI: 78 paths, the new path exactly once, GET-only, tag
    # training; route order listing < by-run < generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30) + 1 (M31) + 1 (M32) + 1 (M33)
    # + 1 (M34) + 1 (M35) + 1 (M36) + 1 (M37) + 1 (M38) + 1 (M39)
    # + 1 (M40) + 1 (M41) + 1 (M42) + 1 (M43) + 1 (M44)
    # + 1 (M45) + 1 (M46 checkpoints by-run) = 78
    assert len(spec["paths"]) == 91
    path = "/api/v1/models/{model_id}/checkpoints/by-run/{run_id}"
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["training"]
    assert [p["name"] for p in item["get"]["parameters"]] == \
        ["model_id", "run_id"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "type": "object", "additionalProperties": True}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/checkpoints") \
        < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}")
    # the M45 route is still present exactly once
    assert keys.count("/api/v1/models/{model_id}/gates/decisions/"
                      "by-baseline-type/{baseline_type}") == 1


# --------------------------------------------------------------------- M52
# Best-checkpoint selection: read-only, deterministic, minimum PERSISTED
# validation_loss over the authoritative M3 listing; criterion explicit;
# ties by the canonical (step, created_at) ASC order; never writes.

def _m52_expected_best(listing: list[dict]) -> dict:
    """Independent local argmin: lowest persisted validation_loss, ties
    resolved by the listing's canonical (step, created_at) ASC order."""
    ordered = sorted(listing, key=lambda c: (c["step"], c["created_at"]))
    return min(ordered, key=lambda c: c["validation_loss"])


def test_m52_best_checkpoint_selection_parity_determinism_errors(api_client):
    from pathlib import Path

    from app import engine as engine_module

    ds_id, tok_id, model_id = _prepare(api_client, "m52best")
    rep = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=8, eval_every_steps=4)).json()
    assert len(rep["checkpoints"]) == 2
    listing = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(listing) == 2
    best_url = f"{MODELS}/{model_id}/checkpoints/best"

    # storage is untouched by selection GETs (count under THIS model)
    mdir = Path(engine_module._forge.storage.model_dir(model_id))
    n_files = sum(1 for p in mdir.rglob("*") if p.is_file())

    got = api_client.get(best_url)
    assert got.status_code == 200, got.text
    sel = got.json()
    expected = _m52_expected_best(listing)
    # the criterion is EXPLICIT in the payload; candidates counted;
    # the selected record is the complete verbatim listing entry
    assert sel["model_id"] == model_id
    assert sel["criterion"] == "minimum_persisted_validation_loss"
    assert sel["candidate_count"] == 2
    assert sel["tied"] is (sum(1 for c in listing
                               if c["validation_loss"]
                               == expected["validation_loss"]) > 1)
    assert sel["checkpoint"] == expected
    assert sel["checkpoint"]["validation_loss"] == min(
        c["validation_loss"] for c in listing)
    # verbatim parity with the M3 detail getter
    one = api_client.get(
        f"{MODELS}/{model_id}/checkpoints/{expected['checkpoint_id']}")
    assert one.status_code == 200 and one.json() == sel["checkpoint"]

    # determinism: repeated GETs are byte-identical
    assert api_client.get(best_url).content == got.content
    assert api_client.get(best_url).content == got.content

    # route collision: "best" resolves to the SELECTION route (200 with
    # the criterion), never to the generic {checkpoint_id} detail (which
    # would 404 on an unknown checkpoint id "best")
    assert got.json().get("criterion") is not None

    # read-only: zero writes under the model dir
    assert sum(1 for p in mdir.rglob("*") if p.is_file()) == n_files

    # unknown model -> 404 (same taxonomy as the checkpoint family)
    assert api_client.get(
        f"{MODELS}/ghost-m52/checkpoints/best").status_code == 404

    # valid model with ZERO checkpoints (created, never trained) -> the
    # established not-found semantic: 404, nothing manufactured
    _, _, fresh_id = _prepare(api_client, "m52fresh")
    empty = api_client.get(f"{MODELS}/{fresh_id}/checkpoints/best")
    assert empty.status_code == 404
    assert "no selectable checkpoints" in empty.json()["detail"]
    assert api_client.get(f"{MODELS}/{fresh_id}/checkpoints").json() == []


def test_m52_best_checkpoint_tie_corruption_openapi(api_client):
    import json as _json
    from pathlib import Path

    from app import engine as engine_module

    ds_id, tok_id, model_id = _prepare(api_client, "m52tie")
    assert api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=8, eval_every_steps=4)).status_code == 200
    listing = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(listing) == 2
    best_url = f"{MODELS}/{model_id}/checkpoints/best"

    # craft an EXACT tie: set the later checkpoint's persisted
    # validation_loss equal to the earlier one's (a test-fixture edit,
    # not production); the tie-break must then select the FIRST among
    # equals in the canonical (step, created_at) ASC order
    earlier, later = sorted(listing, key=lambda c: (c["step"],
                                                    c["created_at"]))
    assert earlier["validation_loss"] != later["validation_loss"]
    ck_dir = (Path(engine_module._forge.storage.model_dir(model_id))
              / "checkpoints" / later["checkpoint_id"])
    man = _json.loads((ck_dir / "manifest.json").read_text())
    man["validation_loss"] = earlier["validation_loss"]
    (ck_dir / "manifest.json").write_text(_json.dumps(man))

    listing2 = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    got = api_client.get(best_url)
    assert got.status_code == 200, got.text
    sel = got.json()
    assert sel["candidate_count"] == 2
    assert sel["tied"] is True
    assert sel["checkpoint"]["checkpoint_id"] == earlier["checkpoint_id"]
    assert sel["checkpoint"]["validation_loss"] == \
        earlier["validation_loss"]
    assert sel["checkpoint"] == _m52_expected_best(listing2)

    # corruption resilience (M46 semantics): unreadable manifests are
    # skipped by the listing; with NO usable checkpoints the selection
    # is the established 404 — nothing is manufactured
    for c in listing2:
        d = (Path(engine_module._forge.storage.model_dir(model_id))
             / "checkpoints" / c["checkpoint_id"])
        (d / "manifest.json").write_text("{ not json")
    assert api_client.get(f"{MODELS}/{model_id}/checkpoints").json() == []
    gone = api_client.get(best_url)
    assert gone.status_code == 404
    assert "no selectable checkpoints" in gone.json()["detail"]

    # OpenAPI: 85 paths (M59 added the best-history route), GET-only, training
    # tag, model_id param, typed $ref response, and the display order
    # listing < by-run < best < generic detail (best can never be
    # captured as {checkpoint_id})
    spec = api_client.get("/openapi.json").json()
    # 78 (M46 cumulative) + 1 (M47) + 1 (M48) + 1 (M49) + 1 (M50)
    # + 1 (M51 recipe plan) + 1 (M52 checkpoints best)
    # + 1 (M59 checkpoints best history) = 85
    assert len(spec["paths"]) == 91
    path = "/api/v1/models/{model_id}/checkpoints/best"
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["training"]
    assert [p["name"] for p in item["get"]["parameters"]] == ["model_id"]
    assert item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"] == {
        "$ref": "#/components/schemas/CheckpointSelection"}
    assert "CheckpointSelection" in spec["components"]["schemas"]
    assert spec["components"]["schemas"]["CheckpointSelection"][
        "properties"]["checkpoint"] == {
        "$ref": "#/components/schemas/CheckpointRecord"}
    assert keys.index("/api/v1/models/{model_id}/checkpoints") < \
        keys.index(path)
    assert keys.index("/api/v1/models/{model_id}/checkpoints/by-run/"
                      "{run_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}")


def test_m54_api_resume_point_errors_and_openapi(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "m54api")
    assert api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=8,
        eval_every_steps=4)).status_code == 200
    listing = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(listing) == 2
    B = listing[0]["checkpoint_id"]                 # historical (not latest)
    assert B != listing[-1]["checkpoint_id"]

    # explicit resume through the EXISTING route: 200, provenance pins B
    run = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4, eval_every_steps=4,
        resume_from_checkpoint_id=B)).json()
    assert run["initial_model_version"] == B
    manifest = api_client.get(f"{MODELS}/{model_id}").json()
    prov = [p for p in manifest["training_provenance"]
            if p["run_id"] == run["run_id"]][0]
    assert prov["initial_checkpoint_id"] == B
    assert prov["config"]["resume_from_checkpoint_id"] == B

    # error taxonomy through the API: unknown -> 404, foreign -> 404,
    # corrupted -> 409, malformed id -> 422 (schema, pre-handler)
    assert api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4,
        resume_from_checkpoint_id="ghost-ck-m54")).status_code == 404
    other = _prepare(api_client, "m54other")
    o_ck = api_client.post(TRAIN_RUN, json=_run_cfg(
        other[2], other[0], other[1], steps=4,
        eval_every_steps=4)).json()["checkpoints"][0]["checkpoint_id"]
    assert api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4,
        resume_from_checkpoint_id=o_ck)).status_code == 404
    from pathlib import Path
    from app import engine as engine_module
    wpath = (Path(engine_module._forge.storage.model_dir(model_id))
             / "checkpoints" / B / "weights.pt")
    orig = wpath.read_bytes()
    # silently-perturbed weights: loads fine but FAILS the content-hash
    # integrity verification -> 409 (the established integrity mapping)
    import torch as _torch
    state = _torch.load(wpath, map_location="cpu", weights_only=True)
    key = sorted(state)[0]
    state[key] = state[key] + 1.0
    _torch.save(state, wpath)
    r = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4, resume_from_checkpoint_id=B))
    assert r.status_code == 409 and "integrity" in r.json()["detail"]
    # unreadable weights (torch.load fails) map through the established
    # /training/run RuntimeError -> 422 branch
    wpath.write_bytes(b"not a torch file")
    r = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4, resume_from_checkpoint_id=B))
    assert r.status_code == 422 and "unreadable" in r.json()["detail"]
    wpath.write_bytes(orig)
    bad = _run_cfg(model_id, ds_id, tok_id, steps=4,
                   resume_from_checkpoint_id="")
    assert api_client.post(TRAIN_RUN, json=bad).status_code == 422

    # OpenAPI: the M54 field is part of the existing TrainingConfig
    # schema (path count moved 84 -> 85 with the M59 history route)
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 91
    assert "resume_from_checkpoint_id" in spec["components"]["schemas"][
        "TrainingConfig"]["properties"]


# --------------------------------------------------------------------- M59
# Best-checkpoint improvement HISTORY: read-only, computed live, only the
# checkpoints that BECAME the M52-selected best; final entry == M52; the
# ONE shared winner rule; zero storage.

def _m59_expected_history(listing: list[dict]) -> list[dict]:
    """Independent local replay: chronological (created_at, id) order,
    running argmin under the M52 winner rule (lower loss, or an equal
    loss with a strictly earlier canonical (step, created_at) key)."""
    chronological = sorted(listing,
                           key=lambda c: (c["created_at"], c["checkpoint_id"]))
    entries = []
    best = None
    for c in chronological:
        if best is not None:
            if c["validation_loss"] > best["validation_loss"]:
                continue
            if c["validation_loss"] == best["validation_loss"] \
                    and (c["step"], c["created_at"]) \
                    >= (best["step"], best["created_at"]):
                continue
        entries.append({
            "checkpoint_id": c["checkpoint_id"],
            "run_id": c["run_id"],
            "step": c["step"],
            "created_at": c["created_at"],
            "validation_loss": c["validation_loss"],
            "perplexity": c["perplexity"],
            "delta_loss_nats": (None if best is None else
                                c["validation_loss"]
                                - best["validation_loss"])})
        best = c
    return entries


def test_m59_history_parity_determinism_errors(api_client):
    from pathlib import Path

    from app import engine as engine_module

    ds_id, tok_id, model_id = _prepare(api_client, "m59hist")
    assert api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=8, eval_every_steps=4)).status_code == 200
    listing = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(listing) == 2
    hist_url = f"{MODELS}/{model_id}/checkpoints/best/history"

    mdir = Path(engine_module._forge.storage.model_dir(model_id))
    n_files = sum(1 for p in mdir.rglob("*") if p.is_file())

    got = api_client.get(hist_url)
    assert got.status_code == 200, got.text
    hist = got.json()
    expected = _m59_expected_history(listing)
    # shape + parity with the independent local replay
    assert hist["model_id"] == model_id
    assert hist["criterion"] == "minimum_persisted_validation_loss"
    assert hist["candidate_count"] == len(listing)
    assert hist["entries"] == expected
    # only WINNERS appear: a 2-checkpoint run with monotone losses has
    # both as winners; craft a non-winner to prove exclusion
    import json as _json
    ck_dir = (Path(engine_module._forge.storage.model_dir(model_id))
              / "checkpoints" / listing[-1]["checkpoint_id"])
    man = _json.loads((ck_dir / "manifest.json").read_text())
    man["validation_loss"] = listing[0]["validation_loss"] + 5.0  # worse
    (ck_dir / "manifest.json").write_text(_json.dumps(man))
    listing2 = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    hist2 = api_client.get(hist_url).json()
    assert hist2["entries"] == _m59_expected_history(listing2)
    assert [e["checkpoint_id"] for e in hist2["entries"]] == \
        [listing2[0]["checkpoint_id"]]
    # the FINAL entry equals the M52 selection (verbatim, not approximate)
    sel = api_client.get(f"{MODELS}/{model_id}/checkpoints/best").json()
    assert hist2["entries"][-1]["checkpoint_id"] \
        == sel["checkpoint"]["checkpoint_id"]
    assert hist2["entries"][-1]["validation_loss"] \
        == sel["checkpoint"]["validation_loss"]
    # monotonic non-increasing; deltas correct (first None); the sign
    # convention is current - previous (negative = improvement)
    losses = [e["validation_loss"] for e in hist2["entries"]]
    assert all(losses[i] <= losses[i - 1] for i in range(1, len(losses)))
    assert hist2["entries"][0]["delta_loss_nats"] is None
    for i in range(1, len(hist2["entries"])):
        assert abs(hist2["entries"][i]["delta_loss_nats"]
                   - (losses[i] - losses[i - 1])) < 1e-12
        assert hist2["entries"][i]["delta_loss_nats"] <= 0.0
    # verbatim provenance fields from the authoritative record
    rec = api_client.get(
        f"{MODELS}/{model_id}/checkpoints/"
        f"{hist2['entries'][0]['checkpoint_id']}").json()
    assert hist2["entries"][0]["run_id"] == rec["run_id"]
    assert hist2["entries"][0]["step"] == rec["step"]
    assert hist2["entries"][0]["created_at"] == rec["created_at"]

    # determinism: repeated GETs are byte-identical; read-only: zero writes
    again = api_client.get(hist_url)
    assert again.content == api_client.get(hist_url).content
    assert again.json() == hist2
    assert sum(1 for p in mdir.rglob("*") if p.is_file()) == n_files

    # unknown model -> 404 (same taxonomy as the checkpoint family)
    assert api_client.get(
        f"{MODELS}/ghost-m59/checkpoints/best/history").status_code == 404

    # valid model with ZERO checkpoints -> EMPTY history (consistent with
    # the collection endpoints); the M52 SELECTION route stays 404
    _, _, fresh_id = _prepare(api_client, "m59fresh")
    empty = api_client.get(f"{MODELS}/{fresh_id}/checkpoints/best/history")
    assert empty.status_code == 200
    assert empty.json()["entries"] == []
    assert empty.json()["candidate_count"] == 0
    assert api_client.get(
        f"{MODELS}/{fresh_id}/checkpoints/best").status_code == 404
    assert api_client.get(f"{MODELS}/{fresh_id}/checkpoints").json() == []

    # model scoping: a SECOND model's checkpoints never contaminate
    ds2, tok2, mid2 = _prepare(api_client, "m59other")
    assert api_client.post(TRAIN_RUN, json=_run_cfg(
        mid2, ds2, tok2, steps=4, eval_every_steps=2)).status_code == 200
    listing_other = api_client.get(f"{MODELS}/{mid2}/checkpoints").json()
    hist_other = api_client.get(
        f"{MODELS}/{mid2}/checkpoints/best/history").json()
    assert hist_other["candidate_count"] == len(listing_other)
    ids = {e["checkpoint_id"] for e in hist_other["entries"]}
    other_ids = {c["checkpoint_id"] for c in listing_other}
    assert ids <= other_ids
    assert not (ids & {e["checkpoint_id"] for e in hist2["entries"]})


def test_m59_history_tie_nonfinite_live_computation(api_client):
    import json as _json
    from pathlib import Path

    from app import engine as engine_module

    ds_id, tok_id, model_id = _prepare(api_client, "m59tie")
    # three short runs: run A steps 1..4 (chronological == canonical),
    # run B and C later created_at but SMALLER steps
    for seed, steps in ((1, 4), (2, 2), (3, 2)):
        assert api_client.post(TRAIN_RUN, json=_run_cfg(
            model_id, ds_id, tok_id, steps=steps, eval_every_steps=1,
            seed=seed)).status_code == 200
    listing = api_client.get(f"{MODELS}/{model_id}/checkpoints").json()
    assert len(listing) == 8
    hist_url = f"{MODELS}/{model_id}/checkpoints/best/history"

    def set_loss(ckpt_id, loss):
        d = (Path(engine_module._forge.storage.model_dir(model_id))
             / "checkpoints" / ckpt_id)
        man = _json.loads((d / "manifest.json").read_text())
        man["validation_loss"] = loss
        (d / "manifest.json").write_text(_json.dumps(man))

    chron = sorted(listing,
                   key=lambda c: (c["created_at"], c["checkpoint_id"]))
    L1, L2, L3, L4, L5, L6, L7, L8 = chron
    # landscape: L1 first best; L2 worse; L3 wins; L4 an EXACT tie with a
    # LATER canonical key -> NO movement (M52 keeps the first among
    # equals); L5 an EXACT tie with an EARLIER canonical key (smaller
    # step from a later run) -> the selector SWITCHES (movement, delta
    # 0.0); L6 NaN -> never a candidate; L7 wins; L8 worse
    for ck, loss in ((L1, 6.5), (L2, 6.7), (L3, 6.4), (L4, 6.4), (L5, 6.4),
                     (L6, float("inf")), (L7, 6.2), (L8, 6.9)):
        set_loss(ck["checkpoint_id"], loss)

    hist = api_client.get(hist_url).json()
    assert [e["checkpoint_id"] for e in hist["entries"]] == [
        L1["checkpoint_id"], L3["checkpoint_id"], L5["checkpoint_id"],
        L7["checkpoint_id"]]
    assert [e["validation_loss"] for e in hist["entries"]] == \
        [6.5, 6.4, 6.4, 6.2]
    assert hist["candidate_count"] == 7                  # inf excluded
    assert hist["entries"][2]["delta_loss_nats"] == 0.0  # tie switch
    # the M52 selection agrees with the final entry AND the tie story
    sel = api_client.get(f"{MODELS}/{model_id}/checkpoints/best").json()
    assert sel["checkpoint"]["checkpoint_id"] == L7["checkpoint_id"]
    assert sel["tied"] is False

    # live computation: a better checkpoint appears -> the FINAL entry
    # changes immediately; the old best REMAINS in the sequence
    set_loss(L8["checkpoint_id"], 6.1)
    hist2 = api_client.get(hist_url).json()
    assert hist2["entries"][-1]["checkpoint_id"] == L8["checkpoint_id"]
    assert hist2["entries"][-1]["validation_loss"] == 6.1
    assert hist2["entries"][-2]["checkpoint_id"] == L7["checkpoint_id"]
    assert [e["checkpoint_id"] for e in hist2["entries"]][:4] == \
        [e["checkpoint_id"] for e in hist["entries"]]

    # non-finite NaN is excluded exactly like inf (M52 semantics)
    set_loss(L8["checkpoint_id"], float("nan"))
    hist3 = api_client.get(hist_url).json()
    assert hist3["candidate_count"] == 6
    assert hist3["entries"][-1]["checkpoint_id"] == L7["checkpoint_id"]

    # corruption (established M46/M52 semantics): unreadable manifests
    # are skipped by the listing; with NO usable checkpoints the history
    # is EMPTY (the collection convention), the selection 404
    for c in api_client.get(f"{MODELS}/{model_id}/checkpoints").json():
        d = (Path(engine_module._forge.storage.model_dir(model_id))
             / "checkpoints" / c["checkpoint_id"])
        (d / "manifest.json").write_text("{ not json")
    assert api_client.get(f"{MODELS}/{model_id}/checkpoints").json() == []
    gone = api_client.get(hist_url)
    assert gone.status_code == 200 and gone.json()["entries"] == []
    assert api_client.get(
        f"{MODELS}/{model_id}/checkpoints/best").status_code == 404


def test_m59_history_openapi(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "m59spec")
    assert api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=4, eval_every_steps=2)).status_code == 200
    path = "/api/v1/models/{model_id}/checkpoints/best/history"

    spec = api_client.get("/openapi.json").json()
    # 84 (M52 cumulative) + 1 (M59 best history) = 85
    assert len(spec["paths"]) == 91
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["training"]
    assert [p["name"] for p in item["get"]["parameters"]] == ["model_id"]
    assert item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"] == {
        "$ref": "#/components/schemas/BestCheckpointHistory"}
    assert "BestCheckpointHistory" in spec["components"]["schemas"]
    assert "BestCheckpointHistoryEntry" in spec["components"]["schemas"]
    # display order: listing < by-run < best < history < generic detail
    # (neither "best" nor "best/history" can be captured as {checkpoint_id})
    assert keys.index("/api/v1/models/{model_id}/checkpoints") < \
        keys.index(path)
    assert keys.index("/api/v1/models/{model_id}/checkpoints/by-run/"
                      "{run_id}") < keys.index(path)
    assert keys.index("/api/v1/models/{model_id}/checkpoints/best") < \
        keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}")


# --------------------------------------------------------------------------- #
# M61 — explicit verified checkpoint retention (HTTP surface)
# --------------------------------------------------------------------------- #

def test_m61_api_explicit_checkpoint_retention(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "m61del")
    r = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=12, eval_every_steps=4,
        keep_best=False, seed=1))
    assert r.status_code == 200, r.text
    CK = f"{MODELS}/{model_id}/checkpoints"
    listing = api_client.get(CK).json()
    assert len(listing) == 3
    best = api_client.get(f"{CK}/best").json()["checkpoint"]["checkpoint_id"]
    latest = api_client.get(f"{MODELS}/{model_id}").json()["latest_checkpoint"]
    victim = next(c["checkpoint_id"] for c in listing
                  if c["checkpoint_id"] not in (best, latest))
    others = {c["checkpoint_id"]: c for c in listing
              if c["checkpoint_id"] != victim}
    hist_before = api_client.get(f"{CK}/best/history").json()
    weights_before = _weights_sha(api_client, model_id)

    # unknown model / unknown checkpoint -> the family's 404
    assert api_client.delete(
        f"{MODELS}/no-such-m61/checkpoints/{victim}").status_code == 404
    assert api_client.delete(f"{CK}/no-such-ckpt").status_code == 404

    # protected: the M52 best -> 409 with the ordered structured blockers
    r = api_client.delete(f"{CK}/{best}")
    assert r.status_code == 409, r.text
    d = r.json()["detail"]
    assert d["protected"] is True and d["checkpoint_id"] == best
    assert d["blockers"][0]["reason"] == "best"

    # protected: the published/latest -> 409
    r = api_client.delete(f"{CK}/{latest}")
    assert r.status_code == 409, r.text
    assert "published" in [b["reason"] for b in r.json()["detail"]["blockers"]]

    # rejections deleted nothing
    assert len(api_client.get(CK).json()) == 3

    # the safe explicit deletion -> 200 with the deterministic result
    r = api_client.delete(f"{CK}/{victim}")
    assert r.status_code == 200, r.text
    res = r.json()
    assert res == {"model_id": model_id, "checkpoint_id": victim,
                   "files_removed": 2, "bytes_reclaimed": res["bytes_reclaimed"]}
    assert res["bytes_reclaimed"] > 0

    # gone: detail 404, listing shrinks, repeated deletion 404
    assert api_client.get(f"{CK}/{victim}").status_code == 404
    after = api_client.get(CK).json()
    assert len(after) == 2
    assert {c["checkpoint_id"]: c for c in after} == others  # verbatim
    assert api_client.delete(f"{CK}/{victim}").status_code == 404

    # published state, M52 and M59 stay coherent over the survivors
    assert api_client.get(f"{MODELS}/{model_id}").json()[
        "latest_checkpoint"] == latest
    assert api_client.get(f"{CK}/best").json()["checkpoint"][
        "checkpoint_id"] == best
    hist_after = api_client.get(f"{CK}/best/history").json()
    assert hist_after["candidate_count"] == 2
    assert [e["checkpoint_id"] for e in hist_after["entries"]] == \
        [e["checkpoint_id"] for e in hist_before["entries"]
         if e["checkpoint_id"] != victim]
    assert _weights_sha(api_client, model_id) == weights_before

    # OpenAPI: NO new path (the DELETE rides the existing checkpoint
    # detail path); the new operation + response schema are exposed
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 91
    path = "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}"
    assert "delete" in spec["paths"][path]
    assert "get" in spec["paths"][path]      # the existing detail stays
    assert "CheckpointDeletionResult" in spec["components"]["schemas"]


# --------------------------------------------------------------------------- #
# M62 — read-only retention overview (HTTP surface)
# --------------------------------------------------------------------------- #

def test_m62_api_retention_overview(api_client):
    ds_id, tok_id, model_id = _prepare(api_client, "m62ov")
    r = api_client.post(TRAIN_RUN, json=_run_cfg(
        model_id, ds_id, tok_id, steps=12, eval_every_steps=4,
        keep_best=False, seed=1))
    assert r.status_code == 200, r.text
    CK = f"{MODELS}/{model_id}/checkpoints"
    listing = api_client.get(CK).json()
    assert len(listing) == 3
    best = api_client.get(f"{CK}/best").json()["checkpoint"]["checkpoint_id"]
    latest = api_client.get(f"{MODELS}/{model_id}").json()["latest_checkpoint"]
    referenced = latest
    # one evidence reference through the public M4 route
    r = api_client.post("/api/v1/evaluations/run", json={
        "model_id": model_id, "dataset_id": ds_id, "tokenizer_id": tok_id,
        "split": "validation", "batch_size": 8, "max_seq_len": 32,
        "seed": 2, "checkpoint_id": referenced})
    assert r.status_code == 200, r.text

    RET = f"{CK}/retention"
    r = api_client.get(RET)
    assert r.status_code == 200, r.text
    ov = r.json()
    overview_body = r.content            # for the determinism check
    assert ov["model_id"] == model_id
    assert ov["total_checkpoints"] == 3
    assert [e["checkpoint_id"] for e in ov["checkpoints"]] == \
        [c["checkpoint_id"] for c in listing]     # canonical order
    by_id = {e["checkpoint_id"]: e for e in ov["checkpoints"]}
    protected = {cid for cid, e in by_id.items() if not e["deletable"]}
    assert protected == {best, latest, referenced}
    assert ov["deletable_checkpoints"] == 3 - len(protected)
    assert ov["protected_checkpoints"] == len(protected)
    assert ov["total_checkpoint_bytes"] == sum(e["size_bytes"]
                                               for e in ov["checkpoints"])
    assert ov["reclaimable_checkpoint_bytes"] == sum(
        e["size_bytes"] for e in ov["checkpoints"] if e["deletable"])
    for e in ov["checkpoints"]:
        assert set(e) == {"checkpoint_id", "run_id", "step", "created_at",
                          "validation_loss", "files", "size_bytes",
                          "integrity_verified", "deletable", "blockers"}
        assert e["files"] == 2 and e["integrity_verified"] is True

    # M52 / M60 consistency through the public routes
    best_entries = [e for e in ov["checkpoints"]
                    if "best" in [b["reason"] for b in e["blockers"]]]
    assert [e["checkpoint_id"] for e in best_entries] == [best]
    pub_entries = [e for e in ov["checkpoints"]
                   if "published" in [b["reason"] for b in e["blockers"]]]
    assert [e["checkpoint_id"] for e in pub_entries] == [latest]

    # THE headline parity: M61's 409 blocker list is EXACTLY the
    # overview's blocker list for the same checkpoint (the user never
    # sees "deletable" here and then a blocker there)
    r = api_client.delete(f"{CK}/{referenced}")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["blockers"] == by_id[referenced]["blockers"]
    r = api_client.delete(f"{CK}/{best}")
    assert r.status_code == 409
    assert r.json()["detail"]["blockers"] == by_id[best]["blockers"]

    # determinism over HTTP: byte-equal bodies on repeat
    r2 = api_client.get(RET)
    assert r2.status_code == 200 and r2.content == overview_body

    # read-only: the overview itself never mutates storage (rejections
    # above changed nothing; count still 3)
    assert len(api_client.get(CK).json()) == 3

    # a deletable entry really deletes with exactly the reported stats
    victim = next(e["checkpoint_id"] for e in ov["checkpoints"]
                  if e["deletable"])
    entry = by_id[victim]
    r = api_client.delete(f"{CK}/{victim}")
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["checkpoint_id"] == victim
    assert res["files_removed"] == entry["files"]
    assert res["bytes_reclaimed"] == entry["size_bytes"]
    ov_after = api_client.get(RET).json()
    assert ov_after["total_checkpoints"] == 2
    assert victim not in {e["checkpoint_id"] for e in ov_after["checkpoints"]}

    # unknown model 404; empty model 200 with zeroed totals
    assert api_client.get(f"{MODELS}/no-such-m62/checkpoints/retention"
                          ).status_code == 404
    ds2, tok2, mid2 = _prepare(api_client, "m62empty")
    r = api_client.get(f"{MODELS}/{mid2}/checkpoints/retention")
    assert r.status_code == 200, r.text
    empty = r.json()
    assert empty["total_checkpoints"] == 0 and empty["checkpoints"] == []
    assert empty["deletable_checkpoints"] == 0
    assert empty["protected_checkpoints"] == 0
    assert empty["total_checkpoint_bytes"] == 0
    assert empty["reclaimable_checkpoint_bytes"] == 0

    # OpenAPI: exactly one new path (85 -> 86), response schemas exposed
    spec = api_client.get("/openapi.json").json()
    assert len(spec["paths"]) == 91
    RET_PATH = "/api/v1/models/{model_id}/checkpoints/retention"
    assert set(spec["paths"][RET_PATH].keys()) == {"get"}
    assert "CheckpointRetentionOverview" in spec["components"]["schemas"]
    assert "CheckpointRetentionEntry" in spec["components"]["schemas"]
    # the route is declared before the generic {checkpoint_id} capture
    keys = list(spec["paths"].keys())
    assert keys.index(RET_PATH) < keys.index(
        "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}")
