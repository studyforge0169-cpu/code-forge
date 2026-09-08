"""Milestone 15 tests: deterministic checkpoint sampling (generation).

Engine tests use a module-scoped environment (dataset + two tokenizers + a
trained main model with checkpoints + a small-vocab model with its own
checkpoint for mismatch coverage), mirroring the M12/M14 test style; API
coverage runs over the shared HTTP TestClient root with per-test HTTP setup
(dataset + tokenizer + model + one short training run to produce a
checkpoint — generation always binds an explicit immutable checkpoint).

Covered: strict strategy schema rules (greedy forbids temperature/seed;
temperature requires both, temperature=0 is never reinterpreted as greedy);
greedy determinism (byte-identical repeats, ids + output + result_hash);
seeded-temperature determinism with the explicit seed; result-hash
sensitivity (prompt / strategy / parameters / checkpoint weights / tokenizer
participate) and exclusions (sample id / timestamps); context-window and
token-limit preflight; unknown model/checkpoint/tokenizer -> 404-class
FileNotFoundError; corrupt checkpoint -> integrity error; tokenizer/model
vocabulary mismatch -> rejection — all with ZERO writes; exactly one
manifest per successful request and nothing else (no weight copies, no
extra files, no .tmp); list/get semantics read-only and deterministic;
decode runs under no_grad through the existing forward/KV-cache path; full
HTTP lifecycle + OpenAPI exposure. M1–M14 behavior is untouched.
"""
from __future__ import annotations

import json
import random

import pytest
import torch
from pydantic import ValidationError

from app.sampling import SamplingEngine
from app.schemas import (
    ModelCreateRequest,
    SampleGenerateRequest,
    SampleStrategy,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
)

_WORDS = ("river mountain cloud forest desert ocean valley island meadow "
          "canyon table chair lamp desk shelf couch rug clock mirror "
          "vase").split()


def _corpus(n: int, tag: str) -> bytes:
    rng = random.Random(hash(tag) & 0xFFFF)
    lines = []
    for i in range(n):
        k = rng.randint(10, 22)
        lines.append(" ".join(rng.choices(_WORDS, k=k)) + f" number {i}")
    return ("\n\n".join(lines) + "\n").encode("utf-8")


def _tiny_model(name: str, vocab: int, seed: int = 1) -> TransformerConfig:
    return TransformerConfig(name=name, vocab_size=vocab,
                             context_length=64, hidden_size=64,
                             n_layers=2, n_heads=4, n_kv_heads=2,
                             intermediate_size=128, seed=seed)


class Env:
    """One temp root: dataset, big + small tokenizers, trained models."""

    def __init__(self, root):
        from app.engine import ModelForge

        self.forge = ModelForge(root=root)
        self.samples = SamplingEngine(self.forge.storage)

    def prepare(self):
        f = self.forge
        up = f.upload_dataset([("a.txt", _corpus(220, "m15-dom"))],
                              name="m15-dom")
        self.ds = up["dataset_id"]
        # big tokenizer: actual vocab must exceed the SMALL model's vocab so
        # the mismatch rejection is real
        self.tok_big = f.train_tokenizer(
            TokenizerConfig(name="m15-tok-big", vocab_size=600),
            dataset_id=self.ds)
        self.tok_small = f.train_tokenizer(
            TokenizerConfig(name="m15-tok-small", vocab_size=300),
            dataset_id=self.ds)
        assert self.tok_big.actual_vocab_size > 300
        assert self.tok_small.actual_vocab_size <= 300
        for tok in (self.tok_big, self.tok_small):
            f.tokenize_dataset(self.ds, tok.id)

        self.big_model = f.create_model(ModelCreateRequest(
            config=_tiny_model("m15-main", 640)))[0].id
        rep = f.run_training(TrainingConfig(
            name="m15-train-big", method="continued_pretraining",
            model_id=self.big_model, dataset_id=self.ds,
            tokenizer_id=self.tok_big.id, learning_rate=3e-3, batch_size=8,
            max_seq_len=32, steps=8, eval_every_steps=4, keep_best=False,
            seed=3))
        self.big_ckpts = [c["checkpoint_id"] for c in rep.checkpoints]
        assert len(self.big_ckpts) == 2

        self.small_model = f.create_model(ModelCreateRequest(
            config=_tiny_model("m15-small", 300, seed=7)))[0].id
        rep = f.run_training(TrainingConfig(
            name="m15-train-small", method="continued_pretraining",
            model_id=self.small_model, dataset_id=self.ds,
            tokenizer_id=self.tok_small.id, learning_rate=3e-3, batch_size=8,
            max_seq_len=32, steps=6, eval_every_steps=3, keep_best=False,
            seed=5))
        self.small_ckpt = rep.checkpoints[0]["checkpoint_id"]

    def ckpt_sha(self, model_id: str, ckpt_id: str) -> str:
        return self.forge.get_checkpoint(model_id, ckpt_id).weights_sha256

    def file_snapshot(self) -> dict[str, bytes]:
        root = self.forge.storage.root
        return {p.relative_to(root).as_posix(): p.read_bytes()
                for p in root.rglob("*") if p.is_file()}

    def no_tmp(self) -> None:
        leftovers = [p for p in self.forge.storage.root.rglob("*")
                     if p.is_file() and ".tmp" in p.name]
        assert leftovers == []


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m15-root"))
    e.prepare()
    return e


PROMPT = "river mountain cloud forest ocean desert valley island"

def _req(**overrides) -> SampleGenerateRequest:
    base = dict(model_id="m", checkpoint_id="c", tokenizer_id="t",
                prompt=PROMPT, strategy="greedy", max_new_tokens=8)
    base.update(overrides)
    return SampleGenerateRequest(**base)


def _g(env, **overrides):
    base = dict(model_id=env.big_model, checkpoint_id=env.big_ckpts[0],
                tokenizer_id=env.tok_big.id, prompt=PROMPT,
                strategy="greedy", max_new_tokens=8)
    base.update(overrides)
    return SampleGenerateRequest(**base)


# =========================================================================== #
# Schema: strict strategy rules
# =========================================================================== #

def test_strategy_schema_rules():
    _req(strategy="temperature", temperature=0.8, seed=5)  # valid shape
    # greedy forbids temperature and seed (never reinterpreted)
    with pytest.raises(ValidationError, match="takes no temperature"):
        _req(strategy="greedy", temperature=0.8)
    with pytest.raises(ValidationError, match="takes no seed"):
        _req(strategy="greedy", seed=5)
    # temperature REQUIRES both temperature in (0,1] and an explicit seed
    with pytest.raises(ValidationError, match="requires an explicit "
                       "temperature"):
        _req(strategy="temperature", seed=5)
    with pytest.raises(ValidationError, match="requires an explicit integer "
                       "seed"):
        _req(strategy="temperature", temperature=0.8)
    # temperature bounds: 0 is rejected (NOT greedy), >1 rejected
    with pytest.raises(ValidationError, match="greater than 0"):
        _req(strategy="temperature", temperature=0.0, seed=5)
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        _req(strategy="temperature", temperature=1.5, seed=5)
    # max_new_tokens bounds + strategy required + no extra fields
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _req(max_new_tokens=0)
    with pytest.raises(ValidationError, match="less than or equal to 512"):
        _req(max_new_tokens=513)
    with pytest.raises(ValidationError, match="Input should be 'greedy' or "
                       "'temperature'"):
        _req(strategy=None)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SampleGenerateRequest(**_req().model_dump(), extra=1)
    with pytest.raises(ValidationError, match="at least 1 character"):
        SampleGenerateRequest(**{**_req().model_dump(), "prompt": ""})


# =========================================================================== #
# Greedy determinism + record shape
# =========================================================================== #

def test_greedy_deterministic_byte_identical_repeats(env):
    r1 = env.samples.run(_g(env))
    r2 = env.samples.run(_g(env))
    assert r1.sample_id != r2.sample_id            # distinct records
    assert r1.generated_token_ids == r2.generated_token_ids
    assert r1.output_text == r2.output_text
    assert r1.result_hash == r2.result_hash
    assert len(r1.generated_token_ids) == 8 == r1.generated_token_count
    assert r1.strategy.value == "greedy"
    assert r1.temperature is None and r1.seed is None
    # audit fields: verified weights hash + tokenizer content hash
    assert r1.checkpoint_weights_sha256 == env.ckpt_sha(env.big_model,
                                                        env.big_ckpts[0])
    assert r1.tokenizer_hash == env.tok_big.tokenizer_hash
    assert r1.prompt == PROMPT and len(r1.prompt_token_ids) > 0
    assert r1.prompt_token_count == len(r1.prompt_token_ids)
    # manifests on disk round-trip (json serialization of list[int])
    disk = env.samples.get_sample(env.big_model, r1.sample_id)
    assert disk.generated_token_ids == r1.generated_token_ids
    assert disk.result_hash == r1.result_hash
    # the record's own output decodes from its ids with the same tokenizer
    hf = env.forge.tokenizers.get_hf(env.tok_big.id)
    assert hf.decode(r1.generated_token_ids) == r1.output_text
    env.no_tmp()


def test_temperature_same_seed_identical_and_seed_participates(env):
    def t(seed, temperature=0.8, **kw):
        return env.samples.run(_g(env, strategy="temperature",
                                  temperature=temperature, seed=seed, **kw))
    a, b = t(7), t(7)
    assert a.generated_token_ids == b.generated_token_ids
    assert a.output_text == b.output_text and a.result_hash == b.result_hash
    assert a.seed == 7 and a.temperature == 0.8
    c = t(8)                        # different seed: allowed to differ
    assert c.result_hash != a.result_hash   # seed is part of the payload
    d = t(7, temperature=0.5)
    assert d.result_hash != a.result_hash   # temperature is part of payload
    env.no_tmp()


def test_result_hash_semantics_and_exclusions(env):
    base = env.samples.run(_g(env))
    # every semantic input participates (guaranteed via the payload)
    assert env.samples.run(_g(env, prompt=PROMPT + " meadow")).result_hash \
        != base.result_hash                      # prompt changed
    assert env.samples.run(_g(env, max_new_tokens=4)).result_hash \
        != base.result_hash                      # parameter changed
    assert env.samples.run(_g(env, checkpoint_id=env.big_ckpts[1])
                           ).result_hash != base.result_hash  # checkpoint
    assert env.samples.run(_g(env, tokenizer_id=env.tok_small.id,
                              checkpoint_id=env.big_ckpts[1])  # big-model ids
                          ).result_hash != base.result_hash    # fit? checked
    # sample id / timestamps / duration are NOT part of the hash: a repeated
    # identical request has the same hash although those fields differ
    again = env.samples.run(_g(env))
    assert again.result_hash == base.result_hash
    assert again.sample_id != base.sample_id
    assert again.created_at != base.created_at
    assert len(base.result_hash) == 64
    env.no_tmp()


# =========================================================================== #
# Limits / context preflight
# =========================================================================== #

def test_context_window_and_token_limits(env):
    one = env.samples.run(_g(env, max_new_tokens=1))
    assert len(one.generated_token_ids) == 1
    # context is 64: any prompt+max_new_tokens > 64 is rejected, never
    # silently truncated (the engine stops at exactly max_new_tokens)
    with pytest.raises(ValueError, match="exceeds the model's context_length"):
        env.samples.run(_g(env, max_new_tokens=64))
    long_prompt = " ".join(["river"] * 60)
    with pytest.raises(ValueError, match="exceeds the model's context_length"):
        env.samples.run(_g(env, prompt=long_prompt, max_new_tokens=8))
    # boundary is accepted: prompt tokens + max_new_tokens == context
    n = len(env.samples.run(_g(env, max_new_tokens=1)).prompt_token_ids)
    fit = env.samples.run(_g(env, max_new_tokens=64 - n))
    assert len(fit.generated_token_ids) == 64 - n
    env.no_tmp()


# =========================================================================== #
# Compatibility rejections: zero writes every time
# =========================================================================== #

def test_unknown_model_checkpoint_tokenizer_rejected_no_writes(env):
    before = env.file_snapshot()
    with pytest.raises(FileNotFoundError, match="model 'no-such-model' not "
                       "found"):
        env.samples.run(SampleGenerateRequest(
            model_id="no-such-model", checkpoint_id=env.big_ckpts[0],
            tokenizer_id=env.tok_big.id, prompt=PROMPT, strategy="greedy",
            max_new_tokens=8))
    with pytest.raises(FileNotFoundError, match="checkpoint 'no-such-ckpt'"):
        env.samples.run(SampleGenerateRequest(
            model_id=env.big_model, checkpoint_id="no-such-ckpt",
            tokenizer_id=env.tok_big.id, prompt=PROMPT, strategy="greedy",
            max_new_tokens=8))
    with pytest.raises(FileNotFoundError, match="tokenizer 'no-such-tok'"):
        env.samples.run(SampleGenerateRequest(
            model_id=env.big_model, checkpoint_id=env.big_ckpts[0],
            tokenizer_id="no-such-tok", prompt=PROMPT, strategy="greedy",
            max_new_tokens=8))
    assert env.file_snapshot() == before
    env.no_tmp()


def test_vocabulary_mismatch_rejected_no_writes(env):
    """Platform compatibility convention (the exact M3/M4 rule): the
    tokenizer's actual vocab must fit the model's vocab_size. A 300-vocab
    model cannot be sampled through the 561+-vocab tokenizer."""
    before = env.file_snapshot()
    with pytest.raises(ValueError,
                       match=f"vocab \\({env.tok_big.actual_vocab_size}\\) "
                             "exceeds the model's vocab_size \\(300\\)"):
        env.samples.run(SampleGenerateRequest(
            model_id=env.small_model, checkpoint_id=env.small_ckpt,
            tokenizer_id=env.tok_big.id, prompt=PROMPT, strategy="greedy",
            max_new_tokens=8))
    # the matching small tokenizer works fine on the small model
    r = env.samples.run(SampleGenerateRequest(
        model_id=env.small_model, checkpoint_id=env.small_ckpt,
        tokenizer_id=env.tok_small.id, prompt=PROMPT, strategy="greedy",
        max_new_tokens=8))
    assert r.strategy.value == "greedy"
    assert all(t < env.tok_small.actual_vocab_size
               for t in r.generated_token_ids)
    assert env.file_snapshot() != before  # exactly the ONE new manifest
    new_files = [k for k in env.file_snapshot() if k not in before]
    assert len(new_files) == 1
    env.no_tmp()


def test_corrupt_checkpoint_refused_no_manifest(env):
    fresh = env.forge.create_model(ModelCreateRequest(
        config=_tiny_model("m15-corrupt-model", 640, seed=11)))[0].id
    rep = env.forge.run_training(TrainingConfig(
        name="m15-train-corrupt", method="continued_pretraining",
        model_id=fresh, dataset_id=env.ds, tokenizer_id=env.tok_big.id,
        learning_rate=3e-3, batch_size=8, max_seq_len=32, steps=4,
        eval_every_steps=4, keep_best=False, seed=9))
    ckpt = rep.checkpoints[0]["checkpoint_id"]
    # tamper the weights archive: content hash no longer matches the manifest
    wpath = (env.forge.storage.root / "models" / fresh / "checkpoints"
             / ckpt / "weights.pt")
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    first = next(iter(state))
    tampered = {k: (v.clone().fill_(0.0) if k == first else v.clone())
                for k, v in state.items()}
    torch.save(tampered, wpath)
    before = env.file_snapshot()
    with pytest.raises(RuntimeError, match="integrity"):
        env.samples.run(SampleGenerateRequest(
            model_id=fresh, checkpoint_id=ckpt, tokenizer_id=env.tok_big.id,
            prompt=PROMPT, strategy="greedy", max_new_tokens=8))
    assert env.file_snapshot() == before
    env.no_tmp()


# =========================================================================== #
# Storage discipline
# =========================================================================== #

def test_one_manifest_per_run_and_zero_extra_files(env):
    fresh = env.forge.create_model(ModelCreateRequest(
        config=_tiny_model("m15-storage-model", 640, seed=13)))[0].id
    rep = env.forge.run_training(TrainingConfig(
        name="m15-train-storage", method="continued_pretraining",
        model_id=fresh, dataset_id=env.ds, tokenizer_id=env.tok_big.id,
        learning_rate=3e-3, batch_size=8, max_seq_len=32, steps=4,
        eval_every_steps=4, keep_best=False, seed=15))
    ckpt = rep.checkpoints[0]["checkpoint_id"]
    before = env.file_snapshot()
    runs = [env.samples.run(SampleGenerateRequest(
        model_id=fresh, checkpoint_id=ckpt, tokenizer_id=env.tok_big.id,
        prompt=PROMPT, strategy="greedy", max_new_tokens=5)) for _ in range(3)]
    after = env.file_snapshot()
    new_files = [k for k in after if k not in before]
    assert len(new_files) == 3            # exactly one manifest per request
    assert all(k.startswith(f"samples/{fresh}/sample-") and k.endswith(
        "/manifest.json") for k in new_files)
    # every sample dir holds ONLY its manifest (no blobs/weights/caches)
    sroot = env.forge.storage.root / "samples" / fresh
    dirs = sorted(d for d in sroot.iterdir() if d.is_dir())
    assert [d.name for d in dirs] == sorted(d.name for d in dirs)
    assert all(sorted(p.name for p in d.iterdir()) == ["manifest.json"]
               for d in dirs)
    # every pre-existing file is byte-identical (no model/checkpoint writes)
    for k, v in before.items():
        assert after[k] == v
    assert [x.sample_id for x in env.samples.list_samples(fresh)] == \
        [r.sample_id for r in runs]
    env.no_tmp()


def test_list_get_semantics_read_only_deterministic(env):
    fresh = env.forge.create_model(ModelCreateRequest(
        config=_tiny_model("m15-list-model", 640, seed=17)))[0].id
    rep = env.forge.run_training(TrainingConfig(
        name="m15-train-list", method="continued_pretraining",
        model_id=fresh, dataset_id=env.ds, tokenizer_id=env.tok_big.id,
        learning_rate=3e-3, batch_size=8, max_seq_len=32, steps=4,
        eval_every_steps=4, keep_best=False, seed=21))
    ckpt = rep.checkpoints[0]["checkpoint_id"]
    ids = []
    for seed in (1, 2):
        ids.append(env.samples.run(SampleGenerateRequest(
            model_id=fresh, checkpoint_id=ckpt, tokenizer_id=env.tok_big.id,
            prompt=PROMPT, strategy="temperature", temperature=0.7, seed=seed,
            max_new_tokens=3)).sample_id)
    got = env.samples.list_samples(fresh)
    assert [s.sample_id for s in got] == ids
    keyed = [(s.created_at, s.sample_id) for s in got]
    assert keyed == sorted(keyed)
    assert env.samples.get_sample(fresh, ids[0]).sample_id == ids[0]
    with pytest.raises(FileNotFoundError):
        env.samples.list_samples("no-such-model")
    with pytest.raises(FileNotFoundError):
        env.samples.get_sample(fresh, "no-such-sample")
    with pytest.raises(FileNotFoundError):
        env.samples.get_sample("no-such-model", ids[0])
    # reads never write
    before = env.file_snapshot()
    env.samples.list_samples(fresh)
    env.samples.get_sample(fresh, ids[1])
    assert env.file_snapshot() == before
    env.no_tmp()


# =========================================================================== #
# Inference-only guarantees (no grad, existing forward path)
# =========================================================================== #

def test_decode_runs_without_grad_through_existing_forward(env, monkeypatch):
    import app.sampling as sampling_mod

    seen = []
    orig_build = sampling_mod.build_transformer

    def wrapped(cfg, parent_state=None, device="cpu"):
        built = orig_build(cfg, parent_state=parent_state, device=device)
        fwd = built.module.forward

        def guarded(*args, **kwargs):
            seen.append(torch.is_grad_enabled())
            return fwd(*args, **kwargs)

        built.module.forward = guarded
        return built

    monkeypatch.setattr(sampling_mod, "build_transformer", wrapped)
    r = env.samples.run(_g(env))
    assert seen and not any(seen), "forward ran with grad enabled"
    assert len(r.generated_token_ids) == 8


# =========================================================================== #
# HTTP API coverage (shared TestClient root; per-test model setup)
# =========================================================================== #

import json as _json  # noqa: E402

MODELS = "/api/v1/models"
GENERATE = "/api/v1/samples/generate"
PROMPT_API = "river mountain cloud forest ocean desert valley island"


def _http_env(api_client, tag: str, vocab: int = 640):
    """Dataset + tokenizer + model + ONE short training run (checkpoints)."""
    corpus = _corpus(200, tag)
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", (f"{tag}.txt", corpus,
                                           "text/plain"))],
                         data={"name": f"api15-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tok = api_client.post("/api/v1/tokenizers/train",
                          data={"config": _json.dumps(
                              {"name": f"api15-{tag}-tok",
                               "vocab_size": 320}),
                                "dataset_id": ds}).json()["tokenizer"]
    api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                    json={"tokenizer_id": tok["id"]})
    cfg = {"name": f"api15-{tag}-model", "vocab_size": vocab,
           "context_length": 64, "hidden_size": 64, "n_layers": 2,
           "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128,
           "seed": 1}
    mid = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    rep = api_client.post("/api/v1/training/run", json={
        "name": f"api15-{tag}-run", "method": "continued_pretraining",
        "model_id": mid, "dataset_id": ds, "tokenizer_id": tok["id"],
        "learning_rate": 3e-3, "batch_size": 8, "steps": 8,
        "max_seq_len": 32, "warmup_steps": 0, "weight_decay": 0.01,
        "adam_beta1": 0.9, "adam_beta2": 0.999, "lr_schedule": "cosine",
        "eval_every_steps": 4, "keep_best": True, "seed": 3})
    assert rep.status_code == 200, rep.text
    ckpt = rep.json()["checkpoints"][0]["checkpoint_id"]
    return {"mid": mid, "tok": tok["id"], "ckpt": ckpt}


def _gen_body(h, **overrides) -> dict:
    body = {"model_id": h["mid"], "checkpoint_id": h["ckpt"],
            "tokenizer_id": h["tok"], "prompt": PROMPT_API,
            "strategy": "greedy", "max_new_tokens": 6}
    body.update(overrides)
    return body


def test_api_generate_list_get_determinism(api_client):
    h = _http_env(api_client, "gen")
    body = _gen_body(h)

    r1 = api_client.post(GENERATE, json=body)
    assert r1.status_code == 200, r1.text
    s1 = r1.json()
    assert s1["strategy"] == "greedy" and s1["temperature"] is None
    assert s1["model_id"] == h["mid"] and s1["checkpoint_id"] == h["ckpt"]
    assert s1["tokenizer_id"] == h["tok"]
    assert s1["generated_token_count"] == 6
    assert len(s1["result_hash"]) == 64
    assert s1["checkpoint_weights_sha256"]
    assert s1["prompt"] == PROMPT_API

    r2 = api_client.post(GENERATE, json=body)
    s2 = r2.json()
    assert s2["generated_token_ids"] == s1["generated_token_ids"]
    assert s2["output_text"] == s1["output_text"]
    assert s2["result_hash"] == s1["result_hash"]
    assert s2["sample_id"] != s1["sample_id"]

    # list: deterministic (created_at, sample_id) order, then get
    lst1 = api_client.get(f"{MODELS}/{h['mid']}/samples")
    assert lst1.status_code == 200
    runs = lst1.json()
    assert len(runs) == 2
    keyed = [(x["created_at"], x["sample_id"]) for x in runs]
    assert keyed == sorted(keyed)
    assert api_client.get(f"{MODELS}/{h['mid']}/samples").text == lst1.text
    one = api_client.get(f"{MODELS}/{h['mid']}/samples/{s1['sample_id']}")
    assert one.status_code == 200
    assert one.json()["result_hash"] == s1["result_hash"]
    assert api_client.get(f"{MODELS}/{h['mid']}/samples/no-such").status_code \
        == 404
    assert api_client.get(f"{MODELS}/no-such-model/samples").status_code == 404
    # temperature over HTTP: same seed -> same ids/hash
    t1 = api_client.post(GENERATE, json=_gen_body(h, strategy="temperature",
                                                  temperature=0.8, seed=11))
    t2 = api_client.post(GENERATE, json=_gen_body(h, strategy="temperature",
                                                  temperature=0.8, seed=11))
    assert t1.status_code == 200 and t2.status_code == 200
    assert t1.json()["generated_token_ids"] == \
        t2.json()["generated_token_ids"]
    assert t1.json()["result_hash"] == t2.json()["result_hash"]
    # no update/delete paths exist (405)
    assert api_client.put(f"{MODELS}/{h['mid']}/samples/{s1['sample_id']}"
                          ).status_code == 405
    assert api_client.delete(f"{MODELS}/{h['mid']}/samples/{s1['sample_id']}"
                             ).status_code == 405


def test_api_failure_paths_and_zero_manifest_growth(api_client):
    h = _http_env(api_client, "fail")
    body = _gen_body(h)
    assert api_client.post(GENERATE, json=body).status_code == 200
    n0 = len(api_client.get(f"{MODELS}/{h['mid']}/samples").json())
    # unknown checkpoint / tokenizer / model -> 404, nothing written
    for bad, want in ((dict(body, checkpoint_id="no-such-ckpt"), 404),
                      (dict(body, tokenizer_id="no-such-tok"), 404),
                      (dict(body, model_id="no-such-model"), 404)):
        r = api_client.post(GENERATE, json=bad)
        assert r.status_code == want, r.text
    assert len(api_client.get(f"{MODELS}/{h['mid']}/samples").json()) == n0
    # schema/parameter violations -> 422, nothing written
    for bad in (dict(body, strategy="greedy", temperature=0.8),
                dict(body, strategy="temperature", temperature=0.8),
                dict(body, strategy="temperature", seed=1),
                dict(body, strategy="temperature", temperature=0.0, seed=1),
                dict(body, max_new_tokens=0),
                dict(body, max_new_tokens=513),
                dict(body, max_new_tokens=100),       # context overflow
                {"model_id": h["mid"], "checkpoint_id": h["ckpt"],
                 "tokenizer_id": h["tok"], "prompt": "", "strategy": "greedy",
                 "max_new_tokens": 4}):
        r = api_client.post(GENERATE, json=bad)
        assert r.status_code == 422, (bad, r.text)
    assert len(api_client.get(f"{MODELS}/{h['mid']}/samples").json()) == n0


def test_api_openapi_exposes_sampling(api_client):
    spec = api_client.get("/openapi.json").json()
    assert "/api/v1/samples/generate" in spec["paths"]
    assert "/api/v1/models/{model_id}/samples" in spec["paths"]
    assert "/api/v1/models/{model_id}/samples/{sample_id}" in spec["paths"]
    assert "SampleRecord" in spec["components"]["schemas"]
    assert "SampleGenerateRequest" in spec["components"]["schemas"]
    strat = [s["enum"] for s in spec["components"]["schemas"].values()
             if s.get("enum") == ["greedy", "temperature"]]
    assert len(strat) == 1


# =========================================================================== #
# M27: read-only per-checkpoint grouping of the sample history
# =========================================================================== #

BY_CKPT = "/api/v1/models/{mid}/samples/by-checkpoint/{ck}"


def _m27_state(env):
    """M27 state on top of the shared module env (cached): samples under
    BOTH main-model checkpoints plus one on the small model's checkpoint,
    and a fully trained fresh model whose checkpoint has zero samples.
    Returns (s_a1, s_a2, s_b1, s_small, empty_model, empty_ck).
    """
    cached = getattr(env, "_m27", None)
    if cached is not None:
        return cached
    f = env.forge
    s_a1 = f.generate_sample(_g(env))                                  # ck0
    s_a2 = f.generate_sample(_g(env, strategy="temperature",
                                temperature=0.8, seed=27,
                                max_new_tokens=6))                     # ck0
    s_b1 = f.generate_sample(_g(env,
                                checkpoint_id=env.big_ckpts[1]))      # ck1
    s_small = f.generate_sample(_g(env, model_id=env.small_model,
                                   checkpoint_id=env.small_ckpt,
                                   tokenizer_id=env.tok_small.id))
    empty_model = f.create_model(ModelCreateRequest(
        config=_tiny_model("m27-empty", 640, seed=27)))[0].id
    f.run_training(TrainingConfig(
        name="m27-empty-train", method="continued_pretraining",
        model_id=empty_model, dataset_id=env.ds,
        tokenizer_id=env.tok_big.id, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, steps=4, eval_every_steps=2, keep_best=False,
        seed=27))
    empty_ck = f.list_checkpoints(empty_model)[0].checkpoint_id
    env._m27 = (s_a1, s_a2, s_b1, s_small, empty_model, empty_ck)
    return env._m27


def test_m27_engine_grouping_parity_order_verbatim(env):
    s_a1, s_a2, s_b1, s_small, empty_model, empty_ck = _m27_state(env)
    f = env.forge
    for model_id, ck in ((env.big_model, env.big_ckpts[0]),
                         (env.big_model, env.big_ckpts[1]),
                         (env.small_model, env.small_ckpt)):
        listing = f.list_samples(model_id)
        got = f.list_samples_for_checkpoint(model_id, ck)
        # parity with the authoritative M15 listing filtered by the
        # persisted checkpoint identity; deterministic (created_at,
        # sample_id) order; membership from the persisted field only
        assert got == [r for r in listing if r.checkpoint_id == ck]
        keyed = [(r.created_at, r.sample_id) for r in got]
        assert keyed == sorted(keyed)
        assert all(r.checkpoint_id == ck and r.model_id == model_id
                   for r in got)
        ids = [r.sample_id for r in got]
        assert len(ids) == len(set(ids))
        # verbatim payload parity with the M15 single-record getter
        for r in got:
            assert r == f.get_sample(model_id, r.sample_id)
    # the M27-created samples sit under exactly their own checkpoints
    ck0_ids = [r.sample_id for r in f.list_samples_for_checkpoint(
        env.big_model, env.big_ckpts[0])]
    ck1_ids = [r.sample_id for r in f.list_samples_for_checkpoint(
        env.big_model, env.big_ckpts[1])]
    assert ck0_ids.count(s_a1.sample_id) == 1
    assert ck0_ids.count(s_a2.sample_id) == 1
    assert ck1_ids.count(s_b1.sample_id) == 1
    assert s_b1.sample_id not in ck0_ids and s_a1.sample_id not in ck1_ids
    assert [r.sample_id for r in f.list_samples_for_checkpoint(
        env.small_model, env.small_ckpt)].count(s_small.sample_id) == 1


def test_m27_engine_empty_404s_cross_model_read_only(env):
    s_a1, s_a2, s_b1, s_small, empty_model, empty_ck = _m27_state(env)
    f = env.forge
    # valid registered checkpoint with no samples -> []
    assert f.list_samples_for_checkpoint(empty_model, empty_ck) == []
    # cross-model: neither model can resolve the other's checkpoint id
    with pytest.raises(FileNotFoundError):
        f.list_samples_for_checkpoint(env.big_model, env.small_ckpt)
    with pytest.raises(FileNotFoundError):
        f.list_samples_for_checkpoint(env.small_model, env.big_ckpts[0])
    # unknown model / unknown checkpoint -> FileNotFoundError (404 at API)
    with pytest.raises(FileNotFoundError):
        f.list_samples_for_checkpoint("ghost-model-27", env.big_ckpts[0])
    with pytest.raises(FileNotFoundError):
        f.list_samples_for_checkpoint(env.big_model, "ghost-ck-27")
    # read-only: the filter never writes sample manifests
    root = f.storage.root / "samples"
    before = {p.relative_to(root).as_posix()
              for p in root.rglob("*") if p.is_file()}
    f.list_samples_for_checkpoint(env.big_model, env.big_ckpts[0])
    f.list_samples_for_checkpoint(empty_model, empty_ck)
    f.list_samples_for_checkpoint(env.small_model, env.small_ckpt)
    after = {p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_m27_engine_repeated_calls_identical(env):
    s_a1, s_a2, s_b1, s_small, empty_model, empty_ck = _m27_state(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_samples_for_checkpoint(env.big_model,
                                           env.big_ckpts[0])]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_samples_for_checkpoint(env.big_model,
                                               env.big_ckpts[0])]
        assert again == first


def test_m27_api_by_checkpoint_grouping_determinism_and_empty(api_client):
    h = _http_env(api_client, "m27")
    ckpts = api_client.get(f"{MODELS}/{h['mid']}/checkpoints").json()
    other_ck = [c["checkpoint_id"] for c in ckpts
                if c["checkpoint_id"] != h["ckpt"]][0]
    g1 = api_client.post(GENERATE, json=_gen_body(h)).json()
    g2 = api_client.post(GENERATE, json=_gen_body(
        h, strategy="temperature", temperature=0.8, seed=27)).json()
    url = BY_CKPT.format(mid=h["mid"], ck=h["ckpt"])
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # exactly g1 + g2 under this checkpoint, once each, in the
    # deterministic M15 order, verbatim equal to the POST responses
    assert {x["sample_id"] for x in recs} == {g1["sample_id"],
                                              g2["sample_id"]}
    assert len(recs) == 2
    keyed = [(x["created_at"], x["sample_id"]) for x in recs]
    assert keyed == sorted(keyed)
    by_id = {x["sample_id"]: x for x in recs}
    assert by_id[g1["sample_id"]] == g1
    assert by_id[g2["sample_id"]] == g2
    # parity with the existing M15 listing filtered by the persisted
    # checkpoint identity
    listing = api_client.get(f"{MODELS}/{h['mid']}/samples").json()
    assert recs == [x for x in listing
                    if x["checkpoint_id"] == h["ckpt"]]
    # the model's OTHER valid checkpoint has no samples -> 200 + []
    empty = api_client.get(BY_CKPT.format(mid=h["mid"], ck=other_ck))
    assert empty.status_code == 200 and empty.json() == []
    # repeated GET returns identical raw bytes (3 repeats)
    raw1 = api_client.get(url).content
    raw2 = api_client.get(url).content
    raw3 = api_client.get(url).content
    assert raw1 == raw2 == raw3 and json.loads(raw1) == recs
    # the generic sample detail getter is not shadowed
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/{g1['sample_id']}").json() == g1


def test_m27_api_404s_isolation_regressions_openapi(api_client):
    h = _http_env(api_client, "m27e")
    # unknown model / unknown checkpoint -> 404
    assert api_client.get(BY_CKPT.format(mid="ghost-model-27",
                                         ck=h["ckpt"])).status_code == 404
    assert api_client.get(BY_CKPT.format(mid=h["mid"],
                                         ck="ghost-ck-27")).status_code == 404
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/by-checkpoint/ck%20id%2027!!") \
        .status_code == 404
    # cross-model isolation: another real model + this model's real
    # checkpoint id -> 404 (checkpoint ids are model-scoped through the
    # M3 registry); each model sees only its own (empty) history
    h2 = _http_env(api_client, "m27iso")
    assert api_client.get(BY_CKPT.format(mid=h2["mid"],
                                         ck=h["ckpt"])).status_code == 404
    assert api_client.get(BY_CKPT.format(mid=h["mid"],
                                         ck=h2["ckpt"])).status_code == 404
    iso = api_client.get(BY_CKPT.format(mid=h2["mid"], ck=h2["ckpt"]))
    assert iso.status_code == 200 and iso.json() == []
    # M15 generation/listing/getter regression (unchanged behavior)
    g = api_client.post(GENERATE, json=_gen_body(h))
    assert g.status_code == 200 and g.json()["checkpoint_id"] == h["ckpt"]
    lst = api_client.get(f"{MODELS}/{h['mid']}/samples")
    assert lst.status_code == 200 and g.json() in lst.json()
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/{g.json()['sample_id']}") \
        .json() == g.json()
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/ghost-sample-27").status_code == 404
    # M20 sample-quality by-checkpoint (a DIFFERENT surface) intact
    sq = api_client.get(f"{MODELS}/{h['mid']}/sample-quality/"
                        f"by-checkpoint/{h['ckpt']}")
    assert sq.status_code == 200 and sq.json() == []
    # M26 comparisons by-checkpoint intact
    cmp_ = api_client.get(f"{MODELS}/{h['mid']}/comparisons/"
                          f"by-checkpoint/{h['ckpt']}")
    assert cmp_.status_code == 200 and cmp_.json() == []
    # OpenAPI: 77 paths, the new path exactly once, GET-only, sampling
    # tag, array of SampleRecord, registered before the generic route
    spec = api_client.get("/openapi.json").json()
    path = "/api/v1/models/{model_id}/samples/by-checkpoint/{checkpoint_id}"
    generic = "/api/v1/models/{model_id}/samples/{sample_id}"
    assert len(spec["paths"]) == 79
    assert list(spec["paths"]).count(path) == 1
    ops = spec["paths"][path]
    assert set(ops) == {"get"} and ops["get"]["tags"] == ["sampling"]
    schema = (ops["get"]["responses"]["200"]["content"]
              ["application/json"]["schema"])
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/SampleRecord"}
    assert "SampleRecord" in spec["components"]["schemas"]
    assert list(spec["paths"]).index(path) < list(spec["paths"]) \
        .index(generic)


# =========================================================================== #
# M32: read-only per-tokenizer grouping of the sample history
# =========================================================================== #

BY_TOK = "/api/v1/models/{mid}/samples/by-tokenizer/{tok}"


def _m32_state(env):
    """M32 state on top of the shared module env (cached): two samples
    under the env's ORIGINAL tokenizer, one under a fresh second
    tokenizer (m32-tok2, vocab 500 — fits the big model), plus a third
    tokenizer (m32-tok3) with ZERO samples anywhere. Returns
    (s_a, s_b, s_c, tok2, tok3).
    """
    cached = getattr(env, "_m32", None)
    if cached is not None:
        return cached
    f = env.forge
    s_a = f.generate_sample(_g(env))
    s_b = f.generate_sample(_g(env, strategy="temperature",
                                temperature=0.8, seed=32,
                                max_new_tokens=6))
    tok2 = f.train_tokenizer(
        TokenizerConfig(name="m32-tok2", vocab_size=500),
        dataset_id=env.ds).id
    s_c = f.generate_sample(_g(env, tokenizer_id=tok2))
    tok3 = f.train_tokenizer(
        TokenizerConfig(name="m32-tok3", vocab_size=450),
        dataset_id=env.ds).id
    env._m32 = (s_a, s_b, s_c, tok2, tok3)
    return env._m32


def test_m32_engine_grouping_parity_order_verbatim(env):
    s_a, s_b, s_c, tok2, tok3 = _m32_state(env)
    f = env.forge
    tok1 = env.tok_big.id
    for model_id, tok in ((env.big_model, tok1),
                          (env.big_model, tok2),
                          (env.small_model, env.tok_small.id)):
        listing = f.list_samples(model_id)
        got = f.list_samples_for_tokenizer(model_id, tok)
        # parity with the authoritative M15 listing filtered by the
        # persisted sample tokenizer identity; deterministic
        # (created_at, sample_id) order; membership from the persisted
        # field only; each sample EXACTLY ONCE
        assert got == [r for r in listing if r.tokenizer_id == tok]
        keyed = [(r.created_at, r.sample_id) for r in got]
        assert keyed == sorted(keyed)
        ids = [r.sample_id for r in got]
        assert len(ids) == len(set(ids))
        assert all(r.tokenizer_id == tok and r.model_id == model_id
                   for r in got)
        # verbatim payload parity with the M15 single-record getter
        for r in got:
            assert r == f.get_sample(model_id, r.sample_id)
    # tok2 holds exactly the one sample generated with it; the
    # persisted tokenizer identity travels VERBATIM (the paired
    # tokenizer_hash audit field is preserved, never re-derived)
    got2 = f.list_samples_for_tokenizer(env.big_model, tok2)
    assert [r.sample_id for r in got2] == [s_c.sample_id]
    assert all(r.tokenizer_id == tok2 for r in got2)
    # explicit partition: groups over EVERY tokenizer that actually
    # has samples under the big model are pairwise disjoint and cover
    # the full listing exactly once (other tests may add samples with
    # the small tokenizer — discovery, not fixed pairs)
    listing_ids = {r.sample_id for r in f.list_samples(env.big_model)}
    groups = {t: {r.sample_id for r in
                  f.list_samples_for_tokenizer(env.big_model, t)}
              for t in {r.tokenizer_id
                        for r in f.list_samples(env.big_model)}}
    flat = [i for g in groups.values() for i in g]
    assert set(flat) == listing_ids
    assert len(flat) == len(set(flat)) == len(listing_ids)
    assert tok1 in groups and {s_a.sample_id, s_b.sample_id} <= groups[tok1]
    assert groups[tok2] == {s_c.sample_id}


def test_m32_engine_empty_404s_cross_model_read_only(env):
    s_a, s_b, s_c, tok2, tok3 = _m32_state(env)
    f = env.forge
    # fresh tokenizer with zero samples anywhere -> []
    assert f.list_samples_for_tokenizer(env.big_model, tok3) == []
    # model-scoped empty: tok2 is globally valid (it generated one of
    # the big model's samples) but the small model has none with it
    assert f.list_samples_for_tokenizer(env.small_model, tok2) == []
    # cross-model isolation: the two listings are disjoint and no
    # model's group ever contains the other's sample ids
    small_ids = {r.sample_id
                 for r in f.list_samples(env.small_model)}
    big_ids = {r.sample_id for r in f.list_samples(env.big_model)}
    assert small_ids.isdisjoint(big_ids)
    for tok in (tok2, tok3, env.tok_big.id, env.tok_small.id):
        big_group = {r.sample_id for r in
                     f.list_samples_for_tokenizer(env.big_model, tok)}
        small_group = {r.sample_id for r in
                       f.list_samples_for_tokenizer(env.small_model, tok)}
        assert big_group <= big_ids and small_group <= small_ids
        assert big_group.isdisjoint(small_group)
    # unknown model / unknown tokenizer -> FileNotFoundError (404 at API)
    with pytest.raises(FileNotFoundError):
        f.list_samples_for_tokenizer("ghost-model-32", env.tok_big.id)
    with pytest.raises(FileNotFoundError):
        f.list_samples_for_tokenizer(env.big_model, "ghost-tok-32")
    # read-only: the filter never writes sample manifests
    root = f.storage.root / "samples"
    before = {p.relative_to(root).as_posix()
              for p in root.rglob("*") if p.is_file()}
    f.list_samples_for_tokenizer(env.big_model, env.tok_big.id)
    f.list_samples_for_tokenizer(env.big_model, tok2)
    f.list_samples_for_tokenizer(env.big_model, tok3)
    f.list_samples_for_tokenizer(env.small_model, env.tok_small.id)
    after = {p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_m32_engine_repeated_calls_identical(env):
    s_a, s_b, s_c, tok2, tok3 = _m32_state(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_samples_for_tokenizer(env.big_model,
                                          env.tok_big.id)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_samples_for_tokenizer(env.big_model,
                                              env.tok_big.id)]
        assert again == first


def _train_extra_tokenizer(api_client, tag: str, ds_id: str,
                           vocab: int) -> str:
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": json.dumps(
                             {"name": f"api15-{tag}-tok",
                              "vocab_size": vocab}),
                               "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    return tr.json()["tokenizer"]["id"]


def test_m32_api_by_tokenizer_grouping_partition_determinism(api_client):
    h = _http_env(api_client, "m32a")
    # a fresh dataset hosts two extra tokenizers: tok2 (with samples)
    # and tok3 (zero samples anywhere); both fit the model vocab 640
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", ("m32a2.txt", _corpus(120, "m32a2"),
                                           "text/plain"))],
                         data={"name": "api15-m32a2-ds"})
    assert up.status_code == 201, up.text
    ds2 = up.json()["dataset_id"]
    tok2 = _train_extra_tokenizer(api_client, "m32a2", ds2, 500)
    tok3 = _train_extra_tokenizer(api_client, "m32a3", ds2, 450)

    g1 = api_client.post(GENERATE, json=_gen_body(h)).json()
    g2 = api_client.post(GENERATE, json=_gen_body(
        h, strategy="temperature", temperature=0.8, seed=7)).json()
    g3 = api_client.post(GENERATE, json=_gen_body(
        h, tokenizer_id=tok2)).json()
    g4 = api_client.post(GENERATE, json=_gen_body(
        h, tokenizer_id=tok2, strategy="temperature", temperature=0.8,
        seed=13)).json()

    url = BY_TOK.format(mid=h["mid"], tok=h["tok"])
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # authoritative-filter parity: exact subset of the M15 listing
    # whose persisted tokenizer_id matches, in the same order
    listing = api_client.get(f"{MODELS}/{h['mid']}/samples").json()
    assert recs == [x for x in listing
                    if x["tokenizer_id"] == h["tok"]]
    assert {x["sample_id"] for x in recs} == {g1["sample_id"],
                                              g2["sample_id"]}
    keyed = [(x["created_at"], x["sample_id"]) for x in recs]
    assert keyed == sorted(keyed)
    # verbatim: each element is byte-equal to its detail-getter payload
    by_id = {x["sample_id"]: x for x in recs}
    assert by_id[g1["sample_id"]] == g1
    assert by_id[g2["sample_id"]] == g2
    for x in recs:
        one = api_client.get(
            f"{MODELS}/{h['mid']}/samples/{x['sample_id']}")
        assert one.status_code == 200 and one.json() == x
    # deterministic: three repeats return identical raw bytes
    raws = {api_client.get(url).content for _ in range(3)}
    assert len(raws) == 1
    # partition: tok2 holds exactly its own two samples, disjoint from
    # the original tokenizer's group, together the full listing
    recs2 = api_client.get(BY_TOK.format(mid=h["mid"], tok=tok2)).json()
    assert {x["sample_id"] for x in recs2} == {g3["sample_id"],
                                               g4["sample_id"]}
    ids1 = {x["sample_id"] for x in recs}
    ids2 = {x["sample_id"] for x in recs2}
    assert ids1.isdisjoint(ids2)
    assert ids1 | ids2 == {x["sample_id"] for x in listing}
    # valid tokenizer with zero samples for the model -> [] (200)
    empty = api_client.get(BY_TOK.format(mid=h["mid"], tok=tok3))
    assert empty.status_code == 200 and empty.json() == []
    # no side effects: the filter itself added no records
    assert len(api_client.get(
        f"{MODELS}/{h['mid']}/samples").json()) == 4


def test_m32_api_404s_isolation_regressions_openapi(api_client):
    h = _http_env(api_client, "m32b")
    h2 = _http_env(api_client, "m32c")     # second real model
    g1 = api_client.post(GENERATE, json=_gen_body(h)).json()

    # 404s: unknown model / unknown tokenizer (two ghost forms)
    assert api_client.get(BY_TOK.format(mid="ghost-model-32",
                                        tok=h["tok"])).status_code == 404
    assert api_client.get(BY_TOK.format(mid=h["mid"],
                                        tok="ghost-tok-32")).status_code == 404
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/by-tokenizer/"
        "m32--not-a-real-tokenizer-id").status_code == 404
    assert api_client.get(BY_TOK.format(mid="ghost-model-32",
                                        tok="ghost-tok-32")).status_code == 404

    # cross-model isolation: tokenizers are global, but the other
    # model's group is empty (scoping from the model's own listing)
    iso = api_client.get(BY_TOK.format(mid=h2["mid"], tok=h["tok"]))
    assert iso.status_code == 200 and iso.json() == []

    # M15 listing/getter intact
    listing = api_client.get(f"{MODELS}/{h['mid']}/samples").json()
    assert [x["sample_id"] for x in listing] == [g1["sample_id"]]
    got = api_client.get(f"{MODELS}/{h['mid']}/samples/{g1['sample_id']}")
    assert got.status_code == 200 and got.json() == g1

    # M27 by-checkpoint regression: same listing, other grouping
    byck = api_client.get(f"{MODELS}/{h['mid']}/samples/by-checkpoint/"
                          f"{h['ckpt']}").json()
    assert [x["sample_id"] for x in byck] == [g1["sample_id"]]
    # generic detail getter still 404s ghost ids (no route capture)
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/ghost-sample-32").status_code == 404

    # M30 evaluation-by-tokenizer regression (training produced evals
    # with h's tokenizer): parity with the M4 listing filtered
    evs = api_client.get(f"{MODELS}/{h['mid']}/evaluations").json()
    evt = api_client.get(f"{MODELS}/{h['mid']}/evaluations/by-tokenizer/"
                         f"{h['tok']}").json()
    assert evt == [e for e in evs if e["tokenizer_id"] == h["tok"]]
    # M31 comparison-by-tokenizer regression: route intact (no
    # comparisons exist for this fresh model -> 200 + [])
    cmp_t = api_client.get(f"{MODELS}/{h['mid']}/comparisons/by-tokenizer/"
                           f"{h['tok']}")
    assert cmp_t.status_code == 200 and cmp_t.json() == []

    # OpenAPI: 64 paths, the new path exactly once, GET-only, tag
    # sampling, SampleRecord items; route order M27 by-checkpoint <
    # by-tokenizer < generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer) + 1 (M32 samples by-tokenizer)
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
    # = 79
    assert len(spec["paths"]) == 79
    path = ("/api/v1/models/{model_id}/samples/by-tokenizer"
            "/{tokenizer_id}")
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["sampling"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/SampleRecord"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/samples/by-checkpoint/"
                      "{checkpoint_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/samples/{sample_id}")


# --------------------------------------------------------------------------- #
# M40: sample history by strategy (read-only grouping, enum contract)
# --------------------------------------------------------------------------- #

def _m40_state(env):
    """M40 state on top of the shared module env (cached): BOTH
    strategies on the big model (greedy x1 + temperature x1 — earlier
    module tests' samples may not exist when tests are deselected),
    plus a fresh model with NO samples at all (the natural valid-empty
    case for both strategies). Returns (s_greedy, s_temp, empty_model).
    """
    cached = getattr(env, "_m40", None)
    if cached is not None:
        return cached
    f = env.forge
    s_greedy = f.generate_sample(_g(env))
    s_temp = f.generate_sample(_g(env, strategy="temperature",
                                  temperature=0.8, seed=40,
                                  max_new_tokens=6))
    empty_model = f.create_model(ModelCreateRequest(
        config=_tiny_model("m40-empty", 300, seed=41)))[0].id
    env._m40 = (s_greedy, s_temp, empty_model)
    return env._m40


def test_m40_engine_filters_by_persisted_strategy_identity(env):
    s_greedy, s_temp, empty_model = _m40_state(env)
    f = env.forge
    assert s_greedy.strategy == SampleStrategy.GREEDY
    assert s_temp.strategy == SampleStrategy.TEMPERATURE
    listing = f.list_samples(env.big_model)
    for strategy in (SampleStrategy.GREEDY, SampleStrategy.TEMPERATURE):
        got = f.list_samples_for_strategy(env.big_model, strategy)
        # parity with the authoritative M15 listing filtered by the
        # persisted strategy; deterministic (created_at, sample_id)
        # order; membership from the persisted field only
        assert got == [r for r in listing if r.strategy == strategy]
        keyed = [(r.created_at, r.sample_id) for r in got]
        assert keyed == sorted(keyed)
        ids = [r.sample_id for r in got]
        assert len(ids) == len(set(ids))
        assert all(r.strategy == strategy
                   and r.model_id == env.big_model for r in got)
        # verbatim payload parity with the M15 single-record getter
        for r in got:
            assert r == f.get_sample(env.big_model, r.sample_id)
    # the fixture's records land in their own groups with the persisted
    # strategy VERBATIM (earlier module tests may have added other
    # samples — membership derives from the listing); greedy records
    # persist temperature=None (the strategy is never inferred from it)
    got_g = f.list_samples_for_strategy(env.big_model,
                                        SampleStrategy.GREEDY)
    got_t = f.list_samples_for_strategy(env.big_model,
                                        SampleStrategy.TEMPERATURE)
    assert s_greedy.sample_id in {r.sample_id for r in got_g}
    assert s_temp.sample_id in {r.sample_id for r in got_t}
    assert all(r.temperature is None for r in got_g)
    # explicit partition: disjoint groups over BOTH enum values whose
    # union is the full listing
    ids_g = {r.sample_id for r in got_g}
    ids_t = {r.sample_id for r in got_t}
    assert ids_g and ids_t
    assert ids_g.isdisjoint(ids_t)
    assert ids_g | ids_t == {r.sample_id for r in listing}


def test_m40_engine_empty_404s_model_scoping_read_only(env):
    s_greedy, s_temp, empty_model = _m40_state(env)
    f = env.forge
    # a model whose ENTIRE M15 listing is empty -> [] for BOTH enum
    # values (the fixture created it with no samples at all — never a
    # 404)
    assert f.list_samples(empty_model) == []
    for strategy in (SampleStrategy.GREEDY, SampleStrategy.TEMPERATURE):
        assert f.list_samples_for_strategy(empty_model, strategy) == []
    # unknown model -> FileNotFoundError (404 at the API); the enum
    # itself needs NO registry lookup (unsupported values are 422 at
    # the API boundary and never reach the engine)
    with pytest.raises(FileNotFoundError):
        f.list_samples_for_strategy("ghost-model-40",
                                    SampleStrategy.GREEDY)
    # cross-model isolation: each model's group is a subset of its own
    # listing and the two listings are disjoint
    big_ids = {r.sample_id for r in f.list_samples(env.big_model)}
    small_ids = {r.sample_id for r in f.list_samples(env.small_model)}
    assert big_ids.isdisjoint(small_ids)
    for strategy in (SampleStrategy.GREEDY, SampleStrategy.TEMPERATURE):
        for mid, ids in ((env.big_model, big_ids),
                         (env.small_model, small_ids)):
            group = {r.sample_id for r in
                     f.list_samples_for_strategy(mid, strategy)}
            assert group <= ids
    # read-only: the filter never writes sample manifests (the samples
    # root is shared across models)
    root = f.storage.root / "samples"
    before = {p.relative_to(root).as_posix()
              for p in root.rglob("*") if p.is_file()}
    for strategy in (SampleStrategy.GREEDY, SampleStrategy.TEMPERATURE):
        f.list_samples_for_strategy(env.big_model, strategy)
        f.list_samples_for_strategy(env.small_model, strategy)
        f.list_samples_for_strategy(empty_model, strategy)
    after = {p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_m40_engine_repeated_calls_identical(env):
    s_greedy, s_temp, empty_model = _m40_state(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_samples_for_strategy(env.big_model,
                                         SampleStrategy.GREEDY)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_samples_for_strategy(env.big_model,
                                             SampleStrategy.GREEDY)]
        assert again == first


# --------------------------------------------------------------------------- #
# M40 API: sample history by strategy (read-only grouping, enum contract)
# --------------------------------------------------------------------------- #

def test_m40_api_by_strategy_grouping_partition_determinism(api_client):
    h = _http_env(api_client, "m40a")
    # one record per strategy on each side: greedy x2 (different
    # prompts) + temperature x1
    s_g1 = api_client.post(GENERATE, json=_gen_body(h)).json()
    s_g2 = api_client.post(
        GENERATE, json=_gen_body(h, prompt=PROMPT_API + " valley")).json()
    s_t1 = api_client.post(
        GENERATE, json=_gen_body(h, strategy="temperature",
                                 temperature=0.8, seed=40)).json()
    assert {s_g1["strategy"], s_g2["strategy"]} == {"greedy"}
    assert s_t1["strategy"] == "temperature"

    listing = api_client.get(f"{MODELS}/{h['mid']}/samples").json()
    assert len(listing) == 3
    url = f"{MODELS}/{h['mid']}/samples/by-strategy"
    got = api_client.get(f"{url}/greedy")
    assert got.status_code == 200, got.text
    recs = got.json()
    # authoritative-filter parity: exact subset of the M15 listing
    # whose persisted strategy matches, in the same order
    assert recs == [r for r in listing if r["strategy"] == "greedy"]
    assert {r["sample_id"] for r in recs} == \
        {s_g1["sample_id"], s_g2["sample_id"]}
    keyed = [(r["created_at"], r["sample_id"]) for r in recs]
    assert keyed == sorted(keyed)
    # verbatim: each element equals its detail-getter payload (prompt,
    # token ids, output text, temperature included)
    for r in recs:
        one = api_client.get(
            f"{MODELS}/{h['mid']}/samples/{r['sample_id']}")
        assert one.status_code == 200 and one.json() == r
    # deterministic: three repeats return identical raw bytes
    for strategy in ("greedy", "temperature"):
        raws = {api_client.get(f"{url}/{strategy}").content
                for _ in range(3)}
        assert len(raws) == 1
    # partition: the two groups are disjoint and cover the listing
    recs_t = api_client.get(f"{url}/temperature").json()
    assert recs_t == [r for r in listing
                      if r["strategy"] == "temperature"]
    assert [r["sample_id"] for r in recs_t] == [s_t1["sample_id"]]
    ids_g = {r["sample_id"] for r in recs}
    ids_t = {r["sample_id"] for r in recs_t}
    assert ids_g.isdisjoint(ids_t)
    assert ids_g | ids_t == {r["sample_id"] for r in listing}
    # no execution side effects: the filter itself added no records
    assert len(api_client.get(
        f"{MODELS}/{h['mid']}/samples").json()) == 3


def test_m40_api_by_strategy_404_422s_isolation_regressions_openapi(
        api_client):
    h = _http_env(api_client, "m40b")
    other = _http_env(api_client, "m40c")
    s = api_client.post(GENERATE, json=_gen_body(h)).json()
    url = f"{MODELS}/{h['mid']}/samples/by-strategy/"

    # 404: unknown model with a VALID strategy (exactly like the
    # sibling groupings)
    assert api_client.get(
        f"{MODELS}/ghost-model-40/samples/by-strategy/greedy"
    ).status_code == 404
    # 422: unsupported strategy values are rejected by the schema enum
    # at the API boundary — before the handler, so the 422 wins even
    # for an UNKNOWN model (never a registry-style 404, never [])
    for bad in ("GREEDY", "temp%20erature", "3", "top_k"):
        got = api_client.get(url + bad)
        assert got.status_code == 422, (bad, got.status_code)
    assert api_client.get(
        f"{MODELS}/ghost-model-40/samples/by-strategy/top_k"
    ).status_code == 422

    # cross-model isolation: the other model's group is empty under
    # BOTH strategies (scoping from the model's own listing)
    for strategy in ("greedy", "temperature"):
        iso = api_client.get(
            f"{MODELS}/{other['mid']}/samples/by-strategy/{strategy}")
        assert iso.status_code == 200 and iso.json() == []

    # M15 listing/getter intact; the generic detail getter still 404s
    # ghost ids (no route capture by the new literal segment)
    listing = api_client.get(f"{MODELS}/{h['mid']}/samples").json()
    assert [r["sample_id"] for r in listing] == [s["sample_id"]]
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/{s['sample_id']}").json() == s
    assert api_client.get(
        f"{MODELS}/{h['mid']}/samples/ghost-sample-40"
    ).status_code == 404

    # M27 by-checkpoint / M32 by-tokenizer regressions: same listing,
    # other groupings, unconfused (listing-derived parity)
    byck = api_client.get(
        f"{MODELS}/{h['mid']}/samples/by-checkpoint/{h['ckpt']}")
    assert byck.status_code == 200
    assert byck.json() == [r for r in listing
                           if r["checkpoint_id"] == h["ckpt"]]
    bytk = api_client.get(
        f"{MODELS}/{h['mid']}/samples/by-tokenizer/{h['tok']}")
    assert bytk.status_code == 200
    assert bytk.json() == [r for r in listing
                           if r["tokenizer_id"] == h["tok"]]
    # M38 evaluations-by-state-kind regression: this harness's
    # training run persists no M4 evaluation records, so both kinds
    # are the natural valid-empty case with listing parity
    evs = api_client.get(f"{MODELS}/{h['mid']}/evaluations").json()
    assert evs == []
    for kind in ("current", "checkpoint"):
        g = api_client.get(
            f"{MODELS}/{h['mid']}/evaluations/by-state-kind/{kind}")
        assert g.status_code == 200
        assert g.json() == [e for e in evs if e["state_kind"] == kind]
    # M39 comparisons-by-verdict regression: the model has no
    # comparisons, so every verdict group is the natural valid empty
    comps = api_client.get(f"{MODELS}/{h['mid']}/comparisons").json()
    assert comps == []
    for v in ("improved", "regressed", "unchanged"):
        g = api_client.get(
            f"{MODELS}/{h['mid']}/comparisons/by-verdict/{v}")
        assert g.status_code == 200 and g.json() == []

    # OpenAPI: 77 paths, the new path exactly once, GET-only, tag
    # sampling, SampleRecord items, strategy $ref SampleStrategy; route
    # order M32 by-tokenizer < by-strategy < generic detail
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
    # + 1 (M47 evaluations by-truncated) = 79
    assert len(spec["paths"]) == 79
    path = "/api/v1/models/{model_id}/samples/by-strategy/{strategy}"
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["sampling"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/SampleRecord"}
    strategy_param = [p for p in item["get"]["parameters"]
                      if p["name"] == "strategy"][0]
    assert strategy_param["schema"] == {
        "$ref": "#/components/schemas/SampleStrategy"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/samples/by-tokenizer/"
                      "{tokenizer_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/samples/{sample_id}")
