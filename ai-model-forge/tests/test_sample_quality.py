"""Milestone 16 tests: per-sample quality measurement of generated text.

The engine receives ONLY ``model_id`` + ``sample_id``; every other input is
resolved from the immutable M15 sample manifest (recorded checkpoint +
tokenizer, verified by content hash) — nothing is auto-selected or supplied
separately. The full prompt + generated token sequence is scored in ONE
context window under the existing M4 causal-LM objective over the GENERATED
continuation targets only (prompt tokens condition but never count; the
first generated token is conditioned on the full prompt; every generated
target counted exactly once; overlong samples rejected 422, never silently
truncated). Metrics: loss_nats + perplexity = exp(min(loss, 100)).

Covered (engine): resolution 404s (unknown model/sample, sample of another
model); corrupt manifests (unreadable JSON, sample_id/model_id mismatch,
result_hash non-reproduction, recorded checkpoint/tokenizer hash mismatch)
-> 409-class integrity errors; invalid measurement state (window overflow,
vocabulary violation, count/id-range inconsistencies) -> 422-class
ValueErrors — all with zero writes; independent metric correctness
(recomputed per-position causal CE over exactly the generated targets,
fp32 tolerance + M4 rounding); exact boundary (window == context) vs one
token beyond (422); determinism (identical metrics + result_hash across
distinct evaluation records, sample manifest untouched); result-hash
semantics (semantic inputs participate; evaluation id/timestamps/duration/
hardware/paths do not); storage discipline (exactly one manifest per
measurement under sample-evaluations/, no auxiliary files, no .tmp, all
pre-existing files byte-identical); read-only deterministic list/get;
no_grad measurement through the existing forward path. Covered (HTTP):
POST /models/{id}/samples/{sid}/quality (body-less), deterministic list
and get, 404/409/422 behavior with zero growth, 405 on edit/delete,
OpenAPI exposure. M1–M15 tests are untouched.
"""
from __future__ import annotations

import json
import math
import random
import shutil

import pytest
import torch

from app.sample_quality import SampleQualityEngine
from app.sampling import SamplingEngine
from app.schemas import (
    ModelCreateRequest,
    SampleGenerateRequest,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
)

_WORDS = ("river mountain cloud forest desert ocean valley island meadow "
          "canyon table chair lamp desk shelf couch rug clock mirror "
          "vase").split()
PROMPT = "river mountain cloud forest ocean desert valley island"


def _corpus(n: int, tag: str) -> bytes:
    rng = random.Random(hash(tag) & 0xFFFF)
    lines = []
    for i in range(n):
        k = rng.randint(10, 22)
        lines.append(" ".join(rng.choices(_WORDS, k=k)) + f" number {i}")
    return ("\n\n".join(lines) + "\n").encode("utf-8")


def _tiny_model(name: str, vocab: int, seed: int = 1,
                ctx: int = 64) -> TransformerConfig:
    return TransformerConfig(name=name, vocab_size=vocab,
                             context_length=ctx, hidden_size=64,
                             n_layers=2, n_heads=4, n_kv_heads=2,
                             intermediate_size=128, seed=seed)


class Env:
    """One temp root: dataset, big + small tokenizers, trained models."""

    def __init__(self, root):
        from app.engine import ModelForge

        self.forge = ModelForge(root=root)
        self.samples = SamplingEngine(self.forge.storage)
        self.sq = SampleQualityEngine(self.forge.storage)

    def prepare(self):
        f = self.forge
        up = f.upload_dataset([("a.txt", _corpus(220, "m16-dom"))],
                              name="m16-dom")
        self.ds = up["dataset_id"]
        self.tok_big = f.train_tokenizer(
            TokenizerConfig(name="m16-tok-big", vocab_size=600),
            dataset_id=self.ds)
        self.tok_small = f.train_tokenizer(
            TokenizerConfig(name="m16-tok-small", vocab_size=300),
            dataset_id=self.ds)
        assert self.tok_big.actual_vocab_size > 300
        assert self.tok_small.actual_vocab_size <= 300
        for tok in (self.tok_big, self.tok_small):
            f.tokenize_dataset(self.ds, tok.id)
        self.big_model = f.create_model(ModelCreateRequest(
            config=_tiny_model("m16-main", 640)))[0].id
        rep = f.run_training(TrainingConfig(
            name="m16-train-big", method="continued_pretraining",
            model_id=self.big_model, dataset_id=self.ds,
            tokenizer_id=self.tok_big.id, learning_rate=3e-3, batch_size=8,
            max_seq_len=32, steps=8, eval_every_steps=4, keep_best=False,
            seed=3))
        self.big_ckpts = [c["checkpoint_id"] for c in rep.checkpoints]
        assert len(self.big_ckpts) == 2
        self.small_model = f.create_model(ModelCreateRequest(
            config=_tiny_model("m16-small", 300, seed=7)))[0].id
        rep = f.run_training(TrainingConfig(
            name="m16-train-small", method="continued_pretraining",
            model_id=self.small_model, dataset_id=self.ds,
            tokenizer_id=self.tok_small.id, learning_rate=3e-3,
            batch_size=8, max_seq_len=32, steps=4, eval_every_steps=4,
            keep_best=False, seed=5))
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
    e = Env(tmp_path_factory.mktemp("m16-root"))
    e.prepare()
    return e


def _g(env, **overrides) -> SampleGenerateRequest:
    base = dict(model_id=env.big_model, checkpoint_id=env.big_ckpts[0],
                tokenizer_id=env.tok_big.id, prompt=PROMPT,
                strategy="greedy", max_new_tokens=8)
    base.update(overrides)
    return SampleGenerateRequest(**base)


def _make_sample(env, **overrides):
    return env.samples.run(_g(env, **overrides))


def _manifest(env, model_id: str, sample_id: str) -> dict:
    root = env.forge.storage.root
    p = root / "samples" / model_id / f"sample-{sample_id}" / "manifest.json"
    return json.loads(p.read_text())


def _craft(env, model_id: str, sample_id: str, source: str, mutate,
           keep_ids: bool = True) -> dict:
    """Fabricate a sample manifest under samples/<model>/sample-<id> starting
    from the dict of the REAL sample ``source`` (so checkpoint/tokenizer
    identity fields stay genuine). ``keep_ids`` rewrites the identity fields
    to the fabricated location. The caller removes the dir afterwards."""
    d = dict(_manifest(env, model_id, source))
    if callable(mutate):
        d = mutate(d)
    else:
        d.update(mutate)
    if keep_ids:
        d["sample_id"] = sample_id
        d["model_id"] = model_id
    root = env.forge.storage.root
    sdir = root / "samples" / model_id / f"sample-{sample_id}"
    sdir.mkdir(parents=True, exist_ok=False)
    (sdir / "manifest.json").write_text(json.dumps(d))
    return d


def _rm_sample(env, model_id: str, sample_id: str) -> None:
    root = env.forge.storage.root
    shutil.rmtree(root / "samples" / model_id / f"sample-{sample_id}")


# =========================================================================== #
# Metric semantics: what exactly is measured
# =========================================================================== #

def test_measure_greedy_sample_exact_fields_and_one_manifest(env):
    s = _make_sample(env)
    before = env.file_snapshot()
    r = env.sq.run(env.big_model, s.sample_id)
    after = env.file_snapshot()
    new = [k for k in after if k not in before]
    assert len(new) == 1 and new[0] == (
        f"sample-evaluations/{env.big_model}/"
        f"evaluation-{r.evaluation_id}/manifest.json")
    assert r.sample_id == s.sample_id
    assert r.sample_result_hash == s.result_hash
    assert r.checkpoint_id == s.checkpoint_id == env.big_ckpts[0]
    assert r.checkpoint_weights_sha256 == env.ckpt_sha(env.big_model,
                                                       env.big_ckpts[0])
    assert r.tokenizer_id == env.tok_big.id
    assert r.tokenizer_hash == env.tok_big.tokenizer_hash
    # exact target accounting: generated targets scored once; the prompt is
    # conditioning only and never counts as a generated-text target
    p = s.prompt_token_count
    assert p > 0 and r.prompt_token_count == p
    assert r.generated_token_count == s.generated_token_count == 8
    assert r.evaluated_token_count == 8 == r.generated_token_count
    assert r.window_token_count == p + 8
    assert r.context_length == 64
    assert r.window_rule == "single_window"
    # metrics sane; perplexity follows the M4 convention (computed from the
    # UNROUNDED loss, so it matches exp(min(rounded_loss, 100)) only within
    # the loss-rounding envelope ~ ppl * 5e-7)
    assert math.isfinite(r.loss_nats) and r.loss_nats >= 0.0
    assert abs(r.perplexity - math.exp(min(r.loss_nats, 100.0))) < 1e-3
    assert len(r.result_hash) == 64 and len(r.token_sequence_sha256) == 64
    # every pre-existing file byte-identical; nothing else was written
    for k, v in before.items():
        assert after[k] == v
    env.no_tmp()


def test_independent_metric_correctness_and_target_accounting(env):
    """Recompute the causal CE from first principles and verify the engine
    scores EXACTLY the generated targets (positions p_len-1 .. T-2), i.e.
    the first generated token is conditioned on the full prompt and prompt
    targets are excluded. Documented tolerance: 1e-6 (identical fp32 math
    on the CPU)."""
    import numpy as np
    import torch.nn.functional as F

    from app.model_builder import build_transformer, restore_state

    s = _make_sample(env)
    r = env.sq.run(env.big_model, s.sample_id)
    ids = s.prompt_token_ids + s.generated_token_ids
    p_len = len(s.prompt_token_ids)
    g_len = len(s.generated_token_ids)

    # independent rebuild under the sample's recorded checkpoint
    model_cfg = env.forge.storage.load_record(env.big_model).config
    state = env.forge.training.verify_checkpoint(env.big_model,
                                                 env.big_ckpts[0])
    model = build_transformer(model_cfg).module
    missing, unexpected = restore_state(model, state)
    assert not missing and not unexpected
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids], dtype=torch.long))
    logits = logits[0].float()                        # (T, V)
    tgt = torch.tensor(ids, dtype=torch.long)
    # independent path A: single batched cross-entropy over exactly the G
    # generated-target rows (the same op the engine performs) — must agree
    # bit-for-bit (identical fp32 math on the CPU)
    score = logits[p_len - 1: p_len + g_len - 1]      # (G, V)
    targets = tgt[p_len:]
    raw = float(F.cross_entropy(score, targets))      # mean over G targets
    # the engine's rounded values must reproduce EXACTLY from the
    # independently recomputed raw loss (identical fp32 ops on the CPU;
    # the record stores loss rounded to 6 decimals like M4)
    assert r.loss_nats == round(raw, 6), (r.loss_nats, raw)
    assert r.perplexity == round(float(np.exp(min(raw, 100.0))), 6)
    # independent path B: per-position CEs (reduction='none') then mean —
    # mathematically identical, allowed fp32 reduction-order tolerance 1e-5
    per_pos = F.cross_entropy(logits[:-1], tgt[1:], reduction="none")
    assert len(per_pos) == p_len + g_len - 1
    generated_targets = per_pos[p_len - 1:]           # exactly G values
    assert len(generated_targets) == g_len == 8
    expected = float(generated_targets.mean())
    assert abs(r.loss_nats - expected) < 1e-5, (r.loss_nats, expected)
    # prompt-only targets (positions 0..p_len-2) are NOT scored, and the
    # whole-window M4-style mean is not what is reported either (fixed-seed
    # env -> deterministic numbers, asserted with a wide margin)
    assert abs(r.loss_nats - float(per_pos[: p_len - 1].mean())) > 1e-3
    assert abs(r.loss_nats - float(per_pos.mean())) > 1e-3
    env.no_tmp()


def test_first_generated_token_conditioned_on_prompt(env):
    """G0 is scored from the logits of the LAST prompt position (full-prompt
    conditioning): dropping a prompt token changes G0's cross-entropy, and
    the engine's metric equals the subset mean starting exactly at p_len-1."""
    import torch.nn.functional as F

    from app.model_builder import build_transformer, restore_state

    s = _make_sample(env)
    model_cfg = env.forge.storage.load_record(env.big_model).config
    state = env.forge.training.verify_checkpoint(env.big_model,
                                                 env.big_ckpts[0])
    model = build_transformer(model_cfg).module
    restore_state(model, state)
    model.eval()
    ids = s.prompt_token_ids + s.generated_token_ids
    p_len = len(s.prompt_token_ids)
    with torch.no_grad():
        logits_full, _ = model(torch.tensor([ids], dtype=torch.long))
        logits_cut, _ = model(torch.tensor(
            [s.prompt_token_ids[:1] + s.generated_token_ids],
            dtype=torch.long))
        logits2, _ = model(torch.tensor([ids], dtype=torch.long))
    tgt0 = torch.tensor([s.generated_token_ids[0]])
    ce_full = float(F.cross_entropy(logits_full[0, p_len - 1:p_len], tgt0))
    # a 1-token prompt leaves G0 conditioned on almost nothing -> different
    ce_short = float(F.cross_entropy(logits_cut[0, 0:1], tgt0))
    assert abs(ce_full - ce_short) > 1e-3     # conditioning demonstrably acts
    allpos = F.cross_entropy(logits2[0, :-1].float(),
                             torch.tensor(ids[1:]), reduction="none")
    r = env.sq.run(env.big_model, s.sample_id)
    assert abs(r.loss_nats - float(allpos[p_len - 1:].mean())) < 1e-6
    env.no_tmp()


# =========================================================================== #
# Context-window behavior
# =========================================================================== #

def test_exact_context_boundary_measured_one_token_beyond_rejected(env):
    n = _make_sample(env, max_new_tokens=1).prompt_token_count
    fit = _make_sample(env, max_new_tokens=64 - n)   # == context exactly
    assert fit.prompt_token_count + fit.generated_token_count == 64
    r = env.sq.run(env.big_model, fit.sample_id)
    assert r.window_token_count == 64 == r.context_length
    assert r.evaluated_token_count == r.generated_token_count == 64 - n

    # one token beyond the context window: a fabricated overlong sample
    # (reachable only by editing a manifest — M15's preflight forbids it)
    # -> deterministic 422-class rejection, never silent truncation
    ids = _manifest(env, env.big_model, fit.sample_id)["generated_token_ids"]
    _craft(env, env.big_model, "overlong000001", source=fit.sample_id,
           mutate=dict(generated_token_ids=ids + [ids[-1]],
                       generated_token_count=len(ids) + 1))
    before = env.file_snapshot()          # crafted manifest is pre-existing
    with pytest.raises(ValueError, match="context window"):
        env.sq.run(env.big_model, "overlong000001")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "overlong000001")
    env.no_tmp()


# =========================================================================== #
# Resolution + integrity + measurement-state failures (zero writes)
# =========================================================================== #

def test_unknown_model_sample_and_cross_model_404(env):
    sid = _make_sample(env).sample_id
    before = env.file_snapshot()
    with pytest.raises(FileNotFoundError, match="model 'no-such-model'"):
        env.sq.run("no-such-model", sid)
    with pytest.raises(FileNotFoundError, match="sample 'no-such-sample'"):
        env.sq.run(env.big_model, "no-such-sample")
    # the same sample id under a DIFFERENT model is not reachable (each
    # sample belongs to exactly one model): 404 semantics
    with pytest.raises(FileNotFoundError, match="sample"):
        env.sq.run(env.small_model, sid)
    assert env.file_snapshot() == before
    env.no_tmp()


def test_corrupt_sample_manifest_rejected(env):
    real = _make_sample(env).sample_id
    # (a) unreadable manifest JSON
    root = env.forge.storage.root
    sdir = root / "samples" / env.big_model / "sample-garbagejson"
    sdir.mkdir(parents=True)
    (sdir / "manifest.json").write_text("not json {{{")
    before = env.file_snapshot()          # crafted file is pre-existing now
    with pytest.raises(RuntimeError, match="corrupt"):
        env.sq.run(env.big_model, "garbagejson")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "garbagejson")
    # (b) identity fields disagree with the manifest location
    _craft(env, env.big_model, "wrong-owner", source=real,
           keep_ids=False,
           mutate=dict(model_id="some-other-model", sample_id="wrong-owner"))
    before = env.file_snapshot()
    with pytest.raises(RuntimeError, match="corrupt"):
        env.sq.run(env.big_model, "wrong-owner")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "wrong-owner")
    # (c) an in-range tampered token id breaks the sample's own result_hash
    def _flip(d):
        ids = list(d["generated_token_ids"])
        x = ids[0]
        ids[0] = x - 1 if x > 0 else 1        # always within actual vocab
        return dict(d, generated_token_ids=ids)

    _craft(env, env.big_model, "tampered-id", source=real, mutate=_flip)
    before = env.file_snapshot()
    with pytest.raises(RuntimeError, match="integrity"):
        env.sq.run(env.big_model, "tampered-id")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "tampered-id")
    env.no_tmp()


def test_inconsistent_counts_and_out_of_vocab_ids_422(env):
    real = _make_sample(env).sample_id
    # generated count says 9 but the id list has 8 -> exact accounting
    # impossible (fails BEFORE any heavy verification; zero writes)
    _craft(env, env.big_model, "count-bad", source=real,
           mutate=dict(generated_token_count=9))
    before = env.file_snapshot()      # crafted file is pre-existing now
    with pytest.raises(ValueError, match="token counts do not match"):
        env.sq.run(env.big_model, "count-bad")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "count-bad")
    # a recorded id at the tokenizer's actual vocab boundary is unmeasurable
    out = env.tok_big.actual_vocab_size
    assert out < 640                 # still below the model vocab: tests
    _craft(env, env.big_model, "id-bad", source=real,
           mutate=lambda d: dict(
               d, generated_token_ids=[out] + d["generated_token_ids"][1:]))
    before = env.file_snapshot()
    with pytest.raises(ValueError, match="outside the tokenizer"):
        env.sq.run(env.big_model, "id-bad")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "id-bad")
    env.no_tmp()


def test_corrupt_checkpoint_refused(env, tmp_path):
    """A genuinely corrupt weights archive under the recorded checkpoint is
    refused by the existing M3 verification with zero writes (isolated root
    so the shared env's checkpoints stay intact)."""
    from app.engine import ModelForge

    f = ModelForge(root=tmp_path)
    # 400 records make the deterministic 90/5/5 record-bucket split
    # guaranteed non-empty (P(empty validation) ~ 0.95**400 ~ 1e-9): a
    # 60-record corpus left ~4.6% of runs with an empty validation split,
    # a latent flake fixed at its root (assertions unchanged).
    up = f.upload_dataset([("a.txt", _corpus(400, "m16-corrupt"))],
                          name="m16-corrupt-ds")
    ds = up["dataset_id"]
    tok = f.train_tokenizer(TokenizerConfig(name="t", vocab_size=256),
                            dataset_id=ds)
    f.tokenize_dataset(ds, tok.id)
    mid = f.create_model(ModelCreateRequest(config=_tiny_model(
        "m16-c", 512, seed=3, ctx=32)))[0].id
    rep = f.run_training(TrainingConfig(
        name="r", method="continued_pretraining", model_id=mid,
        dataset_id=ds, tokenizer_id=tok.id, learning_rate=3e-3,
        batch_size=8, max_seq_len=16, steps=4, eval_every_steps=4,
        keep_best=False, seed=7))
    ckpt = rep.checkpoints[0]["checkpoint_id"]
    s = f.samples.run(SampleGenerateRequest(
        model_id=mid, checkpoint_id=ckpt, tokenizer_id=tok.id,
        prompt="river mountain cloud", strategy="greedy", max_new_tokens=4))
    sq = SampleQualityEngine(f.storage)
    assert sq.run(mid, s.sample_id).perplexity > 0   # intact sanity run
    wpath = f.storage.root / "models" / mid / "checkpoints" / ckpt / \
        "weights.pt"
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    first = next(iter(state))
    tampered = {k: (v.clone().fill_(0.0) if k == first else v.clone())
                for k, v in state.items()}
    torch.save(tampered, wpath)
    before = {p.relative_to(f.storage.root).as_posix(): p.read_bytes()
              for p in f.storage.root.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="integrity"):
        sq.run(mid, s.sample_id)
    assert {p.relative_to(f.storage.root).as_posix(): p.read_bytes()
            for p in f.storage.root.rglob("*") if p.is_file()} == before
    leftovers = [p for p in f.storage.root.rglob("*")
                 if p.is_file() and ".tmp" in p.name]
    assert leftovers == []


def test_recorded_hash_mismatches_rejected(env):
    real = _make_sample(env).sample_id
    # recorded checkpoint weights hash disagrees with the verified bytes
    _craft(env, env.big_model, "ckpt-hash-bad", source=real,
           mutate=dict(checkpoint_weights_sha256="0" * 64))
    before = env.file_snapshot()      # crafted file is pre-existing now
    with pytest.raises(RuntimeError, match="checkpoint weights hash"):
        env.sq.run(env.big_model, "ckpt-hash-bad")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "ckpt-hash-bad")
    # recorded tokenizer hash disagrees with the stored tokenizer content
    _craft(env, env.big_model, "tok-hash-bad", source=real,
           mutate=dict(tokenizer_hash="0" * 64))
    before = env.file_snapshot()
    with pytest.raises(RuntimeError, match="tokenizer hash"):
        env.sq.run(env.big_model, "tok-hash-bad")
    assert env.file_snapshot() == before
    _rm_sample(env, env.big_model, "tok-hash-bad")
    env.no_tmp()


def test_vocabulary_mismatch_via_recorded_tokenizer_422(env):
    """Platform convention (M3/M4/M15): tokenizer actual vocab must fit the
    model's vocab_size. A sample that records the 300-vocab model with the
    big tokenizer is an invalid measurement state."""
    small_s = env.samples.run(SampleGenerateRequest(
        model_id=env.small_model, checkpoint_id=env.small_ckpt,
        tokenizer_id=env.tok_small.id, prompt=PROMPT, strategy="greedy",
        max_new_tokens=4))
    _craft(env, env.small_model, "vocab-bad", source=small_s.sample_id,
           mutate=dict(tokenizer_id=env.tok_big.id,
                       tokenizer_hash=env.tok_big.tokenizer_hash))
    before = env.file_snapshot()      # crafted file is pre-existing now
    with pytest.raises(ValueError, match="exceeds the model's vocab_size"):
        env.sq.run(env.small_model, "vocab-bad")
    assert env.file_snapshot() == before
    _rm_sample(env, env.small_model, "vocab-bad")
    # positive control: the small model's own sample measures fine through
    # its recorded (compatible) tokenizer
    ok = env.sq.run(env.small_model, small_s.sample_id)
    assert ok.evaluated_token_count == 4 == ok.generated_token_count
    env.no_tmp()


# =========================================================================== #
# Determinism + result hash semantics
# =========================================================================== #

def test_repeated_measurement_deterministic_hash_and_no_mutation(env):
    s = _make_sample(env)
    sfile = env.forge.storage.root / "samples" / env.big_model / \
        f"sample-{s.sample_id}" / "manifest.json"
    sbytes = sfile.read_bytes()
    r1 = env.sq.run(env.big_model, s.sample_id)
    r2 = env.sq.run(env.big_model, s.sample_id)
    assert r1.evaluation_id != r2.evaluation_id    # distinct immutable recs
    assert r1.loss_nats == r2.loss_nats
    assert r1.perplexity == r2.perplexity
    assert r1.result_hash == r2.result_hash        # semantic determinism
    assert sfile.read_bytes() == sbytes            # sample never modified
    # a different generated sequence (different length -> different ids and
    # target accounting) yields a different semantic measurement
    s2 = _make_sample(env, max_new_tokens=4)
    r3 = env.sq.run(env.big_model, s2.sample_id)
    assert r3.result_hash != r1.result_hash
    assert r3.token_sequence_sha256 != r1.token_sequence_sha256
    assert r3.evaluated_token_count == 4 == r3.generated_token_count
    env.no_tmp()


def test_result_hash_static_semantics_and_exclusions(env):
    from app.sample_quality import SampleQualityEngine as SQE

    s = _make_sample(env)
    r = env.sq.run(env.big_model, s.sample_id)
    base = SQE.result_hash(r)
    # semantic inputs participate
    assert SQE.result_hash(r.model_copy(
        update={"loss_nats": r.loss_nats + 0.1})) != base
    assert SQE.result_hash(r.model_copy(
        update={"sample_id": "other-sample"})) != base
    assert SQE.result_hash(r.model_copy(
        update={"sample_result_hash": "0" * 64})) != base
    assert SQE.result_hash(r.model_copy(
        update={"checkpoint_weights_sha256": "0" * 64})) != base
    assert SQE.result_hash(r.model_copy(
        update={"token_sequence_sha256": "0" * 64})) != base
    assert SQE.result_hash(r.model_copy(
        update={"evaluated_token_count": r.evaluated_token_count + 1})
    ) != base
    # non-semantic fields do NOT participate
    assert SQE.result_hash(r.model_copy(
        update={"evaluation_id": "deadbeef0001"})) == base
    assert SQE.result_hash(r.model_copy(
        update={"created_at": r.created_at.replace(year=2001)})) == base
    assert SQE.result_hash(r.model_copy(
        update={"duration_seconds": 123.456})) == base
    assert SQE.result_hash(r.model_copy(
        update={"hardware": {"device": "cuda:0"}})) == base
    env.no_tmp()


# =========================================================================== #
# Storage discipline
# =========================================================================== #

def test_one_manifest_per_measurement_zero_aux_files(env):
    s = _make_sample(env)
    before = env.file_snapshot()
    edir = env.forge.storage.root / "sample-evaluations" / env.big_model
    before_dirs = {d.name for d in edir.iterdir()} if edir.exists() else set()
    evals = [env.sq.run(env.big_model, s.sample_id) for _ in range(3)]
    after = env.file_snapshot()
    new = [k for k in after if k not in before]
    assert len(new) == 3
    assert all(k.startswith(f"sample-evaluations/{env.big_model}/"
                            "evaluation-") and k.endswith("/manifest.json")
               for k in new)
    # exactly the three new dirs; each holds ONLY its manifest
    new_dirs = {d.name for d in edir.iterdir() if d.is_dir()} - before_dirs
    assert len(new_dirs) == 3
    assert all(sorted(p.name for p in (edir / d).iterdir())
               == ["manifest.json"] for d in new_dirs)
    # every pre-existing file byte-identical; the sample is untouched
    for k, v in before.items():
        assert after[k] == v
    listed = {e.evaluation_id for e in
              env.sq.list_sample_evaluations(env.big_model)}
    assert {e.evaluation_id for e in evals} <= listed
    env.no_tmp()


def test_list_get_read_only_deterministic(env):
    s = _make_sample(env)
    n0 = len(env.sq.list_sample_evaluations(env.big_model))
    e1 = env.sq.run(env.big_model, s.sample_id)
    e2 = env.sq.run(env.big_model, s.sample_id)
    got = env.sq.list_sample_evaluations(env.big_model)
    ids = [e.evaluation_id for e in got]
    assert {e1.evaluation_id, e2.evaluation_id} <= set(ids)
    assert len(ids) == n0 + 2
    keyed = [(e.created_at, e.evaluation_id) for e in got]
    assert keyed == sorted(keyed)
    assert got[-2].evaluation_id == e1.evaluation_id
    assert got[-1].evaluation_id == e2.evaluation_id
    one = env.sq.get_sample_evaluation(env.big_model, e1.evaluation_id)
    assert one.result_hash == e1.result_hash
    with pytest.raises(FileNotFoundError):
        env.sq.list_sample_evaluations("no-such-model")
    with pytest.raises(FileNotFoundError):
        env.sq.get_sample_evaluation(env.big_model, "no-such-eval")
    with pytest.raises(FileNotFoundError):
        env.sq.get_sample_evaluation("no-such-model", e1.evaluation_id)
    before = env.file_snapshot()
    env.sq.list_sample_evaluations(env.big_model)
    env.sq.get_sample_evaluation(env.big_model, e2.evaluation_id)
    assert env.file_snapshot() == before
    env.no_tmp()


def test_measurement_runs_without_grad_through_existing_forward(
        env, monkeypatch):
    import app.sample_quality as sq_mod

    seen = []
    orig_build = sq_mod.build_transformer

    def wrapped(cfg, parent_state=None, device="cpu"):
        built = orig_build(cfg, parent_state=parent_state, device=device)
        fwd = built.module.forward

        def guarded(*args, **kwargs):
            seen.append(torch.is_grad_enabled())
            return fwd(*args, **kwargs)

        built.module.forward = guarded
        return built

    monkeypatch.setattr(sq_mod, "build_transformer", wrapped)
    s = _make_sample(env)
    r = env.sq.run(env.big_model, s.sample_id)
    assert seen and not any(seen), "forward ran with grad enabled"
    assert math.isfinite(r.loss_nats)
    env.no_tmp()


# =========================================================================== #
# HTTP API coverage (shared TestClient root; per-test model setup)
# =========================================================================== #

MODELS = "/api/v1/models"
GENERATE = "/api/v1/samples/generate"


def _http_env(api_client, tag: str, vocab: int = 640):
    corpus = _corpus(200, tag)
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", (f"{tag}.txt", corpus,
                                           "text/plain"))],
                         data={"name": f"api16-{tag}-ds"})
    assert up.status_code == 201, up.text
    ds = up.json()["dataset_id"]
    tok = api_client.post("/api/v1/tokenizers/train",
                          data={"config": json.dumps(
                              {"name": f"api16-{tag}-tok",
                               "vocab_size": 320}),
                                "dataset_id": ds}).json()["tokenizer"]
    api_client.post(f"/api/v1/datasets/{ds}/tokenize",
                    json={"tokenizer_id": tok["id"]})
    cfg = {"name": f"api16-{tag}-model", "vocab_size": vocab,
           "context_length": 64, "hidden_size": 64, "n_layers": 2,
           "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128,
           "seed": 1}
    mid = api_client.post(MODELS, json={"config": cfg}).json()["model"]["id"]
    rep = api_client.post("/api/v1/training/run", json={
        "name": f"api16-{tag}-run", "method": "continued_pretraining",
        "model_id": mid, "dataset_id": ds, "tokenizer_id": tok["id"],
        "learning_rate": 3e-3, "batch_size": 8, "steps": 8,
        "max_seq_len": 32, "warmup_steps": 0, "weight_decay": 0.01,
        "adam_beta1": 0.9, "adam_beta2": 0.999, "lr_schedule": "cosine",
        "eval_every_steps": 4, "keep_best": True, "seed": 3})
    assert rep.status_code == 200, rep.text
    ckpt = rep.json()["checkpoints"][0]["checkpoint_id"]
    return {"mid": mid, "tok": tok["id"], "ckpt": ckpt}


def _quality_url(mid: str, sample_id: str) -> str:
    return f"{MODELS}/{mid}/samples/{sample_id}/quality"


def test_api_quality_post_list_get_determinism(api_client):
    h = _http_env(api_client, "quality")
    body = {"model_id": h["mid"], "checkpoint_id": h["ckpt"],
            "tokenizer_id": h["tok"], "prompt": PROMPT, "strategy": "greedy",
            "max_new_tokens": 6}
    s1 = api_client.post(GENERATE, json=body).json()
    # POST with an EMPTY body: the sample itself is the source of truth
    q1 = api_client.post(_quality_url(h["mid"], s1["sample_id"]))
    assert q1.status_code == 200, q1.text
    e1 = q1.json()
    assert e1["model_id"] == h["mid"] and e1["sample_id"] == s1["sample_id"]
    assert e1["sample_result_hash"] == s1["result_hash"]
    assert e1["checkpoint_id"] == h["ckpt"]
    assert e1["tokenizer_id"] == h["tok"]
    assert e1["evaluated_token_count"] == 6 == e1["generated_token_count"]
    assert e1["prompt_token_count"] == len(s1["prompt_token_ids"])
    assert e1["window_token_count"] <= 64 and e1["context_length"] == 64
    assert e1["window_rule"] == "single_window"
    assert math.isfinite(e1["loss_nats"])
    # M4 convention: perplexity comes from the unrounded loss
    assert abs(e1["perplexity"]
               - math.exp(min(e1["loss_nats"], 100.0))) < 1e-3
    q2 = api_client.post(_quality_url(h["mid"], s1["sample_id"]))
    assert q2.status_code == 200
    e2 = q2.json()
    assert e2["evaluation_id"] != e1["evaluation_id"]
    assert e2["result_hash"] == e1["result_hash"]
    assert e2["loss_nats"] == e1["loss_nats"]
    # list: deterministic (created_at, evaluation_id) order; get works
    lst = api_client.get(f"{MODELS}/{h['mid']}/sample-quality")
    assert lst.status_code == 200 and len(lst.json()) == 2
    keyed = [(x["created_at"], x["evaluation_id"]) for x in lst.json()]
    assert keyed == sorted(keyed)
    assert api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality").text == lst.text
    one = api_client.get(f"{MODELS}/{h['mid']}/sample-quality/"
                         f"{e1['evaluation_id']}")
    assert one.status_code == 200
    assert one.json()["result_hash"] == e1["result_hash"]
    # no edit/delete endpoints (405)
    url1 = f"{MODELS}/{h['mid']}/sample-quality/{e1['evaluation_id']}"
    assert api_client.put(url1).status_code == 405
    assert api_client.delete(url1).status_code == 405
    # 404s
    assert api_client.post(_quality_url(h["mid"], "no-such-sample")
                           ).status_code == 404
    assert api_client.post(_quality_url("no-such-model", s1["sample_id"])
                           ).status_code == 404
    assert api_client.get(f"{MODELS}/no-such-model/sample-quality"
                          ).status_code == 404
    assert api_client.get(f"{MODELS}/{h['mid']}/sample-quality/no-such"
                          ).status_code == 404


def test_api_failure_paths_zero_growth(api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "fail16")
    body = {"model_id": h["mid"], "checkpoint_id": h["ckpt"],
            "tokenizer_id": h["tok"], "prompt": PROMPT, "strategy": "greedy",
            "max_new_tokens": 6}
    s1 = api_client.post(GENERATE, json=body).json()
    assert api_client.post(_quality_url(h["mid"], s1["sample_id"])
                           ).status_code == 200
    n0 = len(api_client.get(f"{MODELS}/{h['mid']}/sample-quality").json())

    # a second model WITHOUT any training: sample ids are model-scoped, so
    # the same sample id under another model is unknown there (404)
    cfg2 = {"name": "api16-fail16-model2", "vocab_size": 640,
            "context_length": 64, "hidden_size": 64, "n_layers": 2,
            "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128,
            "seed": 9}
    m2 = api_client.post(MODELS, json={"config": cfg2}).json()["model"]["id"]
    assert api_client.post(_quality_url(m2, s1["sample_id"])
                           ).status_code == 404
    assert api_client.post(_quality_url("no-such-model", s1["sample_id"])
                           ).status_code == 404
    assert api_client.post(_quality_url(h["mid"], "no-such-sample")
                           ).status_code == 404

    root = get_forge().storage.root / "samples" / h["mid"]
    base = json.loads(api_client.get(f"{MODELS}/{h['mid']}/samples/"
                                     f"{s1['sample_id']}").text)
    try:
        # fabricated OVERLONG sample (65 > 64 tokens) -> 422, zero writes
        ids = base["generated_token_ids"]
        over = dict(base, generated_token_ids=(ids * 17)[:100],
                    generated_token_count=100, sample_id="overlong16")
        sdir = root / "sample-overlong16"
        sdir.mkdir(parents=True)
        (sdir / "manifest.json").write_text(json.dumps(over))
        r = api_client.post(_quality_url(h["mid"], "overlong16"))
        assert r.status_code == 422, r.text
        # fabricated TAMPERED sample (in-range id flip) -> 409, zero writes
        x0 = base["generated_token_ids"][0]
        tam = dict(base, generated_token_ids=[
            (x0 - 1 if x0 > 0 else 1) if i == 0 else x
            for i, x in enumerate(base["generated_token_ids"])],
            sample_id="tampered16")
        tdir = root / "sample-tampered16"
        tdir.mkdir(parents=True)
        (tdir / "manifest.json").write_text(json.dumps(tam))
        r = api_client.post(_quality_url(h["mid"], "tampered16"))
        assert r.status_code == 409, r.text
    finally:
        shutil.rmtree(root / "sample-overlong16", ignore_errors=True)
        shutil.rmtree(root / "sample-tampered16", ignore_errors=True)
    # no evaluation manifest was created by any failure
    assert len(api_client.get(f"{MODELS}/{h['mid']}/sample-quality")
               .json()) == n0
    leftovers = [p for p in get_forge().storage.root.rglob("*")
                 if p.is_file() and ".tmp" in p.name]
    assert leftovers == []


def test_api_openapi_exposes_sample_quality(api_client):
    spec = api_client.get("/openapi.json").json()
    assert _quality_url("{model_id}", "{sample_id}") in spec["paths"]
    assert "/api/v1/models/{model_id}/sample-quality" in spec["paths"]
    assert ("/api/v1/models/{model_id}/sample-quality/{evaluation_id}"
            in spec["paths"])
    assert "SampleEvaluationRecord" in spec["components"]["schemas"]
    # surface: 46 (M15 era) + 3 (M16) + 0 (M17) + 1 (M18) + 1 (M19)
    # + 1 (M20) + 1 (M21 suite-runs by-suite)
    # + 1 (M22 by-suite summary) + 1 (M23 gates by-policy)
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
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind) = 70
    assert len(spec["paths"]) == 70


# =========================================================================== #
# M18: dedicated read-only records listing endpoint
# (/sample-quality/records)
#
# GET /models/{id}/sample-quality/records returns the FULL immutable M16
# SampleEvaluationRecord payloads (loss_nats/perplexity included) in the
# exact authoritative M16 ordering ((created_at, evaluation_id) ASCENDING)
# as a pure pass-through of SampleQualityEngine.list_sample_evaluations —
# no derived statistics, no aggregation, zero writes. The M16 reference
# listing (/sample-quality), the individual getter and the M17 dashboard
# section stay unchanged. The route is registered before
# /sample-quality/{evaluation_id} so the literal "records" segment wins.
# =========================================================================== #

def _records_url(mid: str) -> str:
    return f"{MODELS}/{mid}/sample-quality/records"


def _gen_sample(api_client, h, **overrides) -> dict:
    body = {"model_id": h["mid"], "checkpoint_id": h["ckpt"],
            "tokenizer_id": h["tok"], "prompt": PROMPT, "strategy": "greedy",
            "max_new_tokens": 6}
    body.update(overrides)
    return api_client.post(GENERATE, json=body).json()


def _http_model_only(api_client, tag: str, seed: int = 1) -> str:
    cfg = {"name": f"api18-{tag}-model", "vocab_size": 640,
           "context_length": 64, "hidden_size": 64, "n_layers": 2,
           "n_heads": 4, "n_kv_heads": 2, "intermediate_size": 128,
           "seed": seed}
    r = api_client.post(MODELS, json={"config": cfg})
    assert r.status_code == 201, r.text
    return r.json()["model"]["id"]


def test_api_records_endpoint_full_parity_with_engine(api_client):
    """New endpoint payloads == SampleQualityEngine.list_sample_evaluations
    (byte-exact) == the individual M16 getter responses; ordering is the
    authoritative (created_at, evaluation_id) ASCENDING."""
    from app.engine import get_forge

    h = _http_env(api_client, "recpar")
    s = _gen_sample(api_client, h)
    for _ in range(2):
        assert api_client.post(_quality_url(h["mid"], s["sample_id"])
                               ).status_code == 200
    r = api_client.get(_records_url(h["mid"]))
    assert r.status_code == 200, r.text
    recs = r.json()
    assert len(recs) == 2
    # authoritative engine representation
    engine = get_forge().list_sample_evaluations(h["mid"])
    assert [x["evaluation_id"] for x in recs] == \
        [e.evaluation_id for e in engine]
    keyed = [(x["created_at"], x["evaluation_id"]) for x in recs]
    assert keyed == sorted(keyed)               # ascending deterministic
    assert recs == [e.model_dump(mode="json") for e in engine]  # parity
    # individual getter parity
    for x in recs:
        one = api_client.get(f"{MODELS}/{h['mid']}/sample-quality/"
                             f"{x['evaluation_id']}")
        assert one.status_code == 200 and one.json() == x


def test_api_records_empty_history_deterministic_no_writes(api_client):
    from app.engine import get_forge

    mid = _http_model_only(api_client, "recempty")
    root = get_forge().storage.root
    before = sorted(p.relative_to(root).as_posix()
                    for p in root.rglob("*") if p.is_file())
    blob = {p: (root / p).read_bytes() for p in before}
    r1 = api_client.get(_records_url(mid))
    assert r1.status_code == 200 and r1.json() == []
    r2 = api_client.get(_records_url(mid))
    assert r2.status_code == 200 and r2.text == r1.text
    after = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*") if p.is_file())
    assert after == before                     # zero new files
    for p in before:
        assert (root / p).read_bytes() == blob[p]   # byte-identical
    assert [p for p in root.rglob("*") if ".tmp" in p.name] == []


def test_api_records_literal_route_not_shadowed_by_evaluation_id(api_client):
    """The literal /records segment resolves to the new list route while the
    parameterized getter keeps working for real ids and 404s for others."""
    h = _http_env(api_client, "recroute")
    s = _gen_sample(api_client, h)
    e = api_client.post(_quality_url(h["mid"], s["sample_id"])).json()
    recs = api_client.get(_records_url(h["mid"]))
    assert recs.status_code == 200 and len(recs.json()) == 1
    # evaluation_id 'records' must NOT be interpreted as an id by the getter
    assert api_client.get(f"{MODELS}/{h['mid']}/sample-quality/no-such-eval"
                          ).status_code == 404
    # parameterized getter for the real id unchanged
    one = api_client.get(f"{MODELS}/{h['mid']}/sample-quality/"
                         f"{e['evaluation_id']}")
    assert one.status_code == 200 and one.json() == recs.json()[0]
    # M16 reference listing unchanged and identical payloads
    lst = api_client.get(f"{MODELS}/{h['mid']}/sample-quality")
    assert lst.status_code == 200 and lst.json() == recs.json()


def test_api_records_unknown_model_404_zero_writes(api_client):
    from app.engine import get_forge

    root = get_forge().storage.root
    before = sorted(p.relative_to(root).as_posix()
                    for p in root.rglob("*") if p.is_file())
    blob = {p: (root / p).read_bytes() for p in before}
    r = api_client.get(_records_url("no-such-model-18"))
    assert r.status_code == 404
    after = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*") if p.is_file())
    assert after == before
    for p in before:
        assert (root / p).read_bytes() == blob[p]
    assert [p for p in root.rglob("*") if ".tmp" in p.name] == []


def test_api_records_model_isolation(api_client):
    """Records belonging to another model never appear; each model sees
    exactly its own sample-evaluations (crafted foreign record included)."""
    from app.engine import get_forge

    h = _http_env(api_client, "reciso")
    s = _gen_sample(api_client, h)
    q = api_client.post(_quality_url(h["mid"], s["sample_id"]))
    assert q.status_code == 200
    a_real_ids = [x["evaluation_id"] for x in
                  api_client.get(_records_url(h["mid"])).json()]
    assert len(a_real_ids) == 1
    m2 = _http_model_only(api_client, "reciso-b", seed=11)
    assert api_client.get(_records_url(m2)).json() == []
    # craft one schema-valid record under model B's directory
    root = get_forge().storage.root
    real = api_client.get(f"{MODELS}/{h['mid']}/sample-quality").json()[0]
    crafted = dict(real, evaluation_id="e-b-craft01", model_id=m2,
                   sample_id="s-b-craft01")
    d = root / "sample-evaluations" / m2 / "evaluation-e-b-craft01"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(crafted))
    try:
        a = api_client.get(_records_url(h["mid"])).json()
        b = api_client.get(_records_url(m2)).json()
        assert [x["evaluation_id"] for x in a] == a_real_ids
        assert [x["evaluation_id"] for x in b] == ["e-b-craft01"]
        assert a[0]["model_id"] == h["mid"] and b[0]["model_id"] == m2
    finally:
        shutil.rmtree(root / "sample-evaluations" / m2, ignore_errors=True)
    assert api_client.get(_records_url(m2)).json() == []


def test_api_records_metric_values_exposed_no_derived_stats(api_client):
    from app.sample_quality import SampleEvaluationRecord

    h = _http_env(api_client, "recmet")
    s = _gen_sample(api_client, h)
    assert api_client.post(_quality_url(h["mid"], s["sample_id"])
                           ).status_code == 200
    recs = api_client.get(_records_url(h["mid"])).json()
    assert len(recs) == 1
    item = recs[0]
    # the actual immutable M16 metric values are present
    assert "loss_nats" in item and "perplexity" in item
    assert isinstance(item["loss_nats"], float) and item["loss_nats"] > 0.0
    assert isinstance(item["perplexity"], float)
    # exact record schema — nothing added, nothing derived
    assert set(item) == set(SampleEvaluationRecord.model_fields)
    blob = api_client.get(_records_url(h["mid"])).text.lower()
    for banned in ("average", "aggregate", "rank", "score", "trend",
                   "verdict", "min_", "max_", "mean", "delta", "statistic"):
        assert banned not in blob, banned


def test_api_records_deterministic_repeat_read_only_dashboard_intact(
        api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "recdet")
    s = _gen_sample(api_client, h)
    for _ in range(2):
        assert api_client.post(_quality_url(h["mid"], s["sample_id"])
                               ).status_code == 200
    root = get_forge().storage.root
    before = sorted(p.relative_to(root).as_posix()
                    for p in root.rglob("*") if p.is_file())
    blob = {p: (root / p).read_bytes() for p in before}
    dash0 = api_client.get(f"{MODELS}/{h['mid']}/dashboard")
    assert dash0.status_code == 200
    d0 = dash0.json()
    assert d0["sample_quality"]["total_count"] == 2
    r1 = api_client.get(_records_url(h["mid"]))
    r2 = api_client.get(_records_url(h["mid"]))
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.text == r2.text and r1.json() == r2.json()  # byte-identical
    # M17 dashboard unchanged semantics + deterministic across reads
    d1 = api_client.get(f"{MODELS}/{h['mid']}/dashboard").json()
    assert d1["result_hash"] == d0["result_hash"] and d1 == d0
    after = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*") if p.is_file())
    assert after == before
    for p in before:
        assert (root / p).read_bytes() == blob[p]
    assert [p for p in root.rglob("*") if ".tmp" in p.name] == []


def test_api_openapi_records_route(api_client):
    spec = api_client.get("/openapi.json").json()
    p = "/api/v1/models/{model_id}/sample-quality/records"
    assert p in spec["paths"]
    op = spec["paths"][p]["get"]
    assert op["tags"] == ["sample-quality"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["type"] == "array"
    assert schema["items"]["$ref"].endswith("SampleEvaluationRecord")
    # both neighbouring routes stay documented
    assert "/api/v1/models/{model_id}/sample-quality" in spec["paths"]
    assert ("/api/v1/models/{model_id}/sample-quality/{evaluation_id}"
            in spec["paths"])
    # surface: 46 (M15 era) + 3 (M16) + 0 (M17) + 1 (M18) + 1 (M19)
    # + 1 (M20) + 1 (M21 suite-runs by-suite)
    # + 1 (M22 by-suite summary) + 1 (M23 gates by-policy)
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
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind) = 70
    assert len(spec["paths"]) == 70


# =========================================================================== #
# M19: per-sample quality history access (by-sample)
#
# list_sample_evaluations_for_sample(model_id, sample_id) + the route
# GET /models/{id}/sample-quality/by-sample/{sample_id} answer: "for a known
# generated sample, what immutable M16 measurements exist?" — the model's
# authoritative M16 listing filtered by the sample's persisted sample_id
# (sample must exist under samples/<model_id>/; nothing inferred from
# filenames). Full verbatim record payloads, exact M16 (created_at,
# evaluation_id) ordering, [] for a measured-less sample, 404 for unknown
# model/sample and for a sample id of another model. Read-only, no derived
# statistics; M16/M17/M18 surfaces unchanged.
# =========================================================================== #

def _by_sample_url(mid: str, sample_id: str) -> str:
    return f"{MODELS}/{mid}/sample-quality/by-sample/{sample_id}"


def test_engine_by_sample_exact_filter_ordering_and_payloads(env):
    """Engine: per-sample listing == full M16 listing filtered by that
    sample's persisted id — exact payloads, exact ordering, no leaks."""
    s_a = _make_sample(env)                     # measured twice below
    s_b = _make_sample(env, max_new_tokens=4)   # measured once
    ev_a1 = env.sq.run(env.big_model, s_a.sample_id)
    ev_a2 = env.sq.run(env.big_model, s_a.sample_id)
    ev_b1 = env.sq.run(env.big_model, s_b.sample_id)
    got_a = env.sq.list_sample_evaluations_for_sample(
        env.big_model, s_a.sample_id)
    assert [r.evaluation_id for r in got_a] == \
        [r.evaluation_id for r in sorted(
            (ev_a1, ev_a2), key=lambda r: (r.created_at, r.evaluation_id))]
    # payloads are the verbatim authoritative records
    full = env.sq.list_sample_evaluations(env.big_model)
    expected = [r for r in full if r.sample_id == s_a.sample_id]
    assert got_a == expected
    assert all(r.sample_id == s_a.sample_id for r in got_a)
    # every record matches the individual getter byte-for-byte
    for r in got_a:
        assert r == env.sq.get_sample_evaluation(env.big_model,
                                                 r.evaluation_id)
    # the other sample's measurements never appear; ordering ascending
    got_b = env.sq.list_sample_evaluations_for_sample(
        env.big_model, s_b.sample_id)
    assert [r.evaluation_id for r in got_b] == [ev_b1.evaluation_id]
    assert all(r.sample_id == s_b.sample_id for r in got_b)
    keyed = [(r.created_at, r.evaluation_id) for r in got_a]
    assert keyed == sorted(keyed)
    env.no_tmp()


def test_engine_by_sample_empty_history(env):
    """A valid sample of the small model with no measurements -> []."""
    s = env.samples.run(SampleGenerateRequest(
        model_id=env.small_model, checkpoint_id=env.small_ckpt,
        tokenizer_id=env.tok_small.id, prompt=PROMPT, strategy="greedy",
        max_new_tokens=4))
    got = env.sq.list_sample_evaluations_for_sample(env.small_model,
                                                    s.sample_id)
    assert got == []
    env.no_tmp()


def test_engine_by_sample_resolution_404s(env):
    sid = _make_sample(env).sample_id
    with pytest.raises(FileNotFoundError, match="model 'no-such-model'"):
        env.sq.list_sample_evaluations_for_sample("no-such-model", sid)
    with pytest.raises(FileNotFoundError, match="sample 'no-such-sample'"):
        env.sq.list_sample_evaluations_for_sample(env.big_model,
                                                  "no-such-sample")
    # the same sample id under a different model is not that model's sample
    with pytest.raises(FileNotFoundError, match="sample"):
        env.sq.list_sample_evaluations_for_sample(env.small_model, sid)
    env.no_tmp()


def test_api_by_sample_history_parity_and_metric_values(api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "bysamp")
    s1 = _gen_sample(api_client, h)
    s2 = _gen_sample(api_client, h, max_new_tokens=4)
    assert api_client.post(_quality_url(h["mid"], s1["sample_id"])
                           ).status_code == 200
    assert api_client.post(_quality_url(h["mid"], s1["sample_id"])
                           ).status_code == 200
    assert api_client.post(_quality_url(h["mid"], s2["sample_id"])
                           ).status_code == 200
    r = api_client.get(_by_sample_url(h["mid"], s1["sample_id"]))
    assert r.status_code == 200, r.text
    recs = r.json()
    assert len(recs) == 2
    # parity: == full records listing filtered by the sample
    engine = get_forge().list_sample_evaluations(h["mid"])
    expected = [e.model_dump(mode="json") for e in engine
                if e.sample_id == s1["sample_id"]]
    assert recs == expected
    assert all(x["sample_id"] == s1["sample_id"] for x in recs)
    # M18 listing filtered to that sample == same payloads
    all_recs = api_client.get(_records_url(h["mid"])).json()
    assert recs == [x for x in all_recs if x["sample_id"] == s1["sample_id"]]
    # metric values present, no derived-statistic keys
    for x in recs:
        assert "loss_nats" in x and "perplexity" in x
        assert x["loss_nats"] > 0.0 and isinstance(x["perplexity"], float)
    blob = api_client.get(_by_sample_url(h["mid"], s1["sample_id"])).text
    for banned in ("average", "aggregate", "rank", "score", "trend",
                   "verdict", "mean", "delta", "statistic"):
        assert banned not in blob.lower(), banned
    # the other sample's history is its own
    r2 = api_client.get(_by_sample_url(h["mid"], s2["sample_id"]))
    assert r2.status_code == 200 and len(r2.json()) == 1
    assert r2.json()[0]["sample_id"] == s2["sample_id"]


def test_api_by_sample_empty_unknown_and_model_isolation(api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "bysampiso")
    s1 = _gen_sample(api_client, h)
    assert api_client.post(_quality_url(h["mid"], s1["sample_id"])
                           ).status_code == 200
    # second model WITHOUT samples: any sample id is unknown there (404),
    # even one that exists under the first model
    m2 = _http_model_only(api_client, "bysampiso-b", seed=23)
    assert api_client.get(_by_sample_url(m2, s1["sample_id"])
                          ).status_code == 404
    assert api_client.get(_by_sample_url("no-such-model", s1["sample_id"])
                          ).status_code == 404
    assert api_client.get(_by_sample_url(h["mid"], "no-such-sample")
                          ).status_code == 404
    # model B gets a REAL sample of its own with one crafted measurement:
    # B sees exactly its own record; A never sees it
    root = get_forge().storage.root
    b_sid = "b-sample-000001"
    # B gets a real (schema-valid) sample manifest: copy A's sample record
    # verbatim and rewrite only the identity fields
    a_sample = api_client.get(f"{MODELS}/{h['mid']}/samples/"
                              f"{s1['sample_id']}").json()
    b_manifest = dict(a_sample, sample_id=b_sid, model_id=m2)
    (root / "samples" / m2 / f"sample-{b_sid}").mkdir(parents=True)
    (root / "samples" / m2 / f"sample-{b_sid}" / "manifest.json").write_text(
        json.dumps(b_manifest))
    real = api_client.get(f"{MODELS}/{h['mid']}/sample-quality").json()[0]
    crafted = dict(real, evaluation_id="e-b-bysamp01", model_id=m2,
                   sample_id=b_sid,
                   sample_result_hash=real["sample_result_hash"])
    d = root / "sample-evaluations" / m2 / "evaluation-e-b-bysamp01"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(crafted))
    try:
        a = api_client.get(_by_sample_url(h["mid"], s1["sample_id"])).json()
        b = api_client.get(_by_sample_url(m2, b_sid)).json()
        assert len(a) == 1 and len(b) == 1
        assert b[0]["evaluation_id"] == "e-b-bysamp01"
        assert a[0]["model_id"] == h["mid"] and b[0]["model_id"] == m2
        # A's by-sample never contains B's evaluation
        assert all(x["model_id"] == h["mid"] for x in a)
    finally:
        shutil.rmtree(root / "samples" / m2 / f"sample-{b_sid}",
                      ignore_errors=True)
        shutil.rmtree(root / "sample-evaluations" / m2, ignore_errors=True)


def test_api_by_sample_deterministic_repeat_and_read_only(api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "bysampdet")
    s1 = _gen_sample(api_client, h)
    s2 = _gen_sample(api_client, h, max_new_tokens=4)
    for sid in (s1["sample_id"], s2["sample_id"]):
        assert api_client.post(_quality_url(h["mid"], sid)
                               ).status_code == 200
    root = get_forge().storage.root
    before = sorted(p.relative_to(root).as_posix()
                    for p in root.rglob("*") if p.is_file())
    blob = {p: (root / p).read_bytes() for p in before}
    dash0 = api_client.get(f"{MODELS}/{h['mid']}/dashboard").json()
    r1 = api_client.get(_by_sample_url(h["mid"], s1["sample_id"]))
    r2 = api_client.get(_by_sample_url(h["mid"], s1["sample_id"]))
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.text == r2.text and r1.json() == r2.json()
    assert api_client.get(_by_sample_url(h["mid"], s2["sample_id"])
                          ).text == api_client.get(
        _by_sample_url(h["mid"], s2["sample_id"])).text
    # M16/M17/M18 surfaces unchanged by the reads
    assert api_client.get(f"{MODELS}/{h['mid']}/sample-quality").text == \
        api_client.get(f"{MODELS}/{h['mid']}/sample-quality").text
    dash1 = api_client.get(f"{MODELS}/{h['mid']}/dashboard").json()
    assert dash1["result_hash"] == dash0["result_hash"] and dash1 == dash0
    assert api_client.get(_records_url(h["mid"])).text == \
        api_client.get(_records_url(h["mid"])).text
    # zero writes, zero new files, zero .tmp
    after = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*") if p.is_file())
    assert after == before
    for p in before:
        assert (root / p).read_bytes() == blob[p]
    assert [p for p in root.rglob("*") if ".tmp" in p.name] == []


def test_api_openapi_by_sample_route(api_client):
    spec = api_client.get("/openapi.json").json()
    p = "/api/v1/models/{model_id}/sample-quality/by-sample/{sample_id}"
    assert p in spec["paths"]
    op = spec["paths"][p]["get"]
    assert op["tags"] == ["sample-quality"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["type"] == "array"
    assert schema["items"]["$ref"].endswith("SampleEvaluationRecord")
    # all neighbouring routes stay documented
    assert "/api/v1/models/{model_id}/sample-quality" in spec["paths"]
    assert "/api/v1/models/{model_id}/sample-quality/records" in spec["paths"]
    assert ("/api/v1/models/{model_id}/sample-quality/{evaluation_id}"
            in spec["paths"])
    # M19–M27 each added exactly one documented route (M27
    # samples by-checkpoint adds one)
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
    # + 1 (M38 evaluations by-state-kind) -> 70
    assert len(spec["paths"]) == 70


# =========================================================================== #
# M20: per-checkpoint quality history access (by-checkpoint)
#
# list_sample_evaluations_for_checkpoint(model_id, checkpoint_id) + the route
# GET /models/{id}/sample-quality/by-checkpoint/{checkpoint_id} answer: "which
# immutable sample-quality evaluations belong to this checkpoint?" — the
# model's authoritative M16 listing filtered by the persisted checkpoint_id
# recorded in each SampleEvaluationRecord, after ownership validation through
# the model's M3 checkpoint registry (unknown model/checkpoint or a checkpoint
# id belonging to another model -> FileNotFoundError -> 404; checkpoint with no
# measurements -> []). Pure data access: no derived fields, no statistics, no
# checkpoint comparison, read-only, zero storage growth.
# =========================================================================== #

def _by_ckpt_url(mid: str, checkpoint_id: str) -> str:
    return (f"{MODELS}/{mid}/sample-quality/by-checkpoint/{checkpoint_id}")


def test_engine_by_checkpoint_empty_history(env):
    """A valid checkpoint of the big model with no measurements -> [].
    big_ckpts[1] is registered (trained) but nothing in the shared env has
    ever been measured under it — must run before any M20 test measures it."""
    got = env.sq.list_sample_evaluations_for_checkpoint(env.big_model,
                                                        env.big_ckpts[1])
    assert got == []
    env.no_tmp()


def test_engine_by_checkpoint_exact_filter_ordering_and_payloads(env):
    """Engine: per-checkpoint listing == full M16 listing filtered by that
    checkpoint's persisted id — exact payloads, exact ordering, no leaks."""
    s_a = _make_sample(env)                     # measured twice (ckpts[0])
    ev_a1 = env.sq.run(env.big_model, s_a.sample_id)
    ev_a2 = env.sq.run(env.big_model, s_a.sample_id)
    # a second checkpoint of the SAME model gets its own measurement
    s_c = _make_sample(env, checkpoint_id=env.big_ckpts[1])
    ev_c1 = env.sq.run(env.big_model, s_c.sample_id)
    assert ev_c1.checkpoint_id == env.big_ckpts[1]
    got_a = env.sq.list_sample_evaluations_for_checkpoint(
        env.big_model, env.big_ckpts[0])
    # the shared env already holds earlier M16 measurements under ckpts[0]:
    # the two new records are present, and the list is EXACTLY the full
    # authoritative listing filtered by the persisted checkpoint id
    got_ids = [r.evaluation_id for r in got_a]
    assert {ev_a1.evaluation_id, ev_a2.evaluation_id} <= set(got_ids)
    # payloads are the verbatim authoritative records
    full = env.sq.list_sample_evaluations(env.big_model)
    expected = [r for r in full if r.checkpoint_id == env.big_ckpts[0]]
    assert got_a == expected
    assert all(r.checkpoint_id == env.big_ckpts[0] for r in got_a)
    # every record matches the individual getter byte-for-byte
    for r in got_a:
        assert r == env.sq.get_sample_evaluation(env.big_model,
                                                 r.evaluation_id)
    # the other checkpoint's measurement never appears; ordering ascending
    got_c = env.sq.list_sample_evaluations_for_checkpoint(
        env.big_model, env.big_ckpts[1])
    assert [r.evaluation_id for r in got_c] == [ev_c1.evaluation_id]
    assert all(r.checkpoint_id == env.big_ckpts[1] for r in got_c)
    assert all(r.evaluation_id != ev_c1.evaluation_id for r in got_a)
    keyed = [(r.created_at, r.evaluation_id) for r in got_a]
    assert keyed == sorted(keyed)
    env.no_tmp()


def test_engine_by_checkpoint_resolution_404s(env):
    ckpt = env.big_ckpts[0]
    with pytest.raises(FileNotFoundError, match="model 'no-such-model'"):
        env.sq.list_sample_evaluations_for_checkpoint("no-such-model", ckpt)
    with pytest.raises(FileNotFoundError, match="no-such-checkpoint"):
        env.sq.list_sample_evaluations_for_checkpoint(env.big_model,
                                                      "no-such-checkpoint")
    # a checkpoint registered under the big model is not the small model's
    # checkpoint: ownership is model-scoped -> FileNotFoundError (404)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        env.sq.list_sample_evaluations_for_checkpoint(env.small_model, ckpt)
    env.no_tmp()


def test_api_by_checkpoint_history_parity_and_metric_values(api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "bysckpt")
    s1 = _gen_sample(api_client, h)
    s2 = _gen_sample(api_client, h, max_new_tokens=4)
    assert api_client.post(_quality_url(h["mid"], s1["sample_id"])
                           ).status_code == 200
    assert api_client.post(_quality_url(h["mid"], s1["sample_id"])
                           ).status_code == 200
    assert api_client.post(_quality_url(h["mid"], s2["sample_id"])
                           ).status_code == 200
    r = api_client.get(_by_ckpt_url(h["mid"], h["ckpt"]))
    assert r.status_code == 200, r.text
    recs = r.json()
    assert len(recs) == 3
    # parity: == full records listing filtered by that checkpoint
    engine = get_forge().list_sample_evaluations(h["mid"])
    expected = [e.model_dump(mode="json") for e in engine
                if e.checkpoint_id == h["ckpt"]]
    assert recs == expected
    assert all(x["checkpoint_id"] == h["ckpt"] for x in recs)
    # M18 listing filtered to that checkpoint == same payloads
    all_recs = api_client.get(_records_url(h["mid"])).json()
    assert recs == [x for x in all_recs if x["checkpoint_id"] == h["ckpt"]]
    # metric values present, no derived-statistic keys
    for x in recs:
        assert "loss_nats" in x and "perplexity" in x
        assert x["loss_nats"] > 0.0 and isinstance(x["perplexity"], float)
    blob = api_client.get(_by_ckpt_url(h["mid"], h["ckpt"])).text
    for banned in ("average", "aggregate", "rank", "score", "trend",
                   "verdict", "mean", "delta", "statistic", "best",
                   "worst"):
        assert banned not in blob.lower(), banned
    # M19 by-sample of s1 remains its own history (unchanged surface)
    b1 = api_client.get(_by_sample_url(h["mid"], s1["sample_id"]))
    assert b1.status_code == 200 and len(b1.json()) == 2
    assert all(x["sample_id"] == s1["sample_id"] for x in b1.json())


def test_api_by_checkpoint_empty_unknown_and_model_isolation(api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "bysckptiso")
    s1 = _gen_sample(api_client, h)
    assert api_client.post(_quality_url(h["mid"], s1["sample_id"])
                           ).status_code == 200
    cks = api_client.get(f"{MODELS}/{h['mid']}/checkpoints").json()
    ckpt_ids = [c["checkpoint_id"] for c in cks]
    assert h["ckpt"] in ckpt_ids and len(ckpt_ids) == 2
    other_ck = [c for c in ckpt_ids if c != h["ckpt"]][0]
    # valid checkpoint of the model with zero measurements -> 200 []
    r = api_client.get(_by_ckpt_url(h["mid"], other_ck))
    assert r.status_code == 200 and r.json() == [], r.text
    # model B (no checkpoints of its own): A's checkpoint id is unknown
    # there -> 404, even though it exists in the storage under model A
    m2 = _http_model_only(api_client, "bysckptiso-b", seed=23)
    assert api_client.get(_by_ckpt_url(m2, h["ckpt"])).status_code == 404
    assert api_client.get(_by_ckpt_url("no-such-model", h["ckpt"])
                          ).status_code == 404
    assert api_client.get(_by_ckpt_url(h["mid"], "no-such-checkpoint")
                          ).status_code == 404
    # model B gets a REAL checkpoint manifest of its own (schema-valid copy
    # of A's with identity fields rewritten) plus one crafted measurement
    # under it: B sees exactly its own record under its own checkpoint; A's
    # by-checkpoint never contains it, and A answers 404 for B's id because
    # that checkpoint id belongs to another model.
    root = get_forge().storage.root
    b_ck = "e-b-ckpt01"
    a_ck = api_client.get(f"{MODELS}/{h['mid']}/checkpoints").json()
    a_ck = next(c for c in a_ck if c["checkpoint_id"] == h["ckpt"])
    b_manifest = dict(a_ck, checkpoint_id=b_ck, model_id=m2)
    ck_dir = root / "models" / m2 / "checkpoints" / b_ck
    ck_dir.mkdir(parents=True)
    (ck_dir / "manifest.json").write_text(json.dumps(b_manifest))
    real = api_client.get(f"{MODELS}/{h['mid']}/sample-quality").json()[0]
    crafted = dict(real, evaluation_id="e-b-ckpt01", model_id=m2,
                   checkpoint_id=b_ck)
    d = root / "sample-evaluations" / m2 / "evaluation-e-b-ckpt01"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(crafted))
    try:
        a = api_client.get(_by_ckpt_url(h["mid"], h["ckpt"]))
        b = api_client.get(_by_ckpt_url(m2, b_ck))
        assert a.status_code == 200 and b.status_code == 200
        assert len(b.json()) == 1 and b.json()[0]["model_id"] == m2
        assert b.json()[0]["evaluation_id"] == "e-b-ckpt01"
        assert len(a.json()) == 1
        assert a.json()[0]["model_id"] == h["mid"]
        assert all(x["model_id"] == h["mid"] for x in a.json())
        # the same checkpoint id under the OTHER model -> 404 (ownership)
        assert api_client.get(_by_ckpt_url(h["mid"], b_ck)
                              ).status_code == 404
        assert api_client.get(_by_ckpt_url(m2, h["ckpt"])
                              ).status_code == 404
    finally:
        shutil.rmtree(root / "models" / m2 / "checkpoints", ignore_errors=True)
        shutil.rmtree(root / "sample-evaluations" / m2, ignore_errors=True)


def test_api_by_checkpoint_deterministic_repeat_and_read_only(api_client):
    from app.engine import get_forge

    h = _http_env(api_client, "bysckptdet")
    s1 = _gen_sample(api_client, h)
    s2 = _gen_sample(api_client, h, max_new_tokens=4)
    for sid in (s1["sample_id"], s2["sample_id"]):
        assert api_client.post(_quality_url(h["mid"], sid)
                               ).status_code == 200
    root = get_forge().storage.root
    before = sorted(p.relative_to(root).as_posix()
                    for p in root.rglob("*") if p.is_file())
    blob = {p: (root / p).read_bytes() for p in before}
    dash0 = api_client.get(f"{MODELS}/{h['mid']}/dashboard").json()
    u = _by_ckpt_url(h["mid"], h["ckpt"])
    r1 = api_client.get(u)
    r2 = api_client.get(u)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.text == r2.text and r1.json() == r2.json()
    # M16/M17/M18/M19 surfaces unchanged by the reads
    assert api_client.get(f"{MODELS}/{h['mid']}/sample-quality").text == \
        api_client.get(f"{MODELS}/{h['mid']}/sample-quality").text
    assert api_client.get(_records_url(h["mid"])).text == \
        api_client.get(_records_url(h["mid"])).text
    assert api_client.get(_by_sample_url(h["mid"], s1["sample_id"])).text == \
        api_client.get(_by_sample_url(h["mid"], s1["sample_id"])).text
    dash1 = api_client.get(f"{MODELS}/{h['mid']}/dashboard").json()
    assert dash1["result_hash"] == dash0["result_hash"] and dash1 == dash0
    # zero writes, zero new files, zero .tmp
    after = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*") if p.is_file())
    assert after == before
    for p in before:
        assert (root / p).read_bytes() == blob[p]
    assert [p for p in root.rglob("*") if ".tmp" in p.name] == []


def test_api_openapi_by_checkpoint_route(api_client):
    spec = api_client.get("/openapi.json").json()
    p = "/api/v1/models/{model_id}/sample-quality/by-checkpoint/{checkpoint_id}"
    assert p in spec["paths"]
    op = spec["paths"][p]["get"]
    assert op["tags"] == ["sample-quality"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["type"] == "array"
    assert schema["items"]["$ref"].endswith("SampleEvaluationRecord")
    # all neighbouring routes stay documented (incl. M18 records + M19
    # by-sample literals, which must precede /{evaluation_id})
    assert "/api/v1/models/{model_id}/sample-quality" in spec["paths"]
    assert "/api/v1/models/{model_id}/sample-quality/records" in spec["paths"]
    assert ("/api/v1/models/{model_id}/sample-quality/by-sample/{sample_id}"
            in spec["paths"])
    assert ("/api/v1/models/{model_id}/sample-quality/{evaluation_id}"
            in spec["paths"])
    # M18–M28 each added exactly one documented route -> 60
    assert len(spec["paths"]) == 70


# =========================================================================== #
# M33: read-only per-tokenizer grouping of the sample-quality history
# =========================================================================== #

BY_TOK33 = "/api/v1/models/{mid}/sample-quality/by-tokenizer/{tok}"


def _m33_state(env):
    """M33 state on top of the shared module env (cached): two measured
    big-model samples under the env's ORIGINAL tokenizer, one measured
    small-model sample under tok_small, plus a fresh third tokenizer
    (m33-tok3) with ZERO measurements anywhere. Returns
    (e1, e2, e3, tok3).
    """
    cached = getattr(env, "_m33", None)
    if cached is not None:
        return cached
    f = env.forge
    s1 = _make_sample(env)
    s2 = _make_sample(env, strategy="temperature", temperature=0.8,
                      seed=33, max_new_tokens=6)
    e1 = env.sq.run(env.big_model, s1.sample_id)
    e2 = env.sq.run(env.big_model, s2.sample_id)
    s_small = env.samples.run(SampleGenerateRequest(
        model_id=env.small_model, checkpoint_id=env.small_ckpt,
        tokenizer_id=env.tok_small.id, prompt=PROMPT,
        strategy="greedy", max_new_tokens=6))
    e3 = env.sq.run(env.small_model, s_small.sample_id)
    tok3 = f.train_tokenizer(
        TokenizerConfig(name="m33-tok3", vocab_size=450),
        dataset_id=env.ds).id
    env._m33 = (e1, e2, e3, tok3)
    return env._m33


def test_m33_engine_grouping_parity_order_verbatim(env):
    e1, e2, e3, tok3 = _m33_state(env)
    f = env.forge
    tok1 = env.tok_big.id
    for model_id, tok in ((env.big_model, tok1),
                          (env.small_model, env.tok_small.id)):
        listing = env.sq.list_sample_evaluations(model_id)
        got = f.list_sample_evaluations_for_tokenizer(model_id, tok)
        # parity with the authoritative M16 listing filtered by the
        # persisted measurement tokenizer identity; deterministic
        # (created_at, evaluation_id) order; membership from the
        # persisted field only; each record EXACTLY ONCE
        assert got == [r for r in listing if r.tokenizer_id == tok]
        keyed = [(r.created_at, r.evaluation_id) for r in got]
        assert keyed == sorted(keyed)
        ids = [r.evaluation_id for r in got]
        assert len(ids) == len(set(ids))
        assert all(r.tokenizer_id == tok and r.model_id == model_id
                   for r in got)
        # verbatim payload parity with the M16 single-record getter
        for r in got:
            assert r == env.sq.get_sample_evaluation(
                model_id, r.evaluation_id)
    # the M33-created measurements sit under exactly their tokenizer
    got1 = f.list_sample_evaluations_for_tokenizer(env.big_model, tok1)
    assert {e1.evaluation_id, e2.evaluation_id} <= \
        {r.evaluation_id for r in got1}
    assert all(r.tokenizer_id == tok1 for r in got1)
    assert e3.evaluation_id in {r.evaluation_id for r in
                                f.list_sample_evaluations_for_tokenizer(
                                    env.small_model, env.tok_small.id)}
    # explicit partition: groups over EVERY tokenizer that actually has
    # records under the big model are pairwise disjoint and cover the
    # listing exactly once (earlier tests may add records — discovery,
    # not fixed pairs)
    listing = env.sq.list_sample_evaluations(env.big_model)
    listing_ids = {r.evaluation_id for r in listing}
    groups = {t: {r.evaluation_id for r in
                  f.list_sample_evaluations_for_tokenizer(env.big_model, t)}
              for t in {r.tokenizer_id for r in listing}}
    flat = [i for g in groups.values() for i in g]
    assert set(flat) == listing_ids
    assert len(flat) == len(set(flat)) == len(listing_ids)
    assert tok1 in groups


def test_m33_engine_empty_404s_cross_model_read_only(env):
    e1, e2, e3, tok3 = _m33_state(env)
    f = env.forge
    # fresh tokenizer with zero measurements anywhere -> []
    assert f.list_sample_evaluations_for_tokenizer(
        env.big_model, tok3) == []
    # model-scoped empty GUARANTEED by the platform vocabulary rule: the
    # small model (vocab 300) can never hold a tok_big (600) record
    assert f.list_sample_evaluations_for_tokenizer(
        env.small_model, env.tok_big.id) == []
    # cross-model isolation: listings are disjoint and no model's group
    # ever contains the other's evaluation ids
    big_ids = {r.evaluation_id
               for r in env.sq.list_sample_evaluations(env.big_model)}
    small_ids = {r.evaluation_id
                 for r in env.sq.list_sample_evaluations(env.small_model)}
    assert big_ids.isdisjoint(small_ids)
    for tok in (env.tok_big.id, env.tok_small.id, tok3):
        bg = {r.evaluation_id for r in
              f.list_sample_evaluations_for_tokenizer(env.big_model, tok)}
        sg = {r.evaluation_id for r in
              f.list_sample_evaluations_for_tokenizer(env.small_model,
                                                      tok)}
        assert bg <= big_ids and sg <= small_ids
        assert bg.isdisjoint(sg)
    # unknown model / unknown tokenizer -> FileNotFoundError (404 at API)
    with pytest.raises(FileNotFoundError):
        f.list_sample_evaluations_for_tokenizer(
            "ghost-model-33", env.tok_big.id)
    with pytest.raises(FileNotFoundError):
        f.list_sample_evaluations_for_tokenizer(
            env.big_model, "ghost-tok-33")
    # read-only: the filter never writes anything
    root = f.storage.root
    before = {p.relative_to(root).as_posix()
              for p in root.rglob("*") if p.is_file()}
    f.list_sample_evaluations_for_tokenizer(env.big_model,
                                            env.tok_big.id)
    f.list_sample_evaluations_for_tokenizer(env.big_model, tok3)
    f.list_sample_evaluations_for_tokenizer(env.small_model,
                                            env.tok_small.id)
    f.list_sample_evaluations_for_tokenizer(env.small_model,
                                            env.tok_big.id)
    after = {p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_m33_engine_repeated_calls_identical(env):
    e1, e2, e3, tok3 = _m33_state(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_sample_evaluations_for_tokenizer(env.big_model,
                                                     env.tok_big.id)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_sample_evaluations_for_tokenizer(
                     env.big_model, env.tok_big.id)]
        assert again == first


def _gen33(h, **overrides) -> dict:
    body = {"model_id": h["mid"], "checkpoint_id": h["ckpt"],
            "tokenizer_id": h["tok"], "prompt": PROMPT,
            "strategy": "greedy", "max_new_tokens": 6}
    body.update(overrides)
    return body


def _train_extra_tokenizer33(api_client, tag: str, ds_id: str,
                             vocab: int) -> str:
    tr = api_client.post("/api/v1/tokenizers/train",
                         data={"config": json.dumps(
                             {"name": f"api16-{tag}-tok",
                              "vocab_size": vocab}),
                               "dataset_id": ds_id})
    assert tr.status_code == 201, tr.text
    return tr.json()["tokenizer"]["id"]


def test_m33_api_by_tokenizer_grouping_partition_determinism(api_client):
    h = _http_env(api_client, "m33a")
    # a fresh dataset hosts two extra tokenizers: tok2 (with measured
    # samples) and tok3 (zero records anywhere); both fit vocab 640
    up = api_client.post("/api/v1/datasets/upload",
                         files=[("files", ("m33a2.txt",
                                           _corpus(120, "m33a2"),
                                           "text/plain"))],
                         data={"name": "api16-m33a2-ds"})
    assert up.status_code == 201, up.text
    ds2 = up.json()["dataset_id"]
    tok2 = _train_extra_tokenizer33(api_client, "m33a2", ds2, 500)
    tok3 = _train_extra_tokenizer33(api_client, "m33a3", ds2, 450)

    s1 = api_client.post(GENERATE, json=_gen33(h)).json()
    e1 = api_client.post(_quality_url(h["mid"],
                                      s1["sample_id"])).json()
    s2 = api_client.post(GENERATE, json=_gen33(
        h, strategy="temperature", temperature=0.8, seed=7)).json()
    e2 = api_client.post(_quality_url(h["mid"],
                                      s2["sample_id"])).json()
    s3 = api_client.post(GENERATE, json=_gen33(
        h, tokenizer_id=tok2)).json()
    e3 = api_client.post(_quality_url(h["mid"],
                                      s3["sample_id"])).json()
    e4 = api_client.post(_quality_url(h["mid"],
                                      s3["sample_id"])).json()

    url = BY_TOK33.format(mid=h["mid"], tok=h["tok"])
    got = api_client.get(url)
    assert got.status_code == 200, got.text
    recs = got.json()
    # authoritative-filter parity: exact subset of the M16 listing
    # whose persisted tokenizer_id matches, in the same order
    listing = api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality").json()
    assert recs == [x for x in listing
                    if x["tokenizer_id"] == h["tok"]]
    assert {x["evaluation_id"] for x in recs} == {e1["evaluation_id"],
                                                  e2["evaluation_id"]}
    keyed = [(x["created_at"], x["evaluation_id"]) for x in recs]
    assert keyed == sorted(keyed)
    # verbatim: each element equals its POST payload and its
    # detail-getter payload
    by_id = {x["evaluation_id"]: x for x in recs}
    assert by_id[e1["evaluation_id"]] == e1
    assert by_id[e2["evaluation_id"]] == e2
    for x in recs:
        one = api_client.get(
            f"{MODELS}/{h['mid']}/sample-quality/{x['evaluation_id']}")
        assert one.status_code == 200 and one.json() == x
    # deterministic: three repeats return identical raw bytes
    raws = {api_client.get(url).content for _ in range(3)}
    assert len(raws) == 1
    # partition: tok2 holds exactly its own two records (one sample
    # measured twice -> distinct evaluation ids, same tokenizer),
    # disjoint from the original tokenizer's group, together the full
    # listing
    recs2 = api_client.get(
        BY_TOK33.format(mid=h["mid"], tok=tok2)).json()
    assert {x["evaluation_id"] for x in recs2} == {e3["evaluation_id"],
                                                   e4["evaluation_id"]}
    assert len({x["evaluation_id"] for x in recs2}) == 2
    ids1 = {x["evaluation_id"] for x in recs}
    ids2 = {x["evaluation_id"] for x in recs2}
    assert ids1.isdisjoint(ids2)
    assert ids1 | ids2 == {x["evaluation_id"] for x in listing}
    # valid tokenizer with zero records for the model -> [] (200)
    empty = api_client.get(BY_TOK33.format(mid=h["mid"], tok=tok3))
    assert empty.status_code == 200 and empty.json() == []
    # no side effects: the filter itself added no records
    assert len(api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality").json()) == 4


def test_m33_api_404s_isolation_regressions_openapi(api_client):
    h = _http_env(api_client, "m33b")
    h2 = _http_env(api_client, "m33c")     # second real model
    s1 = api_client.post(GENERATE, json=_gen33(h)).json()
    e1 = api_client.post(_quality_url(h["mid"],
                                      s1["sample_id"])).json()

    # 404s: unknown model / unknown tokenizer (two ghost forms)
    assert api_client.get(BY_TOK33.format(mid="ghost-model-33",
                                          tok=h["tok"])).status_code == 404
    assert api_client.get(BY_TOK33.format(mid=h["mid"],
                                          tok="ghost-tok-33")).status_code == 404
    assert api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality/by-tokenizer/"
        "m33--not-a-real-tokenizer-id").status_code == 404
    assert api_client.get(BY_TOK33.format(mid="ghost-model-33",
                                          tok="ghost-tok-33")).status_code == 404

    # cross-model isolation: tokenizers are global, but the other
    # model's group is empty (scoping from the model's own listing)
    iso = api_client.get(BY_TOK33.format(mid=h2["mid"], tok=h["tok"]))
    assert iso.status_code == 200 and iso.json() == []

    # M16 generic listing/getter + M18 records unchanged
    listing = api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality").json()
    assert [x["evaluation_id"] for x in listing] == [e1["evaluation_id"]]
    got = api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality/{e1['evaluation_id']}")
    assert got.status_code == 200 and got.json() == e1
    records = api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality/records").json()
    assert records == listing
    # M19 by-sample / M20 by-checkpoint regressions: same listing,
    # other groupings, unconfused
    bysample = api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality/by-sample/"
        f"{s1['sample_id']}").json()
    assert [x["evaluation_id"] for x in bysample] == [e1["evaluation_id"]]
    byck = api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality/by-checkpoint/"
        f"{h['ckpt']}").json()
    assert [x["evaluation_id"] for x in byck] == [e1["evaluation_id"]]
    # generic detail getter still 404s ghost ids (no route capture)
    assert api_client.get(
        f"{MODELS}/{h['mid']}/sample-quality/ghost-eval-33"
    ).status_code == 404
    # M30 evaluation-by-tokenizer regression: parity with the M4
    # listing filtered by persisted tokenizer identity
    evs = api_client.get(f"{MODELS}/{h['mid']}/evaluations").json()
    evt = api_client.get(f"{MODELS}/{h['mid']}/evaluations/by-tokenizer/"
                         f"{h['tok']}").json()
    assert evt == [e for e in evs if e["tokenizer_id"] == h["tok"]]
    # M32 samples-by-tokenizer regression: the measured sample sits in
    # its tokenizer's sample group
    smp = api_client.get(f"{MODELS}/{h['mid']}/samples/by-tokenizer/"
                         f"{h['tok']}").json()
    assert [x["sample_id"] for x in smp] == [s1["sample_id"]]

    # OpenAPI: 70 paths, the new path exactly once, GET-only, tag
    # sample-quality, SampleEvaluationRecord items; route order M20
    # by-checkpoint < by-tokenizer < generic detail
    spec = api_client.get("/openapi.json").json()
    # 55 (pre-M18) + 1 (M18) + 1 (M24) + 1 (M25) + 1 (M26) + 1 (M27)
    # + 1 (M28) + 1 (M29) + 1 (M30 evaluations by-tokenizer)
    # + 1 (M31 comparisons by-tokenizer) + 1 (M32 samples by-tokenizer)
    # + 1 (M33 sample-quality by-tokenizer)
    # + 1 (M34 gate decisions by-comparison)
    # + 1 (M35 workflows by-recipe)
    # + 1 (M36 evaluations by-split)
    # + 1 (M37 comparisons by-split)
    # + 1 (M38 evaluations by-state-kind) = 70
    assert len(spec["paths"]) == 70
    path = ("/api/v1/models/{model_id}/sample-quality/by-tokenizer"
            "/{tokenizer_id}")
    keys = list(spec["paths"])
    assert keys.count(path) == 1
    item = spec["paths"][path]
    assert list(item.keys()) == ["get"]
    assert item["get"]["tags"] == ["sample-quality"]
    schema = item["get"]["responses"]["200"]["content"][
        "application/json"]["schema"]
    assert schema["type"] == "array" and schema["items"] == {
        "$ref": "#/components/schemas/SampleEvaluationRecord"}
    assert "post" not in item
    assert keys.index("/api/v1/models/{model_id}/sample-quality/"
                      "by-checkpoint/{checkpoint_id}") < keys.index(path)
    assert keys.index(path) < keys.index(
        "/api/v1/models/{model_id}/sample-quality/{evaluation_id}")
