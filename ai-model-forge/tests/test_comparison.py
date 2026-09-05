"""Milestone 5 tests: comparison engine (A/B states over identical probes).

Scenarios are built with two deterministic domains that share heads but use
different verb tails (the M3 conflict recipe):
  * IMPROVEMENT: checkpoints of one model trained longer on domain A are
    better on the A probe (measured delta ≈ -2.7 nats).
  * REGRESSION: after continued training on domain B the same model's
    checkpoint regresses on the A probe (measured delta ≈ +0.5 nats).
All comparisons reuse verified M4 evaluations; nothing is trained inside the
comparison logic.
"""
from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from app.schemas import (
    ComparisonRequest,
    ComparisonState,
    EvaluationConfig,
    ModelCreateRequest,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
)

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


class Env:
    """One temp root with two domains, one tokenizer, one trained model."""

    def __init__(self, root):
        from app.engine import ModelForge

        self.forge = ModelForge(root=root)

    def prepare(self):
        f = self.forge
        up_a = f.upload_dataset([("a.txt", _domain_bytes(TAIL_A, 240))], name="m5-domA")
        up_b = f.upload_dataset([("b.txt", _domain_bytes(TAIL_B, 240))], name="m5-domB")
        self.ds_a, self.ds_b = up_a["dataset_id"], up_b["dataset_id"]
        tok = f.train_tokenizer(TokenizerConfig(name="m5-tok", vocab_size=600),
                                dataset_id=self.ds_a)
        self.tok_id = tok.id
        f.tokenize_dataset(self.ds_a, tok.id)
        f.tokenize_dataset(self.ds_b, tok.id)

        cfg = TransformerConfig(name="m5-main", vocab_size=640, context_length=64,
                                hidden_size=64, n_layers=2, n_heads=4,
                                n_kv_heads=2, intermediate_size=128, seed=1)
        self.model_id = f.create_model(ModelCreateRequest(config=cfg))[0].id

        self.tokens_a = f.datasets.tokenized_artifact(
            self.ds_a, 1, tok.id)[0].splits["train"].count
        self.tokens_b = f.datasets.tokenized_artifact(
            self.ds_b, 1, tok.id)[0].splits["train"].count

        # run 1: 30 epochs on domain A -> monotone improvement on the A probe
        r1 = f.run_training(self._train(self.ds_a, 30))
        ck = r1.checkpoints
        assert len(ck) == 15, len(ck)  # eval every 2 epochs
        self.imp_early = f.get_checkpoint(self.model_id, ck[2]["checkpoint_id"])
        self.tol_ckpt = f.get_checkpoint(self.model_id, ck[-3]["checkpoint_id"])
        self.imp_final = f.get_checkpoint(self.model_id, ck[-1]["checkpoint_id"])
        assert self.imp_early.validation_loss > self.imp_final.validation_loss

        # run 2: continue the same model on domain B -> regression on the A probe
        r2 = f.run_training(self._train(self.ds_b, 40))
        self.reg_final = f.get_checkpoint(self.model_id,
                                          r2.checkpoints[-1]["checkpoint_id"])

    def _train(self, ds_id: str, epochs: int) -> TrainingConfig:
        tokens = self.tokens_a if ds_id == self.ds_a else self.tokens_b
        sp_epoch = max(1, (tokens // 32) // 8)
        return TrainingConfig(method="continued_pretraining", model_id=self.model_id,
                              dataset_id=ds_id, tokenizer_id=self.tok_id,
                              learning_rate=3e-3, batch_size=8, max_seq_len=32,
                              epochs=epochs, eval_every_steps=sp_epoch * 2,
                              keep_best=False, seed=1)

    # ------------------------------------------------------------------ #

    def state(self, kind: str, ckpt_id=None) -> dict:
        return {"state_kind": kind,
                **({"checkpoint_id": ckpt_id} if ckpt_id else {})}

    def cmp(self, a_id: str, b_id: str, split: str = "validation",
            ds: str = "ds_a", tolerance: float = 1e-4, **overrides) -> ComparisonRequest:
        base = dict(model_id=self.model_id,
                    state_a=self.state("checkpoint", a_id),
                    state_b=self.state("checkpoint", b_id),
                    dataset_id=getattr(self, ds), split=split,
                    tokenizer_id=self.tok_id, batch_size=8, max_seq_len=32,
                    seed=1, tolerance=tolerance)
        base.update(overrides)
        return ComparisonRequest(**base)

    def eval_direct(self, ckpt_id, split="validation", ds="ds_a", window=32,
                    tokenizer=None, version=None, seed=1) -> EvaluationConfig:
        return EvaluationConfig(
            model_id=self.model_id, checkpoint_id=ckpt_id,
            dataset_id=getattr(self, ds), dataset_version=version,
            split=split, tokenizer_id=tokenizer or self.tok_id,
            batch_size=8, max_seq_len=window, seed=seed)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m5-root"))
    e.prepare()
    return e


@pytest.fixture(scope="module")
def other(env):
    """Second model with an incompatible architecture (vocab 320 / hidden 96),
    evaluated with its own small tokenizer on domain B."""
    forge = env.forge
    tok2 = forge.train_tokenizer(TokenizerConfig(name="m5-other-tok", vocab_size=320),
                                 dataset_id=env.ds_b)
    forge.tokenize_dataset(env.ds_b, tok2.id)
    cfg = TransformerConfig(name="m5-other", vocab_size=320, context_length=64,
                            hidden_size=96, n_layers=2, n_heads=6,
                            n_kv_heads=2, intermediate_size=192, seed=9)
    model_id = forge.create_model(ModelCreateRequest(config=cfg))[0].id
    return model_id, tok2.id


# =========================================================================== #
# Test 1 — improvement scenario
# =========================================================================== #

def test_improvement_scenario(env):
    early, final = env.imp_early, env.imp_final
    rec = env.forge.run_comparison(env.cmp(early.checkpoint_id, final.checkpoint_id))
    assert rec.verdict.value == "improved"
    assert rec.delta_loss_nats < -1.0          # measured ≈ -2.7
    assert rec.loss_a == pytest.approx(early.validation_loss, rel=1e-3)
    assert rec.loss_b == pytest.approx(final.validation_loss, rel=1e-3)
    # evidence chains: state hashes == checkpoint manifests, eval ids real
    assert rec.state_a.state_hash == early.weights_sha256
    assert rec.state_b.state_hash == final.weights_sha256
    assert rec.state_a.evaluation_id != rec.state_b.evaluation_id
    for side, ckpt in ((rec.state_a, early), (rec.state_b, final)):
        ev = env.forge.get_evaluation(env.model_id, side.evaluation_id)
        assert ev.state_kind.value == "checkpoint"
        assert ev.checkpoint_id == ckpt.checkpoint_id
        assert ev.state_hash == ckpt.weights_sha256


# =========================================================================== #
# Test 2 — regression scenario
# =========================================================================== #

def test_regression_scenario(env):
    # state A: after domain-A training; state B: same model after domain-B training
    a, b = env.imp_final, env.reg_final
    rec = env.forge.run_comparison(env.cmp(a.checkpoint_id, b.checkpoint_id))
    assert rec.verdict.value == "regressed"
    assert rec.delta_loss_nats > 0.2           # measured ≈ +0.5
    assert rec.loss_b > rec.loss_a
    # sanity: B really learned domain B (not degenerate)
    ev = env.forge.get_evaluation(env.model_id, rec.state_b.evaluation_id)
    assert ev.loss_nats < 3.0


# =========================================================================== #
# Test 3 — unchanged (same state against itself)
# =========================================================================== #

def test_unchanged_same_state(env):
    ckpt = env.imp_final.checkpoint_id
    rec = env.forge.run_comparison(env.cmp(ckpt, ckpt))
    assert rec.verdict.value == "unchanged"
    assert rec.delta_loss_nats == 0.0
    assert rec.state_a.evaluation_id == rec.state_b.evaluation_id  # one record
    assert rec.state_a.state_hash == rec.state_b.state_hash


# =========================================================================== #
# Test 4 — tolerance flips the verdict
# =========================================================================== #

def test_tolerance_controls_verdict(env):
    a, b = env.tol_ckpt.checkpoint_id, env.imp_final.checkpoint_id  # Δ ≈ +0.0011
    wide = env.forge.run_comparison(env.cmp(a, b, tolerance=0.01))
    assert wide.verdict.value == "unchanged"
    strict = env.forge.run_comparison(env.cmp(a, b, tolerance=1e-6))
    assert strict.verdict.value == "regressed"
    # the other direction flips to improved under the strict tolerance
    rev = env.forge.run_comparison(env.cmp(b, a, tolerance=1e-6))
    assert rev.verdict.value == "improved"
    assert strict.loss_a == wide.loss_a and strict.loss_b == wide.loss_b
    assert wide.tolerance == 0.01 and strict.tolerance == 1e-6


# =========================================================================== #
# Test 5/6 — deterministic comparison + evaluation reuse
# =========================================================================== #

def test_deterministic_repeat_and_evaluation_reuse(env):
    mid = env.model_id
    probe = dict(split="train", seed=999)      # unique probe for exact counting
    n_evals0 = len(env.forge.list_evaluations(mid))

    r1 = env.forge.run_comparison(env.cmp(env.imp_early.checkpoint_id,
                                          env.imp_final.checkpoint_id, **probe))
    assert len(env.forge.list_evaluations(mid)) == n_evals0 + 2  # both created

    r2 = env.forge.run_comparison(env.cmp(env.imp_early.checkpoint_id,
                                          env.imp_final.checkpoint_id, **probe))
    # second run: NO new evaluations (exact reuse), new comparison manifest
    assert len(env.forge.list_evaluations(mid)) == n_evals0 + 2
    assert r1.state_a.evaluation_id == r2.state_a.evaluation_id
    assert r1.state_b.evaluation_id == r2.state_b.evaluation_id
    # metrics + result_hash + verdict identical, ids/timestamps may differ
    for f in ("loss_a", "loss_b", "delta_loss_nats", "verdict", "result_hash"):
        assert getattr(r1, f) == getattr(r2, f), f
    assert r1.comparison_id != r2.comparison_id
    assert len(env.forge.list_comparisons(mid)) >= 2


def test_stale_evaluation_never_reused_for_different_probe(env):
    """An evaluation on a different probe must not satisfy the exact lookup."""
    mid = env.model_id
    before = len(env.forge.list_evaluations(mid))
    # these states already have validation/seed-1 evaluations (earlier tests)
    # and train/seed-999 ones (reuse test); train/seed-1 is fresh for both
    rec = env.forge.run_comparison(env.cmp(env.imp_early.checkpoint_id,
                                           env.imp_final.checkpoint_id,
                                           split="train", seed=1))
    new_evals = [r for r in env.forge.list_evaluations(mid)
                 if r.split.value == "train" and r.seed == 1]
    assert len(env.forge.list_evaluations(mid)) == before + 2  # both created
    assert len(new_evals) == 2
    assert {r.eval_id for r in new_evals} == {rec.state_a.evaluation_id,
                                              rec.state_b.evaluation_id}


# =========================================================================== #
# Tests 7-11 — cross-probe guards (record-level path)
# =========================================================================== #

def _two_side(env, left: EvaluationConfig, right: EvaluationConfig, tolerance=1e-4):
    a = env.forge.evaluation.run(left)
    b = env.forge.evaluation.run(right)
    with pytest.raises(ValueError, match="probe"):
        env.forge.comparison.compare_records(a, b, tolerance=tolerance)


def test_cross_split_refused(env):
    ck = env.imp_final.checkpoint_id
    _two_side(env, env.eval_direct(ck, split="validation"),
              env.eval_direct(ck, split="test"))


def test_cross_dataset_refused(env):
    ck = env.imp_final.checkpoint_id
    _two_side(env, env.eval_direct(ck, ds="ds_a"),
              env.eval_direct(ck, ds="ds_b"))


def test_cross_version_refused(env):
    f = env.forge
    up = f.upload_dataset([("b2.txt", _domain_bytes(TAIL_B, 240))],
                          name="m5-domB", dataset_id=env.ds_b)
    assert up["version"] == 2
    f.tokenize_dataset(env.ds_b, env.tok_id, version=2)
    ck = env.imp_final.checkpoint_id
    _two_side(env, env.eval_direct(ck, ds="ds_b", version=1),
              env.eval_direct(ck, ds="ds_b", version=2))


def test_cross_tokenizer_refused(env):
    f = env.forge
    tok2 = f.train_tokenizer(TokenizerConfig(name="m5-tok2", vocab_size=320),
                             dataset_id=env.ds_b)
    f.tokenize_dataset(env.ds_a, tok2.id)
    ck = env.imp_final.checkpoint_id
    _two_side(env, env.eval_direct(ck),
              env.eval_direct(ck, tokenizer=tok2.id))


def test_cross_window_refused(env):
    ck = env.imp_final.checkpoint_id
    _two_side(env, env.eval_direct(ck, window=32),
              env.eval_direct(ck, window=64))


# =========================================================================== #
# Test 12 — incompatible model configurations refused
# =========================================================================== #

def test_incompatible_configurations_refused(env, other):
    other_id, other_tok = other
    ck = env.imp_final.checkpoint_id
    a = env.forge.evaluation.run(env.eval_direct(ck))
    b = env.forge.evaluation.run(EvaluationConfig(
        model_id=other_id, dataset_id=env.ds_b, split="validation",
        tokenizer_id=other_tok, batch_size=8, max_seq_len=32, seed=1))
    before = len(env.forge.list_comparisons(env.model_id))
    with pytest.raises(ValueError, match="incompatible"):
        env.forge.comparison.compare_records(a, b)
    assert len(env.forge.list_comparisons(env.model_id)) == before  # nothing kept


# =========================================================================== #
# Test 13 — corrupt checkpoint refused before anything is written
# =========================================================================== #

def test_corrupt_checkpoint_refused(env):
    f = env.forge
    # dedicated model + mini run so no shared fixture checkpoint is harmed
    model_id = f.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m5-corrupt", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128, seed=5)))[0].id
    f.run_training(TrainingConfig(method="continued_pretraining", model_id=model_id,
                                  dataset_id=env.ds_a, tokenizer_id=env.tok_id,
                                  learning_rate=3e-3, batch_size=8, max_seq_len=32,
                                  epochs=2, eval_every_steps=7, keep_best=False, seed=1))
    ck = f.list_checkpoints(model_id)[-1]
    wpath = f.training._ckpt_dir(model_id, ck.checkpoint_id) / "weights.pt"

    import torch
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    tampered = {k: v.clone() for k, v in state.items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    weights_before = f.storage.weights_path(model_id).read_bytes()
    comps_before = len(f.list_comparisons(model_id))
    evals_before = len(f.list_evaluations(model_id))
    with pytest.raises(RuntimeError, match="integrity"):
        f.run_comparison(env.cmp(ck.checkpoint_id, ck.checkpoint_id, tolerance=0.1)
                         .model_copy(update={"model_id": model_id}))
    # refused before any evaluation or comparison artifact appeared
    assert f.storage.weights_path(model_id).read_bytes() == weights_before
    assert len(f.list_comparisons(model_id)) == comps_before
    assert len(f.list_evaluations(model_id)) == evals_before


# =========================================================================== #
# Test 14 — read-only comparison
# =========================================================================== #

def test_comparison_is_read_only(env):
    f = env.forge
    mid = env.model_id
    model_dir = f.storage.model_dir(mid)
    before_manifest = (model_dir / "manifest.json").read_bytes()
    before_weights = f.storage.weights_path(mid).read_bytes()
    ckpt_root = f.training._ckpt_root(mid)
    ckpt_before = {p.relative_to(ckpt_root).as_posix(): p.read_bytes()
                   for p in ckpt_root.rglob("*") if p.is_file()}
    eval_root = f.evaluation._evals_root(mid)
    eval_before = {p.relative_to(eval_root).as_posix(): p.read_bytes()
                   for p in eval_root.rglob("*") if p.is_file()}
    prov_before = [p.model_dump(mode="json")
                   for p in f.get_model(mid).training_provenance]

    # fresh probe (seed 77) so exactly two new evaluations must appear
    rec = env.forge.run_comparison(env.cmp(env.imp_early.checkpoint_id,
                                           env.imp_final.checkpoint_id, seed=77))
    assert rec.delta_loss_nats < -1.0

    assert (model_dir / "manifest.json").read_bytes() == before_manifest
    assert f.storage.weights_path(mid).read_bytes() == before_weights
    ckpt_after = {p.relative_to(ckpt_root).as_posix(): p.read_bytes()
                  for p in ckpt_root.rglob("*") if p.is_file()}
    assert ckpt_after == ckpt_before
    assert [p.model_dump(mode="json")
            for p in f.get_model(mid).training_provenance] == prov_before
    eval_after = {p.relative_to(eval_root).as_posix(): p.read_bytes()
                  for p in eval_root.rglob("*") if p.is_file()}
    new_evals = set(eval_after) - set(eval_before)
    assert len(new_evals) == 2 and all("manifest.json" in k for k in new_evals)
    assert all(eval_after[k] == eval_before[k] for k in eval_before)  # untouched


# =========================================================================== #
# Tests 15-17 — list/get semantics
# =========================================================================== #

def test_list_get_comparisons(env):
    lst = env.forge.list_comparisons(env.model_id)
    assert lst == sorted(lst, key=lambda r: (r.created_at, r.comparison_id))
    assert len(lst) >= 6
    one = env.forge.get_comparison(env.model_id, lst[0].comparison_id)
    assert one.comparison_id == lst[0].comparison_id
    with pytest.raises(FileNotFoundError):
        env.forge.get_comparison(env.model_id, "ghost")
    with pytest.raises(FileNotFoundError):
        env.forge.list_comparisons("ghost")
    with pytest.raises(FileNotFoundError):
        env.forge.get_comparison("ghost", "ghost")


def test_unknown_model_refused(env):
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_comparison(env.cmp(env.imp_final.checkpoint_id,
                                         env.imp_final.checkpoint_id)
                                 .model_copy(update={"model_id": "ghost"}))


def test_empty_list_for_known_model(env):
    from app.engine import ModelForge
    fresh = ModelForge(root=env.forge.storage.root)
    model_id = fresh.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m5-fresh", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128)))[0].id
    assert fresh.list_comparisons(model_id) == []


# =========================================================================== #
# Test 18 — missing/invalid states
# =========================================================================== #

def test_missing_and_invalid_states(env):
    # unknown checkpoint id
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_comparison(env.cmp("ghost", env.imp_final.checkpoint_id))
    # schema: contradictory state combinations
    with pytest.raises(ValidationError):
        ComparisonState(state_kind="current", checkpoint_id="x")
    with pytest.raises(ValidationError):
        ComparisonState(state_kind="checkpoint")
    with pytest.raises(ValidationError):
        ComparisonRequest(model_id="m", state_a={"state_kind": "checkpoint"},
                          state_b={"state_kind": "current"},
                          dataset_id="d", split="validation", tokenizer_id="t")


def test_current_state_without_weights_refused(env):
    fresh_id = env.forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m5-naked", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128)))[0].id
    w = env.forge.storage.weights_path(fresh_id)
    w.unlink()
    (env.forge.storage.model_dir(fresh_id) / "weights.sha256").unlink()
    req = env.cmp(env.imp_final.checkpoint_id, env.imp_final.checkpoint_id)
    req = req.model_copy(update={
        "model_id": fresh_id,
        "state_a": ComparisonState(state_kind="current")})
    with pytest.raises(FileNotFoundError, match="no weights"):
        env.forge.run_comparison(req)


# =========================================================================== #
# Test 19 — result hash stability
# =========================================================================== #

def test_result_hash_stability(env):
    a_id, b_id = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    r1 = env.forge.run_comparison(env.cmp(a_id, b_id, seed=123))
    r2 = env.forge.run_comparison(env.cmp(a_id, b_id, seed=123))
    assert r1.result_hash == r2.result_hash
    # hash changes when the probe identity, tolerance or state changes
    assert r1.result_hash != env.forge.run_comparison(
        env.cmp(a_id, b_id, seed=124)).result_hash
    assert r1.result_hash != env.forge.run_comparison(
        env.cmp(a_id, b_id, seed=123, tolerance=0.5)).result_hash
    assert r1.result_hash != env.forge.run_comparison(
        env.cmp(b_id, b_id, seed=123)).result_hash


# =========================================================================== #
# Test 20 — comparison metrics exactly match the referenced evaluations
# =========================================================================== #

def test_comparison_metrics_match_referenced_evaluations(env):
    mid = env.model_id
    rec = env.forge.run_comparison(env.cmp(env.imp_early.checkpoint_id,
                                           env.imp_final.checkpoint_id, seed=321))
    for side, side_loss, side_ppl in ((rec.state_a, rec.loss_a, rec.perplexity_a),
                                      (rec.state_b, rec.loss_b, rec.perplexity_b)):
        ev = env.forge.get_evaluation(mid, side.evaluation_id)
        assert ev.loss_nats == side_loss              # exact, no recomputation
        assert ev.perplexity == side_ppl
        assert ev.token_count == side.token_count
        assert ev.result_hash == side.evaluation_result_hash
        assert ev.state_hash == side.state_hash
    assert rec.delta_loss_nats == pytest.approx(rec.loss_b - rec.loss_a)
    assert rec.delta_perplexity == pytest.approx(rec.perplexity_b - rec.perplexity_a)


# =========================================================================== #
# Extra — current vs checkpoint state kinds remain comparable
# =========================================================================== #

def test_current_vs_checkpoint_comparison(env):
    # after run 2 (keep_best=False) the current weights equal the regression
    # final checkpoint, so current-vs-that-checkpoint must be unchanged
    from app.model_builder import content_hash
    current_hash = content_hash(env.forge.storage.load_weights(env.model_id))
    assert current_hash == env.reg_final.weights_sha256
    req = env.cmp(env.reg_final.checkpoint_id, env.reg_final.checkpoint_id,
                  seed=555).model_copy(update={
                      "state_a": ComparisonState(state_kind="current")})
    rec = env.forge.run_comparison(req)
    assert rec.state_a.state_kind.value == "current"
    assert rec.state_b.state_kind.value == "checkpoint"
    assert rec.state_a.state_hash == rec.state_b.state_hash
    assert rec.verdict.value == "unchanged"
