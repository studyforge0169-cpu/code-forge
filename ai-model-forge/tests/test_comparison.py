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
    EvalStateKind,
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


# =========================================================================== #
# M26: read-only per-checkpoint grouping of the comparison history
# =========================================================================== #

def _m26_env(env):
    """M26 state on top of the shared module env (cached): comparisons
    spread over three checkpoints — a cross-checkpoint A/B, a
    same-checkpoint A=B, a current-vs-checkpoint and a current-vs-current
    — plus a second trained model whose checkpoint has zero comparisons.
    Returns (ckE, ckF, ckR, c_x, c_same, c_cur_ck, c_cur_cur, b, b_ck).
    """
    cached = getattr(env, "_m26_state", None)
    if cached is not None:
        return cached
    f = env.forge
    ckE = env.imp_early.checkpoint_id
    ckF = env.imp_final.checkpoint_id
    ckR = env.reg_final.checkpoint_id
    c_x = f.run_comparison(env.cmp(ckE, ckF, seed=2601))
    c_same = f.run_comparison(env.cmp(ckE, ckE, seed=2602))    # A = B
    c_cur_ck = f.run_comparison(env.cmp(ckR, ckR, seed=2603).model_copy(
        update={"state_a": ComparisonState(state_kind="current")}))
    c_cur_cur = f.run_comparison(env.cmp(ckF, ckF, seed=2604).model_copy(
        update={"state_a": ComparisonState(state_kind="current"),
                "state_b": ComparisonState(state_kind="current")}))
    b = f.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m26-b", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
        seed=26)))[0].id
    up = f.upload_dataset([("m26b.txt", _domain_bytes(TAIL_A, 240))],
                          name="m26-b-ds")
    tb = f.train_tokenizer(TokenizerConfig(name="m26-b-tok", vocab_size=600),
                           dataset_id=up["dataset_id"])
    f.tokenize_dataset(up["dataset_id"], tb.id)
    f.run_training(TrainingConfig(
        method="continued_pretraining", model_id=b, dataset_id=up["dataset_id"],
        tokenizer_id=tb.id, learning_rate=3e-3, batch_size=8, max_seq_len=32,
        steps=8, eval_every_steps=4, keep_best=False, seed=26))
    b_ck = f.list_checkpoints(b)[0].checkpoint_id
    env._m26_state = (ckE, ckF, ckR, c_x, c_same, c_cur_ck, c_cur_cur,
                      b, b_ck)
    return env._m26_state


def test_m26_engine_filters_by_persisted_sides_and_model(env):
    ckE, ckF, ckR, c_x, c_same, c_cur_ck, c_cur_cur, b, b_ck = _m26_env(env)
    f = env.forge
    listing = f.list_comparisons(env.model_id)

    def expected(ck):
        return [r for r in listing
                if any(s.state_kind == EvalStateKind.CHECKPOINT
                       and s.checkpoint_id == ck
                       for s in (r.state_a, r.state_b))]

    for ck in (ckE, ckF, ckR):
        got = f.list_comparisons_for_checkpoint(env.model_id, ck)
        # parity with the authoritative M5 listing filtered by the
        # persisted side states; deterministic (created_at, comparison_id)
        # order; every record involves the checkpoint on >= 1 side
        assert got == expected(ck)
        assert [(r.created_at, r.comparison_id) for r in got] == \
            sorted((r.created_at, r.comparison_id) for r in got)
        assert all(any(s.state_kind == EvalStateKind.CHECKPOINT
                       and s.checkpoint_id == ck
                       for s in (r.state_a, r.state_b)) for r in got)
        assert all(r.model_id == env.model_id for r in got)
        # unique comparison identities (a record never appears twice)
        ids = [r.comparison_id for r in got]
        assert len(ids) == len(set(ids))
        # verbatim payload parity with the M5 single-record getter
        for r in got:
            assert r == f.get_comparison(env.model_id, r.comparison_id)
    # same-checkpoint A=B comparison appears EXACTLY ONCE under ckE
    got_e = f.list_comparisons_for_checkpoint(env.model_id, ckE)
    assert [r.comparison_id for r in got_e].count(c_same.comparison_id) == 1
    # current-vs-checkpoint: appears under ckR exactly once (its current
    # side never matches anything; earlier module tests may have added
    # further ckR records, hence the count-based check)
    got_r = f.list_comparisons_for_checkpoint(env.model_id, ckR)
    assert [r.comparison_id for r in got_r].count(
        c_cur_ck.comparison_id) == 1
    # current-vs-current comparison appears under NO checkpoint
    for ck in (ckE, ckF, ckR):
        assert c_cur_cur.comparison_id not in \
            [r.comparison_id for r in
             f.list_comparisons_for_checkpoint(env.model_id, ck)]


def test_m26_engine_empty_404s_cross_model_and_read_only(env):
    ckE, ckF, ckR, c_x, c_same, c_cur_ck, c_cur_cur, b, b_ck = _m26_env(env)
    f = env.forge
    # valid registered checkpoint with no comparisons -> []
    assert f.list_comparisons_for_checkpoint(b, b_ck) == []
    # cross-model: neither model can resolve the other's checkpoint id
    with pytest.raises(FileNotFoundError):
        f.list_comparisons_for_checkpoint(b, ckE)
    with pytest.raises(FileNotFoundError):
        f.list_comparisons_for_checkpoint(env.model_id, b_ck)
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_comparisons_for_checkpoint("ghost-model-26", ckE)
    # unknown checkpoint -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_comparisons_for_checkpoint(env.model_id, "ghost-ck-26")
    # read-only: the filter never writes comparison manifests
    root = f.storage.model_dir(env.model_id) / "comparisons"
    before = {p.relative_to(root).as_posix()
              for p in root.rglob("*") if p.is_file()}
    f.list_comparisons_for_checkpoint(env.model_id, ckE)
    f.list_comparisons_for_checkpoint(b, b_ck)
    after = {p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_m26_engine_repeated_calls_identical(env):
    ckE, _, _, _, _, _, _, _, _ = _m26_env(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_comparisons_for_checkpoint(env.model_id, ckE)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_comparisons_for_checkpoint(env.model_id, ckE)]
        assert again == first


# =========================================================================== #
# M29: read-only per-dataset grouping of the comparison history
# =========================================================================== #

def _m29_env(env):
    """M29 state on top of the shared module env (cached): ds_b pinned to
    BOTH of its versions (v1 + v2) plus a third dataset with zero
    comparisons. Returns (c_b1, c_b2, ds_c).
    """
    cached = getattr(env, "_m29_state", None)
    if cached is not None:
        return cached
    f = env.forge
    ckE = env.imp_early.checkpoint_id
    ckF = env.imp_final.checkpoint_id
    if 2 not in f.datasets.load_meta(env.ds_b).versions:
        up = f.upload_dataset(
            [("b2.txt", _domain_bytes(TAIL_B, 240))],
            name="m5-domB", dataset_id=env.ds_b)
        assert up["version"] == 2
        f.tokenize_dataset(env.ds_b, env.tok_id, version=2)
    c_b1 = f.run_comparison(env.cmp(ckE, ckF, ds="ds_b",
                                    dataset_version=1, seed=2901))
    c_b2 = f.run_comparison(env.cmp(ckE, ckF, ds="ds_b",
                                    dataset_version=2, seed=2902))
    up_c = f.upload_dataset(
        [("c.txt", _domain_bytes(TAIL_A, 240))], name="m5-domC")
    ds_c = up_c["dataset_id"]
    f.tokenize_dataset(ds_c, env.tok_id)
    env._m29_state = (c_b1, c_b2, ds_c)
    return env._m29_state


def test_m29_engine_filters_by_persisted_dataset_identity(env):
    c_b1, c_b2, ds_c = _m29_env(env)
    f = env.forge
    listing = f.list_comparisons(env.model_id)
    for ds_id in (env.ds_a, env.ds_b, ds_c):
        got = f.list_comparisons_for_dataset(env.model_id, ds_id)
        # parity with the authoritative M5 listing filtered by the
        # persisted shared-probe dataset identity; deterministic
        # (created_at, comparison_id) order; unique comparison ids
        assert got == [r for r in listing if r.dataset_id == ds_id]
        keyed = [(r.created_at, r.comparison_id) for r in got]
        assert keyed == sorted(keyed)
        ids = [r.comparison_id for r in got]
        assert len(ids) == len(set(ids))
        assert all(r.dataset_id == ds_id and r.model_id == env.model_id
                   for r in got)
        for r in got:
            assert r == f.get_comparison(env.model_id, r.comparison_id)
    # ds_b holds exactly the two pinned comparisons, one per version,
    # versions VERBATIM (v1 and v2 both returned — never collapsed to
    # the latest, never rewritten)
    got_b = f.list_comparisons_for_dataset(env.model_id, env.ds_b)
    assert [r.comparison_id for r in got_b] == \
        [c_b1.comparison_id, c_b2.comparison_id]
    assert [r.dataset_version for r in got_b] == [1, 2]
    # each matching comparison appears EXACTLY ONCE even though BOTH
    # sides measure the requested dataset (shared probe; includes the
    # same-checkpoint A=B records from earlier module tests under ds_a)
    got_a = f.list_comparisons_for_dataset(env.model_id, env.ds_a)
    assert all([r.comparison_id for r in got_a].count(r.comparison_id)
               == 1 for r in got_a)
    # unrelated-dataset exclusion: no overlap between dataset groups
    assert ({r.comparison_id for r in got_a}
            & {r.comparison_id for r in got_b}) == set()
    # structural fact: the persisted side representation carries NO
    # dataset fields — the dataset identity is the ONE shared probe
    # persisted once at the top level of the record
    assert not any("dataset" in k
                   for k in type(got_b[0].state_a).model_fields)


def test_m29_engine_empty_404s_isolation_read_only(env):
    c_b1, c_b2, ds_c = _m29_env(env)
    _, _, _, _, _, _, _, b, b_ck = _m26_env(env)
    f = env.forge
    # valid dataset with zero comparisons -> []
    assert f.list_comparisons_for_dataset(env.model_id, ds_c) == []
    # model-scoped empty: the second model has no comparisons at all
    assert f.list_comparisons_for_dataset(b, env.ds_a) == []
    # unknown model / unknown dataset -> FileNotFoundError (404 at API)
    with pytest.raises(FileNotFoundError):
        f.list_comparisons_for_dataset("ghost-model-29", env.ds_a)
    with pytest.raises(FileNotFoundError):
        f.list_comparisons_for_dataset(env.model_id, "ghost-ds-29")
    # cross-model isolation: a's comparison ids never appear under b
    leak = [r.comparison_id for r in
            f.list_comparisons_for_dataset(b, env.ds_a)]
    assert {c_b1.comparison_id, c_b2.comparison_id}.isdisjoint(leak)
    # read-only: the filter never writes comparison manifests
    def comp_files(mid):
        root = f.storage.model_dir(mid) / "comparisons"
        if not root.exists():
            return set()
        return {p.relative_to(root).as_posix()
                for p in root.rglob("*") if p.is_file()}
    before = (comp_files(env.model_id), comp_files(b))
    f.list_comparisons_for_dataset(env.model_id, env.ds_a)
    f.list_comparisons_for_dataset(env.model_id, env.ds_b)
    f.list_comparisons_for_dataset(b, env.ds_a)
    f.list_comparisons_for_dataset(env.model_id, ds_c)
    assert (comp_files(env.model_id), comp_files(b)) == before


def test_m29_engine_repeated_calls_identical(env):
    c_b1, c_b2, ds_c = _m29_env(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_comparisons_for_dataset(env.model_id, env.ds_b)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_comparisons_for_dataset(env.model_id, env.ds_b)]
        assert again == first
