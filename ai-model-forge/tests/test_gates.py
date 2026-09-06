"""Milestone 6 tests: stage-gate engine (policy-driven run decisions).

Uses the M5 two-domain recipe on a module-scoped environment:
  * domain A training improves on the A probe across checkpoints (early ->
    final measured delta ≈ -2.7 nats);
  * continued training on domain B regresses on the A probe (measured
    delta ≈ +0.5 nats against the domain-A final checkpoint);
  * a near-final A checkpoint pair measures Δ ≈ +0.0011 nats (tolerance
    boundary material).

A gate never trains, rolls back or selects anything — a failed checkpoint
gate only carries a suggested rollback target (exercised by the user via the
M3 endpoint). Evidence is reused by identity; each explicit gate request
appends exactly one new immutable decision manifest.
"""
from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from app.schemas import (
    ComparisonState,
    EvaluationConfig,
    GateDecisionResult,
    GatePolicy,
    GateRequest,
    ModelCreateRequest,
    PolicyCreateRequest,
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
        up_a = f.upload_dataset([("a.txt", _domain_bytes(TAIL_A, 240))], name="m6-domA")
        up_b = f.upload_dataset([("b.txt", _domain_bytes(TAIL_B, 240))], name="m6-domB")
        self.ds_a, self.ds_b = up_a["dataset_id"], up_b["dataset_id"]
        tok = f.train_tokenizer(TokenizerConfig(name="m6-tok", vocab_size=600),
                                dataset_id=self.ds_a)
        self.tok_id = tok.id
        f.tokenize_dataset(self.ds_a, tok.id)
        f.tokenize_dataset(self.ds_b, tok.id)

        cfg = TransformerConfig(name="m6-main", vocab_size=640, context_length=64,
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
        assert len(ck) == 15, len(ck)
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
                **( {"checkpoint_id": ckpt_id} if ckpt_id else {})}

    def policy(self, baseline_type: str = "checkpoint", baseline_ckpt=None,
               baseline_hash=None, minimum_loss=None, max_regression_delta=None,
               tolerance: float = 1e-4, split: str = "validation", ds: str = "ds_a",
               window: int = 32, seed=1, name: str = "m6-gate", model_id=None,
               **overrides) -> GatePolicy:
        base = dict(name=name, model_id=model_id or self.model_id,
                    dataset_id=getattr(self, ds), split=split,
                    tokenizer_id=self.tok_id, batch_size=8, max_seq_len=window,
                    seed=seed, tolerance=tolerance,
                    baseline_type=baseline_type,
                    baseline_checkpoint_id=baseline_ckpt,
                    baseline_result_hash=baseline_hash,
                    minimum_loss=minimum_loss,
                    max_regression_delta=max_regression_delta)
        base.update(overrides)
        if base["baseline_checkpoint_id"] is None:
            del base["baseline_checkpoint_id"]
        if base["baseline_result_hash"] is None:
            del base["baseline_result_hash"]
        if base["minimum_loss"] is None:
            del base["minimum_loss"]
        if base["max_regression_delta"] is None:
            del base["max_regression_delta"]
        return GatePolicy(**base)

    def gate(self, policy: GatePolicy, cand_kind: str = "checkpoint",
             cand_ckpt=None, model_id=None) -> GateRequest:
        return GateRequest(model_id=model_id or self.model_id, policy=policy,
                           candidate=self.state(cand_kind, cand_ckpt))

    # ------------------------------------------------------------------ #

    def run_gate(self, *, baseline_type="checkpoint", baseline_ckpt=None,
                 cand_ckpt=None, model_id=None, **policy_kwargs):
        mid = model_id or self.model_id
        pol = self.policy(baseline_type=baseline_type,
                          baseline_ckpt=baseline_ckpt, model_id=mid,
                          **policy_kwargs)
        cand_kind = "current" if cand_ckpt is None else "checkpoint"
        return self.forge.run_gate(self.gate(pol, cand_kind=cand_kind,
                                             cand_ckpt=cand_ckpt,
                                             model_id=mid))

    def gates_root(self, model_id=None) -> list:
        root = self.forge.storage.model_dir(model_id or self.model_id) / "gates"
        return root if root.exists() else []


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m6-root"))
    e.prepare()
    return e


# =========================================================================== #
# Pass semantics
# =========================================================================== #

def test_improvement_candidate_passes(env):
    dec = env.run_gate(baseline_ckpt=env.imp_early.checkpoint_id,
                       cand_ckpt=env.imp_final.checkpoint_id, seed=11)
    assert dec.decision == GateDecisionResult.PASSED
    assert dec.verdict.value == "improved"
    assert dec.delta_loss_nats < -1.0                # measured ≈ -2.7
    assert dec.suggested_checkpoint_id is None and dec.hint is None
    # policy is embedded as evaluated, with the dataset version resolved
    assert dec.policy.dataset_version == 1
    assert dec.policy.model_id == env.model_id
    assert dec.config_hash
    assert dec.baseline is not None and dec.candidate is not None
    assert dec.candidate.state_kind.value == "checkpoint"
    assert dec.baseline.checkpoint_id == env.imp_early.checkpoint_id
    assert dec.candidate.checkpoint_id == env.imp_final.checkpoint_id
    # evidence: state hashes == checkpoint manifests, referenced evals real
    assert dec.baseline.state_hash == env.imp_early.weights_sha256
    assert dec.candidate.state_hash == env.imp_final.weights_sha256
    ev = env.forge.get_evaluation(env.model_id, dec.candidate.evaluation_id)
    assert ev.checkpoint_id == env.imp_final.checkpoint_id
    assert ev.state_hash == env.imp_final.weights_sha256
    assert ev.result_hash == dec.candidate.evaluation_result_hash


def test_unchanged_same_state_passes(env):
    ck = env.imp_final.checkpoint_id
    dec = env.run_gate(baseline_ckpt=ck, cand_ckpt=ck, seed=12)
    assert dec.verdict.value == "unchanged"
    assert dec.delta_loss_nats == 0.0
    assert dec.decision == GateDecisionResult.PASSED
    assert dec.baseline_loss == dec.candidate_loss
    # identical states share one evaluation record (M5 convention)
    assert dec.candidate.evaluation_id == dec.baseline.evaluation_id


def test_regression_within_max_regression_delta_passes(env):
    a = env.imp_final.checkpoint_id          # best on A
    b = env.reg_final.checkpoint_id          # B-trained, regressed on A probe
    dec = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=13,
                       max_regression_delta=1.0)
    assert dec.verdict.value == "regressed"
    assert dec.delta_loss_nats > 0.2         # measured ≈ +0.5
    assert dec.decision == GateDecisionResult.PASSED  # bounded regression
    assert "max_regression_delta" in dec.reason
    assert dec.suggested_checkpoint_id is None
    assert dec.hint is None


def test_absolute_threshold_met_passes(env):
    dec = env.run_gate(baseline_type="minimum_loss", minimum_loss=1e9,
                       cand_ckpt=env.imp_final.checkpoint_id, seed=14)
    assert dec.decision == GateDecisionResult.PASSED
    assert dec.baseline is None                      # threshold-only: no state
    assert dec.comparison_id is None                 # nothing fabricated
    assert dec.comparison_result_hash is None
    assert dec.verdict is None and dec.delta_loss_nats is None
    assert dec.baseline_loss is None
    assert dec.candidate_loss == dec.candidate.loss_nats
    assert dec.suggested_checkpoint_id is None


# =========================================================================== #
# Fail semantics
# =========================================================================== #

def test_regression_fails_without_max_regression_delta(env):
    a = env.imp_final.checkpoint_id
    b = env.reg_final.checkpoint_id
    dec = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=15)
    assert dec.verdict.value == "regressed"
    assert dec.decision == GateDecisionResult.FAILED
    assert "regressed" in dec.reason
    # failed against a checkpoint baseline -> rollback is *suggested* only
    assert dec.suggested_checkpoint_id == a
    assert dec.hint == "rollback recommended"
    assert dec.comparison_id


def test_regression_past_max_regression_delta_fails(env):
    a = env.imp_final.checkpoint_id
    b = env.reg_final.checkpoint_id
    dec = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=16,
                       max_regression_delta=0.1)
    assert dec.verdict.value == "regressed"
    assert dec.decision == GateDecisionResult.FAILED
    assert "max_regression_delta" in dec.reason
    assert dec.suggested_checkpoint_id == a          # suggest, never execute


def test_regression_at_exact_max_regression_boundary(env):
    """Delta exactly at max_regression_delta -> passed; just below -> failed.

    The comparison stores delta rounded to 6 decimals (M5 convention), so the
    boundary is exercised at half-ulp margins around the stored delta.
    """
    a, b = env.imp_final.checkpoint_id, env.reg_final.checkpoint_id
    first = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=17)
    assert first.decision == GateDecisionResult.FAILED
    d = first.delta_loss_nats                       # stored 6-decimal delta
    assert d > 0.2
    cmp_before = len(env.forge.list_comparisons(env.model_id))

    dec_at = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=17,
                          max_regression_delta=d + 1e-6)
    assert dec_at.verdict.value == "regressed"
    assert dec_at.decision == GateDecisionResult.PASSED
    assert dec_at.delta_loss_nats == d              # same evidence reused

    dec_under = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=17,
                             max_regression_delta=d - 1e-6)
    assert dec_under.verdict.value == "regressed"
    assert dec_under.decision == GateDecisionResult.FAILED
    assert len(env.forge.list_comparisons(env.model_id)) == cmp_before
    assert first.comparison_id == dec_at.comparison_id
    assert first.comparison_id == dec_under.comparison_id


def test_minimum_loss_ceiling_fails_a_passing_candidate(env):
    """An improved/unchanged candidate still fails an absolute ceiling."""
    a = env.imp_early.checkpoint_id
    b = env.imp_final.checkpoint_id
    ceiling = env.imp_final.validation_loss * 0.99   # below the candidate loss
    dec = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=18,
                       minimum_loss=ceiling)
    assert dec.verdict.value == "improved"           # loss-only verdict: honest
    assert dec.decision == GateDecisionResult.FAILED  # policy semantics
    assert "minimum_loss" in dec.reason and "ceiling" in dec.reason
    assert dec.suggested_checkpoint_id == a
    # with a generous ceiling the same evidence passes
    ok = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=18,
                      minimum_loss=env.imp_final.validation_loss * 1.5)
    assert ok.verdict.value == "improved"
    assert ok.decision == GateDecisionResult.PASSED
    assert ok.comparison_id == dec.comparison_id     # evidence reused


def test_absolute_threshold_violated_fails(env):
    dec = env.run_gate(baseline_type="minimum_loss", minimum_loss=1e-6,
                       cand_ckpt=env.imp_final.checkpoint_id, seed=19)
    assert dec.decision == GateDecisionResult.FAILED
    assert "threshold" in dec.reason
    assert dec.baseline is None and dec.comparison_id is None
    assert dec.suggested_checkpoint_id is None       # nothing to roll back to


# =========================================================================== #
# Tolerance boundary
# =========================================================================== #

def test_delta_exactly_at_tolerance_unchanged_passes(env):
    """Same state => delta exactly 0.0; tolerance 0.0 is the inclusive edge."""
    ck = env.imp_final.checkpoint_id
    dec = env.run_gate(baseline_ckpt=ck, cand_ckpt=ck, tolerance=0.0, seed=20)
    assert dec.verdict.value == "unchanged"
    assert dec.decision == GateDecisionResult.PASSED
    assert dec.delta_loss_nats == 0.0


def test_delta_just_outside_tolerance_regressed_fails(env):
    """The near-final pair measures Δ ≈ +0.0011; 1e-4 sits just outside."""
    a, b = env.tol_ckpt.checkpoint_id, env.imp_final.checkpoint_id
    dec = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=21)  # tol 1e-4
    assert dec.verdict.value == "regressed"
    assert dec.decision == GateDecisionResult.FAILED
    assert dec.delta_loss_nats > 1e-4
    # the same pair under a wide tolerance is unchanged -> passes
    wide = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=21, tolerance=0.01)
    assert wide.verdict.value == "unchanged"
    assert wide.decision == GateDecisionResult.PASSED


# =========================================================================== #
# Baseline forms
# =========================================================================== #

def test_current_weights_baseline(env):
    # after run 2 (keep_best=False) the live weights equal reg_final
    a = env.reg_final.checkpoint_id
    dec = env.run_gate(baseline_type="current", cand_ckpt=a, seed=22)
    assert dec.verdict.value == "unchanged"
    assert dec.decision == GateDecisionResult.PASSED
    assert dec.baseline.state_kind.value == "current"
    assert dec.baseline.checkpoint_id is None
    assert dec.baseline.state_hash == env.reg_final.weights_sha256
    assert dec.candidate.state_hash == env.reg_final.weights_sha256
    # candidate from the past (better on A) vs current (B-trained) -> improved
    imp = env.run_gate(baseline_type="current",
                       cand_ckpt=env.imp_final.checkpoint_id, seed=23)
    assert imp.verdict.value == "improved"
    assert imp.decision == GateDecisionResult.PASSED


def test_evaluation_result_hash_baseline(env):
    """A previously recorded evaluation result may be the baseline (type C)."""
    f = env.forge
    probe = dict(seed=2024, split="validation")
    rec = f.evaluation.run(EvaluationConfig(
        model_id=env.model_id, checkpoint_id=env.imp_final.checkpoint_id,
        dataset_id=env.ds_a, split=probe["split"], tokenizer_id=env.tok_id,
        batch_size=8, max_seq_len=32, seed=probe["seed"]))

    # worse candidate against the recorded result -> regressed -> failed
    pol = env.policy(baseline_type="evaluation_result_hash",
                     baseline_hash=rec.result_hash, **probe)
    dec = f.run_gate(env.gate(pol, cand_ckpt=env.imp_early.checkpoint_id))
    assert dec.decision == GateDecisionResult.FAILED
    assert dec.verdict.value == "regressed"
    assert dec.delta_loss_nats > 1.0          # early is ≈2.7 nats worse
    assert dec.baseline.evaluation_id == rec.eval_id
    assert dec.baseline.evaluation_result_hash == rec.result_hash
    assert dec.baseline.state_hash == env.imp_final.weights_sha256
    assert dec.comparison_id

    # identical candidate vs its own recorded result -> unchanged -> passed
    pol2 = env.policy(baseline_type="evaluation_result_hash",
                      baseline_hash=rec.result_hash, **probe)
    dec2 = f.run_gate(env.gate(pol2, cand_ckpt=env.imp_final.checkpoint_id))
    assert dec2.verdict.value == "unchanged"
    assert dec2.decision == GateDecisionResult.PASSED


def test_evaluation_result_hash_probe_mismatch_rejected(env):
    """A valid-hash evaluation on a DIFFERENT probe is never the baseline."""
    f = env.forge
    rec = f.evaluation.run(EvaluationConfig(
        model_id=env.model_id, checkpoint_id=env.imp_final.checkpoint_id,
        dataset_id=env.ds_a, split="validation", tokenizer_id=env.tok_id,
        batch_size=8, max_seq_len=32, seed=3030))
    n_evals = len(f.list_evaluations(env.model_id))
    n_gates = len([p for p in env.gates_root().iterdir() if p.is_dir()])
    # policy wants seed 3031 — the stored record (seed 3030) must not match
    pol = env.policy(baseline_type="evaluation_result_hash",
                     baseline_hash=rec.result_hash, seed=3031)
    with pytest.raises(ValueError, match="unrelated evaluation"):
        f.run_gate(env.gate(pol, cand_ckpt=env.imp_final.checkpoint_id))
    assert len(f.list_evaluations(env.model_id)) == n_evals   # nothing ran
    assert len([p for p in env.gates_root().iterdir() if p.is_dir()]) == n_gates


def test_evaluation_result_hash_missing_is_404(env):
    pol = env.policy(baseline_type="evaluation_result_hash",
                     baseline_hash="f" * 64, seed=4040)
    with pytest.raises(FileNotFoundError, match="no evaluation"):
        env.forge.run_gate(env.gate(pol, cand_ckpt=env.imp_final.checkpoint_id))


# =========================================================================== #
# Reuse, determinism, auditability
# =========================================================================== #

def test_repeat_gate_reuses_evidence_and_appends_decision(env):
    f = env.forge
    mid = env.model_id
    probe = dict(split="train", seed=4242)      # unique probe for counting
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    n_ev0 = len(f.list_evaluations(mid))
    n_cm0 = len(f.list_comparisons(mid))
    n_g0 = len(f.list_gate_decisions(mid))

    d1 = env.run_gate(baseline_ckpt=a, cand_ckpt=b, **probe)
    assert len(f.list_evaluations(mid)) == n_ev0 + 2
    assert len(f.list_comparisons(mid)) == n_cm0 + 1
    assert len(f.list_gate_decisions(mid)) == n_g0 + 1

    d2 = env.run_gate(baseline_ckpt=a, cand_ckpt=b, **probe)
    # evidence deduplicated, decision appended (auditability)
    assert len(f.list_evaluations(mid)) == n_ev0 + 2
    assert len(f.list_comparisons(mid)) == n_cm0 + 1
    assert len(f.list_gate_decisions(mid)) == n_g0 + 2
    assert d1.candidate.evaluation_id == d2.candidate.evaluation_id
    assert d1.baseline.evaluation_id == d2.baseline.evaluation_id
    assert d1.comparison_id == d2.comparison_id    # same evidence by identity
    assert d1.result_hash == d2.result_hash
    assert d1.decision_id != d2.decision_id
    # two distinct immutable artifacts exist on disk
    gdirs = {p.name for p in env.gates_root().iterdir()
             if p.is_dir() and p.name.startswith("gate-")}
    assert {f"gate-{d1.decision_id}", f"gate-{d2.decision_id}"} <= gdirs
    assert len(gdirs) == n_g0 + 2


def test_result_hash_determinism_and_sensitivity(env):
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id

    def hash_of(**kw):
        return env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=5150, **kw).result_hash

    base = hash_of()
    assert base == hash_of()                      # repeatable
    assert len(base) == 64
    assert base != hash_of(tolerance=0.5)         # tolerance is semantic
    assert base != hash_of(max_regression_delta=5.0)
    assert base != hash_of(minimum_loss=0.5)
    assert env.run_gate(baseline_ckpt=b, cand_ckpt=a, seed=5150).result_hash != base
    # threshold-only decisions hash deterministically too
    t1 = env.run_gate(baseline_type="minimum_loss", minimum_loss=1e9,
                      cand_ckpt=b, seed=5151)
    t2 = env.run_gate(baseline_type="minimum_loss", minimum_loss=1e9,
                      cand_ckpt=b, seed=5151)
    assert t1.result_hash == t2.result_hash
    assert t1.decision_id != t2.decision_id


def test_evidence_chain_matches_records_exactly(env):
    """GateDecision losses/verdict/delta equal the referenced M4/M5 records."""
    f = env.forge
    mid = env.model_id
    a, b = env.imp_early.checkpoint_id, env.imp_final.checkpoint_id
    dec = env.run_gate(baseline_ckpt=a, cand_ckpt=b, seed=6160)
    ev_a = f.get_evaluation(mid, dec.baseline.evaluation_id)
    ev_b = f.get_evaluation(mid, dec.candidate.evaluation_id)
    cmp_rec = f.get_comparison(mid, dec.comparison_id)
    assert dec.baseline_loss == ev_a.loss_nats == cmp_rec.loss_a
    assert dec.candidate_loss == ev_b.loss_nats == cmp_rec.loss_b
    assert dec.delta_loss_nats == cmp_rec.delta_loss_nats
    assert dec.verdict == cmp_rec.verdict
    assert dec.comparison_result_hash == cmp_rec.result_hash
    assert dec.tolerance == cmp_rec.tolerance
    assert dec.baseline.state_hash == ev_a.state_hash == env.imp_early.weights_sha256
    assert dec.candidate.state_hash == ev_b.state_hash == env.imp_final.weights_sha256
    # persisted manifest equals the returned record byte-for-byte (JSON)
    mpath = (env.gates_root() / f"gate-{dec.decision_id}" / "manifest.json")
    import json
    assert json.loads(mpath.read_text()) == dec.model_dump(mode="json")


def test_list_get_semantics_and_immutability(env):
    f = env.forge
    mid = env.model_id
    lst = f.list_gate_decisions(mid)
    assert lst == sorted(lst, key=lambda r: (r.created_at, r.decision_id))
    assert len(lst) >= 10
    one = f.get_gate_decision(mid, lst[0].decision_id)
    assert one.model_dump() == lst[0].model_dump()
    with pytest.raises(FileNotFoundError):
        f.get_gate_decision(mid, "ghost")
    with pytest.raises(FileNotFoundError):
        f.list_gate_decisions("ghost")
    with pytest.raises(FileNotFoundError):
        f.get_gate_decision("ghost", "ghost")
    # a known model with no gates lists empty
    fresh = f.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m6-fresh", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128)))[0].id
    assert f.list_gate_decisions(fresh) == []


# =========================================================================== #
# Integrity & safety
# =========================================================================== #

@pytest.fixture(scope="module")
def corrupt_env(env, tmp_path_factory):
    """A tiny dedicated model with trained checkpoints for corruption tests."""
    f = env.forge
    model_id = f.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m6-corrupt", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128, seed=5)))[0].id
    f.run_training(TrainingConfig(method="continued_pretraining", model_id=model_id,
                                  dataset_id=env.ds_a, tokenizer_id=env.tok_id,
                                  learning_rate=3e-3, batch_size=8, max_seq_len=32,
                                  epochs=2, eval_every_steps=7, keep_best=False,
                                  seed=1))
    return model_id


def test_corrupt_baseline_checkpoint_refused_before_persistence(env, corrupt_env):
    import torch

    f = env.forge
    mid = corrupt_env
    ckpts = f.list_checkpoints(mid)
    victim, good = ckpts[-1], ckpts[0]
    wpath = f.training._ckpt_dir(mid, victim.checkpoint_id) / "weights.pt"
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    tampered = {k: v.clone() for k, v in state.items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    weights_before = f.storage.weights_path(mid).read_bytes()
    ev0 = len(f.list_evaluations(mid))
    cm0 = len(f.list_comparisons(mid))
    with pytest.raises(RuntimeError, match="integrity"):
        env.run_gate(baseline_ckpt=victim.checkpoint_id,
                     cand_ckpt=good.checkpoint_id, model_id=mid, seed=90)
    # refused before any evaluation/comparison/decision artifact appeared
    assert f.storage.weights_path(mid).read_bytes() == weights_before
    assert len(f.list_evaluations(mid)) == ev0
    assert len(f.list_comparisons(mid)) == cm0
    assert f.list_gate_decisions(mid) == []


def test_corrupt_candidate_state_refused_before_persistence(env, corrupt_env):
    import torch

    f = env.forge
    mid = corrupt_env
    victim = f.list_checkpoints(mid)[-1]
    wpath = f.training._ckpt_dir(mid, victim.checkpoint_id) / "weights.pt"
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    tampered = {k: v.clone() for k, v in state.items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    weights_before = f.storage.weights_path(mid).read_bytes()
    ev0 = len(f.list_evaluations(mid))
    with pytest.raises(RuntimeError, match="integrity"):
        # candidate corrupt; baseline current is fine — still nothing persists
        env.run_gate(baseline_type="current", cand_ckpt=victim.checkpoint_id,
                     model_id=mid, seed=91)
    assert f.storage.weights_path(mid).read_bytes() == weights_before
    assert len(f.list_evaluations(mid)) == ev0
    assert f.list_gate_decisions(mid) == []


def test_gate_is_read_only_and_rollback_never_executed(env):
    """A failed gate writes ONLY its decision manifest; it suggests rollback."""
    f = env.forge
    mid = env.model_id
    model_dir = f.storage.model_dir(mid)
    snap = lambda: {  # noqa: E731
        p.relative_to(model_dir).as_posix(): p.read_bytes()
        for p in model_dir.rglob("*") if p.is_file()}
    before = snap()
    weights_before = f.storage.weights_path(mid).read_bytes()
    prov_before = [p.model_dump(mode="json")
                   for p in f.get_model(mid).training_provenance]
    latest_before = f.get_model(mid).latest_checkpoint

    dec = env.run_gate(baseline_ckpt=env.imp_final.checkpoint_id,
                       cand_ckpt=env.reg_final.checkpoint_id, seed=93)
    assert dec.decision == GateDecisionResult.FAILED
    assert dec.suggested_checkpoint_id == env.imp_final.checkpoint_id
    assert dec.hint == "rollback recommended"

    after = snap()
    added = set(after) - set(before)
    # exactly: 2 fresh evaluations + 1 comparison + the new decision manifest
    expected = {
        f"evaluations/eval-{dec.baseline.evaluation_id}/manifest.json",
        f"evaluations/eval-{dec.candidate.evaluation_id}/manifest.json",
        f"comparisons/comp-{dec.comparison_id}/manifest.json",
        f"gates/gate-{dec.decision_id}/manifest.json",
    }
    assert added == expected
    for k in before:
        assert after[k] == before[k]                # everything else untouched
    assert f.storage.weights_path(mid).read_bytes() == weights_before
    assert f.get_model(mid).latest_checkpoint == latest_before
    assert [p.model_dump(mode="json")
            for p in f.get_model(mid).training_provenance] == prov_before
    # the suggested target is a real, existing M3 checkpoint
    f.get_checkpoint(mid, dec.suggested_checkpoint_id)


def test_suggestion_only_for_checkpoint_baselines(env):
    """A failed gate against 'current' has nothing to suggest."""
    dec = env.run_gate(baseline_type="current",
                       cand_ckpt=env.imp_early.checkpoint_id, seed=94)
    assert dec.verdict.value == "regressed"      # current(B) better than early(A)
    assert dec.decision == GateDecisionResult.FAILED
    assert dec.suggested_checkpoint_id is None
    assert dec.hint is None


# =========================================================================== #
# Validation & 404 surface
# =========================================================================== #

def test_policy_schema_rejects_ambiguous_baselines(env):
    def pol(**kw):
        return GatePolicy(model_id=env.model_id, dataset_id=env.ds_a,
                          split="validation", tokenizer_id=env.tok_id, **kw)

    bad = [
        dict(baseline_type="checkpoint"),                         # no checkpoint id
        dict(baseline_type="checkpoint",
             baseline_checkpoint_id="c1", baseline_result_hash="f" * 64),
        dict(baseline_type="current", baseline_checkpoint_id="c1"),
        dict(baseline_type="current", baseline_result_hash="f" * 64),
        dict(baseline_type="evaluation_result_hash"),             # no hash
        dict(baseline_type="evaluation_result_hash",
             baseline_result_hash="f" * 64, baseline_checkpoint_id="c1"),
        dict(baseline_type="minimum_loss", minimum_loss=1.0,
             baseline_checkpoint_id="c1"),
        dict(baseline_type="minimum_loss", minimum_loss=1.0,
             max_regression_delta=0.5),                           # meaningless
        dict(baseline_type="minimum_loss"),                       # no threshold
        dict(baseline_type="checkpoint", baseline_checkpoint_id="c1",
             max_regression_delta=0.0, tolerance=1e-2),           # max < tol
        dict(baseline_type="current", tolerance=-0.1),
        dict(baseline_type="current", batch_size=0),
        dict(baseline_type="current", max_seq_len=1),
    ]
    for kw in bad:
        with pytest.raises(ValidationError):
            pol(**kw)


def test_request_schema_rejections(env):
    good = env.policy(baseline_type="checkpoint",
                      baseline_ckpt=env.imp_final.checkpoint_id)
    with pytest.raises(ValidationError):          # policy.model_id mismatch
        env.gate(good, model_id="other")
    with pytest.raises(ValidationError):          # candidate current + ckpt id
        env.gate(good, cand_ckpt=env.imp_final.checkpoint_id).model_copy(
            update={"candidate": ComparisonState(state_kind="current",
                                                 checkpoint_id="x")})
    with pytest.raises(ValidationError):          # checkpoint without id
        env.gate(good).model_copy(
            update={"candidate": {"state_kind": "checkpoint"}})
    with pytest.raises(ValidationError):          # extra keys forbidden
        GateRequest(model_id=env.model_id, policy=good,
                    candidate={"state_kind": "current"}, stray=1)
    with pytest.raises(ValidationError):          # unknown baseline_type
        env.policy(baseline_type="zombie")


def test_missing_artifacts_404(env):
    f = env.forge
    a = env.imp_final.checkpoint_id
    # unknown model
    pol = env.policy(model_id="ghost", baseline_type="checkpoint", baseline_ckpt=a)
    with pytest.raises(FileNotFoundError):
        f.run_gate(GateRequest(model_id="ghost", policy=pol,
                               candidate={"state_kind": "checkpoint",
                                          "checkpoint_id": a}))
    # unknown dataset / tokenizer / version
    p1 = env.policy(baseline_type="checkpoint", baseline_ckpt=a,
                    dataset_id="ghost")
    with pytest.raises(FileNotFoundError):
        f.run_gate(env.gate(p1, cand_ckpt=a))
    p2 = env.policy(baseline_type="checkpoint", baseline_ckpt=a,
                    tokenizer_id="ghost")
    with pytest.raises(FileNotFoundError):
        f.run_gate(env.gate(p2, cand_ckpt=a))
    p3 = env.policy(baseline_type="checkpoint", baseline_ckpt=a,
                    dataset_version=99)
    with pytest.raises(FileNotFoundError):
        f.run_gate(env.gate(p3, cand_ckpt=a))
    # unknown checkpoint (baseline and candidate)
    p4 = env.policy(baseline_type="checkpoint", baseline_ckpt="ghost")
    with pytest.raises(FileNotFoundError, match="not found"):
        f.run_gate(env.gate(p4, cand_ckpt=a))
    p5 = env.policy(baseline_type="checkpoint", baseline_ckpt=a)
    with pytest.raises(FileNotFoundError, match="not found"):
        f.run_gate(env.gate(p5, cand_ckpt="ghost"))
    # current state with no weights (remove the initial weights like M5 did)
    naked = f.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m6-naked", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128)))[0].id
    w = f.storage.weights_path(naked)
    w.unlink()
    (f.storage.model_dir(naked) / "weights.sha256").unlink()
    p6 = env.policy(model_id=naked, baseline_type="current")
    with pytest.raises(FileNotFoundError, match="no weights"):
        f.run_gate(env.gate(p6, cand_kind="current", model_id=naked))


def test_probe_compatibility_rejected(env):
    """Probes the engine cannot run (window beyond context) -> ValueError."""
    a = env.imp_final.checkpoint_id
    p = env.policy(baseline_type="checkpoint", baseline_ckpt=a, max_seq_len=128)
    with pytest.raises(ValueError):
        env.forge.run_gate(env.gate(p, cand_ckpt=a))


# =========================================================================== #
# M23: read-only per-policy grouping of the immutable gate-decision history
# =========================================================================== #

def _m23_env(env):
    """Two models, three registered policies, decisions spread across them
    (model_a: pol-a twice + pol-b once + one INLINE decision; model_b:
    its own pol-bb once). Built once per module env and cached — every
    M23 engine test sees the exact same immutable history. Returns
    (d1, d2, d3, d_inline, model_b, d_b).
    """
    cached = getattr(env, "_m23_state", None)
    if cached is not None:
        return cached
    f = env.forge
    base_kw = dict(baseline_type="checkpoint",
                   baseline_ckpt=env.imp_early.checkpoint_id)
    f.register_policy(PolicyCreateRequest(
        policy_id="m23-pol-a", description="m23 policy a",
        policy=env.policy(seed=6301, name="m23-pol-a", **base_kw)))
    f.register_policy(PolicyCreateRequest(
        policy_id="m23-pol-b", description="m23 policy b",
        policy=env.policy(seed=6302, name="m23-pol-b", **base_kw)))
    d1 = f.run_gate(GateRequest(
        model_id=env.model_id, policy_id="m23-pol-a",
        candidate=env.state("checkpoint", env.imp_final.checkpoint_id)))
    d2 = f.run_gate(GateRequest(
        model_id=env.model_id, policy_id="m23-pol-a",
        candidate=env.state("checkpoint", env.reg_final.checkpoint_id)))
    d3 = f.run_gate(GateRequest(
        model_id=env.model_id, policy_id="m23-pol-b",
        candidate=env.state("checkpoint", env.imp_final.checkpoint_id)))
    d_inline = f.run_gate(env.gate(
        env.policy(baseline_type="checkpoint",
                   baseline_ckpt=env.imp_early.checkpoint_id, seed=6303),
        cand_ckpt=env.imp_final.checkpoint_id))          # policy_id is None
    # a second model with its own registered policy + one decision
    m_b = f.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m23-b", vocab_size=640, context_length=64, hidden_size=64,
        n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128,
        seed=63)))[0].id
    f.register_policy(PolicyCreateRequest(
        policy_id="m23-pol-bb", description="m23 policy of model b",
        policy=env.policy(baseline_type="current", seed=6304,
                          name="m23-pol-bb", model_id=m_b)))
    d_b = f.run_gate(GateRequest(model_id=m_b, policy_id="m23-pol-bb",
                                 candidate=env.state("current")))
    env._m23_state = (d1, d2, d3, d_inline, m_b, d_b)
    return env._m23_state


def _m23_gate_files(env, model_ids) -> set[str]:
    out = set()
    for mid in model_ids:
        root = env.gates_root(mid)
        if root == []:
            continue
        for p in root.rglob("*"):
            if p.is_file():
                out.add(f"{mid}:{p.relative_to(root).as_posix()}")
    return out


def test_m23_engine_filters_by_persisted_policy_and_model(env):
    d1, d2, d3, d_inline, m_b, d_b = _m23_env(env)
    f = env.forge
    got = f.list_gate_decisions_for_policy(env.model_id, "m23-pol-a")
    # every record belongs to the requested model AND policy; ordering is
    # the exact M6 convention ((created_at, decision_id) ascending)
    assert [d.decision_id for d in got] == [d1.decision_id, d2.decision_id]
    assert all(d.model_id == env.model_id and d.policy_id == "m23-pol-a"
               for d in got)
    assert [(d.created_at, d.decision_id) for d in got] == \
        sorted((d.created_at, d.decision_id) for d in got)
    # a second policy of the same model returns only its own decision
    got_b = f.list_gate_decisions_for_policy(env.model_id, "m23-pol-b")
    assert [d.decision_id for d in got_b] == [d3.decision_id]
    # payload parity: verbatim GateDecision equality with both the
    # authoritative M6 listing and the M6 single-record getter
    listing = {d.decision_id: d for d in f.list_gate_decisions(env.model_id)}
    for d in got:
        assert d == listing[d.decision_id]
        assert d == f.get_gate_decision(env.model_id, d.decision_id)
    # inline-policy decisions keep policy_id None and never appear
    assert d_inline.policy_id is None
    all_ids = {d.decision_id for d in got} | {d.decision_id for d in got_b}
    assert d_inline.decision_id not in all_ids


def test_m23_engine_valid_policy_without_runs_404s_and_read_only(env):
    d1, d2, d3, d_inline, m_b, d_b = _m23_env(env)
    f = env.forge
    # valid registered policy, but model b never gated under pol-a -> []
    assert f.list_gate_decisions_for_policy(m_b, "m23-pol-a") == []
    # model b's own policy returns exactly its own decision
    got_bb = f.list_gate_decisions_for_policy(m_b, "m23-pol-bb")
    assert [d.decision_id for d in got_bb] == [d_b.decision_id]
    # cross-model: model b never sees model a's pol-a decisions
    assert {d.decision_id for d in got_bb}.isdisjoint(
        {d.decision_id for d in (d1, d2)})
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_gate_decisions_for_policy("ghost-model-23", "m23-pol-a")
    # unknown policy -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_gate_decisions_for_policy(env.model_id, "m23-ghost-policy")
    # read-only: the filter never writes decision manifests
    models = (env.model_id, m_b)
    before = _m23_gate_files(env, models)
    f.list_gate_decisions_for_policy(env.model_id, "m23-pol-a")
    f.list_gate_decisions_for_policy(m_b, "m23-pol-a")
    assert _m23_gate_files(env, models) == before


def test_m23_engine_repeated_calls_identical(env):
    _m23_env(env)
    f = env.forge
    first = [d.model_dump(mode="json")
             for d in f.list_gate_decisions_for_policy(env.model_id,
                                                       "m23-pol-a")]
    for _ in range(3):
        again = [d.model_dump(mode="json")
                 for d in f.list_gate_decisions_for_policy(env.model_id,
                                                           "m23-pol-a")]
        assert again == first
