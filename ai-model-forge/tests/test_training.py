"""Milestone 3 tests: training engine, checkpoints, evaluation gate, rollback."""
from __future__ import annotations

import math
import os
import random

import pytest
import torch
from pydantic import ValidationError

from app.engine import ModelForge
from app.model_builder import content_hash
from app.schemas import (
    ModelCreateRequest,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
)
from app.training import estimate_training_bytes, fit_batch_size, lr_value

# --------------------------------------------------------------------------- #
# Shared scaffold (module scope): tokenizer + datasets + model factory
# --------------------------------------------------------------------------- #

_WORDS = ("river mountain cloud forest desert ocean valley island meadow canyon "
          "table chair lamp desk shelf couch rug clock mirror vase").split()
_TAIL_A = ("flows stands gleams rises falls drifts looms shines hides waits").split()
_TAIL_B = ("sings dances jumps laughs weeps shouts whispers roars murmurs chants").split()


def _word_soup(n: int, start: int = 0) -> list[str]:
    rng = random.Random(start)
    out = []
    for i in range(n):
        k = rng.randint(10, 24)
        out.append(" ".join(rng.choices(_WORDS, k=k)) + f" number {start + i}")
    return out


def _conflict_corpus(a: int, b: int) -> list[str]:
    rng = random.Random(11)
    out = []
    for i in range(a):
        h1, h2 = rng.sample(_WORDS, 2)
        out.append(f"{h1} is {rng.choice(_TAIL_A)} near {h2} with number {i}")
    for i in range(b):
        h1, h2 = rng.sample(_WORDS, 2)
        out.append(f"{h1} is {rng.choice(_TAIL_B)} near {h2} with number {a + i}")
    return out


def _instruction_like(n: int) -> list[str]:
    rng = random.Random(5)
    out = []
    for i in range(n):
        w1, w2 = rng.sample(_WORDS, 2)
        out.append(f"Question: is the {w1} {w2}?\nAnswer: {rng.choice(_TAIL_A)}.")
    return out


def _sentences_bytes(sentences: list[str]) -> bytes:
    return ("\n\n".join(sentences) + "\n").encode("utf-8")


class Env:
    """Everything needed to run small training jobs in one temp root."""

    def __init__(self, root):
        self.forge = ModelForge(root=root)
        self.tokenizer = None
        self.datasets: dict[str, str] = {}

    def prepare(self) -> None:
        base = _word_soup(280) + _instruction_like(200) + _conflict_corpus(220, 180)
        corpus = _sentences_bytes(base)
        up = self.forge.upload_dataset([("base.txt", corpus)], name="t3-base")
        self.datasets["base"] = up["dataset_id"]
        upc = self.forge.upload_dataset([("conflict.txt", _sentences_bytes(_conflict_corpus(220, 180)))],
                                        name="t3-conflict")
        self.datasets["conflict"] = upc["dataset_id"]
        upi = self.forge.upload_dataset([("instr.txt", _sentences_bytes(_instruction_like(260)))],
                                        name="t3-instr")
        self.datasets["instr"] = upi["dataset_id"]

        self.tokenizer = self.forge.train_tokenizer(
            TokenizerConfig(name="t3-tok", vocab_size=600), dataset_id=up["dataset_id"])
        for ds_id in self.datasets.values():
            self.forge.tokenize_dataset(ds_id, self.tokenizer.id)
        self.vocab = self.tokenizer.actual_vocab_size

    def new_model(self, name: str, seed: int | None = None) -> str:
        cfg = TransformerConfig(
            name=name, vocab_size=640, context_length=64, hidden_size=64,
            n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128, seed=seed)
        return self.forge.create_model(ModelCreateRequest(config=cfg))[0].id

    def state_hash(self, model_id: str) -> str:
        return content_hash(self.forge.storage.load_weights(model_id))

    def weights_bytes(self, model_id: str) -> bytes:
        return self.forge.storage.weights_path(model_id).read_bytes()

    def cfg(self, model_id: str, ds_key: str = "base", **overrides) -> TrainingConfig:
        base = dict(method="continued_pretraining", model_id=model_id,
                    dataset_id=self.datasets[ds_key], tokenizer_id=self.tokenizer.id,
                    learning_rate=3e-3, batch_size=8, max_seq_len=32,
                    eval_every_steps=5, seed=1, steps=10)
        base.update(overrides)
        if "epochs" in overrides:
            base.pop("steps", None)  # epochs XOR steps
        return TrainingConfig(**base)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("t3-root"))
    e.prepare()
    return e


# =========================================================================== #
# Test 1 — basic training
# =========================================================================== #

def test_basic_training_overfits_and_updates_lineage(env):
    model_id = env.new_model("t3-basic")
    before = env.state_hash(model_id)

    report = env.forge.run_training(env.cfg(model_id, epochs=8))
    assert report.optimizer_steps > 0
    assert report.final_train_loss < report.initial_train_loss  # designed to overfit
    assert math.isfinite(report.final_train_loss)
    assert report.baseline_validation_loss is not None
    assert report.best_validation_loss is not None
    ckpt_ids = [c["checkpoint_id"] for c in report.checkpoints]
    assert len(ckpt_ids) == len(set(ckpt_ids))
    assert report.checkpoints[-1]["step"] == report.optimizer_steps

    after = env.state_hash(model_id)
    assert after != before  # weights changed
    record = env.forge.get_model(model_id)
    assert record.latest_checkpoint == report.checkpoints[-1]["checkpoint_id"]
    assert record.best_checkpoint is not None
    assert len(record.training_provenance) == 1
    prov = record.training_provenance[0]
    assert prov.dataset_version == report.dataset_version == 1
    assert prov.optimizer_steps == report.optimizer_steps


def test_validation_metrics_finite_and_consistent(env):
    model_id = env.new_model("t3-val")
    report = env.forge.run_training(env.cfg(model_id, steps=10))
    for c in report.checkpoints:
        assert math.isfinite(c["validation_loss"])
        assert math.isfinite(c["perplexity"])
        assert c["perplexity"] == pytest.approx(math.exp(c["validation_loss"]), rel=1e-3)
    if report.best_validation_loss is not None:
        assert report.best_perplexity == pytest.approx(
            math.exp(report.best_validation_loss), rel=1e-3)


# =========================================================================== #
# Test 3 — determinism
# =========================================================================== #

def test_determinism_identical_runs_identical_history(env):
    # identical explicit model seeds -> identical starting weights
    m1, m2 = env.new_model("t3-det-a", seed=42), env.new_model("t3-det-b", seed=42)
    assert env.state_hash(m1) == env.state_hash(m2)
    r1 = env.forge.run_training(env.cfg(m1, steps=18, eval_every_steps=6, seed=99))
    r2 = env.forge.run_training(env.cfg(m2, steps=18, eval_every_steps=6, seed=99))
    assert len(r1.train_loss_history) == len(r2.train_loss_history) == 18
    for a, b in zip(r1.train_loss_history, r2.train_loss_history):
        assert abs(a - b) < 1e-5
    assert r1.checkpoints[-1]["weights_sha256"] == r2.checkpoints[-1]["weights_sha256"]
    assert env.state_hash(m1) == env.state_hash(m2)


# =========================================================================== #
# Test 4 — SFT (still causal LM loss, documented)
# =========================================================================== #

def test_sft_runs_on_instruction_like_text(env):
    model_id = env.new_model("t3-sft")
    report = env.forge.run_training(
        env.cfg(model_id, ds_key="instr", method="sft", epochs=4))
    assert report.method.value == "sft"
    assert report.checkpoints
    assert report.checkpoints[0]["method"] == "sft"
    assert math.isfinite(report.final_train_loss)
    record = env.forge.get_model(model_id)
    assert record.training_provenance[-1].method.value == "sft"


# =========================================================================== #
# Test 5 — configuration validation (schema level)
# =========================================================================== #

def test_epochs_and_steps_both_rejected(env):
    with pytest.raises(ValidationError, match="exactly one"):
        TrainingConfig(model_id="x", dataset_id="y", tokenizer_id="z", max_seq_len=8,
                       epochs=2, steps=2)
    with pytest.raises(ValidationError, match="exactly one"):
        TrainingConfig(model_id="x", dataset_id="y", tokenizer_id="z", max_seq_len=8)


def test_config_value_ranges_rejected():
    def kw(**overrides):
        d = dict(model_id="m", dataset_id="d", tokenizer_id="t", max_seq_len=8, steps=4)
        d.update(overrides)
        return d

    with pytest.raises(ValidationError):
        TrainingConfig(**kw(batch_size=0))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(learning_rate=0.0))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(learning_rate=5.0))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(weight_decay=-0.1))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(weight_decay=1.5))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(gradient_accumulation_steps=0))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(lr_schedule="exponential"))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(max_seq_len=0))
    with pytest.raises(ValidationError):
        TrainingConfig(**kw(warmup_steps=-1))
    TrainingConfig(**kw(lr_schedule="cosine"))  # valid enum accepted


def test_lr_value_schedule_shapes():
    total, warmup, base = 100, 10, 1.0
    assert lr_value(0, total, warmup, base, "linear") == base / warmup  # ramp start
    assert lr_value(warmup - 1, total, warmup, base, "constant") == pytest.approx(base)
    assert lr_value(warmup, total, warmup, base, "constant") == base
    assert lr_value(total - 1, total, warmup, base, "constant") == base
    assert lr_value(total - 1, total, warmup, base, "linear") == pytest.approx(0.0)
    assert lr_value(warmup, total, warmup, base, "cosine") == pytest.approx(base)
    assert lr_value(total - 1, total, warmup, base, "cosine") == pytest.approx(0.0, abs=1e-9)
    mid = lr_value((total + warmup) // 2, total, warmup, base, "cosine")
    assert 0 < mid < base  # monotone decay region
    # monotone non-decreasing warmup
    ramp = [lr_value(s, 50, 20, base, "linear") for s in range(20)]
    assert all(a <= b for a, b in zip(ramp, ramp[1:]))


def test_memory_estimator_reduces_batch(env):
    est_big = estimate_training_bytes(1_000_000, batch=64, seq_len=512, hidden=512, n_layers=12)
    est_small = estimate_training_bytes(1_000_000, batch=1, seq_len=512, hidden=512, n_layers=12)
    assert est_big > est_small
    # tiny budget forces a reduction but never below 1
    batch, reduced = fit_batch_size(64, 1_000_000, 512, 512, 12, budget_bytes=2**30)
    assert reduced and 1 <= batch < 64
    batch2, reduced2 = fit_batch_size(4, 100_000, 32, 64, 2, budget_bytes=2**31)
    assert batch2 == 4 and not reduced2


# =========================================================================== #
# Preflight refusals (weights untouched)
# =========================================================================== #

def test_missing_model_dataset_version_rejected(env):
    model_id = env.new_model("t3-missing")
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_training(env.cfg(model_id).model_copy(update={"model_id": "ghost"}))
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_training(env.cfg(model_id).model_copy(update={"dataset_id": "ghost"}))
    with pytest.raises(FileNotFoundError, match="version 99"):
        env.forge.run_training(env.cfg(model_id).model_copy(update={"dataset_version": 99}))
    with pytest.raises(FileNotFoundError):
        env.forge.run_training(env.cfg(model_id).model_copy(update={"tokenizer_id": "ghost"}))


def test_seq_len_beyond_context_rejected(env):
    model_id = env.new_model("t3-ctx")
    cfg = env.cfg(model_id, steps=4).model_copy(update={"max_seq_len": 128})  # ctx = 64
    with pytest.raises(ValueError, match="context_length"):
        env.forge.run_training(cfg)
    assert env.forge.get_model(model_id).training_provenance == []


def test_missing_tokenization_rejected_weights_unchanged(env):
    """A dataset without a tokenized artifact must refuse cleanly."""
    up = env.forge.upload_dataset([("raw.txt", _sentences_bytes(_word_soup(120, 9000)))],
                                  name="t3-untok")
    model_id = env.new_model("t3-untok-model")
    before_bytes = env.weights_bytes(model_id)
    cfg = env.cfg(model_id).model_copy(update={"dataset_id": up["dataset_id"]})
    with pytest.raises(FileNotFoundError, match="tokenized artifact"):
        env.forge.run_training(cfg)
    assert env.weights_bytes(model_id) == before_bytes  # nothing touched
    assert env.forge.get_model(model_id).training_provenance == []


def test_tampered_dataset_refused_before_training(env):
    import gzip
    import json

    up = env.forge.upload_dataset([("t.txt", _sentences_bytes(_word_soup(160, 7000)))],
                                  name="t3-tamper")
    model_id = env.new_model("t3-tamper-model")
    before_bytes = env.weights_bytes(model_id)

    rec_path = env.forge.datasets._version_dir(up["dataset_id"], 1) / "records.jsonl.gz"
    with gzip.open(rec_path, "rt", encoding="utf-8") as fh:
        lines = fh.readlines()
    rows = [json.loads(x) for x in lines]
    rows[0]["text"] += " EVIL"
    with gzip.open(rec_path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    assert env.forge.verify_dataset(up["dataset_id"])["status"] == "failed"
    with pytest.raises(ValueError, match="integrity"):
        env.forge.run_training(env.cfg(model_id).model_copy(update={"dataset_id": up["dataset_id"]}))
    assert env.weights_bytes(model_id) == before_bytes
    assert env.forge.get_model(model_id).training_provenance == []


# =========================================================================== #
# Tests 8-10 — checkpoint integrity, rollback, keep-best
# =========================================================================== #

def test_checkpoint_verification_and_corruption_refused(env):
    model_id = env.new_model("t3-ckpt-int")
    env.forge.run_training(env.cfg(model_id, steps=12, eval_every_steps=6))
    ckpts = env.forge.list_checkpoints(model_id)
    assert len(ckpts) == 2
    first = ckpts[0]
    assert first.step == 6 and ckpts[1].step == 12
    # lineage: second checkpoint's parent is the first
    assert ckpts[1].parent_checkpoint_id == first.checkpoint_id

    # intact checkpoint verifies
    state = env.forge.training.verify_checkpoint(model_id, first.checkpoint_id)
    assert content_hash(state) == first.weights_sha256

    # corrupt the weights (valid archive, changed values) -> verification and
    # rollback refuse: content hash no longer matches the manifest
    wpath = env.forge.training._ckpt_dir(model_id, first.checkpoint_id) / "weights.pt"
    tampered = {k: v.clone() for k, v in state.items()}
    first_key = next(iter(tampered))
    tampered[first_key].fill_(0.0)
    torch.save(tampered, wpath)
    with pytest.raises(RuntimeError, match="integrity"):
        env.forge.training.verify_checkpoint(model_id, first.checkpoint_id)
    with pytest.raises(RuntimeError, match="integrity"):
        env.forge.training.rollback(model_id, first.checkpoint_id)


def test_rollback_restores_selected_checkpoint(env):
    model_id = env.new_model("t3-rollback")
    env.forge.run_training(env.cfg(model_id, steps=14, eval_every_steps=7))
    ckpts = env.forge.list_checkpoints(model_id)
    target = ckpts[0]  # earlier checkpoint (step 7)

    record = env.forge.rollback_model(model_id, target.checkpoint_id)
    assert env.state_hash(model_id) == target.weights_sha256
    assert record.latest_checkpoint == target.checkpoint_id
    # rollback does not destroy history
    assert [c.checkpoint_id for c in env.forge.list_checkpoints(model_id)] == \
           [c.checkpoint_id for c in ckpts]
    # checkpoint files untouched (immutable)
    assert env.forge.get_checkpoint(model_id, target.checkpoint_id).weights_sha256 == \
        target.weights_sha256


def test_keep_best_rolls_back_when_final_is_worse(env):
    """Controlled degradation: conflicting domains make val loss rise after an
    early best; keep_best must restore the best checkpoint at the end."""
    model_id = env.new_model("t3-keepbest")
    cfg = env.cfg(model_id, ds_key="conflict", epochs=50, eval_every_steps=25,
                  learning_rate=6e-3, weight_decay=0.0, adam_beta2=0.99,
                  warmup_steps=3, seed=1)
    report = env.forge.run_training(cfg)

    assert report.rolled_back_to is not None
    assert report.accepted
    best = env.forge.get_checkpoint(model_id, report.rolled_back_to)
    assert report.rolled_back_to == env.forge.get_model(model_id).best_checkpoint
    # final weights are exactly the best checkpoint's weights
    assert env.state_hash(model_id) == best.weights_sha256
    # sanity: the run did degrade (final checkpoint worse than the best one)
    finals = [c for c in report.checkpoints if c["step"] == report.optimizer_steps]
    assert finals and finals[0]["decision"] == "not_best"
    assert finals[0]["validation_loss"] > best.validation_loss + 0.05


def test_keep_best_false_adopts_final_weights(env):
    model_id = env.new_model("t3-nokeep")
    cfg = env.cfg(model_id, steps=12, eval_every_steps=6, keep_best=False)
    report = env.forge.run_training(cfg)
    assert report.rolled_back_to is None
    final_ckpt = env.forge.get_checkpoint(model_id, report.checkpoints[-1]["checkpoint_id"])
    assert env.state_hash(model_id) == final_ckpt.weights_sha256  # final weights adopted
    record = env.forge.get_model(model_id)
    assert record.best_checkpoint is None  # no 'weights reference' claim when not kept


def test_second_run_baseline_is_previous_state(env):
    model_id = env.new_model("t3-baseline2")
    env.forge.run_training(env.cfg(model_id, steps=10))
    r2 = env.forge.run_training(env.cfg(model_id, steps=10, seed=1))
    # the second run's baseline is the model state left by the first run
    assert r2.baseline_validation_loss is not None and math.isfinite(r2.baseline_validation_loss)
    record = env.forge.get_model(model_id)
    assert len(record.training_provenance) == 2
    assert record.training_provenance[1].initial_checkpoint_id == \
        record.training_provenance[0].final_checkpoint_id


# =========================================================================== #
# Test 11 — storage deduplication (identical hash -> one physical artifact)
# =========================================================================== #

def test_identical_checkpoint_states_stored_once(env):
    """Content addressing: identical state -> one physical artifact.

    Exercised through the engine's own save path: saving the same state twice
    (different ids) must hard-link the second weights file to the first.
    """
    model_id = env.new_model("t3-dedup")
    state = env.forge.storage.load_weights(model_id)
    engine = env.forge.training
    cfg = env.cfg(model_id, steps=4)

    def save_once(parent=None, step=1):
        return engine._save_checkpoint(
            _FakeModule(state), model_id=model_id, run_id="unit",
            parent_checkpoint_id=parent, step=step, epoch=float(step) / 2,
            cfg=cfg, version=1, train_loss=1.0, val_loss=1.1,
            perplexity=3.0, lr=1e-3, decision="not_best")

    c1 = save_once()
    before = _dir_bytes(env.forge.storage.model_dir(model_id) / "checkpoints")
    weights_size = (engine._ckpt_dir(model_id, c1) / "weights.pt").stat().st_size
    c2 = save_once(parent=c1, step=2)

    p1 = engine._ckpt_dir(model_id, c1) / "weights.pt"
    p2 = engine._ckpt_dir(model_id, c2) / "weights.pt"
    assert os.path.samefile(p1, p2)            # one physical file, two entries
    recs = env.forge.list_checkpoints(model_id)
    assert recs[0].weights_sha256 == recs[1].weights_sha256

    after = _dir_bytes(env.forge.storage.model_dir(model_id) / "checkpoints")
    assert after - before < weights_size * 0.5  # second copy added ~0 weight bytes


class _FakeModule:
    """Minimal stand-in exposing state_dict() for the checkpoint saver."""

    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def _dir_bytes(path) -> int:
    """Real disk footprint: hard-linked files share one inode -> counted once."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            st = f.stat()
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    return total


# =========================================================================== #
# Gradient accumulation equivalence + schedule record
# =========================================================================== #

def test_gradient_accumulation_matches_larger_batch(env):
    m1, m2 = env.new_model("t3-accum-a", seed=7), env.new_model("t3-accum-b", seed=7)
    r_a = env.forge.run_training(env.cfg(m1, steps=16, batch_size=4, gradient_accumulation_steps=1,
                                         seed=7, eval_every_steps=16))
    r_b = env.forge.run_training(env.cfg(m2, steps=16, batch_size=1, gradient_accumulation_steps=4,
                                         seed=7, eval_every_steps=16))
    # same token windows per optimizer step -> identical training dynamics
    # (fp32 mean-reduction order differs slightly between the two groupings)
    assert r_a.train_loss_history == pytest.approx(r_b.train_loss_history, abs=1e-5)
    wa, wb = env.forge.storage.load_weights(m1), env.forge.storage.load_weights(m2)
    assert set(wa) == set(wb)
    for key in wa:
        assert torch.allclose(wa[key], wb[key], atol=1e-5, rtol=1e-4), key


def test_warmup_lr_trace_in_checkpoints(env):
    model_id = env.new_model("t3-warmup")
    cfg = env.cfg(model_id, steps=20, eval_every_steps=5, warmup_steps=10,
                  learning_rate=1e-2, seed=2)
    report = env.forge.run_training(cfg)
    lrs = [c["learning_rate"] for c in report.checkpoints]
    assert lrs[0] < lrs[1] <= 1e-2          # warmup ramps
    assert lrs[0] > 0


# =========================================================================== #
# M46: checkpoint history by training run (read-only grouping)
# =========================================================================== #

def test_m46_checkpoints_by_run_parity_partition_order(env):
    model_id = env.new_model("t3-m46-runs")
    run_a = env.forge.run_training(
        env.cfg(model_id, steps=10, eval_every_steps=5)).run_id
    run_b = env.forge.run_training(
        env.cfg(model_id, steps=5, eval_every_steps=5)).run_id
    assert run_a != run_b
    # both runs are registered in the model's own provenance registry
    prov = {p.run_id for p in
            env.forge.get_model(model_id).training_provenance}
    assert {run_a, run_b} <= prov
    all_ckpts = env.forge.list_checkpoints(model_id)
    # authoritative M3 ordering is (step, created_at) ascending
    assert [(c.step, c.created_at) for c in all_ckpts] == \
        sorted((c.step, c.created_at) for c in all_ckpts)
    a = env.forge.list_checkpoints_for_run(model_id, run_a)
    b = env.forge.list_checkpoints_for_run(model_id, run_b)
    # exact listing parity: each group is the listing filtered VERBATIM
    # by the checkpoints' own persisted run_id
    assert a == [c for c in all_ckpts if c.run_id == run_a]
    assert b == [c for c in all_ckpts if c.run_id == run_b]
    assert len(a) == 2 and len(b) == 1
    # TRUE disjoint partition (run_id is REQUIRED — no None case)
    assert {c.checkpoint_id for c in a}.isdisjoint(
        {c.checkpoint_id for c in b})
    assert ({c.checkpoint_id for c in a} | {c.checkpoint_id for c in b}) \
        == {c.checkpoint_id for c in all_ckpts}
    # 2 + 1 == every checkpoint, each exactly once; per-group order is
    # the exact authoritative (step, created_at) order preserved
    for group, run_id in ((a, run_a), (b, run_b)):
        assert [(c.step, c.created_at) for c in group] == \
            sorted((c.step, c.created_at) for c in group)
        assert all(c.run_id == run_id and c.model_id == model_id
                   for c in group)


def test_m46_zero_checkpoint_run_reached_via_listing_resilience(env):
    # Engine reality: steps >= 1 and the FINAL step is always an eval
    # point (training.py builds eval_points as
    # range(eval_every, total+1, eval_every) | {total_steps}), so a
    # healthy registered run ALWAYS produces >= 1 checkpoint. The
    # zero-checkpoint case therefore surfaces only through the M3
    # listing's corruption resilience: unreadable checkpoint manifests
    # are SKIPPED by design, so a registered run whose only manifest
    # is unreadable yields [] — never a 404, never fabricated ids.
    model_id = env.new_model("t3-m46-empty")
    report = env.forge.run_training(
        env.cfg(model_id, steps=2, eval_every_steps=5))
    # the run IS registered in the model's own training_provenance ...
    prov_ids = {p.run_id for p in
                env.forge.get_model(model_id).training_provenance}
    assert report.run_id in prov_ids
    # ... and a healthy run always owns at least one checkpoint
    ckpts = env.forge.list_checkpoints(model_id)
    assert len(ckpts) == 1 and ckpts[0].run_id == report.run_id
    # corrupt the checkpoint manifest (the M3 listing skips it) — the
    # SAME tampering convention as the M3 corruption tests above
    mpath = (env.forge.training._ckpt_dir(model_id, ckpts[0].checkpoint_id)
             / "manifest.json")
    mpath.write_text("{ not json")
    assert env.forge.list_checkpoints(model_id) == []
    # registered run + zero readable checkpoints -> [] (valid empty)
    assert env.forge.list_checkpoints_for_run(
        model_id, report.run_id) == []


def test_m46_unknown_run_model_and_cross_model_isolation(env):
    model_a = env.new_model("t3-m46-iso-a")
    model_b = env.new_model("t3-m46-iso-b")
    run_a = env.forge.run_training(env.cfg(model_a, steps=5)).run_id
    run_b = env.forge.run_training(env.cfg(model_b, steps=5)).run_id
    assert len(env.forge.list_checkpoints_for_run(model_a, run_a)) == 1
    assert len(env.forge.list_checkpoints_for_run(model_b, run_b)) == 1
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        env.forge.list_checkpoints_for_run("ghost-model-46", run_a)
    # unknown run on a valid model -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        env.forge.list_checkpoints_for_run(model_a, "no-such-run-46")
    # a run id belonging to another model -> FileNotFoundError
    # (model-scoped ownership via each model's own provenance)
    with pytest.raises(FileNotFoundError):
        env.forge.list_checkpoints_for_run(model_a, run_b)
    with pytest.raises(FileNotFoundError):
        env.forge.list_checkpoints_for_run(model_b, run_a)


# --------------------------------------------------------------------- M54
# Explicit training resume point: initialize ONE run from an immutable
# checkpoint WITHOUT publishing it first (no rollback, no latest_checkpoint
# mutation to prepare the run). Model-weight resume only.

def test_m54_resume_nondestructive_provenance_and_immutability(env):
    from pathlib import Path

    mid = env.new_model("m54-resume")
    assert env.cfg(mid).resume_from_checkpoint_id is None   # default = old behavior
    env.forge.run_training(env.cfg(mid, steps=10, eval_every_steps=5, seed=1))
    ck = env.forge.list_checkpoints(mid)
    assert len(ck) == 2
    B = ck[0]                                   # a HISTORICAL checkpoint
    latest_before = env.forge.get_model(mid).latest_checkpoint
    assert latest_before != B.checkpoint_id      # latest != resume point
    bdir = (Path(env.forge.storage.model_dir(mid)) / "checkpoints"
            / B.checkpoint_id)
    b_man = (bdir / "manifest.json").read_bytes()
    b_weights = (bdir / "weights.pt").read_bytes()

    rep = env.forge.run_training(env.cfg(
        mid, steps=10, eval_every_steps=5, seed=7,
        resume_from_checkpoint_id=B.checkpoint_id))

    # provenance: the run records its CONCRETE starting checkpoint
    prov = [p for p in env.forge.get_model(mid).training_provenance
            if p.run_id == rep.run_id][0]
    assert prov.initial_checkpoint_id == B.checkpoint_id
    assert prov.parent_checkpoint_id == B.checkpoint_id
    assert prov.config["resume_from_checkpoint_id"] == B.checkpoint_id
    assert rep.initial_model_version == B.checkpoint_id
    # the run's first checkpoint descends from B (existing lineage fields)
    new_ck = sorted((c for c in env.forge.list_checkpoints(mid)
                     if c.run_id == rep.run_id), key=lambda c: c.step)
    assert new_ck[0].parent_checkpoint_id == B.checkpoint_id
    assert new_ck[1].parent_checkpoint_id == new_ck[0].checkpoint_id

    # §7 baseline: the run started from B's state — its baseline evaluation
    # reproduces B's persisted validation_loss (same data/probe config)
    assert round(rep.baseline_validation_loss, 6) == B.validation_loss

    # NON-destructive: latest follows the EXISTING publication semantics
    # (the run's last created checkpoint), never B-unless-created
    m = env.forge.get_model(mid)
    assert m.latest_checkpoint == new_ck[-1].checkpoint_id
    assert m.latest_checkpoint != B.checkpoint_id

    # §14 immutability: B is referenced, never copied or modified
    assert (bdir / "manifest.json").read_bytes() == b_man
    assert (bdir / "weights.pt").read_bytes() == b_weights
    assert len(env.forge.list_checkpoints(mid)) == 4   # 2 + 2, no duplicates


def test_m54_resume_determinism_and_error_taxonomy(env):
    from pathlib import Path

    mid = env.new_model("m54-det")
    env.forge.run_training(env.cfg(mid, steps=10, eval_every_steps=5, seed=1))
    B = env.forge.list_checkpoints(mid)[0]
    kw = dict(steps=10, eval_every_steps=5, seed=7,
              resume_from_checkpoint_id=B.checkpoint_id)
    r2 = env.forge.run_training(env.cfg(mid, **kw))
    r3 = env.forge.run_training(env.cfg(mid, **kw))
    sig = lambda rid: sorted(          # noqa: E731  (step, loss) per checkpoint
        (c.step, c.validation_loss)
        for c in env.forge.list_checkpoints(mid) if c.run_id == rid)
    assert sig(r2.run_id) == sig(r3.run_id)
    assert r2.baseline_validation_loss == r3.baseline_validation_loss

    # unknown checkpoint -> the established 404 taxonomy (engine level)
    with pytest.raises(FileNotFoundError):
        env.forge.run_training(env.cfg(
            mid, steps=4, resume_from_checkpoint_id="no-such-ck-m54"))
    # a checkpoint of ANOTHER model is unknown HERE (model-scoped lookup)
    other = env.new_model("m54-foreign")
    env.forge.run_training(env.cfg(other, steps=10, eval_every_steps=5, seed=1))
    foreign_ck = env.forge.list_checkpoints(other)[0].checkpoint_id
    with pytest.raises(FileNotFoundError):
        env.forge.run_training(env.cfg(
            mid, steps=4, resume_from_checkpoint_id=foreign_ck))
    # corrupted weights -> integrity RuntimeError, nothing written
    wpath = (Path(env.forge.storage.model_dir(mid)) / "checkpoints"
             / B.checkpoint_id / "weights.pt")
    orig = wpath.read_bytes()
    wpath.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="unreadable|integrity"):
        env.forge.run_training(env.cfg(mid, **kw))
    wpath.write_bytes(orig)                    # restore exactly
    # B is intact and resumable again
    r4 = env.forge.run_training(env.cfg(mid, **kw))
    assert r4.baseline_validation_loss == r2.baseline_validation_loss


def test_m54_resume_through_workflow_train_stage(env):
    from app.schemas import (StageType, WorkflowPlan, WorkflowStage)

    mid = env.new_model("m54-wf")
    env.forge.run_training(env.cfg(mid, steps=10, eval_every_steps=5, seed=1))
    B = env.forge.list_checkpoints(mid)[0]
    train_cfg = env.cfg(mid, steps=10, eval_every_steps=5, seed=3,
                        resume_from_checkpoint_id=B.checkpoint_id)
    plan = WorkflowPlan(name="m54-resume-wf", model_id=mid, stages=[
        WorkflowStage(stage_id="tr", type=StageType.TRAIN,
                      training=train_cfg)])
    rec = env.forge.run_workflow(plan)
    assert rec.status.value == "completed"
    prov = [p for p in env.forge.get_model(mid).training_provenance
            if p.run_id == rec.stages[0].artifact.artifact_id][0]
    assert prov.initial_checkpoint_id == B.checkpoint_id
    assert prov.config["resume_from_checkpoint_id"] == B.checkpoint_id
