"""Milestone 4 tests: read-only evaluation engine (current weights + checkpoints).

Covers the 14 required scenarios: M3-consistency of current-state and
checkpoint evaluation (1e-4), determinism (incl. result_hash), corrupt-state
refusal before the loop, invalid state/dataset/version/tokenizer references,
read-only invariants, the train<validation overfit probe, full-split token
accounting under the M3 window/tail policy, capped evaluation semantics and
current-vs-checkpoint comparability.
"""
from __future__ import annotations

import gzip
import json
import math
import random

import pytest
import torch
from pydantic import ValidationError

from app.model_builder import content_hash
from app.schemas import (
    EvalStateKind,
    EvaluationConfig,
    EvaluationSplit,
    ModelCreateRequest,
    TokenizerConfig,
    TrainingConfig,
    TransformerConfig,
)

_WORDS = ("river mountain cloud forest desert ocean valley island meadow canyon "
          "table chair lamp desk shelf couch rug clock mirror vase").split()


def _word_soup(n: int, start: int = 0) -> list[str]:
    rng = random.Random(start)
    out = []
    for i in range(n):
        k = rng.randint(10, 22)
        out.append(" ".join(rng.choices(_WORDS, k=k)) + f" number {start + i}")
    return out


def _bytes(sentences: list[str]) -> bytes:
    return ("\n\n".join(sentences) + "\n").encode("utf-8")


class Env:
    """Module-scope scaffold: two tokenized datasets + tokenizer + models."""

    def __init__(self, root):
        self.forge = None
        self.root = root
        self.tokenizer = None
        self.datasets: dict[str, str] = {}

    def prepare(self) -> None:
        from app.engine import ModelForge

        self.forge = ModelForge(root=self.root)
        up = self.forge.upload_dataset(
            [("base.txt", _bytes(_word_soup(300)))], name="m4-base")
        self.datasets["base"] = up["dataset_id"]
        upp = self.forge.upload_dataset(
            [("probe.txt", _bytes(_word_soup(220, 9000)))], name="m4-probe")
        self.datasets["probe"] = upp["dataset_id"]

        self.tokenizer = self.forge.train_tokenizer(
            TokenizerConfig(name="m4-tok", vocab_size=600),
            dataset_id=self.datasets["base"])
        for ds_id in self.datasets.values():
            self.forge.tokenize_dataset(ds_id, self.tokenizer.id)

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    def new_model(self, name: str, seed: int = 3, vocab_size: int = 640) -> str:
        cfg = TransformerConfig(
            name=name, vocab_size=vocab_size, context_length=64, hidden_size=64,
            n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128, seed=seed)
        return self.forge.create_model(ModelCreateRequest(config=cfg))[0].id

    def train_cfg(self, model_id: str, ds_key: str = "base", **overrides) -> TrainingConfig:
        base = dict(method="continued_pretraining", model_id=model_id,
                    dataset_id=self.datasets[ds_key],
                    tokenizer_id=self.tokenizer.id, learning_rate=3e-3,
                    batch_size=8, max_seq_len=32, eval_every_steps=25,
                    keep_best=False, seed=1, steps=40)
        base.update(overrides)
        if "epochs" in overrides:
            base.pop("steps", None)
        return TrainingConfig(**base)

    def eval_cfg(self, model_id: str, ds_key: str = "base", **overrides) -> EvaluationConfig:
        base = dict(model_id=model_id, dataset_id=self.datasets[ds_key],
                    split=EvaluationSplit.VALIDATION,
                    tokenizer_id=self.tokenizer.id, batch_size=8,
                    max_seq_len=32, seed=11)
        base.update(overrides)
        return EvaluationConfig(**base)

    def run_training(self, model_id: str, ds_key: str = "base", **overrides):
        return self.forge.run_training(self.train_cfg(model_id, ds_key, **overrides))

    def state_hash(self, model_id: str) -> str:
        return content_hash(self.forge.storage.load_weights(model_id))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("m4-root"))
    e.prepare()
    return e


# --------------------------------------------------------------------------- #
# Shared trained fixtures (module scope: each train run executes once)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def trained(env):
    """Model trained with keep_best=False: current weights == final checkpoint."""
    model_id = env.new_model("m4-main")
    report = env.run_training(model_id, ds_key="base", epochs=10)
    ckpts = env.forge.list_checkpoints(model_id)
    assert len(ckpts) >= 2
    return {"model_id": model_id, "report": report, "checkpoints": ckpts}


@pytest.fixture(scope="module")
def probe(env):
    """Word-soup model trained long enough to overfit its training split."""
    model_id = env.new_model("m4-probe-model")
    env.run_training(model_id, ds_key="probe", epochs=30)
    return model_id


# =========================================================================== #
# Test 1 — current weights evaluation == M3 final validation loss
# =========================================================================== #

def test_current_weights_eval_matches_final_validation_loss(env, trained):
    mid = trained["model_id"]
    final_ckpt = trained["checkpoints"][-1]
    rec = env.forge.run_evaluation(env.eval_cfg(mid))
    assert rec.state_kind.value == "current"
    assert rec.checkpoint_id is None
    assert rec.split == EvaluationSplit.VALIDATION
    assert rec.state_hash == env.state_hash(mid)
    assert abs(rec.loss_nats - final_ckpt.validation_loss) < 1e-4
    assert rec.perplexity == pytest.approx(math.exp(rec.loss_nats), rel=1e-3)
    assert rec.token_count > 0
    assert rec.truncated is False
    assert len(rec.result_hash) == 64
    # record echoes the resolved window (max_seq_len=None -> model context here
    # we passed 32 explicitly, so it must echo 32)
    assert rec.config["max_seq_len"] == 32


# =========================================================================== #
# Test 2 — checkpoint evaluation == its manifest validation loss
# =========================================================================== #

def test_checkpoint_eval_matches_manifest_validation_loss(env, trained):
    mid = trained["model_id"]
    ckpt = trained["checkpoints"][0]
    rec = env.forge.run_evaluation(env.eval_cfg(mid, checkpoint_id=ckpt.checkpoint_id))
    assert rec.state_kind.value == "checkpoint"
    assert rec.checkpoint_id == ckpt.checkpoint_id
    assert rec.state_hash == ckpt.weights_sha256
    assert abs(rec.loss_nats - ckpt.validation_loss) < 1e-4


# =========================================================================== #
# Test 3 — deterministic repeated evaluation (metrics + result_hash)
# =========================================================================== #

def test_deterministic_repeated_evaluation(env, trained):
    mid = trained["model_id"]
    a = env.forge.run_evaluation(env.eval_cfg(mid))
    b = env.forge.run_evaluation(env.eval_cfg(mid))
    assert a.eval_id != b.eval_id          # separate immutable records
    for field in ("loss_nats", "perplexity", "token_count",
                  "records_covered", "truncated", "state_hash"):
        assert getattr(a, field) == getattr(b, field), field
    assert a.result_hash == b.result_hash
    # default seed (None) is config-derived and equally deterministic
    c = env.forge.run_evaluation(env.eval_cfg(mid, seed=None))
    d = env.forge.run_evaluation(env.eval_cfg(mid, seed=None))
    assert c.result_hash == d.result_hash
    assert a.result_hash != c.result_hash   # different seed -> different identity


# =========================================================================== #
# Test 4 — corrupt checkpoint refused BEFORE the evaluation loop
# =========================================================================== #

def test_corrupt_checkpoint_refused(env):
    mid = env.new_model("m4-corrupt")
    env.run_training(mid, ds_key="base", steps=8, eval_every_steps=8)
    ckpts = env.forge.list_checkpoints(mid)
    assert len(ckpts) == 1
    ckpt = ckpts[0]
    before_weights = env.forge.storage.weights_path(mid).read_bytes()

    # corrupt the weights (valid archive, changed values) -> hash mismatch
    wpath = env.forge.training._ckpt_dir(mid, ckpt.checkpoint_id) / "weights.pt"
    tampered = {k: v.clone() for k, v in
                torch.load(wpath, map_location="cpu", weights_only=True).items()}
    next(iter(tampered.values())).fill_(0.0)
    torch.save(tampered, wpath)

    with pytest.raises(RuntimeError, match="integrity"):
        env.forge.run_evaluation(env.eval_cfg(mid, checkpoint_id=ckpt.checkpoint_id))
    # refused before any artifact was created + nothing else was touched
    assert env.forge.list_evaluations(mid) == []
    assert env.forge.storage.weights_path(mid).read_bytes() == before_weights
    # corrupted checkpoint also refuses via the shared verifier for current
    # state? no: current weights of this model were never replaced; verify the
    # model still evaluates fine on its own current weights
    rec = env.forge.run_evaluation(env.eval_cfg(mid))
    assert rec.state_kind.value == "current"


# =========================================================================== #
# Test 5 — invalid state configuration rejected
# =========================================================================== #

def test_invalid_state_references_rejected(env, trained):
    mid = trained["model_id"]
    other = env.new_model("m4-other")
    with pytest.raises(ValidationError):
        EvaluationConfig(**{**env.eval_cfg(mid).model_dump(mode="json"),
                            "model_id": ""})            # empty model id
    with pytest.raises(ValidationError):
        EvaluationConfig(**{**env.eval_cfg(mid).model_dump(mode="json"),
                            "checkpoint_id": ""})       # empty checkpoint id
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_evaluation(env.eval_cfg("ghost"))
    with pytest.raises(FileNotFoundError):
        env.forge.run_evaluation(env.eval_cfg(mid, checkpoint_id="ghost"))
    # a checkpoint is scoped to its model -> other model's id is "not found"
    other_ckpt = env.forge.training._ckpt_root(other)
    if other_ckpt.exists():
        ids = [d.name for d in other_ckpt.iterdir() if d.is_dir()]
        if ids:
            with pytest.raises(FileNotFoundError, match="not found"):
                env.forge.run_evaluation(
                    env.eval_cfg(mid, checkpoint_id=ids[0]))


# =========================================================================== #
# Test 6 — invalid split rejected
# =========================================================================== #

def test_invalid_split_rejected(env, trained):
    mid = trained["model_id"]
    for bad in ("trainx", "val", "validation2", "TRAIN", ""):
        with pytest.raises(ValidationError):
            env.eval_cfg(mid, split=bad)


# =========================================================================== #
# Test 7 — invalid dataset/version rejected
# =========================================================================== #

def test_invalid_dataset_version_rejected(env, trained):
    mid = trained["model_id"]
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_evaluation(env.eval_cfg(mid, dataset_id="ghost"))
    with pytest.raises(FileNotFoundError, match="version 99"):
        env.forge.run_evaluation(env.eval_cfg(mid, dataset_version=99))

    # dataset uploaded but never tokenized -> no tokenized artifact
    up = env.forge.upload_dataset([("raw.txt", _bytes(_word_soup(40, 7000)))],
                                  name="m4-untok")
    with pytest.raises(FileNotFoundError, match="tokenized artifact"):
        env.forge.run_evaluation(env.eval_cfg(mid, dataset_id=up["dataset_id"]))

    # corrupted dataset -> refused (integrity), nothing evaluated
    up2 = env.forge.upload_dataset([("t.txt", _bytes(_word_soup(60, 8000)))],
                                   name="m4-tamper")
    rec_path = env.forge.datasets._version_dir(up2["dataset_id"], 1) / "records.jsonl.gz"
    with gzip.open(rec_path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh]
    rows[0]["text"] += " EVIL"
    with gzip.open(rec_path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert env.forge.verify_dataset(up2["dataset_id"])["status"] == "failed"
    with pytest.raises(ValueError, match="integrity"):
        env.forge.run_evaluation(env.eval_cfg(mid, dataset_id=up2["dataset_id"]))
    assert len(env.forge.list_evaluations(mid)) >= 1  # prior evals unaffected


# =========================================================================== #
# Test 8 — invalid tokenizer / vocabulary mismatch rejected
# =========================================================================== #

def test_invalid_tokenizer_and_vocab_rejected(env, trained):
    mid = trained["model_id"]
    with pytest.raises(FileNotFoundError, match="not found"):
        env.forge.run_evaluation(env.eval_cfg(mid, tokenizer_id="ghost"))
    # model whose vocab cannot hold the token ids
    tiny = env.new_model("m4-vocab16", vocab_size=16)
    with pytest.raises(ValueError, match="vocab_size"):
        env.forge.run_evaluation(env.eval_cfg(tiny))


# =========================================================================== #
# Test 9 — read-only evaluation
# =========================================================================== #

def _tree_bytes(path) -> dict[str, bytes]:
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(path))] = p.read_bytes()
    return out


def test_evaluation_is_read_only(env, trained):
    mid = trained["model_id"]
    model_dir = env.forge.storage.model_dir(mid)
    before_manifest = (model_dir / "manifest.json").read_bytes()
    before_weights = env.forge.storage.weights_path(mid).read_bytes()
    ckpt_root = env.forge.training._ckpt_root(mid)
    before_ckpts = _tree_bytes(ckpt_root)
    before_prov = [p.model_dump(mode="json")
                   for p in env.forge.get_model(mid).training_provenance]
    before_evals = env.forge.list_evaluations(mid)

    env.forge.run_evaluation(env.eval_cfg(mid))
    ck = env.forge.list_checkpoints(mid)[0]
    env.forge.run_evaluation(env.eval_cfg(mid, checkpoint_id=ck.checkpoint_id))

    assert (model_dir / "manifest.json").read_bytes() == before_manifest
    assert env.forge.storage.weights_path(mid).read_bytes() == before_weights
    assert env.state_hash(mid) == trained["checkpoints"][-1].weights_sha256
    assert _tree_bytes(ckpt_root) == before_ckpts
    assert [p.model_dump(mode="json")
            for p in env.forge.get_model(mid).training_provenance] == before_prov
    after_evals = env.forge.list_evaluations(mid)
    assert [r.eval_id for r in after_evals[:len(before_evals)]] == \
        [r.eval_id for r in before_evals]     # existing records untouched
    assert len(after_evals) == len(before_evals) + 2


# =========================================================================== #
# Test 10 — train < validation overfit probe
# =========================================================================== #

def test_train_vs_validation_overfit_probe(env, probe):
    train_rec = env.forge.run_evaluation(
        env.eval_cfg(probe, ds_key="probe", split=EvaluationSplit.TRAIN))
    val_rec = env.forge.run_evaluation(
        env.eval_cfg(probe, ds_key="probe", split=EvaluationSplit.VALIDATION))
    assert train_rec.split == EvaluationSplit.TRAIN
    assert val_rec.split == EvaluationSplit.VALIDATION
    # deliberately overfit fixture: the model memorised its training records
    assert train_rec.loss_nats < val_rec.loss_nats - 0.05


# =========================================================================== #
# Test 11 — full split: token count == M3 usable-target accounting
# =========================================================================== #

def test_full_split_token_count_matches_m3_policy(env, trained):
    mid = trained["model_id"]
    ds_id = env.datasets["base"]
    info, _ = env.forge.datasets.tokenized_artifact(ds_id, 1, env.tokenizer.id)
    total = info.splits["validation"].count
    rec = env.forge.run_evaluation(env.eval_cfg(mid))          # window 32
    expected = (total // 32) * (32 - 1)                        # drop tail policy
    assert rec.token_count == expected
    assert rec.truncated is False
    vm = env.forge.datasets.version_manifest(ds_id, 1)
    assert rec.records_covered == vm.splits["validation"].count  # exact records

    # default window (max_seq_len=None -> model context_length 64) also
    # follows the same accounting, and the record echoes the resolved window
    rec64 = env.forge.run_evaluation(env.eval_cfg(mid, max_seq_len=None))
    assert rec64.token_count == (total // 64) * 63
    assert rec64.config["max_seq_len"] == 64
    assert rec64.records_covered == vm.splits["validation"].count


# =========================================================================== #
# Test 12 — capped evaluation (deterministic, truncated, records not guessed)
# =========================================================================== #

def test_capped_evaluation(env, trained):
    # use the (large) train split so the cap really truncates mid-window:
    # 100 targets = 3 full windows of 31 + 7 targets of a partial window
    mid = trained["model_id"]
    cap = 100
    a = env.forge.run_evaluation(env.eval_cfg(
        mid, split=EvaluationSplit.TRAIN, max_eval_tokens=cap, seed=5))
    assert a.token_count == cap          # exact deterministic cap (windows +
                                         # partial final window)
    assert a.truncated is True
    assert a.records_covered is None     # never inferred from the flat stream
    b = env.forge.run_evaluation(env.eval_cfg(
        mid, split=EvaluationSplit.TRAIN, max_eval_tokens=cap, seed=5))
    assert b.loss_nats == a.loss_nats and b.result_hash == a.result_hash

    # a cap beyond the whole split behaves exactly like a full evaluation
    full = env.forge.run_evaluation(env.eval_cfg(mid, split=EvaluationSplit.TRAIN))
    huge = env.forge.run_evaluation(env.eval_cfg(
        mid, split=EvaluationSplit.TRAIN, max_eval_tokens=10**9, seed=5))
    assert huge.truncated is False
    assert huge.token_count == full.token_count
    assert huge.loss_nats == full.loss_nats
    assert huge.records_covered == full.records_covered


# =========================================================================== #
# Test 14 — current vs checkpoint comparison: same schema, correct hashes
# =========================================================================== #

def test_current_and_checkpoint_states_comparable(env, trained):
    mid = trained["model_id"]
    cur = env.forge.run_evaluation(env.eval_cfg(mid))
    ckpt = trained["checkpoints"][0]
    ckp = env.forge.run_evaluation(
        env.eval_cfg(mid, checkpoint_id=ckpt.checkpoint_id))
    for rec in (cur, ckp):
        assert rec.model_id == mid
        assert rec.dataset_id == env.datasets["base"]
        assert rec.dataset_version == 1
        assert rec.split == EvaluationSplit.VALIDATION
        assert rec.tokenizer_id == env.tokenizer.id
        assert math.isfinite(rec.loss_nats) and rec.perplexity > 0
    assert cur.state_hash == env.state_hash(mid)          # current canonical
    assert ckp.state_hash == ckpt.weights_sha256          # checkpoint canonical
    assert cur.state_kind.value != ckp.state_kind.value


# =========================================================================== #
# M24: read-only per-checkpoint grouping of the evaluation history
# =========================================================================== #

def _m24_env(env):
    """One model with three checkpoints (evals on two of them + one
    current-state eval) and a second model with its own checkpoint.
    Built once per module env and cached — every M24 engine test sees the
    exact same immutable history. Returns (a, b, cks_a, ck_evals, cur,
    b_ckpt, b_eval).
    """
    cached = getattr(env, "_m24_state", None)
    if cached is not None:
        return cached
    f = env.forge
    a = env.new_model("m24-main")
    env.run_training(a, ds_key="base", steps=30, eval_every_steps=10)
    cks = f.list_checkpoints(a)
    assert len(cks) >= 3
    ck_a, ck_b, ck_c = (c.checkpoint_id for c in cks[:3])
    e1 = f.run_evaluation(env.eval_cfg(a, checkpoint_id=ck_a, seed=2411))
    e2 = f.run_evaluation(env.eval_cfg(a, checkpoint_id=ck_a, seed=2412))
    e3 = f.run_evaluation(env.eval_cfg(a, checkpoint_id=ck_b, seed=2413))
    cur = f.run_evaluation(env.eval_cfg(a, seed=2414))   # current state
    b = env.new_model("m24-b", seed=24)
    env.run_training(b, ds_key="base", steps=10, eval_every_steps=10)
    b_ckpt = f.list_checkpoints(b)[0].checkpoint_id
    b_eval = f.run_evaluation(env.eval_cfg(b, checkpoint_id=b_ckpt,
                                           seed=2415))
    env._m24_state = (a, b, (ck_a, ck_b, ck_c),
                      (ck_a, [e1, e2]), ck_b, [e3], cur, b_ckpt, b_eval)
    return env._m24_state


def _m24_eval_files(env, model_ids) -> set[str]:
    out = set()
    for mid in model_ids:
        root = env.forge.storage.model_dir(mid) / "evaluations"
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                out.add(f"{mid}:{p.relative_to(root).as_posix()}")
    return out


def test_m24_engine_filters_by_persisted_checkpoint_and_model(env):
    a, b, cks, (ck_a, ck_evals), ck_b, ck_evals_b, cur, b_ckpt, b_eval = \
        _m24_env(env)
    f = env.forge
    got = f.list_evaluations_for_checkpoint(a, ck_a)
    # every record belongs to the requested model AND checkpoint; ordering
    # is the exact M4 convention ((created_at, eval_id) ascending)
    assert [r.eval_id for r in got] == [r.eval_id for r in ck_evals]
    assert all(r.model_id == a and r.state_kind.value == "checkpoint"
               and r.checkpoint_id == ck_a for r in got)
    assert [(r.created_at, r.eval_id) for r in got] == \
        sorted((r.created_at, r.eval_id) for r in got)
    # a second checkpoint returns only its own evaluation
    got_b = f.list_evaluations_for_checkpoint(a, ck_b)
    assert [r.eval_id for r in got_b] == [r.eval_id for r in ck_evals_b]
    # payload parity: verbatim EvaluationRecord equality with both the
    # authoritative M4 listing and the M4 single-record getter
    listing = {r.eval_id: r for r in f.list_evaluations(a)}
    for r in got:
        assert r == listing[r.eval_id]
        assert r == f.get_evaluation(a, r.eval_id)
    # the current-state evaluation (checkpoint_id None) never appears
    assert cur.state_kind.value == "current" and cur.checkpoint_id is None
    assert cur.eval_id not in {r.eval_id for r in got}
    assert cur.eval_id not in {r.eval_id for r in got_b}


def test_m24_engine_empty_404s_cross_model_and_read_only(env):
    a, b, cks, (ck_a, ck_evals), ck_b, _, cur, b_ckpt, b_eval = \
        _m24_env(env)
    f = env.forge
    ck_c = cks[2]
    # valid registered checkpoint with no evaluations -> []
    assert f.list_evaluations_for_checkpoint(a, ck_c) == []
    # the other model's checkpoint history stays separate (exactly its
    # own evaluation, never model a's)
    got_b = f.list_evaluations_for_checkpoint(b, b_ckpt)
    assert [r.eval_id for r in got_b] == [b_eval.eval_id]
    assert {r.eval_id for r in got_b}.isdisjoint(
        {r.eval_id for r in ck_evals})
    # cross-model: model a cannot query model b's checkpoint id (M3
    # registry is model-scoped) and vice versa
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_checkpoint(a, b_ckpt)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_checkpoint(b, ck_a)
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_checkpoint("ghost-model-24", ck_a)
    # unknown checkpoint -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_checkpoint(a, "ghost-ck-24")
    # read-only: the filter never writes evaluation manifests
    before = _m24_eval_files(env, (a, b))
    f.list_evaluations_for_checkpoint(a, ck_a)
    f.list_evaluations_for_checkpoint(a, ck_c)
    assert _m24_eval_files(env, (a, b)) == before


def test_m24_engine_repeated_calls_identical(env):
    a, _, cks, (ck_a, _), _, _, _, _, _ = _m24_env(env)
    f = env.forge
    first = [r.model_dump(mode="json")
             for r in f.list_evaluations_for_checkpoint(a, ck_a)]
    for _ in range(3):
        again = [r.model_dump(mode="json")
                 for r in f.list_evaluations_for_checkpoint(a, ck_a)]
        assert again == first


# =========================================================================== #
# M28: read-only per-dataset grouping of the evaluation history
# =========================================================================== #

def _m28_env(env):
    """M28 state on top of the shared module env (cached): a model with
    evaluations over TWO datasets (two on 'base', one on 'probe'), a
    second model with NO evaluations (model-scoped empty), and a third
    freshly uploaded dataset with zero evaluations.
    Returns (a, b, e_b1, e_b2, e_p1, third_ds).
    """
    cached = getattr(env, "_m28_state", None)
    if cached is not None:
        return cached
    f = env.forge
    a = env.new_model("m28-main", seed=28)
    e_b1 = f.run_evaluation(env.eval_cfg(a, ds_key="base", seed=2811))
    e_b2 = f.run_evaluation(env.eval_cfg(a, ds_key="base", seed=2812))
    e_p1 = f.run_evaluation(env.eval_cfg(a, ds_key="probe", seed=2813))
    b = env.new_model("m28-b", seed=29)          # NO evaluations at all
    third = f.upload_dataset(
        [("third.txt", _bytes(_word_soup(200, 28000)))], name="m28-third")
    f.tokenize_dataset(third["dataset_id"], env.tokenizer.id)
    env._m28_state = (a, b, e_b1, e_b2, e_p1, third["dataset_id"])
    return env._m28_state


def test_m28_engine_filters_by_persisted_dataset_identity(env):
    a, b, e_b1, e_b2, e_p1, third_ds = _m28_env(env)
    f = env.forge
    base_id, probe_id = env.datasets["base"], env.datasets["probe"]
    listing = f.list_evaluations(a)
    for ds_id, expected in ((base_id, [e_b1, e_b2]),
                            (probe_id, [e_p1]),
                            (third_ds, [])):
        got = f.list_evaluations_for_dataset(a, ds_id)
        # parity with the authoritative M4 listing filtered by the
        # persisted dataset identity; deterministic (created_at,
        # eval_id) order; membership from the persisted field only
        assert got == [r for r in listing if r.dataset_id == ds_id]
        assert [r.eval_id for r in got] == [r.eval_id for r in expected]
        keyed = [(r.created_at, r.eval_id) for r in got]
        assert keyed == sorted(keyed)
        assert all(r.dataset_id == ds_id and r.model_id == a for r in got)
        # persisted dataset_version travels VERBATIM (never rewritten)
        # and every payload equals the M4 single-record getter verbatim
        for r, orig in zip(got, expected):
            assert r.dataset_version == orig.dataset_version
            assert r == f.get_evaluation(a, r.eval_id)
    # no cross-dataset leakage: each evaluation appears under exactly
    # its own dataset and under no other
    for ds_id in (base_id, probe_id):
        other = probe_id if ds_id == base_id else base_id
        under = [r.eval_id for r in
                 f.list_evaluations_for_dataset(a, ds_id)]
        under_other = [r.eval_id for r in
                       f.list_evaluations_for_dataset(a, other)]
        assert not (set(under) & set(under_other))


def test_m28_engine_empty_404s_model_scoping_read_only(env):
    a, b, e_b1, e_b2, e_p1, third_ds = _m28_env(env)
    f = env.forge
    base_id = env.datasets["base"]
    # valid dataset with zero evaluations for THIS model -> [] (the
    # model-scoped empty case: the dataset is global, b has no evals)
    assert f.list_evaluations_for_dataset(b, base_id) == []
    assert f.list_evaluations_for_dataset(b, third_ds) == []
    # unknown model / unknown dataset -> FileNotFoundError (404 at API)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_dataset("ghost-model-28", base_id)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_dataset(a, "ghost-ds-28")
    # model scoping: a's evaluation ids never appear under b
    leak = [r.eval_id for r in
            f.list_evaluations_for_dataset(b, base_id)]
    assert {e_b1.eval_id, e_b2.eval_id}.isdisjoint(leak)
    # read-only: the filter never writes evaluation manifests
    before = _m24_eval_files(env, (a, b))
    f.list_evaluations_for_dataset(a, base_id)
    f.list_evaluations_for_dataset(b, base_id)
    f.list_evaluations_for_dataset(a, third_ds)
    after = _m24_eval_files(env, (a, b))
    assert after == before


def test_m28_engine_repeated_calls_identical(env):
    a, b, e_b1, e_b2, e_p1, third_ds = _m28_env(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_evaluations_for_dataset(a, env.datasets["base"])]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_evaluations_for_dataset(a, env.datasets["base"])]
        assert again == first


# =========================================================================== #
# M30: read-only per-tokenizer grouping of the evaluation history
# =========================================================================== #

def _m30_env(env):
    """M30 state on top of the shared module env (cached): a SECOND
    tokenizer (tok2, vocab 320) with which the base dataset is
    tokenized and ONE evaluation of the M28 model runs (partitioning),
    and a THIRD tokenizer (tok3) with zero evaluations.
    Returns (a, e_t1, e_t2, tok1, tok2, tok3).
    """
    cached = getattr(env, "_m30_state", None)
    if cached is not None:
        return cached
    f = env.forge
    a, b, e_b1, e_b2, e_p1, third_ds = _m28_env(env)
    tok1 = env.tokenizer.id
    tok2 = f.train_tokenizer(
        TokenizerConfig(name="m30-tok2", vocab_size=320),
        dataset_id=env.datasets["base"]).id
    f.tokenize_dataset(env.datasets["base"], tok2)
    e_t2 = f.run_evaluation(env.eval_cfg(a, tokenizer_id=tok2, seed=3011))
    tok3 = f.train_tokenizer(
        TokenizerConfig(name="m30-tok3", vocab_size=300),
        dataset_id=env.datasets["probe"]).id
    env._m30_state = (a, (e_b1, e_b2, e_p1), e_t2, tok1,
                      tok2, tok3)
    return env._m30_state


def test_m30_engine_filters_by_persisted_tokenizer_identity(env):
    a, t1_evals, e_t2, tok1, tok2, tok3 = _m30_env(env)
    f = env.forge
    listing = f.list_evaluations(a)
    for tok, expected in ((tok1, list(t1_evals)),
                        (tok2, [e_t2]), (tok3, [])):
        got = f.list_evaluations_for_tokenizer(a, tok)
        # parity with the authoritative M4 listing filtered by the
        # persisted tokenizer identity; deterministic (created_at,
        # eval_id) order; membership from the persisted field only
        assert got == [r for r in listing if r.tokenizer_id == tok]
        assert [r.eval_id for r in got] == [r.eval_id for r in expected]
        keyed = [(r.created_at, r.eval_id) for r in got]
        assert keyed == sorted(keyed)
        assert all(r.tokenizer_id == tok and r.model_id == a for r in got)
        for r in got:
            assert r == f.get_evaluation(a, r.eval_id)
    # persisted tokenizer identity VERBATIM on every record (tok2's id
    # stays exactly tok2 — no substitution, no rewriting)
    got2 = f.list_evaluations_for_tokenizer(a, tok2)
    assert all(r.tokenizer_id == tok2 for r in got2)
    # partition: every evaluation id appears under exactly ONE of the
    # two populated tokenizers; no tokenizer leaks into the other
    under1 = {r.eval_id for r in f.list_evaluations_for_tokenizer(a, tok1)}
    under2 = {r.eval_id for r in f.list_evaluations_for_tokenizer(a, tok2)}
    assert under1 and under2 and under1.isdisjoint(under2)
    assert under1 | under2 == {r.eval_id for r in listing}


def test_m30_engine_empty_404s_model_scoping_read_only(env):
    a, t1_evals, e_t2, tok1, tok2, tok3 = _m30_env(env)
    f = env.forge
    _, b, *_ = _m28_env(env)
    # valid tokenizer with zero evaluations for THIS model -> [] (the
    # model-scoped empty case: tokenizers are global, b has no evals)
    assert f.list_evaluations_for_tokenizer(b, tok1) == []
    assert f.list_evaluations_for_tokenizer(a, tok3) == []
    # unknown model / unknown tokenizer -> FileNotFoundError (404 at API)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_tokenizer("ghost-model-30", tok1)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_tokenizer(a, "ghost-tok-30")
    # model scoping: a's evaluation ids never appear under b
    leak = [r.eval_id for r in f.list_evaluations_for_tokenizer(b, tok1)]
    assert {e.eval_id for e in t1_evals}.isdisjoint(leak)
    assert e_t2.eval_id not in leak
    # read-only: the filter never writes evaluation manifests
    before = _m24_eval_files(env, (a, b))
    f.list_evaluations_for_tokenizer(a, tok1)
    f.list_evaluations_for_tokenizer(a, tok2)
    f.list_evaluations_for_tokenizer(b, tok1)
    f.list_evaluations_for_tokenizer(a, tok3)
    after = _m24_eval_files(env, (a, b))
    assert after == before


def test_m30_engine_repeated_calls_identical(env):
    a, t1_evals, e_t2, tok1, tok2, tok3 = _m30_env(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_evaluations_for_tokenizer(a, tok1)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_evaluations_for_tokenizer(a, tok1)]
        assert again == first


# =========================================================================== #
# M36: read-only per-split grouping of the evaluation history
# =========================================================================== #

def _m36_env(env):
    """M36 state on top of the shared module env (cached): a model with
    evaluations on ALL THREE splits (validation x2, train x1, test x1,
    all over 'base'), plus the M28 empty-history model b. Returns
    (a, b, e_v1, e_v2, e_t1, e_s1).
    """
    cached = getattr(env, "_m36_state", None)
    if cached is not None:
        return cached
    _, b, _, _, _, _ = _m28_env(env)      # b: NO evaluations at all
    f = env.forge
    a = env.new_model("m36-main", seed=36)
    e_v1 = f.run_evaluation(env.eval_cfg(a, ds_key="base", seed=3611))
    e_v2 = f.run_evaluation(env.eval_cfg(a, ds_key="base", seed=3612))
    e_t1 = f.run_evaluation(env.eval_cfg(a, ds_key="base",
                                         split=EvaluationSplit.TRAIN,
                                         seed=3613))
    e_s1 = f.run_evaluation(env.eval_cfg(a, ds_key="base",
                                         split=EvaluationSplit.TEST,
                                         seed=3614))
    env._m36_state = (a, b, e_v1, e_v2, e_t1, e_s1)
    return env._m36_state


def test_m36_engine_filters_by_persisted_split_identity(env):
    a, b, e_v1, e_v2, e_t1, e_s1 = _m36_env(env)
    f = env.forge
    listing = f.list_evaluations(a)
    for split in (EvaluationSplit.VALIDATION, EvaluationSplit.TRAIN,
                  EvaluationSplit.TEST):
        got = f.list_evaluations_for_split(a, split)
        # parity with the authoritative M4 listing filtered by the
        # persisted split; deterministic (created_at, eval_id) order;
        # membership from the persisted field only, never rewritten
        assert got == [r for r in listing if r.split == split]
        keyed = [(r.created_at, r.eval_id) for r in got]
        assert keyed == sorted(keyed)
        ids = [r.eval_id for r in got]
        assert len(ids) == len(set(ids))
        assert all(r.split == split and r.model_id == a for r in got)
        # verbatim payload parity with the M4 single-record getter
        for r in got:
            assert r == f.get_evaluation(a, r.eval_id)
    # the M36-created evaluations sit under exactly their splits
    val_ids = [r.eval_id for r in f.list_evaluations_for_split(
        a, EvaluationSplit.VALIDATION)]
    assert val_ids.count(e_v1.eval_id) == 1
    assert val_ids.count(e_v2.eval_id) == 1
    assert [r.eval_id for r in f.list_evaluations_for_split(
        a, EvaluationSplit.TRAIN)] .count(e_t1.eval_id) == 1
    assert [r.eval_id for r in f.list_evaluations_for_split(
        a, EvaluationSplit.TEST)].count(e_s1.eval_id) == 1
    # explicit partition: groups over EVERY split present in the
    # listing are pairwise disjoint and cover it exactly once
    groups = {sp: {r.eval_id for r in
                   f.list_evaluations_for_split(a, sp)}
              for sp in {r.split for r in listing}}
    flat = [i for g in groups.values() for i in g]
    assert set(flat) == {r.eval_id for r in listing}
    assert len(flat) == len(set(flat)) == len(listing)


def test_m36_engine_empty_404s_model_scoping_read_only(env):
    a, b, e_v1, e_v2, e_t1, e_s1 = _m36_env(env)
    f = env.forge
    # empty-history model -> [] for EVERY valid split (never 404)
    assert f.list_evaluations(b) == []
    for split in (EvaluationSplit.VALIDATION, EvaluationSplit.TRAIN,
                  EvaluationSplit.TEST):
        assert f.list_evaluations_for_split(b, split) == []
    # unknown model -> FileNotFoundError (404 at the API)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_split("ghost-model-36",
                                     EvaluationSplit.VALIDATION)
    # cross-model isolation: the groups of one model never contain
    # the other's evaluation ids (model scoping from the listing)
    a_ids = {r.eval_id for r in f.list_evaluations(a)}
    b_ids = {r.eval_id for r in f.list_evaluations(b)}
    assert a_ids.isdisjoint(b_ids)
    for split in (EvaluationSplit.VALIDATION, EvaluationSplit.TRAIN,
                  EvaluationSplit.TEST):
        got_b = {r.eval_id for r in
                 f.list_evaluations_for_split(b, split)}
        assert got_b <= b_ids and got_b.isdisjoint(a_ids)
    # read-only: the filter never writes evaluation manifests
    def eval_files(mid):
        root = f.storage.model_dir(mid) / "evaluations"
        if not root.exists():
            return set()
        return {p.relative_to(root).as_posix()
                for p in root.rglob("*") if p.is_file()}
    before = (eval_files(a), eval_files(b))
    f.list_evaluations_for_split(a, EvaluationSplit.VALIDATION)
    f.list_evaluations_for_split(a, EvaluationSplit.TEST)
    f.list_evaluations_for_split(b, EvaluationSplit.TRAIN)
    assert (eval_files(a), eval_files(b)) == before


def test_m36_engine_repeated_calls_identical(env):
    a, b, e_v1, e_v2, e_t1, e_s1 = _m36_env(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_evaluations_for_split(a,
                                          EvaluationSplit.VALIDATION)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_evaluations_for_split(
                     a, EvaluationSplit.VALIDATION)]
        assert again == first


# --------------------------------------------------------------------------- #
# M38: evaluation history by state kind (read-only grouping, enum contract)
# --------------------------------------------------------------------------- #

def _m38_env(env):
    """M38 state on top of the shared module env (cached): a TRAINED
    model with evaluations over BOTH state kinds (checkpoint x2 over
    its first stored checkpoint, current x1 over the published
    weights), plus the M28 empty-history model b. Returns
    (a, b, e_c1, e_c2, e_cu).
    """
    cached = getattr(env, "_m38_state", None)
    if cached is not None:
        return cached
    _, b, _, _, _, _ = _m28_env(env)      # b: NO evaluations at all
    f = env.forge
    a = env.new_model("m38-main", seed=38)
    env.run_training(a, ds_key="base", steps=30)
    ck = f.list_checkpoints(a)[0].checkpoint_id
    e_c1 = f.run_evaluation(env.eval_cfg(a, ds_key="base",
                                         checkpoint_id=ck, seed=3811))
    e_c2 = f.run_evaluation(env.eval_cfg(a, ds_key="base",
                                         checkpoint_id=ck, seed=3812))
    e_cu = f.run_evaluation(env.eval_cfg(a, ds_key="base", seed=3813))
    env._m38_state = (a, b, e_c1, e_c2, e_cu)
    return env._m38_state


def test_m38_engine_filters_by_persisted_state_kind_identity(env):
    a, b, e_c1, e_c2, e_cu = _m38_env(env)
    f = env.forge
    listing = f.list_evaluations(a)
    for kind in (EvalStateKind.CHECKPOINT, EvalStateKind.CURRENT):
        got = f.list_evaluations_for_state_kind(a, kind)
        # parity with the authoritative M4 listing filtered by the
        # persisted state_kind; deterministic (created_at, eval_id)
        # order; membership from the persisted field only
        assert got == [r for r in listing if r.state_kind == kind]
        keyed = [(r.created_at, r.eval_id) for r in got]
        assert keyed == sorted(keyed)
        ids = [r.eval_id for r in got]
        assert len(ids) == len(set(ids))
        assert all(r.state_kind == kind and r.model_id == a for r in got)
        # the nullability is a schema CONSEQUENCE of the persisted
        # kind, never the membership source
        assert all((r.checkpoint_id is None)
                   == (kind == EvalStateKind.CURRENT) for r in got)
        for r in got:
            assert r == f.get_evaluation(a, r.eval_id)
    # the fixture's records land in their own groups with the
    # persisted kind VERBATIM (earlier module tests may have added
    # other evaluations — membership derives from the listing)
    got_ck = f.list_evaluations_for_state_kind(a, EvalStateKind.CHECKPOINT)
    got_cu = f.list_evaluations_for_state_kind(a, EvalStateKind.CURRENT)
    assert {e_c1.eval_id, e_c2.eval_id} <= {r.eval_id for r in got_ck}
    assert e_cu.eval_id in {r.eval_id for r in got_cu}
    # explicit partition: disjoint groups over BOTH enum values whose
    # union is the full listing
    ids_ck = {r.eval_id for r in got_ck}
    ids_cu = {r.eval_id for r in got_cu}
    assert ids_ck and ids_cu
    assert ids_ck.isdisjoint(ids_cu)
    assert ids_ck | ids_cu == {r.eval_id for r in listing}


def test_m38_engine_empty_404s_model_scoping_read_only(env):
    a, b, e_c1, e_c2, e_cu = _m38_env(env)
    f = env.forge
    # valid state kind with zero evaluations -> [] (the model-scoped
    # empty case: b has no evaluations under EITHER kind)
    for kind in (EvalStateKind.CHECKPOINT, EvalStateKind.CURRENT):
        assert f.list_evaluations_for_state_kind(b, kind) == []
    # unknown model -> FileNotFoundError (404 at the API); the enum
    # itself needs NO registry lookup (unsupported values are 422 at
    # the API boundary and never reach the engine)
    with pytest.raises(FileNotFoundError):
        f.list_evaluations_for_state_kind("ghost-model-38",
                                          EvalStateKind.CURRENT)
    # cross-model isolation: a's eval ids never appear under b
    for kind in (EvalStateKind.CHECKPOINT, EvalStateKind.CURRENT):
        leak = [r.eval_id for r in
                f.list_evaluations_for_state_kind(b, kind)]
        assert e_c1.eval_id not in leak and e_cu.eval_id not in leak
    # read-only: the filter never writes evaluation manifests
    def eval_files(mid):
        root = f.storage.model_dir(mid) / "evaluations"
        if not root.exists():
            return set()
        return {p.relative_to(root).as_posix()
                for p in root.rglob("*") if p.is_file()}
    before = (eval_files(a), eval_files(b))
    for kind in (EvalStateKind.CHECKPOINT, EvalStateKind.CURRENT):
        f.list_evaluations_for_state_kind(a, kind)
        f.list_evaluations_for_state_kind(b, kind)
    assert (eval_files(a), eval_files(b)) == before


def test_m38_engine_repeated_calls_identical(env):
    a, b, e_c1, e_c2, e_cu = _m38_env(env)
    f = env.forge
    first = [r.model_dump(mode="json") for r in
             f.list_evaluations_for_state_kind(a,
                                               EvalStateKind.CHECKPOINT)]
    for _ in range(3):
        again = [r.model_dump(mode="json") for r in
                 f.list_evaluations_for_state_kind(a,
                                                   EvalStateKind.CHECKPOINT)]
        assert again == first
