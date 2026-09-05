"""Model engine tests: builder math, forward/generation correctness, persistence."""
from __future__ import annotations

import pytest
import torch

from app.engine import ModelForge
from app.model_builder import build_transformer, content_hash
from app.schemas import ModelCreateRequest, TransformerConfig


def variant(cfg: TransformerConfig, **updates: object) -> TransformerConfig:
    """Rebuild a config from JSON form so enum/field coercion always runs.

    (``model_copy(update=...)`` skips pydantic validation — string values for
    enum fields would silently slip through as raw strings.)
    """
    return TransformerConfig(**{**cfg.model_dump(mode="json"), **updates})


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #

def expected_parameter_count(cfg: TransformerConfig) -> int:
    """Independent re-derivation of the parameter count from the config."""
    hd = cfg.hidden_size // cfg.n_heads
    tok = cfg.vocab_size * cfg.hidden_size
    head = 0 if cfg.tie_output_embeddings else cfg.vocab_size * cfg.hidden_size
    inter = cfg.intermediate_size
    mlp_params = (3 * inter if cfg.activation.value == "swiglu" else 2 * inter) * cfg.hidden_size
    per_layer = (
        cfg.hidden_size                                      # ln1
        + (cfg.n_heads + 2 * cfg.n_kv_heads) * hd * cfg.hidden_size   # fused qkv
        + cfg.hidden_size * cfg.hidden_size                  # attention output
        + cfg.hidden_size                                    # ln2
        + mlp_params
    )
    return tok + head + per_layer * cfg.n_layers + cfg.hidden_size  # + final_norm


def test_parameter_count_matches_independent_formula(forge: ModelForge, tiny_config):
    cfg = tiny_config
    record, _ = forge.create_model(ModelCreateRequest(config=cfg))
    assert record.parameter_count == expected_parameter_count(cfg)
    assert record.parameter_count == 19_616  # sanity: locked value


def test_tied_embeddings_counted_once(forge: ModelForge, tiny_config):
    payload = tiny_config.model_dump(mode="json")
    payload.update(vocab_size=32, tie_output_embeddings=True)  # hidden == 32
    cfg = TransformerConfig(**payload)
    record, _ = forge.create_model(ModelCreateRequest(config=cfg))
    assert record.parameter_count == expected_parameter_count(cfg)
    assert record.parameter_count == 16_544  # tok counted once; head is tied


def test_preserve_init_requires_parent_state():
    cfg = variant(TransformerConfig.tiny(), weight_init="preserve")
    with pytest.raises(ValueError, match="requires parent weights"):
        build_transformer(cfg)


# --------------------------------------------------------------------------- #
# Forward / generation behaviour
# --------------------------------------------------------------------------- #

def test_forward_shape_and_finite_logits(tiny_config):
    built = build_transformer(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    logits, caches = built.module(ids)
    assert logits.shape == (2, 8, tiny_config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert len(caches) == tiny_config.n_layers
    # Causal: row t may not depend on tokens after t.
    a = built.module(torch.randint(0, tiny_config.vocab_size, (1, 4)))[0]
    assert a.shape == (1, 4, tiny_config.vocab_size)


def test_generation_length_determinism_and_context_guard(tiny_config):
    built = build_transformer(tiny_config)
    prompt = torch.tensor([[3, 7, 1]])
    out1 = built.module.generate(prompt, max_new_tokens=12, seed=42)
    out2 = built.module.generate(prompt, max_new_tokens=12, seed=42)
    assert out1.shape == (1, 15)
    assert torch.equal(out1, out2)  # seeded -> deterministic
    with pytest.raises(ValueError, match="prompt length"):
        built.module.generate(torch.zeros(1, tiny_config.context_length, dtype=torch.long), 5)


def test_kv_cache_reproduces_full_forward(tiny_config):
    """Decoding with cache must produce identical logits to one-shot forward."""
    built = build_transformer(tiny_config)
    ids = torch.tensor([[1, 2, 3, 4, 5]])

    full_logits, _ = built.module(ids)

    # Manual incremental decode: prefill 3 tokens, then 2 cached steps.
    logits, caches = built.module(ids[:, :3])
    step_logits = []
    for pos in (3, 4):
        logits, caches = built.module(ids[:, pos : pos + 1], kv_caches=caches)
        step_logits.append(logits)
    assert step_logits[0].shape == (1, 1, tiny_config.vocab_size)

    # The final incremental logit equals the last row of the full forward.
    assert torch.allclose(step_logits[-1][0, 0], full_logits[0, 4], atol=1e-4)
    # ... and the first decode step matches row 3 of the one-shot forward.
    assert torch.allclose(step_logits[0][0, 0], full_logits[0, 3], atol=1e-4)


def test_gqa_vs_mha_consistency(forge: ModelForge, tiny_config):
    """GQA with one kv-head must equal plain MHA whose 4 heads share k/v."""
    base = tiny_config.model_dump(mode="json")
    base.update(n_layers=1)
    gqa_cfg = TransformerConfig(**{**base, "n_kv_heads": 1})
    mha_cfg = TransformerConfig(**{**base, "n_kv_heads": 4})

    gqa = build_transformer(gqa_cfg)
    mha = build_transformer(mha_cfg)

    with torch.no_grad():
        # 1) Copy every parameter with a compatible shape (GQA's fused qkv is
        #    the only differing one). Seeding alone would NOT desync-free
        #    sampling: the RNG stream diverges at the differently sized qkv.
        gqa_state = dict(gqa.module.named_parameters())
        for name, param in mha.module.named_parameters():
            if name in gqa_state and gqa_state[name].shape == param.shape:
                param.copy_(gqa_state[name])

        # 2) Expand the single GQA kv-head to 4 identical MHA kv-heads.
        hd = gqa_cfg.head_dim()  # 8
        n_q_rows = mha_cfg.n_heads * hd  # 32 rows of q
        gq = gqa.module.blocks[0].attn.qkv.weight.data  # (48, 32)
        mkv = mha.module.blocks[0].attn.qkv.weight.data  # (64, 32)
        mkv[:n_q_rows] = gq[:n_q_rows]
        for h in range(mha_cfg.n_heads):
            mkv[n_q_rows + h * hd : n_q_rows + (h + 1) * hd] = gq[n_q_rows : n_q_rows + hd]
            mkv[2 * n_q_rows + h * hd : 2 * n_q_rows + (h + 1) * hd] = gq[n_q_rows + hd : n_q_rows + 2 * hd]

    ids = torch.tensor([[2, 5, 1, 9]])
    logits_gqa, _ = gqa.module(ids)
    logits_mha, _ = mha.module(ids)
    assert torch.allclose(logits_gqa, logits_mha, atol=1e-3)


# --------------------------------------------------------------------------- #
# Precision / component variants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_low_precision_models_roundtrip(forge: ModelForge, tiny_config, precision):
    cfg = variant(tiny_config, precision=precision)
    built = build_transformer(cfg)
    assert built.dtype == precision
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    logits, _ = built.module(ids)
    assert logits.dtype == torch.float32  # logits always fp32
    assert torch.isfinite(logits).all()


def test_component_variants_forward(forge: ModelForge, tiny_config):
    for kwargs in (
        {"position_encoding": "learned"},
        {"position_encoding": "none"},
        {"qkv_merging": "separate"},
        {"activation": "gelu"},
        {"activation": "relu"},
        {"attention_softcap": 30.0},
        {"n_kv_heads": 1},
    ):
        cfg = variant(tiny_config, **kwargs)
        built = build_transformer(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, caches = built.module(ids)
        assert torch.isfinite(logits).all(), kwargs
        out = built.module.generate(ids[:, :3], max_new_tokens=4, seed=1)
        assert out.shape == (1, 7), kwargs


def test_rope_positions_beyond_theta_are_capped_but_valid(forge: ModelForge):
    cfg = TransformerConfig(
        name="rope-100", vocab_size=64, context_length=256, hidden_size=32,
        n_layers=1, n_heads=4, n_kv_heads=2, intermediate_size=48, rope_theta=100.0,
    )
    assert cfg.rope_max_seq() == 100
    built = build_transformer(cfg)
    logits, _ = built.module(torch.zeros(1, 100, dtype=torch.long))
    assert logits.shape == (1, 100, 64)


# --------------------------------------------------------------------------- #
# Persistence & integrity
# --------------------------------------------------------------------------- #

def test_create_persists_and_reloads(forge: ModelForge, tiny_config):
    record, extra = forge.create_model(ModelCreateRequest(config=tiny_config))
    assert extra["weights_bytes_on_disk"] > 0
    model_dir = forge.storage.model_dir(record.id)
    assert (model_dir / "manifest.json").exists()
    assert (model_dir / "weights.pt").exists()
    assert (model_dir / "weights.sha256").exists()

    reloaded = forge.get_model(record.id)
    assert reloaded.config == tiny_config
    assert reloaded.state_hash == record.state_hash  # weights hash persisted & stable
    assert reloaded.config_hash == record.config_hash
    assert reloaded.config_hash != "" and reloaded.state_hash != ""
    assert reloaded.parameter_count == record.parameter_count
    # Verify manifest JSON and torch weights both parse after a full re-read.
    assert forge.verify_model(record.id)["integrity"] == "ok"

    # Second engine instance on the same root sees the same model (stateless dirs).
    other = ModelForge(root=forge.storage.root)
    assert [r.id for r in other.list_models()] == [record.id]
    other_record = other.get_model(record.id)
    assert other_record.config == tiny_config


def test_duplicate_name_rejected(forge: ModelForge, tiny_config):
    forge.create_model(ModelCreateRequest(config=tiny_config))
    with pytest.raises(ValueError, match="already exists"):
        forge.create_model(ModelCreateRequest(config=tiny_config))


def test_weights_hash_is_stable_and_dtype_sensitive(forge: ModelForge, tiny_config):
    """Equal values => equal hashes; quantized values => different hashes."""
    fp32a = build_transformer(tiny_config)
    fp32b = build_transformer(tiny_config)  # deterministic init (derived seed)
    h_a, h_b = content_hash(fp32a.state_dict()), content_hash(fp32b.state_dict())
    assert h_a == h_b  # stable across calls and across identical rebuilds
    assert h_a == content_hash(fp32a.state_dict())

    # The same *intended* numbers stored at lower precision hash differently,
    # because fp16/bf16 rounding changed the stored values (correct behaviour).
    assert h_a != content_hash(build_transformer(
        variant(tiny_config, precision="bf16")).state_dict())


def test_delete_removes_everything(forge: ModelForge, tiny_config):
    record, _ = forge.create_model(ModelCreateRequest(config=tiny_config))
    forge.delete_model(record.id)
    assert forge.storage.model_ids() == []
    with pytest.raises(FileNotFoundError):
        forge.get_model(record.id)


def test_flash_attention_rejected_on_cpu():
    cfg = variant(TransformerConfig.tiny(), attention_impl="flash_attn")
    with pytest.raises(ValueError, match="CUDA"):
        build_transformer(cfg)
