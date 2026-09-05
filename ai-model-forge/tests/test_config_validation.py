"""Schema-level tests: the configuration system and its validation rules."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import TransformerConfig, WeightInit


def base_payload() -> dict:
    return TransformerConfig.tiny().model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Valid configurations
# --------------------------------------------------------------------------- #


def test_tiny_config_is_valid():
    cfg = TransformerConfig.tiny()
    assert cfg.vocab_size == 64
    assert cfg.head_dim() == 8
    assert cfg.qkv_output_size() == (4 + 2 * 2) * 8


def test_recommended_minimum_is_valid():
    cfg = TransformerConfig.recommended_minimum()
    assert cfg.head_dim() == 32
    assert cfg.rope_max_seq() == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_encoding": "learned"},
        {"position_encoding": "none"},
        {"qkv_merging": "separate"},
        {"activation": "gelu"},
        {"activation": "relu"},
        {"precision": "bf16"},
        {"precision": "fp16"},
        {"attention_softcap": 50.0},
        {"n_kv_heads": 4},  # plain MHA (kv == heads)
        {"tie_output_embeddings": True, "vocab_size": 32},  # hidden_size == vocab
    ],
)
def test_variants_are_valid(kwargs):
    payload = base_payload()
    payload.update(kwargs)
    TransformerConfig(**payload)


def test_default_seed_is_deterministic_and_stable():
    cfg = TransformerConfig.tiny()
    assert cfg.effective_seed() == cfg.default_seed()
    again = TransformerConfig(**cfg.model_dump(mode="json"))
    assert again.effective_seed() == cfg.effective_seed()
    tweaked = cfg.model_copy(update={"n_layers": 3})
    assert tweaked.effective_seed() != cfg.effective_seed()


def test_explicit_seed_wins_over_derived():
    cfg = TransformerConfig.tiny()
    seeded = cfg.model_copy(update={"seed": 7})
    assert seeded.effective_seed() == 7


# --------------------------------------------------------------------------- #
# Field-level rejections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", 8),            # floor 16
        ("context_length", 2),        # floor 4
        ("hidden_size", 4),           # floor 8
        ("n_layers", 0),
        ("n_heads", 0),
        ("n_kv_heads", 0),
        ("name", "bad name!"),        # charset: '!' not allowed
        ("name", "bad*name"),         # charset: '*' not allowed
        ("architecture", "bert_encoder"),
        ("activation", "sigmoid"),
        ("precision", "f64"),
    ],
)
def test_field_level_rejections(field, value):
    payload = base_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        TransformerConfig(**payload)


def test_extra_fields_rejected():
    payload = base_payload()
    payload["surprise"] = True
    with pytest.raises(ValidationError):
        TransformerConfig(**payload)


# --------------------------------------------------------------------------- #
# Pairwise rules
# --------------------------------------------------------------------------- #


def test_hidden_not_divisible_by_heads_rejected():
    payload = base_payload()
    payload["hidden_size"] = 33  # 33 % 4 != 0
    with pytest.raises(ValidationError, match="divisible by n_heads"):
        TransformerConfig(**payload)


def test_gqa_uneven_groups_rejected():
    payload = base_payload()
    payload["n_heads"] = 4
    payload["n_kv_heads"] = 3  # 4 % 3 != 0
    with pytest.raises(ValidationError, match="multiple of n_kv_heads"):
        TransformerConfig(**payload)


def test_kv_heads_exceeding_heads_rejected():
    payload = base_payload()
    payload["n_heads"] = 2
    payload["n_kv_heads"] = 4
    with pytest.raises(ValidationError, match="cannot exceed"):
        TransformerConfig(**payload)


def test_swiglu_odd_intermediate_rejected():
    payload = base_payload()
    payload["intermediate_size"] = 49
    with pytest.raises(ValidationError, match="even for swiglu"):
        TransformerConfig(**payload)


def test_odd_intermediate_ok_for_relu():
    payload = base_payload()
    payload["activation"] = "relu"
    payload["intermediate_size"] = 49
    TransformerConfig(**payload)


def test_tied_embeddings_require_equal_dims():
    payload = base_payload()
    payload["tie_output_embeddings"] = True  # vocab 64 != hidden 32
    with pytest.raises(ValidationError, match="tie_output_embeddings"):
        TransformerConfig(**payload)


def test_flash_attention_bounds():
    payload = base_payload()
    payload["n_heads"] = 1
    payload["attention_impl"] = "flash_attn"
    with pytest.raises(ValidationError, match="2 <= n_heads"):
        TransformerConfig(**payload)


def test_rope_theta_zero_rejected():
    payload = base_payload()
    payload["rope_theta"] = 0
    with pytest.raises(ValidationError):
        TransformerConfig(**payload)


# --------------------------------------------------------------------------- #
# Derived behaviour
# --------------------------------------------------------------------------- #


def test_rope_cache_capped_by_theta():
    payload = base_payload()
    payload["context_length"] = 512
    cfg = TransformerConfig(**payload)
    assert cfg.rope_max_seq() == 512          # theta 10k >= 10k -> unlimited
    payload["rope_theta"] = 100.0
    cfg = TransformerConfig(**payload)
    assert cfg.rope_max_seq() == 100          # capped at theta
    payload["rope_theta"] = 64.5
    cfg = TransformerConfig(**payload)
    assert cfg.rope_max_seq() == 64           # floored


def test_output_proj_std_scaling():
    payload = base_payload()
    payload["n_layers"] = 2
    payload["init_std"] = 0.1
    cfg = TransformerConfig(**payload)
    assert cfg.output_init_std() == pytest.approx(0.1 / 2.0)


def test_preserve_init_is_schema_valid_but_engine_enforced():
    """PRESERVE passes schema (needs a parent at build time)."""
    payload = base_payload()
    payload["weight_init"] = WeightInit.PRESERVE.value
    cfg = TransformerConfig(**payload)
    assert cfg.weight_init == WeightInit.PRESERVE
