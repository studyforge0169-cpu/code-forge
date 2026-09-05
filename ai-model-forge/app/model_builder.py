"""Configurable decoder-only Transformer: architecture + model factory.

Implements every component requested for milestone 1:
  tokenizer-agnostic embeddings, RoPE / learned / no positional encoding,
  RMSNorm, causal multi-head attention, GQA (n_kv_heads <= n_heads, uniform
  groups), fused or separate Q/K/V projections, attention output projection,
  residual connections, SwiGLU / GELU / ReLU MLP, final norm, LM head
  (optional weight tying), autoregressive generation with KV cache,
  configurable precision.

The architecture is fully driven by ``TransformerConfig`` — the same code
builds a 2-layer toy or a 40-layer model; nothing is hard-coded. Only the
``decoder_only_transformer`` architecture exists so far; unknown
architectures are rejected, never silently mapped.

KV-cache contract
-----------------
Every ``forward`` returns per-layer (k, v) caches in (B, kv_heads, T, head_dim)
layout, so prefill and decoding share ONE code path: decode steps simply pass
the previous caches in. Position alignment inside attention always treats the
query rows as the *last* ``T`` tokens of the concatenated key sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as forge_cfg
from .hardware import detect_hardware
from .schemas import (
    Activation,
    AttentionImpl,
    Normalization,
    PositionEncoding,
    Precision,
    QKVMerging,
    TransformerConfig,
    WeightInit,
)

log = forge_cfg.get_logger("model")

NEG_INF = -3.4e38  # safe "finite -inf" for fp16/bf16 additive masks


# =========================================================================== #
# Layer primitives
# =========================================================================== #


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias). Statistics computed in fp32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """Interleaved-pair rotary embeddings (GPT-NeoX style) with a cached table.

    Buffers are non-persistent: they never enter the state dict, parameter
    counts, or optimizer state.
    """

    def __init__(self, head_dim: int, max_seq: int, theta: float = 10000.0, scaling: float = 1.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head dimension"
        self.head_dim = head_dim
        self.max_seq = max_seq
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        inv_freq = inv_freq / scaling
        pos = torch.arange(max_seq, dtype=torch.float32)
        angles = torch.einsum("m,k->mk", pos, inv_freq)  # (max_seq, head_dim/2)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of x's last dim. x: (..., T, head_dim), positions: (T,)."""
        cos = self.cos_cached[positions]  # (T, head_dim/2)
        sin = self.sin_cached[positions]
        view = (1,) * (x.dim() - 2) + cos.shape  # broadcast over leading dims
        cos, sin = cos.view(view), sin.view(view)
        pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        even, odd = pair[..., 0], pair[..., 1]
        new_even = even * cos - odd * sin
        new_odd = odd * cos + even * sin
        return torch.stack([new_even, new_odd], dim=-1).reshape_as(x)


def causal_additive_mask(q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(q_len, kv_len) additive causal mask; rows = last q_len keys.

    The masked value is ``-finfo.max`` (representable in fp16/bf16; -inf
    would work for fp32 only and NaN-poison fp16 matmuls).
    """
    neg = -torch.finfo(dtype).max
    q_abs = torch.arange(kv_len - q_len, kv_len, device=device)   # absolute positions of query rows
    k_abs = torch.arange(kv_len, device=device)
    allowed = k_abs.unsqueeze(0) <= q_abs.unsqueeze(1)
    mask = torch.full((q_len, kv_len), neg, device=device, dtype=dtype)
    return mask.masked_fill(allowed, 0.0)


# =========================================================================== #
# Attention
# =========================================================================== #


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with GQA, fused or separate projections.

    Projections produce tensors in (B, heads, T, head_dim) layout so that
    rotation, cache concatenation and scoring share a single representation.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim()
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.softcap = cfg.attention_softcap
        self.attn_drop = nn.Dropout(cfg.attention_dropout)

        hidden = cfg.hidden_size
        if cfg.qkv_merging == QKVMerging.FUSED:
            self.qkv = nn.Linear(hidden, cfg.qkv_output_size(), bias=False)
            self.q = self.k = self.v = None
        else:
            self.qkv = None
            self.q = nn.Linear(hidden, hidden, bias=False)
            self.k = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
            self.v = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)

    # ------------------------------------------------------------------ #

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linear projections -> q (B,n_heads,T,hd), k/v (B,n_kv_heads,T,hd)."""
        B, T, _ = x.shape
        if self.qkv is not None:
            qkv = self.qkv(x).view(B, T, self.n_heads + 2 * self.n_kv_heads, self.head_dim)
            q = qkv[:, :, : self.n_heads, :].transpose(1, 2)
            k = qkv[:, :, self.n_heads : self.n_heads + self.n_kv_heads, :].transpose(1, 2)
            v = qkv[:, :, self.n_heads + self.n_kv_heads :, :].transpose(1, 2)
        else:
            q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def rotate(self, q: torch.Tensor, k: torch.Tensor, rope: "RotaryEmbedding", positions: torch.Tensor):
        if rope is not None:
            q = rope.apply(q, positions)
            k = rope.apply(k, positions)
        return q, k

    def core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attention math + cache append. q (B,n_heads,T,hd); k/v (B,n_kv,T,hd).

        Returns (context (B,T,n_heads*hd), new_cache (k, v) full so far).
        """
        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
        new_cache = (k, v)

        groups = self.groups
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        scale = self.head_dim**-0.5
        att = (q * scale) @ k.transpose(-2, -1)  # (B, heads, T, T_kv)
        if self.softcap is not None:
            att = self.softcap * torch.tanh(att / self.softcap)
        if mask is not None:
            att = att + mask.unsqueeze(0).unsqueeze(0)
        att = torch.softmax(att.float(), dim=-1).to(att.dtype)
        if self.training and self.attn_drop.p > 0.0:
            att = self.attn_drop(att)
        out = att @ v  # (B, heads, T, hd)
        B, _, T, hd = out.shape
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * hd)
        return self.o(out), new_cache


# =========================================================================== #
# MLP
# =========================================================================== #


def build_mlp(cfg: TransformerConfig) -> nn.Module:
    if cfg.activation == Activation.SWIGLU:
        return _SwiGLUMLP(cfg)
    return _PlainMLP(cfg)


class _SwiGLUMLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.drop = nn.Dropout(cfg.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.down(F.silu(gate) * up))


class _PlainMLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.drop = nn.Dropout(cfg.mlp_dropout)
        if cfg.activation == Activation.GELU:
            self._act = lambda t: F.gelu(t, approximate="tanh")
        else:
            self._act = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(self._act(self.up(x))))


# =========================================================================== #
# Transformer block
# =========================================================================== #


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int):
        super().__init__()
        assert cfg.normalization == Normalization.RMSNORM, "only rmsnorm is implemented"
        self.ln1 = RMSNorm(cfg.hidden_size, cfg.norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size, cfg.norm_epsilon)
        self.mlp = build_mlp(cfg)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RotaryEmbedding],
        positions: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.ln1(x)
        q, k, v = self.attn.project(h)
        q, k = self.attn.rotate(q, k, rope, positions)
        attn_out, new_cache = self.attn.core(q, k, v, kv_cache, mask)
        x = x + attn_out                     # residual
        x = x + self.mlp(self.ln2(x))        # residual
        return x, new_cache


# =========================================================================== #
# Whole model
# =========================================================================== #


class ForgeTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.architecture.value != "decoder_only_transformer":
            raise ValueError(f"architecture '{cfg.architecture}' is not supported by ForgeTransformer")

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.emb_drop = nn.Dropout(cfg.embedding_dropout)

        self.rope: Optional[RotaryEmbedding] = None
        self.pos_emb: Optional[nn.Parameter] = None
        if cfg.position_encoding == PositionEncoding.ROPE:
            self.rope = RotaryEmbedding(cfg.head_dim(), cfg.rope_max_seq(), cfg.rope_theta, cfg.rope_scaling)
        elif cfg.position_encoding == PositionEncoding.LEARNED:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.context_length, cfg.hidden_size))

        self.blocks = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.norm_epsilon)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_output_embeddings:
            # One shared parameter object: counted, stored and trained once.
            self.lm_head.weight = self.tok_emb.weight

    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """(B, T) token ids -> (logits (B,T,V) fp32, per-layer kv caches)."""
        B, T = input_ids.shape
        device = input_ids.device
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :T, :]
        x = self.emb_drop(x)

        # Resolve cache state and positional bookkeeping once for all layers.
        cache_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            cache_len = kv_caches[0][0].shape[2]
        total_kv = cache_len + T
        if total_kv > self.cfg.context_length:
            raise ValueError(
                f"sequence (cache {cache_len} + new {T}) exceeds context_length "
                f"({self.cfg.context_length})"
            )
        positions = torch.arange(cache_len, cache_len + T, device=device)
        mask = causal_additive_mask(T, total_kv, device, dtype=x.dtype)

        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, self.rope, positions, layer_cache, mask)
            caches.append(new_cache)

        x = self.final_norm(x)
        return self.lm_head(x).float(), caches

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,               # (1, T0) int64
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive decode with an incremental KV cache.

        Returns input + generated ids of shape (1, T0 + n). Deterministic for
        a fixed seed; logits are always sampled in fp32.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("generate() expects one sequence: shape (1, T)")
        if input_ids.shape[1] >= self.cfg.context_length:
            raise ValueError(f"prompt length {input_ids.shape[1]} already at context limit")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 or None")

        self.eval()
        device = input_ids.device
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        max_total = min(input_ids.shape[1] + max_new_tokens, self.cfg.context_length)
        tokens = input_ids.clone()
        logits, caches = self(input_ids)  # prefill; caches warm the decode loop

        generated = 0
        while generated < max_new_tokens and tokens.shape[1] < max_total:
            last_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth = torch.topk(last_logits, min(top_k, last_logits.shape[-1]), dim=-1).values[:, -1:]
                last_logits = last_logits.masked_fill(last_logits < kth, NEG_INF)
            probs = torch.softmax(last_logits.float(), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            generated += 1
            if eos_token_id is not None and int(nxt.item()) == eos_token_id:
                tokens = torch.cat([tokens, nxt], dim=1)
                break
            tokens = torch.cat([tokens, nxt], dim=1)
            logits, caches = self(nxt, kv_caches=caches)  # 1-token decode step
        return tokens


# =========================================================================== #
# Initialization
# =========================================================================== #


@torch.no_grad()
def apply_normal_init(model: nn.Module, cfg: TransformerConfig) -> None:
    """Default forge init: weights N(0, std); residual-output weights scaled down.

    Runs under no_grad (like nn.init): in-place initialization of
    grad-tracking leaves is otherwise rejected by torch.
    """
    out_std = cfg.output_init_std()
    for name, param in model.named_parameters():
        if name == "lm_head.weight" and cfg.tie_output_embeddings:
            continue  # shared with tok_emb — already initialized
        if param.ndim >= 2 and (name.endswith("o.weight") or name.endswith("down.weight")):
            param.normal_(mean=0.0, std=out_std)
        elif param.ndim >= 1:
            param.normal_(mean=0.0, std=cfg.init_std)
        else:
            param.zero_()


def count_parameters(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        if id(param) not in seen:  # tied embeddings counted exactly once
            seen.add(id(param))
            total += param.numel()
    return total


def param_rows_summary(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if name == "lm_head.weight" and param is model.tok_emb.weight:  # type: ignore[attr-defined]
            continue  # aliased row documented on tok_emb instead
        rows.append({"module": name, "shape": list(param.shape), "numel": int(param.numel())})
    return rows


# =========================================================================== #
# Factory
# =========================================================================== #


@dataclass
class BuiltModel:
    config: TransformerConfig
    module: nn.Module
    parameter_count: int
    dtype: str
    seed: int
    param_rows: list[dict[str, Any]] = field(default_factory=list)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.module.state_dict().items()}


def flash_attention_supported() -> bool:
    """flash_attn needs an NVIDIA CUDA GPU (fused kernels are unavailable on CPU)."""
    return detect_hardware().has_gpu


def build_transformer(
    cfg: TransformerConfig,
    parent_state: Optional[dict[str, torch.Tensor]] = None,
    device: str = "cpu",
) -> BuiltModel:
    """Materialize a validated config into a fresh, initialized model.

    ``parent_state`` enables exact weight reuse (future training stages).
    Any shape incompatibility raises *before* bytes are written.
    """
    if cfg.architecture.value != "decoder_only_transformer":
        raise ValueError(f"architecture '{cfg.architecture}' is not implemented yet")
    if cfg.attention_impl == AttentionImpl.FLASH_ATTN and not flash_attention_supported():
        raise ValueError(
            "attention_impl='flash_attn' needs an NVIDIA CUDA GPU with compatible torch "
            "kernels; use 'eager' on this machine (it is fully portable)"
        )
    if cfg.weight_init == WeightInit.PRESERVE and parent_state is None:
        raise ValueError("weight_init='preserve' requires parent weights to reuse")

    seed = cfg.effective_seed()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ForgeTransformer(cfg)
        if parent_state is not None:
            missing, unexpected = restore_state(model, parent_state, strict_shapes=True)
            if missing or unexpected:
                raise ValueError(
                    f"parent state is incompatible with this config "
                    f"({len(missing)} missing / {len(unexpected)} unexpected parameters)"
                )
        else:
            apply_normal_init(model, cfg)

    dtype = {Precision.FP32: torch.float32, Precision.FP16: torch.float16, Precision.BF16: torch.bfloat16}[cfg.precision]
    model.to(dtype=dtype)
    if device != "cpu":
        model.to(device)

    return BuiltModel(
        config=cfg,
        module=model,
        parameter_count=count_parameters(model),
        dtype=cfg.precision.value,
        seed=seed,
        param_rows=param_rows_summary(model),
    )


def restore_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    strict_shapes: bool = True,
) -> tuple[list[str], list[str]]:
    """Copy a state dict into a model. Returns (missing, unexpected) keys."""
    own = model.state_dict()
    missing = [k for k in own if k not in state]
    unexpected = [k for k in state if k not in own]
    for key, tensor in state.items():
        if key not in own:
            continue
        target = own[key]
        if tuple(target.shape) != tuple(tensor.shape):
            if strict_shapes:
                raise ValueError(
                    f"parameter '{key}': parent shape {tuple(tensor.shape)} != "
                    f"expected {tuple(target.shape)}"
                )
            continue
        target.copy_(tensor)
    return missing, unexpected


def content_hash(state: dict[str, torch.Tensor]) -> str:
    """Deterministic content hash over *values* (fp32 canonical form).

    fp16/bf16 weights widen exactly to fp32, so the same numbers yield the
    same hash regardless of the storage dtype. This is the dedup key.
    """
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        t = state[key].detach().float().contiguous().cpu()
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def model_size_bytes(model: nn.Module) -> int:
    per = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
    total = 0
    for param in model.parameters():
        total += param.numel() * per[param.dtype]
    return total
