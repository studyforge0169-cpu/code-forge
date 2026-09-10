"""Data schemas: model configuration, requests, records.

This is the single source of truth for every persisted/transmitted object.
Everything validates through Pydantic so an invalid combination can never
reach disk or the training engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Shared name rule for user-named artifacts (models, tokenizers, datasets).
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$"

# --------------------------------------------------------------------------- #
# Enumerations (extensible; unknown values are rejected, never silently mapped)
# --------------------------------------------------------------------------- #


class Architecture(str, Enum):
    DECODER_ONLY_TRANSFORMER = "decoder_only_transformer"


class Activation(str, Enum):
    SWIGLU = "swiglu"        # gated MLP: hidden -> 2*intermediate, chunked
    GELU = "gelu"            # plain MLP with GELU (tanh approximation)
    RELU = "relu"            # plain MLP with ReLU


class Normalization(str, Enum):
    RMSNORM = "rmsnorm"


class PositionEncoding(str, Enum):
    ROPE = "rope"            # rotary position embeddings (GPT-NeoX style, cached)
    LEARNED = "learned"      # learned positional embeddings
    NONE = "none"


class AttentionImpl(str, Enum):
    EAGER = "eager"          # portable fused-dot-product math on any device
    FLASH_ATTN = "flash_attn"  # flash-attention kernel; requires NVIDIA CUDA GPU


class QKVMerging(str, Enum):
    FUSED = "fused"          # one Linear with (n_heads + 2*n_kv_heads) output rows
    SEPARATE = "separate"    # three independent Linear layers


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class WeightInit(str, Enum):
    NORMAL = "normal"        # default forge scheme (std configurable)
    PRESERVE = "preserve"    # reuse the parent model's weights (stage chaining)


class InitializationScope(str, Enum):
    FULL = "full"            # init all parameters
    # LAYERS_KEEP_SHARED = ... (reserved for continued-pretraining milestones)


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #


class TransformerConfig(BaseModel):
    """Fully configurable decoder-only Transformer.

    Constraints enforced here (plus pairwise rules in the model validator):
      * ``vocab_size >= 16``, ``context_length >= 4``, ``hidden_size >= 8``
      * ``n_layers >= 1``, ``n_heads >= 1``, ``n_kv_heads >= 1``
      * GQA/MHA shape laws:
          - ``hidden_size % n_heads == 0``   (per-head dim must be an integer)
          - ``n_heads % n_kv_heads == 0``    (query groups must be uniform)
          - per-head dim >= 2
      * ``hidden_size >= 2 * n_heads``
      * SwiGLU needs an even ``intermediate_size`` (chunked in two)
      * ``tie_output_embeddings`` requires ``vocab_size == hidden_size``
      * RoPE: the rope cache is capped at ``min(context_length, rope_theta)``
        when ``rope_theta < 10000`` (positions beyond theta are meaningless);
        an explicit ``max_seq_len`` override is therefore optional.
      * flash attention requires 2 <= n_heads <= 256 *and* a CUDA device
        (device requirement is enforced at build time by the model engine).

    Names must match ``^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$``.
    """

    name: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    architecture: Architecture = Architecture.DECODER_ONLY_TRANSFORMER

    # Dimensions
    vocab_size: int = Field(..., ge=16)
    context_length: int = Field(..., ge=4)
    hidden_size: int = Field(..., ge=8)
    n_layers: int = Field(..., ge=1)
    n_heads: int = Field(..., ge=1)
    n_kv_heads: int = Field(..., ge=1)
    intermediate_size: int = Field(..., ge=4)

    # Components
    activation: Activation = Activation.SWIGLU
    normalization: Normalization = Normalization.RMSNORM
    position_encoding: PositionEncoding = PositionEncoding.ROPE
    rope_theta: float = Field(10000.0, gt=0.0)
    rope_scaling: float = Field(1.0, gt=0.0, le=10_000.0)  # linear scaling factor
    attention_impl: AttentionImpl = AttentionImpl.EAGER
    qkv_merging: QKVMerging = QKVMerging.FUSED
    precision: Precision = Precision.FP32
    tie_output_embeddings: bool = False

    # Regularization / numerics
    embedding_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    attention_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    mlp_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    norm_epsilon: float = Field(1e-5, gt=0.0, le=1.0)
    attention_softcap: Optional[float] = Field(None, gt=0.0)  # logit soft-cap (Gemma-style)

    # Initialization
    weight_init: WeightInit = WeightInit.NORMAL
    init_std: float = Field(0.02, gt=0.0)
    init_std_output_proj: Optional[float] = None  # None -> init_std / sqrt(2*n_layers)

    # Optional per-config random seed (None -> derived deterministically from the config)
    seed: Optional[int] = Field(None, ge=0)

    # Reserved for later milestones (accepted now so configs created later by
    # the orchestrator can be persisted without schema churn; unused by the builder).
    max_seq_len: Optional[int] = Field(None, ge=1)
    window_attention: Optional[int] = Field(None, ge=1)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v >= 2**32:
            raise ValueError("seed must fit in 32 bits")
        return v

    @model_validator(mode="after")
    def _validate_pairwise(self) -> "TransformerConfig":
        errors: list[str] = []
        h, heads, kv = self.hidden_size, self.n_heads, self.n_kv_heads

        if kv > heads:
            errors.append(f"n_kv_heads ({kv}) cannot exceed n_heads ({heads})")
        if heads % kv != 0:
            errors.append(f"n_heads ({heads}) must be a multiple of n_kv_heads ({kv}) for uniform GQA groups")
        if h % heads != 0:
            errors.append(f"hidden_size ({h}) must be divisible by n_heads ({heads})")
        head_dim = h // heads
        if head_dim < 2:
            errors.append(f"per-head dimension is {head_dim}; at least 2 is required")
        if h < 2 * heads:
            errors.append(f"hidden_size ({h}) must be at least 2 * n_heads ({2 * heads})")
        if self.activation == Activation.SWIGLU and self.intermediate_size % 2 != 0:
            errors.append(f"intermediate_size ({self.intermediate_size}) must be even for swiglu "
                          "(it is split into gate and up halves)")

        if self.position_encoding == PositionEncoding.ROPE:
            if self.rope_scaling != 1.0 and self.rope_theta < 10000.0:
                errors.append("rope_scaling != 1 is only supported together with rope_theta >= 10000")
        elif self.rope_theta != 10000.0 or self.rope_scaling != 1.0:
            errors.append("rope_theta/rope_scaling only apply when position_encoding == 'rope'")

        if self.attention_impl == AttentionImpl.FLASH_ATTN:
            if not (2 <= heads <= 256):
                errors.append("flash attention requires 2 <= n_heads <= 256")
            if kv > 1 and self.window_attention is not None:
                errors.append("windowed flash attention is not supported with GQA (n_kv_heads > 1)")

        if self.tie_output_embeddings and self.vocab_size != self.hidden_size:
            errors.append("tie_output_embeddings requires vocab_size == hidden_size "
                          "(token embeddings and the LM head must share one matrix)")

        if self.weight_init == WeightInit.PRESERVE:
            # PRESERVE has no meaning on a from-scratch model. The engine also
            # verifies a parent state actually exists at build time.
            pass  # legality depends on the parent model, checked by the engine

        if self.max_seq_len is not None:
            if self.max_seq_len < 1:
                errors.append("max_seq_len must be >= 1")
            if self.max_seq_len > self.context_length:
                errors.append(f"max_seq_len ({self.max_seq_len}) cannot exceed context_length ({self.context_length})")
            if self.position_encoding == PositionEncoding.ROPE and self.max_seq_len > self.rope_max_seq():
                errors.append(
                    f"max_seq_len ({self.max_seq_len}) exceeds the RoPE cache length "
                    f"({self.rope_max_seq()}) implied by rope_theta={self.rope_theta:g}"
                )

        if errors:
            raise ValueError("; ".join(errors))
        return self

    # ------------------------------------------------------------------ #
    # Derived values used by the builder
    # ------------------------------------------------------------------ #

    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    def rope_max_seq(self) -> int:
        """Longest position the RoPE cache covers.

        Positions with angle >= rope_theta are meaningless for direct
        (unscaled) rotary embeddings, so the cache is truncated at theta
        unless theta >= 10000 (treated as "effectively unlimited").
        """
        if self.rope_theta >= 10_000.0:
            return self.context_length
        return min(self.context_length, max(1, int(self.rope_theta // self.rope_scaling)))

    def qkv_output_size(self) -> int:
        """Row count of the fused QKV projection (heads + 2 * kv-heads, in head units)."""
        return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim()

    def default_seed(self) -> int:
        """Deterministic seed derived from the configuration itself."""
        import hashlib
        import json

        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16)

    def effective_seed(self) -> int:
        return self.default_seed() if self.seed is None else self.seed

    def output_init_std(self) -> float:
        if self.init_std_output_proj is not None:
            return self.init_std_output_proj
        return self.init_std / (2.0 * self.n_layers) ** 0.5

    # ------------------------------------------------------------------ #

    @staticmethod
    def tiny() -> "TransformerConfig":
        """Minimal valid config used by unit tests and smoke checks."""
        return TransformerConfig(
            name="forge-tiny",
            vocab_size=64,
            context_length=32,
            hidden_size=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=48,
        )

    @staticmethod
    def recommended_minimum() -> "TransformerConfig":
        """Smallest config that still behaves like a real small LM (~2.5M params)."""
        return TransformerConfig(
            name="forge-mini",
            vocab_size=256,
            context_length=256,
            hidden_size=192,
            n_layers=4,
            n_heads=6,
            n_kv_heads=2,
            intermediate_size=512,
        )


# --------------------------------------------------------------------------- #
# API request / response objects
# --------------------------------------------------------------------------- #


class ModelCreateRequest(BaseModel):
    config: TransformerConfig
    description: Optional[str] = Field(None, max_length=1024)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=64)

    model_config = ConfigDict(extra="forbid")


class ValidationReport(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    name: str = "default"
    storage_root: str
    initialized_at: str
    app_version: str
    schema_version: int
    model_count: int = 0
    dataset_count: int = 0
    tokenizer_count: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


class TokenizerType(str, Enum):
    BYTE_BPE = "byte_bpe"


class TokenizerConfig(BaseModel):
    """Training-time configuration for a byte-level BPE tokenizer.

    ``vocab_size`` is the *target* maximum: real corpora can legitimately
    produce a smaller vocabulary (byte tokens + merges actually observed),
    so the final size is recorded as ``actual_vocab_size`` on the record.

    Special tokens must be non-empty, unique, and contain no whitespace.
    ``seed`` is accepted for forward compatibility (corpus shuffling when
    a future milestone adds it) but is not used today: training is fully
    deterministic because the corpus order is fixed by ingestion.
    """

    name: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    vocab_size: int = Field(..., ge=256, le=65536)
    special_tokens: list[str] = Field(default_factory=lambda: ["<unk>", "<s>", "</s>"])
    byte_fallback: bool = True
    seed: Optional[int] = Field(None, ge=0, lt=2**32)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("special_tokens")
    @classmethod
    def _validate_specials(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("special tokens must be non-empty strings")
        if any(any(ch.isspace() for ch in s) for s in cleaned):
            raise ValueError("special tokens must not contain whitespace")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"special tokens must be unique, got duplicates in {cleaned}")
        return cleaned


class TokenizerRecord(BaseModel):
    """Persisted record of a trained tokenizer (tokenizers/<id>/manifest.json)."""

    id: str
    name: str
    type: TokenizerType = TokenizerType.BYTE_BPE
    requested_vocab_size: int
    actual_vocab_size: int
    special_tokens: list[str]
    byte_fallback: bool = True
    seed: Optional[int] = None
    trained_on_dataset_id: Optional[str] = None
    config_hash: str
    tokenizer_hash: str            # sha256 of tokenizer.json bytes
    created_at: datetime
    schema_version: int = 1
    hardware: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class SourceFileInfo(BaseModel):
    """Per-input-file result: metadata + hash only (never a file copy)."""

    filename: str
    extension: str
    size: int
    sha256: str
    status: str                    # ok | error
    record_count: int = 0
    error_count: int = 0
    columns: Optional[list[str]] = None   # CSV header when present
    error: Optional[str] = None


class CountsInfo(BaseModel):
    input_record_count: int = 0        # rows/paragraphs extracted from files
    empty_record_count: int = 0        # dropped blanks
    invalid_record_count: int = 0      # malformed rows (CSV ragged etc.)
    duplicate_record_count: int = 0    # dropped within this upload (sha256)
    duplicate_of_previous_versions: int = 0  # present in earlier versions
    unique_record_count: int = 0       # stored in this version


class QualityInfo(BaseModel):
    total_characters: int = 0
    total_bytes: int = 0
    min_length: int = 0
    max_length: int = 0
    mean_length: float = 0.0
    median_length: float = 0.0


class SplitInfo(BaseModel):
    count: int = 0
    sha256: str = ""               # hash of ordered record ids in the split


class TokenizedInfo(BaseModel):
    tokenizer_id: str
    tokenizer_name: str = ""
    tokenizer_hash: str = ""
    dtype: str                     # uint16 | uint32
    created_at: datetime
    splits: dict[str, SplitInfo] = Field(default_factory=dict)  # per split: count=token_count


class VersionManifest(BaseModel):
    """Everything needed to reproduce + verify one immutable dataset version."""

    dataset_id: str
    version: int
    created_at: datetime
    schema_version: int = 1
    records_file: str = "records.jsonl.gz"
    allocation: dict[str, Any] = Field(default_factory=lambda: {
        "method": "sha256_bucket_mod1000",
        "boundaries": {"train": 900, "validation": 950, "test": 1000},
        "note": "bucket = int(record_hash[:8], 16) % 1000; <900 train, <950 validation, else test",
    })
    sources: list[SourceFileInfo] = Field(default_factory=list)
    counts: CountsInfo = Field(default_factory=CountsInfo)
    quality: QualityInfo = Field(default_factory=QualityInfo)
    splits: dict[str, SplitInfo] = Field(default_factory=dict)
    tokenized: dict[str, TokenizedInfo] = Field(default_factory=dict)  # tokenizer_id -> info


class DatasetInfo(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    versions: list[int] = Field(default_factory=list)
    latest_version: int = 0
    total_records: int = 0
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


class TrainingMethod(str, Enum):
    CONTINUED_PRETRAINING = "continued_pretraining"
    SFT = "sft"


class LRSchedule(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class CheckpointDecision(str, Enum):
    ACCEPT = "accept"          # validation loss beat the best so far
    NOT_BEST = "not_best"      # kept for history, did not improve


class TrainingConfig(BaseModel):
    """Validated configuration for one training run (CPT or SFT).

    Semantics of this milestone (documented, honest):
      * both methods train with standard causal LM loss over tokenized text;
        no chat templates, role masking or special SFT loss yet
      * ``epochs`` XOR ``steps`` must be set (never both, never neither)
      * ``batch_size`` is the micro-batch size; the effective update batch is
        ``batch_size * gradient_accumulation_steps``
      * training is deterministic for a fixed seed (default: derived from
        the config) on a fixed backend
    """

    name: str = Field(default="run", min_length=1, max_length=64, pattern=NAME_PATTERN)
    method: TrainingMethod = TrainingMethod.CONTINUED_PRETRAINING
    model_id: str = Field(..., min_length=1, max_length=64)
    dataset_id: str = Field(..., min_length=1, max_length=64)
    dataset_version: Optional[int] = Field(None, ge=1)   # None -> latest version
    tokenizer_id: str = Field(..., min_length=1, max_length=64)

    # M54: explicit training resume point — initialize THIS run from an
    # immutable checkpoint of the SAME model, WITHOUT publishing it first
    # (no rollback, no latest_checkpoint mutation to prepare the run; the
    # model's published state only changes through the normal training
    # completion semantics). None (default) = today's behavior: start from
    # the model's current published weights. Model-scoped and verified like
    # every checkpoint reference (manifest + weights content hash); the
    # checkpoint is REFERENCED, never copied. Model-WEIGHT resume only —
    # optimizer/scheduler state is NOT persisted by M3 and is NOT restored.
    resume_from_checkpoint_id: Optional[str] = Field(
        None, min_length=1, max_length=64)

    # M55: DECLARATIVE best-resume — a WORKFLOW training-stage option.
    # True means "resolve the M52 best checkpoint (minimum persisted
    # validation_loss) at workflow resolution time and initialize this
    # run from it". The workflow engine's single resolver pins the
    # concrete id on ``resolved_resume_checkpoint_id`` (the M53 pinned
    # form); the training engine then receives it as a PURE M54
    # ``resume_from_checkpoint_id``. XOR with the explicit id (both set
    # is a contradiction, never a silent preference); a DIRECT training
    # request with resume_from_best=True is rejected — direct runs name
    # the checkpoint explicitly (e.g. GET /checkpoints/best answers it).
    resume_from_best: bool = False
    resolved_resume_checkpoint_id: Optional[str] = Field(
        None, min_length=1, max_length=64)   # pinned by the workflow
    # resolver ONLY (the M52 selection's concrete id); valid only with
    # resume_from_best=True

    # Optimizer / schedule
    learning_rate: float = Field(3e-4, ge=1e-6, le=1.0)
    lr_schedule: LRSchedule = LRSchedule.COSINE
    warmup_steps: int = Field(0, ge=0)
    weight_decay: float = Field(0.01, ge=0.0, le=1.0)
    adam_beta1: float = Field(0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(0.999, gt=0.0, lt=1.0)
    max_grad_norm: Optional[float] = Field(1.0, gt=0.0)  # None disables clipping

    # Data / batching
    batch_size: int = Field(8, ge=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    max_seq_len: int = Field(..., ge=2)      # <= model.context_length (engine check)
    epochs: Optional[int] = Field(None, ge=1)
    steps: Optional[int] = Field(None, ge=1)

    # Evaluation / checkpointing
    eval_every_steps: int = Field(25, ge=1)
    keep_best: bool = True

    seed: Optional[int] = Field(None, ge=0, lt=2**32)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _epochs_xor_steps(self) -> "TrainingConfig":
        if (self.epochs is None) == (self.steps is None):
            raise ValueError("set exactly one of 'epochs' or 'steps' (not both, not neither)")
        if self.resume_from_best and self.resume_from_checkpoint_id is not None:
            raise ValueError(
                "resume_from_best and resume_from_checkpoint_id are "
                "mutually exclusive — the M52 selection decides the "
                "checkpoint OR it is named explicitly, never both")
        if self.resolved_resume_checkpoint_id is not None \
                and not self.resume_from_best:
            raise ValueError(
                "resolved_resume_checkpoint_id is the workflow "
                "resolver's pinned M52 selection and is only valid with "
                "resume_from_best=True")
        return self

    def default_seed(self) -> int:
        import hashlib
        import json

        blob = json.dumps(self.model_dump(mode="json", exclude={"seed"}), sort_keys=True)
        return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16)

    def effective_seed(self) -> int:
        return self.default_seed() if self.seed is None else self.seed


class RunProvenance(BaseModel):
    """One training run appended to the model manifest (lineage record)."""

    run_id: str
    method: TrainingMethod
    config: dict[str, Any]           # full TrainingConfig JSON (single copy per run)
    seed: int
    dataset_id: str
    dataset_version: int
    tokenizer_id: str
    parent_checkpoint_id: Optional[str] = None
    initial_checkpoint_id: Optional[str] = None
    final_checkpoint_id: Optional[str] = None
    requested_batch_size: int = 0
    actual_batch_size: int = 0
    gradient_accumulation_steps: int = 1
    optimizer_steps: int = 0
    epochs_run: float = 0.0
    baseline_validation_loss: Optional[float] = None
    best_validation_loss: Optional[float] = None
    best_perplexity: Optional[float] = None
    initial_train_loss: Optional[float] = None
    final_train_loss: Optional[float] = None
    accepted: bool = False
    rolled_back_to: Optional[str] = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = 0.0


class CheckpointRecord(BaseModel):
    """Immutable per-checkpoint manifest (checkpoints/<id>/manifest.json)."""

    checkpoint_id: str
    model_id: str
    run_id: str
    parent_checkpoint_id: Optional[str] = None   # lineage within/across runs
    step: int                                    # optimizer step at save time
    epoch: float                                 # step / steps-per-epoch
    method: TrainingMethod
    dataset_id: str
    dataset_version: int
    train_loss: float
    validation_loss: float
    perplexity: float
    learning_rate: float
    decision: CheckpointDecision
    weights_sha256: str                          # content hash (dedup key)
    created_at: datetime
    schema_version: int = 1


class CheckpointSelection(BaseModel):
    """Read-only SELECTION of ONE model's best checkpoint under the
    persisted validation-loss criterion (M52) — a computed view, never
    a record.

    "Best" means exactly one thing here: the checkpoint with the
    MINIMUM persisted ``CheckpointRecord.validation_loss`` among the
    model's checkpoints in the authoritative M3 listing (the persisted
    manifest is the source — validation loss is never recomputed and
    never derived from perplexity, decisions, evaluations, ids or
    timestamps). This is NOT a claim of overall model quality: a lower
    validation loss is selection evidence under this one criterion
    only. Ties on the exact minimum resolve by the listing's canonical
    (step, created_at) ASCENDING order — the first checkpoint among
    equals — and are disclosed via ``tied``. Checkpoints whose
    persisted ``validation_loss`` is not finite are never candidates.
    Never persisted, never written; no selection pointer exists.
    """

    model_id: str
    criterion: Literal["minimum_persisted_validation_loss"]
    candidate_count: int = Field(ge=1)
    tied: bool
    checkpoint: CheckpointRecord

    model_config = ConfigDict(extra="forbid")


class RollbackRequest(BaseModel):
    checkpoint_id: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Evaluation (read-only measurement of an existing model state)
# --------------------------------------------------------------------------- #


class EvalStateKind(str, Enum):
    """Which state of a model an evaluation measured."""

    CURRENT = "current"          # the model's published weights.pt
    CHECKPOINT = "checkpoint"    # an immutable stored checkpoint of that model
    BEST = "best"                # M53 WORKFLOW state REFERENCE only: resolved
                                 # to a concrete checkpoint at execution or
                                 # M51-preflight time through the M52 selection
                                 # (minimum persisted validation_loss); never
                                 # persisted on measurement records — the
                                 # concrete resolved id is pinned instead


class EvaluationSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EvaluationConfig(BaseModel):
    """Validated configuration for one read-only evaluation.

    State selection rule: ``model_id`` is ALWAYS required. ``checkpoint_id``
    is scoped to that model — ``None`` means "evaluate the model's current
    published weights", otherwise that exact checkpoint is evaluated. No
    model/checkpoint XOR: a checkpoint id cannot identify its model alone.

    Evaluation never trains and never modifies anything: it measures an
    existing state against one deterministic tokenized split.
    """

    model_id: str = Field(..., min_length=1, max_length=64)
    checkpoint_id: Optional[str] = Field(None, min_length=1, max_length=64)
    dataset_id: str = Field(..., min_length=1, max_length=64)
    dataset_version: Optional[int] = Field(None, ge=1)  # None -> latest version
    split: EvaluationSplit
    tokenizer_id: str = Field(..., min_length=1, max_length=64)

    # Data handling: identical windowing/tail policy as M3 training eval.
    max_eval_tokens: Optional[int] = Field(None, ge=1)  # None -> whole split
    batch_size: int = Field(1, ge=1)
    max_seq_len: Optional[int] = Field(None, ge=2)      # window length; None -> model context

    seed: Optional[int] = Field(None, ge=0, lt=2**32)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def default_seed(self) -> int:
        import hashlib
        import json

        blob = json.dumps(self.model_dump(mode="json", exclude={"seed"}), sort_keys=True)
        return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16)

    def effective_seed(self) -> int:
        return self.default_seed() if self.seed is None else self.seed


class EvaluationRecord(BaseModel):
    """Immutable record of one evaluation (evaluations/eval-<id>/manifest.json).

    ``result_hash`` is a deterministic hash over the evaluation *inputs and
    results* (state hash, dataset/version/tokenizer identity, split, config,
    seed, metrics, counts) — it deliberately excludes eval_id, timestamps and
    paths, so two identical evaluations produce the same result_hash even
    though their manifests differ.
    """

    eval_id: str
    model_id: str
    state_kind: EvalStateKind
    checkpoint_id: Optional[str] = None
    state_hash: str                        # canonical fp32 tensor-content hash of the state
    dataset_id: str
    dataset_version: int
    split: EvaluationSplit
    tokenizer_id: str
    tokenized_bin_sha256: str              # identity of the exact evaluated bin
    loss_nats: float                       # token-weighted mean cross-entropy (nats)
    perplexity: float                      # exp(min(loss, 100))
    token_count: int                       # evaluated target tokens
    records_covered: Optional[int] = None  # split record count (full eval) / None (capped)
    truncated: bool = False                # max_eval_tokens stopped before the split ended
    config: dict[str, Any] = Field(default_factory=dict)  # full EvaluationConfig JSON
    seed: int                              # effective seed used (default derived from config)
    result_hash: str
    hardware: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    duration_seconds: float = 0.0
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Comparison (read-only orchestration over M4 evaluation records)
# --------------------------------------------------------------------------- #


class ComparisonVerdict(str, Enum):
    """Verdict of one comparison — ONLY about measured loss on one probe."""

    IMPROVED = "improved"      # loss_B < loss_A - tolerance
    REGRESSED = "regressed"    # loss_B > loss_A + tolerance
    UNCHANGED = "unchanged"    # |loss_B - loss_A| <= tolerance


class ComparisonState(BaseModel):
    """Which state of the model takes part in a comparison."""

    state_kind: EvalStateKind
    checkpoint_id: Optional[str] = Field(None, min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _kind_matches_id(self) -> "ComparisonState":
        if self.state_kind == EvalStateKind.BEST:
            raise ValueError(
                "state_kind='best' is a WORKFLOW state reference (M53): it is "
                "resolved to a concrete checkpoint by the workflow engine's "
                "single resolver at execution/preflight time; direct "
                "comparison/gate/suite-run requests must use 'current' or "
                "'checkpoint'")
        if self.state_kind == EvalStateKind.CURRENT and self.checkpoint_id is not None:
            raise ValueError(
                "state_kind='current' must have checkpoint_id=None")
        if self.state_kind == EvalStateKind.CHECKPOINT and not self.checkpoint_id:
            raise ValueError(
                "state_kind='checkpoint' requires a non-empty checkpoint_id")
        return self


class ComparisonRequest(BaseModel):
    """Compare two states of ONE model under ONE identical probe.

    Both states are evaluated (or an exact existing immutable evaluation is
    reused) on the same dataset/version/split/tokenizer/window/cap/seed, so
    only the model state differs. ``tolerance`` (>= 0, in nats) decides the
    verdict: |loss_B - loss_A| <= tolerance -> unchanged.
    """

    model_id: str = Field(..., min_length=1, max_length=64)
    state_a: ComparisonState
    state_b: ComparisonState

    # Shared probe conditions (identical for A and B by construction)
    dataset_id: str = Field(..., min_length=1, max_length=64)
    dataset_version: Optional[int] = Field(None, ge=1)  # None -> latest version
    split: EvaluationSplit
    tokenizer_id: str = Field(..., min_length=1, max_length=64)
    max_eval_tokens: Optional[int] = Field(None, ge=1)  # None -> whole split
    batch_size: int = Field(1, ge=1)
    max_seq_len: Optional[int] = Field(None, ge=2)      # window; None -> model context
    seed: Optional[int] = Field(None, ge=0, lt=2**32)

    tolerance: float = Field(1e-4, ge=0.0)              # nats

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def default_seed(self) -> int:
        import hashlib
        import json

        blob = json.dumps(self.model_dump(mode="json", exclude={"seed"}),
                          sort_keys=True)
        return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16)

    def effective_seed(self) -> int:
        return self.default_seed() if self.seed is None else self.seed


class ComparisonSide(BaseModel):
    """Per-state evidence embedded in a comparison record."""

    state_kind: EvalStateKind
    checkpoint_id: Optional[str] = None
    state_hash: str                       # canonical tensor-content hash
    evaluation_id: str                    # immutable M4 record backing this side
    evaluation_result_hash: str = ""      # that record's deterministic result hash
    loss_nats: float
    perplexity: float
    token_count: int


class ComparisonRecord(BaseModel):
    """Immutable record of one comparison (comparisons/comp-<id>/manifest.json).

    ``result_hash`` is deterministic over the immutable inputs/results (model
    config identity, both state hashes, both evaluation result hashes, probe
    identity, tolerance, metrics, verdict) — never comparison_id/timestamps —
    so repeating a comparison against unchanged states reproduces it.
    """

    comparison_id: str
    model_id: str
    config_hash: str                      # identity of the compared architecture
    state_a: ComparisonSide
    state_b: ComparisonSide
    dataset_id: str
    dataset_version: int
    split: EvaluationSplit
    tokenizer_id: str
    max_eval_tokens: Optional[int] = None
    batch_size: int
    max_seq_len: int                      # resolved window used by both evals
    seed: int
    loss_a: float
    loss_b: float
    perplexity_a: float
    perplexity_b: float
    delta_loss_nats: float                # loss_B - loss_A (negative = B improved)
    delta_perplexity: float
    tolerance: float
    verdict: ComparisonVerdict
    result_hash: str
    created_at: datetime
    duration_seconds: float = 0.0
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Stage gates (policy -> evidence -> explicit decision; M6)
# --------------------------------------------------------------------------- #


class GateBaselineType(str, Enum):
    """What a gate policy compares the candidate against."""

    CHECKPOINT = "checkpoint"            # a specific immutable checkpoint
    CURRENT = "current"                  # the model's published current weights
    EVALUATION_RESULT_HASH = "evaluation_result_hash"  # a past immutable evaluation
    MINIMUM_LOSS = "minimum_loss"        # absolute loss threshold only (no state)


class GateDecisionResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class GatePolicy(BaseModel):
    """One immutable policy definition (embedded in each gate decision).

    The policy fixes the probe (dataset/version/split/tokenizer/window/cap/
    batch/seed), the baseline form, the tolerance and optional extra
    constraints. ``extra="forbid"``; ambiguous baseline combinations are
    rejected by the validator. Baselines:

      * CHECKPOINT     -> baseline_checkpoint_id
      * CURRENT        -> (no extra fields)
      * EVALUATION_RESULT_HASH -> baseline_result_hash (a past M4 evaluation)
      * MINIMUM_LOSS   -> minimum_loss (absolute threshold; no comparison)

    ``max_regression_delta`` may only accompany state baselines; ``minimum_loss``
    may additionally ride along with any state baseline as an absolute
    candidate-loss ceiling.
    """

    name: str = Field("gate", min_length=1, max_length=64, pattern=NAME_PATTERN)
    model_id: str = Field(..., min_length=1, max_length=64)
    description: Optional[str] = None

    # Probe (identical for candidate and baseline)
    dataset_id: str = Field(..., min_length=1, max_length=64)
    dataset_version: Optional[int] = Field(None, ge=1)  # None -> latest version
    split: EvaluationSplit
    tokenizer_id: str = Field(..., min_length=1, max_length=64)
    max_eval_tokens: Optional[int] = Field(None, ge=1)  # None -> whole split
    batch_size: int = Field(1, ge=1)
    max_seq_len: Optional[int] = Field(None, ge=2)      # window; None -> model context
    seed: Optional[int] = Field(None, ge=0, lt=2**32)

    # Baseline definition
    baseline_type: GateBaselineType
    baseline_checkpoint_id: Optional[str] = Field(None, min_length=1, max_length=64)
    baseline_result_hash: Optional[str] = None          # 64-hex M4 result hash

    # Decision constraints
    tolerance: float = Field(1e-4, ge=0.0)              # |delta| <= tol -> unchanged
    max_regression_delta: Optional[float] = Field(None, ge=0.0)  # regressed-but-bounded pass
    minimum_loss: Optional[float] = Field(None, ge=0.0) # absolute candidate-loss ceiling

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _baseline_consistent(self) -> "GatePolicy":
        kind = self.baseline_type
        if kind == GateBaselineType.CHECKPOINT:
            if not self.baseline_checkpoint_id:
                raise ValueError(
                    "baseline_type='checkpoint' requires baseline_checkpoint_id")
            if self.baseline_result_hash:
                raise ValueError(
                    "baseline_type='checkpoint' cannot set baseline_result_hash")
        elif kind == GateBaselineType.CURRENT:
            if self.baseline_checkpoint_id or self.baseline_result_hash:
                raise ValueError(
                    "baseline_type='current' takes no baseline id fields")
        elif kind == GateBaselineType.EVALUATION_RESULT_HASH:
            if not self.baseline_result_hash:
                raise ValueError(
                    "baseline_type='evaluation_result_hash' requires "
                    "baseline_result_hash (64-hex evaluation result hash)")
            if self.baseline_checkpoint_id:
                raise ValueError(
                    "baseline_type='evaluation_result_hash' cannot set "
                    "baseline_checkpoint_id")
        elif kind == GateBaselineType.MINIMUM_LOSS:
            if self.baseline_checkpoint_id or self.baseline_result_hash:
                raise ValueError(
                    "baseline_type='minimum_loss' takes no baseline id fields")
            if self.minimum_loss is None:
                raise ValueError(
                    "baseline_type='minimum_loss' requires minimum_loss")
            if self.max_regression_delta is not None:
                raise ValueError(
                    "max_regression_delta is meaningless without a state "
                    "baseline (baseline_type='minimum_loss')")
        else:  # pragma: no cover - enum guards this
            raise ValueError(f"unknown baseline_type '{kind}'")
        if kind != GateBaselineType.MINIMUM_LOSS and self.max_regression_delta is not None:
            if self.tolerance > self.max_regression_delta:
                raise ValueError(
                    "max_regression_delta must be >= tolerance to be meaningful")
        return self

    def default_seed(self) -> int:
        import hashlib
        import json

        blob = json.dumps(self.model_dump(mode="json", exclude={"seed"}),
                          sort_keys=True)
        return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16)

    def effective_seed(self) -> int:
        return self.default_seed() if self.seed is None else self.seed


class GateRequest(BaseModel):
    """Evaluate one candidate model state against one gate policy.

    ``candidate`` uses the M5 ComparisonState convention (current has no
    checkpoint id; checkpoint requires one). Exactly ONE policy source must
    be given: ``policy`` (inline, M6 semantics, embedded in the decision) or
    ``policy_id`` (an immutable M9 registry definition, resolved by the gate
    engine — the executed policy is still embedded in the decision). An
    inline ``policy.model_id`` must equal ``model_id``; a registry policy is
    checked against ``model_id`` at resolution.
    """

    model_id: str = Field(..., min_length=1, max_length=64)
    policy: Optional[GatePolicy] = None
    policy_id: Optional[str] = Field(None, min_length=1, max_length=64,
                                     pattern=NAME_PATTERN)
    candidate: ComparisonState

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _policy_source_consistent(self) -> "GateRequest":
        if (self.policy is None) == (self.policy_id is None):
            raise ValueError(
                "exactly one policy source is required: inline 'policy' or "
                "registry 'policy_id' (not both, not neither)")
        if self.policy is not None and self.policy.model_id != self.model_id:
            raise ValueError(
                f"policy.model_id ('{self.policy.model_id}') must equal "
                f"model_id ('{self.model_id}')")
        return self


class GateDecision(BaseModel):
    """Immutable record of one gate evaluation
    (gates/<decision_id>/manifest.json).

    Evidence chain: decision -> candidate evaluation (+ baseline evaluation
    and comparison when applicable) -> model state/checkpoint. ``result_hash``
    is deterministic over the semantic inputs/results only — never the
    decision id, timestamps or paths — so identical gate requests against
    unchanged states reproduce it.
    """

    decision_id: str
    model_id: str
    config_hash: str                      # architecture identity of the model
    policy: GatePolicy                    # full policy as evaluated (version resolved)
    policy_id: Optional[str] = None       # M9 registry id when resolved (else None)
    policy_config_hash: Optional[str] = None  # M9 definition config hash (else None)
    candidate: ComparisonSide
    baseline: Optional[ComparisonSide] = None   # None for minimum_loss-only gates
    comparison_id: Optional[str] = None         # None for threshold-only gates

    candidate_loss: float
    baseline_loss: Optional[float] = None
    delta_loss_nats: Optional[float] = None     # candidate - baseline
    verdict: Optional[ComparisonVerdict] = None  # None for threshold-only gates
    comparison_result_hash: Optional[str] = None  # deterministic hash of the evidence

    tolerance: float
    max_regression_delta: Optional[float] = None
    minimum_loss: Optional[float] = None

    decision: GateDecisionResult
    reason: str
    suggested_checkpoint_id: Optional[str] = None  # only when rollback makes sense
    hint: Optional[str] = None

    result_hash: str
    created_at: datetime
    duration_seconds: float = 0.0
    schema_version: int = 1


class TrainingReport(BaseModel):
    """Structured result of one synchronous training run."""

    run_id: str
    model_id: str
    method: TrainingMethod
    dataset_id: str
    dataset_version: int
    tokenizer_id: str
    seed: int
    initial_model_version: Optional[str] = None  # latest checkpoint id at start
    final_model_version: Optional[str] = None    # checkpoint id the weights now match
    optimizer_steps: int = 0
    epochs_run: float = 0.0
    requested_batch_size: int = 0
    actual_batch_size: int = 0
    baseline_validation_loss: Optional[float] = None
    baseline_perplexity: Optional[float] = None
    initial_train_loss: Optional[float] = None
    final_train_loss: Optional[float] = None
    best_validation_loss: Optional[float] = None
    best_perplexity: Optional[float] = None
    accepted: bool = False
    rolled_back_to: Optional[str] = None
    train_loss_history: list[float] = Field(default_factory=list)
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    hardware: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Persisted records (manifest contents)
# --------------------------------------------------------------------------- #


class ModelRecord(BaseModel):
    """Everything the forge knows about one model. Weights live in ``weights.pt``."""

    id: str
    name: str
    description: Optional[str] = None
    metadata: dict[str, str] = Field(default_factory=dict)
    architecture: Architecture
    created_at: datetime
    updated_at: datetime
    builder_version: str
    manifest_schema_version: int = 1

    config: TransformerConfig
    config_hash: str
    parameter_count: int
    weight_bytes: int
    dtype: str

    parent_model_id: Optional[str] = None  # set when a model is derived from another
    initializer: str = "normal"            # normal | preserve
    seed: int
    state_hash: str                        # content hash of the weights (dedup key)
    hardware: dict[str, Any] = Field(default_factory=dict)
    weights_file: str = "weights.pt"
    param_rows: list[dict[str, Any]] = Field(default_factory=list)  # informational

    # Training lineage (empty until the first training run)
    latest_checkpoint: Optional[str] = None   # most recent checkpoint created
    best_checkpoint: Optional[str] = None     # best-known checkpoint (weights reference)
    training_provenance: list[RunProvenance] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "architecture": self.architecture.value,
            "parameter_count": self.parameter_count,
            "precision": self.config.precision.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parent_model_id": self.parent_model_id,
        }


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Workflows (ordered, auditable orchestration over M3–M6; M7)
# --------------------------------------------------------------------------- #


class StageType(str, Enum):
    """Stage kinds M7 can execute — exactly the existing engine capabilities.

    ``recipe`` (M14) is a definition-level stage: it appears ONLY inside
    registered workflow recipes and means "splice this referenced recipe's
    stages here, deterministically qualified". Inline workflow plans never
    contain recipe stages — by the time a plan reaches the WorkflowEngine
    every recipe stage has been expanded away by the RecipeEngine.
    """

    TRAIN = "train"            # M3 TrainingEngine (explicit user request only)
    EVALUATE = "evaluate"      # M4 measurement (M5/M6-style exact reuse)
    COMPARE = "compare"        # M5 A/B comparison (equivalent evidence reused)
    GATE = "gate"              # M6 policy decision (branching source)
    SUITE_RUN = "suite_run"    # M10 suite run (multi-probe M4 batch, M11 stage)
    RECIPE = "recipe"          # recipe reference (M14; registered recipes only)


class WorkflowStatus(str, Enum):
    """Terminal status of one workflow run. Synchronous execution persists the
    manifest only after the run ends, so a mid-flight 'running' state is never
    observable and deliberately not modelled."""

    COMPLETED = "completed"    # plan executed through its last stage
    FAILED = "failed"          # a stage raised (missing/corrupt/invalid input)
    STOPPED = "stopped"        # a gate decision failed and no on_fail branch


class ArtifactKind(str, Enum):
    """What a finished stage produced (evidence kind on the run record)."""

    TRAINING_REPORT = "training_report"
    EVALUATION = "evaluation"
    COMPARISON = "comparison"
    GATE_DECISION = "gate_decision"
    SUITE_RUN = "suite_run"


class StageStateRef(BaseModel):
    """Explicit source of a model state inside a workflow stage.

    ``current`` -> the plan model's live published weights.
    ``checkpoint`` -> either a literal immutable M3 checkpoint id or the FINAL
    checkpoint produced by an earlier ``train`` stage (``from_stage``). The
    workflow never guesses which checkpoint to use.
    ``best`` (M53) -> a DECLARATIVE request for the M52 best-checkpoint
    selection (minimum persisted validation_loss over the authoritative M3
    listing, canonical (step, created_at) tie-break): resolved to a concrete
    checkpoint id at execution/preflight time by the workflow engine's
    single resolver and pinned on ``resolved_checkpoint_id``. The recipe
    definition itself stays declarative and is never rewritten; each
    immutable run record carries the pinned concrete id, so a later
    execution that resolves a different checkpoint has a different
    execution identity while history never changes meaning.
    """

    state_kind: EvalStateKind
    checkpoint_id: Optional[str] = Field(None, min_length=1, max_length=64)
    from_stage: Optional[str] = Field(None, min_length=1, max_length=64)
    # M53: the CONCRETE checkpoint id the 'best' selection resolved to at
    # execution/preflight time. Set ONLY by the workflow engine's single
    # resolver and ONLY when state_kind == 'best' (None on declarative
    # definitions); persisted run records carry the pinned id so history
    # never silently changes meaning.
    resolved_checkpoint_id: Optional[str] = Field(
        None, min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _kind_matches_source(self) -> "StageStateRef":
        if self.state_kind == EvalStateKind.CURRENT:
            if self.checkpoint_id is not None or self.from_stage is not None:
                raise ValueError(
                    "state_kind='current' must not set checkpoint_id or from_stage")
            if self.resolved_checkpoint_id is not None:
                raise ValueError(
                    "resolved_checkpoint_id is the pinned M53 'best' "
                    "resolution and is only valid with state_kind='best'")
        elif self.state_kind == EvalStateKind.BEST:
            if self.checkpoint_id is not None or self.from_stage is not None:
                raise ValueError(
                    "state_kind='best' must not set checkpoint_id or "
                    "from_stage (the M52 selection decides the checkpoint; "
                    "an explicit id is a contradiction, not a hint)")
        else:
            if bool(self.checkpoint_id) == bool(self.from_stage):
                raise ValueError(
                    "state_kind='checkpoint' needs exactly one of checkpoint_id "
                    "or from_stage (earlier train stage)")
            if self.resolved_checkpoint_id is not None:
                raise ValueError(
                    "resolved_checkpoint_id is the pinned M53 'best' "
                    "resolution and is only valid with state_kind='best'")
        return self


class WorkflowEvaluationStage(BaseModel):
    """M4 evaluation stage: the full M4 probe config plus an optional explicit
    pointer at an earlier train stage's final checkpoint. When
    ``checkpoint_from_stage`` is set, ``config.checkpoint_id`` must stay unset
    (no ambiguity), and the resolved checkpoint is evaluated on the config's
    probe with M5/M6-style exact evaluation reuse.

    M56: ``checkpoint_from_best`` is the DECLARATIVE best-state selector —
    "evaluate the M52 best checkpoint" (minimum persisted validation_loss).
    The workflow engine's single resolver (M53) pins the SAME per-plan M52
    selection on ``resolved_checkpoint_id`` (the M53 pinned form); execution
    then hands the M4 path a PURE explicit-checkpoint config — the
    evaluation layer never queries "best" (no dynamic selection, no silent
    fallback to current). XOR with the explicit id and with
    ``checkpoint_from_stage`` (a contradiction is rejected, never silently
    preferred); a pre-pinned id is respected (idempotent resolution) and is
    only valid with ``checkpoint_from_best=True``. Direct M4 evaluation
    requests have no workflow context and no such field — they name the
    checkpoint explicitly (GET /checkpoints/best answers it)."""

    config: EvaluationConfig
    checkpoint_from_stage: Optional[str] = Field(None, min_length=1,
                                                 max_length=64)
    # M56: declarative best-evaluation selector (workflow-only); the pin is
    # set by the workflow engine's single resolver ONLY (the M52 selection's
    # concrete id) and is valid only with checkpoint_from_best=True
    checkpoint_from_best: bool = False
    resolved_checkpoint_id: Optional[str] = Field(
        None, min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _not_ambiguous(self) -> "WorkflowEvaluationStage":
        if self.checkpoint_from_stage and self.config.checkpoint_id:
            raise ValueError(
                "checkpoint_from_stage conflicts with config.checkpoint_id: "
                "set exactly one")
        if self.checkpoint_from_best and self.config.checkpoint_id is not None:
            raise ValueError(
                "checkpoint_from_best conflicts with config.checkpoint_id: "
                "the M52 selection decides the checkpoint OR it is named "
                "explicitly, never both")
        if self.checkpoint_from_best and self.checkpoint_from_stage:
            raise ValueError(
                "checkpoint_from_best conflicts with checkpoint_from_stage: "
                "set exactly one state selector")
        if self.resolved_checkpoint_id is not None \
                and not self.checkpoint_from_best:
            raise ValueError(
                "resolved_checkpoint_id is the workflow resolver's pinned "
                "M52 selection and is only valid with "
                "checkpoint_from_best=True")
        return self


class WorkflowComparisonStage(BaseModel):
    """M5 comparison stage: two explicit state sources under ONE shared probe.

    Probe fields mirror ComparisonRequest exactly (same defaults) so the stage
    is seed/identity-compatible with the M5 engine; only the state fields are
    workflow references instead of literal ComparisonState objects.
    """

    state_a: StageStateRef
    state_b: StageStateRef
    dataset_id: str = Field(..., min_length=1, max_length=64)
    dataset_version: Optional[int] = Field(None, ge=1)  # None -> latest
    split: EvaluationSplit
    tokenizer_id: str = Field(..., min_length=1, max_length=64)
    max_eval_tokens: Optional[int] = Field(None, ge=1)
    batch_size: int = Field(1, ge=1)
    max_seq_len: Optional[int] = Field(None, ge=2)      # window; None -> context
    seed: Optional[int] = Field(None, ge=0, lt=2**32)
    tolerance: float = Field(1e-4, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class WorkflowGateStage(BaseModel):
    """M6 gate stage: one gate policy source plus the candidate state source.

    Exactly ONE policy source is allowed: a full inline GatePolicy (M6) or a
    stable M9 registry ``policy_id`` (resolved by the gate engine at run
    start; unknown ids fail the run cleanly before anything executes). The
    baseline stays inside the policy (checkpoint/current/evaluation-hash/
    minimum-loss, M6 rules). Branching happens on the actual GateDecision."""

    policy: Optional[GatePolicy] = None
    policy_id: Optional[str] = Field(None, min_length=1, max_length=64,
                                     pattern=NAME_PATTERN)
    candidate: StageStateRef

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _policy_source_consistent(self) -> "WorkflowGateStage":
        if (self.policy is None) == (self.policy_id is None):
            raise ValueError(
                "a gate stage needs exactly one policy source: inline "
                "'policy' or registry 'policy_id' (not both, not neither)")
        return self


class WorkflowSuiteRunStage(BaseModel):
    """M10 suite-run stage inside a workflow (M11): ONE named probe suite
    against ONE explicitly resolved model state.

    ``suite_id`` names an immutable M9 ProbeSuite (resolved at run start
    through the registry — the workflow never guesses or auto-selects a
    suite). ``state`` uses the exact M7 ``StageStateRef`` representation
    (current weights, a literal immutable checkpoint id, the FINAL
    checkpoint of an earlier train stage via ``from_stage``, or the M53
    declarative ``best`` reference resolved to a concrete checkpoint by
    the workflow engine's single resolver at execution/preflight time)
    so the run record always shows which state was evaluated. The stage
    executes the existing M10 ``SuiteRunEngine`` — it is an orchestration adapter only,
    never a second suite-run implementation and never an aggregate gate.
    """

    suite_id: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    state: StageStateRef

    model_config = ConfigDict(extra="forbid", validate_assignment=True)



class WorkflowRecipeCallStage(BaseModel):
    """A recipe-reference stage inside a registered recipe (M14).

    Names ONE already-registered immutable recipe (``recipe_id``) whose
    ordered stage list is deterministically spliced into the referencing
    recipe at this position when the composition is expanded. The referenced
    recipe must exist at REGISTRATION time (definition composing definition);
    expansion is static and nested references expand recursively. The stage
    itself executes nothing — it disappears during expansion, so branches or
    ``from_stage`` references may never target it (the shared validator
    rejects such targets on the expanded list).
    """

    recipe_id: str = Field(..., min_length=1, max_length=64,
                           pattern=NAME_PATTERN)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkflowStage(BaseModel):
    """One ordered stage. Exactly one payload matching ``type`` is allowed;
    ``on_pass``/``on_fail`` are gate-only forward branches (an absent
    ``on_fail`` on a failed gate means the workflow STOPS).

    ``type=recipe`` (M14) is allowed only inside registered workflow recipes
    (registration validates with ``allow_recipe_stages=True``); inline
    WorkflowPlans reject it — the WorkflowEngine never sees recipe stages.
    """

    stage_id: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    type: StageType
    training: Optional[TrainingConfig] = None            # type=train
    evaluation: Optional[WorkflowEvaluationStage] = None  # type=evaluate
    comparison: Optional[WorkflowComparisonStage] = None  # type=compare
    gate: Optional[WorkflowGateStage] = None              # type=gate
    suite_run: Optional[WorkflowSuiteRunStage] = None     # type=suite_run
    recipe: Optional[WorkflowRecipeCallStage] = None      # type=recipe (M14)
    on_pass: Optional[str] = Field(None, min_length=1, max_length=64)
    on_fail: Optional[str] = Field(None, min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _payload_matches_type(self) -> "WorkflowStage":
        payloads = [p for p in (self.training, self.evaluation,
                                self.comparison, self.gate, self.suite_run,
                                self.recipe)
                    if p is not None]
        if len(payloads) != 1:
            raise ValueError(
                "a stage must set exactly one payload (training/evaluation/"
                "comparison/gate/suite_run/recipe) for its type")
        (payload,) = payloads
        expected = {StageType.TRAIN: TrainingConfig,
                    StageType.EVALUATE: WorkflowEvaluationStage,
                    StageType.COMPARE: WorkflowComparisonStage,
                    StageType.GATE: WorkflowGateStage,
                    StageType.SUITE_RUN: WorkflowSuiteRunStage,
                    StageType.RECIPE: WorkflowRecipeCallStage}[self.type]
        if not isinstance(payload, expected):
            raise ValueError(
                f"stage type '{self.type.value}' requires the matching payload "
                f"type ({expected.__name__})")
        if self.on_pass is not None or self.on_fail is not None:
            if self.type != StageType.GATE:
                raise ValueError(
                    "on_pass/on_fail are gate-only transitions (branching is "
                    "driven by M6 GateDecisions)")
        return self


def validate_plan_stages(stages: list["WorkflowStage"], *,
                         model_id: Optional[str] = None,
                         allow_recipe_stages: bool = False) -> None:
    """Shared structural rules for M7 stage lists (plans AND recipes).

    One code path keeps recipe validation from ever drifting from inline
    plan validation (M12): unique stage ids; every embedded config must
    target ``model_id`` when one is given; ``from_stage``/``checkpoint_
    from_stage`` references must name an EARLIER train stage; gate
    ``on_pass``/``on_fail`` branches must name a LATER existing stage.
    ``WorkflowPlan`` always passes the plan's model (embedded configs must
    agree with it); recipe registration passes ``model_id=None`` because a
    recipe is a model-agnostic definition that binds ONE model per run —
    the model-agreement checks then run at bind time when the recipe is
    re-validated through a real ``WorkflowPlan``. Error messages are
    identical for both paths.

    ``allow_recipe_stages`` (M14): recipe-reference stages are legal ONLY
    inside registered workflow recipe definitions (which are expanded before
    any plan is built); inline plans keep the default False so the
    WorkflowEngine never receives a recipe stage. The shared validator is
    also re-run over every EXPANDED stage list with the default False — a
    leftover recipe stage after expansion is therefore a bug caught here.
    """
    if not allow_recipe_stages:
        for s in stages:
            if s.type == StageType.RECIPE:
                raise ValueError(
                    f"stage '{s.stage_id}' type 'recipe' is only allowed "
                    f"inside registered workflow recipes (it is expanded "
                    f"before a plan is built)")
    ids = [s.stage_id for s in stages]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate stage ids: {ids}")
    index = {sid: i for i, sid in enumerate(ids)}
    for i, s in enumerate(stages):
        if model_id is not None:
            # every embedded config must target the plan's model
            if s.training is not None and s.training.model_id != model_id:
                raise ValueError(
                    f"stage '{s.stage_id}' training config targets model "
                    f"'{s.training.model_id}', plan targets '{model_id}'")
            if s.evaluation is not None and \
                    s.evaluation.config.model_id != model_id:
                raise ValueError(
                    f"stage '{s.stage_id}' evaluation config targets a "
                    f"different model than the plan")
            if s.gate is not None and s.gate.policy is not None \
                    and s.gate.policy.model_id != model_id:
                raise ValueError(
                    f"stage '{s.stage_id}' gate policy targets model "
                    f"'{s.gate.policy.model_id}', plan targets '{model_id}'")
        # from_stage refs: earlier train stages only (checkpoint producers)
        refs: list[tuple[str, str]] = []
        if s.evaluation is not None and s.evaluation.checkpoint_from_stage:
            refs.append((s.evaluation.checkpoint_from_stage, "evaluation"))
        for ref_name, src in (("state_a", s.comparison.state_a if s.comparison
                               else None),
                              ("state_b", s.comparison.state_b if s.comparison
                               else None),
                              ("candidate", s.gate.candidate if s.gate
                               else None),
                              ("suite state", s.suite_run.state
                               if s.suite_run else None)):
            if src is not None and src.from_stage:
                refs.append((src.from_stage, f"{s.stage_id}.{ref_name}"))
        for target, where in refs:
            if target not in index:
                raise ValueError(
                    f"{where} references unknown stage '{target}'")
            if index[target] >= i:
                raise ValueError(
                    f"{where} must reference an EARLIER stage; "
                    f"'{target}' is not before '{s.stage_id}'")
            if stages[index[target]].type != StageType.TRAIN:
                raise ValueError(
                    f"{where} references stage '{target}' which is not a "
                    f"train stage (only train stages produce checkpoints)")
        # transitions: forward-only, existing, gate-only (already enforced)
        for field in ("on_pass", "on_fail"):
            target = getattr(s, field)
            if target is None:
                continue
            if target not in index:
                raise ValueError(
                    f"stage '{s.stage_id}' {field} references unknown "
                    f"stage '{target}'")
            if index[target] <= i:
                raise ValueError(
                    f"stage '{s.stage_id}' {field} must branch FORWARD to "
                    f"a later stage (no cycles); '{target}' is not later")


class WorkflowPlan(BaseModel):
    """One submitted workflow: what SHOULD happen (never mutated afterwards).

    Plans are inline: the executed plan is embedded verbatim in every
    WorkflowRecord, so runs stay auditable without a plan registry. All stages
    operate on ONE model (``model_id``); every embedded config must agree.
    References point strictly backward (earlier train stages) and transitions
    strictly forward, so each stage executes at most once, in plan order.
    Structural validation is shared with M12 workflow recipes
    (``validate_plan_stages``); plans additionally demand that every embedded
    config targets the plan's model, while recipes bind the model per run.
    """

    name: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    model_id: str = Field(..., min_length=1, max_length=64)
    description: Optional[str] = None
    stages: list[WorkflowStage] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _plan_consistent(self) -> "WorkflowPlan":
        validate_plan_stages(self.stages, model_id=self.model_id)
        return self

    def plan_hash(self) -> str:
        import hashlib
        import json

        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class WorkflowArtifact(BaseModel):
    """Evidence one executed stage produced (referenced artifact + hashes).

    ``artifact_id`` is the underlying immutable record's id (run/eval/
    comparison/decision id). ``result_hash``/``state_hash`` are the recorded
    deterministic hashes, so the run chains to its evidence exactly like M6."""

    kind: ArtifactKind
    artifact_id: str
    result_hash: Optional[str] = None       # evaluation/comparison/gate hash
    state_hash: Optional[str] = None        # canonical weights content hash
    checkpoint_id: Optional[str] = None     # final checkpoint of a train stage
    loss_nats: Optional[float] = None
    delta_loss_nats: Optional[float] = None
    verdict: Optional[ComparisonVerdict] = None
    gate_decision: Optional[GateDecisionResult] = None
    accepted: Optional[bool] = None         # training outcome (M3 semantics)
    optimizer_steps: Optional[int] = None
    epochs_run: Optional[float] = None
    best_validation_loss: Optional[float] = None
    final_train_loss: Optional[float] = None
    final_validation_loss: Optional[float] = None


class WorkflowStageResult(BaseModel):
    """What actually happened for one stage (in plan order)."""

    stage_id: str
    type: StageType
    executed: bool = False
    skipped: bool = False
    artifact: Optional[WorkflowArtifact] = None
    error: Optional[str] = None             # recorded, never fabricated as an id


class WorkflowTransition(BaseModel):
    """One control-flow step: which stage produced it and where execution went."""

    stage_id: Optional[str] = None          # None only before the first stage
    decision: Optional[str] = None          # passed | failed | next
    to_stage: Optional[str] = None          # None = terminal (completed/stopped)


class WorkflowRecord(BaseModel):
    """Immutable record of ONE workflow execution (workflows/workflow-<id>/).

    Embeds the executed plan verbatim and chains every stage to its underlying
    immutable artifact (training report / evaluation / comparison / gate
    decision). ``result_hash`` is deterministic over the semantic execution
    only — plan, status, per-stage evidence hashes/results and transitions —
    never the workflow id, recipe provenance, timestamps or paths, so
    identical workflows over identical immutable inputs reproduce it.

    ``recipe_id``/``recipe_hash`` (M12) are additive provenance: they are set
    only when the run was produced by executing a registered workflow recipe
    and stay null for inline plans. The recipe is never copied into the run —
    the record stays reference-oriented (recipe id + config hash + the bound
    plan actually executed).

    ``composition`` (M14) is additive provenance for COMPOSITE recipe runs:
    the deterministic depth-first trace of every referenced recipe that was
    expanded into this run's plan (recipe_id + config_hash per call
    occurrence, in expansion order, the top-level recipe itself excluded —
    it is already ``recipe_id``/``recipe_hash``). Null for inline runs and
    for runs of recipes that reference nothing. All three provenance fields
    are excluded from ``result_hash`` (semantic execution only).
    """

    workflow_id: str
    model_id: str
    config_hash: str                       # architecture identity of the model
    name: str
    plan_hash: str                         # identity of the embedded plan
    plan: WorkflowPlan
    status: WorkflowStatus
    terminal_reason: Optional[str] = None
    failed_stage_id: Optional[str] = None
    stages: list[WorkflowStageResult] = Field(default_factory=list)
    transitions: list[WorkflowTransition] = Field(default_factory=list)
    suggested_checkpoint_id: Optional[str] = None  # terminal failed gate only
    hint: Optional[str] = None
    recipe_id: Optional[str] = None        # M12: recipe that produced this run
    recipe_hash: Optional[str] = None      # M12: that recipe's config hash
    composition: Optional[list["WorkflowRecipeRef"]] = None  # M14: refs expanded
    result_hash: str
    created_at: datetime
    duration_seconds: float = 0.0
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Dashboard (read-only aggregation over immutable histories; M8)
# --------------------------------------------------------------------------- #


class DashboardDiagnostic(BaseModel):
    """Deterministic note about an artifact the dashboard could not use.

    Never fabricated: one diagnostic per unreadable manifest (family + id
    taken from the directory name) or per internal reference to an artifact
    that is missing from the scanned tree. Valid artifacts stay visible.
    """

    family: str                        # artifact family the issue belongs to
    artifact_id: Optional[str] = None  # directory-derived id when applicable
    issue: str                         # deterministic short description


class DashboardEvaluationSeries(BaseModel):
    """Evaluations of ONE state under ONE exact probe, oldest first.

    ``identity`` mirrors the M4/M5/M6 exact-reuse fields (state kind/id/
    canonical hash + dataset/version/split/tokenizer/window/cap/batch/seed);
    ``key`` is a sha256 over its canonical JSON so grouping never depends on
    filesystem order.
    """

    identity: dict[str, Any]
    key: str
    count: int
    records: list[EvaluationRecord]


class DashboardComparisonSeries(BaseModel):
    """Comparisons of ONE ordered state pair under ONE probe + tolerance."""

    identity: dict[str, Any]
    key: str
    count: int
    records: list[ComparisonRecord]


class DashboardGateSeries(BaseModel):
    """Gate decisions grouped by recorded policy configuration identity.

    ``policy_key`` = sha256 over the canonical JSON of the decision's own
    persisted ``policy`` (dashboard-derived, never persisted). Statistics are
    descriptive per-policy pass/fail counts — not a model-quality score.
    """

    policy_key: str
    policy: dict[str, Any]
    statistics: dict[str, int]
    records: list[GateDecision]


class DashboardWorkflowSummary(BaseModel):
    """All workflow runs, oldest first, plus descriptive status counts."""

    counts: dict[str, int]
    records: list[WorkflowRecord]


class DashboardSuiteRunSummary(BaseModel):
    """This model's M10 suite runs, oldest first, plus status counts (M13).

    Counts are execution-bookkeeping only (``completed``/``failed`` exactly as
    M10 persists them — never a score, pass percentage or quality judgment).
    Each record keeps its full per-probe detail (probe → outcome
    created|reused|failed → evaluation id) untouched: ``reused`` describes
    evaluation-artifact reuse, never probe success.
    """

    counts: dict[str, int]
    records: list["SuiteRunRecord"] = Field(default_factory=list)


class DashboardGraphNode(BaseModel):
    """One real artifact node (family + id, both resolvable)."""

    family: str
    artifact_id: str


class DashboardGraphEdge(BaseModel):
    """One recorded reference between two existing artifacts."""

    source: DashboardGraphNode
    target: DashboardGraphNode
    role: str


class DashboardGraph(BaseModel):
    """Deterministic reference graph (nodes/edges sorted canonically)."""

    nodes: list[DashboardGraphNode] = Field(default_factory=list)
    edges: list[DashboardGraphEdge] = Field(default_factory=list)


class DashboardSampleQualityBySample(BaseModel):
    """One sample's measurement history summary (M17 dashboard section).

    References only: sample id + the sample's own immutable result_hash
    (as recorded by its evaluations) + how many immutable M16 measurements
    exist + the newest measurement's evaluation id and timestamp. No metric
    values, no comparisons, no quality interpretation of any kind.
    """

    sample_id: str
    sample_result_hash: str          # recorded on the latest evaluation
    evaluation_count: int            # immutable M16 records for this sample
    latest_evaluation_id: str        # max by (created_at, evaluation_id)
    latest_evaluated_at: datetime


class DashboardSampleQualityLatest(BaseModel):
    """One reference in the newest-first measurement history list (M17)."""

    evaluation_id: str
    sample_id: str
    created_at: datetime


class DashboardSampleQuality(BaseModel):
    """M17 dashboard section: read-only sample-evaluation history structure.

    Derived LIVE from the immutable ``sample-evaluations/<model_id>/``
    manifests of the requested model only (never persisted, never cached).
    ``total_count`` is the number of valid records; ``by_sample`` groups
    them per sample ordered by ``sample_id`` with per-sample latest
    references; ``latest`` lists every record newest-first, ordered by
    (created_at, evaluation_id) DESCENDING, as references only.

    The section deliberately exposes ONLY the existence/history of M16
    measurements: no loss/perplexity values, no averages/minimums/maximums,
    no rankings, no quality verdicts — M16 records are referenced by id,
    their metric payload is not copied here.
    """

    total_count: int
    by_sample: list[DashboardSampleQualityBySample] = Field(
        default_factory=list)
    latest: list[DashboardSampleQualityLatest] = Field(default_factory=list)


class ModelDashboard(BaseModel):
    """Read-only per-model view over the immutable history (M8).

    Everything reported comes from persisted manifests; no new measurement,
    score or ranking is invented. ``result_hash`` covers the canonical JSON
    of the whole dashboard except itself, so identical storage state always
    reproduces it (no request time, random ids or traversal order involved).

    ``sample_quality`` (M17) is a read-only observability section derived
    live from the model's immutable ``sample-evaluations/`` records —
    history structure only (counts + references), never metric values or
    quality judgments.
    """

    model_id: str
    config_hash: str
    summary: dict[str, Any]          # persisted model-manifest facts
    checkpoints: list[CheckpointRecord] = Field(default_factory=list)
    training_runs: list[RunProvenance] = Field(default_factory=list)
    evaluations: list[DashboardEvaluationSeries] = Field(default_factory=list)
    comparisons: list[DashboardComparisonSeries] = Field(default_factory=list)
    gate_decisions: list[DashboardGateSeries] = Field(default_factory=list)
    workflows: DashboardWorkflowSummary = Field(
        default_factory=lambda: DashboardWorkflowSummary(counts={},
                                                         records=[]))
    suite_runs: DashboardSuiteRunSummary = Field(
        default_factory=lambda: DashboardSuiteRunSummary(counts={},
                                                         records=[]))
    artifact_graph: DashboardGraph = Field(default_factory=DashboardGraph)
    diagnostics: list[DashboardDiagnostic] = Field(default_factory=list)
    sample_quality: DashboardSampleQuality = Field(
        default_factory=lambda: DashboardSampleQuality(total_count=0))
    result_hash: str = ""
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Stable policy registry + named probe suites (M9)
# --------------------------------------------------------------------------- #


class PolicyDefinition(BaseModel):
    """Immutable named policy (policies/<policy_id>/manifest.json; M9).

    ``policy`` is the exact M6 GatePolicy the GateEngine evaluates — the
    registry only gives it a stable user-chosen id plus a deterministic
    ``config_hash`` (sha256 over the canonical JSON of ``policy`` alone: no
    ids, timestamps, description or paths), so identical configurations
    always reproduce the same hash. Registration is immutable: the same
    policy_id with any different content is rejected — never overwritten.
    """

    policy_id: str
    description: Optional[str] = None
    policy: GatePolicy
    config_hash: str
    created_at: datetime
    schema_version: int = 1


class PolicyCreateRequest(BaseModel):
    """Register one immutable policy definition (stable id + full M6 semantics)."""

    policy_id: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    description: Optional[str] = Field(None, max_length=512)
    policy: GatePolicy

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SuiteProbe(BaseModel):
    """One probe of a suite: the exact M4 probe-identity fields.

    Mirrors the probe fields ``EvaluationConfig``/``GatePolicy`` already carry
    (dataset/version/split/tokenizer/window/token-cap/batch/seed) with the
    same constraints and ``extra="forbid"`` — no second probe identity is
    invented. Model and state are supplied at use time through
    ``to_evaluation_config``, so an evaluation produced from a suite probe
    carries exactly the identity M4/M5/M6 reuse on.
    """

    dataset_id: str = Field(..., min_length=1, max_length=64)
    dataset_version: Optional[int] = Field(None, ge=1)  # None -> latest version
    split: EvaluationSplit
    tokenizer_id: str = Field(..., min_length=1, max_length=64)
    max_eval_tokens: Optional[int] = Field(None, ge=1)  # None -> whole split
    batch_size: int = Field(1, ge=1)
    max_seq_len: Optional[int] = Field(None, ge=2)      # window; None -> model context
    seed: Optional[int] = Field(None, ge=0, lt=2**32)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_evaluation_config(self, model_id: str,
                             checkpoint_id: Optional[str] = None) -> "EvaluationConfig":
        """Resolve this probe into an M4 ``EvaluationConfig`` (adds model/state).

        The probe fields are passed through verbatim, so the resulting
        evaluation is indistinguishable from one configured directly with the
        same M4 fields.
        """
        return EvaluationConfig(model_id=model_id, checkpoint_id=checkpoint_id,
                                **self.model_dump())


class ProbeSuiteCreateRequest(BaseModel):
    """Register one immutable probe suite (stable id + probe set)."""

    suite_id: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    description: Optional[str] = Field(None, max_length=512)
    probes: list[SuiteProbe] = Field(..., min_length=1, max_length=256)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _no_duplicate_probes(self) -> "ProbeSuiteCreateRequest":
        import json as _json

        keys = [_json.dumps(p.model_dump(mode="json"), sort_keys=True,
                            default=str) for p in self.probes]
        if len(keys) != len(set(keys)):
            raise ValueError("a probe suite may not contain the same probe "
                             "twice (probes are a set of exact M4 probes)")
        return self


class ProbeSuite(BaseModel):
    """Immutable named probe suite (probe-suites/<suite_id>/manifest.json; M9).

    A suite is a named SET of M4 evaluation probes (order is meaningless for
    evaluation, so probes are stored in canonical order). ``probes_hash`` is
    sha256 over the canonical sorted probe JSON — no ids, timestamps,
    description or paths. A suite never computes a score; each probe stays an
    independent exact M4 probe.
    """

    suite_id: str
    description: Optional[str] = None
    probes: list[SuiteProbe] = Field(default_factory=list)
    probes_hash: str
    created_at: datetime
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Explicit multi-probe evaluation batches over named probe suites (M10)
# --------------------------------------------------------------------------- #


class SuiteRunStatus(str, Enum):
    """Status of one suite run (execution bookkeeping only)."""

    COMPLETED = "completed"
    FAILED = "failed"


class SuiteRunRequest(BaseModel):
    """Execute ONE named probe suite against ONE explicit model state.

    The state reuses the M4/M5 ``ComparisonState`` convention (current
    weights, or one immutable checkpoint) — no second state representation
    exists. The suite resolves through the M9 registry; every probe of the
    suite executes as an independent exact M4 evaluation (or is reused when
    an exact evaluation already exists). Nothing here aggregates, scores or
    gates: a suite is a set of probes, never a benchmark.
    """

    model_id: str = Field(..., min_length=1, max_length=64)
    suite_id: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    state: ComparisonState

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SuiteRunProbeResult(BaseModel):
    """Per-probe outcome of a suite run (in canonical suite order).

    ``probe`` is the probe AS EXECUTED (dataset version resolved when the
    probe ran); ``outcome`` is one of ``created | reused | failed`` so a
    failed probe is never confused with reused or newly-created evidence;
    ``evaluation_id`` is the underlying immutable M4 evaluation when one
    exists (created or reused), else None. No evaluation result is ever
    fabricated for a failed probe.
    """

    probe: SuiteProbe
    evaluation_id: Optional[str] = None
    outcome: str                                    # created | reused | failed
    error: Optional[str] = None                     # deterministic failure text


class SuiteRunRecord(BaseModel):
    """Immutable record of ONE suite execution
    (suite-runs/<suite_run_id>/manifest.json; M10).

    References only — evaluations, datasets, tokenizers and weights are never
    copied. ``result_hash`` covers the semantic execution (model + config
    identity, state identity incl. the canonical state hash, suite id +
    probes hash, canonical probe-to-evaluation mapping, status and failure
    text) — never the suite_run_id, timestamps, durations or paths — so two
    logically identical runs over identical immutable evidence reproduce it.
    """

    suite_run_id: str
    model_id: str
    config_hash: str                      # architecture identity of the model
    state: ComparisonState                # evaluated state (kind + checkpoint id)
    state_hash: str                       # canonical fp32 content hash of that state
    suite_id: str
    suite_probes_hash: str                # identity of the suite content at run time

    status: SuiteRunStatus
    probe_count: int                      # execution bookkeeping only (never a score)
    completed_count: int                  # probes with an evaluation (created/reused)
    reused_count: int                     # probes satisfied by pre-existing evidence
    failed_count: int                     # probes that could not execute

    results: list[SuiteRunProbeResult] = Field(default_factory=list)
    result_hash: str
    created_at: datetime
    duration_seconds: float = 0.0
    schema_version: int = 1


class SuiteRunSummary(BaseModel):
    """Read-only bookkeeping summary of ONE model's suite-run history for
    ONE named suite (M22).

    A pure derived view over the immutable SuiteRunRecord manifests: it
    contains only identity/counting bookkeeping (which run ids exist, in
    the deterministic M21 order, and how many) plus the earliest/latest
    recorded run timestamps — never scores, averages, trends or judgments.
    ``earliest_created_at``/``latest_created_at`` are None exactly when the
    model has no runs of the suite (total_count 0). Never persisted, never
    written: recomputed deterministically from the M21 filter on request.
    """

    model_id: str
    suite_id: str
    total_count: int                      # number of suite-run records (>= 0)
    run_ids: list[str] = Field(default_factory=list)
                                           # ASCENDING (created_at, id)
    earliest_created_at: Optional[datetime] = None   # None when no runs
    latest_created_at: Optional[datetime] = None     # None when no runs


# --------------------------------------------------------------------------- #
# Named workflow recipes (immutable reusable M7 plans; M12)
# --------------------------------------------------------------------------- #


class WorkflowRecipeRef(BaseModel):
    """One immutable recipe reference (M14 composition bookkeeping).

    Identifies a referenced recipe by its stable id AND its deterministic
    config hash — never by copying its definition. Used both in stored
    recipe manifests (the direct references a composite is built from) and
    in workflow run records (the deterministic expansion trace).
    """

    recipe_id: str = Field(..., min_length=1, max_length=64,
                           pattern=NAME_PATTERN)
    config_hash: str = Field(..., min_length=64, max_length=64)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkflowRecipeCreateRequest(BaseModel):
    """Register ONE immutable workflow recipe (stable id + M7 stage list).

    The body reuses the exact M7 ``WorkflowStage`` discriminated union — no
    recipe-specific stage language exists, and a ``recipe`` stage (M14)
    simply references another registered recipe. Structural validation runs
    the SAME shared rules as inline ``WorkflowPlan`` validation
    (``validate_plan_stages``) so recipe validation can never drift; the
    model-equality half of plan validation is deferred to run time because a
    recipe is a definition that binds ONE explicitly supplied model per run.
    Recipe stages are accepted here (allow_recipe_stages=True) — they are
    expanded away before any WorkflowPlan is built; inline plans reject them.
    """

    recipe_id: str = Field(..., min_length=1, max_length=64, pattern=NAME_PATTERN)
    description: Optional[str] = Field(None, max_length=512)
    stages: list[WorkflowStage] = Field(..., min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _recipe_consistent(self) -> "WorkflowRecipeCreateRequest":
        validate_plan_stages(self.stages, model_id=None,
                             allow_recipe_stages=True)
        return self


class WorkflowRecipe(BaseModel):
    """Immutable named workflow recipe
    (workflow-recipes/<recipe_id>/manifest.json; M12).

    A recipe is INERT DATA: an ordered M7 stage list (the exact
    ``WorkflowStage`` union, validated by the same rules as inline plans)
    with a user-chosen id and a deterministic ``config_hash``. Registration
    is immutable: the same recipe_id with any different stage content is
    rejected, never overwritten. Execution only ever happens through an
    explicit run request binding one model; the recipe itself never selects,
    schedules or repeats anything.

    ``composition`` (M14): for a COMPOSITE recipe — one whose declared
    stages contain ``recipe`` references — this lists those DIRECT
    references (referenced recipe_id + config_hash, in declared stage
    order); nested references are reachable transitively through the
    referenced recipes' own manifests, so nothing is ever duplicated. Null
    for recipes that reference nothing (all M12 definitions). For composite
    recipes ``config_hash`` is derived from the DECLARED ordered stages PLUS
    the referenced recipes' ids and config hashes, so a composite can never
    drift from the definitions it was built from; plain recipes keep the
    exact M12 hash over the ordered stages alone.
    """

    recipe_id: str
    description: Optional[str] = None
    stages: list[WorkflowStage] = Field(default_factory=list)
    composition: Optional[list[WorkflowRecipeRef]] = None  # M14: direct refs
    config_hash: str
    created_at: datetime
    schema_version: int = 1

    @model_validator(mode="after")
    def _recipe_consistent(self) -> "WorkflowRecipe":
        validate_plan_stages(self.stages, model_id=None,
                             allow_recipe_stages=True)
        return self


class WorkflowRecipeRunRequest(BaseModel):
    """Execute ONE registered workflow recipe against ONE explicit model.

    The model is the ONLY runtime binding — a recipe never auto-selects a
    model, checkpoint, suite or gate. ``model_id`` must name an existing
    model; embedded stage configs pinned to a different model are rejected
    when the recipe is re-validated as a WorkflowPlan (422).
    """

    model_id: str = Field(..., min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkflowRecipeResolution(BaseModel):
    """Read-only RESOLUTION of ONE registered recipe against ONE explicit
    model (M51) — the preflight view of the recipe-run surface.

    Computed deterministically on request by the SAME resolution path
    RecipeEngine.run() uses (recipe lookup -> model validation -> M14
    deterministic expansion -> WorkflowPlan construction with the FULL
    M7 validation incl. embedded-config model agreement). NEVER
    persisted, never written: it is a view, not a record — exactly one
    execution system remains. ``plan`` is the exact expanded,
    model-bound WorkflowPlan the WorkflowEngine WOULD execute (identical
    to the plan embedded in the resulting WorkflowRecord); ``composition``
    is the additive M14 expansion trace (None for plain recipes). The
    ``plan_hash`` of ``plan`` therefore predicts the executed run's
    ``plan_hash``.
    """

    recipe_id: str
    recipe_hash: str                      # the recipe's config_hash
    model_id: str
    plan: WorkflowPlan
    composition: Optional[list["WorkflowRecipeRef"]] = None

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# M15 — checkpoint sampling (generation) schemas
# --------------------------------------------------------------------------- #


class SampleStrategy(str, Enum):
    """Decoding strategies M15 supports — exactly two, both explicit.

    ``greedy`` is deterministic argmax with no RNG; ``temperature`` draws
    from the temperature-scaled distribution using ONE deterministic RNG
    stream seeded by the request's explicit integer seed. A request never
    mixes them and temperature is never reinterpreted as greedy.
    """

    GREEDY = "greedy"
    TEMPERATURE = "temperature"


class SampleGenerateRequest(BaseModel):
    """Generate text from ONE explicitly selected immutable checkpoint.

    Every input is explicit — model_id, checkpoint_id, tokenizer_id, prompt,
    strategy and generation parameters — nothing is ever auto-selected
    (never current/latest/best weights, never a guessed tokenizer). The
    checkpoint is verified (content hash vs manifest) before any decode and
    the tokenizer/model vocabularies are checked for compatibility
    (platform convention: model vocab_size >= tokenizer actual vocab, the
    exact rule M3/M4 use). Greedy forbids temperature/seed; temperature
    requires BOTH an explicit temperature in (0, 1] and an explicit seed.
    max_new_tokens is capped at 512 by the schema; the effective cap is
    model context_length minus the encoded prompt length (checked at
    preflight — never silently truncated).
    """

    model_id: str = Field(..., min_length=1, max_length=64)
    checkpoint_id: str = Field(..., min_length=1, max_length=64)
    tokenizer_id: str = Field(..., min_length=1, max_length=64)
    prompt: str = Field(..., min_length=1, max_length=4096)
    strategy: SampleStrategy
    max_new_tokens: int = Field(..., ge=1, le=512)
    temperature: Optional[float] = Field(None, gt=0.0, le=1.0)
    seed: Optional[int] = Field(None, ge=0, lt=2 ** 32)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _strategy_consistent(self) -> "SampleGenerateRequest":
        if self.strategy == SampleStrategy.GREEDY:
            if self.temperature is not None:
                raise ValueError(
                    "strategy 'greedy' takes no temperature — remove it or "
                    "choose strategy 'temperature' explicitly")
            if self.seed is not None:
                raise ValueError(
                    "strategy 'greedy' takes no seed — greedy is "
                    "deterministic argmax without RNG")
        else:
            if self.temperature is None:
                raise ValueError(
                    "strategy 'temperature' requires an explicit temperature "
                    "in (0, 1]")
            if self.seed is None:
                raise ValueError(
                    "strategy 'temperature' requires an explicit integer "
                    "seed (0 <= seed < 2**32)")
        return self


class SampleRecord(BaseModel):
    """One immutable text-generation sample
    (samples/<model_id>/sample-<id>/manifest.json; M15).

    Full audit + reproduction record of ONE generation request: the exact
    request (model/checkpoint/tokenizer ids, prompt verbatim, strategy,
    parameters), the verified checkpoint identity (weights content hash), the
    tokenizer's content hash, the encoded prompt token ids, the generated
    token ids, and the decoded output text. Nothing else is ever written —
    no weight copies, no blobs, no caches. ``result_hash`` is deterministic
    over the semantic payload only (ids/timestamps/paths/duration excluded),
    so identical requests over identical immutable inputs reproduce it.
    """

    sample_id: str
    model_id: str
    checkpoint_id: str
    checkpoint_weights_sha256: str      # content hash of the verified state
    tokenizer_id: str
    tokenizer_hash: str                 # sha256 of tokenizer.json bytes
    prompt: str                         # request text verbatim (never stripped)
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    output_text: str                    # decoded generated continuation only
    strategy: SampleStrategy
    temperature: Optional[float] = None
    seed: Optional[int] = None
    max_new_tokens: int
    prompt_token_count: int
    generated_token_count: int
    result_hash: str
    created_at: datetime
    duration_seconds: float = 0.0
    hardware: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Sample evaluation (per-sample likelihood measurement; M16)
# --------------------------------------------------------------------------- #


class SampleEvaluationRecord(BaseModel):
    """One immutable per-sample quality measurement (M16).

    Records the causal-LM likelihood measurement of ONE immutable M15
    sample under the sample's own recorded checkpoint and tokenizer
    (sample-driven state resolution — nothing is ever auto-selected or
    supplied separately). Metrics are loss_nats (mean causal
    cross-entropy over the GENERATED continuation target tokens, fp32)
    and perplexity = exp(min(loss_nats, 100)) — the exact M4 convention.

    Target accounting: targets are only the ``generated_token_count``
    generated tokens (the first generated token is conditioned on the full
    prompt); prompt tokens are never scored as generated-text targets.
    ``evaluated_token_count`` therefore always equals the generated count
    (every generated target counted exactly once, no truncation).

    Window rule: the full prompt + continuation token sequence must fit ONE
    model context window (``window_token_count <= context_length``). An
    overlong sample is rejected at preflight (422) — this platform never
    silently truncates and never invents sliding-window accounting.
    ``window_rule`` is the constant string ``single_window``.

    The record deliberately does NOT duplicate the sample's token lists;
    ``sample_result_hash`` (the sample's own recorded result_hash) and
    ``token_sequence_sha256`` (canonical hash of the exact prompt+generated
    token sequences that were measured) bind the record to the immutable
    sample. ``result_hash`` is deterministic over the semantic payload
    (metrics/identities/accounting/sequence digest); evaluation_id,
    timestamps, duration, hardware and paths are excluded, so identical
    samples over identical checkpoint bytes reproduce it byte-for-byte.
    """

    evaluation_id: str
    model_id: str
    sample_id: str
    sample_result_hash: str              # the measured sample's result_hash
    token_sequence_sha256: str           # canonical digest of the measured ids
    checkpoint_id: str
    checkpoint_weights_sha256: str       # verified content hash of the state
    tokenizer_id: str
    tokenizer_hash: str                  # sha256 of tokenizer.json bytes
    prompt_token_count: int              # conditioning tokens (never targets)
    generated_token_count: int           # generated continuation tokens
    evaluated_token_count: int           # == generated count (exact targets)
    context_length: int                  # model context window bound
    window_token_count: int              # prompt + generated tokens evaluated
    window_rule: str                     # constant "single_window"
    loss_nats: float                     # mean causal CE over generated targets
    perplexity: float                    # exp(min(loss_nats, 100))
    result_hash: str
    hardware: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    duration_seconds: float = 0.0
    schema_version: int = 1
