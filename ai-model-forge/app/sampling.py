"""SamplingEngine: deterministic text generation from ONE explicitly
selected, verified, immutable checkpoint (M15).

M15 is the platform's first model-USAGE (inference) primitive: given an
explicit model + checkpoint + tokenizer + prompt + decoding strategy, it
produces exactly ONE immutable sample manifest. It never trains, evaluates,
scores, ranks, judges or modifies anything.

Semantics (documented, honest):

  * explicit state selection ONLY — model_id, checkpoint_id, tokenizer_id,
    prompt, strategy and every generation parameter come from the request;
    nothing is ever auto-selected (never current/latest/best weights, never
    a guessed tokenizer, never a dataset)
  * two decoding strategies, never mixed:
      - ``greedy``: per-step argmax over the logits — no RNG anywhere
      - ``temperature``: logits / temperature -> softmax -> ONE sample via
        torch.multinomial from a per-request ``torch.Generator`` seeded with
        the request's explicit integer seed (CPU generator, fp32 logits);
        the same seed + request reproduces the same token stream
    temperature in (0, 1]; temperature=0 is REJECTED, never reinterpreted
    as greedy (greedy must be requested through ``strategy="greedy"``)
  * vocabulary rule (the platform convention M3/M4 already document): the
    tokenizer's actual vocab must fit the model's ``vocab_size`` (model
    vocab >= tokenizer vocab). Logits are restricted to the tokenizer's
    actual vocab range at every step, so every generated token id is
    decodable — a checkpoint trained by this platform is always usable
    through its tokenizer, exactly as training/evaluation use it
  * no EOS policy: generation stops after exactly ``max_new_tokens`` tokens
    (the stored tokenizers have no post-processor/EOS semantics and the
    models were not trained with EOS boundaries; inventing one would be a
    new rule). ``max_new_tokens`` must therefore fit in the model's context
    window: ``prompt_tokens + max_new_tokens <= context_length`` is enforced
    at preflight (422) — never silently truncated
  * prompt text is encoded verbatim (never stripped) with the stored HF
    tokenizer (no special tokens are added — the stored tokenizers define
    no post-processor) and decoded with that same tokenizer. Raw control
    bytes do not round-trip (the documented M2 tokenizer limitation);
    ``output_text`` is the decoded GENERATED continuation (the prompt is
    kept verbatim in the record, token ids are kept for exact audit)
  * checkpoint integrity: the requested checkpoint is verified through the
    existing M3 machinery (content hash vs its manifest ``weights_sha256``)
    BEFORE anything is loaded or written; a corrupt checkpoint is refused
    with the established integrity error
  * model loading reuses the existing builder/restore primitives exactly as
    M4 does (build_transformer from the model's stored config, restore the
    verified state, eval mode); decode runs under ``torch.no_grad()`` with
    the existing incremental KV-cache forward path; no optimizer, no
    gradients, no training path is ever constructed
  * every successful request writes exactly ONE atomic manifest under
    ``samples/<model_id>/sample-<id>/manifest.json`` (new root-level family
    following the M10 ``suite-runs`` convention); preflight/runtime failure
    creates NO manifest and touches nothing else. No weights, tokenizer
    copies, blobs, caches or temp files are ever written
  * ``result_hash`` follows the platform canonical-hash philosophy: sha256
    over the semantic payload (model/checkpoint identity + verified weights
    hash + tokenizer identity/hash + prompt + strategy + parameters + seed +
    token ids + output text); sample id, timestamps, duration, hardware and
    paths are excluded — identical requests over identical immutable inputs
    reproduce the hash byte-for-byte
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from . import config as forge_cfg
from .hardware import detect_hardware
from .model_builder import build_transformer, content_hash, restore_state
from .schemas import (
    SampleGenerateRequest,
    SampleRecord,
    SampleStrategy,
)
from .storage import Storage, atomic_write_json, read_json
from .tokenizer import TokenizerEngine
from .training import TrainingEngine

log = forge_cfg.get_logger("sampling")

SAMPLES_DIR = "samples"            # root-level family (M10 suite-runs style)
SAMPLE_PREFIX = "sample-"
MANIFEST = "manifest.json"
MAX_NEW_TOKENS_CAP = 512           # schema mirror (documented hard cap)


class SamplingEngine:
    """Everything sampling-related for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.tokenizers = TokenizerEngine(storage)
        self.training = TrainingEngine(storage)   # checkpoint verification

    # ------------------------------------------------------------------ #
    # Paths / registry helpers
    # ------------------------------------------------------------------ #

    def _samples_root(self, model_id: str) -> Path:
        return self.storage.root / SAMPLES_DIR / model_id

    def _sample_dir(self, model_id: str, sample_id: str) -> Path:
        return self._samples_root(model_id) / f"{SAMPLE_PREFIX}{sample_id}"

    def _model_exists(self, model_id: str) -> bool:
        return (self.storage.model_dir(model_id) / "manifest.json").exists()

    def list_samples(self, model_id: str) -> list[SampleRecord]:
        """Immutable sample history of one model, deterministic order
        ((created_at, sample_id)). Unknown model -> FileNotFoundError."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        root = self._samples_root(model_id)
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith(SAMPLE_PREFIX):
                continue
            mpath = d / MANIFEST
            if mpath.exists():
                try:
                    records.append(SampleRecord(**read_json(mpath)))
                except Exception:
                    log.warning("unreadable sample manifest %s", mpath)
        records.sort(key=lambda r: (r.created_at, r.sample_id))
        return records

    def get_sample(self, model_id: str, sample_id: str) -> SampleRecord:
        """One persisted immutable sample (404 for unknown model or sample)."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        path = self._sample_dir(model_id, sample_id) / MANIFEST
        if not path.exists():
            raise FileNotFoundError(
                f"sample '{sample_id}' not found for model '{model_id}'")
        return SampleRecord(**read_json(path))

    # ------------------------------------------------------------------ #
    # The run (preflight everything before creating any artifact)
    # ------------------------------------------------------------------ #

    def run(self, request: SampleGenerateRequest) -> SampleRecord:
        """Generate text from the explicitly selected verified checkpoint.

        Preflight order (each failure -> no manifest, nothing else touched):
        model exists -> checkpoint verified (404 unknown / integrity error
        corrupt) -> tokenizer exists -> tokenizer/model vocabulary
        compatibility (422) -> prompt encodes to >= 1 token (422) ->
        prompt + max_new_tokens fits the model context (422) -> decode.
        """
        start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        spec = detect_hardware()

        # ---- model + checkpoint resolution and verification -------------
        try:
            model_record = self.storage.load_record(request.model_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"model '{request.model_id}' not found") from None
        # verify_checkpoint: FileNotFoundError for unknown checkpoint,
        # RuntimeError (integrity) for corrupt weights — nothing is loaded
        # into a model and nothing is written before this passes.
        state = self.training.verify_checkpoint(
            request.model_id, request.checkpoint_id)
        model_cfg = model_record.config
        state_hash = content_hash(state)

        # ---- tokenizer resolution + vocabulary compatibility ------------
        try:
            tok_record = self.tokenizers.load(request.tokenizer_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"tokenizer '{request.tokenizer_id}' not found") from None
        if tok_record.actual_vocab_size > model_cfg.vocab_size:
            raise ValueError(
                f"tokenizer '{request.tokenizer_id}' vocab "
                f"({tok_record.actual_vocab_size}) exceeds the model's "
                f"vocab_size ({model_cfg.vocab_size}) — the tokenizer and "
                f"model are incompatible (the platform convention requires "
                f"model vocab_size >= tokenizer vocab, exactly as M3/M4 "
                f"enforce it)")
        try:
            hf = self.tokenizers.get_hf(request.tokenizer_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"tokenizer '{request.tokenizer_id}' not found") from None

        # ---- prompt encoding + context fit (never silently truncated) ---
        prompt_ids = hf.encode(request.prompt).ids
        if not prompt_ids:
            raise ValueError(
                "the prompt encodes to zero tokens — provide text the "
                "tokenizer can represent")
        if len(prompt_ids) + request.max_new_tokens > model_cfg.context_length:
            raise ValueError(
                f"prompt ({len(prompt_ids)} tokens) + max_new_tokens "
                f"({request.max_new_tokens}) exceeds the model's "
                f"context_length ({model_cfg.context_length}) — reduce "
                f"max_new_tokens or the prompt")

        # ---- model build + verified state restore (exact M4 convention) --
        device = "cuda" if spec.has_gpu else "cpu"
        model = build_transformer(model_cfg).module
        missing, unexpected = restore_state(model, state)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint state incompatible with the model config: "
                f"{len(missing)} missing, {len(unexpected)} unexpected")
        model.to(device)
        model.eval()
        input_ids = torch.tensor([prompt_ids], dtype=torch.long,
                                 device=device)

        generated = self._decode(
            model, input_ids,
            vocab_size=tok_record.actual_vocab_size,
            strategy=request.strategy, temperature=request.temperature,
            seed=request.seed, max_new_tokens=request.max_new_tokens)
        output_text = hf.decode(generated)
        duration = round(time.monotonic() - start, 3)

        record = SampleRecord(
            sample_id=uuid.uuid4().hex[:12],
            model_id=request.model_id,
            checkpoint_id=request.checkpoint_id,
            checkpoint_weights_sha256=state_hash,
            tokenizer_id=request.tokenizer_id,
            tokenizer_hash=tok_record.tokenizer_hash,
            prompt=request.prompt,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated,
            output_text=output_text,
            strategy=request.strategy,
            temperature=request.temperature,
            seed=request.seed,
            max_new_tokens=request.max_new_tokens,
            prompt_token_count=len(prompt_ids),
            generated_token_count=len(generated),
            result_hash="",  # filled below (needs the rounded outputs)
            created_at=started_at,
            duration_seconds=duration,
            hardware=spec.to_dict(),
        )
        record = record.model_copy(update={
            "result_hash": self.result_hash(record)})
        self._persist(record)
        log.info("sample %s of model %s from checkpoint %s: %s, %d token(s), "
                 "%d prompt token(s)", record.sample_id, record.model_id,
                 record.checkpoint_id, record.strategy.value,
                 record.generated_token_count, record.prompt_token_count)
        return record

    # ------------------------------------------------------------------ #
    # Autoregressive decode (reuses the existing forward + KV cache)
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def _decode(model, input_ids: torch.Tensor, *, vocab_size: int,
                strategy: SampleStrategy, temperature: Optional[float],
                seed: Optional[int], max_new_tokens: int) -> list[int]:
        """Token-by-token decode through the model's incremental KV cache.

        ``vocab_size`` is the TOKENIZER's actual vocab: logits are restricted
        to that range at every step so each generated id is decodable. Greedy
        = argmax (no RNG); temperature = softmax((logits/temp)) over the
        restricted range, sampled once per step from a CPU generator seeded
        with the explicit request seed. fp32 logits throughout. The caller
        already guarantees prompt + max_new_tokens fit the context, so the
        loop always produces exactly max_new_tokens ids (no EOS policy).
        """
        logits, caches = model(input_ids)          # prefill
        generated: list[int] = []
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        for _ in range(max_new_tokens):
            last = logits[:, -1, :vocab_size].float()   # (1, tokenizer vocab)
            if strategy == SampleStrategy.GREEDY:
                nxt = torch.argmax(last, dim=-1, keepdim=True)
            else:
                assert temperature is not None
                probs = torch.softmax(last / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1,
                                        generator=generator)
            generated.append(int(nxt.item()))
            logits, caches = model(nxt, kv_caches=caches)  # 1-token step
        return generated

    # ------------------------------------------------------------------ #
    # Deterministic result hash + persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def result_hash(record: SampleRecord) -> str:
        """Deterministic hash of a sample's semantic payload.

        Includes model/checkpoint identity + the VERIFIED checkpoint weights
        content hash, tokenizer identity + content hash, the prompt verbatim
        and its token ids, strategy, temperature, seed, max_new_tokens, the
        generated token ids and the decoded output text. Excludes sample_id,
        created_at, duration, hardware and filesystem paths, so identical
        requests over identical immutable inputs reproduce the hash.
        """
        payload = {
            "model_id": record.model_id,
            "checkpoint_id": record.checkpoint_id,
            "checkpoint_weights_sha256": record.checkpoint_weights_sha256,
            "tokenizer_id": record.tokenizer_id,
            "tokenizer_hash": record.tokenizer_hash,
            "prompt": record.prompt,
            "prompt_token_ids": record.prompt_token_ids,
            "strategy": record.strategy.value,
            "temperature": record.temperature,
            "seed": record.seed,
            "max_new_tokens": record.max_new_tokens,
            "generated_token_ids": record.generated_token_ids,
            "output_text": record.output_text,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _persist(self, record: SampleRecord) -> None:
        """Write one immutable sample manifest (atomic; never rewritten)."""
        sdir = self._sample_dir(record.model_id, record.sample_id)
        sdir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(sdir / MANIFEST, record.model_dump(mode="json"))
