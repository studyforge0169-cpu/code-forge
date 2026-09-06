"""SampleQualityEngine: per-sample likelihood measurement of one immutable
M15 generated sample under the model's existing M4 causal-LM objective.

Milestone 16 is the platform's first *measurement* layer over generated
text. Given ONLY ``model_id`` + ``sample_id`` it deterministically measures
one immutable M15 sample under the checkpoint and tokenizer the sample
itself records (sample-driven state resolution — nothing is auto-selected
and no checkpoint/tokenizer/metric configuration is ever accepted from the
caller).

Semantics (documented, honest):

  * what is measured: the complete recorded token sequence
    ``context = prompt_token_ids + generated_token_ids`` is fed through the
    model once (under ``torch.no_grad()``, existing forward path) and the
    causal shift-by-one cross-entropy is accumulated ONLY over the target
    positions corresponding to the GENERATED continuation::

        prompt tokens:        P0 P1 P2
        generated tokens:              G0 G1 G2

        causal targets:   P0->P1  P1->P2  P2->G0  G0->G1  G1->G2
        scored targets:                  ^^^^^  ^^^^^  ^^^^^   (G only)

    The first generated token (G0) is conditioned on the full prompt (its
    logits are the last prompt position's), prompt tokens are NEVER scored
    as generated-text targets, and every generated token is scored exactly
    once. ``evaluated_token_count`` therefore equals the generated count.

  * metrics: ``loss_nats`` = mean causal cross-entropy over the generated
    targets (fp32, rounded like M4 to 6 decimals) and
    ``perplexity = exp(min(loss_nats, 100))`` — the exact M4 convention.
    Perplexity is a likelihood measurement under ONE causal-LM objective;
    it is NOT an overall quality judgment.

  * window rule (ONE deterministic rule, no silent truncation): the full
    prompt + continuation must fit ONE model context window
    (``window_token_count <= model.context_length``, with the same RoPE
    cache guard M4 applies). Every sample M15 can legitimately produce
    satisfies this by M15's own generation-time preflight
    (prompt + max_new_tokens <= context_length), so all reachable samples
    are measured completely in a single window. An overlong sample can only
    exist as an edited/inconsistent manifest; inventing sliding/multi-window
    target accounting absent from M3/M4 platform history would risk an
    ambiguous metric, so overlong samples are REJECTED deterministically at
    preflight (422) with zero writes — never truncated, never windowed.

  * verification (every check precedes any write; failure order is fixed
    and documented in ``run``): model/sample resolution -> 404; unreadable
    or self-inconsistent manifest (parse failure, sample_id/model_id fields
    disagreeing with the request) -> integrity error (409); window fit ->
    422; recorded checkpoint verified through the existing M3 machinery
    (unknown -> 404, corrupt -> 409) and its content hash compared with the
    sample's recorded ``checkpoint_weights_sha256`` (mismatch -> 409);
    recorded tokenizer resolved (404) and its content hash compared with
    the sample's recorded ``tokenizer_hash`` (mismatch -> 409); platform
    vocabulary convention ``tokenizer actual vocab <= model vocab_size``
    (violation -> 422); recorded ids within the tokenizer's actual vocab
    and recorded counts equal to the id-list lengths (violation -> 422);
    the sample's own ``result_hash`` must reproduce from its recorded
    fields (failure -> 409). Internally inconsistent samples FAIL rather
    than being silently repaired.

  * storage: one successful measurement creates EXACTLY ONE immutable
    manifest under the new root-level family
    ``sample-evaluations/<model_id>/evaluation-<id>/manifest.json``
    (atomic write). Samples, checkpoints, tokenizers, M4 evaluations,
    workflows, recipes and dashboards are never modified. Repeated requests
    create separate immutable records (no silent deduplication) whose
    semantic ``result_hash`` is byte-identical.

  * ``result_hash`` covers the semantic payload: model/sample identity,
    the sample's own result_hash, a canonical digest of the exact
    prompt/generated token sequences measured, checkpoint identity +
    verified weights hash, tokenizer identity/hash, exact target
    accounting, context/window rule + counts, ``loss_nats`` and
    ``perplexity``. Evaluation id, timestamps, duration, hardware and
    filesystem paths are excluded.

M19 adds ONE read-only per-sample access method
(``list_sample_evaluations_for_sample``): the authoritative M16 listing of
one model filtered by the persisted ``sample_id`` of one sample (the sample
must exist under ``samples/<model_id>/``), preserving the M16 record
payloads and (created_at, evaluation_id) ordering exactly. It is pure data
access — no derived fields, no statistics, no quality judgment.

M20 adds ONE read-only per-checkpoint access method
(``list_sample_evaluations_for_checkpoint``): the same authoritative M16
listing of one model filtered by the persisted ``checkpoint_id`` recorded
in each SampleEvaluationRecord (the checkpoint must be registered in the
model's M3 checkpoint registry; an unknown checkpoint or a checkpoint id
belonging to another model is FileNotFoundError -> 404). Answers only
"which immutable sample-quality evaluations belong to this checkpoint?" —
no derived fields, no statistics, no quality judgment, never judges the
checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from . import config as forge_cfg
from .hardware import detect_hardware
from .model_builder import build_transformer, content_hash, restore_state
from .sampling import SAMPLES_DIR, SAMPLE_PREFIX, SamplingEngine
from .schemas import SampleEvaluationRecord, SampleRecord
from .storage import Storage, atomic_write_json, read_json
from .tokenizer import TokenizerEngine
from .training import TrainingEngine

log = forge_cfg.get_logger("sample_quality")

SAMPLE_EVALUATIONS_DIR = "sample-evaluations"   # new M16 root-level family
EVALUATION_PREFIX = "evaluation-"
MANIFEST = "manifest.json"
WINDOW_RULE = "single_window"   # full sequence in ONE context window or 422


class SampleQualityEngine:
    """Per-sample likelihood measurement for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.tokenizers = TokenizerEngine(storage)
        self.training = TrainingEngine(storage)     # checkpoint verification
        self.sampling = SamplingEngine(storage)     # sample resolution/hash

    # ------------------------------------------------------------------ #
    # Paths / registry helpers
    # ------------------------------------------------------------------ #

    def _evaluations_root(self, model_id: str) -> Path:
        return self.storage.root / SAMPLE_EVALUATIONS_DIR / model_id

    def _evaluation_dir(self, model_id: str, evaluation_id: str) -> Path:
        return (self._evaluations_root(model_id)
                / f"{EVALUATION_PREFIX}{evaluation_id}")

    def _model_exists(self, model_id: str) -> bool:
        return (self.storage.model_dir(model_id) / "manifest.json").exists()

    def list_sample_evaluations(
            self, model_id: str) -> list[SampleEvaluationRecord]:
        """Immutable per-sample measurement history of one model in
        deterministic (created_at, evaluation_id) order. Unknown model ->
        FileNotFoundError; no measurements yet -> []. Read-only."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        root = self._evaluations_root(model_id)
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith(EVALUATION_PREFIX):
                continue
            mpath = d / MANIFEST
            if mpath.exists():
                try:
                    records.append(
                        SampleEvaluationRecord(**read_json(mpath)))
                except Exception:
                    log.warning("unreadable sample-evaluation manifest %s",
                                mpath)
        records.sort(key=lambda r: (r.created_at, r.evaluation_id))
        return records

    def get_sample_evaluation(
            self, model_id: str, evaluation_id: str) -> SampleEvaluationRecord:
        """One persisted immutable measurement (404 for unknown model or
        evaluation). Never mutates anything."""
        if not self._model_exists(model_id):
            raise FileNotFoundError(f"model '{model_id}' not found")
        path = (self._evaluation_dir(model_id, evaluation_id) / MANIFEST)
        if not path.exists():
            raise FileNotFoundError(
                f"sample evaluation '{evaluation_id}' not found for model "
                f"'{model_id}'")
        return SampleEvaluationRecord(**read_json(path))

    # ------------------------------------------------------------------ #
    # M19: per-sample access (read-only)
    # ------------------------------------------------------------------ #

    def list_sample_evaluations_for_sample(
            self, model_id: str, sample_id: str
    ) -> list[SampleEvaluationRecord]:
        """Immutable M16 measurements of ONE sample of ONE model (M19).

        Resolution: unknown model -> FileNotFoundError; the sample must
        exist under samples/<model_id>/ — an unknown sample or a sample id
        that belongs to another model both raise FileNotFoundError via the
        M15 sample getter (sample ids are model-scoped paths; nothing is
        inferred from filenames or graph edges). The result is the model's
        authoritative M16 listing (the exact engine parse + deterministic
        (created_at, evaluation_id) ASCENDING order) filtered by the
        sample's persisted ``sample_id``, so every returned record is a
        complete verbatim SampleEvaluationRecord and records of other
        samples/models never appear. A sample with no measurements returns
        []. Read-only, never writes.
        """
        # existence + ownership: raises FileNotFoundError (404 at the API)
        self.sampling.get_sample(model_id, sample_id)
        return [r for r in self.list_sample_evaluations(model_id)
                if r.sample_id == sample_id]

    # ------------------------------------------------------------------ #
    # M20: per-checkpoint access (read-only)
    # ------------------------------------------------------------------ #

    def list_sample_evaluations_for_checkpoint(
            self, model_id: str, checkpoint_id: str
    ) -> list[SampleEvaluationRecord]:
        """Immutable M16 measurements recorded under ONE checkpoint of ONE
        model (M20).

        Resolution: unknown model or unregistered checkpoint ->
        FileNotFoundError; ownership is validated through the model's M3
        checkpoint registry (``TrainingEngine.get_checkpoint``) — a
        checkpoint id belonging to another model is not registered under
        this model and raises FileNotFoundError, exactly like an unknown
        one (nothing is inferred from filenames or graph edges). The result
        is the model's authoritative M16 listing (the exact engine parse +
        deterministic (created_at, evaluation_id) ASCENDING order) filtered
        by the persisted ``checkpoint_id`` recorded in each
        SampleEvaluationRecord, so every returned record is a complete
        verbatim SampleEvaluationRecord and records of other
        checkpoints/models never appear. A checkpoint with no measurements
        returns []. Read-only, never writes.
        """
        # existence + ownership: raises FileNotFoundError (404 at the API)
        self.training.get_checkpoint(model_id, checkpoint_id)
        return [r for r in self.list_sample_evaluations(model_id)
                if r.checkpoint_id == checkpoint_id]

    def list_sample_evaluations_for_tokenizer(
            self, model_id: str, tokenizer_id: str
    ) -> list[SampleEvaluationRecord]:
        """Immutable M16 sample-quality measurements of ONE model whose
        measured samples were generated with ONE tokenizer (M33).

        Membership comes from the persisted measurement tokenizer
        identity ONLY: every ``SampleEvaluationRecord`` carries a
        required non-nullable top-level ``tokenizer_id`` (M16 measures
        an immutable M15 sample under its own RECORDED state — the
        record persists the sample's tokenizer identity verbatim as
        part of that recorded state), and a measurement belongs to the
        request when its persisted ``tokenizer_id`` equals the
        requested id, matched VERBATIM — never filenames, paths,
        sample ids, checkpoint ids, tokenizer contents or hashes, the
        tokenizer currently registered, or a latest-tokenizer
        substitution (tokenizer ids are opaque ids; M2/M16 have no
        tokenizer versioning and none is introduced here; the record
        is NOT altered for this endpoint). Each record appears EXACTLY
        ONCE (the authoritative listing holds each record exactly
        once). Resolution: unknown model or unknown tokenizer ->
        FileNotFoundError; the tokenizer is validated through the
        existing registry (``TokenizerEngine.load`` — the same
        registry getter ``GET /tokenizers/{id}`` exposes; tokenizers
        are GLOBAL, so the model scoping comes from the model's own
        M16 listing — a model never sees another model's
        measurements). The result keeps the authoritative M16
        (created_at, evaluation_id) ASCENDING order. A valid tokenizer
        with no measurements for the model returns []. Read-only,
        never writes.
        """
        # existence: raises FileNotFoundError (404 at the API)
        self.tokenizers.load(tokenizer_id)
        return [r for r in self.list_sample_evaluations(model_id)
                if r.tokenizer_id == tokenizer_id]

    # ------------------------------------------------------------------ #
    # The run (sample-driven; preflight everything before any write)
    # ------------------------------------------------------------------ #

    def run(self, model_id: str, sample_id: str) -> SampleEvaluationRecord:
        """Measure ONE immutable M15 sample under its recorded state.

        Fixed preflight order (each failure -> NO manifest, nothing else
        touched): model exists (404) -> sample manifest exists and parses
        (404 / corrupt 409) -> manifest self-identity (sample_id/model_id
        fields agree with the request; else 409) -> context-window fit
        (422) -> recorded checkpoint verified (404 / corrupt 409) + its
        content hash equals the sample's recorded checkpoint hash (409) ->
        recorded tokenizer resolved (404) + content hash equals the
        sample's recorded tokenizer hash (409) + platform vocabulary
        convention (422) -> recorded counts match the id lists and every
        recorded id is within the tokenizer's actual vocab (422) -> the
        sample's own result_hash reproduces from its recorded fields (409)
        -> build/restore/eval -> measure -> persist exactly one manifest.
        """
        start = time.monotonic()
        started_at = datetime.now(timezone.utc)
        spec = detect_hardware()

        # ---- 1. model + sample resolution (404 semantics) ----------------
        try:
            model_record = self.storage.load_record(model_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"model '{model_id}' not found") from None
        model_cfg = model_record.config
        path = self._sample_path(model_id, sample_id)
        try:
            raw = read_json(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"sample '{sample_id}' not found for model '{model_id}'"
            ) from None
        except Exception as exc:
            raise RuntimeError(
                f"sample '{sample_id}' manifest is unreadable or corrupt: "
                f"{exc}") from exc
        try:
            sample = SampleRecord(**raw)
        except ValidationError as exc:
            raise RuntimeError(
                f"sample '{sample_id}' manifest failed integrity "
                f"verification (fields do not parse as a sample record): "
                f"{exc}") from exc
        if sample.sample_id != sample_id or sample.model_id != model_id:
            raise RuntimeError(
                f"sample '{sample_id}' manifest is corrupt: its recorded "
                f"sample_id/model_id do not match this location "
                f"({sample.sample_id}/{sample.model_id})")

        # ---- 2. context-window rule: fit ONE window or reject ------------
        prompt_ids = sample.prompt_token_ids
        generated_ids = sample.generated_token_ids
        p_len, g_len = len(prompt_ids), len(generated_ids)
        total = p_len + g_len
        if g_len < 1 or p_len < 1:
            raise ValueError(
                "invalid measurement state: the sample records no prompt "
                "tokens or no generated tokens — there is nothing to score "
                "under the causal-LM objective")
        if (sample.prompt_token_count != p_len
                or sample.generated_token_count != g_len):
            raise ValueError(
                "invalid measurement state: recorded token counts do not "
                "match the recorded id lists — exact target accounting is "
                "impossible")
        if total > model_cfg.context_length or (
                model_cfg.position_encoding.value == "rope"
                and total > model_cfg.rope_max_seq()):
            raise ValueError(
                f"sample '{sample_id}' is {total} tokens long (prompt "
                f"{p_len} + generated {g_len}) but the model's context "
                f"window is {model_cfg.context_length} — this platform's "
                f"samples always fit one window at generation time (M15 "
                f"preflight), so an overlong sample is an inconsistent "
                f"record; measuring it would require inventing "
                f"sliding-window accounting absent from the M3/M4 "
                f"objective, so it is rejected — never silently truncated")

        # ---- 3. recorded checkpoint: verify + hash cross-check ----------
        try:
            state = self.training.verify_checkpoint(
                model_id, sample.checkpoint_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"checkpoint '{sample.checkpoint_id}' not found for model "
                f"'{model_id}'") from None
        state_hash = content_hash(state)
        if state_hash != sample.checkpoint_weights_sha256:
            raise RuntimeError(
                f"sample '{sample_id}' failed integrity verification: its "
                f"recorded checkpoint weights hash does not match the "
                f"verified checkpoint content — refusing to measure a "
                f"sample that points at different bytes")

        # ---- 4. recorded tokenizer: resolve + hash + vocab convention ----
        try:
            tok_record = self.tokenizers.load(sample.tokenizer_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"tokenizer '{sample.tokenizer_id}' not found"
            ) from None
        if tok_record.tokenizer_hash != sample.tokenizer_hash:
            raise RuntimeError(
                f"sample '{sample_id}' failed integrity verification: its "
                f"recorded tokenizer hash does not match the stored "
                f"tokenizer '{sample.tokenizer_id}' content")
        if tok_record.actual_vocab_size > model_cfg.vocab_size:
            raise ValueError(
                f"tokenizer '{sample.tokenizer_id}' vocab "
                f"({tok_record.actual_vocab_size}) exceeds the model's "
                f"vocab_size ({model_cfg.vocab_size}) — the sample's "
                f"recorded tokenizer and model are incompatible (the "
                f"platform convention requires model vocab_size >= "
                f"tokenizer vocab, exactly as M3/M4/M15 enforce it)")
        if any(i < 0 or i >= tok_record.actual_vocab_size
               for i in prompt_ids + generated_ids):
            raise ValueError(
                "invalid measurement state: the sample records token ids "
                f"outside the tokenizer's actual vocab "
                f"({tok_record.actual_vocab_size}) — the sequence cannot "
                "be measured or decoded faithfully")

        # ---- 5. the sample's own result_hash must reproduce --------------
        if SamplingEngine.result_hash(sample) != sample.result_hash:
            raise RuntimeError(
                f"sample '{sample_id}' failed integrity verification: its "
                f"recorded result_hash does not reproduce from its recorded "
                f"fields — the manifest was altered after generation")

        # ---- 6. build + verified restore (exact M4/M15 convention) -------
        device = "cuda" if spec.has_gpu else "cpu"
        model = build_transformer(model_cfg).module
        missing, unexpected = restore_state(model, state)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint state incompatible with the model config: "
                f"{len(missing)} missing, {len(unexpected)} unexpected")
        model.to(device)
        model.eval()

        # ---- 7. measure (causal-LM objective, generated targets only) ----
        loss, perplexity = self._measure(
            model, prompt_ids + generated_ids, p_len, g_len, device)
        duration = round(time.monotonic() - start, 3)

        seq_digest = self.token_sequence_sha256(prompt_ids, generated_ids)
        record = SampleEvaluationRecord(
            evaluation_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            sample_id=sample_id,
            sample_result_hash=sample.result_hash,
            token_sequence_sha256=seq_digest,
            checkpoint_id=sample.checkpoint_id,
            checkpoint_weights_sha256=state_hash,
            tokenizer_id=sample.tokenizer_id,
            tokenizer_hash=sample.tokenizer_hash,
            prompt_token_count=p_len,
            generated_token_count=g_len,
            evaluated_token_count=g_len,
            context_length=model_cfg.context_length,
            window_token_count=total,
            window_rule=WINDOW_RULE,
            loss_nats=round(float(loss), 6),
            perplexity=round(float(perplexity), 6),
            result_hash="",  # filled below (needs the rounded metrics)
            hardware=spec.to_dict(),
            created_at=started_at,
            duration_seconds=duration,
        )
        record = record.model_copy(update={
            "result_hash": self.result_hash(record)})
        self._persist(record)
        log.info("sample evaluation %s of sample %s (model %s): loss %.4f "
                 "ppl %.2f over %d generated target token(s), %d prompt "
                 "token(s), window %d/%d", record.evaluation_id,
                 sample_id, model_id, record.loss_nats, record.perplexity,
                 record.evaluated_token_count, record.prompt_token_count,
                 record.window_token_count, record.context_length)
        return record

    # ------------------------------------------------------------------ #
    # Measurement (pure; no RNG; the exact M4 causal-LM objective)
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def _measure(model: torch.nn.Module, ids: list[int], p_len: int,
                 g_len: int, device: str) -> tuple[float, float]:
        """Causal next-token cross-entropy over the GENERATED targets only.

        ``ids`` is the full prompt + generated sequence (length p_len +
        g_len, already guaranteed to fit one context window). One forward
        pass yields logits for every position; the generated token G_i is
        scored against the logits of position p_len - 1 + i (the FIRST
        generated token is conditioned on the last prompt position, i.e.
        the full prompt). Prompt positions are never targets. The mean over
        the g_len per-target cross-entropies is the token-weighted mean —
        the exact M4 loss per target position, fp32, no RNG anywhere.
        """
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(input_ids)                      # (1, T, V)
        v = logits.shape[-1]
        score = logits[:, p_len - 1: p_len + g_len - 1, :]   # (1, G, V)
        targets = input_ids[:, p_len:]                       # (1, G)
        loss = F.cross_entropy(score.reshape(-1, v),
                               targets.reshape(-1))
        ppl = float(np.exp(min(float(loss), 100.0)))
        return float(loss), ppl

    @staticmethod
    def token_sequence_sha256(prompt_ids: list[int],
                              generated_ids: list[int]) -> str:
        """Canonical digest of the EXACT token sequence a measurement used.

        Binds a sample-evaluation record to the precise prompt/generated
        id lists of the immutable sample (the record stores this digest
        instead of duplicating the lists).
        """
        blob = json.dumps({"prompt_token_ids": prompt_ids,
                           "generated_token_ids": generated_ids},
                          sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ------------------------------------------------------------------ #
    # Deterministic result hash + persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def result_hash(record: SampleEvaluationRecord) -> str:
        """Deterministic hash of a measurement's semantic payload.

        Includes model/sample identity, the sample's own result_hash, the
        canonical digest of the exact measured token sequence, checkpoint
        identity + verified weights hash, tokenizer identity/hash, exact
        target accounting (prompt/generated/evaluated counts), the
        context/window rule + window count + context bound, and the rounded
        metrics. Excludes evaluation_id, created_at, duration, hardware and
        filesystem paths, so identical samples over identical checkpoint
        bytes reproduce the hash byte-for-byte.
        """
        payload = {
            "model_id": record.model_id,
            "sample_id": record.sample_id,
            "sample_result_hash": record.sample_result_hash,
            "token_sequence_sha256": record.token_sequence_sha256,
            "checkpoint_id": record.checkpoint_id,
            "checkpoint_weights_sha256": record.checkpoint_weights_sha256,
            "tokenizer_id": record.tokenizer_id,
            "tokenizer_hash": record.tokenizer_hash,
            "prompt_token_count": record.prompt_token_count,
            "generated_token_count": record.generated_token_count,
            "evaluated_token_count": record.evaluated_token_count,
            "context_length": record.context_length,
            "window_token_count": record.window_token_count,
            "window_rule": record.window_rule,
            "loss_nats": record.loss_nats,
            "perplexity": record.perplexity,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _sample_path(self, model_id: str, sample_id: str) -> Path:
        return (self.storage.root / SAMPLES_DIR / model_id
                / f"{SAMPLE_PREFIX}{sample_id}" / MANIFEST)

    def _persist(self, record: SampleEvaluationRecord) -> None:
        """Write one immutable sample-evaluation manifest (atomic; only
        written after a fully successful measurement)."""
        edir = self._evaluation_dir(record.model_id, record.evaluation_id)
        edir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(edir / MANIFEST, record.model_dump(mode="json"))
