"""Tokenizer engine: deterministic byte-level BPE (Hugging Face ``tokenizers``).

Scope of this milestone:
  * train a byte-level BPE tokenizer from user text (files or a dataset)
  * persist exactly one ``tokenizer.json`` + one ``manifest.json`` per tokenizer
  * encode/decode for later dataset tokenization

Correctness contract (empirically verified in tests):
  * ``decode(encode(text)) == text`` EXACTLY for any text whose characters are
    printable content plus standard whitespace (space, tab, newline, CR) —
    including arbitrary multilingual Unicode and emoji.
  * Raw control bytes (e.g. ``\\x00``-``\\x08``) are NOT preserved: ByteLevel
    treats them as word boundaries, so they do not round-trip. This is the
    documented behaviour of standard byte-level BPE text tokenizers and is
    why the ingestion pipeline only produces real text records.

Determinism: the Rust trainer has no RNG; corpus order is fixed by the
ingestion order, so identical corpora always produce identical vocabularies
(tokenizer.json bytes are byte-identical, hence identical hashes).
``vocab_size`` is a ceiling: small corpora legitimately yield fewer tokens.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from . import config as forge_cfg
from .hardware import detect_hardware
from .schemas import TokenizerConfig, TokenizerRecord
from .storage import (Storage, atomic_delete_dir,
                   atomic_write_bytes, atomic_write_json,
                   hash_file_sha256, read_json)

log = forge_cfg.get_logger("tokenizer")

TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "manifest.json"
SPECIAL_DEFAULT = ["<unk>", "<s>", "</s>"]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class TokenizerEngine:
    """All tokenizer operations for one storage root."""

    def __init__(self, storage: Storage):
        self.storage = storage

    # ------------------------------------------------------------------ #
    # Registry helpers
    # ------------------------------------------------------------------ #

    def _dirs(self) -> list[Path]:
        root = self.storage.tokenizers_dir
        if not root.exists():
            return []
        return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))

    def list(self) -> list[TokenizerRecord]:
        records = []
        for d in self._dirs():
            try:
                records.append(self.load(d.name))
            except Exception:
                log.warning("skipping unreadable tokenizer dir %s", d.name)
        records.sort(key=lambda r: r.created_at)
        return records

    def load(self, tokenizer_id: str) -> TokenizerRecord:
        mpath = self.storage.tokenizer_dir(tokenizer_id) / MANIFEST_FILE
        if not mpath.exists():
            raise FileNotFoundError(f"tokenizer '{tokenizer_id}' not found")
        return TokenizerRecord(**read_json(mpath))

    def exists_id(self, tokenizer_id: str) -> bool:
        return (self.storage.tokenizer_dir(tokenizer_id) / MANIFEST_FILE).exists()

    def verify(self, tokenizer_id: str) -> dict[str, Any]:
        """Integrity check of ONE tokenizer (M65): the manifest must
        resolve (missing/unparseable -> FileNotFoundError /
        ValueError — registry-invisible, the established convention)
        and the ``tokenizer.json`` content hash must equal the
        persisted ``tokenizer_hash``. Read-only; never repairs: any
        mismatch -> status "failed". This is the verifier the M65
        deletion guard runs BEFORE the reference analysis — deletion
        never bypasses integrity validation (the M61 ordering)."""
        record = self.load(tokenizer_id)
        errors: list[str] = []
        tpath = self.storage.tokenizer_dir(tokenizer_id) / TOKENIZER_FILE
        if not tpath.exists():
            errors.append("tokenizer.json missing")
        else:
            digest = hash_file_sha256(tpath)
            if digest != record.tokenizer_hash:
                errors.append(
                    f"tokenizer.json content hash mismatch: {digest} != "
                    f"{record.tokenizer_hash}")
        return {"tokenizer_id": tokenizer_id,
                "status": "ok" if not errors else "failed",
                "errors": errors}

    def get_hf(self, tokenizer_id: str) -> Tokenizer:
        """Load the HF tokenizer object for encoding (not cached: cheap mmap)."""
        path = self.storage.tokenizer_dir(tokenizer_id) / TOKENIZER_FILE
        if not path.exists():
            raise FileNotFoundError(f"tokenizer '{tokenizer_id}' not found")
        return Tokenizer.from_file(str(path))

    def delete(self, tokenizer_id: str) -> tuple[int, int]:
        """Remove ONE tokenizer's directory ATOMICALLY (M65 low-level
        primitive, the ``remove_checkpoint`` pattern): manifest +
        tokenizer.json disappear in ONE ``os.rename`` to a hidden
        ``.tmp-delete-*`` sibling the registry scans skip. Returns
        ``(files_removed, bytes_reclaimed)`` measured from the files as
        they existed immediately before removal. The CALLER (the forge
        facade) owns the safety decision — the FULL M64 reference
        analysis must have passed before this is called, because
        deletion must never become a way to orphan referencing
        evidence."""
        tdir = self.storage.tokenizer_dir(tokenizer_id)
        files = sorted(p for p in tdir.rglob("*") if p.is_file())
        nbytes = sum(p.stat().st_size for p in files)
        atomic_delete_dir(tdir)
        log.info("deleted tokenizer %s (%d files, %d bytes)",
                 tokenizer_id, len(files), nbytes)
        return len(files), nbytes

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        config: TokenizerConfig,
        corpus: Iterable[str],
        trained_on_dataset_id: Optional[str] = None,
    ) -> TokenizerRecord:
        """Train + persist a deterministic byte-level BPE tokenizer.

        ``corpus`` supplies normalized text records; order defines
        determinism (no shuffling, no RNG). Returns the persisted record.
        """
        config = config.model_copy(deep=True)  # re-validate
        # Reject duplicate names (ids stay unique), consistent with models.
        for existing in self.list():
            if existing.name == config.name:
                raise ValueError(
                    f"a tokenizer named '{config.name}' already exists "
                    f"(id {existing.id}); choose a unique name"
                )

        specials = config.special_tokens or SPECIAL_DEFAULT
        tok = Tokenizer(models.BPE(byte_fallback=config.byte_fallback))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        # Corpus records are already stripped; a trailing "\n" gives every
        # record a clean word boundary so merges never span record edges.
        tok.train_from_iterator((r + "\n" for r in corpus), trainer=trainer)

        tokenizer_id = _new_id()
        created = datetime.now(timezone.utc)
        tdir = self.storage.tokenizer_dir(tokenizer_id)
        tdir.mkdir(parents=True, exist_ok=True)

        tokenizer_bytes = tok.to_str().encode("utf-8")
        atomic_write_bytes(tdir / TOKENIZER_FILE, tokenizer_bytes)

        actual_vocab = len(tok.get_vocab())
        record = TokenizerRecord(
            id=tokenizer_id,
            name=config.name,
            type="byte_bpe",
            requested_vocab_size=config.vocab_size,
            actual_vocab_size=actual_vocab,
            special_tokens=specials,
            byte_fallback=config.byte_fallback,
            seed=config.seed,
            trained_on_dataset_id=trained_on_dataset_id,
            config_hash=_config_hash(config),
            tokenizer_hash=hash_file_sha256(tdir / TOKENIZER_FILE),
            created_at=created,
            hardware=detect_hardware().to_dict(),
        )
        atomic_write_json(tdir / MANIFEST_FILE, record.model_dump(mode="json"))
        log.info(
            "trained tokenizer id=%s name=%r vocab=%s (requested %s) on dataset=%s",
            tokenizer_id, config.name, actual_vocab, config.vocab_size,
            trained_on_dataset_id or "<files>",
        )
        return self.load(tokenizer_id)  # round-trip through disk


def _config_hash(cfg: TokenizerConfig) -> str:
    import hashlib

    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
