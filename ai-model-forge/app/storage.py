"""Storage engine: layout, atomic writes, content hashing, deduplication.

Layout (storage root, default ~/ai-model-forge-data, override FORGE_ROOT):

    storage_root/
      project.json                 # project identity
      tmp/                         # atomic-write scratch (cleaned on startup)
      models/
        <model_id>/
          manifest.json            # ALL metadata (single source of truth)
          weights.pt               # current model state (torch state dict)
          weights.sha256           # hex content hash of weights.pt

Design rules honoured here:
  * metadata never duplicated — manifest.json is the one record per model
  * weights stored once; future stage history will reference, not copy
  * every write is atomic (temp file + os.replace) so a crash cannot corrupt
  * content hashing enables later deduplication and integrity checks

The storage class itself is stateless and test-friendly (plain directory).
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterator

from . import config
from .schemas import ModelRecord, TransformerConfig

# Model sub-directory layout
WEIGHTS_FILE = "weights.pt"
HASH_FILE = "weights.sha256"
MANIFEST_FILE = "manifest.json"
PROJECT_FILE = "project.json"


def _freeze_hash(data: dict[str, Any]) -> str:
    """Stable content hash of a JSON-able record (used for dedup/change detection)."""
    import hashlib

    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write raw bytes atomically (temp file + fsync + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Storage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.datasets_dir = self.root / "datasets"
        self.tokenizers_dir = self.root / "tokenizers"
        self.tmp_dir = self.root / "tmp"

    # ------------------------------------------------------------------ #
    # Artifact layout (single place where paths are defined)
    # ------------------------------------------------------------------ #

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def tokenizer_dir(self, tokenizer_id: str) -> Path:
        return self.tokenizers_dir / tokenizer_id

    # ------------------------------------------------------------------ #
    # Project lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self, project_name: str = "default") -> Path:
        import datetime as _dt

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        project_path = self.root / PROJECT_FILE
        if not project_path.exists():
            atomic_write_json(project_path, {
                "name": project_name,
                "initialized_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "app_version": config.APP_VERSION,
                "manifest_schema_version": config.MANIFEST_SCHEMA_VERSION,
            })
        # Always clean stale temp files from crashed writers.
        for stale in self.tmp_dir.glob("*"):
            stale.unlink(missing_ok=True)
        return project_path

    def project_info(self) -> dict[str, Any]:
        self.initialize()
        return read_json(self.root / PROJECT_FILE)

    # ------------------------------------------------------------------ #
    # Model records
    # ------------------------------------------------------------------ #

    def _valid_model_dirs(self) -> Iterator[Path]:
        for entry in sorted(self.models_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name.startswith("tmp"):
                continue
            yield entry

    def model_ids(self) -> list[str]:
        return [d.name for d in self._valid_model_dirs()]

    def save_record(self, record: ModelRecord) -> None:
        """Persist the manifest. record.config is stored as its JSON form."""
        payload = record.model_dump(mode="json")
        payload["config"] = record.config.model_dump(mode="json")
        atomic_write_json(self.model_dir(record.id) / MANIFEST_FILE, payload)

    def load_record(self, model_id: str) -> ModelRecord:
        raw = read_json(self.model_dir(model_id) / MANIFEST_FILE)
        raw["config"] = TransformerConfig(**raw["config"])
        return ModelRecord(**raw)

    def load_all(self) -> list[ModelRecord]:
        records = []
        for model_id in self.model_ids():
            try:
                records.append(self.load_record(model_id))
            except Exception:  # a broken/corrupt manifest must not take the API down
                records.append(None)  # type: ignore[list-item]
        return [r for r in records if r is not None]

    # ------------------------------------------------------------------ #
    # Weights
    # ------------------------------------------------------------------ #

    def weights_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / WEIGHTS_FILE

    def write_weights(self, model_id: str, state: dict[str, Any]) -> Path:
        """Persist a torch state dict atomically (stream to temp, fsync, rename)."""
        import torch

        target = self.weights_path(model_id)
        model_dir = self.model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        tmp = model_dir / f".tmp-weights-{uuid.uuid4().hex}.pt"
        try:
            torch.save(state, tmp)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        # Sidecar content hash for integrity checks / future dedup. Written
        # after the weights file so it always reflects the stored bytes.
        digest = hash_file_sha256(target)
        atomic_write_json(model_dir / HASH_FILE, {"sha256": digest})
        return target

    def load_weights(self, model_id: str, device: str = "cpu") -> dict[str, Any]:
        """Load state back; cross-checks the sidecar hash when present."""
        import torch

        path = self.weights_path(model_id)
        if not path.exists():
            raise FileNotFoundError(f"no weights for model {model_id}")
        state = torch.load(path, map_location=device, weights_only=True)
        sidecar = self.model_dir(model_id) / HASH_FILE
        if sidecar.exists():
            stored = read_json(sidecar).get("sha256")
            actual = hash_file_sha256(path)
            if stored and stored != actual:
                raise RuntimeError(
                    f"weight integrity check failed for model {model_id}: "
                    f"stored sha256 {stored[:16]}… != actual {actual[:16]}…"
                )
        return state

    def verify_integrity(self, model_id: str) -> None:
        """Raise on any mismatch between record, weights and hash sidecar."""
        record = self.load_record(model_id)
        state = self.load_weights(model_id)  # raises on hash mismatch
        if not state:
            raise RuntimeError(f"model {model_id}: empty weight state")
        return record

    def remove(self, model_id: str) -> None:
        shutil.rmtree(self.model_dir(model_id), ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def usage(self) -> dict[str, Any]:
        total_bytes = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size
        return {
            "path": str(self.root),
            "bytes": total_bytes,
            "models": len(self.model_ids()),
            "datasets": len(self._ids_in(self.datasets_dir)),
            "tokenizers": len(self._ids_in(self.tokenizers_dir)),
        }

    @staticmethod
    def _ids_in(directory: Path) -> list[str]:
        if not directory.exists():
            return []
        return sorted(
            p.name for p in directory.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
