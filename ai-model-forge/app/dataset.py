"""Data engine: ingestion, dedup, splitting, versioning, verification, tokenization.

Pipeline implemented here (source files are never copied — only normalized
records, metadata and hashes are stored):

    input files
      -> decode (strict UTF-8)
      -> extract records (txt/md paragraphs, csv rows, json text fields)
      -> drop empty records
      -> sha256 deduplication (within upload)
      -> quality statistics
      -> deterministic split assignment (hash buckets, no RNG)
      -> compressed, versioned snapshot   datasets/<id>/v<N>/records.jsonl.gz

Versioning contract
-------------------
* Every version is a SELF-CONTAINED full snapshot of the unique records seen
  in that upload: ``dataset_id + version`` fully describes the data, and
  historical version directories are never rewritten.
* Version files are written atomically; once ``manifest.json`` exists a
  version is immutable. The only mutable files are the dataset pointer
  (``dataset.json``) and the dedup index (``records.sha256``).
* Records are stored as ``{"id": <sha256 of text>, "text": ...}`` lines.
  Splits are NOT stored as lists: membership derives from the record id
  (deterministic), so consumers can recompute splits from stored ids — and
  verification re-derives them to catch tampering.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from . import config as forge_cfg
from .schemas import (
    CountsInfo,
    DatasetInfo,
    QualityInfo,
    SourceFileInfo,
    SplitInfo,
    TokenizedInfo,
    VersionManifest,
)
from .storage import (Storage, atomic_delete_dir,
                   atomic_write_bytes, atomic_write_json,
                   hash_file_sha256, read_json)

log = forge_cfg.get_logger("dataset")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}
SPLITS = ("train", "validation", "test")
# Allocation: bucket = int(record_id[:8], 16) % 1000
#   < 900 -> train, < 950 -> validation, else test   (90 / 5 / 5)
_BUCKET_TRAIN, _BUCKET_VALIDATION = 900, 950
ALLOCATION_SPEC: dict[str, Any] = {
    "method": "sha256_bucket_mod1000",
    "boundaries": {"train": _BUCKET_TRAIN, "validation": _BUCKET_VALIDATION, "test": 1000},
    "note": "bucket = int(record_hash[:8], 16) % 1000; <900 train, <950 validation, else test",
}

VERSION_MANIFEST = "manifest.json"
RECORDS_FILE = "records.jsonl.gz"
DATASET_META = "dataset.json"
INDEX_FILE = "records.sha256"
TOKENIZED_DIR = "tokenized"
_JSON_DEPTH_LIMIT = 16


# =========================================================================== #
# Record identity & deterministic splitting
# =========================================================================== #


def normalize_text(text: str) -> str:
    """Canonical record form: universal line endings + outer trim.

    Internal content is deliberately preserved (no aggressive rewriting).
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def record_id_of(text: str) -> str:
    """Canonical record identity: sha256 of the normalized UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_of(record_id: str) -> str:
    bucket = int(record_id[:8], 16) % 1000
    if bucket < _BUCKET_TRAIN:
        return "train"
    if bucket < _BUCKET_VALIDATION:
        return "validation"
    return "test"


def split_hash(record_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for rid in record_ids:
        digest.update(rid.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


# =========================================================================== #
# Text extraction: one file -> candidate records + per-file errors
# =========================================================================== #


def _extract_paragraphs(text: str) -> list[str]:
    """TXT / Markdown: split on blank lines and trim.

    Blank blocks are emitted as "" so the caller can count them as empty
    records (they are dropped centrally, never stored).
    """
    return [p.strip() for p in re.split(r"\n[ \t]*\n", text)]


def _extract_csv(text: str, errors: list[str]) -> tuple[list[str], Optional[list[str]]]:
    """CSV: one record per data row; cells trimmed, columns joined " | ".

    With >= 2 content rows the first row is the header (never a record) and
    rows whose width differs are reported with their line numbers and
    skipped — never reshaped or silently dropped. A single content row is
    treated as data (no header). Blank lines/blank-cell rows are emitted as
    "" so the caller can count them as empty records.
    """
    rows: list[tuple[int, list[str]]] = []   # (line number, stripped cells)
    empties = 0
    for lineno, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        cells = [c.strip() for c in row]
        if not row or not any(cells):
            empties += 1
        else:
            rows.append((lineno, cells))

    records: list[str] = [""] * empties
    if not rows:
        return records, None

    header_mode = len(rows) > 1
    header = rows[0][1] if header_mode else None
    for lineno, cells in (rows[1:] if header_mode else rows):
        if header_mode and len(cells) != len(header):
            errors.append(f"row {lineno}: expected {len(header)} columns, got {len(cells)}")
            continue
        records.append(" | ".join(cells))
    return records, header


def _extract_json(text: str, errors: list[str]) -> list[str]:
    """JSON: arrays of records, dicts with a ``text`` field, nested lists.

    Strings are used as-is; a dict with a ``text`` key yields that text
    (other keys are not duplicated into records); dicts without ``text``
    are stored as compact JSON document records; anything else is reported
    per-element (never silently discarded).
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"malformed JSON: {exc.msg} at line {exc.lineno} col {exc.colno}")
        return []

    out: list[str] = []

    def walk(o: Any, depth: int, where: str) -> None:
        if depth > _JSON_DEPTH_LIMIT:
            errors.append(f"{where}: nesting deeper than {_JSON_DEPTH_LIMIT} levels")
            return
        if isinstance(o, str):
            out.append(o.strip())  # may be "" -> counted as an empty record
        elif isinstance(o, list):
            for i, item in enumerate(o):
                walk(item, depth + 1, f"{where}[{i}]")
        elif isinstance(o, dict):
            if "text" in o:
                t = o["text"]
                if isinstance(t, str):
                    if t.strip():
                        out.append(t.strip())
                elif isinstance(t, list):
                    walk(t, depth + 1, f"{where}.text")
                else:
                    errors.append(f"{where}.text: expected string or list of strings")
            else:
                out.append(json.dumps(o, ensure_ascii=False, sort_keys=True))  # document record
        else:
            errors.append(f"{where}: unsupported element type {type(o).__name__}")

    walk(obj, 0, "$")
    return out


def ingest_file(filename: str, data: bytes) -> tuple[SourceFileInfo, list[str]]:
    """Decode + extract candidate records from one file.

    Never raises for content problems: failures are reported on the returned
    SourceFileInfo (status="error" or error entries) so a mixed upload yields
    a useful per-file report instead of failing opaquely.
    """
    info = SourceFileInfo(
        filename=filename,
        extension=Path(filename).suffix.lower(),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        status="ok",
    )
    errors: list[str] = []

    if info.extension not in SUPPORTED_EXTENSIONS:
        info.status, info.error = "error", (
            f"unsupported extension '{info.extension or '(none)'}'; supported: .txt .md .csv .json")
        return info, []
    if not data:
        info.status, info.error = "error", "file is empty (0 bytes)"
        return info, []

    try:
        text = data.decode("utf-8-sig" if info.extension == ".csv" else "utf-8")
    except UnicodeDecodeError as exc:
        info.status, info.error = "error", f"invalid UTF-8 at byte {exc.start}"
        return info, []
    text = normalize_text(text)

    columns: Optional[list[str]] = None
    if info.extension in (".txt", ".md"):
        records = _extract_paragraphs(text)
    elif info.extension == ".csv":
        records, columns = _extract_csv(text, errors)
    else:
        records = _extract_json(text, errors)

    info.record_count = len(records)
    info.error_count = len(errors)
    info.columns = columns
    if errors:
        shown = "; ".join(errors[:3])
        more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
        info.error = shown + more
        if not records:
            info.status = "error"  # e.g. malformed JSON with no usable records
    elif not records:
        info.error = "no text records found in file"
    return info, records


# =========================================================================== #
# Dataset engine
# =========================================================================== #


class DatasetEngine:
    def __init__(self, storage: Storage):
        self.storage = storage

    # ------------------------------------------------------------ registry

    def _meta_path(self, dataset_id: str) -> Path:
        return self.storage.dataset_dir(dataset_id) / DATASET_META

    def _version_dir(self, dataset_id: str, version: int) -> Path:
        return self.storage.dataset_dir(dataset_id) / f"v{version}"

    def _index_path(self, dataset_id: str) -> Path:
        return self.storage.dataset_dir(dataset_id) / INDEX_FILE

    def load_meta(self, dataset_id: str) -> DatasetInfo:
        path = self._meta_path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"dataset '{dataset_id}' not found")
        return DatasetInfo(**read_json(path))

    def _all_dataset_ids(self) -> list[str]:
        root = self.storage.datasets_dir
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))

    def list(self) -> list[DatasetInfo]:
        infos = []
        for ds_id in self._all_dataset_ids():
            try:
                infos.append(self.load_meta(ds_id))
            except Exception:
                log.warning("skipping unreadable dataset dir %s", ds_id)
        infos.sort(key=lambda d: d.created_at)
        return infos

    def version_manifest(self, dataset_id: str, version: int) -> VersionManifest:
        path = self._version_dir(dataset_id, version) / VERSION_MANIFEST
        if not path.exists():
            raise FileNotFoundError(f"dataset '{dataset_id}' version {version} not found")
        return VersionManifest(**read_json(path))

    def get(self, dataset_id: str) -> dict[str, Any]:
        """Full read view: dataset pointer + every version manifest + tokenized summaries."""
        meta = self.load_meta(dataset_id)
        versions = []
        for version in meta.versions:
            vdir = self._version_dir(dataset_id, version)
            manifest = self.version_manifest(dataset_id, version)
            view = manifest.model_dump(mode="json")
            tok_root = vdir / TOKENIZED_DIR
            view["tokenized"] = []
            if tok_root.exists():
                for tdir in sorted(tok_root.iterdir()):
                    if tdir.is_dir():
                        try:
                            view["tokenized"].append(read_json(tdir / VERSION_MANIFEST))
                        except Exception:
                            log.warning("unreadable tokenized artifact %s", tdir)
            versions.append(view)
        return {"dataset": meta.model_dump(mode="json"), "versions": versions}

    # ---------------------------------------------------------- ingestion

    def create_or_append(
        self,
        files: list[tuple[str, bytes]],
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ingest files -> new dataset (v1) or an appended version of an existing one."""
        from .schemas import NAME_PATTERN

        if not files:
            raise ValueError("no files provided")
        if len({fn for fn, _ in files}) != len(files):
            raise ValueError("duplicate filenames in the same upload")

        # 1) per-file extraction (errors are reported per file, never fatal)
        file_infos: list[SourceFileInfo] = []
        candidates: list[str] = []
        for filename, data in files:
            info, records = ingest_file(filename, data)
            file_infos.append(info)
            if info.status == "ok":
                candidates.extend(records)

        failed = [f"{f.filename}: {f.error}" for f in file_infos if f.status == "error"]
        if failed and not candidates:
            raise ValueError("all files failed: " + " | ".join(failed))

        # 2) resolve target: append vs fresh dataset (name rules on create)
        existing = self.load_meta(dataset_id) if dataset_id else None
        if existing is not None:
            version = (existing.versions[-1] if existing.versions else 0) + 1
        else:
            version = 1
            if name is None:
                stem = Path(file_infos[0].filename).stem
                name = re.sub(r"[^A-Za-z0-9 ._-]", "_", stem)[:64] or f"dataset-{uuid.uuid4().hex[:8]}"
            if not re.fullmatch(NAME_PATTERN, name or ""):
                raise ValueError(
                    f"invalid dataset name {name!r}: 1-64 chars of A-Za-z0-9, space, '.', '_', '-'")
            for other in self.list():
                if other.name == name:
                    raise ValueError(
                        f"a dataset named '{name}' already exists (id {other.id}); "
                        "pass its dataset_id to append a new version instead")

        # 3) drop empties + dedupe within this upload (insertion order kept)
        non_empty = [t for t in candidates if t.strip()]
        stored: list[tuple[str, str]] = []  # (record_id, text) in first-seen order
        seen: set[str] = set()
        for text in non_empty:
            rid = record_id_of(text)
            if rid not in seen:
                seen.add(rid)
                stored.append((rid, text))
        empty_count = len(candidates) - len(non_empty)
        duplicate_count = len(non_empty) - len(stored)

        # 4) cross-version duplicate report (versions stay self-contained)
        previous = self._load_index(dataset_id) if existing else set()
        dup_previous = sum(1 for rid, _ in stored if rid in previous)

        if not stored:
            raise ValueError("no usable records extracted")

        # 5) quality stats + deterministic splits
        quality, split_buckets = self._stats_and_splits(stored)
        splits = {s: SplitInfo(count=len(ids), sha256=split_hash(ids)) for s, ids in split_buckets.items()}
        counts = CountsInfo(
            input_record_count=sum(f.record_count for f in file_infos),
            empty_record_count=empty_count,
            invalid_record_count=sum(f.error_count for f in file_infos),
            duplicate_record_count=duplicate_count,
            duplicate_of_previous_versions=dup_previous,
            unique_record_count=len(stored),
        )

        # 6) write the immutable version snapshot (atomic)
        ds_id = existing.id if existing else uuid.uuid4().hex[:12]
        vdir = self._version_dir(ds_id, version)
        if vdir.exists() and not (vdir / VERSION_MANIFEST).exists():
            # orphan from a previously interrupted write; safe to clear
            import shutil

            shutil.rmtree(vdir)
        if vdir.exists():
            raise RuntimeError(f"version directory {vdir} already exists (internal error)")
        vdir.mkdir(parents=True, exist_ok=False)
        rows = [{"id": rid, "text": text} for rid, text in stored]
        atomic_write_bytes(vdir / RECORDS_FILE, _gz_jsonl(rows))
        manifest = VersionManifest(
            dataset_id=ds_id,
            version=version,
            created_at=datetime.now(timezone.utc),
            sources=file_infos,
            counts=counts,
            quality=quality,
            splits=splits,
        )
        atomic_write_json(vdir / VERSION_MANIFEST, manifest.model_dump(mode="json"))

        # 7) update the mutable pointer + dedup index (old versions untouched)
        now = datetime.now(timezone.utc)
        if existing:
            meta = existing.model_copy(update={
                "updated_at": now,
                "versions": [*existing.versions, version],
                "latest_version": version,
                "total_records": len(stored),
            })
        else:
            meta = DatasetInfo(
                id=ds_id, name=name or "", created_at=now, updated_at=now,
                versions=[1], latest_version=1, total_records=len(stored),
            )
        atomic_write_json(self._meta_path(ds_id), meta.model_dump(mode="json"))
        self._append_index(ds_id, [rid for rid, _ in stored])

        log.info("dataset %s v%s: %s unique records (%s input, %s empty, %s dup, %s prev-dup)",
                 ds_id, version, len(stored), counts.input_record_count,
                 empty_count, duplicate_count, dup_previous)
        return {
            "dataset_id": ds_id,
            "name": meta.name,
            "version": version,
            "counts": counts.model_dump(mode="json"),
            "quality": quality.model_dump(mode="json"),
            "splits": {k: v.model_dump(mode="json") for k, v in splits.items()},
            "files": [f.model_dump(mode="json") for f in file_infos],
        }

    @staticmethod
    def _stats_and_splits(stored: list[tuple[str, str]]) -> tuple[QualityInfo, dict[str, list[str]]]:
        lengths = [len(text) for _, text in stored]
        quality = QualityInfo(
            total_characters=sum(lengths),
            total_bytes=sum(len(text.encode("utf-8")) for _, text in stored),
            min_length=min(lengths),
            max_length=max(lengths),
            mean_length=round(statistics.fmean(lengths), 2),
            median_length=float(statistics.median(lengths)),
        )
        buckets: dict[str, list[str]] = {s: [] for s in SPLITS}
        for rid, _ in stored:
            buckets[split_of(rid)].append(rid)
        return quality, buckets

    # ------------------------------------------------------- dedup index

    def _load_index(self, dataset_id: str) -> set[str]:
        path = self._index_path(dataset_id)
        if not path.exists():
            return set()
        with open(path, encoding="ascii") as fh:
            return {line.strip() for line in fh if line.strip()}

    def _append_index(self, dataset_id: str, record_ids: list[str]) -> None:
        merged = sorted(self._load_index(dataset_id) | set(record_ids))
        atomic_write_bytes(self._index_path(dataset_id), ("\n".join(merged) + "\n").encode("ascii"))

    # ------------------------------------------------------------ reading

    def iter_version_records(self, dataset_id: str, version: int) -> Iterator[dict[str, str]]:
        """Stream stored records of one version in order (dicts with id/text)."""
        path = self._version_dir(dataset_id, version) / RECORDS_FILE
        if not path.exists():
            raise FileNotFoundError(f"dataset '{dataset_id}' version {version} has no records file")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)

    # ---------------------------------------------------------- tokenizing

    def tokenized_artifact(self, dataset_id: str, version: int, tokenizer_id: str
                           ) -> tuple[TokenizedInfo, dict[str, Path]]:
        """Resolve the tokenized artifact for (dataset, version, tokenizer).

        Returns (TokenizedInfo, {split: path-to-bin}) or raises with an
        actionable message when the artifact does not exist.
        """
        meta = self.load_meta(dataset_id)
        if version not in meta.versions:
            raise FileNotFoundError(
                f"dataset '{dataset_id}' has no version {version} (available: {meta.versions})")
        tdir = self._version_dir(dataset_id, version) / TOKENIZED_DIR / tokenizer_id
        mpath = tdir / VERSION_MANIFEST
        if not mpath.exists():
            raise FileNotFoundError(
                f"dataset '{dataset_id}' version {version} has no tokenized artifact for "
                f"tokenizer '{tokenizer_id}' — run POST /datasets/{{id}}/tokenize first")
        info = TokenizedInfo(**read_json(mpath))
        paths = {s: tdir / f"{s}.bin" for s in SPLITS}
        missing = [s for s, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"tokenized artifact for tokenizer '{tokenizer_id}' is missing split bins: {missing}")
        return info, paths

    def tokenize(
        self,
        dataset_id: str,
        tokenizer_id: str,
        version: Optional[int] = None,
    ) -> dict[str, Any]:
        """Tokenize one version with a stored tokenizer -> v<N>/tokenized/<tok_id>/.

        Bins are flat little-endian arrays: uint16 when the actual vocabulary
        fits (<= 65536, always true at this milestone's cap), else uint32.
        Each split's bin stores its token count + sha256 in the tokenized
        manifest. Re-running with the same tokenizer deterministically
        rewrites identical bytes (harmless); version files are never touched.
        """
        from .tokenizer import TokenizerEngine

        tok_engine = TokenizerEngine(self.storage)
        record = tok_engine.load(tokenizer_id)
        meta = self.load_meta(dataset_id)
        if version is None:
            version = meta.latest_version
        if version not in meta.versions:
            raise FileNotFoundError(
                f"dataset '{dataset_id}' has no version {version} (available: {meta.versions})")
        self.version_manifest(dataset_id, version)  # raises when the dir is gone

        tokenizer = tok_engine.get_hf(tokenizer_id)
        dtype = np.uint16 if record.actual_vocab_size <= 2**16 else np.uint32

        buckets: dict[str, list[str]] = {s: [] for s in SPLITS}
        for row in self.iter_version_records(dataset_id, version):
            buckets[split_of(row["id"])].append(row["text"])

        tdir = self._version_dir(dataset_id, version) / TOKENIZED_DIR / tokenizer_id
        tdir.mkdir(parents=True, exist_ok=True)
        split_infos: dict[str, SplitInfo] = {}
        for split, texts in buckets.items():
            ids: list[int] = []
            for chunk_start in range(0, len(texts), 512):
                encodings = tokenizer.encode_batch(texts[chunk_start:chunk_start + 512],
                                                   add_special_tokens=False)
                for enc in encodings:
                    ids.extend(enc.ids)
            data = np.asarray(ids, dtype=dtype).tobytes()
            atomic_write_bytes(tdir / f"{split}.bin", data)
            split_infos[split] = SplitInfo(count=len(ids), sha256=hashlib.sha256(data).hexdigest())

        info = TokenizedInfo(
            tokenizer_id=tokenizer_id,
            tokenizer_name=record.name,
            tokenizer_hash=record.tokenizer_hash,
            dtype=np.dtype(dtype).name,
            created_at=datetime.now(timezone.utc),
            splits=split_infos,
        )
        atomic_write_json(tdir / VERSION_MANIFEST, info.model_dump(mode="json"))
        log.info("tokenized dataset %s v%s with tokenizer %s: %s",
                 dataset_id, version, tokenizer_id,
                 {s: i.count for s, i in split_infos.items()})
        return {
            "dataset_id": dataset_id,
            "version": version,
            "tokenizer_id": tokenizer_id,
            "dtype": info.dtype,
            "splits": {k: v.model_dump(mode="json") for k, v in split_infos.items()},
        }

    # ------------------------------------------------------------- verify

    def verify(self, dataset_id: str) -> dict[str, Any]:
        """Deep integrity check over every version (and tokenized artifacts).

        Checks: manifest validity, record count, per-record hash, split
        recomputation vs manifest, tokenized size/hash/counts. Never repairs:
        any mismatch -> overall status "failed".
        """
        meta = self.load_meta(dataset_id)
        checks: list[dict[str, Any]] = []
        ok = True
        for version in meta.versions:
            vcheck: dict[str, Any] = {"version": version, "status": "ok", "errors": []}
            try:
                manifest = self.version_manifest(dataset_id, version)
            except Exception as exc:
                vcheck["status"] = "failed"
                vcheck["errors"].append(f"manifest unreadable: {exc}")
                ok = False
                checks.append(vcheck)
                continue

            count = 0
            split_counts = {s: 0 for s in SPLITS}
            seen_ids: set[str] = set()
            try:
                for row in self.iter_version_records(dataset_id, version):
                    count += 1
                    if row["id"] in seen_ids:
                        raise ValueError(f"duplicate record id at line {count}")
                    seen_ids.add(row["id"])
                    if record_id_of(row.get("text", "")) != row.get("id"):
                        raise ValueError(f"record hash mismatch at line {count}")
                    split_counts[split_of(row["id"])] += 1
            except Exception as exc:
                vcheck["status"] = "failed"
                vcheck["errors"].append(f"records unreadable/corrupt: {exc}")

            if vcheck["status"] == "ok":
                if count != manifest.counts.unique_record_count:
                    vcheck["errors"].append(
                        f"record count {count} != manifest {manifest.counts.unique_record_count}")
                for split in SPLITS:
                    got = split_counts[split]
                    if got != manifest.splits[split].count:
                        vcheck["errors"].append(
                            f"split '{split}': recomputed {got} != manifest {manifest.splits[split].count}")
                if sum(manifest.splits[s].count for s in SPLITS) != manifest.counts.unique_record_count:
                    vcheck["errors"].append("split counts do not sum to the record count")
                if vcheck["errors"]:
                    vcheck["status"] = "failed"

            tok_root = self._version_dir(dataset_id, version) / TOKENIZED_DIR
            if tok_root.exists():
                vcheck["tokenized"] = []
                for tdir in sorted(tok_root.iterdir()):
                    if tdir.is_dir():
                        tcheck = self._verify_tokenized(tdir)
                        vcheck["tokenized"].append(tcheck)
                        if tcheck["status"] != "ok":
                            ok = False
            if vcheck["status"] != "ok":
                ok = False
            checks.append(vcheck)

        return {"dataset_id": dataset_id, "status": "ok" if ok else "failed", "versions": checks}

    @staticmethod
    def _verify_tokenized(tdir: Path) -> dict[str, Any]:
        """Checks are self-sufficient: manifest + bins, no tokenizer needed."""
        tcheck: dict[str, Any] = {"tokenizer_id": tdir.name, "status": "ok", "errors": []}
        try:
            info = TokenizedInfo(**read_json(tdir / VERSION_MANIFEST))
            nbytes = {"uint16": 2, "uint32": 4}.get(info.dtype)
            if nbytes is None:
                tcheck["errors"].append(f"unknown dtype {info.dtype!r}")
                nbytes = 0
            for split, sinfo in info.splits.items():
                bin_path = tdir / f"{split}.bin"
                if not bin_path.exists():
                    tcheck["errors"].append(f"{split}.bin missing")
                    continue
                size = bin_path.stat().st_size
                if size != sinfo.count * nbytes:
                    tcheck["errors"].append(
                        f"{split}.bin size {size} != {sinfo.count} tokens x {nbytes} bytes")
                if hash_file_sha256(bin_path) != sinfo.sha256:
                    tcheck["errors"].append(f"{split}.bin sha256 mismatch")
        except Exception as exc:
            tcheck["errors"].append(f"tokenized manifest unreadable: {exc}")
        if tcheck["errors"]:
            tcheck["status"] = "failed"
        return tcheck

    # ----------------------------------------------------------- deletion

    def delete(self, dataset_id: str) -> tuple[int, int]:
        """Remove ONE dataset's directory ATOMICALLY (M65 low-level
        primitive, the ``remove_checkpoint`` pattern): the dataset's
        whole own tree (meta + every version + records + tokenized
        artifacts) disappears in ONE ``os.rename`` to a hidden
        ``.tmp-delete-*`` sibling the registry scans skip, so no
        observer ever sees a half-deleted dataset. Returns
        ``(files_removed, bytes_reclaimed)`` measured from the files as
        they existed immediately before removal. The CALLER (the forge
        facade) owns the safety decision — the FULL M64 reference
        analysis must have passed before this is called, because
        deletion must never become a way to orphan referencing
        evidence."""
        if not self._meta_path(dataset_id).exists():
            raise FileNotFoundError(f"dataset '{dataset_id}' not found")
        ddir = self.storage.dataset_dir(dataset_id)
        files = sorted(p for p in ddir.rglob("*") if p.is_file())
        nbytes = sum(p.stat().st_size for p in files)
        atomic_delete_dir(ddir)
        log.info("deleted dataset %s (%d files, %d bytes)",
                 dataset_id, len(files), nbytes)
        return len(files), nbytes


def _referencing_tokenizers(storage: Storage, dataset_id: str) -> list[str]:
    """Lightweight dependency scan (tokenizers whose training source is this dataset)."""
    root = storage.tokenizers_dir
    if not root.exists():
        return []
    refs: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        mpath = d / VERSION_MANIFEST
        if mpath.exists():
            try:
                if read_json(mpath).get("trained_on_dataset_id") == dataset_id:
                    refs.append(d.name)
            except Exception:
                continue
    return refs


# =========================================================================== #
# Helpers shared with the tokenizer engine
# =========================================================================== #


def _gz_jsonl(rows: list[dict[str, Any]]) -> bytes:
    """Deterministic gzip (mtime=0) JSONL bytes -> reproducible artifacts."""
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0) as fh:
        for row in rows:
            fh.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return bio.getvalue()


def extract_records_for_training(files: list[tuple[str, bytes]]) -> list[str]:
    """Files -> unique normalized records (same rules as dataset ingestion).

    Used by the tokenizer engine so tokenizer corpora and dataset corpora
    follow identical extraction semantics.
    """
    candidates: list[str] = []
    for filename, data in files:
        info, records = ingest_file(filename, data)
        if info.status == "ok":
            candidates.extend(records)
    stored: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        if not text.strip():
            continue
        rid = record_id_of(text)
        if rid not in seen:
            seen.add(rid)
            stored.append(text)
    return stored
