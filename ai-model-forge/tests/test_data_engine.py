"""Milestone 2 tests: dataset + tokenizer engines (library level)."""
from __future__ import annotations

import gzip
import json

import pytest
from pydantic import ValidationError

from app.dataset import record_id_of, split_of
from app.schemas import TokenizerConfig, TokenizerRecord


def corpus_sentences(n: int = 200) -> list[str]:
    return [
        f"the quick brown fox jumps over the lazy dog number {i} and runs far away into the woods today"
        for i in range(n)
    ] + ["hello world, this is a byte level bpe tokenizer test " * 3]


def corpus_bytes() -> bytes:
    return ("\n\n".join(corpus_sentences()) + "\n").encode("utf-8")


# =========================================================================== #
# Tokenizer engine
# =========================================================================== #


def test_tokenizer_training_works_and_vocab_respected(forge):
    cfg = TokenizerConfig(name="tkn-a", vocab_size=320)
    rec = forge.train_tokenizer(cfg, files=[("corpus.txt", corpus_bytes())])
    assert isinstance(rec, TokenizerRecord)
    assert rec.actual_vocab_size == 320  # corpus rich enough to fill the cap
    assert rec.actual_vocab_size <= rec.requested_vocab_size
    vocab = forge.tokenizers.get_hf(rec.id).get_vocab()
    assert all(s in vocab for s in ["<unk>", "<s>", "</s>"])
    assert rec.trained_on_dataset_id is None
    assert rec.tokenizer_hash and len(rec.tokenizer_hash) == 64


def test_tokenizer_deterministic_across_runs(forge):
    """Same corpus + config -> byte-identical tokenizer artifacts (different names)."""
    r1 = forge.train_tokenizer(TokenizerConfig(name="tkn-det-a", vocab_size=320),
                               files=[("corpus.txt", corpus_bytes())])
    r2 = forge.train_tokenizer(TokenizerConfig(name="tkn-det-b", vocab_size=320),
                               files=[("corpus.txt", corpus_bytes())])
    assert r1.tokenizer_hash == r2.tokenizer_hash
    v1 = forge.tokenizers.get_hf(r1.id).get_vocab()
    v2 = forge.tokenizers.get_hf(r2.id).get_vocab()
    assert v1 == v2


def test_tokenizer_from_dataset_records_training_source(forge):
    up = forge.upload_dataset([("data.txt", corpus_bytes())], name="ds-for-tok")
    cfg = TokenizerConfig(name="tkn-from-ds", vocab_size=256)
    rec = forge.train_tokenizer(cfg, dataset_id=up["dataset_id"])
    assert rec.trained_on_dataset_id == up["dataset_id"]


def test_tokenizer_encode_decode_roundtrip_multilingual(forge):
    cfg = TokenizerConfig(name="tkn-rt", vocab_size=512)
    rec = forge.train_tokenizer(cfg, files=[("corpus.txt", corpus_bytes())])
    tok = forge.tokenizers.get_hf(rec.id)
    samples = [
        "the quick brown fox",                      # plain
        "héllo wörld — café, déjà vu!",             # accents + punctuation
        "你好，世界。Привет мир! Ελληνικά தமிழ் مرحبا",  # multilingual
        "emoji 🎉🎊 and symbols ©®™",               # emoji/symbols
        "line one\nline two\tindented",             # standard whitespace
        "x" * 1000,                                  # long run
        "quotes 'single' \"double\" (parens) [brackets]",  # quotes/brackets
    ]
    for text in samples:
        assert tok.decode(tok.encode(text).ids) == text


def test_tokenizer_roundtrip_all_corpus_records(forge):
    cfg = TokenizerConfig(name="tkn-rt2", vocab_size=512)
    rec = forge.train_tokenizer(cfg, files=[("corpus.txt", corpus_bytes())])
    tok = forge.tokenizers.get_hf(rec.id)
    for sentence in corpus_sentences():
        assert tok.decode(tok.encode(sentence).ids) == sentence


@pytest.mark.parametrize("vocab", [100, 70_000])
def test_tokenizer_invalid_vocab_size_rejected(forge, vocab):
    with pytest.raises(ValidationError):
        TokenizerConfig(name="tkn-bad", vocab_size=vocab)


def test_tokenizer_special_token_collision_rejected():
    with pytest.raises(ValidationError, match="unique"):
        TokenizerConfig(name="tkn-collide", vocab_size=300,
                        special_tokens=["<unk>", "<unk>"])


def test_tokenizer_special_token_whitespace_rejected():
    with pytest.raises(ValidationError, match="whitespace"):
        TokenizerConfig(name="tkn-ws", vocab_size=300, special_tokens=["a b"])


def test_tokenizer_duplicate_name_rejected(forge):
    cfg = TokenizerConfig(name="tkn-dupname", vocab_size=300)
    forge.train_tokenizer(cfg, files=[("corpus.txt", corpus_bytes())])
    with pytest.raises(ValueError, match="already exists"):
        forge.train_tokenizer(cfg, files=[("corpus.txt", corpus_bytes())])


def test_tokenizer_delete(forge):
    cfg = TokenizerConfig(name="tkn-del", vocab_size=300)
    rec = forge.train_tokenizer(cfg, files=[("corpus.txt", corpus_bytes())])
    forge.delete_tokenizer(rec.id)
    assert forge.list_tokenizers() == []
    with pytest.raises(FileNotFoundError):
        forge.get_tokenizer(rec.id)


# =========================================================================== #
# Dataset ingestion formats
# =========================================================================== #


def test_ingest_all_formats(forge):
    files = [
        ("a.txt", "alpha one\n\nalpha two\n".encode()),
        ("b.md", "# Heading\n\nsome **bold** text\n".encode()),
        ("c.csv", "name,qty\napple,3\npear,5\n".encode()),
        ("d.json", json.dumps({"text": ["hello json", {"text": "nested one"}]}).encode()),
    ]
    up = forge.upload_dataset(files, name="mix-ds")
    assert up["version"] == 1
    assert up["counts"]["input_record_count"] == 8   # 2+2+2+2
    assert up["counts"]["empty_record_count"] == 0
    assert up["counts"]["invalid_record_count"] == 0
    assert up["counts"]["duplicate_record_count"] == 0
    assert up["counts"]["unique_record_count"] == 8
    assert up["files"][2]["columns"] == ["name", "qty"]  # csv header recorded
    rows = list(forge.datasets.iter_version_records(up["dataset_id"], 1))
    texts = [r["text"] for r in rows]
    assert "alpha one" in texts and "alpha two" in texts
    assert "some **bold** text" in texts
    assert "apple | 3" in texts and "pear | 5" in texts   # csv header excluded
    assert "hello json" in texts and "nested one" in texts
    assert all(r["id"] == record_id_of(r["text"]) for r in rows)


def test_ingest_dedup_and_empty_and_invalid(forge):
    files = [
        ("dup.txt", "shared record\n\nunique one\n".encode()),
        ("dup2.txt", "shared record\n\nunique two\n".encode()),
        ("empties.json", json.dumps(["a", "", "   ", {"text": "b"}, 42]).encode()),
        ("ragged.csv", "h1,h2\nx,y\nbroken,row,here\nz,9\n".encode()),
    ]
    up = forge.upload_dataset(files, name="dedup-ds")
    counts = up["counts"]
    # txt 2+2; json ["a","","",{text:b},42] -> 4 records + 1 error (42);
    # csv: header h1,h2 excluded; ragged row reported+skipped; rows x|y, z|9 kept
    assert counts["input_record_count"] == 10    # 2+2+4+2 csv data rows
    assert counts["empty_record_count"] == 2
    assert counts["invalid_record_count"] == 2   # json element 42 + ragged csv row
    assert counts["duplicate_record_count"] == 1  # "shared record" twice
    assert counts["unique_record_count"] == 7     # 10 - 2 empty - 2 invalid - 1 dup
    assert up["files"][3]["error_count"] == 1     # ragged csv row reported per-file
    rows = list(forge.datasets.iter_version_records(up["dataset_id"], 1))
    assert len(rows) == 7
    texts = {r["text"] for r in rows}
    assert texts == {"shared record", "unique one", "unique two", "a", "b",
                     "x | y", "z | 9"}


def test_unsupported_and_empty_and_malformed_files(forge):
    with pytest.raises(ValueError, match="unsupported extension"):
        forge.upload_dataset([("x.pdf", b"%PDF-1.4 fake")], name="bad-ext")
    with pytest.raises(ValueError, match="empty"):
        forge.upload_dataset([("x.txt", b"")], name="empty-file")
    with pytest.raises(ValueError, match="invalid UTF-8"):
        forge.upload_dataset([("x.txt", b"\xff\xfe bad")], name="bad-utf8")
    # malformed JSON alone -> every file fails -> global error mentions it
    with pytest.raises(ValueError, match="malformed JSON"):
        forge.upload_dataset([("x.json", b"{not json")], name="bad-json")


def test_mixed_valid_and_invalid_upload_reports_per_file(forge):
    up = forge.upload_dataset(
        [("good.txt", "fine record\n".encode()), ("bad.pdf", b"%PDF-1.4")],
        name="mixed-ds",
    )
    statuses = {f["filename"]: f["status"] for f in up["files"]}
    assert statuses == {"good.txt": "ok", "bad.pdf": "error"}
    assert "unsupported extension" in up["files"][1]["error"]
    assert up["counts"]["unique_record_count"] == 1


def test_duplicate_filename_rejected(forge):
    with pytest.raises(ValueError, match="duplicate filenames"):
        forge.upload_dataset([("a.txt", b"one"), ("a.txt", b"two")], name="dupfn")


def test_source_file_hashes_and_quality_stats(forge):
    import hashlib

    content = ("The quick brown fox.\n" * 10).encode()  # one paragraph (no blank lines)
    up = forge.upload_dataset([("f.txt", content)], name="stats-ds")
    f = up["files"][0]
    assert f["sha256"] == hashlib.sha256(content).hexdigest()
    q = up["quality"]
    text = ("The quick brown fox.\n" * 10).strip()
    assert up["counts"]["unique_record_count"] == 1
    assert q["total_characters"] == len(text)
    assert q["max_length"] == q["min_length"] == q["mean_length"] == len(text)
    assert q["median_length"] == float(len(text))


# =========================================================================== #
# Splits
# =========================================================================== #


def test_splits_deterministic_exhaustive_no_overlap(forge):
    files = [("c.txt", ("\n\n".join(corpus_sentences(300)) + "\n").encode())]
    up = forge.upload_dataset(files, name="split-ds")
    rows = list(forge.datasets.iter_version_records(up["dataset_id"], 1))
    ids = [r["id"] for r in rows]
    got = {"train": [], "validation": [], "test": []}
    for rid in ids:
        got[split_of(rid)].append(rid)
    for split in ("train", "validation", "test"):
        assert up["splits"][split]["count"] == len(got[split])
    assert up["splits"]["train"]["count"] > 0
    assert up["splits"]["validation"]["count"] > 0
    assert up["splits"]["test"]["count"] > 0
    assert sum(len(v) for v in got.values()) == len(ids) == up["counts"]["unique_record_count"]
    # union == all, no overlap
    assert set(ids) == set(got["train"]) | set(got["validation"]) | set(got["test"])
    assert not (set(got["train"]) & set(got["validation"]))
    assert not (set(got["validation"]) & set(got["test"]))
    assert not (set(got["train"]) & set(got["test"]))
    # determinism: identical upload -> identical split assignment
    up2 = forge.upload_dataset(files, name="split-ds2")
    ids2 = [r["id"] for r in forge.datasets.iter_version_records(up2["dataset_id"], 1)]
    assert ids == ids2
    for split in ("train", "validation", "test"):
        assert up["splits"][split]["sha256"] == up2["splits"][split]["sha256"]


# =========================================================================== #
# Versioning
# =========================================================================== #

def _records_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def test_versioning_append_keeps_v1_untouched(forge):
    up1 = forge.upload_dataset([("a.txt", "record one\n\nrecord two\n".encode())], name="ver-ds")
    ds_id = up1["dataset_id"]
    v1_manifest_path = forge.datasets._version_dir(ds_id, 1) / "manifest.json"
    v1_records_path = forge.datasets._version_dir(ds_id, 1) / "records.jsonl.gz"
    m1_before = _records_bytes(v1_manifest_path)
    r1_before = _records_bytes(v1_records_path)

    up2 = forge.upload_dataset(
        [("b.txt", "record three\n\nrecord one\n".encode())],  # "record one" is a cross-version dup
        dataset_id=ds_id,
    )
    assert up2["version"] == 2
    assert up2["counts"]["unique_record_count"] == 2          # three + one (self-contained)
    assert up2["counts"]["duplicate_of_previous_versions"] == 1
    assert _records_bytes(v1_manifest_path) == m1_before      # v1 byte-identical
    assert _records_bytes(v1_records_path) == r1_before

    info = forge.get_dataset(ds_id)
    assert info["dataset"]["versions"] == [1, 2]
    assert info["dataset"]["latest_version"] == 2
    v2_records = list(forge.datasets.iter_version_records(ds_id, 2))
    assert {r["text"] for r in v2_records} == {"record three", "record one"}
    # v1 still fully readable & verifiable
    assert forge.verify_dataset(ds_id)["status"] == "ok"
    v1_records = list(forge.datasets.iter_version_records(ds_id, 1))
    assert {r["text"] for r in v1_records} == {"record one", "record two"}


def test_duplicate_dataset_name_rejected_and_append_without_id(forge):
    forge.upload_dataset([("a.txt", "x\n".encode())], name="named-ds")
    with pytest.raises(ValueError, match="already exists"):
        forge.upload_dataset([("b.txt", "y\n".encode())], name="named-ds")
    # different name -> separate dataset
    forge.upload_dataset([("c.txt", "z\n".encode())], name="named-ds-2")
    assert len(forge.list_datasets()) == 2


# =========================================================================== #
# Integrity verification (incl. tamper detection)
# =========================================================================== #


def test_verify_ok_then_failed_after_records_tamper(forge):
    up = forge.upload_dataset([("a.txt", ("\n\n".join(corpus_sentences(60))).encode())], name="verify-ds")
    ds_id = up["dataset_id"]
    assert forge.verify_dataset(ds_id)["status"] == "ok"

    rec_path = forge.datasets._version_dir(ds_id, 1) / "records.jsonl.gz"
    with gzip.open(rec_path, "rt", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert lines
    rows = [json.loads(x) for x in lines]
    rows[0]["text"] += " TAMPERED"
    with gzip.open(rec_path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = forge.verify_dataset(ds_id)
    assert report["status"] == "failed"
    assert any("hash mismatch" in e for e in report["versions"][0]["errors"])


def test_verify_failed_after_tokenized_bin_tamper(forge):
    up = forge.upload_dataset([("a.txt", ("\n\n".join(corpus_sentences(120)) + "\n").encode())],
                              name="toktamper-ds")
    cfg = TokenizerConfig(name="tok-tamper-tok", vocab_size=320)
    rec = forge.train_tokenizer(cfg, dataset_id=up["dataset_id"])
    forge.tokenize_dataset(up["dataset_id"], rec.id)
    assert forge.verify_dataset(up["dataset_id"])["status"] == "ok"

    bin_path = (forge.datasets._version_dir(up["dataset_id"], 1) / "tokenized" / rec.id / "train.bin")
    data = bytearray(bin_path.read_bytes())
    data[0] ^= 0xFF
    bin_path.write_bytes(bytes(data))

    report = forge.verify_dataset(up["dataset_id"])
    assert report["status"] == "failed"
    assert report["versions"][0]["tokenized"][0]["status"] == "failed"
    assert any("sha256" in e for e in report["versions"][0]["tokenized"][0]["errors"])


def test_delete_protected_while_tokenizer_trains_on_it(forge):
    up = forge.upload_dataset([("a.txt", ("\n\n".join(corpus_sentences(150))).encode())], name="prot-ds")
    ds_id = up["dataset_id"]
    cfg = TokenizerConfig(name="prot-tok", vocab_size=320)
    rec = forge.train_tokenizer(cfg, dataset_id=ds_id)
    with pytest.raises(ValueError, match="tokenizer_training"):
        forge.delete_dataset(ds_id)
    forge.delete_tokenizer(rec.id)
    forge.delete_dataset(ds_id)  # now allowed
    with pytest.raises(FileNotFoundError):
        forge.datasets.load_meta(ds_id)


# =========================================================================== #
# Tokenization
# =========================================================================== #

def _token_counts(hf_tok, texts):
    return sum(len(hf_tok.encode(t).ids) for t in texts)


def test_tokenize_artifact_dtype_counts_and_decode(forge):
    sentences = corpus_sentences(150)
    up = forge.upload_dataset([("a.txt", ("\n\n".join(sentences)).encode())], name="tok-ds")
    cfg = TokenizerConfig(name="tok-ds-tok", vocab_size=320)
    rec = forge.train_tokenizer(cfg, dataset_id=up["dataset_id"])
    hf = forge.tokenizers.get_hf(rec.id)

    result = forge.tokenize_dataset(up["dataset_id"], rec.id)
    assert result["dtype"] == "uint16"
    assert set(result["splits"]) == {"train", "validation", "test"}

    # independent expectation: group stored records by split, encode by hand
    rows = list(forge.datasets.iter_version_records(up["dataset_id"], 1))
    buckets = {"train": [], "validation": [], "test": []}
    for row in rows:
        buckets[split_of(row["id"])].append(row["text"])
    import numpy as np

    vdir = forge.datasets._version_dir(up["dataset_id"], 1) / "tokenized" / rec.id
    total = 0
    for split, texts in buckets.items():
        expected_tokens = _token_counts(hf, texts)
        assert result["splits"][split]["count"] == expected_tokens
        bin_path = vdir / f"{split}.bin"
        arr = np.fromfile(bin_path, dtype=np.uint16)
        assert len(arr) == expected_tokens
        total += expected_tokens
        # decoding the stored token stream reproduces the concatenated records
        assert hf.decode(arr.tolist()) == "".join(texts)
    assert total > 0
    assert forge.verify_dataset(up["dataset_id"])["status"] == "ok"


def test_tokenize_unknown_version_and_tokenizer(forge):
    up = forge.upload_dataset([("a.txt", b"hello world\n")], name="tok404-ds")
    cfg = TokenizerConfig(name="tok404-tok", vocab_size=300)
    rec = forge.train_tokenizer(cfg, files=[("a.txt", b"hello world\n")])
    with pytest.raises(FileNotFoundError):
        forge.tokenize_dataset(up["dataset_id"], rec.id, version=9)
    with pytest.raises(FileNotFoundError):
        forge.tokenize_dataset(up["dataset_id"], "does-not-exist")
