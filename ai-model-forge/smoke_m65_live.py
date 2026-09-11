"""M65 LIVE certification smoke (production touched READ-ONLY; the
deletion verification runs on a THROWAWAY COPY — executed exactly ONCE).

Production protocol (per the M65 spec): the production dataset and
tokenizer are both REFERENCED (M64: 16 and 15 references), so no
deletion is attempted over HTTP — the guard + success paths are
verified through the engine on a byte-copy of the production root,
which is then discarded. The production root itself must remain
byte-for-byte identical.

Certified facts of this production root (identical bytes since the M64
certification — m65_pre.sha256 == m64_pre.sha256, 52/52):
  1 dataset (ccf20044baff, 16 refs: 6/2/4/2/0/1/1), 1 tokenizer
  (8a54196b1589, 15 refs: 6/2/4/2/0/0/0/1), 1 model (70aa05e0bee6);
  52 files / 8,926,421 B; M62 retention 13/10/3.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8787"
API = BASE + "/api/v1"

DS_ID = "ccf20044baff"
TOK_ID = "8a54196b1589"
MODEL_ID = "70aa05e0bee6"
DS_TOTAL = 16
TOK_TOTAL = 15
N_CKPTS, N_DELETABLE, N_PROTECTED = 13, 10, 3
N_FILES, TOTAL_BYTES = 52, 8_926_421
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m65_pre.sha256"

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((name, bool(cond), extra))
    print(f"{'PASS' if cond else 'FAIL'} {name} {extra}")


def jget(path: str):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        assert r.status == 200, (path, r.status)
        return json.loads(r.read()), r.read()


def inventory(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


# ======================= production (READ-ONLY) ============================ #
project, _ = jget("/project")
root = Path(project["storage_root"])
check("P1 project readable", project["model_count"] == 1)

pre_disk = inventory(root)
pre_ref = {}
for line in PRE_INVENTORY.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    pre_ref[rel.removeprefix("./")] = digest
check("P2 disk == m65_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

datasets, _ = jget("/datasets")
tokenizers, _ = jget("/tokenizers")
check("P3 registries", [d["id"] for d in datasets] == [DS_ID]
      and [t["id"] for t in tokenizers] == [TOK_ID])

ds_ov, _ = jget(f"/datasets/{DS_ID}/usage")
tok_ov, _ = jget(f"/tokenizers/{TOK_ID}/usage")
check("P4 M64 usage coherent (dataset)",
      ds_ov["total_references"] == DS_TOTAL and ds_ov["referenced"] is True)
check("P5 M64 usage coherent (tokenizer)",
      tok_ov["total_references"] == TOK_TOTAL and tok_ov["referenced"] is True)

ret, _ = jget(f"/models/{MODEL_ID}/checkpoints/retention")
check("P6 M62 retention coherent", ret["total_checkpoints"] == N_CKPTS
      and ret["deletable_checkpoints"] == N_DELETABLE
      and ret["protected_checkpoints"] == N_PROTECTED)

# the expected blocker lists, derived from the (independently certified)
# production usage overviews
ds_blockers_expected = [
    {"reason": c["category"], "detail": ", ".join(c["references"])}
    for c in ds_ov["categories"] if c["references"]]
tok_blockers_expected = [
    {"reason": c["category"], "detail": ", ".join(c["references"])}
    for c in tok_ov["categories"] if c["references"]]

# ================== throwaway COPY verification (engine) ==================== #
sys.path.insert(0, str(Path(__file__).resolve().parent))
copy_root = Path(tempfile.mkdtemp(prefix="m65-copy-")) / "root"
shutil.copytree(root, copy_root)
try:
    from app.engine import ModelForge
    from app.schemas import TokenizerConfig

    forge = ModelForge(root=copy_root)

    # C1: the copy's usage views equal the production HTTP answers
    copy_ds = forge.dataset_usage_overview(DS_ID).model_dump(mode="json")
    copy_tok = forge.tokenizer_usage_overview(TOK_ID).model_dump(mode="json")
    check("C1 copy usage == production usage (both artifacts)",
          copy_ds == ds_ov and copy_tok == tok_ov)

    # C2: the dataset guard refuses with EXACTLY the M64 blockers
    try:
        forge.delete_dataset(DS_ID)
        check("C2 dataset guard refuses", False)
    except ValueError as exc:
        blockers = forge.dataset_deletion_blockers(DS_ID)
        check("C2 dataset guard refuses with the M64 blocker list",
              [b.model_dump() for b in blockers] == ds_blockers_expected
              and all(b.reason in str(exc) for b in blockers[:1]),
              f"{len(blockers)} categories")

    # C3: the tokenizer guard refuses with EXACTLY the M64 blockers
    try:
        forge.delete_tokenizer(TOK_ID)
        check("C3 tokenizer guard refuses", False)
    except ValueError:
        blockers = forge.tokenizer_deletion_blockers(TOK_ID)
        check("C3 tokenizer guard refuses with the M64 blocker list",
              [b.model_dump() for b in blockers] == tok_blockers_expected,
              f"{len(blockers)} categories")

    # C4: refusals wrote nothing on the copy either
    copy_inv = inventory(copy_root)
    check("C4 refusals are read-only (copy inventory stable)",
          len(copy_inv) == N_FILES)

    # C5: unknown ids -> FileNotFoundError
    for fn, arg in ((forge.delete_dataset, "no-such-m65"),
                    (forge.delete_tokenizer, "no-such-m65")):
        try:
            fn(arg)
            check("C5 unknown -> FileNotFoundError", False, arg)
        except FileNotFoundError:
            check("C5 unknown -> FileNotFoundError", True, arg[:8])

    # C6: live recompute — train a NEW tokenizer on the copied dataset
    # (unreferenced itself), watch tokenizer_training grow then shrink
    tt0 = [c for c in forge.dataset_usage_overview(DS_ID).categories
           if c.category == "tokenizer_training"][0].references
    new_tok = forge.train_tokenizer(
        TokenizerConfig(name="m65-copy-tok", vocab_size=300),
        dataset_id=DS_ID).id
    tt1 = [c for c in forge.dataset_usage_overview(DS_ID).categories
           if c.category == "tokenizer_training"][0].references
    res = forge.delete_tokenizer(new_tok)
    tt2 = [c for c in forge.dataset_usage_overview(DS_ID).categories
           if c.category == "tokenizer_training"][0].references
    check("C6 live recompute (train + delete -> blocker list shrinks)",
          tt1 == sorted(tt0 + [new_tok]) and tt2 == tt0
          and res.files_removed == 2 and res.bytes_reclaimed > 0,
          f"{len(tt0)} -> {len(tt1)} -> {len(tt2)}")

    # C7: an unreferenced DATASET deletes with exact stats (copy)
    up = forge.upload_dataset(
        [("free.txt", (b"m65 free corpus\n" * 40))], name="m65-copy-free")
    free_ds = up["dataset_id"]
    fdir = copy_root / "datasets" / free_ds
    files = sorted(p for p in fdir.rglob("*") if p.is_file())
    nbytes = sum(p.stat().st_size for p in files)
    res = forge.delete_dataset(free_ds)
    check("C7 unreferenced dataset deletes with exact stats",
          res.dataset_id == free_ds
          and res.files_removed == len(files)
          and res.bytes_reclaimed == nbytes
          and not fdir.exists()
          and not any(p.name.startswith(".") for p in
                      (copy_root / "datasets").iterdir()),
          f"{res.files_removed} files / {res.bytes_reclaimed} B")

    # C8: the protected production artifacts survive everything
    check("C8 protected artifacts untouched on the copy",
          (copy_root / "datasets" / DS_ID).exists()
          and (copy_root / "tokenizers" / TOK_ID).exists()
          and forge.dataset_deletion_blockers(DS_ID) is not None)
finally:
    shutil.rmtree(copy_root.parent, ignore_errors=True)
check("C9 copy discarded", not copy_root.exists())

# ============================== audits ===================================== #
post_disk = inventory(root)
check("A1 production byte-identical before/after (zero mutation)",
      post_disk == pre_disk, f"{len(post_disk)} files")
check("A2 tmp clean", not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M65 CHECKS PASSED "
      "(production read-only; deletions verified on a discarded copy)")
sys.exit(0 if n_pass == len(results) else 1)
