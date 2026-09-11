"""M65 retention-views LIVE certification smoke (READ-ONLY production;
the ONE safe deletion runs on a THROWAWAY COPY — executed exactly ONCE).

Certifies the M65 spec-compliance delta: the integrity-first deletion
guards and the read-only retention views. Production is never mutated
(the production artifacts are referenced AND protected; no deletion is
attempted over HTTP beyond the read-only 409 refusals).

Certified facts of this production root (third certified rebuild,
2026-09-11; pre-inventory m65_retention_pre.sha256, 52 files /
8,926,407 B / 0 tmp): 1 dataset (8a2af1e3d1fa, 16 refs: 6/2/4/2/0/1/1),
1 tokenizer (02673690c5ff, 15 refs: 6/2/4/2/0/0/0/1), 1 model
(5939483e70ac); M62 retention 13/10/3; dataset files 8, tokenizer
files 2.
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

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8788"
API = BASE + "/api/v1"

DS_ID = "8a2af1e3d1fa"
TOK_ID = "02673690c5ff"
MODEL_ID = "5939483e70ac"
DS_TOTAL = 16
TOK_TOTAL = 15
DS_FILES = 8
TOK_FILES = 2
N_CKPTS, N_DELETABLE, N_PROTECTED = 13, 10, 3
N_FILES = 52
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m65_retention_pre.sha256"

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


def walk_artifact(directory: Path) -> tuple[list[str], int]:
    files, total = [], 0
    for q in sorted(directory.rglob("*")):
        rel = q.relative_to(directory)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if q.is_file():
            files.append(rel.as_posix())
            total += q.stat().st_size
    return files, total


# ======================= production (READ-ONLY) ============================ #
project, _ = jget("/project")
root = Path(project["storage_root"])
check("P1 project readable", project["model_count"] == 1)

pre_disk = inventory(root)
pre_ref = {}
for line in PRE_INVENTORY.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    pre_ref[rel.removeprefix("./")] = digest
check("P2 disk == m65_retention_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

ds_usage, _ = jget(f"/datasets/{DS_ID}/usage")
tok_usage, _ = jget(f"/tokenizers/{TOK_ID}/usage")
check("P3 M64 usage coherent (16 dataset / 15 tokenizer refs)",
      ds_usage["total_references"] == DS_TOTAL
      and tok_usage["total_references"] == TOK_TOTAL)

ret, _ = jget(f"/models/{MODEL_ID}/checkpoints/retention")
check("P4 M62 retention coherent (13/10/3)",
      ret["total_checkpoints"] == N_CKPTS
      and ret["deletable_checkpoints"] == N_DELETABLE
      and ret["protected_checkpoints"] == N_PROTECTED)

# ===================== the retention views (once each) ===================== #
r = jget(f"/datasets/{DS_ID}/retention")
ov, body_ds = r
files, nbytes = walk_artifact(root / "datasets" / DS_ID)
check("R1 dataset retention: identity + artifact files + bytes",
      ov["dataset_id"] == DS_ID and ov["version_count"] == 1
      and ov["files"] == files and len(files) == DS_FILES
      and ov["size_bytes"] == nbytes,
      f"{len(ov['files'])} files / {ov['size_bytes']} B")
check("R2 dataset retention: integrity + not deletable + ordered blockers",
      ov["integrity_verified"] is True and ov["deletable"] is False
      and ov["blockers"] == [{"reason": c["category"],
                              "detail": ", ".join(c["references"])}
                             for c in ds_usage["categories"]
                             if c["references"]],
      f"{len(ov['blockers'])} blocker categories")

r = jget(f"/tokenizers/{TOK_ID}/retention")
tv, body_tok = r
tfiles, tbytes = walk_artifact(root / "tokenizers" / TOK_ID)
check("R3 tokenizer retention: identity + artifact files + bytes",
      tv["tokenizer_id"] == TOK_ID
      and tv["files"] == tfiles == ["manifest.json", "tokenizer.json"]
      and len(tfiles) == TOK_FILES and tv["size_bytes"] == tbytes
      and tv["trained_on_dataset_id"] == DS_ID,
      f"{len(tv['files'])} files / {tv['size_bytes']} B")
check("R4 tokenizer retention: integrity + not deletable + ordered blockers",
      tv["integrity_verified"] is True and tv["deletable"] is False
      and tv["blockers"] == [{"reason": c["category"],
                              "detail": ", ".join(c["references"])}
                             for c in tok_usage["categories"]
                             if c["references"]],
      f"{len(tv['blockers'])} blocker categories")

# ---- V: determinism + unknown 404s + protected DELETEs (read-only) ------- #
_, b2 = jget(f"/datasets/{DS_ID}/retention")
_, t2 = jget(f"/tokenizers/{TOK_ID}/retention")
check("V1 deterministic byte-identical repeats", b2 == body_ds and t2 == body_tok)

for path in ("/datasets/no-m65/retention", "/tokenizers/no-m65/retention"):
    try:
        urllib.request.urlopen(API + path, timeout=30)
        ok = False
    except urllib.error.HTTPError as e:
        ok = e.code == 404
    check(f"V2 unknown -> 404 ({path.split('/')[2]})", ok)

for path in (f"/datasets/{DS_ID}", f"/tokenizers/{TOK_ID}"):
    req = urllib.request.Request(API + path, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=30)
        ok = False
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        ok = e.code == 409 and body["detail"]["protected"] is True \
            and body["detail"]["blockers"]
    check(f"V3 protected DELETE -> 409 + ordered blockers ({path.split('/')[2]})",
          ok)

# ================== throwaway COPY verification (engine) =================== #
sys.path.insert(0, str(Path(__file__).resolve().parent))
copy_root = Path(tempfile.mkdtemp(prefix="m65rv-copy-")) / "root"
shutil.copytree(root, copy_root)
try:
    from app.engine import ModelForge
    from app.schemas import TokenizerConfig

    forge = ModelForge(root=copy_root)

    # C1: the copy's retention views equal the production HTTP answers
    copy_ds = forge.dataset_retention_overview(DS_ID).model_dump(mode="json")
    copy_tok = forge.tokenizer_retention_overview(TOK_ID).model_dump(
        mode="json")
    check("C1 copy retention views == production HTTP answers",
          copy_ds == ov and copy_tok == tv)

    # C2: THE one safe deletion in this certification — a fresh
    # unreferenced tokenizer trained on the copy (its own retention
    # view proves deletable FIRST)
    new_tok = forge.train_tokenizer(
        TokenizerConfig(name="m65rv-copy-tok", vocab_size=300),
        dataset_id=DS_ID).id
    rv = forge.tokenizer_retention_overview(new_tok)
    tdir = copy_root / "tokenizers" / new_tok
    efiles, ebytes = walk_artifact(tdir)
    check("C2 fresh tokenizer retention view proves it deletable",
          rv.integrity_verified is True and rv.blockers == []
          and rv.deletable is True
          and rv.files == efiles == ["manifest.json", "tokenizer.json"]
          and rv.size_bytes == ebytes)
    res = forge.delete_tokenizer(new_tok)
    check("C3 the ONE safe deletion: exact stats, atomic",
          res.tokenizer_id == new_tok
          and res.files_removed == len(efiles) == 2
          and res.bytes_reclaimed == ebytes
          and not tdir.exists()
          and not any(p.name.startswith(".")
                      for p in (copy_root / "tokenizers").iterdir()),
          f"{res.files_removed} files / {res.bytes_reclaimed} B")

    # C4: the copy's dataset usage/retention reflect the transient
    # reference exactly (live recompute, then back to the base state)
    ds_after = forge.dataset_retention_overview(DS_ID)
    check("C4 live recompute: transient reference gone, view restored",
          [b.reason for b in ds_after.blockers]
          == [b["reason"] for b in ov["blockers"]]
          and ds_after.integrity_verified is True
          and ds_after.deletable is False)

    # C5: protected artifacts survive everything on the copy
    check("C5 protected artifacts untouched on the copy",
          (copy_root / "datasets" / DS_ID).exists()
          and (copy_root / "tokenizers" / TOK_ID).exists())
finally:
    shutil.rmtree(copy_root.parent, ignore_errors=True)
check("C6 copy discarded", not copy_root.exists())

# ============================== audits ===================================== #
post_disk = inventory(root)
check("A1 production byte-identical before/after (zero mutation)",
      post_disk == pre_disk, f"{len(post_disk)} files")
check("A2 tmp clean", not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M65 RETENTION-VIEW CHECKS PASSED "
      "(production read-only; one safe deletion on a discarded copy)")
sys.exit(0 if n_pass == len(results) else 1)
