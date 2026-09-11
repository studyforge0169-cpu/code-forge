"""M63 LIVE certification smoke (READ-ONLY, executed exactly ONCE).

Certifies `GET /api/v1/project/storage` against the production root:
every number is cross-checked against an INDEPENDENT oracle (a direct
filesystem walk + an independent category classifier, plus the M62
route per model) — never against the app's own accounting code.

Zero state-mutating calls: GET only. Expected storage outcome:
byte-for-byte identical before and after (verified against the
pre-captured m63_pre.sha256 inventory AND a before/after walk).

Certified facts of this production root (reconstructed 2026-09-11 via
the committed m62_production_rebuild.py after the second sandbox
re-provision; verified by the M63 dev-time sanity run):
  52 files / 8,926,421 B / 0 tmp; 1 model; 13 checkpoints;
  10 deletable / 3 protected; checkpoint bytes 8,197,262;
  reclaimable 6,305,615; best == published @ 5.996844.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8785"
API = BASE + "/api/v1"

# ---- certified constants of THIS root (facts, hard-coded on purpose) ----
N_FILES = 52
TOTAL_BYTES = 8_926_421
N_MODELS = 1
N_CKPTS = 13
N_DELETABLE = 10
N_PROTECTED = 3
CKPT_BYTES = 8_197_262
RECLAIMABLE_BYTES = 6_305_615
BEST_LOSS = 5.996844
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m63_pre.sha256"

CATEGORIES = ["models", "checkpoints", "model_records", "datasets",
              "tokenizers", "suite_runs", "samples", "sample_evaluations",
              "policies", "probe_suites", "workflow_recipes", "project",
              "unclassified"]

EVIDENCE = {"evaluations", "comparisons", "gates", "workflows"}
ROOT_FAMILIES = {"datasets", "tokenizers", "suite-runs", "samples",
                 "sample-evaluations", "policies", "probe-suites",
                 "workflow-recipes"}

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((name, bool(cond), extra))
    print(f"{'PASS' if cond else 'FAIL'} {name} {extra}")


def get(path: str):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return r.status, r.read()


def jget(path: str):
    status, body = get(path)
    assert status == 200, (path, status, body[:200])
    import json
    return json.loads(body), body


def walk(root: Path) -> dict[str, int]:
    """INDEPENDENT physical walk (the M63 boundary)."""
    out: dict[str, int] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if rel.parts[0] == "tmp":
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        if p.is_file():
            out[rel.as_posix()] = p.stat().st_size
    return out


def classify(rel: str) -> str:
    """INDEPENDENT classifier (documented taxonomy, not the engine's)."""
    parts = rel.split("/")
    if parts[0] == "models":
        if len(parts) == 3:
            return "models"
        if len(parts) >= 4:
            if parts[2] == "checkpoints":
                return "checkpoints"
            if parts[2] in EVIDENCE:
                return "model_records"
        return "unclassified"
    if rel == "project.json":
        return "project"
    if parts[0] in ROOT_FAMILIES:
        return parts[0].replace("-", "_")
    return "unclassified"


def inventory(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.relative_to(root).parts[0] != "tmp":
            out[p.relative_to(root).as_posix()] = \
                hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ============================ prelude (read-only) ========================== #
project, _ = jget("/project")
root = Path(project["storage_root"])
check("P1 project readable", project["model_count"] == N_MODELS,
      f"models={project['model_count']}")

pre_disk = inventory(root)
pre_ref = {}
for line in PRE_INVENTORY.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    pre_ref[rel.removeprefix("./")] = digest
check("P2 disk == m63_pre.sha256 (52/52)",
      pre_disk == pre_ref, f"{len(pre_ref)} files")

models, _ = jget("/models")
model_ids = [m["id"] for m in models]
check("P3 model listing", len(model_ids) == N_MODELS, str(model_ids))

m62: dict[str, dict] = {}
for mid in model_ids:
    ov, _ = jget(f"/models/{mid}/checkpoints/retention")
    m62[mid] = ov
    best, _ = jget(f"/models/{mid}/checkpoints/best")
    model, _ = jget(f"/models/{mid}")
    check(f"P4 {mid}: M62/M52/M60 coherent",
          ov["total_checkpoints"] == N_CKPTS
          and best["checkpoint"]["checkpoint_id"]
          == model["latest_checkpoint"]
          and best["checkpoint"]["validation_loss"] == BEST_LOSS,
          f"ckpts={ov['total_checkpoints']} best==published@"
          f"{best['checkpoint']['validation_loss']}")

# ========================= the M63 overview (once) ========================= #
overview, body = jget("/project/storage")
check("X1 project storage readable", True,
      f"{overview['total_files']} files / {overview['total_bytes']} B")

# ---- V1: totals == independent physical oracle ---- #
files = walk(root)
check("V1 totals match physical oracle",
      overview["total_files"] == len(files) == N_FILES
      and overview["total_bytes"] == sum(files.values()) == TOTAL_BYTES,
      f"{overview['total_files']}/{overview['total_bytes']} vs "
      f"{len(files)}/{sum(files.values())}")

# ---- V2: categories reconcile + exact partition + canonical order ---- #
cats_o: dict[str, list[int]] = {}
for rel, size in files.items():
    c = classify(rel)
    cats_o.setdefault(c, [0, 0])
    cats_o[c][0] += 1
    cats_o[c][1] += size
names = [c["name"] for c in overview["categories"]]
check("V2a canonical category order", names == CATEGORIES, str(names))
ok_v2 = sum(c["files"] for c in overview["categories"]) == overview["total_files"] \
    and sum(c["bytes"] for c in overview["categories"]) == overview["total_bytes"]
for c in overview["categories"]:
    ok_v2 = ok_v2 and [c["files"], c["bytes"]] == cats_o.get(c["name"], [0, 0])
check("V2b categories == independent classifier (exact partition)",
      ok_v2 and overview["categories"][12]["name"] == "unclassified"
      and overview["categories"][12]["files"] == 0)

# ---- V3: per-model rows == that model's OWN M62 route ---- #
rows = {r["model_id"]: r for r in overview["models"]}
check("V3a row set/order == registry listing",
      [r["model_id"] for r in overview["models"]] == model_ids
      and overview["model_count"] == N_MODELS)
ok_v3 = True
for mid, row in rows.items():
    m = m62[mid]
    ok_v3 = ok_v3 and (row["checkpoint_count"], row["deletable_checkpoints"],
                       row["protected_checkpoints"],
                       row["total_checkpoint_bytes"],
                       row["reclaimable_checkpoint_bytes"]) == \
        (m["total_checkpoints"], m["deletable_checkpoints"],
         m["protected_checkpoints"], m["total_checkpoint_bytes"],
         m["reclaimable_checkpoint_bytes"])
    ok_v3 = ok_v3 and row["protected_checkpoint_bytes"] == sum(
        e["size_bytes"] for e in m["checkpoints"] if not e["deletable"])
    ok_v3 = ok_v3 and row["total_model_bytes"] == (
        row["model_bytes"] + row["records_bytes"]
        + row["total_checkpoint_bytes"])
    # per-model physical attribution straight from the walk
    ok_v3 = ok_v3 and row["model_bytes"] == sum(
        s for r, s in files.items()
        if r.split("/")[:2] == ["models", mid] and len(r.split("/")) == 3)
    ok_v3 = ok_v3 and row["records_bytes"] == sum(
        s for r, s in files.items()
        if r.split("/")[:2] == ["models", mid] and len(r.split("/")) >= 4
        and r.split("/")[2] in EVIDENCE)
check("V3b rows == M62 route aggregates (verbatim, all models)", ok_v3)

# ---- V4: project aggregates are exact sums over the rows ---- #
ok_v4 = all(
    overview[f] == sum(r[f] for r in overview["models"]) for f in
    ("checkpoint_count", "deletable_checkpoints", "protected_checkpoints",
     "total_checkpoint_bytes", "reclaimable_checkpoint_bytes",
     "protected_checkpoint_bytes"))
check("V4 project aggregates == sums over rows", ok_v4,
      f"ckpts={overview['checkpoint_count']} "
      f"deletable={overview['deletable_checkpoints']} "
      f"protected={overview['protected_checkpoints']}")

# ---- V5: certified retention facts + reclaimable NEVER overstates ---- #
check("V5 certified retention facts",
      overview["checkpoint_count"] == N_CKPTS
      and overview["deletable_checkpoints"] == N_DELETABLE
      and overview["protected_checkpoints"] == N_PROTECTED
      and overview["total_checkpoint_bytes"] == CKPT_BYTES
      and overview["reclaimable_checkpoint_bytes"] == RECLAIMABLE_BYTES
      and overview["reclaimable_checkpoint_bytes"]
      <= overview["total_checkpoint_bytes"]
      < overview["total_bytes"],
      f"reclaimable={overview['reclaimable_checkpoint_bytes']} "
      f"of ckpt={overview['total_checkpoint_bytes']} "
      f"of total={overview['total_bytes']}")

# ---- V6: healthy storage — physical checkpoints category == registry ---- #
by_cat = {c["name"]: c for c in overview["categories"]}
check("V6 checkpoints category == registry total (healthy)",
      by_cat["checkpoints"]["bytes"] == overview["total_checkpoint_bytes"]
      and by_cat["checkpoints"]["files"] == 2 * overview["checkpoint_count"],
      f"{by_cat['checkpoints']['files']} files / "
      f"{by_cat['checkpoints']['bytes']} B")

# ---- V7: M52 best + M60 published are protected (per model) ---- #
ok_v7 = True
for mid, m in m62.items():
    best, _ = jget(f"/models/{mid}/checkpoints/best")
    model, _ = jget(f"/models/{mid}")
    by_id = {e["checkpoint_id"]: e for e in m["checkpoints"]}
    ok_v7 = ok_v7 and not by_id[best["checkpoint"]["checkpoint_id"]]["deletable"]
    ok_v7 = ok_v7 and not by_id[model["latest_checkpoint"]]["deletable"]
check("V7 M52 best + M60 published protected", ok_v7)

# ---- V8: deterministic byte-identical repeat ---- #
_, body2 = jget("/project/storage")
check("V8 deterministic byte-identical repeat", body2 == body)

# ============================== audits ===================================== #
post_disk = inventory(root)
check("A1 storage byte-identical before/after (zero mutation)",
      post_disk == pre_disk, f"{len(post_disk)} files")
check("A2 tmp clean", not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M63 READ-ONLY CHECKS PASSED "
      "(zero mutations)")
sys.exit(0 if n_pass == len(results) else 1)
