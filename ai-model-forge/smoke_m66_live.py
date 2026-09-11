"""M66 LIVE certification smoke (READ-ONLY, executed exactly ONCE).

Certifies `GET /api/v1/models/{model_id}/usage` against the production
root. Every category is cross-checked against an INDEPENDENT oracle
that parses the raw manifests/layout directly (no app code) AND
against the public listing routes. Zero state-mutating calls: GET only.

Certified facts of this production root (identical bytes since the M65
certification — m66_pre.sha256 == m65_retention_pre.sha256, 52/52):
  1 model (5939483e70ac, live-model); usage categories:
  training_run 6, checkpoint 13, workflow 3, evaluation 4,
  comparison 2, gate 2, suite_run 0, sample 0, sample_quality 0,
  workflow_recipe 1 (m62-live-loop) -> internal 30, external 1,
  total 31; M62 retention 13/10/3; M64 usage 16 (dataset) / 15
  (tokenizer); 52 files / 8,926,407 B.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8789"
API = BASE + "/api/v1"

MODEL_ID = "5939483e70ac"
DS_ID = "8a2af1e3d1fa"
TOK_ID = "02673690c5ff"
CATS = {"training_run": 6, "checkpoint": 13, "workflow": 3, "evaluation": 4,
        "comparison": 2, "gate": 2, "suite_run": 0, "sample": 0,
        "sample_quality": 0, "workflow_recipe": 1}
INTERNAL, EXTERNAL, TOTAL = 30, 1, 31
N_CKPTS, N_DELETABLE, N_PROTECTED = 13, 10, 3
DS_TOTAL, TOK_TOTAL = 16, 15
ORDER = ["training_run", "checkpoint", "workflow", "evaluation",
         "comparison", "gate", "suite_run", "sample", "sample_quality",
         "workflow_recipe"]
INTERNAL_SET = set(ORDER[:6])
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m66_pre.sha256"

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


# ---------------------- the INDEPENDENT oracle (no app code) ---------------- #
INTERNAL_FAMILIES = {"checkpoint": ("checkpoints", "checkpoint_id"),
                     "workflow": ("workflows", "workflow_id"),
                     "evaluation": ("evaluations", "eval_id"),
                     "comparison": ("comparisons", "comparison_id"),
                     "gate": ("gates", "decision_id")}


def oracle(root: Path, model_id: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in ORDER}
    mdir = root / "models" / model_id
    man = json.loads((mdir / "manifest.json").read_text())
    out["training_run"] = sorted(p["run_id"] for p in
                                 man.get("training_provenance", []))
    for cat, (fam, idfield) in INTERNAL_FAMILIES.items():
        froot = mdir / fam
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if d.is_dir():
                    rec = json.loads((d / "manifest.json").read_text())
                    if rec.get("model_id", model_id) == model_id:
                        out[cat].append(rec[idfield])
    sroot = root / "suite-runs"
    if sroot.exists():
        for d in sorted(sroot.iterdir()):
            if d.is_dir():
                rec = json.loads((d / "manifest.json").read_text())
                if rec.get("model_id") == model_id:
                    out["suite_run"].append(rec["suite_run_id"])
    for cat, fam, idfield in (("sample", "samples", "sample_id"),
                              ("sample_quality", "sample-evaluations",
                               "evaluation_id")):
        froot = root / fam / model_id
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if d.is_dir():
                    rec = json.loads((d / "manifest.json").read_text())
                    if rec.get("model_id", model_id) == model_id:
                        out[cat].append(rec[idfield])
    rroot = root / "workflow-recipes"
    if rroot.exists():
        for d in sorted(rroot.iterdir()):
            if d.is_dir():
                rec = json.loads((d / "manifest.json").read_text())
                refs = set()

                def walk(o):
                    if isinstance(o, dict):
                        for k, v in o.items():
                            if k == "model_id":
                                refs.add(v)
                            walk(v)
                    elif isinstance(o, list):
                        for v in o:
                            walk(v)

                walk(rec.get("stages", []))
                if model_id in refs:
                    out["workflow_recipe"].append(rec["recipe_id"])
    return {c: sorted(out[c]) for c in out}


# ============================ prelude (read-only) ========================== #
project, _ = jget("/project")
root = Path(project["storage_root"])
check("P1 project readable", project["model_count"] == 1)

pre_disk = inventory(root)
pre_ref = {}
for line in PRE_INVENTORY.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    pre_ref[rel.removeprefix("./")] = digest
check("P2 disk == m66_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

models, _ = jget("/models")
check("P3 model registry", [m["id"] for m in models] == [MODEL_ID])

ret, _ = jget(f"/models/{MODEL_ID}/checkpoints/retention")
check("P4 M62 retention coherent (13/10/3)",
      ret["total_checkpoints"] == N_CKPTS
      and ret["deletable_checkpoints"] == N_DELETABLE
      and ret["protected_checkpoints"] == N_PROTECTED)

sto, _ = jget("/project/storage")
check("P5 M63 project storage coherent",
      sto["total_files"] == 52 and sto["model_count"] == 1)

ds_usage, _ = jget(f"/datasets/{DS_ID}/usage")
tok_usage, _ = jget(f"/tokenizers/{TOK_ID}/usage")
check("P6 M64 usage coherent (16 dataset / 15 tokenizer)",
      ds_usage["total_references"] == DS_TOTAL
      and tok_usage["total_references"] == TOK_TOTAL)

ora = oracle(root, MODEL_ID)

# ========================= the M66 overview (once) ========================= #
ov, body = jget(f"/models/{MODEL_ID}/usage")
check("U1 usage readable + identity",
      ov["model_id"] == MODEL_ID and ov["name"] == "live-model"
      and ov["architecture"] and ov["parameter_count"] > 0,
      f"total={ov['total_references']}")

check("U2 canonical category order",
      [c["category"] for c in ov["categories"]] == ORDER)

ok_u3 = True
cats = {c["category"]: c["references"] for c in ov["categories"]}
for cat in ORDER:
    ok_u3 = ok_u3 and cats[cat] == ora[cat] and len(cats[cat]) == CATS[cat]
check("U3 every category == independent oracle + certified counts", ok_u3,
      " ".join(f"{c}={len(cats[c])}" for c in ORDER))

check("U4 internal/external splits",
      ov["internal_references"] == INTERNAL
      and ov["external_references"] == EXTERNAL
      and ov["total_references"] == TOTAL
      and ov["referenced"] is True and ov["externally_referenced"] is True,
      f"internal={ov['internal_references']} external={ov['external_references']}")
check("U4b splits are exact sums over the categories",
      ov["internal_references"] == sum(len(cats[c]) for c in INTERNAL_SET)
      and ov["external_references"] == sum(len(cats[c]) for c in ORDER
                                           if c not in INTERNAL_SET))

# ---- U5: parity with the public listing routes ---------------------------- #
ck = sorted(c["checkpoint_id"] for c in
            jget(f"/models/{MODEL_ID}/checkpoints")[0])
ev = sorted(e["eval_id"] for e in
            jget(f"/models/{MODEL_ID}/evaluations")[0])
wf = sorted(w["workflow_id"] for w in
            jget(f"/models/{MODEL_ID}/workflows")[0])
cp = sorted(c["comparison_id"] for c in
            jget(f"/models/{MODEL_ID}/comparisons")[0])
gt = sorted(g["decision_id"] for g in
            jget(f"/models/{MODEL_ID}/gates/decisions")[0])
sr = sorted(s["suite_run_id"] for s in
            jget(f"/models/{MODEL_ID}/suite-runs")[0])
sm = sorted(s["sample_id"] for s in
            jget(f"/models/{MODEL_ID}/samples")[0])
sq = sorted(s["evaluation_id"] for s in
            jget(f"/models/{MODEL_ID}/sample-quality")[0])
mrec = jget(f"/models/{MODEL_ID}")[0]
tr = sorted(p["run_id"] for p in mrec["training_provenance"])
rc = jget("/workflows/recipes")[0]
recipes = sorted(r["recipe_id"] for r in rc
                 if MODEL_ID in json.dumps(r["stages"]))
check("U5 every category == its public listing route",
      cats["checkpoint"] == ck and cats["evaluation"] == ev
      and cats["workflow"] == wf and cats["comparison"] == cp
      and cats["gate"] == gt and cats["suite_run"] == sr
      and cats["sample"] == sm and cats["sample_quality"] == sq
      and cats["training_run"] == tr and cats["workflow_recipe"] == recipes)

# ---- U6: unknown model -> 404 ---------------------------------------------- #
try:
    urllib.request.urlopen(API + "/models/no-m66/usage", timeout=30)
    ok = False
except urllib.error.HTTPError as e:
    ok = e.code == 404
check("U6 unknown model -> 404", ok)

# ---- U7: deterministic byte-identical repeat ------------------------------- #
_, body2 = jget(f"/models/{MODEL_ID}/usage")
check("U7 deterministic byte-identical repeat", body2 == body)

# ---- U8: OpenAPI ----------------------------------------------------------- #
with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
NEW = "/api/v1/models/{model_id}/usage"
deletes = sorted(p for p, ops in spec["paths"].items() if "delete" in ops)
check("U8 OpenAPI: 92 paths, GET-only usage route, no new deletes",
      len(spec["paths"]) == 92
      and set(spec["paths"][NEW].keys()) == {"get"}
      and deletes == ["/api/v1/datasets/{dataset_id}",
                      "/api/v1/models/{model_id}",
                      "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}",
                      "/api/v1/tokenizers/{tokenizer_id}"]
      and "ModelUsageOverview" in spec["components"]["schemas"]
      and "ModelUsageCategory" in spec["components"]["schemas"])

# ============================== audits ===================================== #
post_disk = inventory(root)
check("A1 storage byte-identical before/after (zero mutation)",
      post_disk == pre_disk, f"{len(post_disk)} files")
check("A2 tmp clean", not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M66 READ-ONLY CHECKS PASSED "
      "(zero mutations)")
sys.exit(0 if n_pass == len(results) else 1)
