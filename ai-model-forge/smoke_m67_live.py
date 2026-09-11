"""M67 LIVE certification smoke (executed exactly ONCE).

Certifies explicit VERIFIED model retention against the production
root — read-only over production (a protected DELETE that must 409
with the exact M66-visible external blockers), then the SUCCESS path
on a DISPOSABLE COPY (create + train a fresh internal-only model,
verify safe state, delete it, prove exact atomic removal, discard).

Certified facts of this production root (byte-identical since the M66
certification — m67_pre.sha256 == m66_pre.sha256, 52/52):
  1 model (5939483e70ac, live-model); M66 usage (11 categories):
  training_run 6, checkpoint 13, workflow 3, evaluation 4,
  comparison 2, gate 2, suite_run 0, sample 0, sample_quality 0,
  workflow_recipe 1 (m62-live-loop), policy 0 -> internal 30,
  external 1, total 31; retention: integrity True, deletable False,
  40 files / 8,874,610 B, blockers exactly
  [workflow_recipe: m62-live-loop]; M62 retention 13/10/3; M64 usage
  16 (dataset) / 15 (tokenizer); 52 files / 8,926,407 B.
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

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8790"
API = BASE + "/api/v1"

MODEL_ID = "5939483e70ac"
DS_ID = "8a2af1e3d1fa"
TOK_ID = "02673690c5ff"
RECIPE_HASH = ("4d5301577f3e80b11ad656df267ada60"
               "ee0006ac191a547237f8bdd7c73838d0")
CATS = {"training_run": 6, "checkpoint": 13, "workflow": 3, "evaluation": 4,
        "comparison": 2, "gate": 2, "suite_run": 0, "sample": 0,
        "sample_quality": 0, "workflow_recipe": 1, "policy": 0}
INTERNAL, EXTERNAL, TOTAL = 30, 1, 31
RET_FILES, RET_BYTES = 40, 8874610
N_CKPTS, N_DELETABLE, N_PROTECTED = 13, 10, 3
DS_TOTAL, TOK_TOTAL = 16, 15
ORDER = ["training_run", "checkpoint", "workflow", "evaluation",
         "comparison", "gate", "suite_run", "sample", "sample_quality",
         "workflow_recipe", "policy"]
INTERNAL_SET = set(ORDER[:6])
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m67_pre.sha256"
PROD_ROOT = Path("/home/user/ai-model-forge-data")

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
    """category -> sorted reference ids, from raw manifests/layout."""
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
    for fam, is_recipe in (("workflow-recipes", True), ("policies", False)):
        froot = root / fam
        if not froot.exists():
            continue
        for d in sorted(froot.iterdir()):
            if not d.is_dir():
                continue
            rec = json.loads((d / "manifest.json").read_text())
            if is_recipe:
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
                hit = model_id in refs
                cat, idfield = "workflow_recipe", "recipe_id"
            else:
                hit = rec.get("policy", {}).get("model_id") == model_id
                cat, idfield = "policy", "policy_id"
            if hit:
                out[cat].append(rec[idfield])
    return {c: sorted(out[c]) for c in out}


# ============================ prelude (read-only) ========================== #
project, _ = jget("/project")
root = Path(project["storage_root"])
check("P1 project readable", project["model_count"] == 1
      and str(root) == str(PROD_ROOT))

pre_disk = inventory(root)
pre_ref = {}
for line in PRE_INVENTORY.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    pre_ref[rel.removeprefix("./")] = digest
check("P2 disk == m67_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

models, _ = jget("/models")
check("P3 model registry", [m["id"] for m in models] == [MODEL_ID])

ret62, _ = jget(f"/models/{MODEL_ID}/checkpoints/retention")
check("P4 M62 retention coherent (13/10/3)",
      ret62["total_checkpoints"] == N_CKPTS
      and ret62["deletable_checkpoints"] == N_DELETABLE
      and ret62["protected_checkpoints"] == N_PROTECTED)

sto, _ = jget("/project/storage")
check("P5 M63 project storage coherent",
      sto["total_files"] == 52 and sto["model_count"] == 1)

ds_usage, _ = jget(f"/datasets/{DS_ID}/usage")
tok_usage, _ = jget(f"/tokenizers/{TOK_ID}/usage")
check("P6 M64 usage coherent (16 dataset / 15 tokenizer)",
      ds_usage["total_references"] == DS_TOTAL
      and tok_usage["total_references"] == TOK_TOTAL)

ora = oracle(root, MODEL_ID)

# ===================== M66 usage (unchanged surface) ======================= #
ov, usage_body = jget(f"/models/{MODEL_ID}/usage")
cats = {c["category"]: c["references"] for c in ov["categories"]}
check("U1 M66 usage: 11 canonical categories, certified counts",
      [c["category"] for c in ov["categories"]] == ORDER
      and all(len(cats[c]) == CATS[c] for c in ORDER)
      and ov["internal_references"] == INTERNAL
      and ov["external_references"] == EXTERNAL
      and ov["total_references"] == TOTAL,
      " ".join(f"{c}={len(cats[c])}" for c in ORDER))
check("U2 M66 usage == independent manifest oracle",
      all(cats[c] == ora[c] for c in ORDER))

# ===================== M67 retention + protected DELETE ==================== #
ret, ret_body = jget(f"/models/{MODEL_ID}/retention")
mdir = root / "models" / MODEL_ID
walk_files = sorted(p.relative_to(mdir).as_posix()
                    for p in mdir.rglob("*") if p.is_file())
walk_bytes = sum(p.stat().st_size for p in mdir.rglob("*")
                 if p.is_file())
check("R1 retention: identity + integrity + stats == independent walk",
      ret["model_id"] == MODEL_ID and ret["name"] == "live-model"
      and ret["integrity_verified"] is True
      and ret["files"] == walk_files and ret["files"] and
      len(ret["files"]) == RET_FILES
      and ret["size_bytes"] == walk_bytes == RET_BYTES,
      f"{RET_FILES} files / {RET_BYTES} B")

expected_blockers = [{"category": c, "reference_id": r}
                     for c in ORDER if c not in INTERNAL_SET
                     for r in cats[c]]
check("R2 retention: deletable False, blockers == M66 external refs",
      ret["deletable"] is False
      and [{k: b[k] for k in ("category", "reference_id")}
           for b in ret["blockers"]] == expected_blockers
      and ret["blockers"][0]["detail"] == f"config_hash '{RECIPE_HASH}'",
      f"{len(expected_blockers)} blocker(s)")

# the protected DELETE: 409, structured, zero mutation
req = urllib.request.Request(API + f"/models/{MODEL_ID}",
                             method="DELETE")
try:
    urllib.request.urlopen(req, timeout=30)
    ok, detail = False, {}
except urllib.error.HTTPError as e:
    ok = e.code == 409
    # FastAPI wraps HTTPException payloads in a {"detail": ...} envelope
    detail = json.loads(e.read())["detail"]
check("D1 protected DELETE -> structured 409",
      ok and set(detail) == {"message", "model_id", "protected",
                             "blockers"}
      and detail["model_id"] == MODEL_ID
      and detail["protected"] is True)
check("D2 409 blockers == M66 usage external refs EXACTLY",
      [{k: b[k] for k in ("category", "reference_id")}
       for b in detail["blockers"]] == expected_blockers
      and detail["blockers"] == ret["blockers"],
      "the headline invariant, live")

# determinism (byte-identical repeats)
_, usage_body2 = jget(f"/models/{MODEL_ID}/usage")
_, ret_body2 = jget(f"/models/{MODEL_ID}/retention")
check("D3 deterministic byte-identical repeats",
      usage_body2 == usage_body and ret_body2 == ret_body)

# public listing parity for the referencing external family
recipes = sorted(r["recipe_id"] for r in
                 jget("/workflows/recipes")[0]
                 if MODEL_ID in json.dumps(r["stages"]))
check("D4 public listing parity (recipes)",
      cats["workflow_recipe"] == recipes == ["m62-live-loop"])

# unknown model -> 404 on both endpoints
try:
    urllib.request.urlopen(API + "/models/no-m67/usage", timeout=30)
    u404 = False
except urllib.error.HTTPError as e:
    u404 = e.code == 404
try:
    urllib.request.urlopen(API + "/models/no-m67/retention", timeout=30)
    r404 = False
except urllib.error.HTTPError as e:
    r404 = e.code == 404
try:
    urllib.request.urlopen(
        urllib.request.Request(API + "/models/no-m67", method="DELETE"),
        timeout=30)
    d404 = False
except urllib.error.HTTPError as e:
    d404 = e.code == 404
check("D5 unknown model -> 404 (usage / retention / DELETE)",
      u404 and r404 and d404)

# OpenAPI
with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
NEW_RET = "/api/v1/models/{model_id}/retention"
deletes = sorted(p for p, ops in spec["paths"].items() if "delete" in ops)
check("O1 OpenAPI: 93 paths, retention GET-only, NO new DELETE paths",
      len(spec["paths"]) == 93
      and set(spec["paths"][NEW_RET].keys()) == {"get"}
      and set(spec["paths"]["/api/v1/models/{model_id}"].keys()) ==
      {"get", "delete"}
      and deletes == ["/api/v1/datasets/{dataset_id}",
                      "/api/v1/models/{model_id}",
                      "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}",
                      "/api/v1/tokenizers/{tokenizer_id}"]
      and spec["paths"]["/api/v1/models/{model_id}"]["delete"][
          "responses"]["409"]["content"]["application/json"]["schema"][
          "$ref"] == "#/components/schemas/ModelDeletionBlocked")

# production audit after ALL read-only production calls
check("A1 production byte-identical after protected DELETE (zero "
      "mutation)", inventory(root) == pre_disk, f"{len(pre_disk)} files")
check("A2 tmp clean", not list((root / "tmp").iterdir()))

# ===================== disposable-copy success path ======================== #
# (in-process, on a THROWAWAY COPY of the production root; the copy is
#  discarded at the end — production is never mutated)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.engine import ModelForge  # noqa: E402
from app.schemas import (  # noqa: E402
    ModelCreateRequest, TransformerConfig, TrainingConfig)

copy_root = Path(tempfile.mkdtemp(prefix="m67-copy-"))
try:
    shutil.copytree(PROD_ROOT, copy_root, dirs_exist_ok=True)
    forge = ModelForge(root=copy_root)

    # guard parity ON THE COPY: the production model is still blocked
    copy_before = inventory(copy_root)
    try:
        forge.delete_model(MODEL_ID)
        refused = False
    except ValueError:
        refused = True
    check("C1 copy: production model still refused (guard parity)",
          refused and inventory(copy_root) == copy_before)

    # a fresh model, TRAINED on the copy's existing dataset+tokenizer:
    # internal references (runs + checkpoints) must NOT block
    fresh = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m67-copy-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=99)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=fresh,
        dataset_id=DS_ID, tokenizer_id=TOK_ID, learning_rate=3e-3,
        batch_size=8, max_seq_len=32, eval_every_steps=2,
        keep_best=False, seed=99, steps=4))
    usage = forge.model_usage_overview(fresh)
    ucats = {c.category: c.references for c in usage.categories}
    check("C2 copy: fresh trained model — internal refs, zero external",
          ucats["training_run"] and ucats["checkpoint"]
          and usage.external_references == 0
          and all(ucats[c] == [] for c in ORDER
                  if c not in INTERNAL_SET))

    # safe state confirmed through the retention view, then DELETE
    fret = forge.model_retention_overview(fresh)
    check("C3 copy: retention safe state (integrity ok, deletable)",
          fret.integrity_verified is True and fret.deletable is True
          and fret.blockers == [])
    pre_delete = inventory(copy_root)
    fdir = copy_root / "models" / fresh
    exp_files = sorted(p.relative_to(fdir).as_posix()
                       for p in fdir.rglob("*") if p.is_file())
    exp_bytes = sum(p.stat().st_size for p in fdir.rglob("*")
                    if p.is_file())
    result = forge.delete_model(fresh)
    after_delete = inventory(copy_root)
    check("C4 copy: deletion stats == independent pre-walk",
          result.model_id == fresh
          and result.files_removed == len(exp_files) == len(fret.files)
          and result.bytes_reclaimed == exp_bytes == fret.size_bytes,
          f"{result.files_removed} files / {result.bytes_reclaimed} B")
    check("C5 copy: ONLY the model dir removed, atomically",
          set(after_delete) == set(pre_delete) - {
              f"models/{fresh}/{rel}" for rel in exp_files}
          and not fdir.exists()
          and not any(p.name.startswith(".tmp-delete")
                      for p in (copy_root / "models").iterdir()))
    check("C6 copy: registries shrink; everything else untouched",
          fresh not in forge.storage.model_ids()
          and len(forge.storage.model_ids()) == 1
          and forge.model_retention_overview(MODEL_ID).deletable
          is False)
finally:
    shutil.rmtree(copy_root, ignore_errors=True)
check("C7 copy discarded", not copy_root.exists())

# final production audit
check("A3 production byte-identical, final (52/52)",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M67 CHECKS PASSED "
      "(production untouched; success path on a discarded copy)")
sys.exit(0 if n_pass == len(results) else 1)
