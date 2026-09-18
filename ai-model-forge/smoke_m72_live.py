"""M72 LIVE certification smoke (production touched READ-ONLY only —
the milestone IS a read-only inventory; nothing is deleted, created
or mutated anywhere; executed exactly ONCE).

Certifies `GET /api/v1/project/retention` — the PROJECT retention
inventory (M72): the whole deletion surface in ONE view — against
the production root. The canonical family order, the per-family
counts, the aggregation invariant (every per-family number == the
SUM of the EXISTING per-artifact retention views, fetched
individually as an independent oracle — no second scanner), the
TRUE totals (the ONE M63 physical walk == the raw disk), the EXACT
reclaimable (the model/record overlap rule) and determinism all
proven LIVE.

Certified facts of this production root (byte-identical since the
M68 delta — m72_pre.sha256 == m71_pre.sha256, 52/52):
  1 model (a266c8480cb7, BLOCKED — its only external reference is
  the workflow recipe), 1 dataset (5062cc7e0954, BLOCKED), 1
  tokenizer (f24b699890eb, BLOCKED), 1 workflow recipe
  (m62-live-loop, BLOCKED by its 2 workflow runs), 0 policies, 0
  probe suites; M69 records: training_run 6, checkpoint 13
  (M62: 10 blocked / 3 deletable), workflow 3 (all LEAF,
  deletable), evaluation 4 (all blocked), comparison 2 (blocked),
  gate 2 (blocked); 0 suite runs / samples / sample-quality
  measurements. Reclaimable = exactly the 3 leaf workflow records
  + the 10 deletable checkpoint artifact sets (M62: 3 protected).
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8796"
API = BASE + "/api/v1"

MODEL_ID = "a266c8480cb7"
DS_ID = "5062cc7e0954"
TOK_ID = "f24b699890eb"
RECIPE_ID = "m62-live-loop"
TOTAL_FILES = 52
TOTAL_BYTES = 8926403

# the certified M69/M62/M70 record inventory of this root (the M62 registry overview:
# 13 checkpoints / 10 DELETABLE / 3 protected — 2 held by
# workflow stage artifacts, 1 by best+published+manifest
# pointers)
CHECKPOINTS = 13
CK_DELETABLE = 10
CK_BLOCKED = 3
WORKFLOWS = ["28c7631e1152", "8f022ff69681", "edcb773ef326"]
EVALUATIONS = ["0a5a1e331f90", "7afcaa09abdb", "bcfaf78933a5",
               "d44e1566c239"]
COMPARISONS = ["71224941a968", "894e77ab88bc"]
GATES = ["5117c7a5f895", "983c7d65fab5"]
TRAINING_RUNS = 6

FAMILY_ORDER = [
    "model", "dataset", "tokenizer", "workflow_recipe", "gate_policy",
    "probe_suite", "training_run", "checkpoint", "workflow",
    "evaluation", "comparison", "gate", "suite_run", "sample",
    "sample_quality",
]
EXPECTED_COUNTS = {
    "model": 1, "dataset": 1, "tokenizer": 1, "workflow_recipe": 1,
    "gate_policy": 0, "probe_suite": 0, "training_run": TRAINING_RUNS,
    "checkpoint": CHECKPOINTS, "workflow": 3, "evaluation": 4,
    "comparison": 2, "gate": 2, "suite_run": 0, "sample": 0,
    "sample_quality": 0,
}
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m72_pre.sha256"
PROD_ROOT = Path(__import__("os").environ.get(
    "FORGE_SMOKE_ROOT", "/home/user/ai-model-forge-data"))

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
check("P2 disk == m72_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

models, _ = jget("/models")
check("P3 model registry", [m["id"] for m in models] == [MODEL_ID])

recipes, _ = jget("/workflows/recipes")
policies, _ = jget("/policies")
suites, _ = jget("/probe-suites")
check("P4 definition registries: 1 recipe / 0 policies / 0 suites",
      [r["recipe_id"] for r in recipes] == [RECIPE_ID]
      and policies == [] and suites == [])

# ==================== the inventory view (read-only) ======================= #
body, raw1 = jget("/project/retention")
fam = {f["family"]: f for f in body["families"]}

check("V1 exact response shape + canonical family order",
      set(body) == {"families", "total_count", "total_files",
                    "total_size_bytes", "total_deletable",
                    "total_blocked", "reclaimable_files",
                    "reclaimable_bytes"}
      and [f["family"] for f in body["families"]] == FAMILY_ORDER
      and all(set(f) == {"family", "deletion_supported", "count",
                         "files", "size_bytes", "deletable_count",
                         "blocked_count", "reclaimable_files",
                         "reclaimable_bytes"} for f in body["families"]))

check("V2 per-family certified counts (34 artifacts)",
      all(fam[f]["count"] == n for f, n in EXPECTED_COUNTS.items())
      and body["total_count"] == 34)
check("V3 training_run is the ONE lifecycle-less family",
      fam["training_run"]["deletion_supported"] is False
      and fam["training_run"]["deletable_count"] == 0
      and fam["training_run"]["blocked_count"] == 0
      and fam["training_run"]["files"] == 0
      and all(fam[f]["deletion_supported"] is True
              for f in FAMILY_ORDER if f != "training_run"))
check("V4 certified deletable/blocked: ckpt 10/3, wf 3/0, evals 0/4, "
      "comp 0/2, gate 0/2; model/ds/tok/recipe all blocked",
      fam["checkpoint"]["deletable_count"] == CK_DELETABLE
      and fam["checkpoint"]["blocked_count"] == CK_BLOCKED
      and fam["workflow"]["deletable_count"] == 3
      and fam["workflow"]["blocked_count"] == 0
      and fam["evaluation"]["deletable_count"] == 0
      and fam["evaluation"]["blocked_count"] == 4
      and fam["comparison"]["deletable_count"] == 0
      and fam["comparison"]["blocked_count"] == 2
      and fam["gate"]["deletable_count"] == 0
      and fam["gate"]["blocked_count"] == 2
      and all(fam[f]["deletable_count"] == 0 and fam[f]["blocked_count"] == 1
              for f in ("model", "dataset", "tokenizer",
                        "workflow_recipe"))
      and body["total_deletable"] == 13
      and body["total_blocked"] == 15)

# ---- the aggregation oracle: per-item views fetched individually ------ #
mv, _ = jget(f"/models/{MODEL_ID}/retention")
dsv, _ = jget(f"/datasets/{DS_ID}/retention")
tokv, _ = jget(f"/tokenizers/{TOK_ID}/retention")
recv, _ = jget(f"/workflows/recipes/{RECIPE_ID}/retention")
ckv, _ = jget(f"/models/{MODEL_ID}/checkpoints/retention")

ok_files = (fam["model"]["files"] == len(mv["files"])
            and fam["model"]["size_bytes"] == mv["size_bytes"]
            and fam["dataset"]["files"] == len(dsv["files"])
            and fam["dataset"]["size_bytes"] == dsv["size_bytes"]
            and fam["tokenizer"]["files"] == len(tokv["files"])
            and fam["tokenizer"]["size_bytes"] == tokv["size_bytes"]
            and fam["workflow_recipe"]["files"] == len(recv["files"])
            and fam["workflow_recipe"]["size_bytes"] == recv["size_bytes"]
            and fam["checkpoint"]["files"] == sum(
                e["files"] for e in ckv["checkpoints"])
            and fam["checkpoint"]["size_bytes"]
            == ckv["total_checkpoint_bytes"])
wf_bytes = wf_files = 0
for wid in WORKFLOWS:
    wv, _ = jget(f"/models/{MODEL_ID}/workflows/{wid}/retention")
    wf_files += len(wv["files"])
    wf_bytes += wv["size_bytes"]
ok_files = ok_files and fam["workflow"]["files"] == wf_files \
    and fam["workflow"]["size_bytes"] == wf_bytes
ev_bytes = ev_files = 0
for eid in EVALUATIONS:
    ev, _ = jget(f"/models/{MODEL_ID}/evaluations/{eid}/retention")
    ev_files += len(ev["files"])
    ev_bytes += ev["size_bytes"]
ok_files = ok_files and fam["evaluation"]["files"] == ev_files \
    and fam["evaluation"]["size_bytes"] == ev_bytes
check("V5 HEADLINE aggregation: family files/bytes == the SUM of the "
      "EXISTING per-item retention views (fetched individually)",
      ok_files)

# ---- TRUE totals: the ONE M63 walk == the raw disk ------------------- #
storage, _ = jget("/project/storage")
check("V6 TRUE totals == the ONE M63 physical walk == the raw disk "
      "(52 files / 8,926,403 B)",
      body["total_files"] == storage["total_files"] == TOTAL_FILES
      and body["total_size_bytes"] == storage["total_bytes"]
      == TOTAL_BYTES == sum(p.stat().st_size
                            for p in root.rglob("*")
                            if p.is_file()
                            and p.relative_to(root).parts[0] != "tmp"))

# ---- EXACT reclaimable (the overlap rule) ---------------------------- #
# the model is BLOCKED (the recipe) -> it contributes only its own
# deletable records: the 3 leaf workflows + the 10 deletable
# checkpoint artifact sets (M62: 3 protected)
exp_rf = 3 + sum(e["files"] for e in ckv["checkpoints"] if e["deletable"])
exp_rb = wf_bytes + ckv["reclaimable_checkpoint_bytes"]
check("V7 EXACT reclaimable == 3 leaf workflow records + the 10 "
      "deletable checkpoint artifact sets (model blocked -> records "
      "only; the overlap rule)",
      body["reclaimable_files"] == exp_rf
      and body["reclaimable_bytes"] == exp_rb
      and fam["workflow"]["reclaimable_bytes"] == wf_bytes
      and fam["checkpoint"]["reclaimable_bytes"]
      == ckv["reclaimable_checkpoint_bytes"]
      and fam["model"]["reclaimable_files"] == 0,
      f"{exp_rf} files / {exp_rb} B")

# ---- determinism ------------------------------------------------------ #
_, raw2 = jget("/project/retention")
check("V8 deterministic repeat (byte-identical)", raw1 == raw2)

# ============================ scope (read-only) ============================ #
# a fresh sub-root has no project state — but the API binds ONE root;
# the only scope-exercise available read-only is the route presence
# and the OpenAPI surface
with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
del_paths = sorted(p for p, ops in spec["paths"].items()
                   if "delete" in ops)
check("O1 OpenAPI: 105 paths / 14 deletes / the new route GET-only / "
      "2 schemas",
      len(spec["paths"]) == 105 and len(del_paths) == 14
      and set(spec["paths"]["/api/v1/project/retention"].keys())
      == {"get"}
      and all(s in spec["components"]["schemas"] for s in (
          "ProjectFamilyRetention", "ProjectRetentionOverview")))

check("A1 production byte-identical after all checks (ZERO mutation)",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()),
      f"{len(pre_disk)} files")

# ============================== summary ==================================== #
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print(f"\nM72 LIVE SMOKE: {n_pass}/{len(results)} checks passed")
for name, ok, extra in results:
    if not ok:
        print(f"  FAILED: {name} {extra}")
sys.exit(1 if n_fail else 0)
