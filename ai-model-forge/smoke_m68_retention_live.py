"""M68-DELTA LIVE certification smoke (executed exactly ONCE).

Certifies the retention views (the §5 spec-compliance delta: GET
/models/{id}/suite-runs/{run}/retention, GET .../samples/{sample}/
retention, GET .../sample-quality/{eval}/retention) against the
production root — read-only over production — and the §5 invariant
(retention.deletable == DELETE would succeed; retention.blockers ==
DELETE blockers; DELETE stats == the view's artifact accounting) on a
DISPOSABLE COPY, which is discarded afterwards.

Certified facts of this production root (rebuilt after the FIFTH
sandbox re-provision — same certified structure, new content-derived
ids; m68_retention_pre.sha256, 52/52):
  1 model (a266c8480cb7, live-model); M66 usage (11 categories):
  training_run 6, checkpoint 13, workflow 3, evaluation 4,
  comparison 2, gate 2, suite_run 0, sample 0, sample_quality 0,
  workflow_recipe 1 (m62-live-loop), policy 0 -> internal 30,
  external 1, total 31; M67 retention: integrity True, deletable
  False, 40 files / 8,874,606 B, blockers exactly
  [workflow_recipe: m62-live-loop]; 52 files / 8,926,403 B.
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

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8792"
API = BASE + "/api/v1"

MODEL_ID = "a266c8480cb7"
DS_ID = "5062cc7e0954"
TOK_ID = "f24b699890eb"
CATS = {"training_run": 6, "checkpoint": 13, "workflow": 3, "evaluation": 4,
        "comparison": 2, "gate": 2, "suite_run": 0, "sample": 0,
        "sample_quality": 0, "workflow_recipe": 1, "policy": 0}
ORDER = ["training_run", "checkpoint", "workflow", "evaluation",
         "comparison", "gate", "suite_run", "sample", "sample_quality",
         "workflow_recipe", "policy"]
RET_FILES, RET_BYTES = 40, 8874606
PRE_INVENTORY = Path(__file__).resolve().parent.parent / \
    "m68_retention_pre.sha256"
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


def walk_stats(d: Path) -> tuple[list[str], int]:
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*")
                   if p.is_file())
    return files, sum(p.stat().st_size for p in d.rglob("*")
                      if p.is_file())


# ---------------------- the INDEPENDENT oracle (no app code) ---------------- #
def oracle_external_model_refs(root: Path, model_id: str) -> dict[str, list]:
    out: dict[str, list] = {c: [] for c in
                            ("suite_run", "sample", "sample_quality",
                             "workflow_recipe", "policy")}
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
                if model_id in json.dumps(rec.get("stages", [])):
                    out["workflow_recipe"].append(rec["recipe_id"])
    proot = root / "policies"
    if proot.exists():
        for d in sorted(proot.iterdir()):
            if d.is_dir():
                rec = json.loads((d / "manifest.json").read_text())
                if rec.get("policy", {}).get("model_id") == model_id:
                    out["policy"].append(rec["policy_id"])
    return {c: sorted(v) for c, v in out.items()}


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
check("P2 disk == m68_retention_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

models, _ = jget("/models")
check("P3 model registry", [m["id"] for m in models] == [MODEL_ID])

ov, usage_body = jget(f"/models/{MODEL_ID}/usage")
cats = {c["category"]: c["references"] for c in ov["categories"]}
ora = oracle_external_model_refs(root, MODEL_ID)
check("P4 M66 usage: certified counts + oracle parity",
      [c["category"] for c in ov["categories"]] == ORDER
      and all(len(cats[c]) == CATS[c] for c in ORDER)
      and all(cats[c] == ora[c] for c in ora)
      and ov["internal_references"] == 30
      and ov["external_references"] == 1,
      " ".join(f"{c}={len(cats[c])}" for c in ORDER))

ret, ret_body = jget(f"/models/{MODEL_ID}/retention")
check("P5 M67 retention: blocked by the recipe only",
      ret["integrity_verified"] is True and ret["deletable"] is False
      and len(ret["files"]) == RET_FILES
      and ret["size_bytes"] == RET_BYTES
      and [(b["category"], b["reference_id"]) for b in ret["blockers"]] ==
      [("workflow_recipe", "m62-live-loop")])

# the three new retention routes: production holds no runtime records,
# so its exercisable surface is the unknown-id 404 path
ret404 = True
for path in (f"/models/{MODEL_ID}/suite-runs/no-such/retention",
             f"/models/{MODEL_ID}/samples/no-such/retention",
             f"/models/{MODEL_ID}/sample-quality/no-such/retention",
             "/models/no-m68/suite-runs/x/retention",
             "/models/no-m68/samples/x/retention",
             "/models/no-m68/sample-quality/x/retention"):
    try:
        urllib.request.urlopen(API + path, timeout=30)
        ret404 = False
    except urllib.error.HTTPError as e:
        ret404 = ret404 and e.code == 404
check("R1 retention routes: unknown-id 404s (production has 0 runtime "
      "records)", ret404)

_, usage_body2 = jget(f"/models/{MODEL_ID}/usage")
_, ret_body2 = jget(f"/models/{MODEL_ID}/retention")
check("R2 deterministic byte-identical repeats",
      usage_body2 == usage_body and ret_body2 == ret_body)

# OpenAPI: 96 paths; three GET-only retention routes; 7 deletes (the
# delta added ZERO delete operations); schemas present
with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
RET_PATHS = ["/api/v1/models/{model_id}/suite-runs/{suite_run_id}"
             "/retention",
             "/api/v1/models/{model_id}/samples/{sample_id}/retention",
             "/api/v1/models/{model_id}/sample-quality/{evaluation_id}"
             "/retention"]
deletes = sorted(p for p, ops in spec["paths"].items() if "delete" in ops)
check("R3 OpenAPI: 96 paths, 3 GET-only retention routes, 7 deletes",
      len(spec["paths"]) == 96
      and all(set(spec["paths"][p].keys()) == {"get"} for p in RET_PATHS)
      and len(deletes) == 7
      and all(s in spec["components"]["schemas"] for s in (
          "SuiteRunRetentionOverview", "SampleRetentionOverview",
          "SampleEvaluationRetentionOverview")))

check("A1 production byte-identical after all read-only checks",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()),
      f"{len(pre_disk)} files")

# ===================== disposable-copy: the §5 invariant ==================== #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.engine import ModelForge  # noqa: E402
from app.schemas import (  # noqa: E402
    ComparisonState, EvalStateKind, ModelCreateRequest,
    ProbeSuiteCreateRequest, SampleGenerateRequest, SampleStrategy,
    SuiteProbe, SuiteRunRequest, TransformerConfig, TrainingConfig)

copy_root = Path(tempfile.mkdtemp(prefix="m68r-copy-"))
try:
    shutil.copytree(PROD_ROOT, copy_root, dirs_exist_ok=True)
    forge = ModelForge(root=copy_root)

    # a fresh model + the THREE runtime records, on the copy
    fresh = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m68r-copy-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=77)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=fresh, dataset_id=DS_ID,
        tokenizer_id=TOK_ID, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=2, keep_best=False, seed=77,
        steps=4))
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m68r-suite", description="M68r smoke",
        probes=[SuiteProbe(dataset_id=DS_ID, split="validation",
                           tokenizer_id=TOK_ID, batch_size=8,
                           max_seq_len=32, seed=42)]))
    run = forge.run_suite(SuiteRunRequest(
        model_id=fresh, suite_id="m68r-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    best = forge.select_best_checkpoint(fresh).checkpoint.checkpoint_id
    smp = forge.generate_sample(SampleGenerateRequest(
        model_id=fresh, checkpoint_id=best, tokenizer_id=TOK_ID,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=6))
    qua = forge.evaluate_sample(fresh, smp.sample_id)

    # the model is blocked by all THREE runtime records (canonical order)
    usage = forge.model_usage_overview(fresh)
    ucats = {c.category: c.references for c in usage.categories}
    check("C1 copy: fresh model blocked by all three runtime records",
          ucats["suite_run"] == [run.suite_run_id]
          and ucats["sample"] == [smp.sample_id]
          and ucats["sample_quality"] == [qua.evaluation_id]
          and usage.external_references == 3)

    # ---- the retention views + independent walks ---------------------- #
    vrun = forge.suite_run_retention_overview(fresh, run.suite_run_id)
    vsmp = forge.sample_retention_overview(fresh, smp.sample_id)
    vqua = forge.sample_evaluation_retention_overview(fresh,
                                                      qua.evaluation_id)
    wrun = walk_stats(copy_root / "suite-runs" / run.suite_run_id)
    wsmp = walk_stats(copy_root / "samples" / fresh /
                      f"sample-{smp.sample_id}")
    wqua = walk_stats(copy_root / "sample-evaluations" / fresh /
                      f"evaluation-{qua.evaluation_id}")
    check("C2 copy: leaves deletable; sample blocked; stats == walks",
          vrun.deletable is True and vrun.blockers == []
          and (vrun.files, vrun.size_bytes) == wrun
          and vqua.deletable is True and vqua.blockers == []
          and (vqua.files, vqua.size_bytes) == wqua
          and vsmp.deletable is False and vsmp.integrity_verified is True
          and (vsmp.files, vsmp.size_bytes) == wsmp
          and [b.detail for b in vsmp.blockers] ==
          [f"sample-quality measurement(s) '{qua.evaluation_id}'"])

    # ---- §5 invariant, blocked direction: DELETE refuses, same blockers -
    copy_before = inventory(copy_root)
    try:
        forge.delete_sample(fresh, smp.sample_id)
        refused = False
    except ValueError:
        refused = True
    check("C3 copy: sample DELETE refused; view blockers == guard blockers",
          refused and inventory(copy_root) == copy_before)

    # ---- deletable direction: stats == the views ----------------------- #
    r = forge.delete_sample_evaluation(fresh, qua.evaluation_id)
    check("C4 copy: measurement DELETE stats == its retention view",
          r.files_removed == len(vqua.files)
          and r.bytes_reclaimed == vqua.size_bytes)
    flipped = forge.sample_retention_overview(fresh, smp.sample_id)
    check("C5 copy: sample retention flips to deletable, live",
          flipped.deletable is True and flipped.blockers == [])
    r = forge.delete_sample(fresh, smp.sample_id)
    check("C6 copy: sample DELETE stats == its flipped view",
          r.files_removed == len(flipped.files)
          and r.bytes_reclaimed == flipped.size_bytes)
    r = forge.delete_suite_run(fresh, run.suite_run_id)
    check("C7 copy: suite-run DELETE stats == its retention view",
          r.files_removed == len(vrun.files)
          and r.bytes_reclaimed == vrun.size_bytes)

    # the model's external references drained -> deletable -> stats ==
    # its retention view (the full M67/M68 chain)
    mret = forge.model_retention_overview(fresh)
    check("C8 copy: model unblocked; DELETE stats == retention view",
          mret.deletable is True and mret.blockers == []
          and forge.model_usage_overview(fresh).external_references == 0)
    mres = forge.delete_model(fresh)
    check("C9 copy: model DELETE stats == retention view",
          mres.files_removed == len(mret.files)
          and mres.bytes_reclaimed == mret.size_bytes)

    # the production model on the copy is untouched and still blocked
    check("C10 copy: production model untouched, still blocked",
          forge.model_retention_overview(MODEL_ID).deletable is False)
finally:
    shutil.rmtree(copy_root, ignore_errors=True)
check("C11 copy discarded", not copy_root.exists())

# final production audit
check("A2 production byte-identical, final (52/52)",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M68-DELTA RETENTION-VIEW CHECKS "
      "PASSED (production untouched; §5 invariant on a discarded copy)")
sys.exit(0 if n_pass == len(results) else 1)
