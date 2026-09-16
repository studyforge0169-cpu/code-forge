"""M68 LIVE certification smoke (executed exactly ONCE).

Certifies the explicit suite-run & sample lifecycle against the
production root — read-only over production (the new DELETE routes'
404 behavior on unknown ids, the M67 protected DELETE unchanged), then
the SUCCESS paths on a DISPOSABLE COPY: create a fresh model + suite
run + sample + quality measurement, watch it get BLOCKED, drain the
three references through the new M68 deletions (exact accounting each
time), and finally delete the model — the full M68 payoff — with the
copy returning byte-for-byte to its post-copy state before disposal.

Certified facts of this production root (rebuilt after the fourth
sandbox re-provisioning; m68_pre.sha256, 52/52):
  1 model (3770ca1bfa23, live-model); M66 usage (11 categories):
  training_run 6, checkpoint 13, workflow 3, evaluation 4,
  comparison 2, gate 2, suite_run 0, sample 0, sample_quality 0,
  workflow_recipe 1 (m62-live-loop), policy 0 -> internal 30,
  external 1, total 31; M67 retention: integrity True, deletable
  False, 40 files / 8,874,613 B, blockers exactly
  [workflow_recipe: m62-live-loop]; dataset e7ee867f06df 16 refs;
  tokenizer dbb34eaa8804 15 refs; M62 retention 13/10/3;
  52 files / 8,926,410 B.
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

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8791"
API = BASE + "/api/v1"

MODEL_ID = "3770ca1bfa23"
DS_ID = "e7ee867f06df"
TOK_ID = "dbb34eaa8804"
RECIPE_ID = "m62-live-loop"
CATS = {"training_run": 6, "checkpoint": 13, "workflow": 3, "evaluation": 4,
        "comparison": 2, "gate": 2, "suite_run": 0, "sample": 0,
        "sample_quality": 0, "workflow_recipe": 1, "policy": 0}
INTERNAL, EXTERNAL, TOTAL = 30, 1, 31
RET_FILES, RET_BYTES = 40, 8874613
N_CKPTS, N_DELETABLE, N_PROTECTED = 13, 10, 3
DS_TOTAL, TOK_TOTAL = 16, 15
ORDER = ["training_run", "checkpoint", "workflow", "evaluation",
         "comparison", "gate", "suite_run", "sample", "sample_quality",
         "workflow_recipe", "policy"]
INTERNAL_SET = set(ORDER[:6])
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m68_pre.sha256"
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


def walk_stats(directory: Path) -> tuple[int, int]:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return len(files), sum(p.stat().st_size for p in files)


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
check("P2 disk == m68_pre.sha256 (52/52)", pre_disk == pre_ref,
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

ov, usage_body = jget(f"/models/{MODEL_ID}/usage")
cats = {c["category"]: c["references"] for c in ov["categories"]}
ora = oracle(root, MODEL_ID)
check("P7 M66 usage: certified counts + independent oracle",
      [c["category"] for c in ov["categories"]] == ORDER
      and all(len(cats[c]) == CATS[c] for c in ORDER)
      and all(cats[c] == ora[c] for c in ORDER)
      and ov["internal_references"] == INTERNAL
      and ov["external_references"] == EXTERNAL
      and ov["total_references"] == TOTAL,
      " ".join(f"{c}={len(cats[c])}" for c in ORDER))

ret, ret_body = jget(f"/models/{MODEL_ID}/retention")
check("P8 M67 retention coherent (blocked by the recipe only)",
      ret["integrity_verified"] is True and ret["deletable"] is False
      and len(ret["files"]) == RET_FILES
      and ret["size_bytes"] == RET_BYTES
      and [(b["category"], b["reference_id"]) for b in ret["blockers"]]
      == [("workflow_recipe", RECIPE_ID)])

# ==================== the new routes over production (read-only) =========== #
# production has NO suite runs / samples / measurements: the new DELETE
# routes' production-exercisable surface is exactly the unknown-id 404
def dstatus(path: str) -> int:
    try:
        urllib.request.urlopen(
            urllib.request.Request(API + path, method="DELETE"), timeout=30)
        return 0
    except urllib.error.HTTPError as e:
        return e.code


n1 = all(dstatus(p) == 404 for p in (
    f"/models/{MODEL_ID}/suite-runs/no-m68",
    f"/models/{MODEL_ID}/samples/no-m68",
    f"/models/{MODEL_ID}/sample-quality/no-m68",
    "/models/no-m68/suite-runs/x",
    "/models/no-m68/samples/x",
    "/models/no-m68/sample-quality/x"))
check("N1 new DELETE routes: unknown ids -> 404", n1)

# the M67 protected DELETE is unchanged (still the recipe blocker)
try:
    urllib.request.urlopen(
        urllib.request.Request(API + f"/models/{MODEL_ID}",
                               method="DELETE"), timeout=30)
    ok, detail = False, {}
except urllib.error.HTTPError as e:
    ok = e.code == 409
    detail = json.loads(e.read())["detail"]
check("N2 M67 model DELETE unchanged (409, recipe blocker)",
      ok and detail["protected"] is True
      and [(b["category"], b["reference_id"]) for b in detail["blockers"]]
      == [("workflow_recipe", RECIPE_ID)])

_, usage_body2 = jget(f"/models/{MODEL_ID}/usage")
_, ret_body2 = jget(f"/models/{MODEL_ID}/retention")
check("N3 deterministic byte-identical repeats",
      usage_body2 == usage_body and ret_body2 == ret_body)

with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
deletes = sorted(p for p, ops in spec["paths"].items() if "delete" in ops)
three = ("/api/v1/models/{model_id}/suite-runs/{suite_run_id}",
         "/api/v1/models/{model_id}/samples/{sample_id}",
         "/api/v1/models/{model_id}/sample-quality/{evaluation_id}")
check("O1 OpenAPI: 93 paths (zero new), 7 delete operations",
      len(spec["paths"]) == 93
      and deletes == [
          "/api/v1/datasets/{dataset_id}",
          "/api/v1/models/{model_id}",
          "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}",
          "/api/v1/models/{model_id}/sample-quality/{evaluation_id}",
          "/api/v1/models/{model_id}/samples/{sample_id}",
          "/api/v1/models/{model_id}/suite-runs/{suite_run_id}",
          "/api/v1/tokenizers/{tokenizer_id}"]
      and all(set(spec["paths"][p].keys()) == {"get", "delete"}
              for p in three)
      and spec["paths"]["/api/v1/models/{model_id}/samples/{sample_id}"][
          "delete"]["responses"]["409"]["content"]["application/json"][
          "schema"]["$ref"] == "#/components/schemas/SampleDeletionBlocked"
      and all(s in spec["components"]["schemas"] for s in (
          "SuiteRunDeletionResult", "SampleDeletionResult",
          "SampleEvaluationDeletionResult", "SampleDeletionBlocked")))

check("A1 production byte-identical after the read-only phase",
      inventory(root) == pre_disk, f"{len(pre_disk)} files")

# ===================== disposable-copy success paths ======================= #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.engine import ModelForge  # noqa: E402
from app.schemas import (  # noqa: E402
    ComparisonState, EvalStateKind, ModelCreateRequest,
    ProbeSuiteCreateRequest, SampleGenerateRequest, SampleStrategy,
    SuiteProbe, SuiteRunRequest, TransformerConfig, TrainingConfig)

copy_root = Path(tempfile.mkdtemp(prefix="m68-copy-"))
try:
    shutil.copytree(PROD_ROOT, copy_root, dirs_exist_ok=True)
    forge = ModelForge(root=copy_root)

    # guard parity ON THE COPY: the production model is still blocked
    post_copy = inventory(copy_root)
    try:
        forge.delete_model(MODEL_ID)
        refused = False
    except ValueError:
        refused = True
    check("C1 copy: production model still refused (guard parity)",
          refused and inventory(copy_root) == post_copy)

    # the copy's own probe suite (production has none; an immutable
    # definition that legitimately persists on the copy — the C9
    # full-circle check accounts for exactly this one manifest)
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m68-copy-suite", description="M68 smoke fixture",
        probes=[SuiteProbe(dataset_id=DS_ID, split="validation",
                           tokenizer_id=TOK_ID, batch_size=8,
                           max_seq_len=32, seed=42)]))
    suite_manifest = (copy_root / "probe-suites" / "m68-copy-suite"
                      / "manifest.json")

    # a fresh model, TRAINED on the copy's existing dataset+tokenizer,
    # then blocked by ALL THREE new external reference families
    fresh = forge.create_model(ModelCreateRequest(config=TransformerConfig(
        name="m68-copy-model", vocab_size=640, context_length=64,
        hidden_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=128, seed=99)))[0].id
    forge.run_training(TrainingConfig(
        method="continued_pretraining", model_id=fresh, dataset_id=DS_ID,
        tokenizer_id=TOK_ID, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=2, keep_best=False, seed=99,
        steps=4))
    run = forge.run_suite(SuiteRunRequest(
        model_id=fresh, suite_id="m68-copy-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    best = forge.select_best_checkpoint(fresh).checkpoint.checkpoint_id
    smp = forge.generate_sample(SampleGenerateRequest(
        model_id=fresh, checkpoint_id=best, tokenizer_id=TOK_ID,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=8))
    sq = forge.evaluate_sample(fresh, smp.sample_id)

    ret = forge.model_retention_overview(fresh)
    check("C2 copy: fresh model blocked by all THREE new families",
          ret.deletable is False
          and [(b.category, b.reference_id) for b in ret.blockers] == [
              ("suite_run", run.suite_run_id),
              ("sample", smp.sample_id),
              ("sample_quality", sq.evaluation_id)],
          "canonical order suite_run < sample < sample_quality")

    # the sample deletion is REFUSED (typed blocker), zero mutation
    before_refusal = inventory(copy_root)
    try:
        forge.delete_sample(fresh, smp.sample_id)
        refused = False
    except ValueError:
        refused = True
    blockers = forge.sample_deletion_blockers(fresh, smp.sample_id)
    check("C3 copy: sample deletion refused by its measurement",
          refused and len(blockers) == 1
          and blockers[0].reason == "sample_quality"
          and sq.evaluation_id in blockers[0].detail
          and inventory(copy_root) == before_refusal)

    # drain: measurement -> sample -> suite run, exact accounting each
    # (stats captured from independent walks BEFORE each deletion)
    qdir = (copy_root / "sample-evaluations" / fresh
            / f"evaluation-{sq.evaluation_id}")
    q_stats = walk_stats(qdir)
    rq = forge.delete_sample_evaluation(fresh, sq.evaluation_id)
    check("C4 copy: measurement deleted, exact stats",
          rq.sample_id == smp.sample_id
          and (rq.files_removed, rq.bytes_reclaimed) == q_stats
          and not qdir.exists())

    sdir = copy_root / "samples" / fresh / f"sample-{smp.sample_id}"
    s_stats = walk_stats(sdir)
    rs = forge.delete_sample(fresh, smp.sample_id)
    check("C5 copy: sample deleted (now unblocked), exact stats",
          (rs.files_removed, rs.bytes_reclaimed) == s_stats
          and not sdir.exists()
          and [(b.category, b.reference_id)
               for b in forge.model_retention_overview(fresh).blockers]
          == [("suite_run", run.suite_run_id)])

    # the suite run's probe evaluations are MODEL-OWNED: preserved
    evals_after_probe = len(forge.list_evaluations(fresh))
    rdir = copy_root / "suite-runs" / run.suite_run_id
    r_stats = walk_stats(rdir)
    rr = forge.delete_suite_run(fresh, run.suite_run_id)
    check("C6 copy: suite run deleted, exact stats, NO cascade",
          (rr.files_removed, rr.bytes_reclaimed) == r_stats
          and not rdir.exists()
          and len(forge.list_evaluations(fresh)) == evals_after_probe,
          f"{evals_after_probe} probe evaluations preserved")

    # the payoff: the model is NOW deletable — and deleting it returns
    # the copy byte-for-byte to its post-copy state
    ret = forge.model_retention_overview(fresh)
    check("C7 copy: model unblocked after the drain",
          ret.deletable is True and ret.blockers == []
          and forge.model_usage_overview(fresh).external_references == 0)
    res = forge.delete_model(fresh)
    check("C8 copy: model deleted, exact stats == retention view",
          res.model_id == fresh
          and (res.files_removed, res.bytes_reclaimed)
          == (len(ret.files), ret.size_bytes)
          and not (copy_root / "models" / fresh).exists())
    expected_final = dict(post_copy)
    expected_final["probe-suites/m68-copy-suite/manifest.json"] = \
        hashlib.sha256(suite_manifest.read_bytes()).hexdigest()
    check("C9 copy: FULL-CIRCLE — post-copy state + the one immutable"
          " suite definition, nothing else",
          inventory(copy_root) == expected_final
          and len(forge.storage.model_ids()) == 1
          and not any(p.name.startswith(".tmp-delete")
                      for p in (copy_root / "models").iterdir()))
finally:
    shutil.rmtree(copy_root, ignore_errors=True)
check("C10 copy discarded", not copy_root.exists())

# final production audit
check("A2 production byte-identical, final (52/52)",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M68 CHECKS PASSED "
      "(production untouched; success paths on a discarded copy)")
sys.exit(0 if n_pass == len(results) else 1)
