"""M69 LIVE certification smoke (READ-ONLY over production, executed
exactly ONCE).

Certifies `GET /api/v1/models/{model_id}/records/usage` — the
model-owned record usage overview — against the production root:
every category cross-checked against an INDEPENDENT manifest-parsing
oracle, M66/M62 coherence, the M61 lineage classification, and a
disposable-copy phase proving EXTERNAL reference discovery live
(production holds no suite runs/samples/measurements). Zero state
mutation anywhere; the copy is discarded.

Certified facts of this production root (byte-identical since the
M68 delta — m69_pre.sha256 == m68_retention_pre.sha256, 52/52):
  1 model (a266c8480cb7, live-model); M66 usage: training_run 6,
  checkpoint 13, workflow 3, evaluation 4, comparison 2, gate 2,
  suite_run 0, sample 0, sample_quality 0, workflow_recipe 1,
  policy 0 -> internal 30, external 1; M67 retention: deletable
  False (the m62-live-loop recipe); M69 records usage: 30 records
  (6/13/3/4/2/2), 57 references, 57 internal, 0 external.
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

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8793"
API = BASE + "/api/v1"

MODEL_ID = "a266c8480cb7"
DS_ID = "5062cc7e0954"
TOK_ID = "f24b699890eb"
M66_CATS = {"training_run": 6, "checkpoint": 13, "workflow": 3,
            "evaluation": 4, "comparison": 2, "gate": 2, "suite_run": 0,
            "sample": 0, "sample_quality": 0, "workflow_recipe": 1,
            "policy": 0}
ORDER = ["training_run", "checkpoint", "workflow", "evaluation",
         "comparison", "gate"]
TOT_RECS, TOT_REFS, INT_REFS, EXT_REFS = 30, 57, 57, 0
N_CKPTS, N_DELETABLE, N_PROTECTED = 13, 10, 3
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m69_pre.sha256"
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


# ------------------- the INDEPENDENT oracle (no app code) ------------------ #
KIND_TO_FAMILY = {"training_report": "training_run",
                  "evaluation": "evaluation",
                  "comparison": "comparison",
                  "gate_decision": "gate"}
REF_CATS = ["model", "checkpoint", "run_provenance", "workflow",
            "evaluation", "comparison", "gate", "suite_run", "sample",
            "sample_quality"]


def oracle(root: Path, model_id: str) -> dict:
    refs: dict = {}

    def add(family, rid, category, ref_id):
        refs.setdefault((family, rid), set()).add((category, ref_id))

    mdir = root / "models" / model_id
    man = json.loads((mdir / "manifest.json").read_text())
    if man.get("latest_checkpoint"):
        add("checkpoint", man["latest_checkpoint"], "model", model_id)
    if man.get("best_checkpoint"):
        add("checkpoint", man["best_checkpoint"], "model", model_id)
    for p in man.get("training_provenance", []):
        for key in ("parent_checkpoint_id", "initial_checkpoint_id",
                    "final_checkpoint_id", "rolled_back_to"):
            if p.get(key):
                add("checkpoint", p[key], "run_provenance", model_id)

    def family_records(fam):
        froot = mdir / fam
        out = []
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if d.is_dir():
                    out.append(json.loads(
                        (d / "manifest.json").read_text()))
        return out

    for c in family_records("checkpoints"):
        add("training_run", c["run_id"], "checkpoint",
            c["checkpoint_id"])
        if c.get("parent_checkpoint_id"):
            add("checkpoint", c["parent_checkpoint_id"], "checkpoint",
                c["checkpoint_id"])
    for e in family_records("evaluations"):
        if e.get("checkpoint_id"):
            add("checkpoint", e["checkpoint_id"], "evaluation",
                e["eval_id"])
    for c in family_records("comparisons"):
        for side in (c["state_a"], c["state_b"]):
            if side.get("checkpoint_id"):
                add("checkpoint", side["checkpoint_id"], "comparison",
                    c["comparison_id"])
            add("evaluation", side["evaluation_id"], "comparison",
                c["comparison_id"])
    for g in family_records("gates"):
        for side in (g.get("candidate"), g.get("baseline")):
            if side:
                if side.get("checkpoint_id"):
                    add("checkpoint", side["checkpoint_id"], "gate",
                        g["decision_id"])
                add("evaluation", side["evaluation_id"], "gate",
                    g["decision_id"])
        if g.get("comparison_id"):
            add("comparison", g["comparison_id"], "gate",
                g["decision_id"])
    for w in family_records("workflows"):
        for st in w.get("stages", []):
            art = st.get("artifact")
            if not art:
                continue
            fam = KIND_TO_FAMILY.get(art.get("kind"))
            if fam:
                add(fam, art["artifact_id"], "workflow",
                    w["workflow_id"])
            if art.get("checkpoint_id"):
                add("checkpoint", art["checkpoint_id"], "workflow",
                    w["workflow_id"])
        if w.get("suggested_checkpoint_id"):
            add("checkpoint", w["suggested_checkpoint_id"], "workflow",
                w["workflow_id"])
    sroot = root / "suite-runs"
    if sroot.exists():
        for d in sorted(sroot.iterdir()):
            if not d.is_dir():
                continue
            r = json.loads((d / "manifest.json").read_text())
            if r.get("model_id") != model_id:
                continue
            if r.get("state", {}).get("checkpoint_id"):
                add("checkpoint", r["state"]["checkpoint_id"],
                    "suite_run", r["suite_run_id"])
            for pr in r.get("results", []):
                if pr.get("evaluation_id"):
                    add("evaluation", pr["evaluation_id"], "suite_run",
                        r["suite_run_id"])
    for fam, cat, idf in (("samples", "sample", "sample_id"),
                          ("sample-evaluations", "sample_quality",
                           "evaluation_id")):
        froot = root / fam / model_id
        if froot.exists():
            for d in sorted(froot.iterdir()):
                if not d.is_dir():
                    continue
                rec = json.loads((d / "manifest.json").read_text())
                add("checkpoint", rec["checkpoint_id"], cat, rec[idf])

    families = {
        "training_run": sorted(p["run_id"] for p in
                               man.get("training_provenance", [])),
        "checkpoint": sorted(c["checkpoint_id"] for c in
                             family_records("checkpoints")),
        "workflow": sorted(w["workflow_id"] for w in
                           family_records("workflows")),
        "evaluation": sorted(e["eval_id"] for e in
                             family_records("evaluations")),
        "comparison": sorted(c["comparison_id"] for c in
                             family_records("comparisons")),
        "gate": sorted(g["decision_id"] for g in
                       family_records("gates")),
    }
    order = {c: i for i, c in enumerate(REF_CATS)}
    return {"refs": {k: sorted(v, key=lambda cr: (order[cr[0]], cr[1]))
                     for k, v in refs.items()},
            "families": families}


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
check("P2 disk == m69_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

models, _ = jget("/models")
check("P3 model registry", [m["id"] for m in models] == [MODEL_ID])

usage, _ = jget(f"/models/{MODEL_ID}/usage")
ucats = {c["category"]: c["references"] for c in usage["categories"]}
check("P4 M66 usage coherent (certified counts)",
      all(len(ucats[c]) == M66_CATS[c] for c in M66_CATS)
      and usage["internal_references"] == 30
      and usage["external_references"] == 1)

ret62, _ = jget(f"/models/{MODEL_ID}/checkpoints/retention")
check("P5 M62 retention coherent (13/10/3)",
      ret62["total_checkpoints"] == N_CKPTS
      and ret62["deletable_checkpoints"] == N_DELETABLE
      and ret62["protected_checkpoints"] == N_PROTECTED)

ora = oracle(root, MODEL_ID)

# ===================== the M69 overview (once) ============================= #
ov, body = jget(f"/models/{MODEL_ID}/records/usage")
check("U1 usage readable + identity + canonical order",
      ov["model_id"] == MODEL_ID and ov["name"] == "live-model"
      and [c["category"] for c in ov["categories"]] == ORDER)

counts = {c["category"]: len(c["records"]) for c in ov["categories"]}
check("U2 record counts == the ONE listings == the oracle",
      counts == {"training_run": 6, "checkpoint": 13, "workflow": 3,
                 "evaluation": 4, "comparison": 2, "gate": 2}
      and all(counts[c] == len(ora["families"][c]) for c in ORDER)
      and ov["total_records"] == TOT_RECS,
      " ".join(f"{c}={counts[c]}" for c in ORDER))

ok_u3 = True
for c in ov["categories"]:
    if [r["record_id"] for r in c["records"]] != ora["families"][c["category"]]:
        ok_u3 = False
    for r in c["records"]:
        got = [(x["category"], x["reference_id"]) for x in r["references"]]
        if got != ora["refs"].get((c["category"], r["record_id"]), []):
            ok_u3 = False
        if r["total_references"] != len(r["references"]):
            ok_u3 = False
        if r["external_references"] != sum(
                1 for x in r["references"] if x["external"]):
            ok_u3 = False
check("U3 EVERY record's references == the independent oracle "
      "(ids, categories, order, counts)", ok_u3)

check("U4 aggregates == exact sums + certified totals",
      ov["total_references"] == sum(r["total_references"] for c in
                                    ov["categories"] for r in c["records"])
      == TOT_REFS
      and ov["internal_references"] == INT_REFS
      and ov["external_references"] == EXT_REFS
      and ov["total_records"] == TOT_RECS)

# M66 agreement: the internal category record sets are identical
ok_u5 = True
for c in ORDER:
    if sorted(r["record_id"] for r in
              next(x for x in ov["categories"]
                   if x["category"] == c)["records"]) != \
            sorted(ucats[c]):
        ok_u5 = False
check("U5 M66 agreement: internal category record sets identical", ok_u5)

# M61 lineage classification: checkpoints whose ONLY references are
# lineage edges have EMPTY M62 retention blockers (never blocked)
lin = {r["record_id"] for r in
       next(x for x in ov["categories"] if x["category"] == "checkpoint")
       ["records"]
       if r["references"] and all(
           x["category"] in ("checkpoint", "run_provenance")
           for x in r["references"])}
ret62_entries = {e["checkpoint_id"]: e for e in ret62["checkpoints"]}
ok_u6 = bool(lin) and all(ret62_entries[rid]["blockers"] == []
                          for rid in lin if rid in ret62_entries)
check("U6 M61 lineage consistency: lineage-only checkpoints have "
      "empty M62 blockers", ok_u6, f"{len(lin)} lineage-only")

# unknown model -> 404; determinism; OpenAPI
try:
    urllib.request.urlopen(API + "/models/no-m69/records/usage",
                           timeout=30)
    u404 = False
except urllib.error.HTTPError as e:
    u404 = e.code == 404
_, body2 = jget(f"/models/{MODEL_ID}/records/usage")
with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
check("U7 unknown model -> 404; deterministic byte-identical repeat",
      u404 and body2 == body)
check("U8 OpenAPI: 97 paths, GET-only route, 7 deletes (unchanged)",
      len(spec["paths"]) == 97
      and set(spec["paths"]["/api/v1/models/{model_id}/records/usage"]
              .keys()) == {"get"}
      and len([p for p, ops in spec["paths"].items()
               if "delete" in ops]) == 7
      and all(s in spec["components"]["schemas"] for s in (
          "RecordReference", "ModelOwnedRecordUsage",
          "ModelOwnedRecordCategory", "ModelRecordsUsageOverview")))

check("A1 production byte-identical after all read-only checks",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()),
      f"{len(pre_disk)} files")

# ============== disposable copy: EXTERNAL discovery, live ================== #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.engine import ModelForge  # noqa: E402
from app.schemas import (  # noqa: E402
    ComparisonState, EvalStateKind, ProbeSuiteCreateRequest,
    SampleGenerateRequest, SampleStrategy, SuiteProbe, SuiteRunRequest)

copy_root = Path(tempfile.mkdtemp(prefix="m69-copy-"))
try:
    shutil.copytree(PROD_ROOT, copy_root, dirs_exist_ok=True)
    forge = ModelForge(root=copy_root)

    # external references created on the copy only: a suite run (state
    # checkpoint + probe evaluation) + a sample + a measurement
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m69-suite", description="M69 smoke",
        probes=[SuiteProbe(dataset_id=DS_ID, split="validation",
                           tokenizer_id=TOK_ID, batch_size=8,
                           max_seq_len=32, seed=42)]))
    ck = sorted(c.checkpoint_id for c in
                forge.training.list_checkpoints(MODEL_ID))
    run = forge.run_suite(SuiteRunRequest(
        model_id=MODEL_ID, suite_id="m69-suite",
        state=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                              checkpoint_id=ck[0])))
    smp = forge.generate_sample(SampleGenerateRequest(
        model_id=MODEL_ID, checkpoint_id=ck[1], tokenizer_id=TOK_ID,
        prompt="river mountain cloud", strategy=SampleStrategy.GREEDY,
        max_new_tokens=6))
    qua = forge.evaluate_sample(MODEL_ID, smp.sample_id)

    cov = forge.model_records_usage_overview(MODEL_ID)
    flat = {(c.category, r.record_id): r for c in cov.categories
            for r in c.records}
    ext_entries = [(x.category, x.reference_id)
                   for c in cov.categories for r in c.records
                   for x in r.references if x.external]
    # category iteration order: the checkpoint-category external
    # entries first (state ck, then sample + measurement on the later
    # checkpoint), then the evaluation-category probe entry
    check("C1 copy: EXTERNAL references discovered (4 entries)",
          ext_entries == [
              ("suite_run", run.suite_run_id),      # state checkpoint
              ("sample", smp.sample_id),
              ("sample_quality", qua.evaluation_id),
              ("suite_run", run.suite_run_id)]      # probe evaluation
          and cov.external_references == 4
          # the suite run's probe EXECUTES a new evaluation record:
          # +1 record (evaluation family 4 -> 5) whose own checkpoint
          # reference is +1 INTERNAL entry
          and cov.internal_references == INT_REFS + 1
          and cov.total_references == TOT_REFS + 5
          and cov.total_records == 31,
          "suite-run state ck + probe eval + sample + measurement")
    # the suite run's state checkpoint and probe eval are the targets
    check("C2 copy: external targets correct",
          ("suite_run", run.suite_run_id) in [
              (x.category, x.reference_id) for x in
              flat[("checkpoint", ck[0])].references]
          and any(
              ("suite_run", run.suite_run_id) in [
                  (x.category, x.reference_id) for x in
                  flat[("evaluation", pr.evaluation_id)].references]
              for pr in run.results if pr.evaluation_id))
    # oracle parity on the copy (external families included)
    cora = oracle(copy_root, MODEL_ID)
    ok_c3 = True
    for c in cov.categories:
        for r in c.records:
            if [(x.category, x.reference_id) for x in r.references] != \
                    cora["refs"].get((c.category, r.record_id), []):
                ok_c3 = False
    check("C3 copy: EVERY record == the independent oracle "
          "(external families included)", ok_c3)
    # determinism on the copy
    check("C4 copy: deterministic repeat",
          forge.model_records_usage_overview(
              MODEL_ID).model_dump(mode="json") ==
          cov.model_dump(mode="json"))
finally:
    shutil.rmtree(copy_root, ignore_errors=True)
check("C5 copy discarded", not copy_root.exists())

# final production audit
check("A2 production byte-identical, final (52/52)",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M69 READ-ONLY CHECKS PASSED "
      "(external discovery on a discarded copy)")
sys.exit(0 if n_pass == len(results) else 1)
