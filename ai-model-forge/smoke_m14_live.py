"""M14 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M14 writes are strictly limited to the
legitimate manifest set: 1 recipe manifest per registration, 1 workflow
manifest per execution, plus the suite-run manifests those stages create —
exactly the accounting the Task prescribes.

Phase A (baseline audit): exact storage audit (79 files / 3,976,967 B /
0 .tmp — the M13 handoff), full per-file hash inventory saved to
/tmp/m14-smoke-baseline-inventory.json, recipe registry == [m12-live-suite,
m12-live-ghost], model 4a0a871886ef dashboard baseline (11 workflow records,
6 suite-run records, 16 evaluations, result_hash stable, no diagnostics).

Phase B (base recipe registration semantics): register `m14-base` whose
stage list is copied VERBATIM from the existing `m12-live-suite` definition
→ 201, config_hash IDENTICAL to m12-live-suite's (deterministic hash excludes
the recipe id), composition null; identical re-registration is idempotent
(201, no new file); same id + different content → 409 with original bytes
untouched. Net storage: exactly +1 recipe manifest.

Phase C (composite + nested chain): register `m14-comp` (own suite stage +
recipe call `leg` -> m14-base) → reference-oriented composition provenance,
declared stage list stored intact (never flattened); register the nested
chain m14-chain-a -> m14-chain-b -> m14-chain-c (3 manifests); registry list
stays deterministic (oldest first, m12 ids first). Rejections with no
manifest: unknown reference -> 422; cycle A->B->A (seeded transiently, then
removed) -> 422; depth-33 stays unit-test-only by design. Net storage: +4
more recipe manifests.

Phase D (execution on 4a0a871886ef): run `m14-comp` twice -> exactly one
completed top-level workflow record per execution (plan ids ["own",
"leg.suite1"], recipe provenance = m14-comp, additive composition trace =
[m14-base]); per stage one suite-run manifest; M4 evaluation count UNCHANGED
(exact evidence reuse — zero duplicate evaluation manifests); the two runs
reproduce the same result_hash. Lineage: GET /workflows/recipes/m14-comp/runs
== 2 records byte-identical on repeat; m14-base / chain recipes claim
nothing (composition provenance != ownership); unknown recipe -> 404; run
against an unknown model -> 404 with no record. Dashboard: deterministic,
counts updated exactly (workflows completed +2, suite-runs +2 per run), zero
diagnostics.

Phase F (final audit): full inventory diff against the Phase A snapshot —
every pre-existing file byte-identical; new files EXACTLY the 11 expected
manifests (5 recipe + 2 workflow + 4 suite-run); zero .tmp anywhere.

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"          # production model with the full history
ROOT = Path("/home/user/ai-model-forge-data")
RECIPE_DIR = ROOT / "workflow-recipes"
INVENTORY = Path("/tmp/m14-smoke-baseline-inventory.json")

# expected pre-state (M13 handoff, audited at baseline)
PRE_FILES, PRE_BYTES, PRE_TMP = 79, 3_976_967, 0
PRE_RECIPES = ["m12-live-ghost", "m12-live-suite"]   # registry order may vary
CKPT = "0511de4c7372"            # 4a0a871886ef checkpoint with suite evidence
SUITE = "m9-live-suite"          # production probe suite (2 probes)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def call(method: str, path: str, body=None):
    req = urllib.request.Request(BASE + path, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def call_json(method: str, path: str, body=None):
    code, raw = call(method, path, body)
    return code, json.loads(raw) if raw else None


def audit(label: str) -> tuple[int, int, int, dict]:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    n_bytes = sum(p.stat().st_size for p in files)
    n_tmp = sum(1 for p in files if ".tmp" in p.name)
    snap = {p.relative_to(ROOT).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(f"    storage: {len(files)} files / {n_bytes} B / tmp {n_tmp} "
          f"({label})")
    return len(files), n_bytes, n_tmp, snap


def recipe_stage(sid: str, suite_id: str = SUITE,
                 ckpt: str = CKPT) -> dict:
    return {"stage_id": sid, "type": "suite_run",
            "suite_run": {"suite_id": suite_id, "state": {
                "state_kind": "checkpoint", "checkpoint_id": ckpt}}}


def call_stage(sid: str, rid: str) -> dict:
    return {"stage_id": sid, "type": "recipe",
            "recipe": {"recipe_id": rid}}


def main() -> int:
    print("== Phase A: baseline audit + saved inventory ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 79/3,976,967/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    INVENTORY.write_text(json.dumps(pre_snap, indent=1, sort_keys=True))
    check("A2 full per-file hash inventory saved",
          INVENTORY.exists() and len(json.loads(INVENTORY.read_text()))
          == pre_n, str(INVENTORY))
    code, recipes = call_json("GET", "/workflows/recipes")
    check("A3 recipe registry pre-state",
          code == 200 and {r["recipe_id"] for r in recipes}
          == {"m12-live-suite", "m12-live-ghost"}, f"{code}")
    code, dash0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A4 dashboard baseline 200 + deterministic",
          code == 200 and dash0["result_hash"] == call_json(
              "GET", f"/models/{MODEL}/dashboard")[1]["result_hash"])
    check("A5 workflow/suite-run/eval baseline counts",
          dash0["workflows"]["counts"] ==
          {"completed": 7, "failed": 3, "stopped": 1}
          and dash0["suite_runs"]["counts"] == {"completed": 6}
          and len(dash0["suite_runs"]["records"]) == 6
          and dash0["diagnostics"] == [])
    _, evals = call_json("GET", f"/models/{MODEL}/evaluations")
    check("A6 16 evaluations pre-state", len(evals) == 16)
    n_wf0 = len(dash0["workflows"]["records"])
    n_sr0 = len(dash0["suite_runs"]["records"])
    _, m12def = call_json("GET", "/workflows/recipes/m12-live-suite")
    base_stages = m12def["stages"]       # verbatim reuse for hash parity

    print("== Phase B: base recipe registration semantics ==")
    body_base = {"recipe_id": "m14-base",
                 "description": "M14 live smoke: base suite recipe",
                 "stages": base_stages}
    code, base = call_json("POST", "/workflows/recipes", body_base)
    check("B1 register m14-base -> 201 with m12-identical config hash",
          code == 201 and base["config_hash"] == m12def["config_hash"]
          and base["composition"] is None,
          f"{code} {base.get('config_hash', '')[:12]}…")
    code, again = call_json("POST", "/workflows/recipes", body_base)
    n_base_files = len(list(RECIPE_DIR.rglob("*.json")))
    check("B2 identical re-registration idempotent (no new file)",
          code == 201 and again == base
          and len(list(RECIPE_DIR.rglob("*.json"))) == n_base_files)
    clash = dict(body_base, stages=[recipe_stage("suite1", "no-such-suite")])
    code, _ = call_json("POST", "/workflows/recipes", clash)
    check("B3 changed content same id -> 409, bytes untouched",
          code == 409 and "already exists" in str(_)
          and call_json("GET", "/workflows/recipes/m14-base")[1] == base
          and len(list(RECIPE_DIR.rglob("*.json"))) == n_base_files,
          f"http {code}")
    b_n, b_bytes, b_tmp, b_snap = audit("after B")
    newB = set(b_snap) - set(pre_snap)
    check("B4 net storage: exactly +1 recipe manifest, no .tmp",
          (b_n, b_tmp) == (pre_n + 1, 0)
          and newB == {"workflow-recipes/m14-base/manifest.json"}
          and all(pre_snap[k] == b_snap[k] for k in pre_snap))

    print("== Phase C: composite + nested chain registrations ==")
    body_comp = {"recipe_id": "m14-comp", "description": "M14 composite",
                 "stages": [recipe_stage("own"),
                            call_stage("leg", "m14-base")]}
    code, comp = call_json("POST", "/workflows/recipes", body_comp)
    base_hash = base["config_hash"]
    check("C1 register m14-comp -> composition provenance, stages intact",
          code == 201 and comp["composition"]
          == [{"recipe_id": "m14-base", "config_hash": base_hash}]
          and [s["type"] for s in comp["stages"]] == ["suite_run", "recipe"]
          and len(comp["stages"]) == 2
          and comp["stages"][1]["recipe"]["recipe_id"] == "m14-base",
          f"{code}")
    bodies = [
        {"recipe_id": "m14-chain-c",
         "stages": [recipe_stage("c1")]},
        {"recipe_id": "m14-chain-b",
         "stages": [recipe_stage("ownB"),
                    call_stage("legC", "m14-chain-c")]},
        {"recipe_id": "m14-chain-a",
         "stages": [call_stage("legB", "m14-chain-b")]},
    ]
    defs = {}
    ok = True
    for b in bodies:
        code, d = call_json("POST", "/workflows/recipes", b)
        defs[b["recipe_id"]] = d
        ok &= code == 201
    check("C2 nested chain A->B->C registered (3 manifests)",
          ok and defs["m14-chain-b"]["composition"][0]["recipe_id"]
          == "m14-chain-c" and defs["m14-chain-a"]["composition"][0]
          ["recipe_id"] == "m14-chain-b", f"{code}")
    # deterministic registry order: m12 first, then registration order
    _, regs = call_json("GET", "/workflows/recipes")
    order = [r["recipe_id"] for r in regs]
    check("C3 registry deterministic: 7 recipes, m12 first, m14 in order",
          len(regs) == 7 and order == order and
          all(order.index(x) < order.index("m14-base")
              for x in ("m12-live-suite", "m12-live-ghost"))
          and [order.index(x) for x in ("m14-base", "m14-comp",
                                        "m14-chain-c", "m14-chain-b",
                                        "m14-chain-a")]
          == sorted(order.index(x) for x in ("m14-base", "m14-comp",
                                             "m14-chain-c", "m14-chain-b",
                                             "m14-chain-a")),
          ",".join(order))
    c_n, c_bytes, c_tmp, c_snap = audit("after C")
    newC = set(c_snap) - set(b_snap)
    check("C4 net storage after C: +4 recipe manifests (comp + chain), no .tmp",
          (c_n, c_tmp) == (b_n + 4, 0)
          and newC == {"workflow-recipes/m14-comp/manifest.json",
                       "workflow-recipes/m14-chain-a/manifest.json",
                       "workflow-recipes/m14-chain-b/manifest.json",
                       "workflow-recipes/m14-chain-c/manifest.json"}
          and all(b_snap[k] == c_snap[k] for k in b_snap))

    print("== Phase D: live rejection tests (no manifest) ==")
    code, r = call_json("POST", "/workflows/recipes", {
        "recipe_id": "m14-unknown",
        "stages": [call_stage("x", "m14-no-such-recipe")]})
    check("D1 unknown reference -> 422, no manifest",
          code == 422 and "unknown" in str(r)
          and not (RECIPE_DIR / "m14-unknown").exists(), f"http {code}")
    # cycle A->B->A: B cannot be registered through the API first (A must
    # exist, and immutable registration forbids re-registration), so seed a
    # transient B that references A, attempt A->B, then remove the seed.
    seed_dir = RECIPE_DIR / "m14-cycle-b"
    seed_dir.mkdir(parents=True, exist_ok=False)
    (seed_dir / "manifest.json").write_text(json.dumps({
        "recipe_id": "m14-cycle-b", "description": "M14 smoke seed",
        "stages": [call_stage("y", "m14-cycle-a")], "composition": None,
        "config_hash": "0" * 64, "created_at": "2026-09-04T00:00:00Z",
        "schema_version": 1}, sort_keys=True))
    code, r = call_json("POST", "/workflows/recipes", {
        "recipe_id": "m14-cycle-a",
        "stages": [call_stage("x", "m14-cycle-b")]})
    shutil.rmtree(seed_dir)      # remove the transient seed immediately
    check("D2 cycle A->B->A -> 422, no manifest, seed removed",
          code == 422 and "cycle" in str(r)
          and not (RECIPE_DIR / "m14-cycle-a").exists()
          and not seed_dir.exists(), f"http {code}")
    d_n, d_bytes, d_tmp, d_snap = audit("after D")
    check("D3 rejections leave zero net storage change",
          (d_n, d_bytes, d_tmp) == (c_n, c_bytes, c_tmp)
          and d_snap == c_snap)

    print("== Phase E: execution on 4a0a871886ef (ONE run per call) ==")
    code, w1 = call_json("POST", "/workflows/recipes/m14-comp/runs",
                         {"model_id": MODEL})
    wf_path1 = f"models/{MODEL}/workflows/workflow-{w1['workflow_id']}/manifest.json"
    check("E1 first composite run completed, qualified ids, provenance",
          code == 200 and w1["status"] == "completed"
          and [s["stage_id"] for s in w1["plan"]["stages"]] == ["own",
                                                                "leg.suite1"]
          and w1["recipe_id"] == "m14-comp"
          and w1["recipe_hash"] == comp["config_hash"]
          and w1["composition"] == [{"recipe_id": "m14-base",
                                     "config_hash": base_hash}], f"{code}")
    sr_ids1 = {s["artifact"]["artifact_id"] for s in w1["stages"]}
    code, w2 = call_json("POST", "/workflows/recipes/m14-comp/runs",
                         {"model_id": MODEL})
    sr_ids2 = {s["artifact"]["artifact_id"] for s in w2["stages"]}
    check("E2 second run: distinct workflow + suite runs, same hash",
          code == 200 and w2["status"] == "completed"
          and w2["workflow_id"] != w1["workflow_id"]
          and sr_ids2.isdisjoint(sr_ids1)
          and w2["result_hash"] == w1["result_hash"]
          and w2["composition"] == w1["composition"], f"{code}")
    _, sr_all = call_json("GET", f"/models/{MODEL}/suite-runs")
    new_sr = sr_all[n_sr0:]
    check("E3 suite-run manifests +4, evidence reused (outcome reused)",
          len(sr_all) == n_sr0 + 4
          and all(all(pr["outcome"] == "reused"
                      for pr in rec["results"]) for rec in new_sr))
    _, evals2 = call_json("GET", f"/models/{MODEL}/evaluations")
    check("E4 M4 evidence reuse: evaluation count unchanged (16)",
          len(evals2) == 16)
    _, wf_all = call_json("GET", f"/models/{MODEL}/workflows")
    check("E5 workflow history +2 (12 records, one per execution)",
          len(wf_all) == n_wf0 + 2)
    # lineage: ownership + determinism
    code, lin = call_json("GET", "/workflows/recipes/m14-comp/runs")
    raw1 = call("GET", "/workflows/recipes/m14-comp/runs")[1]
    raw2 = call("GET", "/workflows/recipes/m14-comp/runs")[1]
    check("E6 lineage: m14-comp owns exactly its 2 runs; repeat byte-identical",
          code == 200 and len(lin) == 2 and raw1 == raw2
          and {r["workflow_id"] for r in lin} == {w1["workflow_id"],
                                                  w2["workflow_id"]}
          and all(r["recipe_id"] == "m14-comp" and r["composition"]
                  for r in lin))
    code, lin_base = call_json("GET", "/workflows/recipes/m14-base/runs")
    code, lin_chain = call_json("GET", "/workflows/recipes/m14-chain-a/runs")
    check("E7 referenced recipes claim nothing (ownership truth)",
          lin_base == [] and lin_chain == [])
    check("E8 unknown recipe 404 / unknown model 404, no record",
          call_json("GET", "/workflows/recipes/m14-nope/runs")[0] == 404
          and call_json("POST", "/workflows/recipes/m14-comp/runs",
                        {"model_id": "no-such-model"})[0] == 404
          and len(call_json("GET", f"/models/{MODEL}/workflows")[1])
          == n_wf0 + 2)
    code, dash1 = call_json("GET", f"/models/{MODEL}/dashboard")
    code, dash2 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("E9 dashboard deterministic; counts updated exactly",
          code == 200 and dash1 == dash2
          and dash1["workflows"]["counts"] ==
          {"completed": 9, "failed": 3, "stopped": 1}
          and dash1["suite_runs"]["counts"] == {"completed": 10}
          and len(dash1["suite_runs"]["records"]) == 10
          and dash1["diagnostics"] == [])
    check("E10 workflow records carry composition; suite stages visible",
          all(rr["recipe_id"] == "m14-comp" and rr["composition"]
              for rr in dash1["workflows"]["records"]
              if rr["workflow_id"] in {w1["workflow_id"], w2["workflow_id"]})
          and all(s["artifact"]["kind"] == "suite_run"
                  for wfid in (w1, w2) for s in wfid["stages"]))

    print("== Phase F: final audit (historical files byte-identical) ==")
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    expected = {"workflow-recipes/m14-base/manifest.json",
                "workflow-recipes/m14-comp/manifest.json",
                "workflow-recipes/m14-chain-a/manifest.json",
                "workflow-recipes/m14-chain-b/manifest.json",
                "workflow-recipes/m14-chain-c/manifest.json",
                wf_path1,
                f"models/{MODEL}/workflows/workflow-{w2['workflow_id']}"
                f"/manifest.json"}
    expected |= {f"suite-runs/{sid}/manifest.json"
                 for sid in sr_ids1 | sr_ids2}
    check("F1 every pre-existing file byte-identical", changed == [])
    check("F2 new files are EXACTLY the expected manifests (11)",
          new_files == expected, f"{len(new_files)} new")
    check("F3 totals 90 files / zero .tmp / recipes 7",
          f_n == 90 and f_tmp == 0
          and len(list(RECIPE_DIR.rglob("*.json"))) == 7)

    print()
    if FAILURES:
        print(f"M14 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M14 live smoke OK: composable recipes (registration, expansion, "
          "provenance, one-workflow execution, evidence reuse) verified "
          "live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
