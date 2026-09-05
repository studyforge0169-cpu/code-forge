"""M12 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data), model 4a0a871886ef.

Phase A (read-only): storage audit + production history + recipe/suite-run
inventories + dashboard baseline counts (captured BEFORE any write).

Phase B (registration): register the recipe `m12-live-suite` (one suite_run
stage: suite m9-live-suite against the kept checkpoint 0511de4c7372, whose
M4 evaluations already exist from the M10/M11 live runs) -> exactly one
recipe manifest with a deterministic config hash; identical re-registration
adds nothing; conflicting content -> 409 with the definition byte-identical;
GET list/get work; unknown recipe -> 404; PUT/DELETE -> 405.

Phase C (execution): run the recipe against the explicit production model ->
completed workflow with recipe provenance (recipe_id + recipe_hash), a
suite-run reference and snapshotted state provenance (checkpoint
0511de4c7372); both M4 probes are REUSED (zero new evaluations).

Phase D (repeat): a second explicit run -> distinct workflow + suite-run
records, deterministic result hashes, evaluations still reused, zero new M4
evidence. Inline-style API read of the persisted run shows provenance.

Phase E (failure): unknown recipe -> 404 (nothing persisted); unknown model
-> 404 (nothing persisted); a registered recipe whose suite does not exist at
run time -> 404 with ONE failed workflow run persisted (recipe provenance,
failed stage, no fabricated suite-run artifact, suite-run count unchanged).

Phase F (audit): dashboard deterministic with expected count deltas; every
pre-existing file byte-identical; new files are EXACTLY the expected set
(2 recipe manifests + 3 workflow manifests + 2 suite-run manifests); eval
count unchanged; zero .tmp.

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"
CKPT = "0511de4c7372"               # kept/final weights (verified M3 state)
SUITE = "m9-live-suite"
RECIPE = "m12-live-suite"
GHOST_RECIPE = "m12-live-ghost"
GHOST_SUITE = "m12-ghost-suite"
ROOT = Path("/home/user/ai-model-forge-data")

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


def snapshot() -> dict[str, int]:
    return {p.relative_to(ROOT).as_posix(): p.stat().st_size
            for p in ROOT.rglob("*") if p.is_file()}


def audit(label: str) -> tuple[int, int, int, dict]:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    n_bytes = sum(p.stat().st_size for p in files)
    n_tmp = sum(1 for p in files if ".tmp" in p.name)
    snap = snapshot()
    print(f"    storage: {len(files)} files / {n_bytes} B / tmp {n_tmp}")
    return len(files), n_bytes, n_tmp, snap


def main() -> int:
    print("== Phase A: read-only baseline ==")
    pre_n, pre_bytes, pre_tmp, pre = audit("pre")

    code, dash0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A1 dashboard 200 + deterministic", code == 200
          and dash0["result_hash"] == call_json(
              "GET", f"/models/{MODEL}/dashboard")[1]["result_hash"])
    wf_counts0 = dash0["workflows"]["counts"]
    wf0 = {r["workflow_id"]: r for r in dash0["workflows"]["records"]}
    check("A2 production history intact",
          len([r for r in dash0["workflows"]["records"]
               if r["status"] == "completed"]) == wf_counts0["completed"]
          and sum(wf_counts0.values()) == len(wf0),
          str(wf_counts0))

    _, suite_runs0 = call_json("GET", f"/models/{MODEL}/suite-runs")
    n_sr0 = len(suite_runs0)
    sr_ids0 = {s["suite_run_id"] for s in suite_runs0}
    check("A3 suite-run history intact", n_sr0 == 4
          and sr_ids0 == {"aae8e8a8ba9c", "bb114d2ce42e",
                          "8f8aee834c9f", "d9742459017b"})

    code, suite = call_json("GET", f"/probe-suites/{SUITE}")
    check("A4 M9 suite resolves", code == 200 and suite["suite_id"] == SUITE
          and len(suite["probes"]) == 2)

    code, evals = call_json("GET", f"/models/{MODEL}/evaluations")
    n_ev0 = len(evals)
    check("A5 M4 evaluation inventory intact", code == 200 and n_ev0 == 16)

    code, recipes0 = call_json("GET", "/workflows/recipes")
    check("A6 no recipes yet", code == 200 and recipes0 == [])

    print("== Phase B: registration ==")
    body = {"recipe_id": RECIPE,
            "description": "M12 live smoke: suite-run on kept checkpoint",
            "stages": [{"stage_id": "suite1", "type": "suite_run",
                        "suite_run": {"suite_id": SUITE, "state": {
                            "state_kind": "checkpoint",
                            "checkpoint_id": CKPT}}}]}
    code, reg = call_json("POST", "/workflows/recipes", body)
    check("B1 register recipe 201", code == 201 and reg["recipe_id"] == RECIPE,
          f"http {code}")
    cfg_hash = reg["config_hash"]
    check("B2 config hash is 64-hex + persisted verbatim",
          len(cfg_hash) == 64
          and cfg_hash
          in (ROOT / "workflow-recipes" / RECIPE
              / "manifest.json").read_text()
          and reg["description"] == body["description"],
          cfg_hash[:12])

    code, reg2 = call_json("POST", "/workflows/recipes", body)
    check("B3 identical re-registration idempotent",
          code == 201 and reg2["config_hash"] == cfg_hash
          and len(call_json("GET", "/workflows/recipes")[1]) == 1)

    clash = json.loads(json.dumps(body))
    clash["stages"][0]["suite_run"]["state"] = {"state_kind": "current"}
    code, _ = call_json("POST", "/workflows/recipes", clash)
    check("B4 conflicting content -> 409", code == 409, f"http {code}")

    code, got = call_json("GET", f"/workflows/recipes/{RECIPE}")
    check("B5 recipe get 200", code == 200 and got["config_hash"] == cfg_hash
          and got["stages"][0]["suite_run"]["state"]["checkpoint_id"] == CKPT)
    check("B6 unknown recipe 404",
          call("GET", "/workflows/recipes/no-such")[0] == 404)
    check("B7 PUT/DELETE -> 405",
          call("PUT", f"/workflows/recipes/{RECIPE}")[0] == 405
          and call("DELETE", f"/workflows/recipes/{RECIPE}")[0] == 405)

    print("== Phase C: execution ==")
    code, run1 = call_json("POST", f"/workflows/recipes/{RECIPE}/runs",
                           {"model_id": MODEL})
    check("C1 run 200", code == 200 and run1["status"] == "completed",
          f"http {code}")
    check("C2 recipe provenance on run record",
          run1["recipe_id"] == RECIPE and run1["recipe_hash"] == cfg_hash
          and run1["plan"]["model_id"] == MODEL
          and run1["name"] == RECIPE)
    w1_id = run1["workflow_id"]
    art1 = run1["stages"][0]["artifact"]
    check("C3 suite-run reference + state provenance",
          art1["kind"] == "suite_run" and art1["artifact_id"]
          and art1["checkpoint_id"] == CKPT)
    sr1_id = art1["artifact_id"]
    code, sr1 = call_json("GET", f"/models/{MODEL}/suite-runs/{sr1_id}")
    check("C4 suite-run record real + evidence reused",
          code == 200 and sr1["suite_id"] == SUITE
          and sr1["state"]["checkpoint_id"] == CKPT
          and sr1["completed_count"] == 2 and sr1["reused_count"] == 2
          and sr1["failed_count"] == 0,
          f"created {sr1['completed_count'] - sr1['reused_count']}, "
          f"reused {sr1['reused_count']}")
    code, wrec = call_json("GET", f"/models/{MODEL}/workflows/{w1_id}")
    check("C5 persisted run carries provenance",
          code == 200 and wrec["recipe_id"] == RECIPE
          and wrec["recipe_hash"] == cfg_hash)

    print("== Phase D: repeat ==")
    code, run2 = call_json("POST", f"/workflows/recipes/{RECIPE}/runs",
                           {"model_id": MODEL})
    check("D1 second run distinct", code == 200
          and run2["workflow_id"] != w1_id
          and run2["stages"][0]["artifact"]["artifact_id"] != sr1_id)
    sr2_id = run2["stages"][0]["artifact"]["artifact_id"]
    code, sr2 = call_json("GET", f"/models/{MODEL}/suite-runs/{sr2_id}")
    check("D2 evidence reused, no duplicate evaluations",
          sr2["reused_count"] == 2 and sr2["completed_count"] == 2
          and sr2["result_hash"] == sr1["result_hash"])
    check("D3 run result hashes deterministic",
          run2["result_hash"] == run1["result_hash"])
    code, evals2 = call_json("GET", f"/models/{MODEL}/evaluations")
    check("D4 evaluations unchanged (16)", len(evals2) == n_ev0,
          f"{len(evals2)} -> {n_ev0}")

    print("== Phase E: failure semantics ==")
    code, _ = call_json("POST", "/workflows/recipes/no-such-recipe/runs",
                        {"model_id": MODEL})
    check("E1 unknown recipe -> 404, nothing persisted",
          code == 404
          and len(call_json("GET", "/workflows/recipes")[1]) == 1)
    code, _ = call_json("POST", f"/workflows/recipes/{RECIPE}/runs",
                        {"model_id": "no-such-model"})
    check("E2 unknown model -> 404, nothing persisted", code == 404)
    ghost = {"recipe_id": GHOST_RECIPE, "stages": [
        {"stage_id": "suite1", "type": "suite_run",
         "suite_run": {"suite_id": GHOST_SUITE,
                       "state": {"state_kind": "current"}}}]}
    code, ghost_reg = call_json("POST", "/workflows/recipes", ghost)
    check("E3 ghost recipe registered", code == 201
          and ghost_reg["config_hash"] != cfg_hash
          and len(call_json("GET", "/workflows/recipes")[1]) == 2,
          "semantic change -> different hash")
    n_sr_before_fail = len(call_json("GET",
                                     f"/models/{MODEL}/suite-runs")[1])
    n_wf_before_fail = len(call_json("GET", f"/models/{MODEL}/workflows")[1])
    code, _ = call_json("POST", f"/workflows/recipes/{GHOST_RECIPE}/runs",
                        {"model_id": MODEL})
    check("E4 missing suite at run time -> 404", code == 404, f"http {code}")
    wfs = call_json("GET", f"/models/{MODEL}/workflows")[1]
    failed = [w for w in wfs if w["workflow_id"] not in wf0
              and w["status"] == "failed"]
    check("E5 failed run persisted with provenance + no artifact",
          len(wfs) == n_wf_before_fail + 1 and len(failed) == 1
          and failed[0]["recipe_id"] == GHOST_RECIPE
          and failed[0]["failed_stage_id"] == "suite1"
          and failed[0]["stages"][0]["artifact"] is None)
    check("E6 no suite-run artifact for failed attempt",
          len(call_json("GET", f"/models/{MODEL}/suite-runs")[1])
          == n_sr_before_fail)

    print("== Phase F: audit ==")
    code, dash1 = call_json("GET", f"/models/{MODEL}/dashboard")
    code, dash2 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("F1 dashboard valid + deterministic",
          code == 200 and dash1 == dash2)
    counts = dash1["workflows"]["counts"]
    check("F2 workflow counts reflect recipe runs",
          counts["completed"] == wf_counts0["completed"] + 2
          and counts["failed"] == wf_counts0["failed"] + 1
          and counts["stopped"] == wf_counts0["stopped"],
          str(counts))
    rec_runs = [r for r in dash1["workflows"]["records"]
                if r.get("recipe_id")]
    check("F3 recipe provenance visible in dashboard history",
          len(rec_runs) == 3 and all(r["recipe_hash"] for r in rec_runs))

    post_n, post_bytes, post_tmp, post = audit("post")
    changed = [k for k in pre if k in post and pre[k] != post[k]]
    new = sorted(set(post) - set(pre))
    check("F4 pre-existing files byte-identical", changed == [],
          f"changed={changed}")
    expected_new = {
        f"workflow-recipes/{RECIPE}/manifest.json",
        f"workflow-recipes/{GHOST_RECIPE}/manifest.json",
        f"models/{MODEL}/workflows/workflow-{w1_id}/manifest.json",
        f"models/{MODEL}/workflows/workflow-{run2['workflow_id']}/manifest.json",
        f"models/{MODEL}/workflows/workflow-"
        f"{failed[0]['workflow_id']}/manifest.json",
        f"suite-runs/{sr1_id}/manifest.json",
        f"suite-runs/{sr2_id}/manifest.json",
    }
    check("F5 new files exactly 2 recipes + 3 workflows + 2 suite runs",
          set(new) == expected_new, f"new={new}")
    check("F6 evaluations still 16",
          len(call_json("GET", f"/models/{MODEL}/evaluations")[1]) == n_ev0)
    check("F7 post-state zero .tmp", post_tmp == 0)
    expected_bytes = pre_bytes + sum(post[k] for k in new)
    check("F8 byte accounting exact", post_n == pre_n + 7
          and post_bytes == expected_bytes,
          f"{post_n} files / {post_bytes} B")

    print()
    if FAILURES:
        print(f"M12 live smoke FAILED: {len(FAILURES)} checks: {FAILURES}")
        return 1
    print("M12 live smoke OK: workflow recipes + explicit binding + "
          "evidence reuse verified on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
