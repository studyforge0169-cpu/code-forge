"""M13 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M13 is strictly READ-ONLY observability —
this smoke never writes a single byte to the production root.

Phase A (baseline audit): storage audit (79 files / 3,976,967 B / 0 .tmp),
family inventories, production dashboard baseline (suite-run section counts
and records for 4a0a871886ef; empty section for b5bc905326b6).

Phase B (dashboard suite-run section): model 4a0a871886ef shows exactly its
6 persisted suite runs (aae8e8a8ba9c / bb114d2ce42e / 8f8aee834c9f /
d9742459017b / 2af508035191 / 5684649d0ced, all completed, 2 per-probe
references each with real M4 evaluation ids, outcomes recorded verbatim);
counts {completed: 6}; ordering (created_at, suite_run_id); model
b5bc905326b6 shows the deterministic empty section; no score/quality tokens
anywhere in the section.

Phase C (artifact graph): suite_run nodes for all 6 runs; workflow ->
suite_run edges (role stage_artifact) exactly for the workflow stage
artifacts that reference them; suite_run -> evaluation edges (role probe)
for every per-probe reference; every edge endpoint is a real node;
diagnostics == [] on both models.

Phase D (recipe lineage): GET /workflows/recipes/m12-live-suite/runs ->
exactly 2 completed runs (54e78451453b, 900624426305) annotated with
model_id/recipe_id/recipe_hash/status/created_at, ordered
(created_at, workflow_id), records verbatim to the per-model dashboard
records; m12-live-ghost -> exactly 1 failed run (6ea007c7ac98); unknown
recipe -> 404; repeated GETs byte-identical and write-free.

Phase E (final audit): repeated dashboard + lineage reads are byte-
deterministic, result_hash is stable, and the storage audit is identical
to Phase A (same files, same bytes, zero .tmp).

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"          # production model with the full history
EMPTY_MODEL = "b5bc905326b6"    # production model with no artifacts at all
SUITE_IDS = ["aae8e8a8ba9c", "bb114d2ce42e", "8f8aee834c9f",
             "d9742459017b", "5684649d0ced", "2af508035191"]
RECIPE_SUITE = "m12-live-suite"  # 2 completed recipe-bound runs
RECIPE_GHOST = "m12-live-ghost"  # 1 failed recipe-bound run
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


def audit(label: str) -> tuple[int, int, int, dict]:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    n_bytes = sum(p.stat().st_size for p in files)
    n_tmp = sum(1 for p in files if ".tmp" in p.name)
    snap = {p.relative_to(ROOT).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(f"    storage: {len(files)} files / {n_bytes} B / tmp {n_tmp} "
          f"({label})")
    return len(files), n_bytes, n_tmp, snap


def main() -> int:
    print("== Phase A: read-only baseline audit ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")

    code, dash0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A1 dashboard 200", code == 200, f"http {code}")
    h0 = dash0["result_hash"]
    check("A2 result_hash 64-hex + repeat-stable",
          len(h0) == 64 and dash0 == call_json(
              "GET", f"/models/{MODEL}/dashboard")[1])
    sec0 = dash0["suite_runs"]
    check("A3 suite-run section present with 6 completed records",
          sec0["counts"] == {"completed": 6}
          and [r["suite_run_id"] for r in sec0["records"]] == SUITE_IDS
          and all(r["status"] == "completed" and r["model_id"] == MODEL
                  for r in sec0["records"]))
    check("A4 workflow history counts intact (7 completed/3 failed/1 stopped)",
          dash0["workflows"]["counts"] ==
          {"completed": 7, "failed": 3, "stopped": 1})
    check("A5 healthy store: zero diagnostics", dash0["diagnostics"] == [])
    code, empty_dash = call_json("GET", f"/models/{EMPTY_MODEL}/dashboard")
    check("A6 second model: 200 + deterministic empty suite-run section",
          code == 200
          and empty_dash["suite_runs"] == {"counts": {}, "records": []}
          and empty_dash == call_json(
              "GET", f"/models/{EMPTY_MODEL}/dashboard")[1])
    _, evals = call_json("GET", f"/models/{MODEL}/evaluations")
    eval_ids = {e["eval_id"] for e in evals}
    check("A7 M4 evaluation inventory intact", len(eval_ids) == 16)
    _, workflows = call_json("GET", f"/models/{MODEL}/workflows")
    check("A8 workflow inventory intact", len(workflows) == 11)

    print("== Phase B: dashboard suite-run section detail ==")
    dash1 = call_json("GET", f"/models/{MODEL}/dashboard")[1]
    sec = dash1["suite_runs"]
    by_id = {r["suite_run_id"]: r for r in sec["records"]}
    check("B1 records verbatim to the M10 engine view",
          set(by_id) == set(SUITE_IDS))
    _, sr_list = call_json("GET", f"/models/{MODEL}/suite-runs")
    engine_by_id = {r["suite_run_id"]: r for r in sr_list}
    check("B2 every dashboard record equals the suite-run endpoint record",
          all(by_id[sid] == engine_by_id[sid] for sid in SUITE_IDS))
    order = [r["suite_run_id"] for r in sec["records"]]
    check("B3 ordering is (created_at, suite_run_id)",
          order == [r["suite_run_id"] for r in sorted(
              sr_list, key=lambda r: (r["created_at"], r["suite_run_id"]))])
    created = by_id["aae8e8a8ba9c"]
    check("B4 first run created its M4 evaluations (outcome verbatim)",
          [p["outcome"] for p in created["results"]] ==
          ["created", "created"])
    check("B5 later runs reused exact evidence (outcome verbatim)",
          all(set(p["outcome"] for p in by_id[sid]["results"]) == {"reused"}
              for sid in SUITE_IDS[1:]))
    check("B6 every per-probe evaluation_id is a real M4 evaluation",
          all(p["evaluation_id"] in eval_ids
              for sid in SUITE_IDS for p in by_id[sid]["results"]))
    code, suite_def = call_json("GET", "/probe-suites/m9-live-suite")
    check("B7 suite definition resolves with 2 probes",
          code == 200 and len(suite_def["probes"]) == 2)
    def_seeds = {p["seed"] for p in suite_def["probes"]}
    check("B8 per-probe detail preserved as recorded",
          all(len(by_id[sid]["results"]) == 2
              and all(p["error"] is None
                      and p["evaluation_id"]
                      and p["probe"]["seed"] in def_seeds
                      and p["probe"]["dataset_id"]
                      and p["probe"]["split"] == "validation"
                      for p in by_id[sid]["results"])
              for sid in SUITE_IDS))
    blob = json.dumps(sec)
    check("B9 no invented score/quality/benchmark anywhere in the section",
          not any(w in blob for w in
                  ("score", "quality", "rank", "benchmark", "leaderboard",
                   "pass_rate", "accuracy")))
    check("B10 empty model section deterministic again + no suite nodes",
          call_json("GET", f"/models/{EMPTY_MODEL}/dashboard")[1] ==
          empty_dash
          and all(n["family"] != "suite_run"
                  for n in empty_dash["artifact_graph"]["nodes"]))

    print("== Phase C: artifact graph ==")
    g = dash1["artifact_graph"]
    nodes = {(n["family"], n["artifact_id"]) for n in g["nodes"]}
    check("C1 suite_run nodes for all 6 production runs",
          all(("suite_run", sid) in nodes for sid in SUITE_IDS))
    edges = {(e["source"]["family"], e["source"]["artifact_id"],
              e["target"]["family"], e["target"]["artifact_id"], e["role"])
             for e in g["edges"]}
    wf_sr = [(w["workflow_id"], w["stages"][0]["artifact"]["artifact_id"])
             for w in workflows
             if w["stages"][0]["artifact"]
             and w["stages"][0]["artifact"]["kind"] == "suite_run"]
    check("C2 workflow -> suite_run edges exactly for recorded stage "
          "artifacts", len(wf_sr) == 4
          and all(("workflow", wid, "suite_run", rid, "stage_artifact")
                  in edges for wid, rid in wf_sr))
    probe_edges = [(sid, p["evaluation_id"])
                   for sid in SUITE_IDS for p in by_id[sid]["results"]]
    check("C3 suite_run -> evaluation edges for every per-probe reference",
          len(probe_edges) == 12
          and all(("suite_run", sid, "evaluation", ev, "probe") in edges
                  for sid, ev in probe_edges))
    check("C4 every edge endpoint is a real node",
          all((e[0], e[1]) in nodes and (e[2], e[3]) in nodes for e in edges))
    check("C5 graph healthy: zero diagnostics on both models",
          dash1["diagnostics"] == []
          and empty_dash["diagnostics"] == [])

    print("== Phase D: recipe-run lineage ==")
    code, lin_suite = call_json(
        "GET", f"/workflows/recipes/{RECIPE_SUITE}/runs")
    check("D1 m12-live-suite lineage: exactly 2 runs, 200",
          code == 200 and len(lin_suite) == 2, f"http {code} n={len(lin_suite)}")
    code, lin_ghost = call_json(
        "GET", f"/workflows/recipes/{RECIPE_GHOST}/runs")
    check("D2 m12-live-ghost lineage: exactly 1 failed run",
          code == 200 and len(lin_ghost) == 1
          and lin_ghost[0]["status"] == "failed")
    code, _ = call_json("GET", "/workflows/recipes/no-such-recipe/runs")
    check("D3 unknown recipe -> 404", code == 404, f"http {code}")
    code, rec_def = call_json(
        "GET", f"/workflows/recipes/{RECIPE_SUITE}")
    cfg_hash = rec_def["config_hash"]
    check("D4 lineage records annotated with provenance",
          all(r["recipe_id"] == RECIPE_SUITE
              and r["recipe_hash"] == cfg_hash
              and r["model_id"] == MODEL
              and r["status"] == "completed"
              and r["created_at"] and r["workflow_id"]
              for r in lin_suite))
    wf_by_id = {w["workflow_id"]: w for w in workflows}
    check("D5 lineage records verbatim to the per-model workflow records",
          all(lin_suite[i] == wf_by_id[lin_suite[i]["workflow_id"]]
              for i in range(len(lin_suite))))
    check("D6 ordering is (created_at, workflow_id)",
          [r["workflow_id"] for r in lin_suite] ==
          [r["workflow_id"] for r in sorted(
              lin_suite, key=lambda r: (r["created_at"],
                                        r["workflow_id"]))])
    check("D7 recipe-bound workflows visible with provenance in history",
          all(w["recipe_id"] in (RECIPE_SUITE, RECIPE_GHOST)
              and w["recipe_hash"] for w in workflows
              if w["recipe_id"] is not None)
          and sum(1 for w in workflows if w["recipe_id"]) == 3)
    check("D8 inline workflow runs keep null provenance",
          all(w["recipe_id"] is None and w["recipe_hash"] is None
              for w in workflows if w["recipe_id"] is None))
    check("D9 empty-model workflow history + dashboard consistent",
          empty_dash["workflows"] == {"counts": {}, "records": []})
    check("D10 no recipe score/quality anywhere in lineage responses",
          not any(w in (json.dumps(lin_suite) + json.dumps(lin_ghost))
                  for w in ("score", "quality", "rank", "benchmark",
                            "leaderboard")))

    print("== Phase E: repeated reads + final audit ==")
    raw_a = call("GET", f"/models/{MODEL}/dashboard")[1]
    raw_b = call("GET", f"/models/{MODEL}/dashboard")[1]
    check("E1 dashboard byte-identical across repeated GETs", raw_a == raw_b)
    raw_l1 = call("GET", f"/workflows/recipes/{RECIPE_SUITE}/runs")[1]
    raw_l2 = call("GET", f"/workflows/recipes/{RECIPE_SUITE}/runs")[1]
    check("E2 lineage byte-identical across repeated GETs", raw_l1 == raw_l2)
    raw_e1 = call("GET", f"/models/{EMPTY_MODEL}/dashboard")[1]
    raw_e2 = call("GET", f"/models/{EMPTY_MODEL}/dashboard")[1]
    check("E3 empty-model dashboard byte-identical too", raw_e1 == raw_e2)
    h_final = call_json("GET", f"/models/{MODEL}/dashboard")[1]["result_hash"]
    check("E4 dashboard result_hash stable across the whole smoke",
          h_final == h0)
    post_n, post_bytes, post_tmp, post_snap = audit("post")
    check("E5 storage audit identical to Phase A",
          post_n == pre_n and post_bytes == pre_bytes
          and post_tmp == pre_tmp == 0
          and post_snap == pre_snap,
          f"{pre_n}/{pre_bytes}/tmp{pre_tmp} -> "
          f"{post_n}/{post_bytes}/tmp{post_tmp}")
    unchanged = [p for p in pre_snap if pre_snap[p] != post_snap.get(p)]
    check("E6 zero new/changed files and zero .tmp",
          unchanged == [] and post_tmp == 0, str(unchanged[:5]))

    print()
    if FAILURES:
        print(f"M13 live smoke FAILED: {len(FAILURES)} failing check(s)")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M13 live smoke OK: read-only suite-run dashboard + graph + "
          "recipe lineage verified on production (zero writes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
