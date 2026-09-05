"""M11 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data), model 4a0a871886ef.

Phase A (read-only): storage audit + production history + M9 suite + M10
suite-run inventory.

Phase B (suite-run workflow stages, minimal writes): two equivalent
workflows, each with one `suite_run` stage over the kept checkpoint
0511de4c7372 and the registered suite m9-live-suite (whose M10 live run
already created the exact M4 evaluations) -> every probe is reused; each
invocation adds exactly one workflow manifest + one suite-run manifest and
ZERO evaluations. The workflow records reference the suite-run artifacts with
explicit state provenance.

Phase C (failure): a workflow whose suite stage names an unknown suite -> the
M7 failure path persists one failed workflow run (no fabricated suite-run
artifact, HTTP 404).

Phase D: dashboard determinism + byte-level audit (pre-existing files must be
byte-identical; new files are exactly the expected manifests; zero .tmp).

Exit code 0 = all evidence checks passed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"
CKPT = "0511de4c7372"               # kept/final weights (verified M3 state)
SUITE = "m9-live-suite"
ROOT = Path("/home/user/ai-model-forge-data")
MODEL_DIR = ROOT / "models" / MODEL
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


def call_json(method: str, path: str, body=None, expect: int = 200):
    code, raw = call(method, path, body)
    if code != expect:
        raise AssertionError(f"{method} {path}: expected {expect}, "
                             f"got {code}: {raw[:300]}")
    return json.loads(raw) if raw else None


def audit() -> dict:
    out = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            out[rel] = {"size": p.stat().st_size,
                        "mtime_ns": p.stat().st_mtime_ns,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def dir_ids(path: Path) -> set[str]:
    return {p.name for p in path.iterdir() if p.is_dir()} if path.exists() \
        else set()


def suite_stage(sid: str, suite_id: str, ckpt: str) -> dict:
    return {"stage_id": sid, "type": "suite_run",
            "suite_run": {"suite_id": suite_id,
                          "state": {"state_kind": "checkpoint",
                                    "checkpoint_id": ckpt}}}


def main() -> int:
    print("M11 live HTTP smoke on production FORGE_ROOT")
    print("=" * 78)

    # ---------------- Phase A: read-only baseline --------------------------
    pre = audit()
    check("A1 pre-state zero .tmp", all(".tmp" not in k for k in pre))
    n_files, n_bytes = len(pre), sum(v["size"] for v in pre.values())
    print(f"    storage: {n_files} files / {n_bytes} B")
    wf_counts0 = call_json("GET", f"/models/{MODEL}/dashboard") \
        ["workflows"]["counts"]
    n_wf0 = len(call_json("GET", f"/models/{MODEL}/workflows"))
    n_sr0 = len(call_json("GET", f"/models/{MODEL}/suite-runs"))
    n_ev0 = len(call_json("GET", f"/models/{MODEL}/evaluations"))
    check("A2 production history intact",
          len(call_json("GET", f"/models/{MODEL}/checkpoints")) >= 3
          and n_ev0 >= 16
          and len(call_json("GET", f"/models/{MODEL}/comparisons")) >= 8
          and len(call_json("GET", f"/models/{MODEL}/gates/decisions")) >= 11
          and n_wf0 >= 5 and n_sr0 >= 2,
          f"wf={n_wf0} suite-runs={n_sr0} evals={n_ev0}")
    suite = call_json("GET", f"/probe-suites/{SUITE}")
    check("A3 M9 suite resolves", suite["suite_id"] == SUITE
          and len(suite["probes"]) == 2)

    # ---------------- Phase B: suite-run workflow stages -------------------
    wf_body = {"name": "m11-live-suite-1", "model_id": MODEL,
               "stages": [suite_stage("suite1", SUITE, CKPT)]}
    code, raw = call("POST", "/workflows/run", wf_body)
    check("B1 workflow with suite stage 200", code == 200, f"http {code}")
    w1 = json.loads(raw)
    check("B2 workflow completed", w1["status"] == "completed", w1["status"])
    art = w1["stages"][0]["artifact"]
    check("B3 stage artifact references a suite run",
          art["kind"] == "suite_run" and art["artifact_id"]
          and art["checkpoint_id"] == CKPT and art["result_hash"])
    sr1 = call_json("GET", f"/models/{MODEL}/suite-runs/{art['artifact_id']}")
    check("B4 suite-run record has explicit state provenance",
          sr1["state"]["checkpoint_id"] == CKPT
          and sr1["state"]["state_kind"] == "checkpoint"
          and sr1["suite_probes_hash"] == suite["probes_hash"])
    check("B5 M10 evidence reused (M4 evals exist from the M10 live run)",
          [x["outcome"] for x in sr1["results"]] == ["reused", "reused"])
    n_sr1 = len(call_json("GET", f"/models/{MODEL}/suite-runs"))
    n_wf1 = len(call_json("GET", f"/models/{MODEL}/workflows"))
    check("B6 exactly one new workflow + one new suite run",
          n_wf1 == n_wf0 + 1 and n_sr1 == n_sr0 + 1, f"wf+1 sr+1")
    check("B7 no new evaluations created",
          len(call_json("GET", f"/models/{MODEL}/evaluations")) == n_ev0)

    # second equivalent workflow -> reuse again, new records only
    wf_body2 = dict(wf_body, name="m11-live-suite-2")
    code, raw = call("POST", "/workflows/run", wf_body2)
    w2 = json.loads(raw)
    sr2 = call_json("GET", f"/models/{MODEL}/suite-runs/"
                           f"{w2['stages'][0]['artifact']['artifact_id']}")
    check("B8 second workflow reuses every evaluation",
          code == 200 and w2["status"] == "completed"
          and [x["outcome"] for x in sr2["results"]] == ["reused", "reused"])
    check("B9 suite-run result_hash deterministic across invocations",
          sr2["result_hash"] == sr1["result_hash"])
    check("B10 distinct immutable invocations",
          w2["workflow_id"] != w1["workflow_id"]
          and sr2["suite_run_id"] != sr1["suite_run_id"])
    check("B11 no duplicate M4 evidence",
          len(call_json("GET", f"/models/{MODEL}/evaluations")) == n_ev0)

    # ---------------- Phase C: failure semantics ---------------------------
    n_sr_before_fail = len(call_json("GET", f"/models/{MODEL}/suite-runs"))
    bad_body = {"name": "m11-live-fail", "model_id": MODEL,
                "stages": [suite_stage("suite1", "ghost-suite", CKPT)]}
    code, raw = call("POST", "/workflows/run", bad_body)
    check("C1 unknown suite -> 404 (after failed run persisted)", code == 404,
          f"http {code}")
    runs = call_json("GET", f"/models/{MODEL}/workflows")
    failed = [w for w in runs if w["name"] == "m11-live-fail"]
    check("C2 failed run recorded with stage id and no fabricated artifact",
          len(failed) == 1 and failed[0]["status"] == "failed"
          and failed[0]["failed_stage_id"] == "suite1"
          and failed[0]["stages"][0]["artifact"] is None
          and failed[0]["stages"][0]["error"]
          and "ghost-suite" in failed[0]["stages"][0]["error"])
    n_sr_after_fail = len(call_json("GET", f"/models/{MODEL}/suite-runs"))
    check("C3 no suite-run artifact for the failed attempt",
          n_sr_after_fail == n_sr_before_fail)

    # ---------------- Phase D: dashboard + byte audit -----------------------
    dash1 = call_json("GET", f"/models/{MODEL}/dashboard")
    dash2 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("D1 dashboard valid + deterministic",
          dash1["diagnostics"] == []
          and dash1["result_hash"] == dash2["result_hash"])
    wf_rows = [w for w in dash1["workflows"]["records"]
               if w["workflow_id"] in (w1["workflow_id"], w2["workflow_id"])]
    check("D2 dashboard renders suite-run workflow stages",
          len(wf_rows) == 2
          and all(s["artifact"]["kind"] == "suite_run"
                  for w in wf_rows for s in w["stages"]))
    expected_counts = {k: wf_counts0.get(k, 0) for k in wf_counts0}
    expected_counts["completed"] = wf_counts0.get("completed", 0) + 2
    expected_counts["failed"] = wf_counts0.get("failed", 0) + 1
    check("D3 workflow history counts reflect new runs",
          dash1["workflows"]["counts"] == expected_counts,
          f"{dash1['workflows']['counts']}")
    post = audit()
    changed = [k for k in pre if pre[k] != post.get(k)]
    check("D4 all pre-existing files byte-identical", changed == [],
          f"changed={changed[:5]}")
    new_files = set(post) - set(pre)
    expected = {f"models/{MODEL}/workflows/workflow-{w1['workflow_id']}/manifest.json",
                f"models/{MODEL}/workflows/workflow-{w2['workflow_id']}/manifest.json",
                f"models/{MODEL}/workflows/workflow-{failed[0]['workflow_id']}/manifest.json",
                f"suite-runs/{sr1['suite_run_id']}/manifest.json",
                f"suite-runs/{sr2['suite_run_id']}/manifest.json"}
    check("D5 new files exactly 3 workflow + 2 suite-run manifests",
          new_files == expected, f"new={sorted(new_files)}")
    check("D6 post-state zero .tmp", all(".tmp" not in k for k in post))
    n2 = len(post)
    b2 = sum(v["size"] for v in post.values())
    print(f"    final storage: {n2} files / {b2} B "
          f"(workflows {n_wf0} -> {n_wf0 + 3}, suite runs {n_sr0} -> "
          f"{n_sr0 + 2}, evaluations {n_ev0} -> {n_ev0}, zero .tmp)")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("\nM11 live smoke OK: suite-run workflow stages + evidence reuse "
          "verified on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
