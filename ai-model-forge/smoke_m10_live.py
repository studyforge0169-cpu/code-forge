"""M10 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data), model 4a0a871886ef.

Phase A (read-only): storage audit + production history + the existing M9
suite (m9-live-suite) verified through the API.

Phase B (suite runs, minimal writes): execute the registered suite against
the kept checkpoint 0511de4c7372 (an explicit immutable state). Probes with
no exact M4 evidence are evaluated once; everything is then reused by the
second identical run — exactly k new evaluation manifests + 2 new suite-run
manifests, nothing else.

Phase C (errors/immutability): unknown suite 404, malformed request 422,
unknown state 404, no update/delete endpoints (405), dashboard valid and
deterministic, byte-level audit of every pre-existing file.

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


def eval_dir_ids() -> set[str]:
    d = MODEL_DIR / "evaluations"
    return {p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set()


def suite_run_dir_ids() -> set[str]:
    d = ROOT / "suite-runs"
    return {p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set()


def main() -> int:
    print("M10 live HTTP smoke on production FORGE_ROOT")
    print("=" * 78)

    # ---------------- Phase A: read-only baseline --------------------------
    pre = audit()
    check("A1 pre-state zero .tmp", all(".tmp" not in k for k in pre),
          f"{sum(1 for k in pre if '.tmp' in k)}")
    n_files, n_bytes = len(pre), sum(v["size"] for v in pre.values())
    print(f"    storage: {n_files} files / {n_bytes} B")
    n_ev0 = len(call_json("GET", f"/models/{MODEL}/evaluations"))
    check("A2 production history intact",
          len(call_json("GET", f"/models/{MODEL}/checkpoints")) >= 3
          and n_ev0 >= 14
          and len(call_json("GET", f"/models/{MODEL}/comparisons")) >= 8
          and len(call_json("GET", f"/models/{MODEL}/gates/decisions")) >= 11
          and len(call_json("GET", f"/models/{MODEL}/workflows")) >= 5,
          f"evals={n_ev0}")
    suite = call_json("GET", f"/probe-suites/{SUITE}")
    check("A3 existing M9 suite resolves", suite["suite_id"] == SUITE
          and len(suite["probes"]) == 2 and len(suite["probes_hash"]) == 64,
          f"{len(suite['probes'])} probes, hash {suite['probes_hash'][:12]}…")
    dash0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A4 dashboard valid pre-run",
          dash0["diagnostics"] == [] and len(dash0["result_hash"]) == 64)
    check("A5 no suite runs yet", suite_run_dir_ids() == set())

    # ---------------- Phase B: suite runs ----------------------------------
    body = {"model_id": MODEL, "suite_id": SUITE,
            "state": {"state_kind": "checkpoint", "checkpoint_id": CKPT}}
    code, raw = call("POST", "/suite-runs", body)
    check("B1 suite-run 200", code == 200, f"http {code}")
    r1 = json.loads(raw)
    check("B2 run completed with per-probe results",
          r1["status"] == "completed" and r1["probe_count"] == 2
          and r1["completed_count"] == 2
          and r1["reused_count"] + r1["failed_count"] <= 2
          and len(r1["result_hash"]) == 64)
    ids1 = [x["evaluation_id"] for x in r1["results"]]
    created1 = [x["evaluation_id"] for x in r1["results"]
                if x["outcome"] == "created"]
    check("B3 every probe maps to a real evaluation id", all(ids1))
    # returned ids resolve through the existing M4 endpoint
    resolve_ok = all(call_json("GET", f"/models/{MODEL}/evaluations/{eid}")
                     .get("eval_id") == eid for eid in ids1)
    check("B4 evaluation ids resolve", resolve_ok)
    ev_dirs_after1 = eval_dir_ids()
    check("B5 new evaluation manifests == probes that needed them",
          len(ev_dirs_after1) == n_ev0 + len(created1),
          f"+{len(created1)} of {len(ids1)} probes created")
    # state identity recorded on the run
    check("B6 run references the explicit checkpoint state",
          r1["state"]["checkpoint_id"] == CKPT
          and r1["state"]["state_kind"] == "checkpoint"
          and len(r1["state_hash"]) == 64
          and r1["suite_probes_hash"] == suite["probes_hash"])

    # second identical run -> full reuse, no duplicate evidence
    code, raw = call("POST", "/suite-runs", body)
    r2 = json.loads(raw)
    check("B7 second run reuses every evaluation",
          [x["outcome"] for x in r2["results"]] == ["reused", "reused"]
          and [x["evaluation_id"] for x in r2["results"]] == ids1)
    check("B8 deterministic result_hash across runs",
          r2["result_hash"] == r1["result_hash"])
    check("B9 no duplicate evaluation artifacts",
          eval_dir_ids() == ev_dirs_after1)
    check("B10 exactly two new suite-run manifests",
          suite_run_dir_ids() == {r1["suite_run_id"], r2["suite_run_id"]})

    # list + detail
    listing = call_json("GET", f"/models/{MODEL}/suite-runs")
    check("B11 deterministic list order",
          [x["suite_run_id"] for x in listing] == [r1["suite_run_id"],
                                                   r2["suite_run_id"]])
    got = call_json("GET", f"/models/{MODEL}/suite-runs/{r1['suite_run_id']}")
    check("B12 detail returns the persisted record",
          got["result_hash"] == r1["result_hash"]
          and got["created_at"] == r1["created_at"])
    code, _ = call("GET", f"/models/{MODEL}/suite-runs/ghost-run")
    check("B13 unknown run -> 404", code == 404, f"http {code}")
    code, _ = call("GET", f"/models/b5bc905326b6/suite-runs/{r1['suite_run_id']}")
    check("B14 run of another model -> 404", code == 404, f"http {code}")

    # ---------------- Phase C: errors, immutability, audit -----------------
    code, _ = call("POST", "/suite-runs",
                   {"model_id": MODEL, "suite_id": "ghost-suite",
                    "state": {"state_kind": "checkpoint",
                              "checkpoint_id": CKPT}})
    check("C1 unknown suite -> 404", code == 404, f"http {code}")
    code, _ = call("POST", "/suite-runs",
                   {"model_id": MODEL, "suite_id": SUITE,
                    "state": {"state_kind": "checkpoint"}})
    check("C2 malformed state -> 422", code == 422, f"http {code}")
    code, _ = call("POST", "/suite-runs",
                   {"model_id": MODEL, "suite_id": SUITE,
                    "state": {"state_kind": "checkpoint",
                              "checkpoint_id": "ghost-ck"}})
    check("C3 unknown state -> 404", code == 404, f"http {code}")
    check("C4 no update/delete endpoints",
          call("PUT", f"/models/{MODEL}/suite-runs/{r1['suite_run_id']}")[0]
          == 405
          and call("DELETE", f"/models/{MODEL}/suite-runs/{r1['suite_run_id']}")[0]
          == 405
          and call("PATCH", "/suite-runs")[0] == 405)
    # dashboard remains valid + deterministic (now includes the new evals)
    dash1 = call_json("GET", f"/models/{MODEL}/dashboard")
    dash2 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("C5 dashboard valid and deterministic",
          dash1["diagnostics"] == []
          and dash1["result_hash"] == dash2["result_hash"])
    check("C6 dashboard renders the suite-run evaluations",
          {x["evaluation_id"] for x in r1["results"]}
          <= {r["eval_id"] for g in dash1["evaluations"] for r in g["records"]})
    # byte-level audit: only the expected new artifacts, all else identical
    post = audit()
    changed = [k for k in pre if pre[k] != post.get(k)]
    expected_new = {f"suite-runs/{r1['suite_run_id']}/manifest.json",
                    f"suite-runs/{r2['suite_run_id']}/manifest.json"}
    ev_dirs_after2 = eval_dir_ids()
    for eid in created1:
        expected_new.add(f"models/{MODEL}/evaluations/eval-{eid}/manifest.json")
    check("C7 all pre-existing files byte-identical", changed == [],
          f"changed={changed[:5]}")
    new_files = sorted(set(post) - set(pre))
    check("C8 new files exactly 2 run manifests + needed evaluations",
          set(new_files) == expected_new, f"new={new_files}")
    check("C9 post-state zero .tmp", all(".tmp" not in k for k in post))
    n2 = len(post)
    b2 = sum(v["size"] for v in post.values())
    print(f"    final storage: {n2} files / {b2} B "
          f"(evals {n_ev0} -> {len(ev_dirs_after2)}, "
          f"suite runs 0 -> {len(suite_run_dir_ids())}, zero .tmp)")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("\nM10 live smoke OK: suite runs + exact-evidence reuse verified "
          "on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
