"""M9 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data).

Phase A (read-only): storage audit + production history verified through the
existing endpoints + M8 dashboard unchanged.

Phase B (registry, minimal writes): register one policy + one probe suite,
retrieve/resolve both, verify idempotent identical registration, 409 on
changed content, 422 on malformed, 404 on unknown ids.

Phase C (one real gate path): a policy whose probe/baseline EXACTLY match an
existing production comparison (discovered at runtime) is evaluated through
the registry id — the M6 engine reuses the existing evaluations/comparison,
so exactly ONE new immutable decision manifest is appended (plus the two
definition manifests from phase B). All pre-existing files must remain
byte-identical.

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


def main() -> int:
    print("M9 live HTTP smoke on production FORGE_ROOT")
    print("=" * 78)

    # ---------------- Phase A: read-only baseline --------------------------
    pre = audit()
    check("A1 pre-state zero .tmp", all(".tmp" not in k for k in pre),
          f"{sum(1 for k in pre if '.tmp' in k)}")
    print(f"    storage: {len(pre)} files / "
          f"{sum(v['size'] for v in pre.values())} B")
    n_ck = len(call_json("GET", f"/models/{MODEL}/checkpoints"))
    n_ev = len(call_json("GET", f"/models/{MODEL}/evaluations"))
    n_co = len(call_json("GET", f"/models/{MODEL}/comparisons"))
    n_ga = len(call_json("GET", f"/models/{MODEL}/gates/decisions"))
    n_wf = len(call_json("GET", f"/models/{MODEL}/workflows"))
    check("A2 production history intact",
          n_ck >= 3 and n_ev >= 14 and n_co >= 8 and n_ga >= 10 and n_wf >= 5,
          f"{n_ck} ck/{n_ev} ev/{n_co} co/{n_ga} ga/{n_wf} wf")
    dash0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A3 dashboard valid (no diagnostics, stable hash)",
          dash0["diagnostics"] == [] and len(dash0["result_hash"]) == 64
          and dash0["result_hash"] ==
          call_json("GET", f"/models/{MODEL}/dashboard")["result_hash"])

    # registry starts empty on production
    check("A4 registry empty before M9 use",
          call_json("GET", "/policies") == []
          and call_json("GET", "/probe-suites") == [])

    # ---------------- Phase B: definitions ---------------------------------
    # policy: reuse the probe identity of a real production comparison with
    # two checkpoint states (discovered at runtime -> evidence reuse in C)
    comp_rows = [r for g in dash0["comparisons"] for r in g["records"]]
    comp = next((r for r in comp_rows
                 if r["state_a"]["state_kind"] == "checkpoint"
                 and r["state_b"]["state_kind"] == "checkpoint"
                 and r["state_a"]["checkpoint_id"]
                 and r["state_b"]["checkpoint_id"]), None)
    check("B0 reusable comparison found on production", comp is not None,
          f"{len(comp_rows)} comparison records scanned")
    if comp is None:
        print("cannot continue without a checkpoint-pair comparison")
        return 1
    ck_base = comp["state_a"]["checkpoint_id"]
    ck_cand = comp["state_b"]["checkpoint_id"]
    policy = {"name": "m9-live-guard", "model_id": MODEL,
              "dataset_id": comp["dataset_id"],
              "dataset_version": comp["dataset_version"],
              "split": comp["split"], "tokenizer_id": comp["tokenizer_id"],
              "max_eval_tokens": comp["max_eval_tokens"],
              "batch_size": comp["batch_size"], "max_seq_len": comp["max_seq_len"],
              "seed": comp["seed"], "baseline_type": "checkpoint",
              "baseline_checkpoint_id": ck_base, "tolerance": comp["tolerance"],
              "max_regression_delta": max(comp["tolerance"], 1e6)}
    code, raw = call("POST", "/policies", {"policy_id": "m9-live-policy",
                                           "description": "M9 live smoke policy",
                                           "policy": policy})
    check("B1 register policy 201", code == 201, f"http {code}")
    pol = json.loads(raw)
    check("B2 policy has deterministic 64-hex config hash",
          len(pol["config_hash"]) == 64)
    code, raw2 = call("POST", "/policies", {"policy_id": "m9-live-policy",
                                            "policy": policy})
    check("B3 identical re-register idempotent",
          code == 201 and json.loads(raw2)["created_at"] == pol["created_at"])
    changed = dict(policy, max_regression_delta=0.5)
    code, _ = call("POST", "/policies", {"policy_id": "m9-live-policy",
                                         "policy": changed})
    check("B4 changed content under same id -> 409", code == 409,
          f"http {code}")
    mal = dict(policy)
    del mal["baseline_checkpoint_id"]    # checkpoint baseline without id -> invalid
    code, _ = call("POST", "/policies", {"policy_id": "m9-live-mal",
                                         "policy": mal})
    check("B5 malformed policy -> 422", code == 422, f"http {code}")
    code, _ = call("GET", "/policies/m9-live-ghost")
    check("B6 unknown policy -> 404", code == 404, f"http {code}")
    got = call_json("GET", f"/policies/m9-live-policy")
    check("B7 policy get/list resolve identical semantics",
          got["config_hash"] == pol["config_hash"]
          and any(p["policy_id"] == "m9-live-policy"
                  for p in call_json("GET", "/policies")))
    # deterministic suite: two probes of the SAME production probe identity
    ev0 = call_json("GET", f"/models/{MODEL}/evaluations")[0]
    probe_tpl = {"dataset_id": ev0["dataset_id"], "split": "validation",
                 "tokenizer_id": ev0["tokenizer_id"],
                 "batch_size": ev0["config"]["batch_size"],
                 "max_seq_len": ev0["config"]["max_seq_len"]}
    suite = [dict(probe_tpl, seed=ev0["seed"]),
             dict(probe_tpl, seed=ev0["seed"] + 1)]
    code, raw = call("POST", "/probe-suites",
                     {"suite_id": "m9-live-suite", "probes": suite})
    check("B8 register suite 201", code == 201, f"http {code}")
    su = json.loads(raw)
    code, raw2 = call("POST", "/probe-suites",
                      {"suite_id": "m9-live-suite", "probes": list(reversed(suite))})
    check("B9 identical suite (reversed input) idempotent + canonical",
          code == 201 and json.loads(raw2)["probes_hash"] == su["probes_hash"]
          and json.loads(raw2)["created_at"] == su["created_at"])
    code, _ = call("POST", "/probe-suites",
                   {"suite_id": "m9-live-suite",
                    "probes": [dict(probe_tpl, seed=ev0["seed"] + 9)]})
    check("B10 changed suite content -> 409", code == 409, f"http {code}")
    code, _ = call("GET", "/probe-suites/m9-live-ghost")
    check("B11 unknown suite -> 404", code == 404, f"http {code}")
    check("B12 suite get/list resolve",
          call_json("GET", "/probe-suites/m9-live-suite")["probes_hash"]
          == su["probes_hash"]
          and any(s["suite_id"] == "m9-live-suite"
                  for s in call_json("GET", "/probe-suites")))
    check("B13 no PUT/PATCH/DELETE on definitions",
          call("PUT", "/policies/m9-live-policy")[0] == 405
          and call("DELETE", "/policies/m9-live-policy")[0] == 405
          and call("PATCH", "/probe-suites/m9-live-suite")[0] == 405)

    # ---------------- Phase C: one real gate path via registry id ----------
    code, raw = call("POST", "/gates/evaluate",
                     {"model_id": MODEL, "policy_id": "m9-live-policy",
                      "candidate": {"state_kind": "checkpoint",
                                    "checkpoint_id": ck_cand}})
    check("C1 gate via policy_id 200", code == 200, f"http {code}")
    dec = json.loads(raw)
    check("C2 decision provenance recorded",
          dec["policy_id"] == "m9-live-policy"
          and dec["policy_config_hash"] == pol["config_hash"]
          and dec["decision"] in ("passed", "failed"))
    check("C3 evidence reused (comparison == discovered record)",
          dec["comparison_id"] == comp["comparison_id"],
          f"comparison {dec['comparison_id']}")
    # historical + new decisions visible on the dashboard with provenance
    dash1 = call_json("GET", f"/models/{MODEL}/dashboard")
    rows = [r for g in dash1["gate_decisions"] for r in g["records"]
            if r["decision_id"] == dec["decision_id"]]
    check("C4 dashboard renders registry decision with provenance",
          len(rows) == 1 and rows[0]["policy_id"] == "m9-live-policy"
          and rows[0]["policy_config_hash"] == pol["config_hash"])
    check("C5 dashboard still deterministic",
          dash1["result_hash"] ==
          call_json("GET", f"/models/{MODEL}/dashboard")["result_hash"]
          and dash1["result_hash"] != dash0["result_hash"])  # new evidence
    check("C6 workflows unaffected",
          call_json("GET", f"/models/{MODEL}/workflows") is not None)

    # ---------------- Phase D: byte-level audit ----------------------------
    post = audit()
    pre_keys = set(pre)
    post_keys = set(post)
    changed = [k for k in pre_keys if pre[k] != post.get(k)]
    new = sorted(post_keys - pre_keys)
    check("D1 all pre-existing files byte-identical", changed == [],
          f"changed={changed[:5]}")
    expected_new = {
        "policies/m9-live-policy/manifest.json",
        "probe-suites/m9-live-suite/manifest.json",
        f"models/{MODEL}/gates/gate-{dec['decision_id']}/manifest.json",
    }
    check("D2 new files exactly the definitions + one decision",
          set(new) == expected_new,
          f"new={new}")
    check("D3 post-state zero .tmp", all(".tmp" not in k for k in post))
    print(f"    final storage: {len(post)} files / "
          f"{sum(v['size'] for v in post.values())} B "
          f"(+{len(new)} definition/decision manifests, "
          f"all pre-existing files byte-identical)")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("\nM9 live smoke OK: registry + one real registry-gate path verified "
          "on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
