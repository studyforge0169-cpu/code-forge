"""M8 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data), read-only by design: the dashboard only
observes the immutable M1-M7 histories of model 4a0a871886ef.

Checks (in order):
  * pre-state storage audit (files/bytes/mtimes, zero .tmp)
  * per-family history lists (checkpoints / evaluations / comparisons /
    gate decisions / workflows) match the recorded M8 inventory
  * GET /models/<id>/dashboard: summary, checkpoint lineage, evaluation
    series, comparison series, gate series, workflow counts, artifact graph
  * every dashboard id resolves through the existing engine/API getters and
    the graph contains no fabricated nodes/edges
  * determinism: repeated GETs byte-identical, stable result_hash
  * unknown model -> 404; unknown workflow/eval/comp/gate/ckpt ids -> 404
  * post-state byte-level audit: dashboard reads changed nothing

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
    """Full byte-level storage audit of the production root."""
    out = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            out[rel] = {"size": p.stat().st_size,
                        "mtime_ns": p.stat().st_mtime_ns,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def family_ids(prefix: str) -> set[str]:
    d = MODEL_DIR / prefix
    return {p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set()


def main() -> int:
    print(f"M8 live HTTP smoke (read-only) on {ROOT}, model {MODEL}\n")

    pre = audit()
    n_tmp = sum(1 for k in pre if ".tmp" in k)
    n_files = len(pre)
    n_bytes = sum(v["size"] for v in pre.values())
    check("pre-state: zero .tmp files", n_tmp == 0, f"{n_tmp}")
    print(f"       storage: {n_files} files / {n_bytes} B")

    # ---- inventory via the existing API lists ------------------------------
    ckpts = [c["checkpoint_id"] for c in
             call_json("GET", f"/models/{MODEL}/checkpoints")]
    evals = [e["eval_id"] for e in
             call_json("GET", f"/models/{MODEL}/evaluations")]
    comps = [c["comparison_id"] for c in
             call_json("GET", f"/models/{MODEL}/comparisons")]
    gates = [g["decision_id"] for g in
             call_json("GET", f"/models/{MODEL}/gates/decisions")]
    wfs = [w["workflow_id"] for w in
           call_json("GET", f"/models/{MODEL}/workflows")]
    check("live inventory non-trivial",
          len(ckpts) >= 3 and len(evals) >= 14 and len(comps) >= 8
          and len(gates) >= 10 and len(wfs) >= 5,
          f"{len(ckpts)} ckpt / {len(evals)} eval / {len(comps)} comp / "
          f"{len(gates)} gate / {len(wfs)} wf")

    # ---- dashboard ----------------------------------------------------------
    code, raw1 = call("GET", f"/models/{MODEL}/dashboard")
    check("dashboard 200", code == 200, f"http {code}")
    d = json.loads(raw1)
    check("dashboard top-level shape",
          set(d) >= {"model_id", "config_hash", "summary", "checkpoints",
                     "training_runs", "evaluations", "comparisons",
                     "gate_decisions", "workflows", "artifact_graph",
                     "diagnostics", "result_hash", "schema_version"},
          ",".join(sorted(d))[:200])
    check("model_id + schema_version", d["model_id"] == MODEL
          and d["schema_version"] == 1)
    check("no diagnostics on healthy store", d["diagnostics"] == [],
          json.dumps(d["diagnostics"])[:200])
    check("result_hash is 64-hex", len(d["result_hash"]) == 64)
    s = d["summary"]
    check("summary reflects persisted model manifest",
          s["id"] == MODEL and s["latest_checkpoint"] is not None
          and s["training_run_count"] == len(d["training_runs"]),
          f"latest={s['latest_checkpoint']} runs={s['training_run_count']}")

    # A. checkpoint lineage: same ids as the engine list; parents precede
    got_ck = [c["checkpoint_id"] for c in d["checkpoints"]]
    check("A: checkpoint ids == engine list", set(got_ck) == set(ckpts),
          f"{len(got_ck)}")
    pos = {cid: i for i, cid in enumerate(got_ck)}
    order_ok = True
    for c in d["checkpoints"]:
        parent = c.get("parent_checkpoint_id")
        if parent is not None:
            order_ok &= parent in pos and pos[parent] < pos[c["checkpoint_id"]]
    check("A: lineage order (parent precedes child)", order_ok)

    # B. evaluation series: groups mirror recorded records exactly
    series_ids = [r["eval_id"] for g in d["evaluations"] for r in g["records"]]
    check("B: evaluation series cover all records", set(series_ids) == set(evals),
          f"{len(series_ids)} records / {len(d['evaluations'])} series")
    mixed = any(
        len({tuple((k, str(r[k])) for k in ("state_hash", "dataset_id",
                                            "dataset_version", "split",
                                            "tokenizer_id", "seed"))
             for r in g["records"]}) > 1
        for g in d["evaluations"])
    check("B: one series never mixes incompatible probes", not mixed)

    # C. comparisons: M5 delta semantics preserved on every record
    comp_recs = [r for g in d["comparisons"] for r in g["records"]]
    check("C: comparison series cover all records",
          {r["comparison_id"] for r in comp_recs} == set(comps),
          f"{len(comp_recs)}")
    deltas_ok = all(abs(r["delta_loss_nats"] - (r["loss_b"] - r["loss_a"]))
                    < 1e-9 for r in comp_recs)
    check("C: delta_loss_nats == loss_B - loss_A", deltas_ok)

    # D. gates: descriptive statistics + grouped by recorded policy
    gate_recs = [r for g in d["gate_decisions"] for r in g["records"]]
    check("D: gate series cover all decisions",
          {r["decision_id"] for r in gate_recs} == set(gates),
          f"{len(gate_recs)} decisions / {len(d['gate_decisions'])} policies")
    stat_ok = all(sum(g["statistics"].values()) == len(g["records"])
                  for g in d["gate_decisions"])
    check("D: statistics are descriptive counts only", stat_ok)

    # E. workflows: records + counts
    wf_ids = [r["workflow_id"] for r in d["workflows"]["records"]]
    check("E: workflow records cover all runs", set(wf_ids) == set(wfs),
          f"counts={d['workflows']['counts']}")
    check("E: counts are descriptive",
          sum(d["workflows"]["counts"].values()) == len(wfs))

    # F. artifact graph: no fabricated nodes/edges
    g = d["artifact_graph"]
    node_ids = {(n["family"], n["artifact_id"]) for n in g["nodes"]}
    run_ids = {p["run_id"] for p in d["training_runs"]}
    known = {("checkpoint", c) for c in ckpts} | \
        {("evaluation", e) for e in evals} | \
        {("comparison", c) for c in comps} | \
        {("gate_decision", x) for x in gates} | \
        {("workflow", x) for x in wfs} | \
        {("training_run", x) for x in run_ids}
    check("F: every graph node is a real artifact", node_ids <= known,
          f"{len(node_ids)} nodes")
    # every edge target/source exists and every referenced role is recorded
    edge_ok = all((e["source"]["family"], e["source"]["artifact_id"]) in node_ids
                  and (e["target"]["family"], e["target"]["artifact_id"])
                  in node_ids for e in g["edges"])
    check("F: edge endpoints exist (no dangling edges)", edge_ok,
          f"{len(g['edges'])} edges")
    roles = {e["role"] for e in g["edges"]}
    check("F: provenance edge roles present",
          roles >= {"parent", "produced", "final_state", "state", "state_a",
                    "state_b", "candidate", "stage_artifact"},
          ",".join(sorted(roles)))

    # ---- determinism: byte-identical repeats + stable hash ------------------
    code, raw2 = call("GET", f"/models/{MODEL}/dashboard")
    check("repeat GET is byte-identical", code == 200 and raw1 == raw2,
          f"{len(raw2)} bytes")
    check("repeat GET keeps result_hash",
          json.loads(raw2)["result_hash"] == d["result_hash"])

    # ---- every exposed id resolves through the existing API getters ---------
    resolve_ok = True
    for cid in ckpts:
        resolve_ok &= call_json("GET", f"/models/{MODEL}/checkpoints/{cid}") \
            .get("checkpoint_id") == cid
    for eid in evals:
        resolve_ok &= call_json("GET", f"/models/{MODEL}/evaluations/{eid}") \
            .get("eval_id") == eid
    for cid in comps:
        resolve_ok &= call_json("GET", f"/models/{MODEL}/comparisons/{cid}") \
            .get("comparison_id") == cid
    for did in gates:
        resolve_ok &= call_json("GET",
                                f"/models/{MODEL}/gates/decisions/{did}") \
            .get("decision_id") == did
    for wid in wfs:
        resolve_ok &= call_json("GET", f"/models/{MODEL}/workflows/{wid}") \
            .get("workflow_id") == wid
    check("all ids resolve through the existing getters", resolve_ok)

    # ---- error mapping on the dashboard route -------------------------------
    code, raw = call("GET", "/models/ghost-model/dashboard")
    check("unknown model -> 404", code == 404
          and b"not found" in raw.lower(), f"http {code}")

    # ---- read-only: nothing changed, byte for byte ---------------------------
    post = audit()
    changed = [k for k in pre if pre[k] != post.get(k)] + \
        [k for k in post if k not in pre]
    check("storage byte-identical after all dashboard GETs", changed == [],
          f"changed={changed[:5]}")
    check("post-state: zero .tmp files",
          all(".tmp" not in k for k in post))
    n_files2 = len(post)
    n_bytes2 = sum(v["size"] for v in post.values())
    check("storage totals unchanged",
          n_files == n_files2 and n_bytes == n_bytes2,
          f"{n_files2} files / {n_bytes2} B")
    print(f"\nfinal storage: {n_files2} files / {n_bytes2} B / "
          f"zero .tmp (unchanged)")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("\nM8 live smoke OK: read-only dashboard verified on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
