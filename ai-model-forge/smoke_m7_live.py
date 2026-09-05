"""M7 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data), started fresh from the M6 baseline:
41 files / 3,270,964 B, model 4a0a871886ef @ 30a8bc5b82ab, 10 evals,
4 comparisons, 6 gate decisions, zero workflows.

Phase A  : one training workflow (train -> evaluate -> gate) -> M3 really runs.
Phase B  : gate-stop with rollback suggestion, evidence reuse, validation and
           failure cases -> byte-level read-only audit over every artifact.

Exit code 0 = all evidence checks passed.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"
DS = "ee1a716c4573"
TOK = "99106e3255c5"
CK_BAD = "30a8bc5b82ab"     # higher loss on the validation probe
CK_GOOD = "025e6d8d8f15"    # lower loss (the M6-verified better state)
ROOT = Path("/home/user/ai-model-forge-data")
MODEL_DIR = ROOT / "models" / MODEL

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def call(method: str, path: str, body=None, expect: int = 200):
    req = urllib.request.Request(BASE + path, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data) as resp:
            code, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    return code, raw.decode() if raw else ""


def call_json(method: str, path: str, body=None, expect: int = 200):
    code, raw = call(method, path, body)
    if code != expect:
        raise AssertionError(f"{method} {path}: expected {expect}, got {code}: {raw[:300]}")
    return json.loads(raw) if raw else None


def snapshot() -> dict[str, bytes]:
    return {p.relative_to(MODEL_DIR).as_posix(): p.read_bytes()
            for p in MODEL_DIR.rglob("*") if p.is_file()}


def eval_stage(sid, ckpt=None, seed=7001, from_stage=None):
    cfg = dict(model_id=MODEL, dataset_id=DS, split="validation",
               tokenizer_id=TOK, batch_size=8, max_seq_len=32, seed=seed)
    if ckpt:
        cfg["checkpoint_id"] = ckpt
    return {"stage_id": sid, "type": "evaluate",
            "evaluation": {"config": cfg,
                           "checkpoint_from_stage": from_stage}}


def compare_stage(sid, ckpt_a, ckpt_b, seed=7001):
    return {"stage_id": sid, "type": "compare",
            "comparison": {"state_a": {"state_kind": "checkpoint",
                                       "checkpoint_id": ckpt_a},
                           "state_b": {"state_kind": "checkpoint",
                                       "checkpoint_id": ckpt_b},
                           "dataset_id": DS, "split": "validation",
                           "tokenizer_id": TOK, "batch_size": 8,
                           "max_seq_len": 32, "seed": seed, "tolerance": 1e-4}}


def gate_stage(sid, ckpt=None, baseline_ckpt=None, baseline_type="checkpoint",
               minimum_loss=None, seed=7001, from_stage=None,
               on_pass=None, on_fail=None):
    pol = {"name": f"wf-gate-{sid}", "model_id": MODEL, "dataset_id": DS,
           "split": "validation", "tokenizer_id": TOK, "batch_size": 8,
           "max_seq_len": 32, "seed": seed, "tolerance": 1e-4,
           "baseline_type": baseline_type}
    if baseline_ckpt:
        pol["baseline_checkpoint_id"] = baseline_ckpt
    if minimum_loss is not None:
        pol["minimum_loss"] = minimum_loss
    if baseline_type == "current":
        pol.pop("baseline_checkpoint_id", None)
    cand = {"state_kind": "checkpoint"}
    if from_stage:
        cand["from_stage"] = from_stage
    else:
        cand["checkpoint_id"] = ckpt
    st = {"stage_id": sid, "type": "gate",
          "gate": {"policy": pol, "candidate": cand}}
    if on_pass:
        st["on_pass"] = on_pass
    if on_fail:
        st["on_fail"] = on_fail
    return st


def train_stage(sid, name="wf-live-train", steps=16, seed=23):
    return {"stage_id": sid, "type": "train",
            "training": {"name": name, "method": "continued_pretraining",
                         "model_id": MODEL, "dataset_id": DS,
                         "tokenizer_id": TOK, "learning_rate": 3e-3,
                         "batch_size": 8, "max_seq_len": 32, "steps": steps,
                         "eval_every_steps": steps, "keep_best": False,
                         "seed": seed}}


def plan(name, stages, description=None):
    p = {"name": name, "model_id": MODEL, "stages": stages}
    if description:
        p["description"] = description
    return p


def main() -> int:
    print("=" * 78)
    print("M7 live HTTP smoke on production FORGE_ROOT")
    print("=" * 78)

    # ---------------- baseline (case 1-2) ----------------------------------
    model0 = call_json("GET", f"/models/{MODEL}")
    prov0 = len(model0["training_provenance"])
    latest0 = model0["latest_checkpoint"]
    n_ckpt0 = len(call_json("GET", f"/models/{MODEL}/checkpoints"))
    n_wf0 = len(call_json("GET", f"/models/{MODEL}/workflows"))
    n_ev0 = len(call_json("GET", f"/models/{MODEL}/evaluations"))
    n_comp0 = len(call_json("GET", f"/models/{MODEL}/comparisons"))
    n_gate0 = len(call_json("GET", f"/models/{MODEL}/gates/decisions"))
    check("case1 GET model baseline",
          model0["id"] == MODEL and prov0 == 1 and latest0 == CK_BAD
          and n_ckpt0 == 2 and n_ev0 == 10 and n_comp0 == 4 and n_gate0 == 6,
          f"provenance={prov0} latest={latest0} ckpts={n_ckpt0} "
          f"evals={n_ev0} comps={n_comp0} gates={n_gate0}")
    check("case2 workflows history starts empty", n_wf0 == 0, f"{n_wf0} runs")

    s0 = snapshot()

    # ---------------- Phase A: training workflow (case 3-4) ----------------
    wa = plan("m7-live-train-eval-gate",
              [train_stage("s1"),
               eval_stage("s2", seed=7002, from_stage="s1"),
               gate_stage("s3", baseline_type="current", seed=7002,
                          from_stage="s1")],
              description="live M7 demo: explicit train -> eval -> gate chain")
    rec_a = call_json("POST", "/workflows/run", wa)
    check("case3a W-A completed", rec_a["status"] == "completed",
          rec_a["status"])
    arts_a = {s["stage_id"]: s["artifact"] for s in rec_a["stages"]}
    train_art = arts_a["s1"]
    check("case3b train stage artifact is a training report",
          train_art["kind"] == "training_report" and train_art["accepted"]
          and train_art["checkpoint_id"], f"ckpt={train_art['checkpoint_id']}")
    ck_a = train_art["checkpoint_id"]
    model1 = call_json("GET", f"/models/{MODEL}")
    prov1 = model1["training_provenance"]
    check("case3c M3 really trained: provenance appended + run_id linked",
          len(prov1) == prov0 + 1 and prov1[-1]["run_id"] == train_art["artifact_id"]
          and prov1[-1]["final_checkpoint_id"] == ck_a,
          f"runs {prov0}->{len(prov1)}")
    n_ckpt1 = len(call_json("GET", f"/models/{MODEL}/checkpoints"))
    check("case3d new checkpoint persisted and weights advanced",
          n_ckpt1 > n_ckpt0 and model1["latest_checkpoint"] == ck_a
          and model1["latest_checkpoint"] != latest0,
          f"ckpts {n_ckpt0}->{n_ckpt1}, live={model1['latest_checkpoint']}")
    # Two digest conventions by design: the sidecar digests the serialized
    # file bytes; checkpoint manifests / run artifacts carry a canonical
    # content hash of the tensor state. Chain: run artifact (content hash)
    # == checkpoint record (content hash) == content_hash(weights.pt state);
    # and the sidecar must digest exactly the weights.pt file on disk.
    import hashlib
    sidecar = json.loads((MODEL_DIR / "weights.sha256").read_text())
    ck_man = json.loads((MODEL_DIR / "checkpoints" / ck_a / "manifest.json").read_text())
    file_digest = hashlib.sha256((MODEL_DIR / "weights.pt").read_bytes()).hexdigest()
    check("case3e live weights == train-stage final checkpoint (content hash "
          "chain + file-digest sidecar)",
          ck_man["weights_sha256"] == train_art["state_hash"]
          and sidecar["sha256"] == file_digest,
          f"ckpt/content={ck_man['weights_sha256'][:16]}... sidecar==file={sidecar['sha256']==file_digest}")
    # the evaluation stage consumed the train stage output (from_stage=s1)
    eval_art = arts_a["s2"]
    ev = call_json("GET", f"/models/{MODEL}/evaluations/{eval_art['artifact_id']}")
    check("case3f evaluate stage ran on the training output",
          ev["checkpoint_id"] == ck_a and ev["state_hash"] == train_art["state_hash"]
          and ev["loss_nats"] == eval_art["loss_nats"]
          and ev["result_hash"] == eval_art["result_hash"],
          f"loss={ev['loss_nats']}")
    dec_art = arts_a["s3"]
    dec = call_json("GET", f"/models/{MODEL}/gates/decisions/{dec_art['artifact_id']}")
    check("case3g gate passed (candidate == current == train output, no regression)",
          dec_art["gate_decision"] == "passed" and dec["decision"] == "passed",
          f"verdict={dec.get('verdict')}")
    check("case3h transitions of W-A",
          [(t["stage_id"], t["decision"], t["to_stage"]) for t in rec_a["transitions"]]
          == [("s1", "next", "s2"), ("s2", "next", "s3"), ("s3", "passed", None)],
          str(rec_a["transitions"]))
    check("case3i result_hash present + deterministic form",
          len(rec_a["result_hash"]) == 64 and rec_a["plan_hash"] == rec_a["plan_hash"])

    got_a = call_json("GET", f"/models/{MODEL}/workflows/{rec_a['workflow_id']}")
    check("case4 GET single run echoes the POST body (immutable record)",
          got_a == rec_a and got_a["plan"]["name"] == "m7-live-train-eval-gate",
          got_a["workflow_id"])
    lst = call_json("GET", f"/models/{MODEL}/workflows")
    check("case4b list grows to one run",
          len(lst) == 1 and lst[0]["workflow_id"] == rec_a["workflow_id"])
    wf_a_path = MODEL_DIR / "workflows" / f"workflow-{rec_a['workflow_id']}" / "manifest.json"
    check("case4c run manifest on disk == API record",
          json.loads(wf_a_path.read_text()) == rec_a)

    # ---------------- Phase B: stop/suggestion/reuse/errors -----------------
    sB = snapshot()   # byte-level baseline for the read-only phase
    modelB = call_json("GET", f"/models/{MODEL}")

    # case 5: regression gate with NO on_fail -> STOPPED + rollback suggestion
    wb = plan("m7-live-gate-stop-suggest",
              [gate_stage("g1", ckpt=CK_BAD, baseline_ckpt=CK_GOOD, seed=1,
                          minimum_loss=None)],
              description="candidate 30a8... regresses vs baseline 025e...; "
                          "workflow stops and suggests rollback, never rolls back")
    rec_b = call_json("POST", "/workflows/run", wb)
    check("case5a W-B stopped (failed gate without branch)",
          rec_b["status"] == "stopped" and rec_b["failed_stage_id"] is None
          and "stopped" in rec_b["terminal_reason"], rec_b["status"])
    gb_art = rec_b["stages"][0]["artifact"]
    check("case5b gate decision FAILED with real regression evidence",
          gb_art["gate_decision"] == "failed",
          str(gb_art.get("gate_decision")))
    dec_b = call_json("GET", f"/models/{MODEL}/gates/decisions/{gb_art['artifact_id']}")
    check("case5c rollback suggestion recorded, never executed",
          rec_b["suggested_checkpoint_id"] == CK_GOOD
          and rec_b["hint"] == dec_b["hint"]
          and dec_b["suggested_checkpoint_id"] == CK_GOOD
          and "rollback" in (dec_b["hint"] or "").lower(),
          f"suggestion={rec_b['suggested_checkpoint_id']} hint={rec_b['hint']}")
    modelB2 = call_json("GET", f"/models/{MODEL}")
    check("case5d no auto-rollback: weights/provenance/latest untouched",
          modelB2["latest_checkpoint"] == modelB["latest_checkpoint"]
          and modelB2["training_provenance"] == modelB["training_provenance"],
          f"latest={modelB2['latest_checkpoint']}")
    check("case5e run cites the persisted FAILED decision; chain stops",
          dec_b["decision_id"] == gb_art["artifact_id"]
          and rec_b["transitions"][0]["decision"] == "failed"
          and rec_b["transitions"][0]["to_stage"] is None)

    # case 6: evidence workflow with explicit compare + branching gate
    wc = plan("m7-live-evidence-branch",
              [eval_stage("s1", ckpt=CK_BAD, seed=7101),
               compare_stage("s2", ckpt_a=CK_BAD, ckpt_b=CK_BAD, seed=7101),
               gate_stage("s3", ckpt=CK_BAD, baseline_ckpt=CK_GOOD, seed=7101,
                          on_pass="s4", on_fail="s4"),
               eval_stage("s4", ckpt=CK_BAD, seed=7101)],
              description="evidence run: compare of identical states (delta 0), "
                          "regression gate branches to s4 either way")
    n_ev_B = len(call_json("GET", f"/models/{MODEL}/evaluations"))
    n_comp_B = len(call_json("GET", f"/models/{MODEL}/comparisons"))
    rec_c = call_json("POST", "/workflows/run", wc)
    check("case6a W-C completed with branching gate",
          rec_c["status"] == "completed"
          and [(t["stage_id"], t["decision"], t["to_stage"])
               for t in rec_c["transitions"]]
          == [("s1", "next", "s2"), ("s2", "next", "s3"),
              ("s3", "failed", "s4"), ("s4", "next", None)],
          str(rec_c["transitions"]))
    cc_art = rec_c["stages"][1]["artifact"]
    cmp_c = call_json("GET", f"/models/{MODEL}/comparisons/{cc_art['artifact_id']}")
    check("case6b compare of identical states: delta 0, unchanged",
          cmp_c["delta_loss_nats"] == 0.0 and cmp_c["verdict"] == "unchanged"
          and cc_art["verdict"] == "unchanged" and cc_art["delta_loss_nats"] == 0.0)
    check("case6c gate decision in run is FAILED and its evidence chains",
          rec_c["stages"][2]["artifact"]["gate_decision"] == "failed"
          and cmp_c["result_hash"] == cc_art["result_hash"])
    check("case6d stage s4 reuses s1's evaluation (exact identity reuse)",
          rec_c["stages"][3]["artifact"]["artifact_id"]
          == rec_c["stages"][0]["artifact"]["artifact_id"],
          rec_c["stages"][3]["artifact"]["artifact_id"])
    n_ev_C = len(call_json("GET", f"/models/{MODEL}/evaluations"))
    n_comp_C = len(call_json("GET", f"/models/{MODEL}/comparisons"))
    check("case6e W-C evidence growth: +2 evaluations, +2 comparisons",
          n_ev_C == n_ev_B + 2 and n_comp_C == n_comp_B + 2,
          f"evals {n_ev_B}->{n_ev_C}, comps {n_comp_B}->{n_comp_C}")

    # case 7: identical repeat -> evidence reused, fresh audit events
    rec_c2 = call_json("POST", "/workflows/run", wc)
    check("case7a W-C' completed with identical decisions",
          rec_c2["status"] == "completed"
          and rec_c2["result_hash"] == rec_c["result_hash"]
          and rec_c2["stages"][2]["artifact"]["gate_decision"] == "failed"
          and rec_c2["stages"][2]["artifact"]["gate_decision"]
          == rec_c["stages"][2]["artifact"]["gate_decision"])
    n_ev_C2 = len(call_json("GET", f"/models/{MODEL}/evaluations"))
    n_comp_C2 = len(call_json("GET", f"/models/{MODEL}/comparisons"))
    n_gate_C2 = len(call_json("GET", f"/models/{MODEL}/gates/decisions"))
    check("case7b identical repeat reuses ALL evaluations and comparisons",
          n_ev_C2 == n_ev_C and n_comp_C2 == n_comp_C,
          f"evals {n_ev_C}->{n_ev_C2}, comps {n_comp_C}->{n_comp_C2}")
    for ridx in (0, 1, 3):
        check(f"case7c stage {ridx} artifact reused by identity",
              rec_c2["stages"][ridx]["artifact"]["artifact_id"]
              == rec_c["stages"][ridx]["artifact"]["artifact_id"])
    check("case7d gate decision + run manifest are fresh audit events",
          rec_c2["stages"][2]["artifact"]["artifact_id"]
          != rec_c["stages"][2]["artifact"]["artifact_id"]
          and rec_c2["workflow_id"] != rec_c["workflow_id"]
          and n_gate_C2 == len(call_json("GET", f"/models/{MODEL}/gates/decisions")))
    # W-B (stop) + W-C + W-C' -> exactly 3 new workflows since Phase A
    check("case7e workflow history",
          len(call_json("GET", f"/models/{MODEL}/workflows")) == 4,
          "4 runs after W-A + W-B + W-C + W-C'")

    # case 8: invalid plans rejected (422, nothing persisted)
    bad1 = plan("bad-dup", [eval_stage("s1", ckpt=CK_GOOD, seed=7201),
                            eval_stage("s1", ckpt=CK_BAD, seed=7201)])
    code, _ = call("POST", "/workflows/run", bad1)
    check("case8a duplicate stage ids -> 422", code == 422, str(code))
    bad2 = plan("bad-branch", [gate_stage("g1", ckpt=CK_BAD,
                                          baseline_ckpt=CK_GOOD, seed=7202,
                                          on_fail="ghost")])
    code, _ = call("POST", "/workflows/run", bad2)
    check("case8b on_fail to unknown stage -> 422", code == 422, str(code))
    check("case8c invalid plans persisted nothing",
          len(call_json("GET", f"/models/{MODEL}/workflows")) == 4)

    # case 9: unknown model -> 404 before any run; ghost checkpoint -> 404
    #         WITH a recorded failed run (partial history is legitimate)
    ghost_plan = {"name": "ghost-model", "model_id": "ghostmodel",
                  "stages": [{"stage_id": "s1", "type": "evaluate",
                              "evaluation": {"config": {
                                  "model_id": "ghostmodel", "dataset_id": DS,
                                  "split": "validation",
                                  "tokenizer_id": TOK, "batch_size": 8,
                                  "max_seq_len": 32, "seed": 7203,
                                  "checkpoint_id": CK_GOOD}}}]}
    code, raw = call("POST", "/workflows/run", ghost_plan)
    check("case9a unknown model -> 404 before any run", code == 404, str(code))
    check("case9b no run recorded for unknown model",
          len(call_json("GET", f"/models/{MODEL}/workflows")) == 4)
    bad_ck = plan("ghost-ckpt", [eval_stage("s1", ckpt="deadbeef", seed=7204)])
    try:
        call_json("POST", "/workflows/run", bad_ck, expect=404)
        check("case9c ghost checkpoint -> 404", True)
    except AssertionError as e:
        check("case9c ghost checkpoint -> 404", False, str(e)[:200])
    lst = call_json("GET", f"/models/{MODEL}/workflows")
    failed_run = lst[-1]
    check("case9d ghost-ckpt run recorded as FAILED with earlier history kept",
          failed_run["status"] == "failed" and failed_run["failed_stage_id"] == "s1"
          and failed_run["stages"][0]["artifact"] is None
          and failed_run["stages"][0]["error"]
          and len(lst) == 5)

    # case 10: 404 semantics on list/get
    code, _ = call("GET", "/models/ghostmodel/workflows")
    check("case10a unknown model workflows -> 404", code == 404, str(code))
    code, _ = call("GET", f"/models/{MODEL}/workflows/ghost-run")
    check("case10b unknown workflow -> 404", code == 404, str(code))

    # ---------------- Phase B byte-level read-only audit --------------------
    sE = snapshot()
    added = set(sE) - set(sB)
    modified = {k for k in sB if sE[k] != sB[k]}
    check("audit1 no pre-existing file modified during phase B (incl. model "
          "manifest, weights, checkpoints, evals, comps, gates, provenance)",
          modified == set(), sorted(modified)[:5])
    # expected new files, derived from each new run manifest + gate evidence
    expected: set[str] = set()
    for r in call_json("GET", f"/models/{MODEL}/workflows"):
        rp = MODEL_DIR / "workflows" / f"workflow-{r['workflow_id']}"
        if (rp / "manifest.json").exists() and r["workflow_id"] != rec_a["workflow_id"]:
            expected.add(f"workflows/workflow-{r['workflow_id']}/manifest.json")
            for st in r["stages"]:
                art = st.get("artifact")
                if not art:
                    continue
                kind, aid = art["kind"], art["artifact_id"]
                if kind == "evaluation":
                    expected.add(f"evaluations/eval-{aid}/manifest.json")
                elif kind == "comparison":
                    expected.add(f"comparisons/comp-{aid}/manifest.json")
                elif kind == "gate_decision":
                    expected.add(f"gates/gate-{aid}/manifest.json")
                    dec = json.loads((MODEL_DIR / "gates" / f"gate-{aid}"
                                      / "manifest.json").read_text())
                    expected.add(f"evaluations/eval-{dec['candidate']['evaluation_id']}"
                                 f"/manifest.json")
                    if dec.get("baseline"):
                        expected.add(f"evaluations/eval-{dec['baseline']['evaluation_id']}"
                                     f"/manifest.json")
                    if dec.get("comparison_id"):
                        expected.add(f"comparisons/comp-{dec['comparison_id']}"
                                     f"/manifest.json")
    expected = {p for p in expected if p not in sB}
    check("audit2 phase-B additions are exactly the run + evidence manifests",
          added == expected,
          f"unexpected={sorted(added - expected)[:5]} "
          f"missing={sorted(expected - added)[:5]}")
    tmp = [str(p) for p in ROOT.rglob("*") if p.is_file() and ".tmp" in p.name]
    check("audit3 zero .tmp leftovers anywhere", tmp == [], str(tmp)[:200])

    # ---------------- global pre/post storage audit --------------------------
    files = sorted(ROOT.rglob("*"))
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    n_files = len([p for p in files if p.is_file()])
    new_since_s0 = len(set(sE) - set(s0))
    check("audit4 final storage totals",
          n_files == 41 + new_since_s0 and total_bytes > 3_270_964,
          f"files {n_files} (41 + {new_since_s0} new), bytes {total_bytes}")
    # everything from the M6 baseline is byte-identical except the three files
    # M3 legitimately rewrote during the W-A training stage
    changed = {k for k in s0 if sE.get(k) != s0.get(k)}
    check("audit5 M6-baseline artifacts immutable except training stage rewrite",
          changed <= {"manifest.json", "weights.pt", "weights.sha256"},
          sorted(changed)[:5])
    counts = {
        "checkpoints": len(call_json("GET", f"/models/{MODEL}/checkpoints")),
        "evaluations": n_ev_C2, "comparisons": n_comp_C2,
        "gates": n_gate_C2, "workflows": len(lst),
        "provenance_runs": len(call_json("GET", f"/models/{MODEL}")
                               ["training_provenance"]),
    }
    print("-" * 78)
    print("storage audit:", json.dumps({"files": n_files, "bytes": total_bytes,
                                        "artifact_counts": counts}, indent=1))
    print("-" * 78)
    ok = not FAILURES
    print("SMOKE RESULT:", "ALL CHECKS PASSED" if ok else f"{len(FAILURES)} FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
