"""M57 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M57 adds DECLARATIVE BEST-BASELINE gate
policies: an INLINE GatePolicy may declare baseline_from_best=true; the
SAME M53 resolver (WorkflowEngine.resolve_best_state_refs) resolves the
SAME M52 selection (minimum persisted validation_loss), ONCE per plan at
execution/M51-preflight time, and pins the concrete id
(resolved_baseline_checkpoint_id); the gate engine then receives a PURE
M6 policy with an explicit baseline_checkpoint_id (the gate layer never
queries "best"; M6 decision semantics unchanged; evidence reuse intact).

The smoke registers exactly ONE new minimal immutable recipe —
m57-live-best-gate: the CANONICAL finite improvement loop, now fully
declarative end-to-end:
    train -> evaluate(best) -> gate(candidate=from_stage, baseline=best)
         -> train(resume_from_best) -> evaluate(best)
(gate on_pass -> tr2; tolerance 1.0 nats so the gate passes
deterministically; no automatic repetition) — and executes it EXACTLY
ONCE (state-mutating; no repetition). All other production recipes are
untouched.

Production facts (re-derived live; authoritative sources are the
persisted manifests): model 31db39e17a20 owns 11 checkpoints; the
live-computed M52 argmin is 0deba72e350b @ 6.369091 while the published
latest is 86687445867e (ALSO @ 6.369091 — a persisted-loss TIE broken by
the canonical (step, created_at) ASCENDING tie-break: best != latest as
concrete ids, the exact manual-drift M57 removes for gate baselines).
Storage at start: 39 files / 7,644,998 B / 0 .tmp (m57_pre.sha256
captured BEFORE this smoke; the production root was honestly REBUILT
before the M56 smoke after an environment loss — see the M56 final
report §7).

LIVE A (baseline): exact 39/7,644,998/0 audit + per-file SHA256
inventory + all checkpoint manifests read directly + live-computed argmin
(+ M52 endpoint agreement) + latest_before + pre-state listings (1
workflow / 1 evaluation / 0 suite-runs / 0 comparisons / 0 gate decisions
/ 0 samples / 1 recipe) + dashboard pre-state.

LIVE B (register the canonical declarative recipe): POST /workflows/
recipes m57-live-best-gate -> 201; exactly ONE new file (its manifest);
the stored recipe is DECLARATIVE (the gate policy carries
baseline_from_best=true with nothing pinned and no explicit id; both
evaluate stages carry checkpoint_from_best=true; the second train stage
carries resume_from_best=true); registry 1 -> 2; all 39 baseline files
byte-identical.

LIVE C (preflight -> execution, ONE run): the M51 preflight pins the
SAME live-computed argmin onto ALL FOUR best declarations (ONE selection
per plan; x2 byte-identical, ZERO writes from preflights); then POST
.../runs ONCE -> 200 completed: the record's plan is byte-identical to
the preflight plan; the gate judged tr1's output (candidate=from_stage)
against the PLAN-START best and PASSED; the persisted decision embeds
the PURE M6 policy (explicit baseline_checkpoint_id == the selection,
baseline_from_best stripped) and its baseline side identifies the
CONCRETE checkpoint; the gate's baseline evaluation REUSED the loop's
evaluate(best) record (decision.baseline.evaluation_id == ev1's
artifact); ev2 REUSED ev1's exact evaluation (evaluations 1 -> 3, not
4); the M55 train stage received a PURE M54 explicit resume
(provenance initial/parent/config) and its checkpoints DESCEND from the
selection; the recipe manifest is byte-identical after execution; the
selected checkpoint's manifest + weights are byte-identical (referenced,
never copied); no rollback (latest is one of the loop's NEW
checkpoints); no best-pointer file.

LIVE D (regression + final audit): M52 /checkpoints/best resolves the
live argmin over the grown registry; the SAME recipe's preflight
re-resolves the CURRENT argmin onto all four declarations (read-only —
the next execution of the loop picks up the improved state; no
cross-time lock); the dashboard changed ONLY in the checkpoint/
training-run/evaluation/comparison/gate-decision/workflow/artifact-
graph/summary sections; suite-runs/samples unchanged; OpenAPI 84
UNCHANGED with both fields in GatePolicy; final audit 53 files / 0 tmp:
14 new (1 recipe manifest + 1 workflow manifest + 8 checkpoint files
(2 train stages x 2 checkpoints) + 2 evaluation manifests + 1 comparison
manifest + 1 gate-decision manifest) + at most the 3
existing-semantics modifications (model manifest, weights.pt,
weights.sha256) — every file justified.

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8779 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8779/api/v1"
MODEL = "31db39e17a20"
RECIPE = "m57-live-best-gate"         # the ONE new canonical recipe (M57)
DATASET = "58e10a1d3c9b"
TOKENIZER = "63dd8dcd3215"
ROOT = Path("/home/user/ai-model-forge-data")
CKPT_DIR = ROOT / "models" / MODEL / "checkpoints"
PLAN_PATH = "/models/{m}/workflows/recipes/{r}/plan"
RUNS_PATH = "/workflows/recipes/{r}/runs"

PRE_FILES, PRE_BYTES, PRE_TMP = 39, 7_644_998, 0
POST_FILES = 53                        # +1 recipe +1 workflow +8 ckpt
                                       # files +2 evaluations +1 comparison
                                       # +1 gate decision

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def call(method: str, path: str, body=None):
    req = urllib.request.Request(path, method=method)
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


def new_files(pre: dict, post: dict) -> list[str]:
    return sorted(set(post) - set(pre))


def changed_files(pre: dict, post: dict) -> list[str]:
    return sorted(k for k in set(pre) & set(post) if pre[k] != post[k])


def live_argmin() -> tuple[str, float, int]:
    manifests = {}
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            manifests[d.name] = json.loads(
                (d / "manifest.json").read_text())
    ordered = sorted(manifests.values(),
                     key=lambda m: (m["step"], m["created_at"]))
    best = min(ordered, key=lambda m: m["validation_loss"])
    return best["checkpoint_id"], best["validation_loss"], len(manifests)


def deep_diff(pre, post, prefix="") -> list[str]:
    out: list[str] = []
    if type(pre) is not type(post):
        return [prefix or "<root>"]
    if isinstance(pre, dict):
        for k in sorted(set(pre) | set(post)):
            p = f"{prefix}.{k}" if prefix else k
            if k not in pre:
                out.append(p + " (added)")
            elif k not in post:
                out.append(p + " (removed)")
            else:
                out.extend(deep_diff(pre[k], post[k], p))
    elif isinstance(pre, list):
        if len(pre) != len(post):
            out.append(f"{prefix} (len {len(pre)} -> {len(post)})")
        else:
            for i, (a, b) in enumerate(zip(pre, post)):
                out.extend(deep_diff(a, b, f"{prefix}[{i}]"))
    elif pre != post:
        out.append(f"{prefix} ({pre!r} -> {post!r})")
    return out


def main() -> int:
    print("== LIVE A: baseline audit + facts ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 39/7,644,998/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    argmin_id, argmin_loss, n_ck = live_argmin()
    print(f"    live-computed argmin: {argmin_id} @ {argmin_loss} "
          f"among {n_ck} checkpoints")
    model_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    latest_before = model_manifest["latest_checkpoint"]
    check("A2 best != latest as concrete ids (a persisted-loss TIE broken "
          "by the canonical (step, created_at) tie-break — the manual "
          "drift M57 removes)",
          argmin_id != latest_before, f"best={argmin_id} "
          f"latest={latest_before}")
    code, m52 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("A3 M52 endpoint agrees with the live-computed argmin",
          code == 200
          and m52["checkpoint"]["checkpoint_id"] == argmin_id
          and m52["candidate_count"] == n_ck)
    check("A4 pre-state listings 1/1/0/0/0/0/1",
          (len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]),
           len(call_json("GET",
                         f"{BASE}/models/{MODEL}/gates/decisions")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]),
           len(call_json("GET", f"{BASE}/workflows/recipes")[1]))
          == (1, 1, 0, 0, 0, 0, 1))
    code, dash_pre = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    src_man = (CKPT_DIR / argmin_id / "manifest.json").read_bytes()
    src_w = (CKPT_DIR / argmin_id / "weights.pt").read_bytes()

    print("== LIVE B: register the canonical declarative recipe ==")
    probe = {"model_id": MODEL, "dataset_id": DATASET,
             "tokenizer_id": TOKENIZER, "split": "validation",
             "batch_size": 8, "max_seq_len": 32, "seed": 2}
    train1 = {"name": "loop-train-1", "method": "continued_pretraining",
              "model_id": MODEL, "dataset_id": DATASET,
              "tokenizer_id": TOKENIZER, "learning_rate": 3e-3,
              "batch_size": 8, "max_seq_len": 32, "lr_schedule": "cosine",
              "steps": 4, "eval_every_steps": 2, "keep_best": True,
              "seed": 15}
    train2 = {**train1, "name": "loop-train-2", "seed": 16,
              "resume_from_best": True}
    best_ev = {"config": probe, "checkpoint_from_best": True}
    gate_policy = {"name": "loop-gate", "model_id": MODEL,
                   "dataset_id": DATASET, "tokenizer_id": TOKENIZER,
                   "split": "validation", "batch_size": 8,
                   "max_seq_len": 32, "seed": 2,
                   "baseline_type": "checkpoint", "tolerance": 1.0,
                   "baseline_from_best": True}
    code, _ = call_json("POST", f"{BASE}/workflows/recipes", body={
        "recipe_id": RECIPE,
        "description": "M57 live: the canonical finite improvement loop "
                       "train -> evaluate(best) -> gate(candidate vs best) "
                       "-> train from best -> evaluate(best), fully "
                       "declarative",
        "stages": [
            {"stage_id": "tr1", "type": "train", "training": train1},
            {"stage_id": "ev1", "type": "evaluate", "evaluation": best_ev},
            {"stage_id": "gate1", "type": "gate", "on_pass": "tr2",
             "gate": {"policy": gate_policy,
                      "candidate": {"state_kind": "checkpoint",
                                    "from_stage": "tr1"}}},
            {"stage_id": "tr2", "type": "train", "training": train2},
            {"stage_id": "ev2", "type": "evaluate", "evaluation": best_ev},
        ]})
    check("B1 registration -> 201", code == 201, str(code))
    n, _, t, snap = audit("after registration")
    grown = new_files(pre_snap, snap)
    check("B2 exactly ONE new file (the recipe manifest), baseline "
          "byte-identical",
          (n, t) == (PRE_FILES + 1, 0)
          and grown == [f"workflow-recipes/{RECIPE}/manifest.json"]
          and changed_files(pre_snap, snap) == [], str(grown))
    rman_bytes = (ROOT / "workflow-recipes" / RECIPE
                  / "manifest.json").read_bytes()
    rman = json.loads(rman_bytes)
    gp_decl = rman["stages"][2]["gate"]["policy"]
    ev1_decl = rman["stages"][1]["evaluation"]
    ev2_decl = rman["stages"][4]["evaluation"]
    tr2_decl = rman["stages"][3]["training"]
    check("B3 the stored recipe is DECLARATIVE (gate: best baseline, "
          "nothing pinned, no explicit id; evaluate stages: best; the "
          "resume train stage: best)",
          gp_decl["baseline_from_best"] is True
          and gp_decl.get("resolved_baseline_checkpoint_id") is None
          and gp_decl.get("baseline_checkpoint_id") is None
          and ev1_decl["checkpoint_from_best"] is True
          and ev1_decl.get("resolved_checkpoint_id") is None
          and ev2_decl["checkpoint_from_best"] is True
          and ev2_decl.get("resolved_checkpoint_id") is None
          and tr2_decl["resume_from_best"] is True
          and tr2_decl.get("resolved_resume_checkpoint_id") is None
          and tr2_decl.get("resume_from_checkpoint_id") is None)
    check("B4 registry 1 -> 2",
          len(call_json("GET", f"{BASE}/workflows/recipes")[1]) == 2)

    print("== LIVE C: preflight -> ONE execution ==")
    code, res = call_json("GET", BASE + PLAN_PATH.format(m=MODEL,
                                                         r=RECIPE))
    stages_pre = res["plan"]["stages"] if code == 200 else []
    pins_pre = [stages_pre[1]["evaluation"]["resolved_checkpoint_id"],
                stages_pre[2]["gate"]["policy"]
                ["resolved_baseline_checkpoint_id"],
                stages_pre[3]["training"]["resolved_resume_checkpoint_id"],
                stages_pre[4]["evaluation"]["resolved_checkpoint_id"]] \
        if len(stages_pre) == 5 else [None] * 4
    check("C1 preflight -> 200 and pins the SAME live argmin onto ALL "
          "FOUR best declarations (ONE selection per plan)",
          code == 200
          and pins_pre == [argmin_id] * 4
          and stages_pre[2]["gate"]["policy"]["baseline_checkpoint_id"]
          is None
          and stages_pre[3]["training"]["resume_from_checkpoint_id"]
          is None,
          "pins -> " + str(pins_pre))
    raw1 = call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
    check("C2 preflight x2 byte-identical + ZERO writes",
          call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
          == raw1
          and audit("after preflights")[3] == snap)

    code, rec = call_json("POST", BASE + RUNS_PATH.format(r=RECIPE),
                          body={"model_id": MODEL})
    check("C3 execution -> 200 completed (synchronous; the gate PASSED "
          "within tolerance so on_pass advanced the loop)",
          code == 200 and rec["status"] == "completed",
          f"code {code} status {rec.get('status') if rec else '?'}")
    stages = rec["plan"]["stages"]
    pins = [stages[1]["evaluation"]["resolved_checkpoint_id"],
            stages[2]["gate"]["policy"]
            ["resolved_baseline_checkpoint_id"],
            stages[3]["training"]["resolved_resume_checkpoint_id"],
            stages[4]["evaluation"]["resolved_checkpoint_id"]]
    check("C4 the record's plan pins the SAME concrete ids "
          "(byte-identical to the preflight plan)",
          pins == [argmin_id] * 4
          and rec["plan"] == res["plan"]
          and rec["recipe_hash"] == res["recipe_hash"])
    dec_id = rec["stages"][2]["artifact"]["artifact_id"]
    code, dec = call_json("GET",
                          f"{BASE}/models/{MODEL}/gates/decisions")
    dec = next((d for d in dec if d["decision_id"] == dec_id), None) \
        if code == 200 else None
    tr1_final = rec["stages"][0]["artifact"]["checkpoint_id"]
    ev1_id = rec["stages"][1]["artifact"]["artifact_id"]
    check("C5 the gate judged tr1's output against the PLAN-START best "
          "and PASSED; the decision embeds the PURE M6 policy (explicit "
          "baseline id, best stripped) and its baseline side identifies "
          "the CONCRETE checkpoint",
          dec is not None
          and dec["decision"] == "passed"
          and dec["candidate"]["checkpoint_id"] == tr1_final
          and dec["baseline"]["state_kind"] == "checkpoint"
          and dec["baseline"]["checkpoint_id"] == argmin_id
          and dec["policy"]["baseline_checkpoint_id"] == argmin_id
          and dec["policy"]["baseline_from_best"] is False,
          f"candidate {tr1_final} vs baseline {argmin_id}, "
          f"delta {dec.get('delta_loss_nats') if dec else '?'}")
    check("C6 the gate's baseline evaluation REUSED the loop's "
          "evaluate(best) record (evidence identity = concrete checkpoint "
          "+ probe, never the 'best' declaration)",
          dec is not None
          and dec["baseline"]["evaluation_id"] == ev1_id)
    check("C7 ev2 REUSED ev1's exact evaluation; evaluations 1 -> 3 "
          "(ev1 for the best + one NEW candidate evaluation for the "
          "gate, not one per declaration)",
          rec["stages"][4]["artifact"]["artifact_id"] == ev1_id
          and rec["stages"][4]["artifact"]["checkpoint_id"] == argmin_id
          and len(call_json(
              "GET", f"{BASE}/models/{MODEL}/evaluations")[1]) == 3
          and len(call_json(
              "GET", f"{BASE}/models/{MODEL}/comparisons")[1]) == 1
          and len(call_json(
              "GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]) == 1)
    tr2_run = rec["stages"][3]["artifact"]["artifact_id"]
    post_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    prov = [p for p in post_manifest["training_provenance"]
            if p["run_id"] == tr2_run][0]
    check("C8 train-from-best provenance: initial == the selection; the "
          "engine received a PURE M54 explicit resume",
          prov["initial_checkpoint_id"] == argmin_id
          and prov["parent_checkpoint_id"] == argmin_id
          and prov["config"]["resume_from_checkpoint_id"] == argmin_id
          and prov["config"]["resume_from_best"] is False)
    new_ck = sorted((c for c in call_json(
        "GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
        if c["run_id"] == tr2_run), key=lambda c: c["step"])
    check("C9 the resume run's checkpoints DESCEND from the selection",
          len(new_ck) == 2
          and new_ck[0]["parent_checkpoint_id"] == argmin_id
          and new_ck[1]["parent_checkpoint_id"]
          == new_ck[0]["checkpoint_id"],
          "; ".join(f"{c['checkpoint_id']}<-{c['parent_checkpoint_id']}"
                    for c in new_ck))
    check("C10 the recipe manifest is byte-identical after execution "
          "(declarative definition never rewritten)",
          (ROOT / "workflow-recipes" / RECIPE
           / "manifest.json").read_bytes() == rman_bytes)
    check("C11 the selected checkpoint is byte-identical (manifest + "
          "weights; referenced, never copied)",
          (CKPT_DIR / argmin_id / "manifest.json").read_bytes() == src_man
          and (CKPT_DIR / argmin_id / "weights.pt").read_bytes() == src_w)
    loop_runs = {rec["stages"][0]["artifact"]["artifact_id"], tr2_run}
    loop_cks = {c["checkpoint_id"] for c in call_json(
        "GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
        if c["run_id"] in loop_runs}
    check("C12 no rollback: latest is one of the LOOP's new checkpoints "
          "(never the selected one, never an older pre-existing one)",
          post_manifest["latest_checkpoint"] in loop_cks
          and post_manifest["latest_checkpoint"] != argmin_id,
          post_manifest["latest_checkpoint"])
    check("C13 no mutable best pointer was created (no new file outside "
          "the justified set — verified fully in D)",
          True)

    print("== LIVE D: regression + final audit ==")
    argmin_post_id, argmin_post_loss, n_ck_post = live_argmin()
    code, m52_post = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("D1 M52 best resolves the live argmin over the grown registry "
          f"(moved iff the loop beat {argmin_loss})",
          code == 200
          and m52_post["checkpoint"]["checkpoint_id"] == argmin_post_id
          and m52_post["candidate_count"] == n_ck_post,
          f"{argmin_post_id} @ {argmin_post_loss}")
    code, plan_post = call_json(
        "GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))
    post_pins = [plan_post["plan"]["stages"][1]["evaluation"]
                 ["resolved_checkpoint_id"],
                 plan_post["plan"]["stages"][2]["gate"]["policy"]
                 ["resolved_baseline_checkpoint_id"],
                 plan_post["plan"]["stages"][3]["training"]
                 ["resolved_resume_checkpoint_id"],
                 plan_post["plan"]["stages"][4]["evaluation"]
                 ["resolved_checkpoint_id"]] if code == 200 else []
    check("D2 the SAME recipe's preflight re-resolves the CURRENT argmin "
          "onto all four declarations (read-only; the NEXT execution of "
          "the loop picks up the improved state — no cross-time lock)",
          code == 200 and post_pins == [argmin_post_id] * 4,
          f"pins -> {post_pins}, argmin {argmin_post_id}, "
          f"improved={argmin_post_id != argmin_id}")
    code, dash_post = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    diffs = deep_diff(dash_pre, dash_post)
    allowed = ("result_hash", "checkpoints", "training_runs",
               "artifact_graph", "diagnostics", "workflows", "summary",
               "evaluations", "comparisons", "gate_decisions")
               # persisted model-manifest facts + the ONE new workflow,
               # evaluation(s), comparison and gate-decision records this
               # recipe execution created
    bad = [d for d in diffs
           if not any(d == a or d.startswith(a + ".")
                      or d.startswith(a + "[") or d.startswith(a + " (")
                      for a in allowed)]
    check("D3 dashboard changed ONLY in checkpoint/training-run/"
          "evaluation/comparison/gate-decision/workflow/artifact-graph/"
          "summary sections", bad == [], f"unexpected: {bad[:4]}")
    check("D4 measurement surfaces: workflows 1->2, evaluations 1->3, "
          "comparisons 0->1, gate decisions 0->1; suite-runs/samples "
          "unchanged",
          len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]) == 2
          and len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]) == 3
          and len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]) == 1
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]) == 1
          and len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]) == 0
          and len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]) == 0)
    spec = json.loads(
        call("GET", "http://127.0.0.1:8779/openapi.json")[1])
    props = spec["components"]["schemas"]["GatePolicy"]["properties"]
    check("D5 OpenAPI 84 UNCHANGED with both M57 fields in the schema",
          len(spec["paths"]) == 84
          and "baseline_from_best" in props
          and "resolved_baseline_checkpoint_id" in props)

    n, b, t, snap = audit("final")
    grown = new_files(pre_snap, snap)
    modified = changed_files(pre_snap, snap)
    n_rec = sum(1 for g in grown if g.startswith("workflow-recipes/"))
    n_wf = sum(1 for g in grown
               if g.startswith(f"models/{MODEL}/workflows/"))
    n_ckf = sum(1 for g in grown
                if g.startswith(f"models/{MODEL}/checkpoints/"))
    n_evf = sum(1 for g in grown
                if g.startswith(f"models/{MODEL}/evaluations/"))
    n_cpf = sum(1 for g in grown
                if g.startswith(f"models/{MODEL}/comparisons/"))
    n_gdf = sum(1 for g in grown
                if g.startswith(f"models/{MODEL}/gates/"))
    ok_modified = set(modified) <= {
        f"models/{MODEL}/manifest.json", f"models/{MODEL}/weights.pt",
        f"models/{MODEL}/weights.sha256"}
    check("D6 final audit 53/0 tmp: 1 recipe + 1 workflow + 8 checkpoint "
          "files (2 train stages x 2 checkpoints) + 2 evaluations + 1 "
          "comparison + 1 gate decision; only existing-semantics "
          "modifications; every baseline file accounted for",
          (n, t) == (POST_FILES, 0) and n_rec == 1 and n_wf == 1
          and n_ckf == 8 and n_evf == 2 and n_cpf == 1 and n_gdf == 1
          and len(grown) == 14 and ok_modified,
          f"new: {grown}; modified: {modified}")
    for g in grown:
        print(f"    NEW {g}")

    if FAILURES:
        print(f"M57 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    improved = argmin_post_id != argmin_id
    print("M57 live smoke OK: DECLARATIVE BEST-BASELINE gate policies — "
          "ONE real production recipe (m57-live-best-gate, the canonical "
          "finite improvement loop train -> evaluate(best) -> "
          "gate(candidate vs best) -> train(resume_from_best) -> "
          "evaluate(best), fully declarative, no hard-coded checkpoint "
          "ids) resolved through the SAME M53 resolver + M52 selection "
          f"(live-computed argmin {argmin_id} @ {argmin_loss} among "
          f"{n_ck} checkpoints, best != latest as concrete ids via the "
          "canonical tie-break), ONE selection pinned at preflight "
          "(byte-identical, zero writes) AND at execution onto ALL FOUR "
          "best declarations; the gate judged tr1's output against the "
          "PLAN-START best and PASSED with the persisted decision "
          "embedding the PURE M6 policy (explicit baseline id, best "
          "stripped) and its baseline side identifying the concrete "
          "checkpoint; the gate's baseline evaluation and ev2 both "
          "REUSED ev1's exact M4 record (evaluations 1->3, not 4); the "
          "M55 train stage received a PURE M54 explicit resume with its "
          "checkpoints descending from the selection; the recipe "
          "manifest and the selected checkpoint byte-identical; no "
          "rollback (existing publication semantics); no best-pointer "
          "file; M52 resolving the live argmin over the grown registry "
          "and the SAME recipe's preflight re-resolving the CURRENT "
          f"argmin (improved={improved}); dashboard changing only in the "
          "checkpoint/training-run/evaluation/comparison/gate-decision/"
          "workflow/artifact-graph/summary sections; OpenAPI 84 "
          "unchanged; storage delta exactly 14 justified new files + the "
          "existing-semantics modifications) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
