"""M58 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M58 adds BOUNDED FINITE recipe
repetitions: WorkflowRecipeRunRequest.repetitions=N (default 1, bounded
1..16) executes the recipe N times SEQUENTIALLY through the EXISTING
single-run path — each iteration a FULL independent resolution +
execution, so every 'best' declaration (M53 refs, M55 resume, M56
evaluate, M57 gate baseline) re-resolves at THAT iteration's plan start
and may advance between iterations. N normal immutable records; no
wrapper record; no repetition storage.

The smoke registers exactly ONE new minimal immutable recipe —
m58-live-loop-x2: the CANONICAL finite improvement loop
    train -> evaluate(best) -> gate(candidate=from_stage, baseline=best)
         -> train(resume_from_best) -> evaluate(best)
(gate on_pass -> tr2; tolerance 1.0 nats so the gate passes
deterministically) — and executes it with repetitions=2 EXACTLY ONCE
(state-mutating; no repetition of the smoke itself). All other
production recipes are untouched.

Production facts (re-derived live; authoritative sources are the
persisted manifests): model 31db39e17a20 owns 15 checkpoints; the
live-computed M52 argmin is 1b6442a0fc58 @ 6.331234 while the published
latest is 7a5e2e33f75e (ALSO @ 6.331234 — a persisted-loss TIE broken
by the canonical (step, created_at) ASCENDING tie-break). Storage at
start: 53 files / 10,193,156 B / 0 .tmp (m58_pre.sha256 captured BEFORE
this smoke; the production root was honestly REBUILT before the M56
smoke after an environment loss — see the M56 final report §7).

LIVE A (baseline): exact 53/10,193,156/0 audit + per-file SHA256
inventory + all checkpoint manifests read directly + live-computed argmin
(+ M52 endpoint agreement) + latest_before + pre-state listings (2
workflows / 3 evaluations / 0 suite-runs / 1 comparison / 1 gate decision
/ 0 samples / 2 recipes) + dashboard pre-state.

LIVE B (register the canonical declarative recipe): POST /workflows/
recipes m58-live-loop-x2 -> 201; exactly ONE new file (its manifest);
the stored recipe is DECLARATIVE and carries NO repetition field (no
'repetitions' KEY anywhere in the manifest structure — the count is a
request parameter, never part of the definition; the word appears only
inside the human description text); registry 2 -> 3; all 53 baseline
files byte-identical.

EXECUTION NOTE (honest): the smoke was executed EXACTLY ONCE. One
original check (B3) was an over-broad FACT test — it searched for the
substring 'repetitions' anywhere in the manifest JSON, and the recipe's
own human description text ("... executed twice (repetitions=2) ...")
contains that word; the implementation was correct (no 'repetitions'
KEY exists anywhere in the definition). The corrected check walks the
manifest STRUCTURE for the key and was verified read-only post-hoc
(no re-execution, M55/M56/M57 discipline).

LIVE C (preflight -> ONE repeated execution): the M51 preflight (a
SINGLE-iteration resolution preview, unchanged) pins the live argmin
onto all four best declarations (x2 byte-identical, ZERO writes); then
POST .../runs {"repetitions": 2} ONCE -> 200 with the ordered batch:
repetitions=2, executed=2, stopped_early=false, two full records;
iteration 1's pins == the pre-run argmin (== the preflight);
iteration 2's pins == the M52 argmin over the registry EXCLUDING
iteration 2's own run outputs (its own plan-start state — recomputed
live from the persisted manifests); the two records are normal
immutable WorkflowRecords with recipe provenance; each iteration's
gate passed against its own plan-start best with the PURE M6 policy
embedded; plan hashes differ exactly when the resolutions differ
(advance reported honestly); evidence reuse active (ev2 reuses ev1 in
both iterations; identical content replays reuse evaluations and the
comparison); the recipe manifest byte-identical after execution; the
selected checkpoints byte-identical; no rollback; no best-pointer file.

LIVE D (regression + final audit): M52 /checkpoints/best resolves the
live argmin over the grown registry; the M35 by-recipe history lists
BOTH records of the batch; the same recipe's single-iteration preflight
re-resolves the CURRENT argmin; the dashboard changed ONLY in the
checkpoint/training-run/evaluation/comparison/gate-decision/workflow/
artifact-graph/summary sections; suite-runs/samples unchanged; OpenAPI
84 UNCHANGED with the bounded repetitions field (default 1, maximum 16)
and the WorkflowRecipeRepetitionRun component; final audit: exact
structural breakdown — 1 recipe + 2 workflow records + 16 checkpoint
files (2 iterations x 2 train stages x 2 checkpoints x 2 files) + 2
gate decisions + bounded evaluations/comparisons (reuse-dependent:
1..4 / 1..2) + at most the 3 existing-semantics modifications — every
file justified, zero tmp.

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8780 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8780/api/v1"
MODEL = "31db39e17a20"
RECIPE = "m58-live-loop-x2"          # the ONE new canonical recipe (M58)
DATASET = "58e10a1d3c9b"
TOKENIZER = "63dd8dcd3215"
ROOT = Path("/home/user/ai-model-forge-data")
CKPT_DIR = ROOT / "models" / MODEL / "checkpoints"
PLAN_PATH = "/models/{m}/workflows/recipes/{r}/plan"
RUNS_PATH = "/workflows/recipes/{r}/runs"

PRE_FILES, PRE_BYTES, PRE_TMP = 53, 10_193_156, 0

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


def argmin_over(manifests: dict) -> dict:
    ordered = sorted(manifests.values(),
                     key=lambda m: (m["step"], m["created_at"]))
    return min(ordered, key=lambda m: m["validation_loss"])


def load_ckpts() -> dict:
    out = {}
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            out[d.name] = json.loads((d / "manifest.json").read_text())
    return out


def live_argmin() -> tuple[str, float, int]:
    manifests = load_ckpts()
    best = argmin_over(manifests)
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
    check("A1 exact baseline audit 53/10,193,156/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    argmin_id, argmin_loss, n_ck = live_argmin()
    print(f"    live-computed argmin: {argmin_id} @ {argmin_loss} "
          f"among {n_ck} checkpoints")
    model_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    latest_before = model_manifest["latest_checkpoint"]
    check("A2 best != latest as concrete ids (persisted-loss TIE broken "
          "by the canonical (step, created_at) tie-break)",
          argmin_id != latest_before, f"best={argmin_id} "
          f"latest={latest_before}")
    code, m52 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("A3 M52 endpoint agrees with the live-computed argmin",
          code == 200
          and m52["checkpoint"]["checkpoint_id"] == argmin_id
          and m52["candidate_count"] == n_ck)
    check("A4 pre-state listings 2/3/0/1/1/0/2",
          (len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]),
           len(call_json("GET",
                         f"{BASE}/models/{MODEL}/gates/decisions")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]),
           len(call_json("GET", f"{BASE}/workflows/recipes")[1]))
          == (2, 3, 0, 1, 1, 0, 2))
    code, dash_pre = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")

    print("== LIVE B: register the canonical declarative recipe ==")
    probe = {"model_id": MODEL, "dataset_id": DATASET,
             "tokenizer_id": TOKENIZER, "split": "validation",
             "batch_size": 8, "max_seq_len": 32, "seed": 2}
    train1 = {"name": "loop-train-1", "method": "continued_pretraining",
              "model_id": MODEL, "dataset_id": DATASET,
              "tokenizer_id": TOKENIZER, "learning_rate": 3e-3,
              "batch_size": 8, "max_seq_len": 32, "lr_schedule": "cosine",
              "steps": 4, "eval_every_steps": 2, "keep_best": True,
              "seed": 17}
    train2 = {**train1, "name": "loop-train-2", "seed": 18,
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
        "description": "M58 live: the canonical finite improvement loop "
                       "executed twice (repetitions=2) — each iteration "
                       "resolves best independently",
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

    def _has_key(node, key) -> bool:
        if isinstance(node, dict):
            return key in node or any(_has_key(v, key)
                                      for v in node.values())
        if isinstance(node, list):
            return any(_has_key(v, key) for v in node)
        return False

    check("B3 the stored recipe is DECLARATIVE and carries NO repetition "
          "FIELD (the count is a request parameter, never part of the "
          "immutable definition — no 'repetitions' KEY anywhere in the "
          "manifest structure; the word may appear only inside the "
          "human description text)",
          not _has_key(rman, "repetitions")
          and rman["stages"][2]["gate"]["policy"]["baseline_from_best"]
          is True
          and rman["stages"][1]["evaluation"]["checkpoint_from_best"]
          is True
          and rman["stages"][3]["training"]["resume_from_best"] is True)
    check("B4 registry 2 -> 3",
          len(call_json("GET", f"{BASE}/workflows/recipes")[1]) == 3)

    print("== LIVE C: preflight -> ONE repeated execution (repetitions=2)")
    code, res = call_json("GET", BASE + PLAN_PATH.format(m=MODEL,
                                                         r=RECIPE))
    stages_pre = res["plan"]["stages"] if code == 200 else []
    pins_pre = [stages_pre[1]["evaluation"]["resolved_checkpoint_id"],
                stages_pre[2]["gate"]["policy"]
                ["resolved_baseline_checkpoint_id"],
                stages_pre[3]["training"]["resolved_resume_checkpoint_id"],
                stages_pre[4]["evaluation"]["resolved_checkpoint_id"]] \
        if len(stages_pre) == 5 else [None] * 4
    check("C1 M51 preflight (SINGLE-iteration preview, unchanged) pins "
          "the live argmin onto all four best declarations",
          code == 200 and pins_pre == [argmin_id] * 4,
          "pins -> " + str(pins_pre))
    raw1 = call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
    check("C2 preflight x2 byte-identical + ZERO writes",
          call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
          == raw1
          and audit("after preflights")[3] == snap)

    code, batch = call_json("POST", BASE + RUNS_PATH.format(r=RECIPE),
                            body={"model_id": MODEL, "repetitions": 2})
    check("C3 repeated execution -> 200 with the ordered batch "
          "(repetitions=2, executed=2, stopped_early=false)",
          code == 200 and batch is not None
          and batch["repetitions"] == 2 and batch["executed"] == 2
          and batch["stopped_early"] is False
          and len(batch["workflow_ids"]) == 2
          and len(batch["records"]) == 2
          and [r["workflow_id"] for r in batch["records"]]
          == batch["workflow_ids"],
          f"code {code}")
    rec1, rec2 = batch["records"] if code == 200 and batch else ({}, {})
    check("C4 both iterations COMPLETED; two NORMAL immutable records "
          "with recipe provenance (no wrapper record)",
          rec1.get("status") == "completed"
          and rec2.get("status") == "completed"
          and rec1.get("recipe_id") == RECIPE
          and rec2.get("recipe_id") == RECIPE
          and rec1.get("composition") is None
          and rec2.get("composition") is None
          and len(call_json(
              "GET", f"{BASE}/models/{MODEL}/workflows")[1]) == 4)

    def pins_of(rec):
        st = rec["plan"]["stages"]
        return (st[1]["evaluation"]["resolved_checkpoint_id"],
                st[2]["gate"]["policy"]
                ["resolved_baseline_checkpoint_id"],
                st[3]["training"]["resolved_resume_checkpoint_id"],
                st[4]["evaluation"]["resolved_checkpoint_id"])

    p1 = pins_of(rec1)
    check("C5 iteration 1 pinned the PRE-RUN argmin (== the preflight "
          "prediction) onto all four declarations",
          p1 == (argmin_id,) * 4, f"pins1 -> {p1}")
    # iteration 2 resolved independently: its pins describe the registry
    # at ITS plan start = everything except its own runs' outputs
    iter2_runs = {rec2["stages"][0]["artifact"]["artifact_id"],
                  rec2["stages"][3]["artifact"]["artifact_id"]}
    ckpts = load_ckpts()
    pre_iter2 = {cid: m for cid, m in ckpts.items()
                 if m["run_id"] not in iter2_runs}
    expected2 = argmin_over(pre_iter2)["checkpoint_id"]
    p2 = pins_of(rec2)
    check("C6 iteration 2 RE-RESOLVED at its own plan start (the M52 "
          "argmin over the registry EXCLUDING its own outputs — computed "
          "live from the persisted manifests)",
          p2 == (expected2,) * 4,
          f"pins2 -> {p2}, expected {expected2}")
    advanced = p2[0] != p1[0]
    check("C7 plan hashes differ exactly when the resolutions differ "
          "(deterministic identity over the RESOLVED plan)",
          (rec1["plan_hash"] != rec2["plan_hash"]) == (p1 != p2),
          f"advanced={advanced}, hash1={rec1['plan_hash'][:12]}, "
          f"hash2={rec2['plan_hash'][:12]}")
    for i, rec in enumerate((rec1, rec2), start=1):
        dec = next(d for d in call_json(
            "GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]
            if d["decision_id"]
            == rec["stages"][2]["artifact"]["artifact_id"])
        check(f"C8 iteration {i}: the gate judged tr1's output against "
              "ITS plan-start best and PASSED; the decision embeds the "
              "PURE M6 policy; ev2 REUSED ev1's exact evaluation",
              dec["decision"] == "passed"
              and dec["policy"]["baseline_from_best"] is False
              and dec["policy"]["baseline_checkpoint_id"] == pins_of(rec)[1]
              and dec["baseline"]["checkpoint_id"] == pins_of(rec)[1]
              and dec["candidate"]["checkpoint_id"]
              == rec["stages"][0]["artifact"]["checkpoint_id"]
              and (rec["stages"][4]["artifact"]["artifact_id"]
                   == rec["stages"][1]["artifact"]["artifact_id"]),
              f"delta {dec.get('delta_loss_nats')}")
    check("C9 the recipe manifest is byte-identical after the repeated "
          "execution (declarative definition never rewritten)",
          (ROOT / "workflow-recipes" / RECIPE
           / "manifest.json").read_bytes() == rman_bytes)
    post_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    loop_runs = {rec1["stages"][0]["artifact"]["artifact_id"],
                 rec1["stages"][3]["artifact"]["artifact_id"],
                 rec2["stages"][0]["artifact"]["artifact_id"],
                 rec2["stages"][3]["artifact"]["artifact_id"]}
    loop_cks = {c["checkpoint_id"] for c in call_json(
        "GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
        if c["run_id"] in loop_runs}
    check("C10 no rollback: latest is one of the LOOP's new checkpoints "
          "(never a selected best, never an older pre-existing one)",
          post_manifest["latest_checkpoint"] in loop_cks
          and post_manifest["latest_checkpoint"] not in {p1[0], p2[0]},
          post_manifest["latest_checkpoint"])
    check("C11 no mutable best pointer / repetition storage was created "
          "(no new file outside the justified set — verified fully in D)",
          True)

    print("== LIVE D: regression + final audit ==")
    argmin_post_id, argmin_post_loss, n_ck_post = live_argmin()
    code, m52_post = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("D1 M52 best resolves the live argmin over the grown registry",
          code == 200
          and m52_post["checkpoint"]["checkpoint_id"] == argmin_post_id
          and m52_post["candidate_count"] == n_ck_post,
          f"{argmin_post_id} @ {argmin_post_loss} (was {argmin_id} @ "
          f"{argmin_loss})")
    hist = call_json(
        "GET", f"{BASE}/models/{MODEL}/workflows/by-recipe/{RECIPE}")[1]
    check("D2 the M35 by-recipe history lists BOTH records of the batch",
          {w["workflow_id"] for w in hist} >= set(batch["workflow_ids"]),
          f"{len(hist)} records")
    code, plan_post = call_json(
        "GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))
    preflight_pin = (plan_post["plan"]["stages"][1]["evaluation"]
                     ["resolved_checkpoint_id"] if code == 200 else None)
    check("D3 the same recipe's single-iteration preflight re-resolves "
          "the CURRENT argmin (read-only; preflight stays a ONE-iteration "
          "preview — later iterations cannot be predicted)",
          code == 200
          and [preflight_pin,
               plan_post["plan"]["stages"][2]["gate"]["policy"]
               ["resolved_baseline_checkpoint_id"]]
          == [argmin_post_id] * 2,
          f"preflight pins {preflight_pin}, argmin {argmin_post_id}")
    code, dash_post = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    diffs = deep_diff(dash_pre, dash_post)
    allowed = ("result_hash", "checkpoints", "training_runs",
               "artifact_graph", "diagnostics", "workflows", "summary",
               "evaluations", "comparisons", "gate_decisions")
    bad = [d for d in diffs
           if not any(d == a or d.startswith(a + ".")
                      or d.startswith(a + "[") or d.startswith(a + " (")
                      for a in allowed)]
    check("D4 dashboard changed ONLY in checkpoint/training-run/"
          "evaluation/comparison/gate-decision/workflow/artifact-graph/"
          "summary sections", bad == [], f"unexpected: {bad[:4]}")
    check("D5 measurement surfaces: workflows 2->4, evaluations 3->%d, "
          "comparisons 1->%d, gate decisions 1->3; suite-runs/samples "
          "unchanged" % (
              len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]),
              len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1])),
          len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]) == 4
          and len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]) in (4, 5, 6, 7)
          and len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]) in (2, 3)
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]) == 3
          and len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]) == 0
          and len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]) == 0)
    spec = json.loads(
        call("GET", "http://127.0.0.1:8780/openapi.json")[1])
    props = spec["components"]["schemas"]["WorkflowRecipeRunRequest"][
        "properties"]
    check("D6 OpenAPI 84 UNCHANGED; repetitions bounded (default 1, "
          "maximum 16); the batch response component exists",
          len(spec["paths"]) == 84
          and props["repetitions"]["default"] == 1
          and props["repetitions"]["maximum"] == 16
          and "WorkflowRecipeRepetitionRun" in spec["components"]["schemas"])

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
    n_other = len(grown) - n_rec - n_wf - n_ckf - n_evf - n_cpf - n_gdf
    ok_modified = set(modified) <= {
        f"models/{MODEL}/manifest.json", f"models/{MODEL}/weights.pt",
        f"models/{MODEL}/weights.sha256"}
    check("D7 final audit: exact structural breakdown — 1 recipe + 2 "
          "workflow records + 16 checkpoint files (2 iterations x 2 train "
          "stages x 2 checkpoints x 2 files) + 2 gate decisions + "
          "reuse-bounded evaluations (1..4) and comparisons (1..2); only "
          "existing-semantics modifications; zero tmp; no other families",
          t == 0 and n_rec == 1 and n_wf == 2 and n_ckf == 16
          and n_gdf == 2 and 1 <= n_evf <= 4 and 1 <= n_cpf <= 2
          and n_other == 0 and ok_modified,
          f"total {n} files (+{len(grown)}): recipe {n_rec}, wf {n_wf}, "
          f"ckpt {n_ckf}, eval {n_evf}, cmp {n_cpf}, gate {n_gdf}, "
          f"other {n_other}; modified {modified}")
    for g in grown:
        print(f"    NEW {g}")

    if FAILURES:
        print(f"M58 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M58 live smoke OK: BOUNDED FINITE recipe repetitions — ONE "
          "real production recipe (m58-live-loop-x2, the canonical "
          "finite improvement loop) executed with repetitions=2 through "
          "the EXISTING single-run path: TWO normal immutable workflow "
          "records (no wrapper, no repetition storage), iteration 1 "
          f"pinning the pre-run argmin {argmin_id} @ {argmin_loss} "
          "(== the single-iteration M51 preflight, byte-identical, zero "
          "writes) and iteration 2 RE-RESOLVING at its own plan start "
          f"({expected2}; advanced={advanced}); each iteration's gate "
          "judged its tr1 output against ITS plan-start best and passed "
          "with the PURE M6 policy embedded; evidence reuse active "
          "(ev2 reuses ev1; replays reuse evaluations/comparisons); "
          "plan hashes differing exactly when the resolutions differ; "
          "the recipe manifest byte-identical (the count is a request "
          "parameter, never part of the definition); no rollback; no "
          "best-pointer file; M52 resolving the live argmin "
          f"({argmin_post_id} @ {argmin_post_loss}) over the grown "
          "registry; M35 by-recipe history listing both records; "
          "dashboard changing only in the expected sections; OpenAPI 84 "
          "unchanged with the bounded field; storage delta exactly the "
          f"per-iteration artifacts ({len(grown)} new files: 1 recipe + "
          f"2 workflows + 16 checkpoint files + {n_evf} evaluations + "
          f"{n_cpf} comparisons + 2 gate decisions) + the "
          "existing-semantics modifications) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
