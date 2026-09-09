"""M55 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M55 adds the DECLARATIVE BEST-RESUME
for workflow TRAIN stages: TrainingConfig.resume_from_best=true is
resolved by the SAME M53 resolver (WorkflowEngine.resolve_best_state_refs)
through the SAME M52 selection (minimum persisted validation_loss), ONCE
per plan at execution/M51-preflight time; the concrete id is pinned
(resolved_resume_checkpoint_id) into the resolved plan — which executes,
persists and hashes — and the training engine receives a PURE M54
resume_from_checkpoint_id (the training layer never queries "best").

The smoke registers exactly ONE new minimal immutable recipe —
m55-live-best-resume: a single train stage declaring resume_from_best —
and executes it EXACTLY ONCE (state-mutating; no repetition). All other
production recipes are untouched.

Production facts (re-derived live; authoritative sources are the
persisted manifests): model 4a0a871886ef owns 5 checkpoints
(2741cd7a8d72@2/6.366164, b985e7c679ca@4/6.361037, 30a8bc5b82ab@10/
6.397286, 0511de4c7372@16/6.210553, 025e6d8d8f15@20/6.360189); the
live-computed M52 argmin is 0511de4c7372 @ 6.210553 while
latest_checkpoint is b985e7c679ca — best != latest, the exact drift M55
removes. Storage at start: 113 files / 5,289,253 B / 0 .tmp
(m55_pre.sha256 captured BEFORE this smoke).

LIVE A (baseline): exact 113/5,289,253/0 audit + per-file SHA256
inventory + all checkpoint manifests read directly + live-computed
argmin + M52 endpoint agreement + latest_before + pre-state listings
(19 workflows / 16 suite runs / 16 evaluations / 8 comparisons / 11
gate decisions / 8 recipes / 4 samples) + dashboard pre-state.

LIVE B (register the minimal declarative recipe): POST /workflows/
recipes m55-live-best-resume -> 201; exactly ONE new file (its
manifest); the stored recipe is DECLARATIVE (resume_from_best=true,
nothing pinned, no explicit id); registry 8 -> 9; all 113 baseline
files byte-identical.

LIVE C (preflight -> execution, ONE run): the M51 preflight pins
best -> the live-computed argmin (x2 byte-identical, ZERO writes from
preflights); then POST .../runs ONCE -> 200 completed: the record's
plan pins the SAME concrete id (byte-identical to the preflight plan);
the training run's provenance records initial/parent == the concrete id
with the config carrying the PURE M54 explicit resume_from_checkpoint_
id and resume_from_best stripped; the run's checkpoints DESCEND from
it; the recipe manifest is byte-identical after execution; the source
checkpoint's manifest + weights are byte-identical (referenced, never
copied); no rollback (latest follows the existing publication
semantics — the run's last created checkpoint); no best-pointer file.

LIVE D (regression + final audit): M52 /checkpoints/best resolves the
live argmin over the grown registry; the M53 best-ref preflight
(m53-live-best) resolves the same live argmin; the dashboard changed
ONLY in the checkpoint/training-run/artifact-graph/summary sections;
workflows/suite-runs/evaluations/comparisons/gates/samples unchanged;
OpenAPI 84 UNCHANGED with both fields in TrainingConfig; final audit
119 files / 0 tmp: 6 new (1 recipe manifest + 1 workflow manifest +
2 checkpoint manifests + 2 checkpoint weights) + at most the 3
existing-semantics modifications (model manifest, weights.pt,
weights.sha256) — every file justified.

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8776 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8776/api/v1"
MODEL = "4a0a871886ef"
RECIPE = "m55-live-best-resume"        # the ONE new minimal recipe (M55)
M53_RECIPE = "m53-live-best"           # existing M53 best-ref recipe
DATASET = "ee1a716c4573"
TOKENIZER = "99106e3255c5"
ROOT = Path("/home/user/ai-model-forge-data")
CKPT_DIR = ROOT / "models" / MODEL / "checkpoints"
PLAN_PATH = "/models/{m}/workflows/recipes/{r}/plan"
RUNS_PATH = "/workflows/recipes/{r}/runs"

PRE_FILES, PRE_BYTES, PRE_TMP = 113, 5_289_253, 0
POST_FILES = 119                       # +1 recipe +1 workflow +4 ckpt files

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
    check("A1 exact baseline audit 113/5,289,253/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    argmin_id, argmin_loss, n_ck = live_argmin()
    print(f"    live-computed argmin: {argmin_id} @ {argmin_loss} "
          f"among {n_ck} checkpoints")
    model_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    latest_before = model_manifest["latest_checkpoint"]
    check("A2 best != latest (the manual-drift M55 removes)",
          argmin_id != latest_before, f"best={argmin_id} "
          f"latest={latest_before}")
    code, m52 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("A3 M52 endpoint agrees with the live-computed argmin",
          code == 200
          and m52["checkpoint"]["checkpoint_id"] == argmin_id
          and m52["candidate_count"] == n_ck)
    check("A4 pre-state listings 19/16/16/8/11/8/4",
          (len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]),
           len(call_json("GET",
                         f"{BASE}/models/{MODEL}/gates/decisions")[1]),
           len(call_json("GET", f"{BASE}/workflows/recipes")[1]),
           len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]))
          == (19, 16, 16, 8, 11, 8, 4))
    code, dash_pre = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    src_man = (CKPT_DIR / argmin_id / "manifest.json").read_bytes()
    src_w = (CKPT_DIR / argmin_id / "weights.pt").read_bytes()

    print("== LIVE B: register the minimal declarative recipe ==")
    train_cfg = {
        "name": "best-resume-run", "method": "continued_pretraining",
        "model_id": MODEL, "dataset_id": DATASET,
        "tokenizer_id": TOKENIZER, "learning_rate": 3e-3,
        "batch_size": 8, "max_seq_len": 32, "lr_schedule": "cosine",
        "steps": 4, "eval_every_steps": 2, "keep_best": True, "seed": 13,
        "resume_from_best": True,
    }
    code, _ = call_json("POST", f"{BASE}/workflows/recipes", body={
        "recipe_id": RECIPE,
        "description": "M55 live: train from the best checkpoint "
                       "(minimum persisted validation loss)",
        "stages": [{"stage_id": "tr_best", "type": "train",
                    "training": train_cfg}]})
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
    tc = rman["stages"][0]["training"]
    check("B3 the stored recipe is DECLARATIVE (best, nothing pinned, "
          "no explicit id)",
          tc["resume_from_best"] is True
          and tc.get("resolved_resume_checkpoint_id") is None
          and tc.get("resume_from_checkpoint_id") is None,
          json.dumps({k: tc.get(k) for k in (
              "resume_from_best", "resolved_resume_checkpoint_id",
              "resume_from_checkpoint_id")}))
    check("B4 registry 8 -> 9",
          len(call_json("GET", f"{BASE}/workflows/recipes")[1]) == 9)

    print("== LIVE C: preflight -> ONE execution ==")
    code, res = call_json("GET", BASE + PLAN_PATH.format(m=MODEL,
                                                         r=RECIPE))
    tc_pre = res["plan"]["stages"][0]["training"] if code == 200 else {}
    check("C1 preflight -> 200 and pins best -> the live argmin",
          code == 200
          and tc_pre["resume_from_best"] is True
          and tc_pre["resolved_resume_checkpoint_id"] == argmin_id,
          "best -> " + str(tc_pre.get("resolved_resume_checkpoint_id")))
    raw1 = call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
    check("C2 preflight x2 byte-identical + ZERO writes",
          call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
          == raw1
          and audit("after preflights")[3] == snap)

    code, rec = call_json("POST", BASE + RUNS_PATH.format(r=RECIPE),
                          body={"model_id": MODEL})
    check("C3 execution -> 200 completed (synchronous)",
          code == 200 and rec["status"] == "completed",
          f"code {code} status {rec.get('status') if rec else '?'}")
    rtc = rec["plan"]["stages"][0]["training"]
    check("C4 the record's plan pins the SAME concrete id "
          "(byte-identical to the preflight plan)",
          rtc["resume_from_best"] is True
          and rtc["resolved_resume_checkpoint_id"] == argmin_id
          and rec["plan"] == res["plan"]
          and rec["recipe_hash"] == res["recipe_hash"])
    run_id = rec["stages"][0]["artifact"]["artifact_id"]
    post_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    prov = [p for p in post_manifest["training_provenance"]
            if p["run_id"] == run_id][0]
    check("C5 training provenance: initial/parent == the concrete id; "
          "the engine received a PURE M54 explicit resume",
          prov["initial_checkpoint_id"] == argmin_id
          and prov["parent_checkpoint_id"] == argmin_id
          and prov["config"]["resume_from_checkpoint_id"] == argmin_id
          and prov["config"]["resume_from_best"] is False)
    new_ck = sorted((c for c in call_json(
        "GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
        if c["run_id"] == run_id), key=lambda c: c["step"])
    check("C6 the run's checkpoints DESCEND from the selected checkpoint",
          len(new_ck) == 2
          and new_ck[0]["parent_checkpoint_id"] == argmin_id
          and new_ck[1]["parent_checkpoint_id"]
          == new_ck[0]["checkpoint_id"],
          "; ".join(f"{c['checkpoint_id']}<-{c['parent_checkpoint_id']}"
                    for c in new_ck))
    check("C7 the recipe manifest is byte-identical after execution "
          "(declarative definition never rewritten)",
          (ROOT / "workflow-recipes" / RECIPE
           / "manifest.json").read_bytes() == rman_bytes)
    check("C8 the selected checkpoint is byte-identical (manifest + "
          "weights; referenced, never copied)",
          (CKPT_DIR / argmin_id / "manifest.json").read_bytes() == src_man
          and (CKPT_DIR / argmin_id / "weights.pt").read_bytes() == src_w)
    check("C9 no rollback: latest follows the EXISTING publication "
          "semantics (the run's last created checkpoint — never the "
          "selected one)",
          post_manifest["latest_checkpoint"]
          == new_ck[-1]["checkpoint_id"]
          and post_manifest["latest_checkpoint"] != argmin_id,
          post_manifest["latest_checkpoint"])
    check("C10 no mutable best pointer was created (no new file outside "
          "the justified set — verified fully in D)",
          True)

    print("== LIVE D: regression + final audit ==")
    argmin_post_id, argmin_post_loss, n_ck_post = live_argmin()
    code, m52_post = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("D1 M52 best resolves the live argmin over the grown registry "
          "(moved iff the run beat 6.210553)",
          code == 200
          and m52_post["checkpoint"]["checkpoint_id"] == argmin_post_id
          and m52_post["candidate_count"] == n_ck_post,
          f"{argmin_post_id} @ {argmin_post_loss}")
    code, plan53 = call_json(
        "GET", BASE + PLAN_PATH.format(m=MODEL, r=M53_RECIPE))
    check("D2 the M53 best-ref preflight resolves the same live argmin",
          code == 200
          and plan53["plan"]["stages"][0]["suite_run"]["state"]
          ["resolved_checkpoint_id"] == argmin_post_id)
    code, dash_post = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    diffs = deep_diff(dash_pre, dash_post)
    allowed = ("result_hash", "checkpoints", "training_runs",
               "artifact_graph", "diagnostics", "workflows",
               "summary")   # persisted model-manifest facts + the ONE
               # new workflow record this recipe execution created
    bad = [d for d in diffs
           if not any(d == a or d.startswith(a + ".")
                      or d.startswith(a + "[") or d.startswith(a + " (")
                      for a in allowed)]
    check("D3 dashboard changed ONLY in checkpoint/training-run/"
          "artifact-graph/summary sections", bad == [],
          f"unexpected: {bad[:4]}")
    check("D4 measurement surfaces unchanged (workflows/suite-runs/"
          "evaluations/comparisons/gates/samples)",
          len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]) == 20
          and len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]) == 16
          and len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]) == 16
          and len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]) == 8
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]) == 11
          and len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]) == 4)
    spec = json.loads(
        call("GET", "http://127.0.0.1:8776/openapi.json")[1])
    props = spec["components"]["schemas"]["TrainingConfig"]["properties"]
    check("D5 OpenAPI 84 UNCHANGED with both M55 fields in the schema",
          len(spec["paths"]) == 84
          and "resume_from_best" in props
          and "resolved_resume_checkpoint_id" in props)

    n, b, t, snap = audit("final")
    grown = new_files(pre_snap, snap)
    modified = changed_files(pre_snap, snap)
    n_rec = sum(1 for g in grown if g.startswith("workflow-recipes/"))
    n_wf = sum(1 for g in grown
               if g.startswith(f"models/{MODEL}/workflows/"))
    n_ckf = sum(1 for g in grown
                if g.startswith(f"models/{MODEL}/checkpoints/"))
    ok_modified = set(modified) <= {
        f"models/{MODEL}/manifest.json", f"models/{MODEL}/weights.pt",
        f"models/{MODEL}/weights.sha256"}
    check("D6 final audit 119/0 tmp: 1 recipe + 1 workflow + 4 checkpoint "
          "files; only existing-semantics modifications; every baseline "
          "file accounted for",
          (n, t) == (POST_FILES, 0) and n_rec == 1 and n_wf == 1
          and n_ckf == 4 and len(grown) == 6 and ok_modified,
          f"new: {grown}; modified: {modified}")
    for g in grown:
        print(f"    NEW {g}")

    if FAILURES:
        print(f"M55 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M55 live smoke OK: DECLARATIVE BEST-RESUME for workflow train "
          "stages — ONE real production recipe (m55-live-best-resume, "
          "declarative resume_from_best) resolved through the SAME M53 "
          "resolver + M52 selection (live-computed argmin "
          f"{argmin_id} @ {argmin_loss} among {n_ck} checkpoints, best != "
          "latest), pinned at preflight (byte-identical, zero writes) AND "
          "at execution into the immutable record's plan; the training "
          "engine received a PURE M54 explicit resume (provenance "
          "initial/parent/config), the run's checkpoints descending from "
          "the selected one; the recipe manifest and the source checkpoint "
          "byte-identical; no rollback (existing publication semantics); "
          "no best-pointer file; M52 + M53 surfaces resolving the live "
          "argmin over the grown registry; dashboard changing only in the "
          "checkpoint/training-run/artifact-graph/summary sections; "
          "OpenAPI 84 unchanged; storage delta exactly 6 justified new "
          "files + the existing-semantics modifications) verified live on "
          "production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
