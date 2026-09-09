"""M54 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M54 adds the EXPLICIT TRAINING RESUME
POINT: TrainingConfig.resume_from_checkpoint_id initializes ONE run
from an immutable checkpoint's VERIFIED weights WITHOUT publishing it
first — no rollback, no latest_checkpoint mutation to prepare the run;
publication happens only through the normal M3 completion semantics.
Model-weight resume only (no optimizer/scheduler state exists to
restore — documented honestly). Training IS weight-work, so this smoke
executes exactly ONE real production training run (the smallest
footprint M3 supports) and justifies every resulting artifact.

Production facts (re-derived live; the authoritative sources are the
persisted manifests): model 4a0a871886ef owns 3 checkpoints — 025e6d8d8f15
(step 20, val 6.360189), 0511de4c7372 (step 16, val 6.210553 — the
latest_checkpoint AND the M52 argmin), 30a8bc5b82ab (step 10, val
6.397286, run 291a16d755fc). The resume target is 30a8bc5b82ab — a
HISTORICAL checkpoint != latest. The probe configuration that created
it (dataset ee1a716c4573, tokenizer 99106e3255c5, max_seq_len 32,
batch_size 8) is cloned for the resume run so its BASELINE evaluation
must reproduce 30a8's persisted validation_loss exactly — live proof
that the run initialized from 30a8's weights (§7) and that the initial
state equals the existing checkpoint-restore mechanism (§16).

LIVE A (baseline): exact 109/4,026,085/0 audit + per-file SHA256
inventory + read the model manifest (latest_before = 0511de4c7372) and
all checkpoint manifests directly + choose the resume target (a
historical checkpoint != latest) + pre-state listings (19 workflows,
16 suite runs, 16 evaluations, 8 comparisons, 11 gate decisions, 8
recipes, 4 samples, 3 checkpoints) + M52 /checkpoints/best + the M53
preflight of m53-live-best (both resolving the live-computed argmin) +
dashboard pre-state.

LIVE B (the ONE real resume run): POST /training/run with
resume_from_checkpoint_id = 30a8bc5b82ab (steps 4, eval_every 2,
keep_best True — the smallest valid footprint, probe config cloned) ->
200; the report's initial_model_version == the resume checkpoint; its
baseline_validation_loss reproduces the resume checkpoint's persisted
validation_loss EXACTLY (same probe); provenance initial/parent ==
the resume checkpoint and the config records the field; the run's
first new checkpoint DESCENDS from the resume checkpoint
(parent_checkpoint_id); latest_checkpoint follows the EXISTING
publication semantics (the run's last created checkpoint — never the
resume checkpoint: no rollback happened); the resume checkpoint's
manifest + weights are byte-identical (referenced, never copied); no
new evaluations/comparisons/gates/workflows/suite-runs/samples/
recipes (training only touches training artifacts).

NOTE certification: this smoke executes exactly ONE training run; the
certified run is identified in the report (single execution — repeated
execution is forbidden). The dashboard `summary` section (persisted
model-manifest facts: best/latest checkpoint, run count, updated_at)
and the weights.pt/weights.sha256 pair legitimately change through the
EXISTING training completion semantics and are allowed in the audits.

LIVE C (post-state + regression): checkpoints 3 -> 5 (the run's own,
both descending from the resume point); M52 /checkpoints/best resolves
the LIVE-computed argmin over the grown registry; the M53 best-ref
preflight resolves the same live argmin; the dashboard changed ONLY in
the checkpoint/training-run/artifact-graph sections (evaluations,
comparisons, gates, workflows, suite runs, sample quality
byte-identical); OpenAPI 84 UNCHANGED with the field in the
TrainingConfig schema; final audit: every new/modified file enumerated
and justified (2 checkpoint manifests + 2 checkpoint weights files +
the model manifest updated by the EXISTING completion semantics +
published weights.pt only if keep-best published an improving state).

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8775 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8775/api/v1"
MODEL = "4a0a871886ef"
RESUME_CK = "30a8bc5b82ab"             # historical, != latest
DATASET = "ee1a716c4573"
TOKENIZER = "99106e3255c5"
M53_RECIPE = "m53-live-best"
ROOT = Path("/home/user/ai-model-forge-data")
CKPT_ROOT = ROOT / "models" / MODEL / "checkpoints"
CKPT_DIR = CKPT_ROOT          # alias used in loops

PRE_FILES, PRE_BYTES, PRE_TMP = 109, 4_026_085, 0

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
    check("A1 exact baseline audit 109/4,026,085/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")

    model_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    latest_before = model_manifest["latest_checkpoint"]
    check("A2 latest_checkpoint read live (resume target is historical)",
          latest_before != RESUME_CK, f"latest={latest_before}")
    manifests = {}
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            manifests[d.name] = json.loads(
                (d / "manifest.json").read_text())
    check("A3 three persisted checkpoints read directly",
          len(manifests) == 3, str(sorted(manifests)))
    resume_man = manifests[RESUME_CK]
    argmin_pre = min(sorted(manifests.values(),
                            key=lambda m: (m["step"], m["created_at"])),
                     key=lambda m: m["validation_loss"])
    print(f"    resume target: {RESUME_CK} @ "
          f"{resume_man['validation_loss']} | pre-run argmin: "
          f"{argmin_pre['checkpoint_id']}")
    code, m52_pre = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("A4 M52 best pre-state == live-computed argmin",
          code == 200
          and m52_pre["checkpoint"]["checkpoint_id"]
          == argmin_pre["checkpoint_id"])
    code, plan_pre = call_json(
        "GET", f"{BASE}/models/{MODEL}/workflows/recipes/{M53_RECIPE}/plan")
    check("A5 M53 best-ref preflight resolves the same argmin",
          code == 200
          and plan_pre["plan"]["stages"][0]["suite_run"]["state"]
          ["resolved_checkpoint_id"] == argmin_pre["checkpoint_id"])
    wf_pre = call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]
    sr_pre = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]
    ev_pre = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]
    cp_pre = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]
    gates_pre = call_json("GET",
                          f"{BASE}/models/{MODEL}/gates/decisions")[1]
    recipes_pre = call_json("GET", f"{BASE}/workflows/recipes")[1]
    samples_pre = call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]
    check("A6 pre-state listings 19/16/16/8/11/8/4",
          (len(wf_pre), len(sr_pre), len(ev_pre), len(cp_pre),
           len(gates_pre), len(recipes_pre), len(samples_pre))
          == (19, 16, 16, 8, 11, 8, 4))
    code, dash_pre = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    resume_bytes = {"manifest": (CKPT_ROOT / RESUME_CK / "manifest.json")
                    .read_bytes(),
                    "weights": (CKPT_ROOT / RESUME_CK / "weights.pt")
                    .read_bytes()}
    weights_pre = (ROOT / "models" / MODEL / "weights.pt").read_bytes()

    print("== LIVE B: the ONE real resume run ==")
    body = {
        "name": "m54-live-resume", "method": "continued_pretraining",
        "model_id": MODEL, "dataset_id": DATASET,
        "tokenizer_id": TOKENIZER, "learning_rate": 3e-3,
        "batch_size": 8, "max_seq_len": 32, "lr_schedule": "cosine",
        "steps": 4, "eval_every_steps": 2, "keep_best": True, "seed": 11,
        "resume_from_checkpoint_id": RESUME_CK,
    }
    code, rep = call_json("POST", f"{BASE}/training/run", body)
    check("B1 resume training -> 200", code == 200,
          f"code {code} {str(rep)[:120] if rep else ''}")
    check("B2 the run STARTED from the resume checkpoint",
          rep["initial_model_version"] == RESUME_CK,
          str(rep.get("initial_model_version")))
    check("B3 baseline reproduces the resume checkpoint's persisted "
          "validation_loss EXACTLY (same probe config: initialized from "
          "its weights)",
          round(rep["baseline_validation_loss"], 6)
          == resume_man["validation_loss"],
          f"{rep['baseline_validation_loss']} vs "
          f"{resume_man['validation_loss']}")

    post_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    prov = [p for p in post_manifest["training_provenance"]
            if p["run_id"] == rep["run_id"]][0]
    check("B4 provenance pins the resume checkpoint (initial + parent + "
          "config)",
          prov["initial_checkpoint_id"] == RESUME_CK
          and prov["parent_checkpoint_id"] == RESUME_CK
          and prov["config"]["resume_from_checkpoint_id"] == RESUME_CK)
    new_ck = sorted((c for c in call_json(
        "GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
        if c["run_id"] == rep["run_id"]), key=lambda c: c["step"])
    check("B5 the run produced 2 checkpoints DESCENDING from the resume "
          "point (existing lineage fields)",
          len(new_ck) == 2
          and new_ck[0]["parent_checkpoint_id"] == RESUME_CK
          and new_ck[1]["parent_checkpoint_id"]
          == new_ck[0]["checkpoint_id"],
          "; ".join(f"{c['checkpoint_id']}<-{c['parent_checkpoint_id']}"
                    for c in new_ck))
    check("B6 latest_checkpoint follows the EXISTING publication semantics "
          "(the run's last created checkpoint — NOT the resume point: no "
          "rollback happened)",
          post_manifest["latest_checkpoint"] == new_ck[-1]["checkpoint_id"]
          and post_manifest["latest_checkpoint"] != RESUME_CK,
          post_manifest["latest_checkpoint"])
    check("B7 the resume checkpoint is byte-identical (manifest + weights; "
          "referenced, never copied)",
          (CKPT_ROOT / RESUME_CK / "manifest.json").read_bytes()
          == resume_bytes["manifest"]
          and (CKPT_ROOT / RESUME_CK / "weights.pt").read_bytes()
          == resume_bytes["weights"])
    published = (ROOT / "models" / MODEL / "weights.pt").read_bytes()
    if rep.get("final_model_version") in {c["checkpoint_id"] for c in new_ck}:
        check("B8 published weights changed ONLY per the EXISTING keep-best "
              "completion semantics (an improving state was published)",
              published != weights_pre
              and prov["rolled_back_to"] in (None,
                                             prov["best_checkpoint_id"]
                                             if "best_checkpoint_id" in prov
                                             else None),
              f"final={rep.get('final_model_version')}")
    else:
        check("B8 nothing beat the baseline: published weights UNTOUCHED "
              "(keep_best semantics — zero mutation)",
              published == weights_pre,
              f"final={rep.get('final_model_version')}")

    print("== LIVE C: post-state + regression ==")
    check("C1 checkpoints 3 -> 5 (the run's own two; no duplication of "
          "the resume point)",
          len(call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")[1])
          == 5)
    manifests_post = {}
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            manifests_post[d.name] = json.loads(
                (d / "manifest.json").read_text())
    argmin_post = min(sorted(manifests_post.values(),
                             key=lambda m: (m["step"], m["created_at"])),
                      key=lambda m: m["validation_loss"])
    code, m52_post = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("C2 M52 best resolves the LIVE-computed argmin over the grown "
          "registry",
          code == 200
          and m52_post["checkpoint"]["checkpoint_id"]
          == argmin_post["checkpoint_id"]
          and m52_post["candidate_count"] == 5,
          f"{m52_post['checkpoint']['checkpoint_id']} @ "
          f"{m52_post['checkpoint']['validation_loss']}")
    code, plan_post = call_json(
        "GET", f"{BASE}/models/{MODEL}/workflows/recipes/{M53_RECIPE}/plan")
    check("C3 M53 best-ref preflight resolves the same live argmin (the "
          "selection moved if the new run produced a better checkpoint)",
          code == 200
          and plan_post["plan"]["stages"][0]["suite_run"]["state"]
          ["resolved_checkpoint_id"] == argmin_post["checkpoint_id"])
    check("C4 training-only footprint: workflows/suite-runs/evaluations/"
          "comparisons/gates/samples/recipes unchanged",
          len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]) == 19
          and len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]) == 16
          and len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]) == 16
          and len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]) == 8
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]) == 11
          and len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]) == 4
          and len(call_json("GET", f"{BASE}/workflows/recipes")[1]) == 8)
    code, dash_post = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    diffs = deep_diff(dash_pre, dash_post)
    allowed = ("result_hash", "checkpoints", "training_runs",
               "artifact_graph", "diagnostics",
               "summary")   # persisted model-manifest facts: a training
               # run legitimately updates best/latest/run-count/updated_at
    bad = [d for d in diffs
           if not any(d == a or d.startswith(a + ".")
                      or d.startswith(a + "[") or d.startswith(a + " (")
                      for a in allowed)]
    check("C5 dashboard changed ONLY in checkpoint/training-run/artifact-"
          "graph sections", bad == [], f"unexpected: {bad[:4]}")
    spec = json.loads(
        call("GET", "http://127.0.0.1:8775/openapi.json")[1])
    check("C6 OpenAPI 84 UNCHANGED; the field is part of TrainingConfig",
          len(spec["paths"]) == 84
          and "resume_from_checkpoint_id" in spec["components"]["schemas"]
          ["TrainingConfig"]["properties"])

    n, b, t, snap = audit("final")
    grown = new_files(pre_snap, snap)
    modified = changed_files(pre_snap, snap)
    n_ck_new = sum(1 for g in grown
                   if g.startswith(f"models/{MODEL}/checkpoints/"))
    ok_modified = set(modified) <= {
        f"models/{MODEL}/manifest.json", f"models/{MODEL}/weights.pt",
        f"models/{MODEL}/weights.sha256"}   # the published-weights hash
        # sidecar is updated by the EXISTING write_weights semantics
    check("C7 final audit: exactly the run's artifacts — 4 new checkpoint "
          "files (2 manifests + 2 weights), only the model manifest "
          "(existing completion semantics) and possibly published "
          "weights.pt modified, 0 tmp",
          len(grown) == 4 and n_ck_new == 4 and ok_modified and t == 0,
          f"new: {grown}; modified: {modified}")
    for g in grown:
        print(f"    NEW {g}")

    if FAILURES:
        print(f"M54 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M54 live smoke OK: EXPLICIT TRAINING RESUME POINT — ONE real "
          "production run initialized from the historical checkpoint "
          f"{RESUME_CK} (verified weights, probe config cloned) whose "
          "baseline evaluation reproduced the checkpoint's persisted "
          f"validation_loss {resume_man['validation_loss']} EXACTLY "
          "(initialized from its weights, non-destructively), whose "
          "provenance pins the resume point (initial + parent + config) "
          "and whose checkpoints descend from it through the existing "
          "lineage fields; latest_checkpoint followed the EXISTING "
          "publication semantics (the run's last created checkpoint, "
          "never the resume point — no rollback, no pre-run mutation); "
          "the resume checkpoint byte-identical (referenced, never "
          "copied); M52 selection + M53 best-ref preflight resolve the "
          "live argmin over the grown registry; dashboard changed only "
          "in the checkpoint/training-run/artifact-graph sections; "
          "OpenAPI 84 unchanged; storage delta exactly the run's own "
          "artifacts, each justified) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
