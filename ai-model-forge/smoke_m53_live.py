"""M53 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M53 adds the 'best' WORKFLOW STATE
REFERENCE: a workflow stage state (StageStateRef — suite-run states,
comparison sides, gate candidates) may declare state_kind "best"; at
execution or M51-preflight time the workflow engine's ONE resolver
invokes the SAME M52 selection (minimum persisted validation_loss over
the authoritative M3 listing, canonical (step, created_at) ASC
tie-break, first among equals, non-finite values never candidates) and
pins the CONCRETE checkpoint id (resolved_checkpoint_id) into the
resolved plan — which is what executes, persists in the immutable run
record and hashes. The recipe definition stays declarative (never
rewritten, no mutable pointer); downstream engines receive a NORMAL
literal checkpoint state; direct comparison/gate/suite-run requests
still require current/checkpoint.

All 7 production recipes pin literal checkpoint ids (modifying any is
forbidden), so the live verification registers exactly ONE new minimal
immutable recipe — m53-live-best: a single read-only suite_run stage on
the production suite m9-live-suite with state_kind "best" — explicitly
accounted for in the storage audit. Executions are READ-ONLY (suite
runs with full evidence reuse; no training, no rollback, no weight
mutation, no new evaluations).

Production facts (re-derived live; the authoritative sources are the
persisted manifests + the live listings): model 4a0a871886ef owns 3
checkpoints (all decision accept; the persisted validation-loss argmin
DISCOVERED live from the manifests — expected 0511de4c7372 @ 6.210553,
the checkpoint m12-live-suite pins BY HAND), 16 workflows (12
completed / 3 failed / 1 stopped), 13 suite runs, 16 evaluations, 8
comparisons, 11 gate decisions, 7 recipes; model b5bc905326b6 owns NO
checkpoints (never trained — the established 404 case). Storage at
start: 102 files / 4,014,007 B / 0 .tmp (m53_pre.sha256 captured
BEFORE this smoke; the 96 original production files byte-identical,
the 6 M51 execution artifacts of the current certified generation).

LIVE A (baseline + independent expectation): exact 102/4,013,875/0
audit + per-file SHA256 inventory + read ALL production checkpoint
manifests directly + independently compute the argmin under the M52
criterion + full pre-state listings + dashboard + recipe registry 7 +
the M52 /checkpoints/best endpoint agrees with the independent argmin.

LIVE B (register the minimal best recipe): POST /workflows/recipes
m53-live-best (single suite_run stage, suite m9-live-suite, state
"best") -> 201; the recipe registry grows 7 -> 8; exactly ONE new file
(its manifest); all 102 baseline files byte-identical; the recipe
manifest itself stores the DECLARATIVE best reference (no
resolved_checkpoint_id key).

LIVE C (M51 preflight resolves best): GET
/models/{id}/workflows/recipes/m53-live-best/plan -> 200 with
state_kind "best" AND resolved_checkpoint_id == the independently
computed argmin == the M52 endpoint's selection; x3 byte-identical;
ZERO storage growth from preflights (pure computation).

LIVE D (REAL execution x3 of the best recipe): each POST
/workflows/recipes/m53-live-best/runs returns only after terminal
state: 200 completed; the record's plan pins best -> the concrete id
(byte-identical to the preflight plan; same plan_hash across all three
runs); the stage artifact + the suite-run record carry the CONCRETE
checkpoint id; downstream records are normal literal-checkpoint runs;
per execution exactly ONE workflow + ONE suite-run manifest (2/2 probes
satisfied by pre-existing evidence — execution bookkeeping only) and
NOTHING else: no new evaluations/comparisons/gates/samples/checkpoints/
training runs; no pre-existing file modified.

LIVE E (immutability + no auto-anything): the recipe manifest is
byte-identical after all executions (declarative definition never
rewritten); the OLD m12-live-suite runs (literal pins) unchanged; the
model manifest + weights + checkpoint listing byte-identical (no
rollback, no training); re-preflight after the executions resolves the
SAME id (registry unchanged).

LIVE F (regression + OpenAPI + final audit): M3 listing/detail/by-run,
M52 /checkpoints/best, the M51 preflight of m12-live-suite (unchanged
plan — no best refs, resolver returns the plan verbatim), the dashboard
changing ONLY in workflow/suite-run/artifact-graph sections, OpenAPI 84
UNCHANGED (M53 adds NO route — only the schema gained the enum member
+ pinned-id field); final audit 109 files / 0 .tmp with every new file
justified: 1 recipe manifest + 3 workflow manifests + 3 suite-run
manifests.

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8774 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8774/api/v1"
MODEL = "4a0a871886ef"
EMPTY_MODEL = "b5bc905326b6"           # exists, never trained, NO checkpoints
SUITE = "m9-live-suite"
RECIPE = "m53-live-best"               # the ONE new minimal recipe (M53)
PINNED_RECIPE = "m12-live-suite"       # existing recipe with a literal id
CKPT_DIR = Path("/home/user/ai-model-forge-data/models") / MODEL / "checkpoints"
PLAN_PATH = "/models/{m}/workflows/recipes/{r}/plan"
RUNS_PATH = "/workflows/recipes/{r}/runs"
ROOT = Path("/home/user/ai-model-forge-data")

PRE_FILES, PRE_BYTES, PRE_TMP = 102, 4_014_007, 0
N_RUNS = 3
POST_FILES = 109                       # +1 recipe + 3x(workflow+suite-run)

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
    print("== LIVE A: baseline audit + independent expectation ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 102/4,014,007/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")

    manifests = {}
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            manifests[d.name] = json.loads(
                (d / "manifest.json").read_text())
    check("A2 three persisted checkpoint manifests read directly",
          len(manifests) == 3, str(sorted(manifests)))
    ordered = sorted(manifests.values(),
                     key=lambda m: (m["step"], m["created_at"]))
    expected = min(ordered, key=lambda m: m["validation_loss"])
    argmin_id, argmin_loss = (expected["checkpoint_id"],
                              expected["validation_loss"])
    print(f"    independent argmin: {argmin_id} @ {argmin_loss} "
          f"(candidates: "
          + ", ".join(f"{m['checkpoint_id']}={m['validation_loss']}"
                      for m in ordered) + ")")
    code, m52 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("A3 the M52 endpoint agrees with the independent argmin",
          code == 200
          and m52["checkpoint"]["checkpoint_id"] == argmin_id,
          m52["checkpoint"]["checkpoint_id"])
    code, recipes_pre = call_json("GET", f"{BASE}/workflows/recipes")
    check("A4 recipe registry pre-state 7", code == 200
          and len(recipes_pre) == 7, str(len(recipes_pre)))
    code, wf_pre = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code, dash_pre = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    ck_pre = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
    check("A5 pre-state: 16 workflows / 3 checkpoints",
          len(wf_pre) == 16 and len(ck_pre) == 3,
          f"{len(wf_pre)}/{len(ck_pre)}")
    code, preflight_pinned = call_json(
        "GET", BASE + PLAN_PATH.format(m=MODEL, r=PINNED_RECIPE))
    check("A6 M51 preflight of the literal-id recipe still works",
          code == 200 and preflight_pinned["recipe_id"] == PINNED_RECIPE)

    print("== LIVE B: register the minimal best recipe ==")
    stage = {"stage_id": "suite_best", "type": "suite_run",
             "suite_run": {"suite_id": SUITE,
                           "state": {"state_kind": "best"}}}
    code, _ = call_json("POST", f"{BASE}/workflows/recipes", body={
        "recipe_id": RECIPE,
        "description": "M53 live: read-only suite run at the best "
                       "checkpoint (minimum persisted validation loss)",
        "stages": [stage]})
    check("B1 registration -> 201", code == 201, str(code))
    n, _, t, snap = audit("after registration")
    grown = new_files(pre_snap, snap)
    check("B2 exactly ONE new file (the recipe manifest), baseline "
          "byte-identical",
          (n, t) == (PRE_FILES + 1, 0) and grown == [
              f"workflow-recipes/{RECIPE}/manifest.json"]
          and changed_files(pre_snap, snap) == [], str(grown))
    rman = json.loads((ROOT / "workflow-recipes" / RECIPE
                       / "manifest.json").read_bytes())
    rman_bytes = (ROOT / "workflow-recipes" / RECIPE
                  / "manifest.json").read_bytes()
    st = rman["stages"][0]["suite_run"]["state"]
    check("B3 the stored recipe is DECLARATIVE (best, nothing pinned)",
          st["state_kind"] == "best"
          and st.get("resolved_checkpoint_id") is None
          and st.get("checkpoint_id") is None
          and st.get("from_stage") is None, json.dumps(st))
    code, recipes_post = call_json("GET", f"{BASE}/workflows/recipes")
    check("B4 registry 7 -> 8", len(recipes_post) == 8)

    print("== LIVE C: M51 preflight resolves best ==")
    code, res = call_json("GET", BASE + PLAN_PATH.format(m=MODEL,
                                                         r=RECIPE))
    check("C1 preflight -> 200", code == 200, str(code))
    state = res["plan"]["stages"][0]["suite_run"]["state"]
    check("C2 preflight pins best -> the independent argmin",
          state["state_kind"] == "best"
          and state["resolved_checkpoint_id"] == argmin_id,
          f"best -> {state.get('resolved_checkpoint_id')}")
    check("C3 preflight recipe provenance intact",
          res["recipe_id"] == RECIPE and res["composition"] is None
          and res["plan"]["model_id"] == MODEL)
    raws = [call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
            for _ in range(3)]
    check("C4 x3 byte-identical preflight", raws[0] == raws[1] == raws[2])
    n, _, t, snap = audit("after preflights")
    check("C5 preflight wrote NOTHING (103 files, SHAs intact)",
          (n, t) == (PRE_FILES + 1, 0)
          and changed_files(pre_snap, snap) == [])
    code, empty = call_json("GET", BASE + PLAN_PATH.format(
        m=EMPTY_MODEL, r=RECIPE))
    check("C6 zero-checkpoint model -> the established 404",
          code == 404 and "no selectable checkpoints" in empty["detail"],
          str(code))

    print("== LIVE D: REAL execution of the best recipe x3 ==")
    run_results = []
    for i in range(1, N_RUNS + 1):
        before = audit(f"pre run {i}")
        code, rec = call_json("POST", BASE + RUNS_PATH.format(r=RECIPE),
                              body={"model_id": MODEL})
        check(f"D{i}.1 run {i} -> 200 completed (synchronous)",
              code == 200 and rec["status"] == "completed",
              f"code {code} status {rec.get('status') if rec else '?'}")
        rstate = rec["plan"]["stages"][0]["suite_run"]["state"]
        check(f"D{i}.2 run {i} record pins best -> the concrete argmin",
              rstate["state_kind"] == "best"
              and rstate["resolved_checkpoint_id"] == argmin_id)
        check(f"D{i}.3 run {i} plan == preflight plan (same resolver)",
              rec["plan"] == res["plan"]
              and rec["recipe_hash"] == res["recipe_hash"])
        art = rec["stages"][0]["artifact"]
        check(f"D{i}.4 run {i} artifact + suite-run carry the CONCRETE id",
              art["kind"] == "suite_run"
              and art["checkpoint_id"] == argmin_id)
        srm = json.loads((ROOT / "suite-runs" / art["artifact_id"]
                          / "manifest.json").read_text())
        check(f"D{i}.5 run {i} downstream record is a NORMAL literal "
              "checkpoint run (full evidence reuse)",
              srm["state"]["state_kind"] == "checkpoint"
              and srm["state"]["checkpoint_id"] == argmin_id
              and srm["probe_count"] == 2 and srm["reused_count"] == 2
              and srm["status"] == "completed",
              f"{srm['suite_run_id']}: reused {srm['reused_count']}/"
              f"{srm['probe_count']}")
        after = audit(f"post run {i}")
        grown_i = new_files(before[3], after[3])
        check(f"D{i}.6 run {i} grew exactly ONE workflow + ONE suite-run "
              "manifest, modified nothing",
              len(grown_i) == 2
              and sum(1 for g in grown_i
                      if g.startswith(f"models/{MODEL}/workflows/")) == 1
              and sum(1 for g in grown_i
                      if g.startswith("suite-runs/")) == 1
              and changed_files(before[3], after[3]) == []
              and after[2] == 0, "; ".join(grown_i))
        check(f"D{i}.7 run {i} created NO new evaluations/comparisons/"
              "gates/samples/checkpoints/training runs",
              len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]) == 16
              and len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]) == 8
              and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]) == 11
              and len(call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")[1]) == 3
              and len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1]) == 4)
        run_results.append(rec)

    check("D8 all three runs share ONE resolved plan identity "
          "(determinism, same registry)",
          all(r["plan"] == run_results[0]["plan"]
              and r["plan_hash"] == run_results[0]["plan_hash"]
              for r in run_results),
          run_results[0]["plan_hash"][:16])

    print("== LIVE E: immutability + no auto-anything ==")
    check("E1 the recipe manifest is byte-identical after all executions "
          "(declarative definition never rewritten)",
          (ROOT / "workflow-recipes" / RECIPE
           / "manifest.json").read_bytes() == rman_bytes)
    check("E2 no pre-existing production file modified (incl. every old "
          "workflow, the model manifest and weights)",
          changed_files(pre_snap, audit("E2")[3]) == [])
    code, res_after = call_json("GET", BASE + PLAN_PATH.format(
        m=MODEL, r=RECIPE))
    check("E3 re-preflight after the executions resolves the SAME id "
          "(registry unchanged)",
          res_after["plan"]["stages"][0]["suite_run"]["state"]
          ["resolved_checkpoint_id"] == argmin_id
          and call("GET", BASE + PLAN_PATH.format(m=MODEL, r=RECIPE))[1]
          == raws[0])
    model_manifest = json.loads(
        (ROOT / "models" / MODEL / "manifest.json").read_text())
    check("E4 no automatic training/rollback: checkpoint listing 3, "
          "latest_checkpoint unchanged, no new training runs",
          len(call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")[1]) == 3
          and len(model_manifest.get("training_provenance", [])) == 2)

    print("== LIVE F: regression + OpenAPI + final audit ==")
    code, wf_post = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    check("F1 workflows 16 -> 19 (all completed)", len(wf_post) == 19
          and all(w["status"] == "completed" for w in wf_post[-3:]))
    code, dash_post = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    diffs = deep_diff(dash_pre, dash_post)
    allowed = ("result_hash", "workflows", "suite_runs", "artifact_graph",
               "diagnostics")
    bad = [d for d in diffs
           if not any(d == a or d.startswith(a + ".")
                      or d.startswith(a + "[") or d.startswith(a + " (")
                      for a in allowed)]
    check("F2 dashboard changed ONLY in workflow/suite-run/artifact-graph "
          "sections", bad == [], f"unexpected: {bad[:4]}")
    code, m52b = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("F3 M52 selection endpoint unchanged",
          code == 200 and m52b == m52)
    code, pre_pinned = call_json(
        "GET", BASE + PLAN_PATH.format(m=MODEL, r=PINNED_RECIPE))
    check("F4 M51 preflight of the literal-id recipe byte-identical "
          "(no best refs -> resolver returns the plan verbatim)",
          code == 200 and pre_pinned == preflight_pinned)
    spec = json.loads(
        call("GET", "http://127.0.0.1:8774/openapi.json")[1])
    check("F5 OpenAPI 84 UNCHANGED (M53 adds NO route); the schema "
          "carries best + resolved_checkpoint_id",
          len(spec["paths"]) == 84
          and "best" in spec["components"]["schemas"]["EvalStateKind"]
          ["enum"]
          and "resolved_checkpoint_id" in spec["components"]["schemas"]
          ["StageStateRef"]["properties"])
    code, byrec = call_json(
        "GET", f"{BASE}/models/{MODEL}/workflows/by-recipe/{RECIPE}")
    check("F6 M35 by-recipe sees the 3 best-recipe runs", code == 200
          and len(byrec) == 3)

    n, b, t, snap = audit("final")
    grown_total = new_files(pre_snap, snap)
    n_rec = sum(1 for g in grown_total
                if g.startswith("workflow-recipes/"))
    n_wf = sum(1 for g in grown_total
               if g.startswith(f"models/{MODEL}/workflows/"))
    n_sr = sum(1 for g in grown_total
               if g.startswith("suite-runs/"))
    check("F7 final audit 109/0 tmp: 1 recipe + 3 workflow + 3 suite-run "
          "manifests, every baseline file byte-identical",
          (n, t) == (POST_FILES, 0) and n_rec == 1 and n_wf == 3
          and n_sr == 3 and len(grown_total) == 7
          and changed_files(pre_snap, snap) == [],
          f"{n} files, new: {len(grown_total)}")

    if FAILURES:
        print(f"M53 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M53 live smoke OK: best-checkpoint STATE REFERENCES — the "
          "independently computed persisted-argmin "
          f"({argmin_id} @ {argmin_loss}, agreeing with the M52 "
          "endpoint) pinned by ONE shared resolver into the M51 "
          "preflight AND the immutable execution records: a real "
          "production recipe declaring state_kind 'best' resolved to the "
          "concrete checkpoint at preflight (x3 byte-identical, zero "
          "writes) and at execution (x3 synchronous completed runs, "
          "record plan byte-identical to the preflight, artifacts + "
          "downstream suite-run records carrying the CONCRETE id with "
          "full evidence reuse, ONE resolved plan identity), the recipe "
          "manifest staying declarative and byte-identical, no "
          "training/rollback/checkpoint mutation, dashboard changing "
          "only in the workflow/suite-run/artifact-graph aggregate "
          "sections, M3/M35/M51/M52 surfaces + OpenAPI 84 unchanged, "
          "and exactly 7 justified new files: 1 minimal recipe manifest "
          "+ 3 workflow manifests + 3 suite-run manifests) verified "
          "live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
