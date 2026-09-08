"""M51 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M51 is the FIRST milestone since M12
whose smoke legitimately EXECUTES a real production recipe: it adds one
read-only MODEL-SCOPED resolution preflight,
GET /models/{id}/workflows/recipes/{recipe_id}/plan,
which answers "what EXACTLY would this registered recipe execute
against this model?" through the SAME resolution path as the existing
M12 synchronous execution surface (recipe lookup -> model validation ->
M14 deterministic expansion -> WorkflowPlan construction with the FULL
M7 validation) — WITHOUT executing anything (zero persistence), while
execution itself stays exactly where it was:
POST /workflows/recipes/{recipe_id}/runs through the sole
WorkflowEngine. The smoke must therefore prove BOTH halves live:
the read-only preflight (zero writes, byte-identical repeats, plan
BEFORE any execution) AND the unchanged execution semantics on a REAL
production recipe selected from persisted evidence.

Production facts (re-derived live; the authoritative sources are the
persisted manifests and the live listings): model 4a0a871886ef owns 13
workflow runs (completed 9 / failed 3 / stopped 1), 10 suite runs, 16
evaluations, 8 comparisons, 11 gate decisions, 4 samples, 3 checkpoints
(runs 291a16d755fc -> 2 / 85438934f86a -> 1); 7 registered recipes, of
which m12-live-suite (1 suite_run stage, suite m9-live-suite @
checkpoint 0511de4c7372, config_hash 0efdb646bfde..., composition
null) was ALREADY executed twice before (workflows 54e78451453b +
900624426305, each with a suite-run artifact of probe_count 2 /
reused_count 2 — FULL evidence reuse) — a genuinely safe target: no
training, no weight mutation, evidence-reuse semantics, and the two
prior runs are byte-level proof of exactly what one more execution
must produce. Model b5bc905326b6 owns NO history. M17 dashboard
result_hash (pre)
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp (m51_pre.sha256 captured BEFORE this
smoke).

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + per-file SHA256
inventory to /tmp/m51-smoke-baseline-inventory.json + full pre-state
of every M3-M50 surface (listings + dashboard JSON saved) + recipe
registry 7 + m51_pre.sha256 equivalence.

LIVE B (preflight BEFORE executing): GET plan of m12-live-suite under
4a0a871886ef -> 200 with the expanded model-bound plan (exactly ONE
suite_run stage 'suite1' on suite m9-live-suite @ checkpoint
0511de4c7372), recipe_hash == the PERSISTED manifest config_hash read
from disk, composition null (plain recipe); the resolved plan equals
the plan of the PRIOR persisted run (the preflight predicts the next
execution byte-for-byte); x3 byte-identical repeats; ZERO storage
growth (all 96 SHAs intact); the same recipe resolves under the
zero-history model b5bc905326b6 (model-bound echo, still zero
writes); unknown recipe / unknown model -> 404 with nothing persisted.

LIVE C (REAL execution x3 of the SAME production recipe): each
POST /workflows/recipes/m12-live-suite/runs returns only after the
terminal state (synchronous), 200 completed, with recipe provenance
(recipe_id + recipe_hash + composition) and a plan BYTE-IDENTICAL to
the LIVE B preflight; every new storage file is enumerated and
justified against the baseline inventory — per execution exactly ONE
new workflow manifest (models/{id}/workflows/workflow-*/manifest.json)
+ ONE new suite-run manifest (suite-runs/{id}/manifest.json, probe
count 2 / reused_count 2 / completed / checkpoint 0511de4c7372 — FULL
evidence reuse: the suite was already run at this exact state), and
NOTHING else: no new evaluations, comparisons, gates, samples or
checkpoints; every one of the 96 baseline files stays byte-identical
(immutability of all pre-existing production artifacts); final state
13 -> 16 workflows (completed 9 -> 12), 10 -> 13 suite runs, 16 / 8 /
11 / 4 / 3 unchanged.

LIVE D (regression + OpenAPI + final audit): the dashboard hash
changes LEGITIMATELY (it is a computed aggregate over the run history
M51 extended) — the smoke asserts the dashboard diff is EXACTLY the
workflow/suite-run/artifact-graph sections while evaluations,
comparisons, gate decisions, checkpoints, training runs and
sample-quality stay byte-identical; the M42 by-status grouping and the
M35 by-recipe grouping reflect the new runs; OpenAPI 83 with the new
path exactly once, GET-only; registries unchanged (7 recipes, 2
models); final audit 102 files / 0 .tmp with every new file justified;
no baseline manifest modified (m51_pre.sha256 semantics).

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8772 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8772/api/v1"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO history at all
CK_0511 = "0511de4c7372"
SUITE = "m9-live-suite"
RECIPE = "m12-live-suite"             # executed twice before (M12/M35 era)
RECIPE_HASH_12 = "0efdb646bfde"       # first 12 hex of persisted config_hash
PRIOR_RESULT_HASH = ("b2caea90b53818761a4e20533f671bd421207d1f3c1c6bfe8027"
                     "7408822d9ac6")
DASH_HASH = "f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838"
PLAN_PATH = "/models/{m}/workflows/recipes/{r}/plan"   # appended to BASE
RUNS_PATH = "/workflows/recipes/{r}/runs"              # appended to BASE
NEW_PATH = ("/api/v1/models/{model_id}/workflows/recipes/{recipe_id}/plan")
BY_RECIPE_PATH = ("/api/v1/models/{model_id}/workflows/by-recipe/{recipe_id}")
BY_STATUS_PATH = "/api/v1/models/{model_id}/workflows/by-status/{status}"
# fetch variants appended to BASE (which already ends in /api/v1)
BY_RECIPE_FETCH = "/models/{m}/workflows/by-recipe/{r}"
BY_STATUS_FETCH = "/models/{m}/workflows/by-status/{st}"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m51-smoke-baseline-inventory.json")

PRE_FILES, PRE_BYTES, PRE_TMP = 96, 4_002_745, 0
POST_FILES = 102                      # 96 + 3 x (1 workflow + 1 suite-run)
N_RUNS = 3

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


def plan_url(model: str = MODEL, recipe: str = RECIPE) -> str:
    return BASE + PLAN_PATH.format(m=model, r=recipe)


def run_url(recipe: str = RECIPE) -> str:
    return BASE + RUNS_PATH.format(r=recipe)


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
    print("== LIVE A: baseline audit + saved inventory ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 96/4,002,745/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    INVENTORY.write_text(json.dumps(pre_snap, indent=1, sort_keys=True))

    code, listing = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    wf_pre = listing
    code, sr_pre = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    code, ev_pre = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code, cp_pre = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    code, ck_pre = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code, sp_pre = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code, dash_pre = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code, recipes = call_json("GET", f"{BASE}/workflows/recipes")
    check("A2 pre-state listings 13/10/16/8/3/4",
          (len(wf_pre), len(sr_pre), len(ev_pre), len(cp_pre),
           len(ck_pre), len(sp_pre)) == (13, 10, 16, 8, 3, 4),
          f"{len(wf_pre)}/{len(sr_pre)}/{len(ev_pre)}/{len(cp_pre)}/"
          f"{len(ck_pre)}/{len(sp_pre)}")
    from collections import Counter
    wf_status_pre = Counter(w["status"] for w in wf_pre)
    check("A3 workflow status distribution completed 9 / failed 3 / stopped 1",
          wf_status_pre == {"completed": 9, "failed": 3, "stopped": 1},
          str(dict(wf_status_pre)))
    code, gates_pre = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")
    check("A4 gate decisions 11", code == 200 and len(gates_pre) == 11,
          str(len(gates_pre)))
    check("A5 recipe registry 7 (all production recipes)",
          code == 200 and len(recipes) == 7, str(len(recipes)))
    check("A6 dashboard pre-hash",
          dash_pre["result_hash"] == DASH_HASH,
          dash_pre["result_hash"][:16])
    ck_dist = Counter(c["run_id"] for c in ck_pre)
    check("A7 checkpoints 3 with run distribution 2/1",
          dict(ck_dist) == {"291a16d755fc": 2, "85438934f86a": 1},
          str(dict(ck_dist)))

    # the persisted recipe manifest is the authoritative definition
    rman_bytes = (ROOT / "workflow-recipes" / RECIPE
                  / "manifest.json").read_bytes()
    rman = json.loads(rman_bytes)
    check("A8 persisted recipe is the known safe suite_run target",
          len(rman["stages"]) == 1
          and rman["stages"][0]["type"] == "suite_run"
          and rman["stages"][0]["suite_run"]["suite_id"] == SUITE
          and rman["stages"][0]["suite_run"]["state"]["checkpoint_id"]
          == CK_0511,
          f"config_hash {rman['config_hash'][:12]}")
    prior_wf = json.loads(
        (ROOT / "models" / MODEL / "workflows" / "workflow-900624426305"
         / "manifest.json").read_text())
    check("A9 one prior run of this recipe exists (execution precedent)",
          prior_wf["recipe_id"] == RECIPE
          and prior_wf["status"] == "completed")

    print("== LIVE B: read-only preflight BEFORE any execution ==")
    code, res = call_json("GET", plan_url())
    check("B1 preflight 200", code == 200, str(code))
    stage = res["plan"]["stages"][0]
    check("B2 resolved plan is the single known suite_run stage",
          len(res["plan"]["stages"]) == 1
          and stage["stage_id"] == "suite1"
          and stage["type"] == "suite_run"
          and stage["suite_run"]["suite_id"] == SUITE
          and stage["suite_run"]["state"]["checkpoint_id"] == CK_0511
          and stage["suite_run"]["state"]["state_kind"] == "checkpoint")
    check("B3 resolution provenance echoes recipe + model",
          res["recipe_id"] == RECIPE
          and res["model_id"] == MODEL
          and res["plan"]["model_id"] == MODEL)
    check("B4 recipe_hash == persisted manifest config_hash",
          res["recipe_hash"] == rman["config_hash"],
          res["recipe_hash"][:12])
    check("B5 plain recipe: composition null", res["composition"] is None)
    def _sem(o):
        if isinstance(o, dict):
            return {k: _sem(v) for k, v in o.items() if v is not None}
        if isinstance(o, list):
            return [_sem(v) for v in o]
        return o

    check("B6 preflight plan predicts the prior execution exactly "
          "(the M12-era manifest simply predates the M14 'recipe' "
          "stage field: canonical equality ignoring null keys)",
          _sem(res["plan"]) == _sem(prior_wf["plan"]))

    b_raw = [call("GET", plan_url())[1] for _ in range(3)]
    check("B7 x3 byte-identical preflight repeats",
          b_raw[0] == b_raw[1] == b_raw[2])
    n, b, t, snap = audit("after preflight GETs")
    check("B8 preflight wrote NOTHING (96 files, SHAs intact)",
          (n, b, t) == (PRE_FILES, PRE_BYTES, PRE_TMP) and snap == pre_snap)

    code, res_other = call_json("GET", plan_url(model=OTHER_MODEL))
    check("B9 same recipe resolves under the zero-history model",
          code == 200
          and res_other["model_id"] == OTHER_MODEL
          and res_other["recipe_hash"] == res["recipe_hash"]
          and res_other["plan"]["stages"] == res["plan"]["stages"])
    code, _ = call_json("GET", plan_url(recipe="no-such-recipe"))
    check("B10 unknown recipe -> 404 (same taxonomy as a run)",
          code == 404, str(code))
    code, _ = call_json("GET", plan_url(model="no-such-model"))
    check("B11 unknown model -> 404", code == 404, str(code))
    n, _, t, snap = audit("after B9-B11")
    check("B12 error preflights wrote NOTHING",
          (n, t) == (PRE_FILES, PRE_TMP) and snap == pre_snap)

    print("== LIVE C: REAL execution of the production recipe x3 ==")
    run_results = []
    for i in range(1, N_RUNS + 1):
        before = audit(f"pre run {i}")
        code, rec = call_json("POST", run_url(),
                              body={"model_id": MODEL})
        check(f"C{i}.1 synchronous run {i} -> 200 completed",
              code == 200 and rec["status"] == "completed",
              f"code {code} status {rec.get('status') if rec else '?'}")
        check(f"C{i}.2 run {i} provenance == preflight provenance",
              rec["recipe_id"] == res["recipe_id"]
              and rec["recipe_hash"] == res["recipe_hash"]
              and rec["composition"] == res["composition"])
        check(f"C{i}.3 run {i} plan == preflight plan (byte-identical)",
              rec["plan"] == res["plan"])
        art = rec["stages"][0]["artifact"]
        check(f"C{i}.4 run {i} stage artifact is a fully-reused suite run",
              art is not None and art["kind"] == "suite_run"
              and art["checkpoint_id"] == CK_0511
              and art["result_hash"] == PRIOR_RESULT_HASH,
              f"artifact {art['artifact_id'] if art else None}")
        after = audit(f"post run {i}")
        grown = new_files(before[3], after[3])
        touched = changed_files(before[3], after[3])
        check(f"C{i}.5 run {i} grew exactly ONE workflow + ONE suite-run "
              "manifest", len(grown) == 2
              and sum(1 for g in grown
                      if g.startswith(f"models/{MODEL}/workflows/")) == 1
              and sum(1 for g in grown
                      if g.startswith("suite-runs/")) == 1,
              "; ".join(grown))
        check(f"C{i}.6 run {i} modified NO pre-existing file",
              touched == [] and after[2] == 0, "; ".join(touched))
        srm = json.loads((ROOT / [g for g in grown
                                  if g.startswith("suite-runs/")][0]
                          ).read_text())
        check(f"C{i}.7 run {i} new suite-run: FULL evidence reuse "
              "(2/2 probes, completed, same checkpoint)",
              srm["probe_count"] == 2 and srm["reused_count"] == 2
              and srm["status"] == "completed"
              and srm["state"]["checkpoint_id"] == CK_0511,
              f"{srm['suite_run_id']}: reused {srm['reused_count']}/"
              f"{srm['probe_count']}")
        check(f"C{i}.8 run {i} created NO new evaluations/comparisons/"
              "gates/samples/checkpoints",
              len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1])
              == len(ev_pre)
              and len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1])
              == len(cp_pre)
              and len(call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")[1])
              == len(ck_pre)
              and len(call_json("GET", f"{BASE}/models/{MODEL}/samples")[1])
              == len(sp_pre))
        run_results.append(rec)

    check("C9 all three runs share one plan + provenance (determinism)",
          all(r["plan"] == run_results[0]["plan"]
              and r["recipe_hash"] == run_results[0]["recipe_hash"]
              for r in run_results))

    print("== LIVE D: regression + OpenAPI + final audit ==")
    code, wf_post = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code, sr_post = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    wf_status_post = Counter(w["status"] for w in wf_post)
    check("D1 workflows 13 -> 16 (completed 9 -> 12, rest unchanged)",
          len(wf_post) == 16
          and wf_status_post == {"completed": 12, "failed": 3, "stopped": 1},
          str(dict(wf_status_post)))
    check("D2 suite runs 10 -> 13 (one per execution — execution "
          "bookkeeping, evidence itself reused)",
          len(sr_post) == 13)
    code, dash_post = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    diffs = deep_diff(dash_pre, dash_post)
    allowed = ("result_hash", "workflows", "suite_runs", "artifact_graph",
               "diagnostics")
    bad = [d for d in diffs
           if not any(d == a or d.startswith(a + ".") or d.startswith(a + "[")
                      or d.startswith(a + " (") for a in allowed)]
    check("D3 dashboard changed ONLY in workflow/suite-run/artifact-graph "
          "sections (computed aggregate over the extended history)",
          dash_post["result_hash"] != DASH_HASH and bad == [],
          f"{len(diffs)} changed paths, unexpected: {bad[:4]}")
    check("D4 dashboard measurements byte-identical (evaluations, "
          "comparisons, gates, checkpoints, training runs, sample quality)",
          dash_post["evaluations"] == dash_pre["evaluations"]
          and dash_post["comparisons"] == dash_pre["comparisons"]
          and dash_post["gate_decisions"] == dash_pre["gate_decisions"]
          and dash_post["checkpoints"] == dash_pre["checkpoints"]
          and dash_post["training_runs"] == dash_pre["training_runs"]
          and dash_post["sample_quality"] == dash_pre["sample_quality"])

    code, bystat = call_json(
        "GET", BASE + BY_STATUS_FETCH.format(m=MODEL, st="completed"))
    check("D5 M42 by-status reflects the new runs (12 completed)",
          code == 200 and len(bystat) == 12, str(len(bystat)))
    code, byrec = call_json(
        "GET", BASE + BY_RECIPE_FETCH.format(m=MODEL, r=RECIPE))
    check("D6 M35 by-recipe reflects the new runs (2 -> 5)",
          code == 200 and len(byrec) == 5, str(len(byrec)))

    spec = json.loads(
        call("GET", "http://127.0.0.1:8772/openapi.json")[1])
    keys = list(spec["paths"])
    check("D7 OpenAPI 83 with the M51 path exactly once",
          len(keys) == 83 and keys.count(NEW_PATH) == 1)
    item = spec["paths"][NEW_PATH]
    check("D8 new path is GET-only, workflow-tagged, typed response",
          list(item) == ["get"] and item["get"]["tags"] == ["workflows"]
          and item["get"]["responses"]["200"]["content"]
          ["application/json"]["schema"]
          == {"$ref": "#/components/schemas/WorkflowRecipeResolution"})

    n, b, t, snap = audit("final")
    grown_total = new_files(pre_snap, snap)
    touched_total = changed_files(pre_snap, snap)
    check("D9 final audit 102 files / 0 tmp, every baseline file "
          "byte-identical",
          (n, t) == (POST_FILES, 0) and touched_total == [],
          f"{n} files, modified: {touched_total[:4]}")
    n_wf = sum(1 for g in grown_total
               if g.startswith(f"models/{MODEL}/workflows/"))
    n_sr = sum(1 for g in grown_total
               if g.startswith("suite-runs/"))
    check("D10 all 6 new files justified: 3 workflow manifests + "
          "3 suite-run manifests (immutable execution records of the "
          "3 real recipe runs; nothing else)",
          len(grown_total) == 6 and n_wf == 3 and n_sr == 3,
          "; ".join(grown_total))
    code, recipes_post = call_json("GET", f"{BASE}/workflows/recipes")
    check("D11 registries unchanged (7 recipes, no new recipe registered)",
          len(recipes_post) == 7)
    rman2 = (ROOT / "workflow-recipes" / RECIPE
             / "manifest.json").read_bytes()
    check("D12 the executed recipe's own manifest is byte-identical "
          "(immutable definition)",
          rman2 == rman_bytes)

    if FAILURES:
        print(f"M51 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M51 live smoke OK: read-only recipe RESOLUTION preflight "
          "(GET .../recipes/{r}/plan) sharing the run() resolution path — "
          "expanded model-bound plan + recipe provenance + M14 composition "
          "BEFORE any execution, byte-identical x3, ZERO persistence (all "
          "96 baseline SHAs intact), 404s matching the run taxonomy — AND "
          "the unchanged synchronous execution surface proven live by 3 "
          "REAL executions of the production recipe m12-live-suite against "
          "the production model (each returning only after terminal "
          "state, plan byte-identical to the preflight, FULL evidence "
          "reuse 2/2 probes, no new evaluations/comparisons/gates/samples/"
          "checkpoints, exactly 6 justified new files: 3 workflow "
          "manifests + 3 suite-run manifests, every pre-existing "
          "production artifact byte-identical, dashboard changing ONLY in "
          "the workflow/suite-run/artifact-graph aggregate sections, "
          "OpenAPI 83) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
