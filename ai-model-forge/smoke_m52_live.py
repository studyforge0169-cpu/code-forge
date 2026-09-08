"""M52 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M52 adds ONE read-only MODEL-SCOPED
selection primitive,
GET /models/{id}/checkpoints/best,
answering "which of this model's checkpoints has the MINIMUM
PERSISTED validation_loss?" — a deterministic computed view over the
authoritative M3 checkpoint listing: the persisted manifest is the
source (validation loss read VERBATIM, never recomputed, never derived
from perplexity, decisions, evaluations, ids or timestamps), ties on
the exact minimum resolve by the listing's canonical (step, created_at)
ASCENDING order (first among equals, DISCLOSED via the tied flag),
non-finite persisted values are never candidates, and "best" means
exactly this criterion — never a claim of overall model quality. The
smoke must DISCOVER the expected selection live from the persisted
checkpoint manifests (never hardcode the value), prove selection
correctness + metadata integrity + determinism, the established
empty/unknown 404 semantics, the route-order guarantee (best is the
SELECTION route, never the {checkpoint_id} detail), representative
M3-M51 regressions (incl. the M51 recipe plan preflight), and ZERO
production storage growth.

Production facts (re-derived live; the authoritative source is the
persisted manifests + the live listings): model 4a0a871886ef owns 3
checkpoints (all decision accept, persisted validation_loss values
discovered live from the manifests — the argmin is expected to be
0511de4c7372 @ 6.210553, the checkpoint the production recipe
m12-live-suite pins, but the smoke computes it independently); 16
workflows (12 completed / 3 failed / 1 stopped), 13 suite runs, 16
evaluations, 8 comparisons, 11 gate decisions, 4 samples, 7 recipes;
the zero-history model b5bc905326b6 owns NO checkpoints (never
trained) — the legitimate empty case. M52 is READ-ONLY: expected
production storage delta is ZERO (102 files / 4,013,875 B / 0 .tmp,
m52_pre.sha256 captured BEFORE this smoke; every SHA must stay
byte-identical; no selection pointer, no cache, no index).

LIVE A (baseline + independent expectation): exact 102/4,013,875/0
audit + per-file SHA256 inventory to /tmp/m52-smoke-baseline-
inventory.json + read ALL production checkpoint manifests DIRECTLY
from disk, independently compute the argmin under the criterion (min
persisted validation_loss; canonical (step, created_at) ASC
tie-break) + record every candidate id/loss + full pre-state of the
M3-M51 surfaces + dashboard + OpenAPI 84 pre-state.

LIVE B (selection correctness): GET best -> 200; returned checkpoint
id == the independently computed argmin; returned validation_loss ==
the persisted manifest value (exact float); the FULL record matches
the authoritative manifest VERBATIM; criterion explicit
("minimum_persisted_validation_loss"); candidate_count == the number
of persisted candidates; tied flag == the independently computed tie
fact; verbatim parity with the M3 listing entry and the M3 detail
getter payload.

LIVE C (determinism): x3 byte-identical response bodies.

LIVE D (established edge semantics): the zero-history model
b5bc905326b6 (valid, never trained, zero checkpoints) -> 404 with the
"no selectable checkpoints" detail (nothing manufactured); unknown
model -> 404; the M3 listing of the empty model is 200 [].

LIVE E (route collision): /checkpoints/best resolves to the SELECTION
route (200 with the criterion key — the generic {checkpoint_id}
detail route would 404 an unknown id "best"); a REAL checkpoint id
still resolves through the detail route (verbatim record, no
criterion key); OpenAPI display order listing < by-run < best <
generic detail.

LIVE F (regression + OpenAPI + final audit): checkpoint listing (3,
canonical order), detail, by-run (M46: run distribution 2/1),
evaluations 16, comparisons 8, gate decisions 11, workflows 16
(12/3/1), suite runs 13, recipes 7, the M51 recipe plan preflight
(m12-live-suite resolves with the same plan as its persisted runs),
dashboard result_hash UNCHANGED (byte-identical computed aggregate),
OpenAPI 84 with the new path exactly once / GET-only / $ref
CheckpointSelection; final audit 102/4,013,875/0 with EVERY SHA
byte-identical (zero storage growth, no cache/index/pointer files).

Start the server first:
    cd ai-model-forge && FORGE_ROOT=/home/user/ai-model-forge-data \
        python -m uvicorn app.api:app --host 127.0.0.1 --port 8773 \
        --log-level warning
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8773/api/v1"
MODEL = "4a0a871886ef"
EMPTY_MODEL = "b5bc905326b6"           # exists, never trained, NO checkpoints
CKPT_DIR = Path("/home/user/ai-model-forge-data/models") / MODEL / "checkpoints"
RECIPE = "m12-live-suite"              # pins the expected best checkpoint
RUN_A = "291a16d755fc"                 # M46: -> 2 checkpoints
RUN_B = "85438934f86a"                 # M46: -> 1 checkpoint
NEW_PATH = "/api/v1/models/{model_id}/checkpoints/best"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m52-smoke-baseline-inventory.json")

PRE_FILES, PRE_BYTES, PRE_TMP = 102, 4_013_875, 0

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


def main() -> int:
    print("== LIVE A: baseline audit + independent expectation ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 102/4,013,875/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    INVENTORY.write_text(json.dumps(pre_snap, indent=1, sort_keys=True))

    # independent expectation: read EVERY checkpoint manifest directly
    # from disk and compute the argmin under the criterion ourselves
    manifests = {}
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            manifests[d.name] = json.loads(
                (d / "manifest.json").read_text())
    check("A2 three persisted checkpoint manifests read directly",
          len(manifests) == 3, str(sorted(manifests)))
    for cid, m in sorted(manifests.items()):
        print(f"    candidate {cid}: validation_loss "
              f"{m['validation_loss']} (step {m['step']}, decision "
              f"{m['decision']})")
    ordered = sorted(manifests.values(),
                     key=lambda m: (m["step"], m["created_at"]))
    expected = min(ordered, key=lambda m: m["validation_loss"])
    min_loss = expected["validation_loss"]
    n_tied = sum(1 for m in manifests.values()
                 if m["validation_loss"] == min_loss)
    print(f"    independent argmin: {expected['checkpoint_id']} @ "
          f"{min_loss} (tied candidates: {n_tied})")
    check("A3 independent argmin computed from persisted manifests only",
          expected["checkpoint_id"] in manifests)

    listing = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
    check("A4 M3 listing pre-state: 3 checkpoints in canonical order",
          len(listing) == 3
          and [(c["step"], c["created_at"]) for c in listing]
          == sorted((c["step"], c["created_at"]) for c in listing))
    code, dash_pre = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A5 dashboard pre-state captured", code == 200)
    wf_pre = call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]
    ev_pre = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]
    cp_pre = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]
    gates_pre = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]
    check("A6 pre-state listings 16/16/8/11",
          (len(wf_pre), len(ev_pre), len(cp_pre), len(gates_pre))
          == (16, 16, 8, 11),
          f"{len(wf_pre)}/{len(ev_pre)}/{len(cp_pre)}/{len(gates_pre)}")
    recipes = call_json("GET", f"{BASE}/workflows/recipes")[1]
    check("A7 recipe registry 7", len(recipes) == 7)

    print("== LIVE B: selection correctness ==")
    code, raw = call("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("B1 GET best -> 200", code == 200, str(code))
    sel = json.loads(raw)
    ck = sel["checkpoint"]
    check("B2 returned checkpoint == independently computed argmin",
          ck["checkpoint_id"] == expected["checkpoint_id"],
          f"{ck['checkpoint_id']} vs {expected['checkpoint_id']}")
    check("B3 returned validation_loss == persisted manifest value",
          ck["validation_loss"] == min_loss,
          f"{ck['validation_loss']}")
    check("B4 full record matches the authoritative manifest verbatim",
          ck == manifests[ck["checkpoint_id"]]
          and ck == [c for c in listing
                     if c["checkpoint_id"] == ck["checkpoint_id"]][0])
    one = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/"
                     f"{ck['checkpoint_id']}")[1]
    check("B5 verbatim parity with the M3 detail getter", one == ck)
    check("B6 criterion explicit + candidates counted + tie disclosed",
          sel["model_id"] == MODEL
          and sel["criterion"] == "minimum_persisted_validation_loss"
          and sel["candidate_count"] == 3
          and sel["tied"] is (n_tied > 1))

    print("== LIVE C: determinism ==")
    raws = [call("GET", f"{BASE}/models/{MODEL}/checkpoints/best")[1]
            for _ in range(3)]
    check("C1 x3 byte-identical response bodies",
          raws[0] == raws[1] == raws[2] == raw)

    print("== LIVE D: established edge semantics ==")
    code, empty = call_json("GET",
                            f"{BASE}/models/{EMPTY_MODEL}/checkpoints/best")
    check("D1 valid zero-checkpoint model -> established 404, nothing "
          "manufactured",
          code == 404 and "no selectable checkpoints" in empty["detail"],
          f"{code} {empty.get('detail', '')[:60]}")
    check("D2 the empty model's M3 listing is 200 []",
          call_json("GET", f"{BASE}/models/{EMPTY_MODEL}/checkpoints")
          == (200, []))
    code, _ = call_json("GET", f"{BASE}/models/ghost-m52/checkpoints/best")
    check("D3 unknown model -> 404", code == 404, str(code))

    print("== LIVE E: route collision ==")
    code, probe = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/best")
    check("E1 'best' resolves to the SELECTION route (criterion key "
          "present — the generic detail route would 404 an unknown id)",
          code == 200 and "criterion" in probe)
    real_id = expected["checkpoint_id"]
    code, detail = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/"
                             f"{real_id}")
    check("E2 a real checkpoint id still resolves through the detail "
          "route (no criterion key, verbatim record)",
          code == 200 and "criterion" not in detail
          and detail == manifests[real_id])

    print("== LIVE F: regression + OpenAPI + final audit ==")
    listing2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")[1]
    check("F1 M3 listing unchanged (byte-identical)", listing2 == listing)
    byrun = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/by-run/"
                      f"{RUN_A}")[1]
    check("F2 M46 by-run unchanged (run distribution 2/1)",
          len(byrun) == 2
          and len(call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/"
                              f"by-run/{RUN_B}")[1]) == 1)
    check("F3 evaluations/comparisons/gates/workflows/suite-runs/"
          "recipes unchanged",
          len(call_json("GET", f"{BASE}/models/{MODEL}/evaluations")[1]) == 16
          and len(call_json("GET", f"{BASE}/models/{MODEL}/comparisons")[1]) == 8
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]) == 11
          and len(call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]) == 16
          and len(call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]) == 13
          and len(call_json("GET", f"{BASE}/workflows/recipes")[1]) == 7)
    code, plan = call_json(
        "GET", f"{BASE}/models/{MODEL}/workflows/recipes/{RECIPE}/plan")
    prior = json.loads(
        (ROOT / "models" / MODEL / "workflows" / "workflow-884d2dc725a5"
         / "manifest.json").read_text())
    check("F4 M51 recipe plan preflight unchanged (same plan as the "
          "persisted run)",
          code == 200 and plan["recipe_id"] == RECIPE
          and plan["plan"]["stages"] == prior["plan"]["stages"]
          and plan["plan"]["stages"][0]["suite_run"]["state"]
          ["checkpoint_id"] == expected["checkpoint_id"])
    code, dash_post = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("F5 dashboard result_hash UNCHANGED (read-only milestone; the "
          "dashboard aggregates persisted facts, a computed selection "
          "view is not one)",
          dash_post["result_hash"] == dash_pre["result_hash"]
          and dash_post == dash_pre)

    spec = json.loads(call("GET", "http://127.0.0.1:8773/openapi.json")[1])
    keys = list(spec["paths"])
    check("F6 OpenAPI 84 with the M52 path exactly once",
          len(keys) == 84 and keys.count(NEW_PATH) == 1)
    item = spec["paths"][NEW_PATH]
    check("F7 GET-only, training tag, model_id param, typed $ref response",
          list(item) == ["get"] and item["get"]["tags"] == ["training"]
          and [p["name"] for p in item["get"]["parameters"]] == ["model_id"]
          and item["get"]["responses"]["200"]["content"]["application/json"]
          ["schema"] == {"$ref": "#/components/schemas/"
                                  "CheckpointSelection"})
    check("F8 display order listing < by-run < best < generic detail",
          keys.index("/api/v1/models/{model_id}/checkpoints")
          < keys.index("/api/v1/models/{model_id}/checkpoints/by-run/"
                       "{run_id}")
          < keys.index(NEW_PATH)
          < keys.index("/api/v1/models/{model_id}/checkpoints/"
                       "{checkpoint_id}"))

    n, b, t, snap = audit("final")
    check("F9 ZERO storage growth: 102/4,013,875/0, every SHA "
          "byte-identical, no cache/index/pointer files",
          (n, b, t) == (PRE_FILES, PRE_BYTES, PRE_TMP) and snap == pre_snap)

    if FAILURES:
        print(f"M52 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M52 live smoke OK: deterministic best-checkpoint SELECTION "
          "under the persisted validation-loss criterion — the argmin "
          "DISCOVERED live from the persisted manifests "
          f"({expected['checkpoint_id']} @ {min_loss}, the checkpoint the "
          "production recipe pins), returned record verbatim-equal to the "
          "authoritative manifest/listing/detail, criterion explicit in "
          "the payload, tie fact disclosed, x3 byte-identical, valid "
          "zero-checkpoint model -> established 404 (nothing "
          "manufactured), unknown model -> 404, 'best' provably resolved "
          "by the SELECTION route (never captured as a checkpoint id), "
          "M3/M46/M4-M51 surfaces + M51 preflight + dashboard hash "
          "unchanged, OpenAPI 84, ZERO production storage growth) "
          "verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
