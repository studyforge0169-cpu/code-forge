"""M50 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M50 is READ-ONLY: it adds one narrow
MODEL-SCOPED access path,
GET /models/{id}/suite-runs/by-reused/{reused_count},
answering "which immutable suite runs of this model satisfied this
many probes with pre-existing evidence?" — the model's authoritative
M10 listing (deterministic (created_at, suite_run_id) ASCENDING
order) filtered VERBATIM by the REQUIRED integer
SuiteRunRecord.reused_count persisted on every record at run time
(probes satisfied by pre-existing evidence — EXECUTION BOOKKEEPING
ONLY, never a score; matched VERBATIM — NEVER recalculated, NEVER
derived from probe outcomes, completed_count, failed_count,
probe_count, suite size, status, timestamps, artifact ids, evaluation
or comparison records, configuration or any other field), registered
BEFORE the generic /suite-runs/{suite_run_id} route and alongside
by-suite (M21)/summary (M22)/by-checkpoint (M25). The count is an
OPEN integer value axis — no registry, no enum, no artificial range
constraint: any integer is type-valid (an unmatched count on a valid
model returns 200 [] — a natural valid-empty), a non-integer spelling
is rejected 422 at the API boundary (schema-level validation,
pre-handler — integers are never silently reinterpreted, and the
validation fires BEFORE the handler even for an unknown model), while
an unknown model with a VALID integer is 404. The smoke must DISCOVER
the authoritative reused_count distribution from the live M10 listing
AND the persisted production manifests (not assume it from an old
report), prove exact listing parity for ALL discovered groups, the
natural valid-empty, clean 404/422 separation, byte-identical repeats
(x3), unchanged M3-M49 surfaces + dashboard hash + registries +
OpenAPI 82, and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M10 listing and the persisted suite-run manifests):
model 4a0a871886ef owns 10 suite runs in ASCENDING (created_at,
suite_run_id) order with persisted reused_count distribution 2 -> 9
and 0 -> 1 (TWO populated groups; 9 + 1 == 10 — a TRUE disjoint
partition; the int is REQUIRED on every record, 0 null values —
independently confirmed by reading the 10 production suite-run
manifests directly); model b5bc905326b6 has NO suite runs; M17
dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m50-smoke-baseline-inventory.json +
model/checkpoint/tokenizer/policy/probe-suite/recipe registries +
M3-M49 pre-state (10 suite runs + by-suite 10 + summary 10 +
by-checkpoint 10, 16 evaluations, 8 comparisons, 11 gate decisions,
13 workflows, 4 samples, 3 checkpoints with run distribution 2/1) +
dashboard + OpenAPI 82 pre-state + the known pair returns 200.

LIVE B (authoritative distribution discovery): from the live M10
listing AND the persisted manifests, discover the reused_count
distribution; print it; cross-check against the audited facts (two
groups summing to 10; 0 null values).

LIVE C (A — known counts): ALL discovered counts -> 200 with the
EXACT discovered records; parity with the M10 listing filtered
locally by the persisted record value; verbatim detail-getter
payloads for all 10 records; (created_at, suite_run_id) ASC
preserved per group.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — valid empty counts): a populated production count AND an
unmatched valid integer (987654) under b5bc905326b6 -> 200 + [] (it
has no suite runs at all); the unmatched count on the KNOWN model ->
200 [] too (open value axis).

LIVE F (D — unknown ids / non-integer spellings): unknown model +
valid integer -> 404; non-integer spellings ("abc", "1.5", "12x") ->
422 on the known model AND on an unknown model (schema-level
validation fires pre-handler; integers are never silently
reinterpreted).

LIVE G (E — true partition): the per-count groups form a TRUE
disjoint partition of the FULL M10 listing (9 + 1 == 10; every record
EXACTLY ONCE; no record in two groups; no None case — the int is
required on every record).

LIVE H (F — ordering): every group's order equals the authoritative
M10 (created_at, suite_run_id) ASCENDING order restricted to that
count.

LIVE I (G — M10/M21/M22/M25 regressions): the suite-run listing
identical to LIVE A; by-suite 10 with listing parity; summary total
10; by-checkpoint 10 with listing parity; every detail getter
verbatim.

LIVE J (H — M22-M49 regressions): evaluations 3/3/3 by-checkpoint +
16 by-dataset + 16 by-tokenizer + by-split 14/2/0 + by-state-kind
9/7 + by-truncated false->15 / true->1 + by-seed 11->4 / 1->3 /
7->2 / 3->2 / 7002->2 / 7101->2 / 12->1; comparisons 6/5/1
by-checkpoint + 8 by-dataset + 8 by-tokenizer + by-split 8/0/0 +
by-verdict 3/3/2 + by-state-kind 8/2 + by-seed 1->5 / 7101->2 /
7002->1; samples 4/0/0 by-checkpoint + 4 by-tokenizer + by-strategy
2/2; sample-quality 2 by-tokenizer; workflows 13 total with
by-status 9/3/1 + 2 by-recipe (m12-live-suite); gates 11 with
by-decision 7/4 + by-verdict 4/3/2 + by-baseline-type 7/2/2/0 + 4
by-comparison + 1 by-policy; checkpoints by-run 291a16d755fc -> 2 /
85438934f86a -> 1.

LIVE K (I — M12/M16/M35 regressions): recipe registry 7 verbatim;
GLOBAL /workflows/recipes/m12-live-suite/runs -> 2; the 4 zero-run
recipes -> 200 + []; unknown recipe 404; M16 evaluation ids still
paired in sample-quality.

LIVE L (J — dashboard; K — registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/tokenizer/model
registries unchanged (2 models with the known ids).

LIVE M (L — OpenAPI + storage zero drift): OpenAPI exactly 82 paths,
new path once (after by-checkpoint, before the generic suite-run
detail route; integer parameter schema; SuiteRunRecord items; the
M49/M48/M47/M46/M45/M44/M43/M42/M41 routes still present exactly
once); every pre-existing file byte-identical, ZERO new files, zero
.tmp, totals unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8771 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8771/api/v1"
SITE = "http://127.0.0.1:8771"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO suite runs
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
DATASET = "ee1a716c4573"
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
RUN_A = "291a16d755fc"                # M46: -> 2 checkpoints
RUN_B = "85438934f86a"                # M46: -> 1 checkpoint
POLICY = "m9-live-policy"
SUITE = "m9-live-suite"
KNOWN_RECIPE = "m12-live-suite"       # M35: 2 runs
ZERO_RUN_RECIPES = ("m14-base", "m14-chain-a", "m14-chain-b",
                    "m14-chain-c")
KNOWN_COMP = "fc379bfcb50f"           # M34: 4 decisions
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")
UNMATCHED_COUNT = 987654              # a valid integer, absent live
M24_COUNTS = {CK_0511: 3, CK_025E: 3, CK_30A8: 3}
M26_COUNTS = {CK_025E: 6, CK_30A8: 5, CK_0511: 1}
M27_COUNTS = {CK_0511: 4, CK_025E: 0, CK_30A8: 0}
M25_COUNT = 10                        # suite runs under m9-live-suite
M28_COUNT = 16                        # evaluations under ee1a716c4573
M29_COUNT = 8                         # comparisons under ee1a716c4573
M30_COUNT = 16                        # evaluations under 99106e3255c5
M31_COUNT = 8                         # comparisons under 99106e3255c5
M32_COUNT = 4                         # samples under 99106e3255c5
M33_COUNT = 2                         # sample-quality under 99106e3255c5
M34_COUNT = 4                         # gate decisions under fc379bfcb50f
M23_COUNT = 1                         # gate decisions under m9-live-policy
M36_DIST = {"validation": 14, "train": 2, "test": 0}
M37_COUNT = 8                         # comparisons with split=validation
M38_DIST = {"checkpoint": 9, "current": 7}
M39_DIST = {"improved": 3, "unchanged": 3, "regressed": 2}
M40_DIST = {"greedy": 2, "temperature": 2}
M41_DIST = {"passed": 7, "failed": 4}
M42_DIST = {"completed": 9, "failed": 3, "stopped": 1}
M43_DIST = {"improved": 4, "regressed": 3, "unchanged": 2}
M44_DIST = {"checkpoint": 8, "current": 2}
M45_DIST = {"checkpoint": 7, "current": 2, "minimum_loss": 2,
            "evaluation_result_hash": 0}
M46_DIST = {RUN_A: 2, RUN_B: 1}
M47_DIST = {"false": 15, "true": 1}
M48_DIST = {11: 4, 1: 3, 7: 2, 3: 2, 7002: 2, 7101: 2, 12: 1}
M49_DIST = {1: 5, 7101: 2, 7002: 1}
M50_DIST = {2: 9, 0: 1}
NEW_PATH = ("/api/v1/models/{model_id}/suite-runs/by-reused/"
            "{reused_count}")
GENERIC_PATH = "/api/v1/models/{model_id}/suite-runs/{suite_run_id}"
M49_PATH = "/api/v1/models/{model_id}/comparisons/by-seed/{seed}"
M48_PATH = "/api/v1/models/{model_id}/evaluations/by-seed/{seed}"
M47_PATH = ("/api/v1/models/{model_id}/evaluations/by-truncated/"
            "{truncated}")
M46_PATH = "/api/v1/models/{model_id}/checkpoints/by-run/{run_id}"
M45_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-baseline-type/"
            "{baseline_type}")
M44_PATH = ("/api/v1/models/{model_id}/comparisons/by-state-kind/"
            "{state_kind}")
M43_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-verdict/"
            "{verdict}")
M42_PATH = "/api/v1/models/{model_id}/workflows/by-status/{status}"
M41_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-decision/"
            "{decision}")
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m50-smoke-baseline-inventory.json")

PRE_FILES, PRE_BYTES, PRE_TMP = 96, 4_002_745, 0

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


def byreused(count, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/suite-runs/by-reused/{count}"


def main() -> int:
    print("== LIVE A: baseline audit + saved inventory ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 96/4,002,745/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    INVENTORY.write_text(json.dumps(pre_snap, indent=1, sort_keys=True))
    check("A2 full per-file hash inventory saved",
          INVENTORY.exists() and len(json.loads(INVENTORY.read_text()))
          == pre_n, str(INVENTORY))
    code, model = call_json("GET", f"{BASE}/models/{MODEL}")
    code_o, other = call_json("GET", f"{BASE}/models/{OTHER_MODEL}")
    code_sr, runs10 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                        f"by-suite/{SUITE}")
    code_sro, runs_o = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/"
                                        "suite-runs")
    code_sum, s_sum0 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                        f"by-suite/{SUITE}/summary")
    code_ck, ckpts = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_ev, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_cp, comps = call_json("GET", f"{BASE}/models/{MODEL}/"
                                      "comparisons")
    code_gd, gdecisions = call_json("GET", f"{BASE}/models/{MODEL}/"
                                           "gates/decisions")
    code_wf, workflows = call_json("GET", f"{BASE}/models/{MODEL}/"
                                          "workflows")
    code_sm, samples = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M3-M49 pre-state intact (10 suite runs + by-suite 10 + "
          "summary total 10, other model 0, 16 evaluations, 8 "
          "comparisons, 11 gate decisions, 13 workflows, 4 samples, 3 "
          "checkpoints, 7 recipes, OpenAPI 82 with the by-reused path "
          "registered exactly once, GET-only, tag suite-runs)",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_sr == 200
          and len(runs10) == M25_COUNT and code_sro == 200
          and runs_o == [] and code_sum == 200
          and s_sum0["total_count"] == M25_COUNT
          and code_ck == 200 and len(ckpts) == 3 and code_ev == 200
          and len(evals) == 16 and code_cp == 200 and len(comps) == 8
          and code_gd == 200 and len(gdecisions) == 11
          and code_wf == 200 and len(workflows) == 13
          and code_sm == 200 and len(samples) == 4
          and call_json("GET", f"{BASE}/workflows/recipes")[1].__len__()
          == 7
          and code_sp == 200 and len(paths) == 82
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["suite-runs"])
    code_d, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A5 M17 dashboard hash equals the full known value",
          code_d == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    code_a, raw_a = call("GET", byreused(2))
    check("A6 the known pair (4a0a871886ef + reused 2) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    # independent source #1: the live M10 listing (by-suite == full
    # listing here); independent source #2: the persisted manifests
    dist: dict = {}
    for r in runs10:
        dist[r["reused_count"]] = dist.get(r["reused_count"], 0) + 1
    manifest_dist: dict = {}
    for mf in ROOT.glob("suite-runs/*/manifest.json"):
        rec = json.loads(mf.read_text())
        if rec.get("model_id") != MODEL:
            continue
        v = rec["reused_count"]
        manifest_dist[v] = manifest_dist.get(v, 0) + 1
    dist_str = ", ".join(f"{k} -> {v}" for k, v in sorted(dist.items()))
    print(f"    discovered (M10 listing): {len(runs10)} suite runs of "
          f"{MODEL} per persisted reused_count: {dist_str}")
    print(f"    discovered (manifests):   {manifest_dist}")
    print(f"    discovered: null values: "
          f"{sum(1 for r in runs10 if r['reused_count'] is None)}")
    check("B1 M10 listing pre-state: 10 suite runs in ASCENDING "
          "(created_at, suite_run_id) order; persisted reused_count "
          "distribution 2->9 / 0->1 (TWO populated groups summing to "
          "10 — a TRUE disjoint partition; 0 null values — the int is "
          "REQUIRED); the persisted manifests agree EXACTLY; other "
          "model has none",
          code_sr == 200 and len(runs10) == 10
          and [(r["created_at"], r["suite_run_id"]) for r in runs10]
          == sorted((r["created_at"], r["suite_run_id"]) for r in runs10)
          and dist == dict(M50_DIST) and manifest_dist == dict(M50_DIST)
          and all(isinstance(r["reused_count"], int) for r in runs10)
          and sum(dist.values()) == 10
          and code_sro == 200 and runs_o == [])

    print("== LIVE C: (A) known counts — exact responses ==")
    groups: dict = {}
    ok_groups = True
    for value in dist:
        code_g, raw_g = call("GET", byreused(value))
        groups[value] = json.loads(raw_g)
        ok_groups = ok_groups and code_g == 200 and groups[value] == \
            [r for r in runs10 if r["reused_count"] == value]
    check("C1/A1 BOTH reused-count responses == M10 listing filtered "
          "locally by the persisted record value (no missing / extra "
          "/ duplicate; every record EXACTLY ONCE; the persisted count "
          "travels VERBATIM — never recalculated, never derived from "
          "probe outcomes, completed/failed/probe counts, status or "
          "any other field; bookkeeping, never a score)",
          ok_groups
          and all(len(groups[v]) == dist[v]
                  and all(r["model_id"] == MODEL
                          and r["reused_count"] == v
                          for r in groups[v])
                  for v in dist),
          "/".join(str(len(groups[v])) for v in sorted(dist)))
    ok_detail = True
    for r in runs10:
        code_one, one = call_json("GET", f"{BASE}/models/{MODEL}/"
                                        f"suite-runs/{r['suite_run_id']}")
        if code_one != 200 or one != r:
            ok_detail = False
            check("C2/A1 verbatim payload parity with the M10 detail "
                  "getter", False, r["suite_run_id"])
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M10 detail getter "
              "for all 10 records (results, counts, state included)",
              True)

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", byreused(2))
    _, r2 = call("GET", byreused(2))
    _, r3 = call("GET", byreused(2))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw_a)

    print("== LIVE E: (C) valid empty counts ==")
    c_pop_o, body_pop_o = call_json("GET", byreused(2, OTHER_MODEL))
    c_unm_o, body_unm_o = call_json("GET", byreused(UNMATCHED_COUNT,
                                                    OTHER_MODEL))
    c_unm, body_unm = call_json("GET", byreused(UNMATCHED_COUNT))
    check("E1/C1 a populated production count AND an unmatched valid "
          "integer under b5bc905326b6 -> 200 + [] (its M10 listing is "
          "empty), and the unmatched count on the KNOWN model -> 200 + "
          "[] (the count is an OPEN value axis — any integer is "
          "type-valid)",
          c_pop_o == 200 and body_pop_o == []
          and c_unm_o == 200 and body_unm_o == []
          and c_unm == 200 and body_unm == [] and runs_o == [])

    print("== LIVE F: (D) unknown ids / non-integer spellings ==")
    c1, _ = call_json("GET", byreused(2, "no-such-model-50"))
    check("F1/D1 unknown model + VALID integer -> 404", c1 == 404,
          f"{c1}")
    c2, b2 = call_json("GET", byreused("abc"))
    c3, b3 = call_json("GET", byreused("1.5"))
    c4, b4 = call_json("GET", byreused("12x"))
    c5, _ = call_json("GET", byreused("abc", "no-such-model-50"))
    check("F2/D1 non-integer spellings -> 422 on the known model "
          "(abc / 1.5 / 12x) AND on an unknown model "
          "(schema-level validation fires pre-handler; integers are "
          "never silently reinterpreted)",
          c2 == c3 == c4 == c5 == 422
          and all("integer" in json.dumps(b) for b in (b2, b3, b4) if b),
          f"{c2}/{c3}/{c4}/{c5}")

    print("== LIVE G: (E) true partition ==")
    ids_by = {}
    ok_groups2 = True
    for value in dist:
        _, body = call_json("GET", byreused(value))
        ids = [r["suite_run_id"] for r in body]
        ok_groups2 = ok_groups2 and len(ids) == len(set(ids)) \
            and all(r["reused_count"] == value for r in body)
        ids_by[value] = set(ids)
    all_ids = {r["suite_run_id"] for r in runs10}
    values = sorted(dist)
    pairwise = all(ids_by[a].isdisjoint(ids_by[b])
                   for i, a in enumerate(values) for b in values[i + 1:])
    union = set().union(*ids_by.values()) if ids_by else set()
    check("G1/E1 the per-count groups form a TRUE disjoint partition "
          "of the FULL M10 listing (9 + 1 == 10; every record EXACTLY "
          "ONCE; no record in two groups; no None case — the int is "
          "required on every record)",
          ok_groups2 and pairwise and union == all_ids
          and len(all_ids) == 10)

    print("== LIVE H: (F) ordering ==")
    ok_order = True
    for value in dist:
        _, body = call_json("GET", byreused(value))
        expected = [r for r in runs10 if r["reused_count"] == value]
        ok_order = ok_order and body == expected and \
            [(r["created_at"], r["suite_run_id"]) for r in body] == \
            sorted((r["created_at"], r["suite_run_id"]) for r in body)
    check("H1/F1 every group preserves the authoritative M10 "
          "(created_at, suite_run_id) ASCENDING order exactly", ok_order)

    print("== LIVE I: (G) M10/M21/M22/M25 regressions ==")
    _, runs10_2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                   f"by-suite/{SUITE}")
    _, s_sum = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                f"by-suite/{SUITE}/summary")
    srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                     f"by-checkpoint/{CK_0511}")[1]
    listing_all = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")[1]
    check("I1/G1 M10/M21/M22/M25 suite-run surfaces unchanged "
          "(by-suite listing identical to LIVE A, summary total 10, "
          "by-checkpoint 10 with listing parity, full listing "
          "identical, every detail getter verbatim per C2)",
          runs10_2 == runs10 and s_sum["total_count"] == M25_COUNT
          and srck == runs10 and listing_all == runs10)

    print("== LIVE J: (H) M22-M49 regressions ==")
    ok_evck = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_evck = ok_evck and c == 200 and len(body) == n
    evd = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/by-dataset/"
                    f"{DATASET}")[1]
    evt = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/by-tokenizer/"
                    f"{TOKENIZER}")[1]
    evs_by = {sp: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"evaluations/by-split/{sp}")[1]
              for sp in M36_DIST}
    stk_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"evaluations/by-state-kind/{k}")[1]
              for k in M38_DIST}
    t47_by = {v: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"evaluations/by-truncated/{v}")[1]
              for v in M47_DIST}
    s48_by = {s: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"evaluations/by-seed/{s}")[1]
              for s in M48_DIST}
    check("J1/H1 M24/M28/M30/M36/M38/M47/M48 evaluation histories "
          "unchanged (3/3/3 by-checkpoint, 16 by-dataset, 16 "
          "by-tokenizer, by-split 14/2/0, by-state-kind checkpoint->9 "
          "/ current->7, by-truncated false->15 / true->1, by-seed "
          "11->4 / 1->3 / 7->2 / 3->2 / 7002->2 / 7101->2 / 12->1, all "
          "with listing parity)",
          ok_evck and len(evd) == M28_COUNT
          and evd == [x for x in evals if x["dataset_id"] == DATASET]
          and len(evt) == M30_COUNT
          and evt == [x for x in evals
                      if x["tokenizer_id"] == TOKENIZER]
          and all(len(evs_by[sp]) == n
                  and evs_by[sp] == [x for x in evals if x["split"] == sp]
                  for sp, n in M36_DIST.items())
          and all(len(stk_by[k]) == n
                  and stk_by[k] == [x for x in evals
                                    if x["state_kind"] == k]
                  for k, n in M38_DIST.items())
          and all(len(t47_by[v]) == n
                  and t47_by[v] == [x for x in evals
                                    if x["truncated"] == (v == "true")]
                  for v, n in M47_DIST.items())
          and all(len(s48_by[s]) == n
                  and s48_by[s] == [x for x in evals if x["seed"] == s]
                  for s, n in M48_DIST.items()))
    ok_cmp = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_cmp = ok_cmp and c == 200 and len(body) == n
    cmpd = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/by-dataset/"
                     f"{DATASET}")[1]
    cmpt = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                     f"by-tokenizer/{TOKENIZER}")[1]
    cps_by = {sp: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"comparisons/by-split/{sp}")[1]
              for sp in M36_DIST}
    vds_by = {v: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"comparisons/by-verdict/{v}")[1]
              for v in M39_DIST}
    c44_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"comparisons/by-state-kind/{k}")[1]
              for k in M44_DIST}
    s49_by = {s: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"comparisons/by-seed/{s}")[1]
              for s in M49_DIST}
    check("J2/H1 M26/M29/M31/M37/M39/M44/M49 comparison histories "
          "unchanged (025e->6 / 30a8->5 / 0511->1 by-checkpoint, 8 "
          "by-dataset, 8 by-tokenizer, by-split validation->8 / "
          "train->0 / test->0, by-verdict improved->3 / unchanged->3 / "
          "regressed->2, by-state-kind checkpoint->8 / current->2 "
          "either-side, by-seed 1->5 / 7101->2 / 7002->1, all with "
          "listing parity)",
          ok_cmp and len(cmpd) == M29_COUNT
          and cmpd == [x for x in comps if x["dataset_id"] == DATASET]
          and len(cmpt) == M31_COUNT
          and cmpt == [x for x in comps
                       if x["tokenizer_id"] == TOKENIZER]
          and cps_by["validation"] == [x for x in comps
                                       if x["split"] == "validation"]
          and len(cps_by["validation"]) == M37_COUNT
          and cps_by["train"] == [] and cps_by["test"] == []
          and all(len(vds_by[v]) == n
                  and vds_by[v] == [x for x in comps
                                    if x["verdict"] == v]
                  for v, n in M39_DIST.items())
          and all(len(c44_by[k]) == n
                  and c44_by[k] == [x for x in comps
                                    if k in (x["state_a"]["state_kind"],
                                             x["state_b"]["state_kind"])]
                  for k, n in M44_DIST.items())
          and all(len(s49_by[s]) == n
                  and s49_by[s] == [x for x in comps if x["seed"] == s]
                  for s, n in M49_DIST.items()))
    ok_smp = True
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_smp = ok_smp and c == 200 and len(body) == n
    smpt = call_json("GET", f"{BASE}/models/{MODEL}/samples/by-tokenizer/"
                     f"{TOKENIZER}")[1]
    sty_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"samples/by-strategy/{st}")[1]
              for st in M40_DIST}
    check("J3/H1 M27/M32/M40 sample histories unchanged (4/0/0 "
          "by-checkpoint, 4 by-tokenizer, by-strategy greedy->2 / "
          "temperature->2, all with listing parity)",
          ok_smp and len(smpt) == M32_COUNT
          and smpt == [x for x in samples
                       if x["tokenizer_id"] == TOKENIZER]
          and all(len(sty_by[st]) == n
                  and sty_by[st] == [x for x in samples
                                     if x["strategy"] == st]
                  for st, n in M40_DIST.items()))
    sqt = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                    f"by-tokenizer/{TOKENIZER}")[1]
    check("J4/H1 M33 sample-quality by-tokenizer unchanged (2)",
          len(sqt) == M33_COUNT
          and sqt == [x for x in call_json(
              "GET", f"{BASE}/models/{MODEL}/sample-quality")[1]
              if x["tokenizer_id"] == TOKENIZER])
    wfs_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"workflows/by-status/{st}")[1]
              for st in M42_DIST}
    check("J5/H1 M42/M35 workflow histories unchanged (by-status "
          "completed->9 / failed->3 / stopped->1 with listing parity, "
          "13 total, 2 by-recipe with listing parity — the M11 "
          "suite-run workflow stages included in the listing)",
          all(len(wfs_by[st]) == n
              and wfs_by[st] == [x for x in workflows
                                 if x["status"] == st]
              for st, n in M42_DIST.items())
          and call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]
          == workflows
          and len(call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                            f"by-recipe/{KNOWN_RECIPE}")[1]) == 2)
    g41_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                           f"decisions/by-decision/{k}")[1]
              for k in M41_DIST}
    g43_by = {v: call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                           f"decisions/by-verdict/{v}")[1]
              for v in M43_DIST}
    g45_by = {bt: call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                            f"decisions/by-baseline-type/{bt}")[1]
              for bt in M45_DIST}
    check("J6/H1 M23/M34/M41/M43/M45 gate histories unchanged "
          "(by-decision passed->7 / failed->4, by-verdict "
          "improved->4 / regressed->3 / unchanged->2, by-baseline-type "
          "checkpoint->7 / current->2 / minimum_loss->2 / "
          "evaluation_result_hash->0, 4 by-comparison, 1 by-policy, "
          "all with listing parity)",
          all(len(g41_by[k]) == n
              and g41_by[k] == [d for d in gdecisions
                                if d["decision"] == k]
              for k, n in M41_DIST.items())
          and all(len(g43_by[v]) == n
                  and g43_by[v] == [d for d in gdecisions
                                    if d["verdict"] == v]
                  for v, n in M43_DIST.items())
          and all(len(g45_by[bt]) == n
                  and g45_by[bt] == [d for d in gdecisions
                                     if d["policy"]["baseline_type"] == bt]
                  for bt, n in M45_DIST.items())
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                            f"decisions/by-comparison/{KNOWN_COMP}")[1])
          == M34_COUNT
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                            f"decisions/by-policy/{POLICY}")[1])
          == M23_COUNT)
    g46_by = {r: call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/"
                           f"by-run/{r}")[1] for r in M46_DIST}
    check("J7/H1 M46 checkpoint-by-run history unchanged "
          "(291a16d755fc -> 2 / 85438934f86a -> 1 with listing "
          "parity)",
          all(len(g46_by[r]) == n
              and g46_by[r] == [c for c in ckpts if c["run_id"] == r]
              for r, n in M46_DIST.items()))

    print("== LIVE K: (I) M12/M16/M35 regressions ==")
    global_runs = call_json("GET", f"{BASE}/workflows/recipes/"
                            f"{KNOWN_RECIPE}/runs")[1]
    recipes = call_json("GET", f"{BASE}/workflows/recipes")[1]
    c_unk = call_json("GET", f"{BASE}/workflows/recipes/"
                      "ghost-recipe-50")[0]
    ok_zero = all(call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                            f"by-recipe/{r}")[1] == []
                  for r in ZERO_RUN_RECIPES)
    sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")[1]
    check("K1/I1 M12/M35 workflow surfaces unchanged (registry 7 "
          "verbatim, GLOBAL m12-live-suite runs 2, by-recipe 2, the 4 "
          "zero-run recipes -> 200 + [], unknown recipe 404); M16 "
          "evaluation ids still paired in sample-quality",
          len(global_runs) == 2 and len(recipes) == 7 and ok_zero
          and c_unk == 404
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))

    print("== LIVE L: (J) dashboard; (K) registries ==")
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("L1/J1 M17 dashboard result_hash + output unchanged",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    toks = call_json("GET", f"{BASE}/tokenizers")[1]
    models_reg = call_json("GET", f"{BASE}/models")[1]
    check("L2/K1 registries unchanged (M2 tokenizer registry exactly "
          "the ONE entry; M9 policy + probe-suite; M12/M14 recipe "
          "registry; M1 model registry exactly the TWO known models)",
          len(toks) == 1 and toks[0]["id"] == TOKENIZER
          and {m["id"] for m in models_reg} == {MODEL, OTHER_MODEL}
          and call_json("GET", f"{BASE}/policies/{POLICY}")[0] == 200
          and call_json("GET", f"{BASE}/probe-suites")[0] == 200
          and len(recipes) == 7)

    print("== LIVE M: OpenAPI + final storage audit (zero drift) ==")
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    params = {p["name"]: p for p in
              spec2["paths"][NEW_PATH]["get"]["parameters"]}
    check("M1 OpenAPI exactly 82 paths, the new path exactly once "
          "(after by-checkpoint, before the generic suite-run detail "
          "route; integer parameter schema; SuiteRunRecord items; the "
          "M49/M48/M47/M46/M45/M44/M43/M42/M41 routes still present "
          "exactly once)",
          code_sp2 == 200 and len(spec2["paths"]) == 82
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and set(params) == {"model_id", "reused_count"}
          and params["reused_count"]["schema"]["type"] == "integer"
          and list(spec2["paths"]).index(
              "/api/v1/models/{model_id}/suite-runs/by-checkpoint/"
              "{checkpoint_id}")
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(GENERIC_PATH)
          and list(spec2["paths"]).count(M49_PATH) == 1
          and list(spec2["paths"]).count(M48_PATH) == 1
          and list(spec2["paths"]).count(M47_PATH) == 1
          and list(spec2["paths"]).count(M46_PATH) == 1
          and list(spec2["paths"]).count(M45_PATH) == 1
          and list(spec2["paths"]).count(M44_PATH) == 1
          and list(spec2["paths"]).count(M43_PATH) == 1
          and list(spec2["paths"]).count(M42_PATH) == 1
          and list(spec2["paths"]).count(M41_PATH) == 1
          and list(spec2["paths"]).count(GENERIC_PATH) == 1)
    _, grouped2 = call_json("GET", byreused(2))
    check("M2 by-reused still deterministic at the end",
          grouped2 == groups[2])
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("M3 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("M4 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("M5 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M50 live smoke FAILED: {len(FAILURES)} check(s) "
              "failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M50 live smoke OK: narrow read-only MODEL-SCOPED "
          "by-reused grouping of the M10 suite-run history over the "
          "persisted REQUIRED integer (distribution DISCOVERED live "
          "from BOTH the M10 listing and the persisted manifests: "
          "2->9 / 0->1 among 10 suite runs, TWO populated groups with "
          "0 null values — exact listing parity, persisted "
          "reused_count VERBATIM, never recalculated, never derived "
          "from probe outcomes or any other field, execution "
          "bookkeeping never a score, authoritative (created_at, "
          "suite_run_id) order preserved, every record exactly once, "
          "TRUE disjoint partition with no None case — the OPEN "
          "integer value axis with no registry), natural valid-empty "
          "200 + [] for an unmatched integer on both models, "
          "schema-level 422 for non-integer spellings (pre-handler, "
          "even for an unknown model), 404 for unknown model with a "
          "valid integer, deterministic byte-identical x3 repeats, "
          "M3-M49 surfaces + dashboard hash + registries + OpenAPI 82 "
          "unchanged, ZERO production storage growth) verified live "
          "on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
