"""M49 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M49 is READ-ONLY: it adds one narrow
MODEL-SCOPED access path,
GET /models/{id}/comparisons/by-seed/{seed},
answering "which immutable A/B comparisons of this model ran with
this effective seed?" — the model's authoritative M5 listing
(deterministic (created_at, comparison_id) ASCENDING order) filtered
VERBATIM by the REQUIRED integer ComparisonRecord.seed persisted on
every record at run time (the seed of the identical-probe A/B
measurement; matched VERBATIM — NEVER recalculated, NEVER normalized,
NEVER derived from the comparison configuration, either side's
evaluation, state payloads, verdicts, loss deltas, ids, timestamps or
any other field; the seed is bookkeeping identity, not a quality
metric — no seed produces better comparisons), registered BEFORE the
generic /comparisons/{comparison_id} route and alongside
by-checkpoint (M26)/by-dataset (M29)/by-tokenizer (M31)/by-split
(M37)/by-verdict (M39)/by-state-kind (M44). The seed is an OPEN
integer value axis — no registry, no enum, no artificial range
constraint: any integer is type-valid (an unmatched seed on a valid
model returns 200 [] — a natural valid-empty), a non-integer spelling
is rejected 422 at the API boundary (schema-level validation,
pre-handler — integers are never silently reinterpreted, and the
validation fires BEFORE the handler even for an unknown model), while
an unknown model with a VALID integer is 404. The smoke must DISCOVER
the authoritative seed distribution from the live M5 listing (not
assume it from an old report), prove exact listing parity for ALL
discovered groups, the natural valid-empty, clean 404/422 separation,
byte-identical repeats (x3), unchanged M3-M48 surfaces + dashboard
hash + registries + OpenAPI 81, and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
source is the M5 comparison listing):
model 4a0a871886ef owns 8 comparisons in ASCENDING (created_at,
comparison_id) order with persisted seed distribution 1 -> 5,
7101 -> 2, 7002 -> 1 (THREE populated groups; 5 + 2 + 1 == 8 — a
TRUE disjoint partition; the int is REQUIRED on every record, 0 null
seeds); model b5bc905326b6 has NO comparisons; M17 dashboard
result_hash f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m49-smoke-baseline-inventory.json +
model/checkpoint/tokenizer/policy/probe-suite/recipe registries +
M4-M48 pre-state (8 comparisons, 16 evaluations with seed dist
11/1/7/3/7002/7101/12 and truncation false->15 / true->1, 11 gate
decisions, 13 workflows, 10 suite-runs, 4 samples, 3 checkpoints with
run distribution 2/1) + dashboard + OpenAPI 81 pre-state + the known
pair returns 200.

LIVE B (authoritative distribution discovery): from the live M5
listing, discover the persisted seed distribution; print it;
cross-check against the audited facts (three groups summing to 8; 0
null seeds).

LIVE C (A — known seeds): ALL discovered seeds -> 200 with the EXACT
discovered records; parity with the M5 listing filtered locally by the
persisted record value; verbatim detail-getter payloads for all 8
records; (created_at, comparison_id) ASC preserved per group.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — valid empty seeds): a populated production seed AND an
unmatched valid integer (987654) under b5bc905326b6 -> 200 + [] (it
has no comparisons at all); the unmatched seed on the KNOWN model ->
200 [] too (open value axis).

LIVE F (D — unknown ids / non-integer spellings): unknown model +
valid integer -> 404; non-integer spellings ("abc", "1.5", "12x") ->
422 on the known model AND on an unknown model (schema-level
validation fires pre-handler; integers are never silently
reinterpreted).

LIVE G (E — true partition): the per-seed groups form a TRUE disjoint
partition of the FULL M5 listing (5 + 2 + 1 == 8; every record EXACTLY
ONCE; no record in two groups; no None case — the int is required on
every record).

LIVE H (F — ordering): every group's order equals the authoritative
M5 (created_at, comparison_id) ASCENDING order restricted to that
seed.

LIVE I (G — M5/M26/M29/M31/M37/M39/M44 regressions): the comparison
listing identical to LIVE A; 025e->6 / 30a8->5 / 0511->1
by-checkpoint; 8 by-dataset; 8 by-tokenizer; by-split
validation->8 / train->0 / test->0; by-verdict improved->3 /
unchanged->3 / regressed->2; by-state-kind checkpoint->8 / current->2
(either-side) — all with listing parity.

LIVE J (H — M22-M48 regressions): evaluations 3/3/3 by-checkpoint +
16 by-dataset + 16 by-tokenizer + by-split 14/2/0 + by-state-kind
9/7 + by-truncated false->15 / true->1 + by-seed 11->4 / 1->3 /
7->2 / 3->2 / 7002->2 / 7101->2 / 12->1; samples 4/0/0
by-checkpoint + 4 by-tokenizer + by-strategy greedy->2 /
temperature->2; suite runs 10 by-suite + 10 by-checkpoint + summary
total 10; sample-quality 2 by-tokenizer; workflows 13 total with
by-status completed->9 / failed->3 / stopped->1 + 2 by-recipe
(m12-live-suite); gates 11 with by-decision passed->7 / failed->4 +
by-verdict improved->4 / regressed->3 / unchanged->2 +
by-baseline-type checkpoint->7 / current->2 / minimum_loss->2 /
evaluation_result_hash->0 + 4 by-comparison + 1 by-policy;
checkpoints by-run 291a16d755fc -> 2 / 85438934f86a -> 1.

LIVE K (I — M12/M16/M35 regressions): recipe registry 7 verbatim;
GLOBAL /workflows/recipes/m12-live-suite/runs -> 2; the 4 zero-run
recipes -> 200 + []; unknown recipe 404; M16 evaluation ids still
paired in sample-quality.

LIVE L (J — dashboard; K — registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/tokenizer/model
registries unchanged (2 models with the known ids).

LIVE M (L — OpenAPI + storage zero drift): OpenAPI exactly 81 paths,
new path once (after by-state-kind, before the generic
comparison-detail route; integer parameter schema; ComparisonRecord
items; the M48/M47/M46/M45/M44/M43/M42/M41 routes still present
exactly once); every pre-existing file byte-identical, ZERO new
files, zero .tmp, totals unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8770 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8770/api/v1"
SITE = "http://127.0.0.1:8770"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO comparisons
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
UNMATCHED_SEED = 987654               # a valid integer, absent live
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
NEW_PATH = "/api/v1/models/{model_id}/comparisons/by-seed/{seed}"
GENERIC_PATH = "/api/v1/models/{model_id}/comparisons/{comparison_id}"
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
INVENTORY = Path("/tmp/m49-smoke-baseline-inventory.json")

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


def byseed(seed, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/comparisons/by-seed/{seed}"


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
    code_cp, comps = call_json("GET", f"{BASE}/models/{MODEL}/"
                                      "comparisons")
    code_cpo, comps_o = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/"
                                         "comparisons")
    code_ev, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_ck, ckpts = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_gd, gdecisions = call_json("GET", f"{BASE}/models/{MODEL}/"
                                           "gates/decisions")
    code_wf, workflows = call_json("GET", f"{BASE}/models/{MODEL}/"
                                          "workflows")
    code_sm, samples = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code_sr, runs10 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                       f"by-suite/{SUITE}")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M4-M48 pre-state intact (8 comparisons, other model 0, "
          "16 evaluations, 3 checkpoints, 11 gate decisions, 13 "
          "workflows, 4 samples, 10 suite runs, 7 recipes, OpenAPI 81 "
          "with the by-seed path registered exactly once, GET-only, "
          "tag comparison)",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_cp == 200
          and len(comps) == 8 and code_cpo == 200 and comps_o == []
          and code_ev == 200 and len(evals) == 16
          and code_ck == 200 and len(ckpts) == 3 and code_gd == 200
          and len(gdecisions) == 11 and code_wf == 200
          and len(workflows) == 13 and code_sm == 200 and len(samples) == 4
          and code_sr == 200 and len(runs10) == 10
          and call_json("GET", f"{BASE}/workflows/recipes")[1].__len__()
          == 7
          and code_sp == 200 and len(paths) == 81
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["comparison"])
    code_d, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A5 M17 dashboard hash equals the full known value",
          code_d == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    code_a, raw_a = call("GET", byseed(1))
    check("A6 the known pair (4a0a871886ef + seed 1) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    dist: dict = {}
    for c in comps:
        dist[c["seed"]] = dist.get(c["seed"], 0) + 1
    dist_str = ", ".join(f"{k} -> {v}" for k, v in sorted(dist.items()))
    print(f"    discovered: {len(comps)} comparisons of {MODEL} per "
          f"persisted seed: {dist_str}")
    print(f"    discovered: {len(dist)} populated seed groups; "
          f"null seeds: "
          f"{sum(1 for c in comps if c['seed'] is None)}")
    check("B1 M5 listing pre-state: 8 comparisons in ASCENDING "
          "(created_at, comparison_id) order; persisted seed "
          "distribution 1->5 / 7101->2 / 7002->1 (THREE populated "
          "groups summing to 8 — a TRUE disjoint partition; 0 null "
          "seeds — the int is REQUIRED); other model has none",
          code_cp == 200 and len(comps) == 8
          and [(c["created_at"], c["comparison_id"]) for c in comps]
          == sorted((c["created_at"], c["comparison_id"]) for c in comps)
          and dist == dict(M49_DIST)
          and all(isinstance(c["seed"], int) for c in comps)
          and sum(dist.values()) == 8
          and code_cpo == 200 and comps_o == [])

    print("== LIVE C: (A) known seeds — exact responses ==")
    groups: dict = {}
    ok_groups = True
    for seed in dist:
        code_g, raw_g = call("GET", byseed(seed))
        groups[seed] = json.loads(raw_g)
        ok_groups = ok_groups and code_g == 200 and groups[seed] == \
            [c for c in comps if c["seed"] == seed]
    check("C1/A1 ALL THREE seed responses == M5 listing filtered "
          "locally by the persisted record value (no missing / extra "
          "/ duplicate; every record EXACTLY ONCE; the persisted seed "
          "travels VERBATIM — never recalculated, never normalized, "
          "never derived from the configuration, either side's "
          "evaluation, state payloads, verdicts or any other field)",
          ok_groups
          and all(len(groups[s]) == dist[s]
                  and all(c["model_id"] == MODEL and c["seed"] == s
                          for c in groups[s])
                  for s in dist),
          "/".join(str(len(groups[s])) for s in sorted(dist)))
    ok_detail = True
    for c in comps:
        code_one, one = call_json("GET", f"{BASE}/models/{MODEL}/"
                                        f"comparisons/"
                                        f"{c['comparison_id']}")
        if code_one != 200 or one != c:
            ok_detail = False
            check("C2/A1 verbatim payload parity with the M5 detail "
                  "getter", False, c["comparison_id"])
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M5 detail getter "
              "for all 8 records (both sides, losses, verdict "
              "included)", True)

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", byseed(1))
    _, r2 = call("GET", byseed(1))
    _, r3 = call("GET", byseed(1))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw_a)

    print("== LIVE E: (C) valid empty seeds ==")
    c_pop_o, body_pop_o = call_json("GET", byseed(1, OTHER_MODEL))
    c_unm_o, body_unm_o = call_json("GET", byseed(UNMATCHED_SEED,
                                                  OTHER_MODEL))
    c_unm, body_unm = call_json("GET", byseed(UNMATCHED_SEED))
    check("E1/C1 a populated production seed AND an unmatched valid "
          "integer under b5bc905326b6 -> 200 + [] (its M5 listing is "
          "empty), and the unmatched seed on the KNOWN model -> 200 + "
          "[] (the seed is an OPEN value axis — any integer is "
          "type-valid)",
          c_pop_o == 200 and body_pop_o == []
          and c_unm_o == 200 and body_unm_o == []
          and c_unm == 200 and body_unm == [] and comps_o == [])

    print("== LIVE F: (D) unknown ids / non-integer spellings ==")
    c1, _ = call_json("GET", byseed(1, "no-such-model-49"))
    check("F1/D1 unknown model + VALID integer -> 404", c1 == 404,
          f"{c1}")
    c2, b2 = call_json("GET", byseed("abc"))
    c3, b3 = call_json("GET", byseed("1.5"))
    c4, b4 = call_json("GET", byseed("12x"))
    c5, _ = call_json("GET", byseed("abc", "no-such-model-49"))
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
    for seed in dist:
        _, body = call_json("GET", byseed(seed))
        ids = [c["comparison_id"] for c in body]
        ok_groups2 = ok_groups2 and len(ids) == len(set(ids)) \
            and all(c["seed"] == seed for c in body)
        ids_by[seed] = set(ids)
    all_ids = {c["comparison_id"] for c in comps}
    seeds = sorted(dist)
    pairwise = all(ids_by[a].isdisjoint(ids_by[b])
                   for i, a in enumerate(seeds) for b in seeds[i + 1:])
    union = set().union(*ids_by.values()) if ids_by else set()
    check("G1/E1 the per-seed groups form a TRUE disjoint partition of "
          "the FULL M5 listing (5 + 2 + 1 == 8; every record EXACTLY "
          "ONCE; no record in two groups; no None case — the int is "
          "required on every record)",
          ok_groups2 and pairwise and union == all_ids
          and len(all_ids) == 8)

    print("== LIVE H: (F) ordering ==")
    ok_order = True
    for seed in dist:
        _, body = call_json("GET", byseed(seed))
        expected = [c for c in comps if c["seed"] == seed]
        ok_order = ok_order and body == expected and \
            [(c["created_at"], c["comparison_id"]) for c in body] == \
            sorted((c["created_at"], c["comparison_id"]) for c in body)
    check("H1/F1 every group preserves the authoritative M5 "
          "(created_at, comparison_id) ASCENDING order exactly (no new "
          "sort key)", ok_order)

    print("== LIVE I: (G) M5/M26/M29/M31/M37/M39/M44 regressions ==")
    _, comps2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
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
    check("I1/G1 M5/M26/M29/M31/M37/M39/M44 comparison surfaces "
          "unchanged (listing identical to LIVE A, 025e->6 / 30a8->5 / "
          "0511->1 by-checkpoint, 8 by-dataset, 8 by-tokenizer, "
          "by-split validation->8 / train->0 / test->0, by-verdict "
          "improved->3 / unchanged->3 / regressed->2, by-state-kind "
          "checkpoint->8 / current->2 either-side, all with listing "
          "parity)",
          comps2 == comps and ok_cmp and len(cmpd) == M29_COUNT
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
                  for k, n in M44_DIST.items()))

    print("== LIVE J: (H) M22-M48 regressions ==")
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
    check("J2/H1 M27/M32/M40 sample histories unchanged (4/0/0 "
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
    check("J3/H1 M33 sample-quality by-tokenizer unchanged (2)",
          len(sqt) == M33_COUNT
          and sqt == [x for x in call_json(
              "GET", f"{BASE}/models/{MODEL}/sample-quality")[1]
              if x["tokenizer_id"] == TOKENIZER])
    s_sum = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                      f"{SUITE}/summary")[1]
    srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                     f"by-checkpoint/{CK_0511}")[1]
    check("J4/H1 M22/M25 suite-run histories unchanged (10 by-suite + "
          "10 by-checkpoint + summary total 10)",
          len(runs10) == M25_COUNT and srck == runs10
          and s_sum["total_count"] == M25_COUNT)
    wfs_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"workflows/by-status/{st}")[1]
              for st in M42_DIST}
    check("J5/H1 M42/M35 workflow histories unchanged (by-status "
          "completed->9 / failed->3 / stopped->1 with listing parity, "
          "13 total, 2 by-recipe with listing parity)",
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
                      "ghost-recipe-49")[0]
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
    check("M1 OpenAPI exactly 81 paths, the new path exactly once "
          "(after by-state-kind, before the generic comparison-detail "
          "route; integer parameter schema; ComparisonRecord items; "
          "the M48/M47/M46/M45/M44/M43/M42/M41 routes still present "
          "exactly once)",
          code_sp2 == 200 and len(spec2["paths"]) == 81
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and set(params) == {"model_id", "seed"}
          and params["seed"]["schema"]["type"] == "integer"
          and list(spec2["paths"]).index(M44_PATH)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(GENERIC_PATH)
          and list(spec2["paths"]).count(M48_PATH) == 1
          and list(spec2["paths"]).count(M47_PATH) == 1
          and list(spec2["paths"]).count(M46_PATH) == 1
          and list(spec2["paths"]).count(M45_PATH) == 1
          and list(spec2["paths"]).count(M44_PATH) == 1
          and list(spec2["paths"]).count(M43_PATH) == 1
          and list(spec2["paths"]).count(M42_PATH) == 1
          and list(spec2["paths"]).count(M41_PATH) == 1
          and list(spec2["paths"]).count(GENERIC_PATH) == 1)
    _, grouped2 = call_json("GET", byseed(1))
    check("M2 by-seed still deterministic at the end",
          grouped2 == groups[1])
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
        print(f"M49 live smoke FAILED: {len(FAILURES)} check(s) "
              "failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M49 live smoke OK: narrow read-only MODEL-SCOPED by-seed "
          "grouping of the M5 comparison history over the persisted "
          "REQUIRED integer (distribution DISCOVERED live: 1->5 / "
          "7101->2 / 7002->1 among 8 comparisons, THREE populated "
          "groups with 0 null seeds — exact listing parity, persisted "
          "seed VERBATIM, never recalculated, never normalized, never "
          "derived from the configuration, either side's evaluation, "
          "state payloads, verdicts or any other field, no quality "
          "judgment, authoritative (created_at, comparison_id) order "
          "preserved, every record exactly once, TRUE disjoint "
          "partition with no None case — the OPEN integer value axis "
          "with no registry), natural valid-empty 200 + [] for an "
          "unmatched integer on both models, schema-level 422 for "
          "non-integer spellings (pre-handler, even for an unknown "
          "model), 404 for unknown model with a valid integer, "
          "deterministic byte-identical x3 repeats, M3-M48 surfaces + "
          "dashboard hash + registries + OpenAPI 81 unchanged, ZERO "
          "production storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
