"""M45 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M45 is READ-ONLY: it adds one narrow
MODEL-SCOPED access path,
GET /models/{id}/gates/decisions/by-baseline-type/{baseline_type},
answering "which immutable gate decisions of this model compared their
candidate against this kind of baseline?" — the model's authoritative
M6 listing filtered by the persisted NESTED policy field
policy.baseline_type (every GateDecision embeds its GatePolicy
VERBATIM — the M6 run persists the policy exactly as evaluated — and
every policy carries the REQUIRED GateBaselineType schema enum:
checkpoint / current / evaluation_result_hash / minimum_loss; matched
VERBATIM — NEVER derived from checkpoint id presence, comparison or
evaluation references, policy contents outside baseline_type, the gate
result, the verdict, loss deltas or timestamps; no gate is
re-evaluated), registered BEFORE the generic /gates/decisions/
{decision_id} route and alongside by-policy (M23)/by-comparison
(M34)/by-decision (M41)/by-verdict (M43). Because the field is
REQUIRED, the groups form a TRUE disjoint partition with no None
case; the enum exposes the COMPLETE contract including
evaluation_result_hash, which naturally returns [] where no such
gates exist. Unsupported baseline-type values are rejected with 422
by the schema enum BEFORE the handler runs (never a registry-style
404; true even for an unknown model); an unknown model with a VALID
baseline type is 404. The smoke must DISCOVER the authoritative
baseline-type distribution from the live M6 listing and the schema
enum (not assume it from an old report), prove exact listing parity
for ALL FOUR groups, the natural valid-empty cases, clean 404/422
separation, byte-identical repeats (x3), unchanged M2-M44 surfaces +
dashboard hash + registries + OpenAPI 77, and ZERO production storage
growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M6 listing and the GateBaselineType schema enum):
model 4a0a871886ef owns 11 gate decisions in ASCENDING (created_at,
decision_id) order with persisted policy.baseline_type distribution
checkpoint -> 7, current -> 2, minimum_loss -> 2, and
evaluation_result_hash -> 0 (a natural valid-empty; the three
populated groups form a TRUE disjoint partition of all 11); model
b5bc905326b6 has NO gate decisions; M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m45-smoke-baseline-inventory.json +
model/checkpoint registries + M4/M5/M6/M11/M15/M16/M18-M44 pre-state
(including the M45 pre-state: 7 checkpoint-baseline decisions) +
dashboard + OpenAPI 77 pre-state + the known pair returns 200.

LIVE B (authoritative distribution discovery): from the live M6
listing and the OpenAPI GateBaselineType schema enum, discover the
nested policy.baseline_type distribution; print it; cross-check
against the audited facts.

LIVE C (A — known baseline types): ALL FOUR enum values -> 200 with
the EXACT discovered records (7/2/2/0); parity with the M6 listing
filtered locally by the persisted nested field; verbatim
detail-getter payloads; (created_at, decision_id) ASC.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — valid empty baseline types): ALL enum values -> 200 + []
under b5bc905326b6 (it has no gate decisions at all);
evaluation_result_hash is a VALID value that returns [] even on the
known model — no route is omitted for empty categories.

LIVE F (D/E — unknown ids / unsupported values): unknown model +
valid baseline type -> 404; unsupported baseline-type value -> 422 on
the known model (incl. case variant + numeric + non-enum word);
unsupported baseline type + UNKNOWN model -> 422 (schema validation
fires pre-handler).

LIVE G (F — true partition): the four groups form a TRUE disjoint
partition of the full M6 listing (7 + 2 + 2 + 0 == 11; every record
EXACTLY ONCE; no record in two groups; no None case — the field is
required).

LIVE H (G — ordering): every group's order equals the authoritative
M6 listing order restricted to that baseline type.

LIVE I (H — M23/M34/M41/M43 regressions): gate decisions 1 by-policy
(m9-live-policy) + 4 by-comparison (fc379bfcb50f) + by-decision
passed->7 / failed->4 + by-verdict improved->4 / regressed->3 /
unchanged->2 (null excluded) + the M6 listing unchanged, all with
listing parity.

LIVE J (I — M22-M44 regressions): evaluations 3/3/3 by-checkpoint +
16 by-dataset + 16 by-tokenizer + by-split validation->14 / train->2
/ test->0 + by-state-kind checkpoint->9 / current->7; comparisons
6/5/1 by-checkpoint + 8 by-dataset + 8 by-tokenizer + by-split
validation->8 / train->0 / test->0 + by-verdict improved->3 /
unchanged->3 / regressed->2 + by-state-kind checkpoint->8 / current->2
(either-side); samples 4/0/0 by-checkpoint + 4 by-tokenizer +
by-strategy greedy->2 / temperature->2; suite runs 10 by-suite + 10
by-checkpoint + summary; sample-quality 2 by-tokenizer; workflows 13
total with by-status completed->9 / failed->3 / stopped->1 + 2
by-recipe (m12-live-suite).

LIVE K (J — M12/M16/M35 regressions): recipe registry 7 verbatim;
GLOBAL /workflows/recipes/m12-live-suite/runs -> 2; M35 by-recipe
m12-live-suite -> 2 + the 4 zero-run recipes -> 200 + []; M16
evaluation ids still paired in sample-quality.

LIVE L (K — dashboard; L — registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/tokenizer registries
unchanged.

LIVE M (M — OpenAPI + storage zero drift): OpenAPI exactly 77 paths,
new path once (after the by-verdict gate route, before the generic
gate-decision route; the M44 comparisons-by-state-kind, M43
gate-by-verdict, M42 workflows-by-status and M41 by-decision routes
still present exactly once); every pre-existing file byte-identical,
ZERO new files, zero .tmp, totals unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \\
        app.api:app --host 127.0.0.1 --port 8766 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8766/api/v1"
SITE = "http://127.0.0.1:8766"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO gate decisions
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
DATASET = "ee1a716c4573"
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
POLICY = "m9-live-policy"
SUITE = "m9-live-suite"
KNOWN_RECIPE = "m12-live-suite"       # M35: 2 runs
ZERO_RUN_RECIPES = ("m14-base", "m14-chain-a", "m14-chain-b",
                    "m14-chain-c")
KNOWN_COMP = "fc379bfcb50f"           # M34: 4 decisions
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")
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
EXPECTED_BASELINE_TYPES = ("checkpoint", "current", "minimum_loss",
                           "evaluation_result_hash")
NEW_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-baseline-type/"
            "{baseline_type}")
GENERIC_PATH = ("/api/v1/models/{model_id}/gates/decisions/"
                "{decision_id}")
M43_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-verdict/"
            "{verdict}")
M44_PATH = ("/api/v1/models/{model_id}/comparisons/by-state-kind/"
            "{state_kind}")
M42_PATH = "/api/v1/models/{model_id}/workflows/by-status/{status}"
M41_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-decision/"
            "{decision}")
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m45-smoke-baseline-inventory.json")

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


def bybt(bt: str, model: str = MODEL) -> str:
    return (f"{BASE}/models/{model}/gates/decisions/by-baseline-type/"
            f"{bt}")

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
    code_c, cks = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    check("A3 both models resolve; the 3 known checkpoints register "
          "(other model has none)",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_c == 200
          and {c["checkpoint_id"] for c in cks}
          == {CK_0511, CK_025E, CK_30A8}
          and call_json("GET", f"{BASE}/models/{OTHER_MODEL}/checkpoints")
          [1] == [])
    code_gd, gdecisions = call_json("GET", f"{BASE}/models/{MODEL}/"
                                           "gates/decisions")
    code_gdo, gdecisions_o = call_json("GET", f"{BASE}/models/"
                                              f"{OTHER_MODEL}/"
                                              "gates/decisions")
    code_g45c, gd45c = call_json("GET", bybt("checkpoint"))
    code_g43i, gd43i = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                        "decisions/by-verdict/improved")
    code_g41p, gd41p = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                        "decisions/by-decision/passed")
    code_cp, comps = call_json("GET", f"{BASE}/models/{MODEL}/"
                                      "comparisons")
    code_c44, c44ck = call_json("GET", f"{BASE}/models/{MODEL}/"
                                       "comparisons/by-state-kind/"
                                       "checkpoint")
    code_wf, workflows = call_json("GET", f"{BASE}/models/{MODEL}/"
                                          "workflows")
    code_w42, wf42c = call_json("GET", f"{BASE}/models/{MODEL}/"
                                        "workflows/by-status/completed")
    code_ev, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_sm, samples = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code_rc, recipes = call_json("GET", f"{BASE}/workflows/recipes")
    code_runs, global_runs = call_json(
        "GET", f"{BASE}/workflows/recipes/{KNOWN_RECIPE}/runs")
    code_gr, gd34 = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                     f"decisions/by-comparison/"
                                     f"{KNOWN_COMP}")
    code_gp, gd23 = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                     f"decisions/by-policy/{POLICY}")
    code_byrec, byrec_runs = call_json("GET", f"{BASE}/models/{MODEL}/"
                                              f"workflows/by-recipe/"
                                              f"{KNOWN_RECIPE}")
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code6, g = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}")
    code7, s = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}/summary")
    code10, srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                    f"by-checkpoint/{CK_0511}")
    code15, evt = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-tokenizer/{TOKENIZER}")
    code16, cmpt = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                    f"by-tokenizer/{TOKENIZER}")
    code17, smpt = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                    f"by-tokenizer/{TOKENIZER}")
    code18, sqt = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   f"by-tokenizer/{TOKENIZER}")
    code_spv, spv = call_json("GET", f"{BASE}/models/{MODEL}/"
                                     "evaluations/by-split/validation")
    code_cpv, cpv = call_json("GET", f"{BASE}/models/{MODEL}/"
                                     "comparisons/by-split/validation")
    code_stk, stk = call_json("GET", f"{BASE}/models/{MODEL}/"
                                     "evaluations/by-state-kind/checkpoint")
    code_vd, vd = call_json("GET", f"{BASE}/models/{MODEL}/"
                                   "comparisons/by-verdict/improved")
    code_sty, sty = call_json("GET", f"{BASE}/models/{MODEL}/"
                                     "samples/by-strategy/greedy")
    code14, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr, probes = call_json("GET", f"{BASE}/probe-suites")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M4/M5/M6/M11/M15/M16/M18-M44 pre-state intact (11 gate "
          "decisions, other model 0, 7 checkpoint-baseline decisions, "
          "4 by-verdict improved, 7 by-decision passed, 8 comparisons, "
          "8 either-side checkpoint comparisons, 13 workflow runs, 9 "
          "by-status completed workflows, 16 evaluations, 4 samples, 7 "
          "recipes, 2 global m12-live-suite runs, 2 by-recipe runs, 4 "
          "by-comparison decisions, 1 by-policy decision, 2 "
          "sample-quality, 10 by-suite, summary 10, 10 by-checkpoint "
          "suite runs, 16/8/4/2 by-tokenizer "
          "evals/comparisons/samples/sample-quality, 14 validation-split "
          "evaluations, 8 validation-split comparisons, 9 checkpoint-kind "
          "evaluations, 3 improved-verdict comparisons, 2 greedy-strategy "
          "samples)",
          code_gd == 200 and len(gdecisions) == 11 and code_gdo == 200
          and gdecisions_o == [] and code_g45c == 200
          and len(gd45c) == 7 and code_g43i == 200
          and len(gd43i) == 4 and code_g41p == 200
          and len(gd41p) == 7 and code_cp == 200
          and len(comps) == 8 and code_c44 == 200
          and len(c44ck) == 8 and code_wf == 200
          and len(workflows) == 13 and code_w42 == 200
          and len(wf42c) == 9 and code_ev == 200
          and len(evals) == 16 and code_sm == 200 and len(samples) == 4
          and code_rc == 200 and len(recipes) == 7 and code_runs == 200
          and len(global_runs) == 2 and code_byrec == 200
          and len(byrec_runs) == 2 and code_gr == 200
          and len(gd34) == 4 and code_gp == 200 and len(gd23) == 1
          and code2 == 200 and len(sq) == 2 and code6 == 200
          and len(g) == 10 and code7 == 200 and s["total_count"] == 10
          and code10 == 200 and len(srck) == 10 and code15 == 200
          and len(evt) == 16 and code16 == 200 and len(cmpt) == 8
          and code17 == 200 and len(smpt) == 4 and code18 == 200
          and len(sqt) == 2 and code_p == 200
          and code_spv == 200 and len(spv) == 14
          and code_cpv == 200 and len(cpv) == 8
          and code_stk == 200 and len(stk) == 9
          and code_vd == 200 and len(vd) == 3
          and code_sty == 200 and len(sty) == 2
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code14 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 77 paths, the new by-baseline-type "
          "path registered exactly once with only GET",
          code_sp == 200 and len(paths) == 77
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["gates"])
    code_a, raw_a = call("GET", bybt("checkpoint"))
    check("A7 the known pair (4a0a871886ef + checkpoint) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    schema = spec["components"]["schemas"]["GateBaselineType"]
    enum = tuple(schema.get("enum", ()))
    dist = {bt: sum(1 for d in gdecisions
                    if d["policy"]["baseline_type"] == bt)
            for bt in enum}
    dist_str = ", ".join(f"{k} -> {v}" for k, v in sorted(dist.items()))
    print(f"    discovered: {len(gdecisions)} gate decisions of {MODEL} "
          f"per policy.baseline_type: {dist_str}")
    print(f"    discovered: GateBaselineType enum values: {list(enum)}")
    check("B1 M6 listing pre-state: 11 gate decisions in ASCENDING "
          "(created_at, decision_id) order; nested baseline-type "
          "distribution checkpoint->7, current->2, minimum_loss->2, "
          "evaluation_result_hash->0; other model has none",
          code_gd == 200 and len(gdecisions) == 11
          and [(d["created_at"], d["decision_id"]) for d in gdecisions]
          == sorted((d["created_at"], d["decision_id"])
                    for d in gdecisions)
          and dist == dict(M45_DIST)
          and all(d["policy"]["baseline_type"] in enum
                  for d in gdecisions)
          and code_gdo == 200 and gdecisions_o == [])
    check("B2 the schema enum holds exactly the four baseline types",
          set(enum) == set(EXPECTED_BASELINE_TYPES) and len(enum) == 4,
          str(enum))

    print("== LIVE C: (A) known baseline types — exact responses ==")
    groups: dict[str, list] = {}
    ok_groups = True
    for bt in EXPECTED_BASELINE_TYPES:
        code_g, raw_g = call("GET", bybt(bt))
        groups[bt] = json.loads(raw_g)
        ok_groups = ok_groups and code_g == 200 and groups[bt] == \
            [d for d in gdecisions
             if d["policy"]["baseline_type"] == bt]
    check("C1/A1 ALL FOUR baseline-type responses == M6 listing "
          "filtered locally by the persisted NESTED policy.baseline_type "
          "(no missing / extra / duplicate; every record EXACTLY ONCE; "
          "the persisted nested field travels VERBATIM — never derived "
          "from checkpoint ids, references or results; "
          "evaluation_result_hash is a VALID value that naturally "
          "returns [])",
          ok_groups
          and all(len(groups[bt]) == M45_DIST[bt]
                  and all(d["model_id"] == MODEL
                          and d["policy"]["baseline_type"] == bt
                          for d in groups[bt])
                  for bt in EXPECTED_BASELINE_TYPES),
          "/".join(str(len(groups[bt]))
                   for bt in EXPECTED_BASELINE_TYPES))
    ok_detail = True
    for bt in EXPECTED_BASELINE_TYPES:
        for d in groups[bt]:
            code_d, one = call_json("GET", f"{BASE}/models/{MODEL}/"
                                           f"gates/decisions/"
                                           f"{d['decision_id']}")
            if code_d != 200 or one != d:
                ok_detail = False
                check("C2/A1 verbatim payload parity with the M6 "
                      "detail getter", False, d["decision_id"])
                break
        if not ok_detail:
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M6 detail getter "
              "for all 11 records (embedded policy, verdict, evidence "
              "chain included)", True)

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", bybt("checkpoint"))
    _, r2 = call("GET", bybt("checkpoint"))
    _, r3 = call("GET", bybt("checkpoint"))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw_a)

    print("== LIVE E: (C) valid empty baseline types ==")
    ok_other = True
    for bt in enum:
        c_o, body_o = call_json("GET", bybt(bt, OTHER_MODEL))
        ok_other = ok_other and c_o == 200 and body_o == []
    check("E1/C1 ALL enum baseline types under b5bc905326b6 -> 200 + [] "
          "(its M6 listing is empty; every baseline-type value is "
          "schema-validated even when the model has no history)",
          ok_other and gdecisions_o == [])

    print("== LIVE F: (D/E) unknown ids / unsupported values ==")
    c1, _ = call_json("GET", bybt("checkpoint", "no-such-model-45"))
    check("F1/D1 unknown model + VALID baseline type -> 404",
          c1 == 404, f"{c1}")
    c2, b2 = call_json("GET", bybt("CHECKPOINT"))
    c3, b3 = call_json("GET", bybt("check%20point"))
    c4, b4 = call_json("GET", bybt("1"))
    check("F2/E1 unsupported baseline-type values -> 422 on the known "
          "model (schema enum rejection — case variant, interior-space, "
          "numeric, non-enum word — never a 404, never [])",
          c2 == c3 == c4 == 422
          and all("enum" in json.dumps(b) for b in (b2, b3, b4) if b),
          f"{c2}/{c3}/{c4}")
    c5, _ = call_json("GET", bybt("baseline", "no-such-model-45"))
    check("F3/E1 unsupported baseline type + UNKNOWN model -> 422 "
          "(validation fires pre-handler; the 422/404 precedence is "
          "the schema's, not a manual conversion)",
          c5 == 422, f"{c5}")

    print("== LIVE G: (F) true partition ==")
    ids_by = {}
    ok_groups2 = True
    for bt in enum:
        _, body = call_json("GET", bybt(bt))
        ids = [d["decision_id"] for d in body]
        ok_groups2 = ok_groups2 and len(ids) == len(set(ids)) \
            and all(d["policy"]["baseline_type"] == bt for d in body)
        ids_by[bt] = set(ids)
    all_ids = {d["decision_id"] for d in gdecisions}
    pairwise = all(ids_by[a].isdisjoint(ids_by[b])
                   for i, a in enumerate(enum) for b in enum[i + 1:])
    check("G1/F1 the four groups form a TRUE disjoint partition of the "
          "FULL M6 listing (7 + 2 + 2 + 0 == 11; every record EXACTLY "
          "ONCE; no record in two groups; no None case — the nested "
          "field is required)",
          ok_groups2 and pairwise
          and set().union(*ids_by.values()) == all_ids
          and len(set().union(*ids_by.values())) == 11)

    print("== LIVE H: (G) ordering ==")
    ok_order = True
    for bt in enum:
        _, body = call_json("GET", bybt(bt))
        expected = [d for d in gdecisions
                    if d["policy"]["baseline_type"] == bt]
        ok_order = ok_order and body == expected and \
            [(d["created_at"], d["decision_id"]) for d in body] == \
            sorted((d["created_at"], d["decision_id"]) for d in body)
    check("H1/G1 every group preserves the authoritative M6 "
          "(created_at, decision_id) ASCENDING order exactly", ok_order)

    print("== LIVE I: (H) M23/M34/M41/M43 + M6 regressions ==")
    _, gd342 = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                f"decisions/by-comparison/{KNOWN_COMP}")
    _, gd232 = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                f"decisions/by-policy/{POLICY}")
    _, gd41p2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                 "decisions/by-decision/passed")
    g43_by = {v: call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                           f"decisions/by-verdict/{v}")[1]
              for v in M43_DIST}
    _, gdecisions2 = call_json("GET", f"{BASE}/models/{MODEL}/"
                                       "gates/decisions")
    check("I1/H1 M23/M34/M41/M43 gate histories + the M6 listing "
          "unchanged (4 by-comparison with listing parity, 1 by-policy "
          "with listing parity, by-decision passed->7 with listing "
          "parity, by-verdict improved->4 / regressed->3 / unchanged->2 "
          "with listing parity and the 2 null-verdict records in NO "
          "group, listing identical to LIVE A)",
          gd342 == gd34 and len(gd342) == M34_COUNT
          and gd342 == [d for d in gdecisions
                        if d.get("comparison_id") == KNOWN_COMP]
          and gd232 == gd23 and len(gd232) == M23_COUNT
          and gd232 == [d for d in gdecisions
                        if d.get("policy_id") == POLICY]
          and gd41p2 == gd41p and len(gd41p2) == M41_DIST["passed"]
          and gd41p2 == [d for d in gdecisions
                         if d["decision"] == "passed"]
          and all(len(g43_by[v]) == n
                  and g43_by[v] == [d for d in gdecisions
                                    if d["verdict"] == v]
                  for v, n in M43_DIST.items())
          and gd43i == g43_by["improved"]
          and gdecisions2 == gdecisions)

    print("== LIVE J: (I) M22-M44 regressions ==")
    ok_suite = (code6 == 200 and len(g) == M25_COUNT
                and code7 == 200 and s["total_count"] == M25_COUNT
                and code10 == 200 and len(srck) == M25_COUNT
                and call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                  f"by-suite/{SUITE}")[1] == g)
    check("J1/I1 M22 suite-run histories unchanged (10 by-suite + 10 "
          "by-checkpoint + summary total 10)", ok_suite)
    ok_evck = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_evck = ok_evck and c == 200 and len(body) == n
    _, evd2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-dataset/{DATASET}")
    _, evt2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-tokenizer/{TOKENIZER}")
    evs_by = {sp: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"evaluations/by-split/{sp}")[1]
              for sp in M36_DIST}
    stk_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"evaluations/by-state-kind/{k}")[1]
              for k in M38_DIST}
    check("J2/I1 M24/M28/M30/M36/M38 evaluation histories unchanged "
          "(3/3/3 by-checkpoint, 16 by-dataset, 16 by-tokenizer, "
          "by-split 14/2/0, by-state-kind checkpoint->9 / current->7, "
          "all with listing parity)",
          ok_evck and len(evd2) == M28_COUNT
          and evd2 == [x for x in evals if x["dataset_id"] == DATASET]
          and len(evt2) == M30_COUNT and evt2 == evt
          and all(len(evs_by[sp]) == n
                  and evs_by[sp] == [x for x in evals
                                     if x["split"] == sp]
                  for sp, n in M36_DIST.items())
          and all(len(stk_by[k]) == n
                  and stk_by[k] == [x for x in evals
                                    if x["state_kind"] == k]
                  for k, n in M38_DIST.items())
          and stk_by["checkpoint"] == stk)
    ok_cmp = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_cmp = ok_cmp and c == 200 and len(body) == n
    _, cmpd2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-dataset/{DATASET}")
    _, cmpt2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-tokenizer/{TOKENIZER}")
    cps_by = {sp: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"comparisons/by-split/{sp}")[1]
              for sp in M36_DIST}
    vds_by = {v: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"comparisons/by-verdict/{v}")[1]
              for v in M39_DIST}
    c44_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"comparisons/by-state-kind/{k}")[1]
              for k in M44_DIST}
    check("J3/I1 M26/M29/M31/M37/M39/M44 comparison histories unchanged "
          "(025e->6 / 30a8->5 / 0511->1 by-checkpoint, 8 by-dataset, 8 "
          "by-tokenizer, by-split validation->8 / train->0 / test->0, "
          "by-verdict improved->3 / unchanged->3 / regressed->2, "
          "by-state-kind checkpoint->8 / current->2 either-side, all "
          "with listing parity)",
          ok_cmp and len(cmpd2) == M29_COUNT
          and cmpd2 == [x for x in comps if x["dataset_id"] == DATASET]
          and len(cmpt2) == M31_COUNT and cmpt2 == cmpt
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
          and c44_by["checkpoint"] == c44ck and vds_by["improved"] == vd)
    ok_smp = True
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_smp = ok_smp and c == 200 and len(body) == n
    _, smpt2 = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                f"by-tokenizer/{TOKENIZER}")
    sty_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"samples/by-strategy/{st}")[1]
              for st in M40_DIST}
    check("J4/I1 M27/M32/M40 sample histories unchanged (4/0/0 "
          "by-checkpoint, 4 by-tokenizer, by-strategy greedy->2 / "
          "temperature->2, all with listing parity)",
          ok_smp and len(smpt2) == M32_COUNT
          and smpt2 == [x for x in samples
                        if x["tokenizer_id"] == TOKENIZER]
          and all(len(sty_by[st]) == n
                  and sty_by[st] == [x for x in samples
                                     if x["strategy"] == st]
                  for st, n in M40_DIST.items())
          and sty_by["greedy"] == sty)
    _, sqt2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                               f"by-tokenizer/{TOKENIZER}")
    check("J5/I1 M33 sample-quality by-tokenizer unchanged (2)",
          len(sqt2) == M33_COUNT and sqt2 == sqt)
    wfs_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"workflows/by-status/{st}")[1]
              for st in M42_DIST}
    _, workflows2 = call_json("GET", f"{BASE}/models/{MODEL}/"
                                     "workflows")
    _, byrec2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                                 f"by-recipe/{KNOWN_RECIPE}")
    check("J6/I1 M42/M35 workflow histories unchanged (by-status "
          "completed->9 / failed->3 / stopped->1 with listing parity, "
          "13 total, 2 by-recipe with listing parity)",
          all(len(wfs_by[st]) == n
              and wfs_by[st] == [x for x in workflows
                                 if x["status"] == st]
              for st, n in M42_DIST.items())
          and wfs_by["completed"] == wf42c
          and workflows2 == workflows
          and byrec2 == byrec_runs and len(byrec2) == 2)

    print("== LIVE K: (J) M12/M16/M35 regressions ==")
    _, global_runs2 = call_json(
        "GET", f"{BASE}/workflows/recipes/{KNOWN_RECIPE}/runs")
    _, recipes3 = call_json("GET", f"{BASE}/workflows/recipes")
    c_unk, _ = call_json("GET", f"{BASE}/workflows/recipes/"
                                "ghost-recipe-45")
    ok_zero = all(call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                            f"by-recipe/{r}")[1] == []
                  for r in ZERO_RUN_RECIPES)
    _, byrec3 = call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                                 f"by-recipe/{KNOWN_RECIPE}")
    check("K1/J1 M12/M35 workflow surfaces unchanged (registry 7 "
          "verbatim, GLOBAL m12-live-suite runs 2, by-recipe 2, the 4 "
          "zero-run recipes -> 200 + [], unknown recipe 404)",
          global_runs2 == global_runs and len(global_runs2) == 2
          and recipes3 == recipes and len(recipes3) == 7
          and byrec3 == byrec_runs and len(byrec3) == 2 and ok_zero
          and c_unk == 404)
    _, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    check("K2/J1 M16 evaluation ids still paired in sample-quality",
          {x["evaluation_id"] for x in sq2} == set(M16_EVAL_IDS)
          and sq2 == sq)

    print("== LIVE L: (K) dashboard; (L) registries ==")
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("L1/K1 M17 dashboard result_hash + output unchanged",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_t, toks = call_json("GET", f"{BASE}/tokenizers")
    code_tg, tok_get = call_json("GET", f"{BASE}/tokenizers/{TOKENIZER}")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr2, probes2 = call_json("GET", f"{BASE}/probe-suites")
    check("L2/L1 registries unchanged (M2 tokenizer registry exactly "
          "the ONE entry + get; M9 policy + probe-suite; M12/M14 "
          "recipe registry already verified verbatim in K1)",
          code_t == 200 and len(toks) == 1
          and toks[0]["id"] == TOKENIZER
          and code_tg == 200 and tok_get["id"] == TOKENIZER
          and code_p2 == 200 and pol2 == pol
          and code_pr2 == 200 and probes2 == probes)

    print("== LIVE M: OpenAPI + final storage audit (zero drift) ==")
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    check("M1 OpenAPI exactly 77 paths, the new path exactly once "
          "(after the by-verdict gate route, before the generic "
          "gate-decision route; the M44 comparisons-by-state-kind, M43 "
          "gate-by-verdict, M42 workflows-by-status and M41 by-decision "
          "routes still present exactly once)",
          code_sp2 == 200 and len(spec2["paths"]) == 77
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(M43_PATH)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(GENERIC_PATH)
          and list(spec2["paths"]).count(M44_PATH) == 1
          and list(spec2["paths"]).count(M43_PATH) == 1
          and list(spec2["paths"]).count(M42_PATH) == 1
          and list(spec2["paths"]).count(M41_PATH) == 1)
    _, grouped2 = call_json("GET", bybt("checkpoint"))
    check("M2 by-baseline-type still deterministic at the end",
          grouped2 == groups["checkpoint"])
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
        print(f"M45 live smoke FAILED: {len(FAILURES)} check(s) "
              "failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M45 live smoke OK: narrow read-only MODEL-SCOPED "
          "by-baseline-type grouping of the M6 gate-decision history "
          "over the persisted NESTED policy field (distribution "
          "DISCOVERED live: checkpoint->7, current->2, minimum_loss->2, "
          "evaluation_result_hash->0 among 11 gate decisions — exact "
          "listing parity, persisted nested policy.baseline_type "
          "VERBATIM (never derived from checkpoint ids, references or "
          "results, never re-evaluated), authoritative order, every "
          "record exactly once, TRUE disjoint partition with no None "
          "case and the COMPLETE enum contract including the naturally "
          "empty evaluation_result_hash —, schema-enum 422 for "
          "unsupported baseline-type values even pre-handler and even "
          "for an unknown model, valid empty 200 + [] for every "
          "baseline type under b5bc905326b6, clean 404 for unknown "
          "model with a valid baseline type, deterministic "
          "byte-identical x3 repeats, M12/M16/M22-M44 surfaces + "
          "dashboard hash + registries + OpenAPI 77 unchanged, ZERO "
          "production storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
