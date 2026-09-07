"""M37 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M37 is READ-ONLY: it adds one narrow
MODEL-SCOPED access path, GET /models/{id}/comparisons/by-split/{split},
answering "which immutable M5 comparison records of this model measured
this dataset split?" — the model's authoritative M5 listing filtered by
the persisted top-level split recorded in each ComparisonRecord (the
shared-probe contract: a comparison exists only when BOTH sides measure
the SAME dataset/version/split/tokenizer/window/seed probe, so the
split is a property of the comparison itself — matched VERBATIM against
the EvaluationSplit schema enum, never inferred from filenames,
checkpoint ids, dataset identities, nested evaluation records or
timestamps), registered BEFORE the generic /comparisons/{comparison_id}
route and alongside by-checkpoint/by-dataset/by-tokenizer. Unsupported
split values are rejected with 422 by the schema enum BEFORE the
handler runs (never a registry-style 404; true even for an unknown
model); an unknown model with a VALID split is 404; a valid split with
zero matching records is 200 + [] (never 404). Each record appears
EXACTLY ONCE and the groups partition the full M5 listing. The smoke
must DISCOVER the authoritative comparison split distribution from the
live M5 listing and the schema enum (not assume it from an old report),
prove exact listing parity, the natural valid-empty cases, clean
404/422 separation, byte-identical repeats (x3), unchanged M2-M36
surfaces + dashboard hash + registries + OpenAPI 69, and ZERO
production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M5 listing and the EvaluationSplit schema enum): model
4a0a871886ef owns 8 comparisons in ASCENDING (created_at,
comparison_id) order, ALL with persisted split "validation" (train ->
0, test -> 0 — the natural valid-enum empty cases); model b5bc905326b6
has NO comparisons; M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m37-smoke-baseline-inventory.json +
model/checkpoint registries + M4/M5/M6/M11/M16/M18-M36 pre-state +
dashboard + OpenAPI 69 pre-state + the known pair returns 200.

LIVE B (authoritative distribution discovery): from the live M5
listing and the OpenAPI EvaluationSplit schema enum, discover the
per-split comparison distribution and the enum values; print them;
cross-check against the audited facts.

LIVE C (A — known split): validation -> 200 with the EXACT discovered
records; parity with the M5 listing filtered locally; verbatim
detail-getter payloads; (created_at, comparison_id) ASC.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — valid empty splits): train and test -> 200 + [] on the
known model; ALL enum splits -> 200 + [] under b5bc905326b6 (it has no
comparisons at all).

LIVE F (D/E — unknown ids / unsupported values): unknown model + valid
split -> 404; unsupported split value -> 422 on the known model
(incl. case variant + numeric); unsupported split + UNKNOWN model ->
422 (schema validation fires pre-handler).

LIVE G (F — partition): the groups over all enum splits partition the
full M5 listing exactly (8 + 0 + 0 == 8; every record EXACTLY ONCE; no
record in two groups; every record's persisted split verbatim).

LIVE H (G — ordering): every group's order equals the authoritative
M5 listing order restricted to that split.

LIVE I (H — M26/M29/M31 regressions): comparisons 6/5/1 by-checkpoint
+ 8 by-dataset + 8 by-tokenizer + parity with the filtered M5
listing.

LIVE J (I — M22-M34 + M36 regressions): evaluations 3/3/3
by-checkpoint + 16 by-dataset + 16 by-tokenizer; M36 evaluations
by-split validation -> 14 / train -> 2 / test -> 0 + parity; suite
runs 10 by-suite + 10 by-checkpoint + summary; samples 4/0/0
by-checkpoint + 4 by-tokenizer; sample-quality 2 by-tokenizer; gate
decisions 4 by-comparison (fc379bfcb50f) + 1 by-policy (m9-live-policy).

LIVE K (J — M12/M16/M35 regressions): recipe registry 7 verbatim;
GLOBAL /workflows/recipes/m12-live-suite/runs -> 2; M35 by-recipe
m12-live-suite -> 2 + the 4 zero-run recipes -> 200 + []; M16
evaluation ids still paired in sample-quality.

LIVE L (K — dashboard; L — registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/tokenizer registries
unchanged.

LIVE M (M — OpenAPI + storage zero drift): OpenAPI exactly 69 paths,
new path once (after the by-tokenizer comparison route, before the
generic comparison route); every pre-existing file byte-identical,
ZERO new files, zero .tmp, totals unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8758 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8758/api/v1"
SITE = "http://127.0.0.1:8758"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO evaluations
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
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_DIST = {"validation": 8, "train": 0, "test": 0}
NEW_PATH = "/api/v1/models/{model_id}/comparisons/by-split/{split}"
GENERIC_PATH = "/api/v1/models/{model_id}/comparisons/{comparison_id}"
TOK_PATH = ("/api/v1/models/{model_id}/comparisons/by-tokenizer/"
            "{tokenizer_id}")
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m37-smoke-baseline-inventory.json")

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


def bysplit(split: str, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/comparisons/by-split/{split}"



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
    code_cp, comps = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    code_cpo, comps_o = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/"
                                        "comparisons")
    code_ev, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_evo, evals_o = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/"
                                         "evaluations")
    code_rc, recipes = call_json("GET", f"{BASE}/workflows/recipes")
    code_runs, global_runs = call_json(
        "GET", f"{BASE}/workflows/recipes/{KNOWN_RECIPE}/runs")
    code_gr, gd34 = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                     f"decisions/by-comparison/"
                                     f"{KNOWN_COMP}")
    code_gp, gd23 = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                     f"decisions/by-policy/{POLICY}")
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
    code_byrec, byrec_runs = call_json("GET", f"{BASE}/models/{MODEL}/"
                                              f"workflows/by-recipe/"
                                              f"{KNOWN_RECIPE}")
    code_evs, evs_val = call_json("GET", f"{BASE}/models/{MODEL}/"
                                         "evaluations/by-split/validation")
    code14, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr, probes = call_json("GET", f"{BASE}/probe-suites")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M4/M5/M6/M11/M16/M18-M36 pre-state intact (8 comparisons, "
          "other model 0, 16 evaluations, other model 0, 7 recipes, 2 "
          "global m12-live-suite runs, 2 by-recipe runs, 4 "
          "by-comparison decisions, 1 by-policy decision, 2 "
          "sample-quality, 10 by-suite, summary 10, 10 by-checkpoint "
          "suite runs, 16/8/4/2 by-tokenizer evals/comparisons/samples/"
          "sample-quality, 14 validation-split evaluations)",
          code_cp == 200 and len(comps) == 8 and code_cpo == 200
          and comps_o == [] and code_ev == 200 and len(evals) == 16
          and code_evo == 200 and evals_o == []
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
          and code_evs == 200 and len(evs_val) == 14
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code14 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 69 paths, the new by-split comparison "
          "path registered exactly once with only GET",
          code_sp == 200 and len(paths) == 69
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["comparison"])
    code_a, raw_a = call("GET", bysplit("validation"))
    check("A7 the known pair (4a0a871886ef + validation) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    schema = spec["components"]["schemas"]["EvaluationSplit"]
    enum = tuple(schema.get("enum", ()))
    by_split: dict[str, list[str]] = {}
    for c in comps:
        by_split.setdefault(c["split"], []).append(c["comparison_id"])
    dist_str = ", ".join(f"{k} -> {len(v)}"
                         for k, v in sorted(by_split.items()))
    print(f"    discovered: {len(comps)} comparisons of {MODEL} per "
          f"split: {dist_str}")
    print(f"    discovered: EvaluationSplit enum values: {list(enum)}")
    check("B1 M5 listing pre-state: 8 comparisons in ASCENDING "
          "(created_at, comparison_id) order; distribution "
          "validation->8, train->0, test->0; other model has none",
          code_cp == 200 and len(comps) == 8
          and [(c["created_at"], c["comparison_id"]) for c in comps]
          == sorted((c["created_at"], c["comparison_id"]) for c in comps)
          and {sp: len(by_split.get(sp, [])) for sp in enum}
          == dict(EXPECTED_DIST)
          and all(c["split"] in enum for c in comps)
          and code_cpo == 200 and comps_o == [])
    check("B2 the schema enum holds exactly train/validation/test",
          set(enum) == set(EXPECTED_SPLITS) and len(enum) == 3,
          str(enum))

    print("== LIVE C: (A) known split — exact response ==")
    code, raw1 = call("GET", bysplit("validation"))
    grouped = json.loads(raw1)
    filtered = [c for c in comps if c["split"] == "validation"]
    ids = [c["comparison_id"] for c in grouped]
    check("C1/A1 response == M5 listing filtered locally by the "
          "persisted shared-probe split (no missing / extra / "
          "duplicate; every record EXACTLY ONCE; each record's "
          "persisted split verbatim)",
          code == 200 and grouped == filtered
          and len(ids) == len(set(ids)) == len(filtered) == 8
          and all(c["model_id"] == MODEL and c["split"] == "validation"
                  for c in grouped),
          f"{code}/{len(grouped)}")
    ok_detail = True
    for c in grouped:
        code_d, one = call_json("GET", f"{BASE}/models/{MODEL}/"
                                       f"comparisons/"
                                       f"{c['comparison_id']}")
        if code_d != 200 or one != c:
            ok_detail = False
            check("C2/A1 verbatim payload parity with the M5 detail "
                  "getter", False, c["comparison_id"])
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M5 detail getter "
              "for all 8 records (verdict/per-side losses + split "
              "included)", True)

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", bysplit("validation"))
    _, r2 = call("GET", bysplit("validation"))
    _, r3 = call("GET", bysplit("validation"))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw1)

    print("== LIVE E: (C) valid empty splits ==")
    ok_empty = True
    for sp in ("train", "test"):
        c_e, body_e = call_json("GET", bysplit(sp))
        ok_empty = ok_empty and c_e == 200 and body_e == []
    check("E1/C1 train AND test (valid enum values with zero matching "
          "records) -> 200 + [] on the known model (NEVER 404)",
          ok_empty)
    ok_other = True
    for sp in enum:
        c_o, body_o = call_json("GET", bysplit(sp, OTHER_MODEL))
        ok_other = ok_other and c_o == 200 and body_o == []
    check("E2/C1 ALL enum splits under b5bc905326b6 -> 200 + [] (its "
          "M5 listing is empty; the split value is schema-validated "
          "even when the model has no history)",
          ok_other and comps_o == [])

    print("== LIVE F: (D/E) unknown ids / unsupported values ==")
    c1, _ = call_json("GET", bysplit("validation", "no-such-model-37"))
    check("F1/D1 unknown model + VALID split -> 404",
          c1 == 404, f"{c1}")
    c2, b2 = call_json("GET", bysplit("valid%20ation"))
    c3, b3 = call_json("GET", bysplit("VALIDATION"))
    c4, b4 = call_json("GET", bysplit("3"))
    check("F2/E1 unsupported split values -> 422 on the known model "
          "(schema enum rejection — interior-space, case variant, "
          "numeric — never a 404, never [])",
          c2 == c3 == c4 == 422
          and all("enum" in json.dumps(b) for b in (b2, b3, b4) if b),
          f"{c2}/{c3}/{c4}")
    c5, _ = call_json("GET", bysplit("banana-split", "no-such-model-37"))
    check("F3/E1 unsupported split + UNKNOWN model -> 422 (validation "
          "fires pre-handler; the 422/404 precedence is the schema's, "
          "not a manual conversion)",
          c5 == 422, f"{c5}")

    print("== LIVE G: (F) partition ==")
    seen: set[str] = set()
    ok_groups = True
    for sp in enum:
        _, body = call_json("GET", bysplit(sp))
        ok_groups = ok_groups and all(c["split"] == sp for c in body)
        seen |= {c["comparison_id"] for c in body}
    all_ids = {c["comparison_id"] for c in comps}
    check("G1/F1 the groups over all enum splits partition the FULL M5 "
          "listing exactly (8 + 0 + 0 == 8; every record EXACTLY "
          "ONCE; no record in two groups)",
          ok_groups and seen == all_ids and len(seen) == 8)

    print("== LIVE H: (G) ordering ==")
    ok_order = True
    for sp in enum:
        _, body = call_json("GET", bysplit(sp))
        expected = [c for c in comps if c["split"] == sp]
        ok_order = ok_order and body == expected and \
            [(c["created_at"], c["comparison_id"]) for c in body] == \
            sorted((c["created_at"], c["comparison_id"]) for c in body)
    check("H1/G1 every group preserves the authoritative M5 "
          "(created_at, comparison_id) ASCENDING order exactly",
          ok_order)

    print("== LIVE I: (H) M26/M29/M31 regressions ==")
    ok_ck = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_ck = ok_ck and c == 200 and len(body) == n
    _, cmpd2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-dataset/{DATASET}")
    _, cmpt2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-tokenizer/{TOKENIZER}")
    check("I1/H1 M26/M29/M31 comparison histories unchanged (025e->6 / "
          "30a8->5 / 0511->1 by-checkpoint, 8 by-dataset, 8 "
          "by-tokenizer, parity with the filtered M5 listing)",
          ok_ck and len(cmpd2) == M29_COUNT
          and cmpd2 == [x for x in comps if x["dataset_id"] == DATASET]
          and len(cmpt2) == M31_COUNT and cmpt2 == cmpt)

    print("== LIVE J: (I) M22-M34 + M36 regressions ==")
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
    _, evs2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               "by-split/validation")
    _, evs_trn = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  "by-split/train")
    _, evs_tst = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  "by-split/test")
    check("J2/I1 M24/M28/M30 + M36 evaluation histories unchanged "
          "(3/3/3 by-checkpoint, 16 by-dataset, 16 by-tokenizer, "
          "by-split validation->14 / train->2 / test->0, parity with "
          "the filtered M4 listing)",
          ok_evck and len(evd2) == M28_COUNT
          and evd2 == [x for x in evals if x["dataset_id"] == DATASET]
          and len(evt2) == M30_COUNT and evt2 == evt
          and evs2 == evs_val and len(evs2) == 14
          and evs2 == [x for x in evals if x["split"] == "validation"]
          and len(evs_trn) == 2
          and evs_trn == [x for x in evals if x["split"] == "train"]
          and evs_tst == [])
    ok_smp = True
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_smp = ok_smp and c == 200 and len(body) == n
    _, smpt2 = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                f"by-tokenizer/{TOKENIZER}")
    _, sqt2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                               f"by-tokenizer/{TOKENIZER}")
    check("J3/I1 M27/M32/M33 sample + sample-quality histories "
          "unchanged (4/0/0 by-checkpoint, 4 by-tokenizer samples, 2 "
          "by-tokenizer sample-quality)",
          ok_smp and len(smpt2) == M32_COUNT and smpt2 == smpt
          and len(sqt2) == M33_COUNT and sqt2 == sqt)
    _, gd342 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                f"by-comparison/{KNOWN_COMP}")
    _, gd232 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                f"by-policy/{POLICY}")
    check("J4/I1 M34/M23 gate-decision histories unchanged "
          "(fc379bfcb50f -> 4 by-comparison, m9-live-policy -> 1 "
          "by-policy)",
          gd342 == gd34 and len(gd342) == M34_COUNT
          and gd232 == gd23 and len(gd232) == 1)

    print("== LIVE K: (J) M12/M16/M35 regressions ==")
    _, global_runs2 = call_json(
        "GET", f"{BASE}/workflows/recipes/{KNOWN_RECIPE}/runs")
    _, recipes3 = call_json("GET", f"{BASE}/workflows/recipes")
    c_unk, _ = call_json("GET", f"{BASE}/workflows/recipes/"
                                "ghost-recipe-37")
    ok_zero = all(call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                            f"by-recipe/{r}")[1] == []
                  for r in ZERO_RUN_RECIPES)
    _, byrec2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                                 f"by-recipe/{KNOWN_RECIPE}")
    check("K1/J1 M12/M35 workflow surfaces unchanged (registry 7 "
          "verbatim, GLOBAL m12-live-suite runs 2, by-recipe 2, the 4 "
          "zero-run recipes -> 200 + [], unknown recipe 404)",
          global_runs2 == global_runs and len(global_runs2) == 2
          and recipes3 == recipes and len(recipes3) == 7
          and byrec2 == byrec_runs and len(byrec2) == 2 and ok_zero
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
    check("M1 OpenAPI exactly 69 paths, the new path exactly once "
          "(after the by-tokenizer comparison route, before the "
          "generic comparison route; the M36 evaluations-by-split "
          "route still present exactly once)",
          code_sp2 == 200 and len(spec2["paths"]) == 69
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(TOK_PATH)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(GENERIC_PATH)
          and list(spec2["paths"]).count("/api/v1/models/{model_id}/"
                                         "evaluations/by-split/"
                                         "{split}") == 1)
    _, grouped2 = call_json("GET", bysplit("validation"))
    check("M2 by-split still deterministic at the end", grouped2 == grouped)
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
        print(f"M37 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M37 live smoke OK: narrow read-only MODEL-SCOPED by-split "
          "grouping of the M5 comparison history (distribution "
          "DISCOVERED live: validation->8, train->0, test->0 among 8 "
          "comparisons — exact listing parity, persisted shared-probe "
          "split identity VERBATIM, authoritative order, every record "
          "exactly once, full partition —, schema-enum 422 for "
          "unsupported split values even pre-handler and even for an "
          "unknown model, valid empty 200 + [] for train/test and for "
          "all splits under b5bc905326b6, clean 404 for unknown model "
          "with a valid split, deterministic byte-identical x3 "
          "repeats, M12/M16/M22-M36 surfaces + dashboard hash + "
          "registries + OpenAPI 69 unchanged, ZERO production storage "
          "growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
