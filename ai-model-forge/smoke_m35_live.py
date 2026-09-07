"""M35 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M35 is READ-ONLY: it adds one narrow
MODEL-SCOPED access path, GET /models/{id}/workflows/by-recipe/{recipe_id},
answering "which immutable M11 workflow runs of this model were
executed from this registered recipe?" — the model's authoritative M11
listing filtered by the persisted top-level recipe_id recorded in each
WorkflowRecord (matched VERBATIM, with the recorded recipe_hash
provenance preserved exactly — never inferred from filenames, stage
contents, statuses or the recipe's current definition), after the
recipe is validated through the GLOBAL M12/M14 registry
(RecipeEngine.get; unknown recipe -> 404 — never []). Ad-hoc runs keep
recipe_id=null and belong to NO by-recipe group. Each matching run
appears EXACTLY ONCE. The GLOBAL M12 cross-model
/workflows/recipes/{recipe_id}/runs lineage surface stays unchanged.
The smoke must DISCOVER the authoritative run distribution from the
live M11 listing and recipe registry (not assume it from an old
report), prove exact listing parity, the natural valid-empty cases,
clean 404s, byte-identical repeats (x3), unchanged M2-M34 surfaces +
dashboard hash + registries + OpenAPI 67, and ZERO production storage
growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M11 listing and the M12/M14 recipe registry): model
4a0a871886ef owns 13 workflows in ASCENDING (created_at, workflow_id)
order with persisted recipe grouping None -> 8 (ad-hoc),
m12-live-suite -> 2, m14-comp -> 2, m12-live-ghost -> 1; the recipe
registry holds 7 recipes (m12-live-suite, m12-live-ghost, m14-base,
m14-comp, m14-chain-a/b/c) of which m14-base, m14-chain-a, m14-chain-b,
m14-chain-c have ZERO runs (natural valid-empty cases -> 200 + []);
model b5bc905326b6 has NO workflows; the GLOBAL M12 surface lists 2
runs for m12-live-suite; M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m35-smoke-baseline-inventory.json +
model/checkpoint registries + M6/M11/M16/M18-M34 pre-state +
dashboard + OpenAPI 67 pre-state + the known pair returns 200.

LIVE B (authoritative distribution discovery): from the live M11
listing and recipe registry, discover the per-recipe run distribution
(incl. the ad-hoc null group), the zero-run recipes and the known
recipe; print the values; cross-check against the audited facts.

LIVE C (A — known recipe): the known recipe -> 200 with the EXACT
discovered records; parity with the M11 listing filtered locally;
verbatim detail-getter payloads; recipe_hash provenance == the
registry definition's config_hash; (created_at, workflow_id) ASC.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — valid empty recipe): every discovered zero-run registered
recipe -> 200 + [].

LIVE F (D/E — unknown ids): unknown model -> 404 (even with the real
recipe id); unknown well-formed + malformed recipe -> 404.

LIVE G (F — cross-model isolation): the known recipe under
b5bc905326b6 -> 200 + [] (it has no workflows at all).

LIVE H (G — ad-hoc isolation): runs whose persisted recipe_id is null
appear in NO group; groups over all recipes partition exactly the
recipe-attributed runs.

LIVE I (H — ordering): every recipe group's order equals the
authoritative M11 listing order restricted to that recipe.

LIVE J (I — M12 regression): the GLOBAL /workflows/recipes/{rid}/runs
surface unchanged (exact records incl. model_id; unknown 404) + the
recipe registry list/get verbatim.

LIVE K (J — M34 regression): gate decisions by-comparison unchanged
(fc379bfcb50f -> 4) + by-policy unchanged (m9-live-policy -> 1).

LIVE L (K — M24-M33 regressions): evaluations 3/3/3 by-checkpoint +
16 by-dataset + 16 by-tokenizer; suite runs 10 by-suite + 10
by-checkpoint + summary; comparisons 6/5/1 by-checkpoint + 8
by-dataset + 8 by-tokenizer; samples 4/0/0 by-checkpoint + 4
by-tokenizer; sample-quality 2 by-tokenizer; the generic M6/M11
listings + detail getters verbatim.

LIVE M (L — dashboard; M — registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/tokenizer registries
unchanged.

LIVE N (N — OpenAPI + storage zero drift): OpenAPI exactly 67 paths,
new path once (after the M11 listing route, before the generic
workflow route); every pre-existing file byte-identical, ZERO new
files, zero .tmp, totals unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8756 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8756/api/v1"
SITE = "http://127.0.0.1:8756"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO workflows
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
KNOWN_RECIPE = "m12-live-suite"       # expected most runs (2)
ZERO_RUN_RECIPES = ("m14-base", "m14-chain-a", "m14-chain-b",
                    "m14-chain-c")
POLICY = "m9-live-policy"
SUITE = "m9-live-suite"
KNOWN_COMP = "fc379bfcb50f"           # M34: 4 decisions
DATASET = "ee1a716c4573"
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")
M24_COUNTS = {CK_0511: 3, CK_025E: 3, CK_30A8: 3}
M25_COUNT = 10
M26_COUNTS = {CK_025E: 6, CK_30A8: 5, CK_0511: 1}
M27_COUNTS = {CK_0511: 4, CK_025E: 0, CK_30A8: 0}
M28_COUNT = 16                        # evaluations under ee1a716c4573
M29_COUNT = 8                         # comparisons under ee1a716c4573
M30_COUNT = 16                        # evaluations under 99106e3255c5
M31_COUNT = 8                         # comparisons under 99106e3255c5
M32_COUNT = 4                         # samples under 99106e3255c5
M33_COUNT = 2                         # sample-quality under 99106e3255c5
M34_COUNT = 4                         # gate decisions under fc379bfcb50f
NEW_PATH = "/api/v1/models/{model_id}/workflows/by-recipe/{recipe_id}"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m35-smoke-baseline-inventory.json")

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


def byrec(rid: str, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/workflows/by-recipe/{rid}"


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
    code_wf, workflows = call_json("GET",
                                   f"{BASE}/models/{MODEL}/workflows")
    code_wo, workflows_o = call_json("GET",
                                     f"{BASE}/models/{OTHER_MODEL}/"
                                     "workflows")
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
    code14, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr, probes = call_json("GET", f"{BASE}/probe-suites")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M6/M11/M16/M18-M34 pre-state intact (13 workflows, other "
          "model 0, 7 recipes, 2 global m12-live-suite runs, 4 "
          "by-comparison decisions, 1 by-policy decision, 2 "
          "sample-quality, 10 by-suite, summary 10, 10 by-checkpoint "
          "suite runs, 16/8/4/2 by-tokenizer evals/comparisons/samples/"
          "sample-quality)",
          code_wf == 200 and len(workflows) == 13 and code_wo == 200
          and workflows_o == [] and code_rc == 200
          and len(recipes) == 7 and code_runs == 200
          and len(global_runs) == 2 and code_gr == 200
          and len(gd34) == 4 and code_gp == 200 and len(gd23) == 1
          and code2 == 200 and len(sq) == 2 and code6 == 200
          and len(g) == 10 and code7 == 200 and s["total_count"] == 10
          and code10 == 200 and len(srck) == 10 and code15 == 200
          and len(evt) == 16 and code16 == 200 and len(cmpt) == 8
          and code17 == 200 and len(smpt) == 4 and code18 == 200
          and len(sqt) == 2 and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code14 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 67 paths, the new by-recipe path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 67
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["workflows"])
    code_a, raw_a = call("GET", byrec(KNOWN_RECIPE))
    check("A7 the known pair (4a0a871886ef + m12-live-suite) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    code_rc2, recipes2 = call_json("GET", f"{BASE}/workflows/recipes")
    rids = {r["recipe_id"] for r in recipes2}
    by_recipe: dict[str, list[str]] = {}
    for w in workflows:
        key = w["recipe_id"] if w["recipe_id"] else "<null>"
        by_recipe.setdefault(key, []).append(w["workflow_id"])
    dist_str = ", ".join(f"{k} -> {len(v)}"
                         for k, v in sorted(by_recipe.items()))
    print(f"    discovered: {len(workflows)} workflows of {MODEL} per "
          f"recipe: {dist_str}")
    zero_run = sorted(rids - {k for k in by_recipe if k != "<null>"})
    print(f"    discovered: zero-run registered recipes: {zero_run}")
    check("B1 M11 listing pre-state: 13 workflows in ASCENDING "
          "(created_at, workflow_id) order; distribution None->8, "
          "m12-live-suite->2, m14-comp->2, m12-live-ghost->1; 7 "
          "registered recipes; the 4 zero-run ones; other model has "
          "none",
          code_wf == 200 and len(workflows) == 13
          and [(w["created_at"], w["workflow_id"]) for w in workflows]
          == sorted((w["created_at"], w["workflow_id"])
                    for w in workflows)
          and len(by_recipe.get(KNOWN_RECIPE, [])) == 2
          and len(by_recipe.get("m14-comp", [])) == 2
          and len(by_recipe.get("m12-live-ghost", [])) == 1
          and len(by_recipe.get("<null>", [])) == 8
          and len(rids) == 7
          and zero_run == list(ZERO_RUN_RECIPES)
          and code_wo == 200 and workflows_o == [])
    check("B2 the known recipe is registered and resolvable (and the "
          "zero-run ones too)",
          call_json("GET", f"{BASE}/workflows/recipes/"
                  f"{KNOWN_RECIPE}")[0] == 200
          and all(call_json("GET", f"{BASE}/workflows/recipes/"
                           f"{r}")[0] == 200 for r in ZERO_RUN_RECIPES))

    print("== LIVE C: (A) known recipe — exact response ==")
    code, raw1 = call("GET", byrec(KNOWN_RECIPE))
    grouped = json.loads(raw1)
    filtered = [w for w in workflows
                if w["recipe_id"] == KNOWN_RECIPE]
    ids = [w["workflow_id"] for w in grouped]
    definition = call_json("GET",
                           f"{BASE}/workflows/recipes/{KNOWN_RECIPE}")[1]
    check("C1/A1 response == M11 listing filtered locally by the "
          "persisted recipe identity (no missing / extra / duplicate; "
          "every run EXACTLY ONCE; recipe_hash provenance == the "
          "registry definition's config_hash)",
          code == 200 and grouped == filtered
          and len(ids) == len(set(ids)) == len(filtered) == 2
          and all(w["recipe_hash"] == definition["config_hash"]
                  and w["model_id"] == MODEL
                  and w["recipe_id"] == KNOWN_RECIPE
                  for w in grouped),
          f"{code}/{len(grouped)}")
    ok_detail = True
    for w in grouped:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                                  f"{w['workflow_id']}")
        if c != 200 or one != w:
            ok_detail = False
            check("C2/A1 verbatim payload parity with the M11 detail "
                  "getter", False, w["workflow_id"])
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M11 detail getter "
              "for both records (status, stages, transitions, "
              "result_hash included)", True)

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", byrec(KNOWN_RECIPE))
    _, r2 = call("GET", byrec(KNOWN_RECIPE))
    _, r3 = call("GET", byrec(KNOWN_RECIPE))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw1)

    print("== LIVE E: (C) valid empty recipes ==")
    ok_empty = True
    for rid in zero_run:
        code_e, body = call_json("GET", byrec(rid))
        ok_empty = ok_empty and code_e == 200 and body == []
    check("E1/C1 every discovered zero-run registered recipe -> 200 + [] "
          "(m14-base, m14-chain-a, m14-chain-b, m14-chain-c)", ok_empty)

    print("== LIVE F: (D/E) unknown ids ==")
    c1, _ = call_json("GET", byrec(KNOWN_RECIPE, "no-such-model-35"))
    check("F1/D1 unknown model -> 404 (even with the real recipe id; a "
          "valid recipe never makes an unknown model valid)",
          c1 == 404, f"{c1}")
    c2, _ = call_json("GET", byrec("ghost-recipe-35"))
    check("F2/E1 unknown well-formed recipe id -> 404 (valid recipe "
          "with zero runs is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{byrec('recipe%20id%20with%20spaces!!')}")
    check("F3/E1 malformed unknown recipe id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE G: (F) cross-model isolation ==")
    c4, body = call_json("GET", byrec(KNOWN_RECIPE, OTHER_MODEL))
    check("G1/F1 the known recipe under b5bc905326b6 -> 200 + [] (its "
          "M11 listing is empty; recipes are global, history is "
          "model-scoped)",
          c4 == 200 and body == [] and workflows_o == [], f"{c4}")

    print("== LIVE H: (G) ad-hoc isolation ==")
    non_null_total = sum(len(v) for k, v in by_recipe.items()
                         if k != "<null>")
    ok_groups = True
    seen: set[str] = set()
    for rid in {k for k in by_recipe if k != "<null>"}:
        _, body = call_json("GET", byrec(rid))
        ok_groups = ok_groups and all(
            w["recipe_id"] == rid for w in body)
        seen |= {w["workflow_id"] for w in body}
    null_ids = {w["workflow_id"] for w in workflows
                if w["recipe_id"] is None}
    attributed = {w["workflow_id"] for w in workflows
                  if w["recipe_id"] is not None}
    check("H1/G1 ad-hoc (null recipe_id) runs appear in NO group; the "
          "groups over all recipes partition exactly the "
          "recipe-attributed runs",
          ok_groups and seen == attributed and len(seen)
          == non_null_total and null_ids.isdisjoint(seen))

    print("== LIVE I: (H) ordering ==")
    ok_order = True
    for rid in {k for k in by_recipe if k != "<null>"}:
        _, body = call_json("GET", byrec(rid))
        expected = [w for w in workflows if w["recipe_id"] == rid]
        ok_order = ok_order and body == expected and \
            [(w["created_at"], w["workflow_id"]) for w in body] == \
            sorted((w["created_at"], w["workflow_id"]) for w in body)
    check("I1/H1 every recipe group preserves the authoritative M11 "
          "(created_at, workflow_id) ASCENDING order exactly",
          ok_order)

    print("== LIVE J: (I) M12 regression ==")
    _, global_runs2 = call_json(
        "GET", f"{BASE}/workflows/recipes/{KNOWN_RECIPE}/runs")
    _, recipes3 = call_json("GET", f"{BASE}/workflows/recipes")
    c5, _ = call_json("GET", f"{BASE}/workflows/recipes/ghost-recipe-35")
    check("J1/I1 the GLOBAL M12 recipe-runs surface unchanged (2 runs "
          "for m12-live-suite, same records incl. model_id; unknown "
          "recipe 404) + the recipe registry list verbatim",
          global_runs2 == global_runs and len(global_runs2) == 2
          and all(w["model_id"] == MODEL for w in global_runs2)
          and recipes3 == recipes2 and c5 == 404)

    print("== LIVE K: (J) M34 regression ==")
    _, gd342 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                f"by-comparison/{KNOWN_COMP}")
    _, gd232 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                f"by-policy/{POLICY}")
    check("K1/J1 M34 by-comparison unchanged (fc379bfcb50f -> 4) + M23 "
          "by-policy unchanged (m9-live-policy -> 1)",
          gd342 == gd34 and len(gd342) == 4
          and gd232 == gd23 and len(gd232) == 1)

    print("== LIVE L: (K) M24-M33 regressions ==")
    ok_more = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_more = ok_more and c == 200 and len(body) == n
    _, evd2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-dataset/{DATASET}")
    code_e, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    check("L1/K1 M24/M28 evaluation histories unchanged (3/3/3 "
          "by-checkpoint, 16 by-dataset, parity with the filtered M4 "
          "listing)",
          ok_more and len(evd2) == M28_COUNT
          and evd2 == [x for x in evals if x["dataset_id"] == DATASET])
    ok_cmp = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_cmp = ok_cmp and c == 200 and len(body) == n
    code_c5, comps = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    _, cmpd2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-dataset/{DATASET}")
    check("L2/K1 M26/M29 comparison histories unchanged (025e->6 / "
          "30a8->5 / 0511->1, 8 by-dataset, parity with the filtered "
          "M5 listing)",
          ok_cmp and code_c5 == 200 and len(comps) == 8
          and len(cmpd2) == M29_COUNT
          and cmpd2 == [x for x in comps
                        if x["dataset_id"] == DATASET])
    _, evt2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-tokenizer/{TOKENIZER}")
    _, cmpt2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-tokenizer/{TOKENIZER}")
    check("L3/K1 M30/M31 by-tokenizer histories unchanged (16 "
          "evaluations / 8 comparisons under 99106e3255c5)",
          len(evt2) == M30_COUNT and evt2 == evt
          and len(cmpt2) == M31_COUNT and cmpt2 == cmpt)
    ok_smp = True
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_smp = ok_smp and c == 200 and len(body) == n
    _, smpt2 = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                f"by-tokenizer/{TOKENIZER}")
    check("L4/K1 M27/M32 sample histories unchanged (4/0/0 "
          "by-checkpoint, 4 by-tokenizer)",
          ok_smp and len(smpt2) == M32_COUNT and smpt2 == smpt)
    _, sqt2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                               f"by-tokenizer/{TOKENIZER}")
    check("L5/K1 M33 sample-quality by-tokenizer unchanged (2) + the "
          "M11 listing still deterministic",
          len(sqt2) == M33_COUNT and sqt2 == sqt
          and call_json("GET", f"{BASE}/models/{MODEL}/workflows")
          [1] == workflows)

    print("== LIVE M: (L) dashboard; (M) registries ==")
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("M1/L1 M17 dashboard result_hash + output unchanged",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_t, toks = call_json("GET", f"{BASE}/tokenizers")
    code_tg, tok_get = call_json("GET",
                                 f"{BASE}/tokenizers/{TOKENIZER}")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr2, probes2 = call_json("GET", f"{BASE}/probe-suites")
    check("M2/M1 registries unchanged (M2 tokenizer registry exactly "
          "the ONE entry + get; M9 policy + probe-suite; M12/M14 "
          "recipe registry already verified verbatim in J1)",
          code_t == 200 and len(toks) == 1
          and toks[0]["id"] == TOKENIZER
          and code_tg == 200 and tok_get["id"] == TOKENIZER
          and code_p2 == 200 and pol2 == pol
          and code_pr2 == 200 and probes2 == probes)

    print("== LIVE N: OpenAPI + final storage audit (zero drift) ==")
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    generic = "/api/v1/models/{model_id}/workflows/{workflow_id}"
    listing_path = "/api/v1/models/{model_id}/workflows"
    global_runs_path = ("/api/v1/workflows/recipes/{recipe_id}/runs")
    check("N1 OpenAPI exactly 67 paths, the new path exactly once "
          "(after the M11 listing route, before the generic workflow "
          "route; the GLOBAL M12 runs route still present exactly "
          "once)",
          code_sp2 == 200 and len(spec2["paths"]) == 67
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(listing_path)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(generic)
          and list(spec2["paths"]).count(global_runs_path) == 1)
    _, grouped2 = call_json("GET", byrec(KNOWN_RECIPE))
    check("N2 by-recipe still deterministic at the end",
          grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("N3 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("N4 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("N5 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M35 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M35 live smoke OK: narrow read-only MODEL-SCOPED by-recipe "
          "grouping of the M11 workflow history (distribution "
          "DISCOVERED live: None->8 ad-hoc, m12-live-suite->2, "
          "m14-comp->2, m12-live-ghost->1 among 13 workflows — exact "
          "listing parity, persisted identity + recipe_hash provenance "
          "VERBATIM, authoritative order, every run exactly once, "
          "ad-hoc runs in NO group —, registry-verified valid empty "
          "200 + [] for the 4 zero-run recipes and for b5bc905326b6, "
          "clean 404s incl. cross-model, deterministic byte-identical "
          "x3 repeats, GLOBAL M12 recipe-runs surface + M23-M34 "
          "surfaces + dashboard hash + registries + OpenAPI 67 "
          "unchanged, ZERO production storage growth) verified live "
          "on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
