"""M34 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M34 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/gates/decisions/by-comparison/{comparison_id},
answering "which immutable M6 gate decisions of this model judged this
M5 comparison?" — the model's authoritative M6 listing filtered by the
persisted top-level comparison_id recorded in each GateDecision
(matched VERBATIM — never inferred from filenames, gate-directory
names, checkpoint ids, policy ids or hashes, and never re-derived from
the comparison's current content), after the comparison is validated
through the model's OWN M5 registry (ComparisonEngine.get_comparison;
unknown or other-model comparison -> 404; comparisons are
model-scoped). Legacy direct-evaluation decisions keep
comparison_id=null and belong to NO by-comparison group. Each matching
decision appears EXACTLY ONCE. The smoke must DISCOVER the
authoritative decision distribution from the live M6 listing (not
assume it from an old report), prove exact listing parity, the natural
valid-comparison empty case, clean 404s, byte-identical repeats (x3),
unchanged M2-M33 surfaces + dashboard hash + registries + OpenAPI 66,
and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M6 listing and the M5 comparison registry): model
4a0a871886ef owns 11 gate decisions in ASCENDING (created_at,
decision_id) order 8931835d4af6, baf767bdcbe1, 9d4facca5153,
ef8ba75f9e43, 5b0493c0fbed, 8968151a08bd, a82c95374a5a, e33c99f2f8ca,
0dae7b2c456e, 5c86e4494d2f, 6921d3b29b9d; grouped by persisted
comparison_id: fc379bfcb50f -> 4, d62f89e97c85 -> 2, 786de08efe4c -> 1,
5c5ff22151ed -> 1, d9a62dde016b -> 1, null -> 2; the model's 8 M5
comparisons include baa361012e00, d683f9b81195, 729f9c55ea89 with ZERO
decisions (natural valid-id empty cases -> 200 + []); exactly ONE
policy-attributed decision (m9-live-policy -> 1; the M23 surface);
model b5bc905326b6 has NO decisions; M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m34-smoke-baseline-inventory.json +
model/tokenizer/checkpoint registries + M6/M16/M18-M33 pre-state +
dashboard + OpenAPI 66 pre-state + the known pair returns 200.

LIVE B (authoritative distribution discovery): from the live M6
listing and M5 comparison registry, discover the per-comparison
decision distribution (incl. the null group), the policy attribution
and the zero-decision comparisons; print the values; cross-check
against the audited facts; pick the known comparison (most decisions)
and one empty comparison for the checks below.

LIVE C (A — known comparison): the known comparison -> 200 with the
EXACT discovered records; parity with the M6 listing filtered locally;
verbatim detail-getter payloads; authoritative (created_at,
decision_id) ASCENDING order.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — empty comparison): a valid zero-decision comparison ->
200 + [].

LIVE F (D/E — unknown ids): unknown model -> 404 (even with the real
comparison id); unknown well-formed + malformed comparison -> 404.

LIVE G (F — cross-model isolation): the real comparison id under
b5bc905326b6 -> 404; no decision of one model ever appears under the
other.

LIVE H (G — null ids): decisions whose persisted comparison_id is
null appear in NO group; the groups over all non-null comparisons
partition exactly the non-null decisions.

LIVE I (H — M23 regression): gate-decision-by-policy unchanged
(m9-live-policy -> the exact discovered decision(s)).

LIVE J (I — M24-M33 regressions): evaluations by-checkpoint (3/3/3)
+ by-dataset (16) + by-tokenizer (16); suite runs by-checkpoint (10)
+ by-suite (10/summary); comparisons by-checkpoint (6/5/1) +
by-dataset (8) + by-tokenizer (8); samples by-checkpoint (4/0/0) +
by-tokenizer (4); sample-quality by-tokenizer (2); the generic M5/M6
listings + detail getters verbatim.

LIVE K (J/K — dashboard + registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/workflow/checkpoint
registries unchanged; M2 tokenizer registry unchanged.

LIVE L (L — OpenAPI + storage zero drift): OpenAPI exactly 66 paths,
new path once (after M23 by-policy, before the generic decision
route); every pre-existing file byte-identical, ZERO new files, zero
.tmp, totals unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8755 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8755/api/v1"
SITE = "http://127.0.0.1:8755"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO gate decisions
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
DECISION_IDS = ("8931835d4af6", "baf767bdcbe1", "9d4facca5153",
                "ef8ba75f9e43", "5b0493c0fbed", "8968151a08bd",
                "a82c95374a5a", "e33c99f2f8ca", "0dae7b2c456e",
                "5c86e4494d2f", "6921d3b29b9d")   # ASC (created_at,id)
KNOWN_COMP = "fc379bfcb50f"           # expected most decisions (4)
EMPTY_COMPS = ("baa361012e00", "d683f9b81195", "729f9c55ea89")
POLICY = "m9-live-policy"
SUITE = "m9-live-suite"
SAMPLE = "f8e66f9c7b50"
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
NEW_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-comparison/"
            "{comparison_id}")
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m34-smoke-baseline-inventory.json")

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


def bycomp(cid: str, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/gates/decisions/by-comparison/{cid}"


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
    code_d, decisions = call_json("GET",
                                  f"{BASE}/models/{MODEL}/gates/decisions")
    code_do, decisions_o = call_json(
        "GET", f"{BASE}/models/{OTHER_MODEL}/gates/decisions")
    code_w, wfs = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code6, g = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}")
    code7, s = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}/summary")
    code9, gd = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"by-policy/{POLICY}")
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
    code_rc, recipes = call_json("GET", f"{BASE}/workflows/recipes")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M6/M16/M18-M33 pre-state intact (11 decisions, other "
          "model 0, 2 sample-quality, 10 by-suite, summary 10, 1 "
          "by-policy, 10 by-checkpoint suite runs, 16/8/4/2 "
          "by-tokenizer evals/comparisons/samples/sample-quality)",
          code_d == 200 and len(decisions) == 11 and code_do == 200
          and decisions_o == [] and code_w == 200
          and code2 == 200 and len(sq) == 2 and code6 == 200
          and len(g) == 10 and code7 == 200 and s["total_count"] == 10
          and code9 == 200 and len(gd) == 1 and code10 == 200
          and len(srck) == 10 and code15 == 200 and len(evt) == 16
          and code16 == 200 and len(cmpt) == 8 and code17 == 200
          and len(smpt) == 4 and code18 == 200 and len(sqt) == 2
          and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code14 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 66 paths, the new by-comparison path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 66
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["gates"])
    code_a, raw_a = call("GET", bycomp(KNOWN_COMP))
    check("A7 the known pair (4a0a871886ef + fc379bfcb50f) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    code_t, toks = call_json("GET", f"{BASE}/tokenizers")
    code_c5, comps = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    by_comp: dict[str, list[str]] = {}
    for x in decisions:
        key = x["comparison_id"] if x["comparison_id"] else "<null>"
        by_comp.setdefault(key, []).append(x["decision_id"])
    dist_str = ", ".join(f"{k} -> {len(v)}"
                         for k, v in sorted(by_comp.items()))
    print(f"    discovered: {len(decisions)} decisions of {MODEL} per "
          f"comparison: {dist_str}")
    comp_ids = [c["comparison_id"] for c in comps]
    empty = [c for c in comp_ids
             if c not in {k for k in by_comp if k != "<null>"}]
    print(f"    discovered: zero-decision comparisons: {empty}")
    check("B1 M6 listing pre-state: 11 decisions in ASCENDING "
          "(created_at, decision_id) order; distribution "
          "fc379bfcb50f->4, d62f89e97c85->2, 786de08efe4c->1, "
          "5c5ff22151ed->1, d9a62dde016b->1, null->2; 8 comparisons; "
          "3 zero-decision ones; other model has none",
          code_d == 200
          and [x["decision_id"] for x in decisions] == list(DECISION_IDS)
          and [(x["created_at"], x["decision_id"]) for x in decisions]
          == sorted((x["created_at"], x["decision_id"])
                    for x in decisions)
          and by_comp.get(KNOWN_COMP) is not None
          and len(by_comp[KNOWN_COMP]) == 4
          and len(by_comp.get("d62f89e97c85", [])) == 2
          and len(by_comp.get("786de08efe4c", [])) == 1
          and len(by_comp.get("5c5ff22151ed", [])) == 1
          and len(by_comp.get("d9a62dde016b", [])) == 1
          and len(by_comp.get("<null>", [])) == 2
          and code_c5 == 200 and len(comps) == 8
          and set(empty) == set(EMPTY_COMPS) and code_do == 200
          and decisions_o == [])
    check("B2 the known comparison is registered under the model's M5 "
          "registry and the empty ones too",
          call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                  f"{KNOWN_COMP}")[0] == 200
          and all(call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                           f"{c}")[0] == 200 for c in EMPTY_COMPS))

    print("== LIVE C: (A) known comparison — exact response ==")
    code, raw1 = call("GET", bycomp(KNOWN_COMP))
    grouped = json.loads(raw1)
    filtered = [x for x in decisions
                if x["comparison_id"] == KNOWN_COMP]
    ids = [x["decision_id"] for x in grouped]
    check("C1/A1 response == M6 listing filtered locally by the "
          "persisted comparison identity (no missing / extra / "
          "duplicate; every decision EXACTLY ONCE)",
          code == 200 and grouped == filtered
          and len(ids) == len(set(ids)) == len(filtered) == 4,
          f"{code}/{len(grouped)}")
    ok_detail = True
    for x in grouped:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"{x['decision_id']}")
        if c != 200 or one != x:
            ok_detail = False
            check("C2/A1 verbatim payload parity with the M6 detail "
                  "getter", False, x["decision_id"])
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M6 detail getter "
              "for all 4 records (verdict, decision, losses, delta, "
              "reason included)", True)
    check("C3/A1 authoritative (created_at, decision_id) ASCENDING "
          "order preserved",
          [(x["created_at"], x["decision_id"]) for x in grouped]
          == sorted((x["created_at"], x["decision_id"]) for x in grouped)
          and all(x["comparison_id"] == KNOWN_COMP
                  and x["model_id"] == MODEL for x in grouped))

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", bycomp(KNOWN_COMP))
    _, r2 = call("GET", bycomp(KNOWN_COMP))
    _, r3 = call("GET", bycomp(KNOWN_COMP))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw1)

    print("== LIVE E: (C) empty comparison ==")
    ok_empty = True
    for c in EMPTY_COMPS:
        code_e, body = call_json("GET", bycomp(c))
        ok_empty = ok_empty and code_e == 200 and body == []
    check("E1/C1 valid zero-decision comparisons -> 200 + [] (all 3 "
          "discovered: baa361012e00, d683f9b81195, 729f9c55ea89)",
          ok_empty)

    print("== LIVE F: (D/E) unknown ids ==")
    c1, _ = call_json("GET", bycomp(KNOWN_COMP, "no-such-model-34"))
    check("F1/D1 unknown model -> 404 (even with the real comparison "
          "id)", c1 == 404, f"{c1}")
    c2, _ = call_json("GET", bycomp("ghost-comp-34"))
    check("F2/E1 unknown well-formed comparison id -> 404 (valid "
          "model, empty history is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{bycomp('comp%20id%20with%20spaces!!')}")
    check("F3/E1 malformed unknown comparison id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE G: (F) cross-model isolation ==")
    c4, _ = call_json("GET", bycomp(KNOWN_COMP, OTHER_MODEL))
    check("G1/F1 the real comparison id under b5bc905326b6 -> 404 "
          "(comparisons are model-scoped through the M5 registry); "
          "b5bc905326b6 has zero decisions", c4 == 404
          and decisions_o == [], f"{c4}")

    print("== LIVE H: (G) null comparison ids ==")
    non_null_total = sum(len(v) for k, v in by_comp.items() if k != "<null>")
    ok_groups = True
    seen: set[str] = set()
    for cid in {k for k in by_comp if k != "<null>"}:
        _, body = call_json("GET", bycomp(cid))
        ok_groups = ok_groups and all(
            x["comparison_id"] == cid for x in body)
        seen |= {x["decision_id"] for x in body}
    null_ids = {x["decision_id"] for x in decisions
                if x["comparison_id"] is None}
    non_null_ids = {x["decision_id"] for x in decisions
                    if x["comparison_id"] is not None}
    check("H1/G1 null-comparison decisions appear in NO group; the "
          "groups over all non-null comparisons partition exactly the "
          "non-null decisions",
          ok_groups and seen == non_null_ids and len(seen)
          == non_null_total and null_ids.isdisjoint(seen))

    print("== LIVE I: (H) M23 regression ==")
    _, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                              f"by-policy/{POLICY}")
    check("I1/H1 M23 by-policy unchanged (m9-live-policy -> the exact "
          "discovered decision, parity with the filtered listing)",
          gd2 == gd and len(gd2) == 1
          and gd2 == [x for x in decisions
                      if x.get("policy_id") == POLICY])

    print("== LIVE J: (I) M24-M33 regressions ==")
    ok_more = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_more = ok_more and c == 200 and len(body) == n
    _, evd2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-dataset/{DATASET}")
    code_e, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    check("J1/I1 M24/M28 evaluation histories unchanged (3/3/3 "
          "by-checkpoint, 16 by-dataset, parity with the filtered M4 "
          "listing)",
          ok_more and len(evd2) == M28_COUNT
          and evd2 == [x for x in evals if x["dataset_id"] == DATASET])
    ok_cmp = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_cmp = ok_cmp and c == 200 and len(body) == n
    _, cmpd2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-dataset/{DATASET}")
    check("J2/I1 M26/M29 comparison histories unchanged (025e->6 / "
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
    check("J3/I1 M30/M31 by-tokenizer histories unchanged (16 "
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
    check("J4/I1 M27/M32 sample histories unchanged (4/0/0 "
          "by-checkpoint, 4 by-tokenizer)",
          ok_smp and len(smpt2) == M32_COUNT and smpt2 == smpt)
    _, sqt2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                               f"by-tokenizer/{TOKENIZER}")
    code_sq, sqs = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    check("J5/I1 M33 sample-quality by-tokenizer unchanged (2 under "
          "99106e3255c5) + the M16 listing intact",
          len(sqt2) == M33_COUNT and sqt2 == sqt
          and code_sq == 200 and len(sqs) == 2
          and {x["evaluation_id"] for x in sqs} == set(M16_EVAL_IDS))
    code_d2, decisions2 = call_json("GET",
                                    f"{BASE}/models/{MODEL}/gates/"
                                    "decisions")
    ok_det = True
    for did in DECISION_IDS:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"{did}")
        ok_det = ok_det and c == 200 \
            and one == decisions[DECISION_IDS.index(did)]
    check("J6/I1 M6 generic listing + all 11 detail getters unchanged",
          code_d2 == 200 and decisions2 == decisions and ok_det)

    print("== LIVE K: (J/K) dashboard + registries ==")
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("K1/J1 M17 dashboard result_hash + output unchanged",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_t2, toks2 = call_json("GET", f"{BASE}/tokenizers")
    code_tg, tok_get = call_json("GET",
                                 f"{BASE}/tokenizers/{TOKENIZER}")
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr2, probes2 = call_json("GET", f"{BASE}/probe-suites")
    code_rc2, recipes2 = call_json("GET", f"{BASE}/workflows/recipes")
    check("K2/K1 M2 tokenizer, M11 workflows, M3 checkpoint, M9 "
          "policy/probe-suite, M12/M14 recipe registries unchanged "
          "(tokenizer registry still exactly the ONE entry)",
          code_t2 == 200 and len(toks2) == 1
          and toks2[0]["id"] == TOKENIZER and toks2 == toks
          and code_tg == 200 and tok_get["id"] == TOKENIZER
          and code_w2 == 200 and wfs2 == wfs and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol
          and code_pr2 == 200 and probes2 == probes
          and code_rc2 == 200 and recipes2 == recipes)

    print("== LIVE L: OpenAPI + final storage audit (zero drift) ==")
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    generic = "/api/v1/models/{model_id}/gates/decisions/{decision_id}"
    m23 = ("/api/v1/models/{model_id}/gates/decisions/by-policy/"
           "{policy_id}")
    check("L1 OpenAPI exactly 66 paths, the new path exactly once "
          "(after M23 by-policy, before the generic decision route)",
          code_sp2 == 200 and len(spec2["paths"]) == 66
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(m23)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(generic))
    _, grouped2 = call_json("GET", bycomp(KNOWN_COMP))
    check("L2 by-comparison still deterministic at the end",
          grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("L3 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("L4 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("L5 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M34 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M34 live smoke OK: narrow read-only by-comparison grouping "
          "of the M6 gate-decision history (distribution DISCOVERED "
          "live: fc379bfcb50f->4, d62f89e97c85->2, three->1, "
          "null->2 among 11 decisions — exact listing parity, "
          "persisted identity VERBATIM, authoritative order, every "
          "decision exactly once, null ids in NO group —, "
          "registry-verified valid empty 200 + [] for the 3 "
          "zero-decision comparisons, clean 404s incl. cross-model, "
          "deterministic byte-identical x3 repeats, M23 + M2-M33 "
          "surfaces + dashboard hash + registries + OpenAPI 66 "
          "unchanged, ZERO production storage growth) verified live "
          "on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
