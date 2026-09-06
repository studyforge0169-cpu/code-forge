"""M30 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M30 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/evaluations/by-tokenizer/{tokenizer_id},
answering "which immutable M4 evaluations of this model measured with
this tokenizer?" — the model's authoritative M4 listing filtered by
the persisted tokenizer identity recorded in each EvaluationRecord
(top-level tokenizer_id, matched VERBATIM — never inferred from
filenames or substituted with the latest tokenizer), after the
tokenizer is validated through the existing registry
(TokenizerEngine.load; tokenizers are GLOBAL; model scoping comes from
the model's own M4 listing, exactly like M28/M29). The smoke must
DISCOVER the authoritative tokenizer/evaluation distribution from the
live registry and M4 listing (not assume it from an old report), prove
exact listing parity for the known tokenizer, the natural
model-scoped empty case, clean 404s, byte-identical repeats (x3),
unchanged M2-M29 surfaces + dashboard hash + registries + OpenAPI 62,
and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the tokenizer registry and the M4 listing): exactly ONE
tokenizer exists, 99106e3255c5 (only entry under <root>/tokenizers/);
model 4a0a871886ef owns 16 evaluations, ALL persisted with top-level
tokenizer_id=99106e3255c5 (14 validation + 2 train splits over dataset
ee1a716c4573 v1), in ASCENDING (created_at, eval_id) order
a439eb92f9cd, b0502d871114, 7a16eaa12120, 0cc96a125976, 425003213a0b,
a884bf729ff7, b89a94306ce8, 75835b64d6af, 90aa392b9a2b, 340f5adbf881,
c739c66638e9, c35a1c9902fe, fe7b42cdb411, ebb9b7eccbe3,
0704399fea7b, 75a23351e91c; model b5bc905326b6 exists with NO
evaluations (natural model-scoped empty case: valid global tokenizer +
model with empty history -> 200 + []); M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m30-smoke-baseline-inventory.json +
model/tokenizer/checkpoint registries + M4 listing + M16/M18-M29
pre-state + dashboard + OpenAPI 62 pre-state.

LIVE B (authoritative distribution discovery): from the live tokenizer
registry and M4 listing, discover tokenizer ids, evaluation counts per
tokenizer and the owning model; print the values; cross-check against
the audited facts.

LIVE C (known tokenizer): 4a0a871886ef + 99106e3255c5 -> 200 with the
EXACT discovered count (16) and ids in the ASCENDING M4 order;
persisted tokenizer identity on every record; verbatim detail-getter
parity for every record.

LIVE D (exact parity): the response equals GET /models/4a0a871886ef/
evaluations filtered locally by persisted tokenizer identity — no
missing, no extra, no duplicate.

LIVE E (ordering): (created_at, eval_id) ASCENDING preserved exactly.

LIVE F (deterministic repeats): three GETs raw-byte-identical.

LIVE G (valid tokenizer + second model): b5bc905326b6 +
99106e3255c5 -> the correct model-scoped result ([] — its M4 listing
is empty; registry-verified first).

LIVE H (unknown model): -> 404 (even with the real tokenizer id).

LIVE I (unknown tokenizer): well-formed and malformed unknown ids ->
404.

LIVE J (M2 tokenizer registry unchanged): GET /tokenizers list +
GET /tokenizers/{id} verbatim; unknown id still 404.

LIVE K (M4 generic evaluation listing unchanged).

LIVE L (M24 checkpoint history unchanged: 3/3/3).

LIVE M (M28 dataset history unchanged: 16 under ee1a716c4573).

LIVE N (M29 comparison history unchanged: 8 under ee1a716c4573).

LIVE O (dashboard output/hash unchanged).

LIVE P (policy/probe/suite/recipe registries unchanged).

LIVE Q (OpenAPI exactly 62 paths, new path once, before the generic
evaluation route).

LIVE R (storage zero drift): every pre-existing file byte-identical,
ZERO new files, zero .tmp, zero storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8751 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8751/api/v1"
SITE = "http://127.0.0.1:8751"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO evaluations
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
EVAL_IDS = ("a439eb92f9cd", "b0502d871114", "7a16eaa12120",
            "0cc96a125976", "425003213a0b", "a884bf729ff7",
            "b89a94306ce8", "75835b64d6af", "90aa392b9a2b",
            "340f5adbf881", "c739c66638e9", "c35a1c9902fe",
            "fe7b42cdb411", "ebb9b7eccbe3", "0704399fea7b",
            "75a23351e91c")   # ASCENDING (created_at, eval_id)
DATASET = "ee1a716c4573"
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
UNKNOWN_TOK = "ffffffffffff"        # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"
POLICY = "m9-live-policy"
SAMPLE = "f8e66f9c7b50"
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")
M24_COUNTS = {CK_0511: 3, CK_025E: 3, CK_30A8: 3}
M25_COUNT = 10
M26_COUNTS = {CK_025E: 6, CK_30A8: 5, CK_0511: 1}
M27_COUNTS = {CK_0511: 4, CK_025E: 0, CK_30A8: 0}
M29_COUNT = 8                         # comparisons under ee1a716c4573
NEW_PATH = "/api/v1/models/{model_id}/evaluations/by-tokenizer/{tokenizer_id}"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m30-smoke-baseline-inventory.json")

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


def bytok(tok: str, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/evaluations/by-tokenizer/{tok}"


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
    code_w, wfs = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  "records")
    code4, bs = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-sample/{SAMPLE}")
    code5, bc = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-checkpoint/{CK_0511}")
    code6, g = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}")
    code7, s = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}/summary")
    code8, evck = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{CK_0511}")
    code9, gd = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"by-policy/{POLICY}")
    code10, srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                    f"by-checkpoint/{CK_0511}")
    code11, smp = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{CK_0511}")
    code12, evd = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-dataset/{DATASET}")
    code13, cmpd = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                    f"by-dataset/{DATASET}")
    code14, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr, probes = call_json("GET", f"{BASE}/probe-suites")
    code_rc, recipes = call_json("GET", f"{BASE}/workflows/recipes")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M11/M16/M18-M29 pre-state intact (workflows 200, 2 sample "
          "records, 10 by-suite, summary 10, 3 by-checkpoint evals, 1 "
          "gate, 10 by-checkpoint suite runs, 4 by-checkpoint samples, "
          "16 by-dataset evals, 8 by-dataset comparisons)",
          code_w == 200 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(rec) == 2 and code4 == 200
          and len(bs) == 2 and code5 == 200 and len(bc) == 2
          and code6 == 200 and len(g) == 10 and code7 == 200
          and s["total_count"] == 10 and code8 == 200 and len(evck) == 3
          and code9 == 200 and len(gd) == 1 and code10 == 200
          and len(srck) == 10 and code11 == 200 and len(smp) == 4
          and code12 == 200 and len(evd) == 16 and code13 == 200
          and len(cmpd) == 8 and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code14 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 62 paths, the new by-tokenizer path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 62
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["evaluation"])

    print("== LIVE B: authoritative distribution discovery ==")
    code_t, toks = call_json("GET", f"{BASE}/tokenizers")
    code_tg, tok_get = call_json("GET", f"{BASE}/tokenizers/{TOKENIZER}")
    code, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_oe, evals_o = call_json("GET",
                                 f"{BASE}/models/{OTHER_MODEL}/evaluations")
    dist: dict[str, list[str]] = {}
    for x in evals:
        dist.setdefault(x["tokenizer_id"], []).append(x["eval_id"])
    print(f"    discovered: {len(toks)} tokenizer(s) in the registry: "
          + ", ".join(t["id"] for t in toks))
    print(f"    discovered: {len(evals)} evaluations of {MODEL} per "
          "tokenizer: "
          + ", ".join(f"{tok} -> {len(ids)}"
                      for tok, ids in sorted(dist.items())))
    check("B1 the tokenizer registry holds exactly the ONE known "
          "tokenizer, resolvable via GET /tokenizers/{id}",
          code_t == 200 and len(toks) == 1
          and toks[0]["id"] == TOKENIZER
          and code_tg == 200 and tok_get["id"] == TOKENIZER)
    check("B2 M4 listing pre-state: 16 evaluations, ALL persisted with "
          "tokenizer 99106e3255c5, in ASCENDING (created_at, eval_id) "
          "order; other model has none",
          code == 200 and len(evals) == 16 and code_oe == 200
          and evals_o == []
          and [x["eval_id"] for x in evals] == list(EVAL_IDS)
          and [(x["created_at"], x["eval_id"]) for x in evals]
          == sorted((x["created_at"], x["eval_id"]) for x in evals)
          and set(dist) == {TOKENIZER})

    print("== LIVE C: known tokenizer (16 evaluations) ==")
    url_c = bytok(TOKENIZER)
    code, raw1 = call("GET", url_c)
    grouped = json.loads(raw1)
    check("C1 HTTP 200 with exactly the discovered 16 records",
          code == 200 and len(grouped) == 16, f"{code}/{len(grouped)}")
    check("C2 the exact discovered ids", 
          [x["eval_id"] for x in grouped] == list(EVAL_IDS))
    check("C3 every record's persisted identity: model + tokenizer id "
          "exactly as requested (VERBATIM)",
          all(x["model_id"] == MODEL and x["tokenizer_id"] == TOKENIZER
              for x in grouped))
    for x in grouped:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"{x['eval_id']}")
        if c != 200 or one != x:
            check("C4 verbatim payload parity with the M4 detail getter",
                  False, x["eval_id"])
            break
    else:
        check("C4 verbatim payload parity with the M4 detail getter for "
              "all 16 records (loss/perplexity/state/dataset identity)",
              True)

    print("== LIVE D: exact parity with the M4 listing ==")
    filtered = [x for x in evals if x["tokenizer_id"] == TOKENIZER]
    ids = [x["eval_id"] for x in grouped]
    check("D1 response == M4 listing filtered locally by persisted "
          "tokenizer identity (no missing / extra / duplicate)",
          grouped == filtered and len(ids) == len(set(ids))
          == len(filtered) == 16)

    print("== LIVE E: authoritative ordering preserved ==")
    check("E1 (created_at, eval_id) ASCENDING exactly as the M4 listing",
          [(x["created_at"], x["eval_id"]) for x in grouped]
          == sorted((x["created_at"], x["eval_id"]) for x in grouped))

    print("== LIVE F: deterministic repeats (x3) ==")
    _, r1 = call("GET", url_c)
    _, r2 = call("GET", url_c)
    _, r3 = call("GET", url_c)
    check("F1 three GETs raw-byte-identical", r1 == r2 == r3 == raw1)

    print("== LIVE G: valid tokenizer + second model ==")
    ok_pre = code_tg == 200 and code_oe == 200 and evals_o == []
    c, body = call_json("GET", bytok(TOKENIZER, OTHER_MODEL))
    check("G1 registry-verified: tokenizer valid + b5bc905326b6's M4 "
          "listing empty -> 200 + [] (model-scoped history)",
          ok_pre and c == 200 and body == [], f"{c}")

    print("== LIVE H: unknown model ==")
    c1, _ = call_json("GET", bytok(TOKENIZER, "no-such-model-30"))
    check("H1 unknown model -> 404 (even with the real tokenizer id)",
          c1 == 404, f"{c1}")

    print("== LIVE I: unknown tokenizer ==")
    c2, _ = call_json("GET", bytok(UNKNOWN_TOK))
    check("I1 unknown well-formed tokenizer id -> 404 (valid model, "
          "empty history is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{bytok('tok%20id%20with%20spaces!!')}")
    check("I2 malformed unknown tokenizer id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE J-N + O/P: regression sweep ==")
    code_t2, toks2 = call_json("GET", f"{BASE}/tokenizers")
    code_tg2, tok_get2 = call_json("GET", f"{BASE}/tokenizers/{TOKENIZER}")
    c4, _ = call_json("GET", f"{BASE}/tokenizers/ghost-tok-30")
    check("J1/J2 M2 tokenizer registry unchanged (list + get verbatim; "
          "unknown id still 404)",
          code_t2 == 200 and toks2 == toks and code_tg2 == 200
          and tok_get2 == tok_get and c4 == 404)
    code, evals2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    ok_detail = True
    for eid in EVAL_IDS:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"{eid}")
        ok_detail = ok_detail and c == 200 \
            and one == evals[EVAL_IDS.index(eid)]
    check("K1/K2 M4 generic listing + all 16 detail getters unchanged",
          code == 200 and evals2 == evals and ok_detail)
    ok_m = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_m = ok_m and c == 200 and len(body) == n
    check("L1 M24 checkpoint history unchanged (3/3/3)", ok_m)
    _, evd2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-dataset/{DATASET}")
    check("M1 M28 dataset history unchanged (16, parity with the "
          "filtered listing)",
          len(evd2) == 16
          and evd2 == [x for x in evals if x["dataset_id"] == DATASET])
    code_cmp, comps = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    _, cmpd2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-dataset/{DATASET}")
    check("N1 M29 comparison history unchanged (8, parity with the "
          "filtered M5 listing)",
          code_cmp == 200 and len(comps) == 8 and len(cmpd2) == 8
          and cmpd2 == [x for x in comps if x["dataset_id"] == DATASET])
    ok_more = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_more = ok_more and c == 200 and len(body) == n
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_more = ok_more and c == 200 and len(body) == n
    code2, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}/summary")
    code9, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"by-policy/{POLICY}")
    code10, srck2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                     f"by-checkpoint/{CK_0511}")
    check("N2 M6 gates + M16/M18-M20/M21/M22/M23/M25/M26/M27 outputs "
          "unchanged",
          ok_more and code2 == 200 and sq2 == sq and code6 == 200
          and g2 == g and code7 == 200 and s2 == s and code9 == 200
          and gd2 == gd and code10 == 200 and srck2 == srck
          and len(srck2) == M25_COUNT)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("O1 M17 dashboard result_hash + output unchanged",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr2, probes2 = call_json("GET", f"{BASE}/probe-suites")
    code_rc2, recipes2 = call_json("GET", f"{BASE}/workflows/recipes")
    check("P1 M11 workflows, M3 checkpoint registry, M9 policy registry, "
          "M9 probe-suite registry, M12/M14 recipe registry unchanged",
          code_w2 == 200 and wfs2 == wfs and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol
          and code_pr2 == 200 and probes2 == probes
          and code_rc2 == 200 and recipes2 == recipes)

    print("== LIVE Q: OpenAPI ==")
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    generic = "/api/v1/models/{model_id}/evaluations/{eval_id}"
    m28 = ("/api/v1/models/{model_id}/evaluations/by-dataset/"
           "{dataset_id}")
    check("Q1 OpenAPI exactly 62 paths, the new path exactly once "
          "(after M28 by-dataset, before the generic evaluation route)",
          code_sp2 == 200 and len(spec2["paths"]) == 62
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(m28)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(generic))

    print("== LIVE R: final storage audit (zero drift) ==")
    _, grouped2 = call_json("GET", url_c)
    check("R0 by-tokenizer still deterministic at the end",
          grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("R1 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("R2 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("R3 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M30 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M30 live smoke OK: narrow read-only by-tokenizer grouping "
          "of the M4 evaluation history (distribution DISCOVERED live: "
          "one tokenizer 99106e3255c5, all 16 evaluations with it — "
          "exact listing parity, persisted identity VERBATIM, "
          "authoritative order —, registry-verified model-scoped empty "
          "200 + [] for b5bc905326b6, clean 404s, deterministic "
          "byte-identical x3 repeats, M2-M29 surfaces + dashboard hash "
          "+ registries + OpenAPI 62 unchanged, ZERO production "
          "storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
