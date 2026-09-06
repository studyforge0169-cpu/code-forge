"""M32 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M32 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/samples/by-tokenizer/{tokenizer_id},
answering "which immutable M15 samples of this model were generated
with this tokenizer?" — the model's authoritative M15 listing filtered
by the persisted sample tokenizer identity recorded in each
SampleRecord (top-level tokenizer_id, matched VERBATIM — never
inferred from filenames, paths, checkpoint metadata, prompt text,
generated tokens or hashes, and never substituted with the latest
tokenizer; the paired tokenizer_hash audit field is preserved
verbatim), after the tokenizer is validated through the existing
GLOBAL registry (TokenizerEngine.load; model scoping comes from the
model's own M15 listing, exactly like M27/M30/M31). Each matching
sample appears EXACTLY ONCE. The smoke must DISCOVER the authoritative
tokenizer/sample distribution from the live registry and M15 listing
(not assume it from an old report), prove exact listing parity for the
known tokenizer, the natural model-scoped empty case, clean 404s,
byte-identical repeats (x3), unchanged M2-M31 surfaces + dashboard
hash + registries + OpenAPI 64, and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the tokenizer registry and the M15 listing): exactly ONE
tokenizer exists, 99106e3255c5 (only entry under <root>/tokenizers/);
model 4a0a871886ef owns 4 samples, ALL persisted with tokenizer_id=
99106e3255c5, ALL generated from checkpoint 0511de4c7372 (the other
known checkpoints 025e6d8d8f15 / 30a8bc5b82ab have 0 each — M27), in
ASCENDING (created_at, sample_id) order f8e66f9c7b50, e2b5166fa549,
05820e8bc68a, 4e8463e0df06; model b5bc905326b6 exists with NO samples
(natural model-scoped empty case: valid global tokenizer + model with
empty history -> 200 + []); M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m32-smoke-baseline-inventory.json +
model/tokenizer/checkpoint registries + M15 listing + M16/M18-M31
pre-state + dashboard + OpenAPI 64 pre-state + the known pair returns
200.

LIVE B (authoritative distribution discovery): from the live tokenizer
registry and M15 listing, discover tokenizer ids, sample counts per
tokenizer/checkpoint and the owning model; print the values;
cross-check against the audited facts.

LIVE C (exact parity): the response equals GET /models/4a0a871886ef/
samples filtered locally by persisted tokenizer identity — no missing,
no extra, no duplicate.

LIVE D (persisted identity VERBATIM): every record carries
model 4a0a871886ef + tokenizer_id 99106e3255c5 exactly as persisted,
with verbatim detail-getter parity (prompt/token ids/output text/
strategy/result_hash included).

LIVE E (ordering): (created_at, sample_id) ASCENDING preserved exactly
as the M15 listing.

LIVE F (deterministic repeats): three GETs raw-byte-identical.

LIVE G (valid tokenizer + second model): b5bc905326b6 +
99106e3255c5 -> the correct model-scoped result ([] — its M15 listing
is empty; registry-verified first).

LIVE H (unknown model): -> 404 (even with the real tokenizer id).

LIVE I (unknown tokenizer): well-formed and malformed unknown ids ->
404.

LIVE J (M2 tokenizer registry unchanged): GET /tokenizers list +
GET /tokenizers/{id} verbatim; unknown id still 404.

LIVE K (M15 generic sample listing unchanged): 4 records, all detail
getters verbatim.

LIVE L (M27 sample-by-checkpoint history unchanged: 0511->4 /
025e->0 / 30a8->0).

LIVE M (M30 evaluation-by-tokenizer history unchanged: 16 under
99106e3255c5).

LIVE N (M31 comparison-by-tokenizer history unchanged: 8 under
99106e3255c5).

LIVE O (M24/M28 evaluation histories unchanged: 3/3/3 by-checkpoint,
16 by-dataset).

LIVE P (M26/M29 comparison histories unchanged: 025e->6 / 30a8->5 /
0511->1 by-checkpoint, 8 by-dataset).

LIVE Q (dashboard output/hash unchanged).

LIVE R (policy/probe/suite/recipe/workflow/checkpoint registries
unchanged).

LIVE S (OpenAPI exactly 64 paths, new path once, after M27
by-checkpoint, before the generic sample route).

LIVE T (storage zero drift): every pre-existing file byte-identical,
ZERO new files, zero .tmp, zero storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8753 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8753/api/v1"
SITE = "http://127.0.0.1:8753"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO samples
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
SAMPLE_IDS = ("f8e66f9c7b50", "e2b5166fa549", "05820e8bc68a",
              "4e8463e0df06")         # ASC (created_at, sample_id)
DATASET = "ee1a716c4573"
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
UNKNOWN_TOK = "ffffffffffff"        # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"
POLICY = "m9-live-policy"
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
NEW_PATH = "/api/v1/models/{model_id}/samples/by-tokenizer/{tokenizer_id}"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m32-smoke-baseline-inventory.json")

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
    return f"{BASE}/models/{model}/samples/by-tokenizer/{tok}"


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
                                 f"by-sample/{SAMPLE_IDS[0]}")
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
    code15, evt = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-tokenizer/{TOKENIZER}")
    code16, cmpt = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                    f"by-tokenizer/{TOKENIZER}")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr, probes = call_json("GET", f"{BASE}/probe-suites")
    code_rc, recipes = call_json("GET", f"{BASE}/workflows/recipes")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M11/M16/M18-M31 pre-state intact (workflows 200, 2 sample "
          "records, 10 by-suite, summary 10, 3 by-checkpoint evals, 1 "
          "gate, 10 by-checkpoint suite runs, 4 by-checkpoint samples, "
          "16 by-dataset evals, 8 by-dataset comparisons, 16 "
          "by-tokenizer evaluations, 8 by-tokenizer comparisons)",
          code_w == 200 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(rec) == 2 and code4 == 200
          and len(bs) == 2 and code5 == 200 and len(bc) == 2
          and code6 == 200 and len(g) == 10 and code7 == 200
          and s["total_count"] == 10 and code8 == 200 and len(evck) == 3
          and code9 == 200 and len(gd) == 1 and code10 == 200
          and len(srck) == 10 and code11 == 200 and len(smp) == 4
          and code12 == 200 and len(evd) == 16 and code13 == 200
          and len(cmpd) == 8 and code15 == 200 and len(evt) == 16
          and code16 == 200 and len(cmpt) == 8 and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code14 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 64 paths, the new by-tokenizer path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 64
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["sampling"])
    code_a, raw_a = call("GET", bytok(TOKENIZER))
    check("A7 the known pair (4a0a871886ef + 99106e3255c5) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    code_t, toks = call_json("GET", f"{BASE}/tokenizers")
    code_tg, tok_get = call_json("GET", f"{BASE}/tokenizers/{TOKENIZER}")
    code, samples = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code_os, samples_o = call_json("GET",
                                   f"{BASE}/models/{OTHER_MODEL}/samples")
    by_tok: dict[str, list[str]] = {}
    by_ck: dict[str, int] = {}
    for x in samples:
        by_tok.setdefault(x["tokenizer_id"], []).append(x["sample_id"])
        by_ck[x["checkpoint_id"]] = by_ck.get(x["checkpoint_id"], 0) + 1
    print(f"    discovered: {len(toks)} tokenizer(s) in the registry: "
          + ", ".join(t["id"] for t in toks))
    print(f"    discovered: {len(samples)} samples of {MODEL} per "
          "tokenizer: "
          + ", ".join(f"{tok} -> {len(ids)}"
                      for tok, ids in sorted(by_tok.items())))
    print(f"    discovered: samples per checkpoint: {by_ck}")
    check("B1 the tokenizer registry holds exactly the ONE known "
          "tokenizer, resolvable via GET /tokenizers/{id}",
          code_t == 200 and len(toks) == 1
          and toks[0]["id"] == TOKENIZER
          and code_tg == 200 and tok_get["id"] == TOKENIZER)
    check("B2 M15 listing pre-state: 4 samples, ALL persisted with "
          "tokenizer 99106e3255c5, ALL from checkpoint 0511de4c7372, "
          "in ASCENDING (created_at, sample_id) order; other model "
          "has none",
          code == 200 and len(samples) == 4 and code_os == 200
          and samples_o == []
          and [x["sample_id"] for x in samples] == list(SAMPLE_IDS)
          and [(x["created_at"], x["sample_id"]) for x in samples]
          == sorted((x["created_at"], x["sample_id"]) for x in samples)
          and set(by_tok) == {TOKENIZER} and by_ck == {CK_0511: 4})

    print("== LIVE C: exact parity with the M15 listing ==")
    code, raw1 = call("GET", bytok(TOKENIZER))
    grouped = json.loads(raw1)
    filtered = [x for x in samples if x["tokenizer_id"] == TOKENIZER]
    ids = [x["sample_id"] for x in grouped]
    check("C1 response == M15 listing filtered locally by the persisted "
          "sample tokenizer identity (no missing / extra / duplicate; "
          "every sample EXACTLY ONCE)",
          code == 200 and grouped == filtered
          and len(ids) == len(set(ids)) == len(filtered) == 4,
          f"{code}/{len(grouped)}")

    print("== LIVE D: persisted identity VERBATIM ==")
    check("D1 every record's persisted identity: model + tokenizer id "
          "exactly as persisted (no substitution/rewriting)",
          all(x["model_id"] == MODEL and x["tokenizer_id"] == TOKENIZER
              for x in grouped))
    ok_detail = True
    for x in grouped:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                  f"{x['sample_id']}")
        if c != 200 or one != x:
            ok_detail = False
            check("D2 verbatim payload parity with the M15 detail getter",
                  False, x["sample_id"])
            break
    if ok_detail:
        check("D2 verbatim payload parity with the M15 detail getter for "
              "all 4 records (prompt, token ids, output text, strategy, "
              "result_hash included)", True)

    print("== LIVE E: authoritative ordering preserved ==")
    check("E1 (created_at, sample_id) ASCENDING exactly as the M15 "
          "listing",
          [(x["created_at"], x["sample_id"]) for x in grouped]
          == sorted((x["created_at"], x["sample_id"]) for x in samples
                    if x["tokenizer_id"] == TOKENIZER))

    print("== LIVE F: deterministic repeats (x3) ==")
    _, r1 = call("GET", bytok(TOKENIZER))
    _, r2 = call("GET", bytok(TOKENIZER))
    _, r3 = call("GET", bytok(TOKENIZER))
    check("F1 three GETs raw-byte-identical", r1 == r2 == r3 == raw1)

    print("== LIVE G: valid tokenizer + second model ==")
    ok_pre = code_tg == 200 and code_os == 200 and samples_o == []
    c, body = call_json("GET", bytok(TOKENIZER, OTHER_MODEL))
    check("G1 registry-verified: tokenizer valid + b5bc905326b6's M15 "
          "listing empty -> 200 + [] (model-scoped history)",
          ok_pre and c == 200 and body == [], f"{c}")

    print("== LIVE H: unknown model ==")
    c1, _ = call_json("GET", bytok(TOKENIZER, "no-such-model-32"))
    check("H1 unknown model -> 404 (even with the real tokenizer id)",
          c1 == 404, f"{c1}")

    print("== LIVE I: unknown tokenizer ==")
    c2, _ = call_json("GET", bytok(UNKNOWN_TOK))
    check("I1 unknown well-formed tokenizer id -> 404 (valid model, "
          "empty history is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{bytok('tok%20id%20with%20spaces!!')}")
    check("I2 malformed unknown tokenizer id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE J-N + O/P/Q/R: regression sweep ==")
    code_t2, toks2 = call_json("GET", f"{BASE}/tokenizers")
    code_tg2, tok_get2 = call_json("GET", f"{BASE}/tokenizers/{TOKENIZER}")
    c4, _ = call_json("GET", f"{BASE}/tokenizers/ghost-tok-32")
    check("J1/J2 M2 tokenizer registry unchanged (list + get verbatim; "
          "unknown id still 404)",
          code_t2 == 200 and toks2 == toks and code_tg2 == 200
          and tok_get2 == tok_get and c4 == 404)
    code, samples2 = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    ok_detail = True
    for sid in SAMPLE_IDS:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/samples/{sid}")
        ok_detail = ok_detail and c == 200 \
            and one == samples[SAMPLE_IDS.index(sid)]
    check("K1/K2 M15 generic listing + all 4 detail getters unchanged",
          code == 200 and samples2 == samples and ok_detail)
    ok_m = True
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_m = ok_m and c == 200 and len(body) == n
    check("L1 M27 sample-by-checkpoint history unchanged "
          "(0511->4 / 025e->0 / 30a8->0)", ok_m)
    _, evt2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-tokenizer/{TOKENIZER}")
    check("M1 M30 evaluation-by-tokenizer history unchanged (16 under "
          "99106e3255c5)", len(evt2) == M30_COUNT and evt2 == evt)
    _, cmpt2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                f"by-tokenizer/{TOKENIZER}")
    check("N1 M31 comparison-by-tokenizer history unchanged (8 under "
          "99106e3255c5)", len(cmpt2) == M31_COUNT and cmpt2 == cmpt)

    ok_more = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_more = ok_more and c == 200 and len(body) == n
    _, evd2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-dataset/{DATASET}")
    code_e, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    check("O1 M24/M28 evaluation histories unchanged (M24 3/3/3 "
          "by-checkpoint, M28 16 by-dataset, parity with the filtered "
          "M4 listing)",
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
    check("P1 M26/M29 comparison histories unchanged (M26 025e->6 / "
          "30a8->5 / 0511->1, M29 8 by-dataset, parity with the "
          "filtered M5 listing)",
          ok_cmp and code_c5 == 200 and len(comps) == 8
          and len(cmpd2) == 8
          and cmpd2 == [x for x in comps if x["dataset_id"] == DATASET])
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("Q1 M17 dashboard result_hash + output unchanged",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr2, probes2 = call_json("GET", f"{BASE}/probe-suites")
    code_rc2, recipes2 = call_json("GET", f"{BASE}/workflows/recipes")
    check("R1 M11 workflows, M3 checkpoint registry, M9 policy registry, "
          "M9 probe-suite registry, M12/M14 recipe registry unchanged",
          code_w2 == 200 and wfs2 == wfs and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol
          and code_pr2 == 200 and probes2 == probes
          and code_rc2 == 200 and recipes2 == recipes)

    print("== LIVE S: OpenAPI ==")
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    generic = "/api/v1/models/{model_id}/samples/{sample_id}"
    m27 = ("/api/v1/models/{model_id}/samples/by-checkpoint/"
           "{checkpoint_id}")
    check("S1 OpenAPI exactly 64 paths, the new path exactly once "
          "(after M27 by-checkpoint, before the generic sample route)",
          code_sp2 == 200 and len(spec2["paths"]) == 64
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(m27)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(generic))

    print("== LIVE T: final storage audit (zero drift) ==")
    _, grouped2 = call_json("GET", bytok(TOKENIZER))
    check("T0 by-tokenizer still deterministic at the end",
          grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("T1 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("T2 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("T3 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M32 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M32 live smoke OK: narrow read-only by-tokenizer grouping "
          "of the M15 sample history (distribution DISCOVERED live: "
          "one tokenizer 99106e3255c5, all 4 samples of 4a0a871886ef "
          "with it, all from checkpoint 0511de4c7372 — exact listing "
          "parity, persisted identity VERBATIM, authoritative order, "
          "every sample exactly once —, registry-verified model-scoped "
          "empty 200 + [] for b5bc905326b6, clean 404s, deterministic "
          "byte-identical x3 repeats, M2-M31 surfaces + dashboard hash "
          "+ registries + OpenAPI 64 unchanged, ZERO production "
          "storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
