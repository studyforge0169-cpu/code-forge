"""M29 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M29 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/comparisons/by-dataset/{dataset_id},
answering "which immutable M5 comparisons of this model measured this
dataset?" — the model's authoritative M5 listing filtered by the
persisted shared-probe dataset identity (a comparison persists exactly
ONE top-level dataset_id + dataset_version — both sides measure the
same probe by construction; per-side dataset identities cannot occur),
each matching comparison EXACTLY ONCE (dedup by comparison identity),
versions VERBATIM, after the dataset is validated through the M2
registry (datasets are GLOBAL; model scoping comes from the model's
own M5 listing, exactly like M28). The smoke must DISCOVER the
authoritative comparison/dataset distribution from the live M5 listing
(not assume it from an old report), prove exact listing parity for the
known dataset, same-dataset A=B dedup, version integrity, the natural
model-scoped empty case, clean 404s, byte-identical repeats (x3),
unchanged M2-M28 surfaces + dashboard hash + registries + OpenAPI 61,
and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M2 registry and the M5 listing): exactly ONE dataset
exists, ee1a716c4573 (v1); model 4a0a871886ef owns 8 comparisons, ALL
persisted with the shared probe (dataset_id=ee1a716c4573,
dataset_version=1), in ASCENDING (created_at, comparison_id) order
fc379bfcb50f (A=30a8, B=025e, improved), baa361012e00 (A=30a8, B=025e,
improved), d683f9b81195 (A=current, B=025e, improved), 786de08efe4c
(A=B=025e, unchanged — same-checkpoint A=B), 5c5ff22151ed (A=current,
B=0511, unchanged), d9a62dde016b (A=025e, B=30a8, regressed),
729f9c55ea89 (A=B=30a8, unchanged — same-checkpoint A=B), d62f89e97c85
(A=025e, B=30a8, regressed); sides carry NO dataset fields; model
b5bc905326b6 exists with NO comparisons (natural model-scoped empty
case); M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m29-smoke-baseline-inventory.json +
model/dataset/checkpoint registries + M5 listing + M16/M18-M28
pre-state + dashboard + OpenAPI 61 pre-state.

LIVE B (authoritative comparison discovery): from the live M5 listing,
discover per-dataset counts, ids, side identities (state kinds +
checkpoint ids + whether sides carry dataset fields) and both-side
overlap; print the values; cross-check against the audited facts.

LIVE C (known dataset): 4a0a871886ef + ee1a716c4573 -> 200 with the
EXACT discovered count (8) and ids in the ASCENDING M5 order;
persisted dataset identity on every record; verbatim detail-getter
parity for every record (verdict/losses included).

LIVE D (exact parity): the response equals GET /models/4a0a871886ef/
comparisons filtered locally by persisted dataset identity — no
missing, no extra, no duplicate.

LIVE E (same-dataset A=B): EVERY matching comparison has both sides on
the shared probe dataset; explicitly verify the two same-checkpoint
A=B records (786de08efe4c, 729f9c55ea89) appear EXACTLY ONCE.

LIVE F (version integrity): every returned dataset_version equals the
persisted M5 listing value VERBATIM (all v1; no rewriting).

LIVE G (deterministic repeats): three GETs of the known-dataset
request raw-byte-identical.

LIVE H (valid empty case): b5bc905326b6 (registry-verified: dataset
valid, model's M5 listing empty) + ee1a716c4573 -> 200 + [].

LIVE I (unknown model): -> 404 (even with the real dataset id).

LIVE J (unknown dataset): well-formed and malformed unknown ids -> 404.

LIVE K (cross-model isolation): registry-verify FIRST, then
b5bc905326b6 + ee1a716c4573 -> 200 + []; none of the 8 comparison ids
leaks; the generic comparison detail getter still resolves (no
shadowing); the M26 by-checkpoint route still returns 6/5/1.

LIVE L (regression + final audit): M2 dataset registry +
GET /datasets/{id}, M4 evaluations, M5 list/get, M6 gate decisions,
M20 sample-quality by-checkpoint, M21 by-suite, M22 summary, M23
by-policy, M24 evaluations by-checkpoint (3/3/3), M25 suite-runs
by-checkpoint (10), M26 comparisons by-checkpoint (6/5/1), M27
samples by-checkpoint (4/0/0), M28 evaluations by-dataset (16),
M17 dashboard hash, policies, probe suites, checkpoint registry,
OpenAPI 61 with the new path exactly once — all unchanged; inventory
diff — every pre-existing file byte-identical, ZERO new files, zero
.tmp, zero storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8750 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8750/api/v1"
SITE = "http://127.0.0.1:8750"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO comparisons
DATASET = "ee1a716c4573"              # the ONE production dataset (v1)
COMP_IDS = ("fc379bfcb50f", "baa361012e00", "d683f9b81195",
            "786de08efe4c", "5c5ff22151ed", "d9a62dde016b",
            "729f9c55ea89", "d62f89e97c85")   # ASCENDING (created, id)
AB_SAME_CK = ("786de08efe4c", "729f9c55ea89")  # same-checkpoint A=B
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
UNKNOWN_DS = "ffffffffffff"          # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"              # the live production suite
POLICY = "m9-live-policy"            # the live production policy (1 gate)
SAMPLE = "f8e66f9c7b50"              # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")    # ASCENDING (created, id)
M24_COUNTS = {CK_0511: 3, CK_025E: 3, CK_30A8: 3}
M25_COUNT = 10                        # suite runs under 0511de4c7372
M26_COUNTS = {CK_025E: 6, CK_30A8: 5, CK_0511: 1}
M27_COUNTS = {CK_0511: 4, CK_025E: 0, CK_30A8: 0}
M28_COUNT = 16                        # evaluations over ee1a716c4573
NEW_PATH = "/api/v1/models/{model_id}/comparisons/by-dataset/{dataset_id}"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m29-smoke-baseline-inventory.json")

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


def byds(ds: str, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/comparisons/by-dataset/{ds}"


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
    code13, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr, probes = call_json("GET", f"{BASE}/probe-suites")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M11/M16/M18-M28 pre-state intact (workflows 200, 2 sample "
          "records, 10 by-suite, summary 10, 3 by-checkpoint evals, 1 "
          "gate, 10 by-checkpoint suite runs, 4 by-checkpoint samples, "
          "16 by-dataset evaluations)",
          code_w == 200 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(rec) == 2 and code4 == 200
          and len(bs) == 2 and code5 == 200 and len(bc) == 2
          and code6 == 200 and len(g) == 10 and code7 == 200
          and s["total_count"] == 10 and code8 == 200 and len(evck) == 3
          and code9 == 200 and len(gd) == 1 and code10 == 200
          and len(srck) == 10 and code11 == 200 and len(smp) == 4
          and code12 == 200 and len(evd) == 16 and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code13 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 61 paths, the new by-dataset path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 61
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["comparison"])

    print("== LIVE B: authoritative comparison discovery ==")
    code_ds, datasets = call_json("GET", f"{BASE}/datasets")
    code_dsv, dsv = call_json("GET", f"{BASE}/datasets/{DATASET}")
    code, comps = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    code_oc, comps_o = call_json("GET",
                                 f"{BASE}/models/{OTHER_MODEL}/comparisons")
    dist: dict[tuple, list[str]] = {}
    for x in comps:
        dist.setdefault((x["dataset_id"], x["dataset_version"]),
                        []).append(x["comparison_id"])
    n_ck_sides = sum(1 for x in comps for s in (x["state_a"], x["state_b"])
                     if s["state_kind"] == "checkpoint")
    print(f"    discovered: {len(datasets)} dataset(s): "
          + ", ".join(f"{d['id']} (versions {d['versions']})"
                      for d in datasets))
    print(f"    discovered: {len(comps)} comparisons of {MODEL} over "
          + ", ".join(f"({ds} v{v}) -> {len(ids)}"
                      for (ds, v), ids in sorted(dist.items())))
    print(f"    discovered: {n_ck_sides} checkpoint sides + "
          f"{2 * len(comps) - n_ck_sides} current sides; "
          f"sides carry dataset fields: "
          f"{'dataset_id' in comps[0]['state_a']}")
    check("B1 M2 registry holds exactly the ONE known dataset (v1)",
          code_ds == 200 and len(datasets) == 1
          and datasets[0]["id"] == DATASET
          and datasets[0]["versions"] == [1]
          and code_dsv == 200 and dsv["dataset"]["id"] == DATASET
          and [v["version"] for v in dsv["versions"]] == [1])
    check("B2 M5 listing pre-state: 8 comparisons, ALL persisted with "
          "the shared probe (ee1a716c4573, v1), in ASCENDING "
          "(created_at, comparison_id) order; other model has none; "
          "sides carry NO dataset fields (single shared identity)",
          code == 200 and len(comps) == 8 and code_oc == 200
          and comps_o == []
          and [x["comparison_id"] for x in comps] == list(COMP_IDS)
          and [(x["created_at"], x["comparison_id"]) for x in comps]
          == sorted((x["created_at"], x["comparison_id"]) for x in comps)
          and set(dist) == {(DATASET, 1)}
          and n_ck_sides == 14
          and "dataset_id" not in comps[0]["state_a"])
    check("B3 the two same-checkpoint A=B records exist as audited "
          "(786de08efe4c A=B=025e..., 729f9c55ea89 A=B=30a8...)",
          all(x["state_a"]["checkpoint_id"] == x["state_b"]["checkpoint_id"]
              and x["state_a"]["state_kind"] == "checkpoint"
              for x in comps if x["comparison_id"] in AB_SAME_CK)
          and {x["comparison_id"] for x in comps
               if x["state_a"]["checkpoint_id"]
               == x["state_b"]["checkpoint_id"]} == set(AB_SAME_CK))

    print("== LIVE C: known dataset (8 comparisons) ==")
    url_c = byds(DATASET)
    code, raw1 = call("GET", url_c)
    grouped = json.loads(raw1)
    check("C1 HTTP 200 with exactly the discovered 8 records",
          code == 200 and len(grouped) == 8, f"{code}/{len(grouped)}")
    check("C2 the exact discovered ids in ASCENDING (created_at, "
          "comparison_id) M5 order",
          [x["comparison_id"] for x in grouped] == list(COMP_IDS)
          and [(x["created_at"], x["comparison_id"]) for x in grouped]
          == sorted((x["created_at"], x["comparison_id"]) for x in grouped))
    check("C3 every record's persisted identity: model + dataset id "
          "exactly as requested",
          all(x["model_id"] == MODEL and x["dataset_id"] == DATASET
              for x in grouped))
    for x in grouped:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                  f"{x['comparison_id']}")
        if c != 200 or one != x:
            check("C4 verbatim payload parity with the M5 detail getter",
                  False, x["comparison_id"])
            break
    else:
        check("C4 verbatim payload parity with the M5 detail getter for "
              "all 8 records (verdict/losses included)", True)

    print("== LIVE D: exact parity with the M5 listing ==")
    filtered = [x for x in comps if x["dataset_id"] == DATASET]
    ids = [x["comparison_id"] for x in grouped]
    check("D1 response == M5 listing filtered locally by persisted "
          "dataset identity (no missing / extra / duplicate)",
          grouped == filtered and len(ids) == len(set(ids))
          == len(filtered) == 8)

    print("== LIVE E: same-dataset A=B behavior ==")
    # EVERY matching comparison has both sides on the shared probe
    # dataset (by construction); the two same-checkpoint A=B records
    # must appear EXACTLY ONCE
    check("E1 the two same-checkpoint A=B records appear EXACTLY ONCE "
          "each (dedup by comparison identity, not side combinations)",
          ids.count(AB_SAME_CK[0]) == 1 and ids.count(AB_SAME_CK[1]) == 1
          and len(ids) == len(set(ids)))

    print("== LIVE F: version integrity ==")
    check("F1 every returned dataset_version equals the persisted M5 "
          "listing value VERBATIM (all v1; no rewriting)",
          all(x["dataset_version"] == y["dataset_version"]
              for x, y in zip(grouped, filtered))
          and {x["dataset_version"] for x in grouped} == {1})

    print("== LIVE G: deterministic repeats (x3) ==")
    _, r1 = call("GET", url_c)
    _, r2 = call("GET", url_c)
    _, r3 = call("GET", url_c)
    check("G1 three GETs raw-byte-identical", r1 == r2 == r3 == raw1)

    print("== LIVE H: valid empty case ==")
    ok_pre = code_dsv == 200 and code_oc == 200 and comps_o == []
    c, body = call_json("GET", byds(DATASET, OTHER_MODEL))
    check("H1 registry-verified: dataset valid + b5bc905326b6's M5 "
          "listing empty -> 200 + [] (model-scoped history)",
          ok_pre and c == 200 and body == [], f"{c}")

    print("== LIVE I: unknown model ==")
    c1, _ = call_json("GET", byds(DATASET, "no-such-model-29"))
    check("I1 unknown model -> 404 (even with the real dataset id)",
          c1 == 404, f"{c1}")

    print("== LIVE J: unknown dataset ==")
    c2, _ = call_json("GET", byds(UNKNOWN_DS))
    check("J1 unknown well-formed dataset id -> 404 (valid model, "
          "empty history is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{byds('ds%20id%20with%20spaces!!')}")
    check("J2 malformed unknown dataset id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE K: cross-model isolation + no shadowing ==")
    c4, leak = call_json("GET", byds(DATASET, OTHER_MODEL))
    check("K1 none of the 8 comparison ids leaks into the other model's "
          "response (its history is [])",
          c4 == 200 and leak == []
          and all(cid not in (leak or []) for cid in COMP_IDS))
    c5, one = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                               f"{COMP_IDS[0]}")
    check("K2 by-dataset does not shadow the M5 detail getter "
          "(fc379bfcb50f resolves verbatim)",
          c5 == 200 and one == comps[0])
    c6, _ = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                             "ghost-comp-29")
    check("K3 unknown comparison id via detail getter still 404",
          c6 == 404, f"{c6}")
    ok_m26 = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_m26 = ok_m26 and c == 200 and len(body) == n
    check("K4 M26 by-checkpoint route unchanged (6/5/1)", ok_m26)

    print("== LIVE L: regression + final audit (zero storage growth) ==")
    code, comps2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    check("L1 M5 listing unchanged", code == 200 and comps2 == comps)
    ok_detail = True
    for cid in COMP_IDS:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                  f"{cid}")
        ok_detail = ok_detail and c == 200 \
            and one == comps[COMP_IDS.index(cid)]
    check("L2 M5 detail getter unchanged for all 8 records", ok_detail)
    code_ds2, datasets2 = call_json("GET", f"{BASE}/datasets")
    code_dsv2, dsv2 = call_json("GET", f"{BASE}/datasets/{DATASET}")
    check("L3 M2 dataset registry + GET /datasets/{id} unchanged",
          code_ds2 == 200 and datasets2 == datasets
          and code_dsv2 == 200 and dsv2 == dsv)
    code_ev, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    _, evd2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-dataset/{DATASET}")
    check("L4 M4 evaluations + M28 by-dataset unchanged (16 records, "
          "parity with the filtered listing)",
          code_ev == 200 and len(evals) == 16 and len(evd2) == 16
          and evd2 == [x for x in evals if x["dataset_id"] == DATASET])
    ok_m = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_m = ok_m and c == 200 and len(body) == n
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_m = ok_m and c == 200 and len(body) == n
    check("L5 M24 (3/3/3) / M27 (4/0/0) by-checkpoint surfaces "
          "unchanged", ok_m)
    code2, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code5, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/{CK_0511}")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}/summary")
    code9, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"by-policy/{POLICY}")
    code10, srck2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                     f"by-checkpoint/{CK_0511}")
    check("L6 M6 gates + M16/M18-M20/M21/M22/M23/M25 outputs unchanged",
          code2 == 200 and sq2 == sq and code5 == 200 and bc2 == bc
          and code6 == 200 and g2 == g and code7 == 200 and s2 == s
          and code9 == 200 and gd2 == gd and code10 == 200
          and srck2 == srck and len(srck2) == M25_COUNT)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("L7 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr2, probes2 = call_json("GET", f"{BASE}/probe-suites")
    check("L8 M11 workflow history, M3 checkpoint registry, M9 policy "
          "registry and M9 probe-suite registry unchanged",
          code_w2 == 200 and wfs2 == wfs and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol
          and code_pr2 == 200 and probes2 == probes)
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    generic = "/api/v1/models/{model_id}/comparisons/{comparison_id}"
    m26 = ("/api/v1/models/{model_id}/comparisons/by-checkpoint/"
           "{checkpoint_id}")
    check("L9 OpenAPI still 61 paths with the new path exactly once "
          "(after M26 by-checkpoint, before the generic route)",
          code_sp2 == 200 and len(spec2["paths"]) == 61
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(m26)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(generic))
    _, grouped2 = call_json("GET", url_c)
    check("L10 by-dataset still deterministic at the end",
          grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("L11 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("L12 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("L13 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M29 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M29 live smoke OK: narrow read-only by-dataset grouping of "
          "the M5 comparison history (distribution DISCOVERED live: one "
          "dataset ee1a716c4573 v1, all 8 comparisons over it with the "
          "exact audited ids and side identities — exact listing "
          "parity, same-dataset A=B records EXACTLY ONCE, versions "
          "VERBATIM (all v1) —, registry-verified model-scoped empty "
          "200 + [] for b5bc905326b6, clean 404s, deterministic "
          "byte-identical x3 repeats, M2-M28 surfaces + dashboard hash "
          "+ registries + OpenAPI 61 unchanged, ZERO production "
          "storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
