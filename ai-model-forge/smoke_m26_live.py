"""M26 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M26 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/comparisons/by-checkpoint/{checkpoint_id},
answering "which immutable M5 comparisons involve this checkpoint state
on EITHER side?" — the model's authoritative M5 listing filtered by the
persisted side states recorded in each ComparisonRecord (state_a/state_b
with state_kind="checkpoint" + checkpoint_id; TWO-SIDED semantics: a
checkpoint_id-looking value on a current-state side never matches),
after checkpoint ownership is validated through the M3 checkpoint
registry (the same model-scoped path M20/M24/M25 use). The smoke must
prove exact M5-listing parity for all three live checkpoints (6/5/1
records), A=B dedup (each same-checkpoint comparison appears EXACTLY
ONCE under its checkpoint), current-side exclusion (current-vs-
checkpoint records appear only under their checkpoint side),
overlapping membership (4 records legitimately under two checkpoints,
once each — the union is 8 unique comparisons, never 12 slots), clean
404s (unknown model / unknown checkpoint / cross-model), byte-identical
repeats, unchanged M5/M6/M16-M25 surfaces + dashboard hash + registries
+ OpenAPI 58, and ZERO production storage growth.

Production facts (audited from the manifests at milestone start; the
authoritative source is the live M5 listing — the smoke re-derives
every expectation from it): model 4a0a871886ef owns 3 checkpoints
(025e6d8d8f15, 0511de4c7372, 30a8bc5b82ab) and 8 comparisons in
ASCENDING (created_at, comparison_id) order fc379bfcb50f, baa361012e00,
d683f9b81195, 786de08efe4c, 5c5ff22151ed, d9a62dde016b, 729f9c55ea89,
d62f89e97c85. Membership on EITHER side: 025e6d8d8f15 -> 6 (fc379,
baa36, d683f, 786de, d9a62, d62f8; two of them A=B/current cases), 30a8
-> 5 (fc379, baa36, d9a62, 729f, d62f8), 0511 -> 1 (5c5ff). Same-
checkpoint A=B records: 786de08efe4c (A=B=025e), 729f9c55ea89 (A=B=
30a8). Current-vs-checkpoint records: d683f9b81195 (A=current, B=025e),
5c5ff22151ed (A=current, B=0511). Four records involve BOTH 025e+30a8.
Model b5bc905326b6 exists with NO checkpoints and NO comparisons
(cross-model probe); M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp. NOTE (honesty): the M26 prompt's
stated per-checkpoint counts were transposed (0511<->30a8); these
values come from the authoritative manifests/listing per the prompt's
own "do not invent ids" rule.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file SHA256
inventory saved to /tmp/m26-smoke-baseline-inventory.json + model,
checkpoint & comparison registries + M5 listing / M6 gates / M16-M25 /
dashboard / OpenAPI pre-state.

LIVE B (checkpoint 025e6d8d8f15): by-checkpoint -> 200 with EXACTLY 6
records — the exact known ids in the ASCENDING M5 order — verified
against the authoritative M5 listing filtered by the persisted sides;
the A=B record 786de08efe4c appears EXACTLY ONCE; verbatim payload
parity with the M5 detail getter.

LIVE C (checkpoint 30a8bc5b82ab): -> 200 with EXACTLY 5 records; the
A=B record 729f9c55ea89 appears EXACTLY ONCE; listing parity.

LIVE D (checkpoint 0511de4c7372): -> 200 with EXACTLY 1 record
(5c5ff22151ed — included via its checkpoint side B only).

LIVE E (dedup + partition): 6 + 5 + 1 = 12 membership slots over 8
UNIQUE comparisons; the union equals the full M5 listing; the 4
double-membership records appear once under EACH checkpoint; the two
A=B records appear exactly once under their own checkpoint and under
no other.

LIVE F (current-state exclusion): the two current-vs-checkpoint records
appear ONLY under their checkpoint side's history and never under the
other checkpoints; no production comparison has both sides current
(and none could ever match a by-checkpoint query).

LIVE G (deterministic repeat): repeated GETs raw-byte-identical for all
three checkpoints.

LIVE H (unknown model): -> 404.

LIVE I (unknown checkpoint): well-formed and malformed unknown ids ->
404.

LIVE J (cross-model isolation + no shadowing): b5bc905326b6 +
0511de4c7372 (owned by 4a0a871886ef) -> 404 via the model-scoped M3
registry (b5bc905326b6 has no checkpoints — only registry-valid ids
probed, no guessed statuses); the other model's comparison history is
[] (no comparison id leaks); by-checkpoint does not shadow
GET /comparisons/{comparison_id} (detail getter works, ghost id 404s).

LIVE K (regression + final audit): M5 list/get, M6 gate decisions, M16
sample-quality, M20 by-checkpoint sample-quality, M21 by-suite, M22
summary, M23 by-policy, M24 evaluations by-checkpoint, M25 suite-runs
by-checkpoint, dashboard hash, policy/checkpoint registries, OpenAPI 58
with the new path exactly once — all unchanged; inventory diff — every
pre-existing file byte-identical, ZERO new files, zero .tmp, zero
storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8747 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8747/api/v1"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO checkpoints/comparisons
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
CK_0511 = "0511de4c7372"
CKPTS = (CK_025E, CK_0511, CK_30A8)   # registry order for the model
COMP_IDS = ("fc379bfcb50f", "baa361012e00", "d683f9b81195",
            "786de08efe4c", "5c5ff22151ed", "d9a62dde016b",
            "729f9c55ea89", "d62f89e97c85")   # ASCENDING (created_at, id)
MEMBERS = {                            # expected per-checkpoint membership
    CK_025E: ("fc379bfcb50f", "baa361012e00", "d683f9b81195",
              "786de08efe4c", "d9a62dde016b", "d62f89e97c85"),
    CK_30A8: ("fc379bfcb50f", "baa361012e00", "d9a62dde016b",
              "729f9c55ea89", "d62f89e97c85"),
    CK_0511: ("5c5ff22151ed",),
}
AB_SAME = {"786de08efe4c": CK_025E, "729f9c55ea89": CK_30A8}
CURRENT_VS_CK = {"d683f9b81195": CK_025E, "5c5ff22151ed": CK_0511}
UNKNOWN_CK = "ffffffffffff"          # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"              # the live production suite
POLICY = "m9-live-policy"            # the live production policy (1 gate)
SAMPLE = "f8e66f9c7b50"              # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")    # ASCENDING (created, id)
NEW_PATH = ("/api/v1/models/{model_id}/comparisons/by-checkpoint/"
            "{checkpoint_id}")
SITE = "http://127.0.0.1:8747"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m26-smoke-baseline-inventory.json")

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


def byck(ck: str, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/comparisons/by-checkpoint/{ck}"


def expected_from_listing(listing, ck: str) -> list[dict]:
    return [c for c in listing
            if any(s["state_kind"] == "checkpoint"
                   and s["checkpoint_id"] == ck
                   for s in (c["state_a"], c["state_b"]))]


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
          and {c["checkpoint_id"] for c in cks} == set(CKPTS)
          and call_json("GET", f"{BASE}/models/{OTHER_MODEL}/checkpoints")
          [1] == [])
    code, comps = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    code_o, comps_o = call_json("GET",
                                f"{BASE}/models/{OTHER_MODEL}/comparisons")
    n_ck_sides = sum(1 for c in comps for s in (c["state_a"],
                                                c["state_b"])
                     if s["state_kind"] == "checkpoint")
    n_cur_sides = sum(1 for c in comps for s in (c["state_a"],
                                                 c["state_b"])
                      if s["state_kind"] == "current")
    by_cid = {c["comparison_id"]: c for c in comps}
    check("A4 M5 listing pre-state: exactly the 8 known comparisons in "
          "ASCENDING (created_at, comparison_id) order; other model "
          "has none",
          code == 200 and len(comps) == 8 and code_o == 200
          and comps_o == []
          and [c["comparison_id"] for c in comps] == list(COMP_IDS)
          and [(c["created_at"], c["comparison_id"]) for c in comps]
          == sorted((c["created_at"], c["comparison_id"]) for c in comps))
    check("A5 persisted side facts: 14 checkpoint sides + 2 current "
          "sides (the two current-vs-checkpoint records); the two A=B "
          "records exactly as audited; no both-sides-current record",
          n_ck_sides == 14 and n_cur_sides == 2
          and {cid: (by_cid[cid]["state_a"]["checkpoint_id"],
                     by_cid[cid]["state_b"]["checkpoint_id"])
               for cid in AB_SAME}
          == {cid: (ck, ck) for cid, ck in AB_SAME.items()}
          and {cid: (by_cid[cid]["state_a"]["checkpoint_id"],
                     by_cid[cid]["state_b"]["checkpoint_id"])
               for cid in CURRENT_VS_CK}
          == {cid: (None, ck) for cid, ck in CURRENT_VS_CK.items()}
          and not any(c["state_a"]["state_kind"] == "current"
                      and c["state_b"]["state_kind"] == "current"
                      for c in comps))
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
    code8, ev = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                 f"by-checkpoint/{CK_0511}")
    code9, gd = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"by-policy/{POLICY}")
    code10, srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                    f"by-checkpoint/{CK_0511}")
    code11, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("A6 M11/M16/M18-M25 pre-state intact (workflows 200, 2 sample "
          "records, 10 by-suite, summary 10, 3 checkpoint evals, 1 "
          "gate, 10 by-checkpoint suite runs)",
          code_w == 200 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(rec) == 2 and code4 == 200
          and len(bs) == 2 and code5 == 200 and len(bc) == 2
          and code6 == 200 and len(g) == 10 and code7 == 200
          and s["total_count"] == 10 and code8 == 200 and len(ev) == 3
          and code9 == 200 and len(gd) == 1 and code10 == 200
          and len(srck) == 10 and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A7 M17 dashboard hash equals the full known value",
          code11 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A8 OpenAPI pre-state: 58 paths, the new by-checkpoint path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 58
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["comparison"])

    grouped_pre = {}
    print("== LIVE B: checkpoint 025e6d8d8f15 (6 comparisons) ==")
    url_b = byck(CK_025E)
    code, raw1 = call("GET", url_b)
    grouped_pre[CK_025E] = json.loads(raw1)
    check("B1 HTTP 200 with exactly 6 records",
          code == 200 and len(grouped_pre[CK_025E]) == 6,
          f"{code}/{len(grouped_pre[CK_025E])}")
    check("B2 the exact audited ids in ASCENDING (created_at, "
          "comparison_id) M5 order",
          [x["comparison_id"] for x in grouped_pre[CK_025E]]
          == list(MEMBERS[CK_025E])
          and [(x["created_at"], x["comparison_id"])
               for x in grouped_pre[CK_025E]]
          == sorted((x["created_at"], x["comparison_id"])
                    for x in grouped_pre[CK_025E]))
    check("B3 every record involves 025e6d8d8f15 on >= 1 side "
          "(persisted identity)",
          all(x["model_id"] == MODEL
              and any(s["state_kind"] == "checkpoint"
                      and s["checkpoint_id"] == CK_025E
                      for s in (x["state_a"], x["state_b"]))
              for x in grouped_pre[CK_025E]))
    check("B4 parity with the M5 listing filtered by the persisted "
          "sides", grouped_pre[CK_025E] == expected_from_listing(comps,
                                                                 CK_025E))
    check("B5 the same-checkpoint A=B record 786de08efe4c appears "
          "EXACTLY ONCE (dedup by comparison identity, not side "
          "combos)",
          [x["comparison_id"] for x in grouped_pre[CK_025E]]
          .count("786de08efe4c") == 1)
    for x in grouped_pre[CK_025E]:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                  f"{x['comparison_id']}")
        if c != 200 or one != x:
            check("B6 verbatim payload parity with the M5 detail getter",
                  False, x["comparison_id"])
            break
    else:
        check("B6 verbatim payload parity with the M5 detail getter for "
              "all 6 records (verdict/losses included)", True)
    check("B7 verbatim ComparisonRecord payloads (same field set as the "
          "M5 listing)",
          all(set(x) == set(comps[0]) and x["verdict"] in
              ("improved", "regressed", "unchanged")
              and len(x["result_hash"]) == 64
              for x in grouped_pre[CK_025E]))

    print("== LIVE C: checkpoint 30a8bc5b82ab (5 comparisons) ==")
    code, raw_c = call("GET", byck(CK_30A8))
    grouped_pre[CK_30A8] = json.loads(raw_c)
    check("C1 HTTP 200 with exactly 5 records", code == 200
          and len(grouped_pre[CK_30A8]) == 5,
          f"{code}/{len(grouped_pre[CK_30A8])}")
    check("C2 the exact audited ids in the ASCENDING M5 order",
          [x["comparison_id"] for x in grouped_pre[CK_30A8]]
          == list(MEMBERS[CK_30A8]))
    check("C3 parity with the filtered M5 listing; every record "
          "involves 30a8bc5b82ab on >= 1 side",
          grouped_pre[CK_30A8] == expected_from_listing(comps, CK_30A8)
          and all(x["model_id"] == MODEL
                  and any(s["state_kind"] == "checkpoint"
                          and s["checkpoint_id"] == CK_30A8
                          for s in (x["state_a"], x["state_b"]))
                  for x in grouped_pre[CK_30A8]))
    check("C4 the same-checkpoint A=B record 729f9c55ea89 appears "
          "EXACTLY ONCE",
          [x["comparison_id"] for x in grouped_pre[CK_30A8]]
          .count("729f9c55ea89") == 1)

    print("== LIVE D: checkpoint 0511de4c7372 (1 comparison) ==")
    code, raw_d = call("GET", byck(CK_0511))
    grouped_pre[CK_0511] = json.loads(raw_d)
    check("D1 HTTP 200 with exactly 1 record (5c5ff22151ed)", code == 200
          and [x["comparison_id"] for x in grouped_pre[CK_0511]]
          == ["5c5ff22151ed"], f"{code}")
    check("D2 that record is current-vs-checkpoint and is included via "
          "its checkpoint side B only",
          grouped_pre[CK_0511][0]["state_a"]["state_kind"] == "current"
          and grouped_pre[CK_0511][0]["state_a"]["checkpoint_id"] is None
          and grouped_pre[CK_0511][0]["state_b"]["checkpoint_id"]
          == CK_0511)
    check("D3 parity with the filtered M5 listing",
          grouped_pre[CK_0511] == expected_from_listing(comps, CK_0511))

    print("== LIVE E: dedup + partition verification ==")
    slots = {ck: [x["comparison_id"] for x in grouped_pre[ck]]
             for ck in CKPTS}
    all_ids = [i for ck in CKPTS for i in slots[ck]]
    check("E1 6 + 5 + 1 = 12 membership slots across the three "
          "checkpoint histories",
          [len(slots[ck]) for ck in (CK_025E, CK_30A8, CK_0511)]
          == [6, 5, 1])
    check("E2 no duplicate comparison id WITHIN any checkpoint history "
          "(A=B appears once, not twice)",
          all(len(ids) == len(set(ids)) for ids in slots.values()))
    union = set(all_ids)
    check("E3 union over the three histories = 8 UNIQUE comparisons "
          "(overlaps legit, never double-counted)",
          len(all_ids) == 12 and len(union) == 8
          and union == {c["comparison_id"] for c in comps})
    both = {cid for cid in union
            if sum(cid in slots[ck] for ck in CKPTS) == 2}
    check("E4 the 4 double-membership records (fc379, baa36, d9a62, "
          "d62f8 under BOTH 025e and 30a8) each appear once per "
          "checkpoint",
          both == {"fc379bfcb50f", "baa361012e00", "d9a62dde016b",
                   "d62f89e97c85"}
          and all(slots[CK_025E].count(cid) == 1
                  and slots[CK_30A8].count(cid) == 1 for cid in both))
    check("E5 each A=B record lives ONLY under its own checkpoint",
          all(cid in slots[ck] and sum(cid in slots[k]
                                       for k in CKPTS) == 1
              for cid, ck in AB_SAME.items()))

    print("== LIVE F: current-state sides never match ==")
    for cid, ck in CURRENT_VS_CK.items():
        others = [k for k in CKPTS if k != ck]
        check(f"F {cid} (current-vs-{ck}) appears under {ck} exactly "
              "once and under NO other checkpoint",
              slots[ck].count(cid) == 1
              and all(cid not in slots[k] for k in others))
    check("F3 no comparison with BOTH sides current can ever appear "
          "(none exists; current sides keep checkpoint_id null)",
          not any(c["state_a"]["state_kind"] == "current"
                  and c["state_b"]["state_kind"] == "current"
                  for c in comps)
          and all(s["checkpoint_id"] is None
                  for c in comps for s in (c["state_a"], c["state_b"])
                  if s["state_kind"] == "current"))

    print("== LIVE G: deterministic repeats ==")
    for ck in CKPTS:
        _, raw_x = call("GET", byck(ck))
        _, raw_y = call("GET", byck(ck))
        check(f"G {ck}: repeated GETs raw-byte-identical",
              raw_x == raw_y
              and json.loads(raw_x) == grouped_pre[ck])

    print("== LIVE H: unknown model ==")
    c1, _ = call_json("GET", byck(CK_025E, "no-such-model-26"))
    check("H1 unknown model -> 404 (even with a real checkpoint id)",
          c1 == 404, f"{c1}")

    print("== LIVE I: unknown checkpoint ==")
    c2, _ = call_json("GET", byck(UNKNOWN_CK))
    check("I1 unknown well-formed checkpoint id -> 404 (valid model, "
          "empty history is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{byck('ck%20id%20with%20spaces!!')}")
    check("I2 malformed unknown checkpoint id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE J: cross-model isolation + no shadowing ==")
    c4, _ = call_json("GET", byck(CK_0511, OTHER_MODEL))
    check("J1 other real model + this model's real registry-valid "
          "checkpoint id -> 404 (model-scoped M3 registry; the other "
          "model owns no checkpoints)", c4 == 404, f"{c4}")
    c5, leak = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/comparisons")
    check("J2 none of the 8 comparison ids is reachable under the other "
          "model (its history is [])",
          c5 == 200 and leak == []
          and all(cid not in (leak or []) for cid in COMP_IDS))
    c6, one = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                               f"{COMP_IDS[0]}")
    check("J3 by-checkpoint does not shadow the M5 detail getter "
          "(fc379bfcb50f resolves verbatim)",
          c6 == 200 and one == comps[0])
    c7, _ = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                             "ghost-comp-26")
    check("J4 unknown comparison id via detail getter still 404",
          c7 == 404, f"{c7}")

    print("== LIVE K: regression + final audit (zero storage growth) ==")
    code, comps2 = call_json("GET", f"{BASE}/models/{MODEL}/comparisons")
    check("K1 M5 listing unchanged", code == 200 and comps2 == comps)
    ok_detail = True
    for cid in COMP_IDS:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                  f"{cid}")
        ok_detail = ok_detail and c == 200 \
            and one == comps[COMP_IDS.index(cid)]
    check("K2 M5 detail getter unchanged for all 8 records", ok_detail)
    code2, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   "records")
    code4, bs2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-sample/{SAMPLE}")
    code5, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/{CK_0511}")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}/summary")
    code8, ev2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"by-checkpoint/{CK_0511}")
    code9, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"by-policy/{POLICY}")
    code10, srck2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                     f"by-checkpoint/{CK_0511}")
    check("K3 M16/M18/M19/M20/M21/M22/M23/M24/M25 outputs unchanged",
          code2 == 200 and sq2 == sq and code3 == 200 and rec2 == rec
          and code4 == 200 and bs2 == bs and code5 == 200 and bc2 == bc
          and code6 == 200 and g2 == g and code7 == 200 and s2 == s
          and code8 == 200 and ev2 == ev and code9 == 200 and gd2 == gd
          and code10 == 200 and srck2 == srck)
    code, gates_all = call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                                       "decisions")
    check("K4 M6 gate-decision history surface intact (listing 200)",
          code == 200 and isinstance(gates_all, list))
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("K5 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("K6 M11 workflow history, M3 checkpoint registry and M9 "
          "policy registry byte-identical to pre-state",
          code_w2 == 200 and wfs2 == wfs and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol)
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    check("K7 OpenAPI still 58 paths with the new path exactly once "
          "(no drift during the smoke)",
          code_sp2 == 200 and len(spec2["paths"]) == 58
          and set(spec2["paths"][NEW_PATH]) == {"get"})
    ok_again = True
    for ck in CKPTS:
        _, body = call_json("GET", byck(ck))
        ok_again = ok_again and body == grouped_pre[ck]
    check("K8 by-checkpoint still deterministic at the end (all three "
          "checkpoints)", ok_again)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("K9 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("K10 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("K11 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M26 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M26 live smoke OK: narrow read-only two-sided by-checkpoint "
          "grouping of the M5 comparison history (exact listing parity "
          "for all three live checkpoints — 6/5/1 records with the "
          "exact audited ids —, A=B dedup exactly once, current-state "
          "sides never matching, 4 double-membership records once per "
          "checkpoint, union = 8 unique comparisons, clean 404s incl. "
          "cross-model isolation through the model-scoped M3 registry, "
          "deterministic byte-identical repeats, M5/M6/M16-M25 surfaces "
          "+ dashboard hash + registries + OpenAPI 58 unchanged, ZERO "
          "production storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
