"""M23 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M23 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/gates/decisions/by-policy/{policy_id},
answering "which immutable gate decisions of this model were produced
under this registered policy?" — the model's authoritative M6 listing
filtered by the persisted policy_id recorded in each GateDecision, after
policy existence is verified through the M9 policy registry. The smoke
must prove exact M6-listing parity for the live policy, 200 + [] for a
valid policy with no decisions under another model, clean 404s (unknown
model / unknown policy), cross-model isolation, exclusion of
inline-policy decisions (policy_id null), byte-identical repeats,
unchanged M6/M9/M18–M22 surfaces + dashboard hash, and ZERO production
storage growth.

Production facts (audited at Phase A): model 4a0a871886ef owns exactly
11 gate decisions (ASCENDING (created_at, decision_id) order
8931835d4af6, baf767bdcbe1, 9d4facca5153, ef8ba75f9e43, 5b0493c0fbed,
8968151a08bd, a82c95374a5a, e33c99f2f8ca, 0dae7b2c456e, 5c86e4494d2f,
6921d3b29b9d); exactly ONE carries the persisted
policy_id='m9-live-policy' (decision 6921d3b29b9d, verdict passed,
created 2026-09-04T09:04:08.780839Z); the other 10 are inline decisions
(policy_id null); policy m9-live-policy is the single registered M9
policy; model b5bc905326b6 exists but owns NO gate decisions
(cross-model / empty-grouping probe); M16/M21/M22-era facts (sample
evals 8ff910cf2a9e / 31a283413c75 under checkpoint 0511de4c7372; 10
suite runs of m9-live-suite; M17 dashboard result_hash
f48557fe8ab1...) — 96 files / 4,002,745 B / 0 .tmp.

Phase A (baseline audit): exact 96 / 4,002,745 / 0 audit + full per-file
hash inventory saved to /tmp/m23-smoke-baseline-inventory.json + model,
policy & gate registry facts + M6 listing / M9 policy / M18 records /
M19 by-sample / M20 by-checkpoint / M21 by-suite / M22 summary /
dashboard pre-state.

Phase B (new endpoint): by-policy of m9-live-policy on 4a0a871886ef ->
200 with EXACTLY 1 record (6921d3b29b9d, passed, policy_id
m9-live-policy); the persisted model_id/policy_id/decision_id/created_at
verified against the authoritative M6 listing; payload parity with the
M6 listing (filtered by persisted policy_id) and the M6 single-record
getter; verbatim GateDecision field set (no derived statistics);
inline-policy decisions never appear; raw-response byte-identical
repeat; M6 listing unchanged by the reads.

Phase C (404 semantics, cross-model isolation, storage-neutral): unknown
model by-policy -> 404; unknown well-formed policy id -> 404; malformed
(percent-encoded) policy id -> 404 (no crash); CROSS-MODEL probe — model
b5bc905326b6 (real) + the real policy m9-live-policy -> 200 + [] (valid
policy, no decisions) and the foreign decision id never leaks; unknown
policy under the other model -> 404; by-policy never shadows the M6
detail getter (a real decision id still resolves, a ghost id still
404s); M9 policy registry still resolves the policy.

Phase D (final audit): M18 records / M19 by-sample / M20 by-checkpoint /
M21 by-suite / M22 summary outputs unchanged; M17 dashboard result_hash
unchanged; M6 listing + M9 policy definition byte-identical to
pre-state; by-policy still deterministic at the end; inventory diff vs
Phase A — every pre-existing file byte-identical, ZERO new files, zero
.tmp, zero storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8744 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8744/api/v1"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, owns NO gate decisions
POLICY = "m9-live-policy"             # the live production policy
UNKNOWN_POLICY = "m-ghost-policy-23"  # well-formed, nonexistent
DECISION = "6921d3b29b9d"             # the single m9-live-policy decision
DECISION_AT = "2026-09-04T09:04:08.780839Z"
ALL_DECISIONS = ("8931835d4af6", "baf767bdcbe1", "9d4facca5153",
                 "ef8ba75f9e43", "5b0493c0fbed", "8968151a08bd",
                 "a82c95374a5a", "e33c99f2f8ca", "0dae7b2c456e",
                 "5c86e4494d2f", "6921d3b29b9d")  # ASCENDING (created, id)
SUITE = "m9-live-suite"               # the live production suite (10 runs)
CKPT = "0511de4c7372"                 # the measured production checkpoint
SAMPLE = "f8e66f9c7b50"               # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")     # ASCENDING (created, id)
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m23-smoke-baseline-inventory.json")

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


def main() -> int:
    print("== Phase A: baseline audit + saved inventory ==")
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
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("A3 both models + the policy resolve in the registries",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_p == 200
          and pol["policy_id"] == POLICY)
    code, decs = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")
    code_o, decs_o = call_json("GET",
                               f"{BASE}/models/{OTHER_MODEL}/gates/decisions")
    check("A4 M6 listing pre-state: exactly the 11 known decisions in "
          "ASCENDING order; other model has none",
          code == 200 and len(decs) == 11 and code_o == 200
          and decs_o == []
          and [x["decision_id"] for x in decs] == list(ALL_DECISIONS))
    check("A5 exactly one decision carries the persisted "
          "policy_id=m9-live-policy (6921d3b29b9d, passed)",
          [x["decision_id"] for x in decs
           if x.get("policy_id") == POLICY] == [DECISION]
          and len([x for x in decs if x.get("policy_id") is None]) == 10)
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  "records")
    code4, bs = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-sample/{SAMPLE}")
    code5, bc = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-checkpoint/{CKPT}")
    code6, g = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}")
    code7, s = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}/summary")
    code8, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A6 M16/M18/M19/M20/M21/M22 pre-state intact (2 records, 10 "
          "runs, summary 10)",
          code2 == 200 and len(sq) == 2 and code3 == 200 and len(rec) == 2
          and code4 == 200 and len(bs) == 2 and code5 == 200
          and len(bc) == 2 and code6 == 200 and len(g) == 10
          and code7 == 200 and s["total_count"] == 10
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A7 M17 dashboard hash begins f48557fe8ab1",
          code8 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])

    print("== Phase B: new endpoint /gates/decisions/by-policy ==")
    url = f"{BASE}/models/{MODEL}/gates/decisions/by-policy/{POLICY}"
    code, raw1 = call("GET", url)
    grouped = json.loads(raw1)
    check("B1 HTTP 200 with exactly 1 gate decision",
          code == 200 and len(grouped) == 1, f"{code}/{len(grouped)}")
    one = grouped[0] if grouped else {}
    check("B2 the decision is 6921d3b29b9d (passed, created "
          "2026-09-04T09:04:08.780839Z)",
          one.get("decision_id") == DECISION
          and one.get("decision") == "passed"
          and one.get("created_at") == DECISION_AT)
    check("B3 persisted identity: model_id + policy_id as requested",
          one.get("model_id") == MODEL and one.get("policy_id") == POLICY)
    check("B4 parity with the M6 listing filtered by persisted policy_id",
          grouped == [x for x in decs if x.get("policy_id") == POLICY])
    c, single = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"{DECISION}")
    check("B5 parity with the M6 single-record getter",
          c == 200 and single == one)
    check("B6 full verbatim GateDecision payload (same field set as the "
          "M6 listing), no derived statistics",
          set(one) == set(decs[0]) and one.get("policy_config_hash")
          == pol["config_hash"] and one.get("policy_id") == POLICY)
    check("B7 inline-policy decisions (policy_id null) never appear",
          all(x.get("policy_id") == POLICY for x in grouped)
          and len(grouped) == 1)
    code2, raw2 = call("GET", url)
    check("B8 repeated GET raw-byte-identical", code2 == 200 and raw2 == raw1)
    code3, raw3 = call("GET", url)
    check("B9 second repeat raw-byte-identical",
          code3 == 200 and raw3 == raw1)
    check("B10 M6 listing unchanged by the reads",
          call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")[1]
          == decs)

    print("== Phase C: 404 semantics + cross-model isolation ==")
    c1, _ = call_json("GET", f"{BASE}/models/no-such-model-23/"
                             f"gates/decisions/by-policy/{POLICY}")
    check("C1 unknown model by-policy -> 404", c1 == 404, f"{c1}")
    c2, _ = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                             f"by-policy/{UNKNOWN_POLICY}")
    check("C2 unknown well-formed policy id -> 404", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                             "by-policy/pol%20id%20with%20spaces!!")
    check("C3 malformed unknown policy id -> 404 (no crash)",
          c3 == 404, f"{c3}")
    c4, empty = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/gates/"
                                 f"decisions/by-policy/{POLICY}")
    check("C4 cross-model isolation: other real model + the real policy "
          "-> 200 + [] (valid policy, no decisions)",
          c4 == 200 and empty == [], f"{c4}")
    check("C5 the foreign decision id never leaks",
          empty == [] and DECISION not in (empty or []))
    c6, _ = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/gates/"
                             f"decisions/by-policy/{UNKNOWN_POLICY}")
    check("C6 unknown policy under the other model -> 404", c6 == 404,
          f"{c6}")
    c7, det = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                               f"{DECISION}")
    check("C7 by-policy does not shadow the M6 detail getter",
          c7 == 200 and det["decision_id"] == DECISION)
    c8, _ = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                             "ghost-dec-23")
    check("C8 unknown decision id via detail getter still 404", c8 == 404,
          f"{c8}")
    c9, _ = call_json("GET", f"{BASE}/models/no-such-model-23/"
                             "gates/decisions")
    check("C9 M6 listing unknown model still 404", c9 == 404, f"{c9}")
    c10, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("C10 M9 policy registry still resolves the policy",
          c10 == 200 and pol2 == pol)

    print("== Phase D: prior endpoints + final audit (zero storage growth) ==")
    code, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code2, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   "records")
    code3, bs2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-sample/{SAMPLE}")
    code4, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/{CKPT}")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                 f"{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                 f"{SUITE}/summary")
    check("D1 M18/M19/M20/M21/M22 outputs unchanged",
          code == 200 and sq2 == sq and code2 == 200 and rec2 == rec
          and code3 == 200 and bs2 == bs and code4 == 200 and bc2 == bc
          and code6 == 200 and g2 == g and code7 == 200 and s2 == s)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("D2 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code, decs2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions")
    code_p2, pol3 = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("D3 M6 listing + M9 policy byte-identical to pre-state",
          code == 200 and decs2 == decs and code_p2 == 200 and pol3 == pol)
    code, grouped2 = call_json("GET", url)
    check("D4 by-policy still deterministic at the end",
          code == 200 and grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("D5 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("D6 zero new files", new_files == set(), f"{len(new_files)} new")
    check("D7 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M23 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M23 live smoke OK: narrow read-only by-policy grouping (exact "
          "M6-record parity for the live policy — 1 decision with the "
          "persisted model/policy identity —, 200 + [] for a valid policy "
          "without decisions under another model, clean 404s incl. "
          "cross-model isolation, inline decisions excluded, "
          "deterministic byte-identical repeats, M6/M9/M18/M19/M20/M21/M22 "
          "surfaces + dashboard hash unchanged, ZERO production storage "
          "growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
