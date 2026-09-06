"""M24 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M24 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/evaluations/by-checkpoint/{checkpoint_id},
answering "which immutable M4 evaluations measured this checkpoint?" —
the model's authoritative M4 listing filtered by the persisted
checkpoint identity recorded in each EvaluationRecord, after checkpoint
ownership is validated through the M3 checkpoint registry (the same
model-scoped path M20 uses). The smoke must prove exact M4-listing parity
for every live checkpoint, current-state exclusion, 200 + [] semantics
only for VALID checkpoints (none exist in production — the empty case is
covered by the isolated unit/API fixtures), clean 404s (unknown model /
unknown checkpoint / cross-model), byte-identical repeats, unchanged
M4/M18/M19/M20/M21/M22/M23 surfaces + dashboard hash, and ZERO
production storage growth.

Production facts (audited at Phase A): model 4a0a871886ef owns 16 M4
evaluations — 9 checkpoint-grouped under exactly 3 checkpoints
(025e6d8d8f15: 7a16eaa12120, 90aa392b9a2b, ebb9b7eccbe3;
0511de4c7372: c739c66638e9, 0704399fea7b, 75a23351e91c; 30a8bc5b82ab:
b0502d871114, 75835b64d6af, fe7b42cdb411) and 7 current-state
evaluations (a439eb92f9cd, 0cc96a125976, 425003213a0b, a884bf729ff7,
b89a94306ce8, 340f5adbf881, c35a1c9902fe) that must NEVER appear; model
b5bc905326b6 exists with NO evaluations and NO checkpoints (cross-model
probe); suite m9-live-suite has 10 runs; policy m9-live-policy has 1
gate decision; M17 dashboard result_hash f48557fe8ab1... — 96 files /
4,002,745 B / 0 .tmp.

LIVE TEST A (baseline audit): exact 96 / 4,002,745 / 0 audit + full
per-file hash inventory saved to /tmp/m24-smoke-baseline-inventory.json +
model/checkpoint registry facts + M4 listing + M18/M19/M20/M21/M22/M23/
dashboard pre-state.

LIVE TEST B (first known checkpoint): by-checkpoint of 0511de4c7372 ->
200 with EXACTLY 3 records (c739c66638e9, 0704399fea7b, 75a23351e91c);
every record's persisted model_id / checkpoint_id / eval_id / created_at
verified against the authoritative M4 listing.

LIVE TEST C (all three checkpoints): 3 + 3 + 3 = 9 evaluations, no
duplication between checkpoint histories, union equals the listing's
checkpoint-grouped set.

LIVE TEST D (current-state exclusion): the 9 checkpoint-associated ids
vs the full 16-record listing — the 7 current-state ids appear in NO
checkpoint response.

LIVE TEST E (deterministic repeat): repeated GETs raw-byte-identical.

LIVE TEST F (valid empty case): production has no unevaluated
checkpoint (all 3 have 3 evaluations) — the 200 + [] case is covered by
the isolated unit/API fixtures; production is NOT mutated to manufacture
it (guarded here by: every checkpoint responds 200, never 404).

LIVE TEST G (errors): unknown model -> 404; unknown well-formed /
malformed checkpoint id -> 404.

LIVE TEST H (cross-model isolation): b5bc905326b6 + 4a0a871886ef's real
checkpoint id -> 404 (model-scoped M3 registry); no foreign evaluation
leaks.

LIVE TEST I (regression): M4 listing/get, M18 records, M19 by-sample,
M20 by-checkpoint (sample-quality), M21 by-suite, M22 summary, M23
by-policy, dashboard hash, policy registry, checkpoint registry — all
unchanged.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8745 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8745/api/v1"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO checkpoints/evaluations
CKPTS = {"025e6d8d8f15": ["7a16eaa12120", "90aa392b9a2b", "ebb9b7eccbe3"],
         "0511de4c7372": ["c739c66638e9", "0704399fea7b", "75a23351e91c"],
         "30a8bc5b82ab": ["b0502d871114", "75835b64d6af", "fe7b42cdb411"]}
CURRENT_IDS = ("a439eb92f9cd", "0cc96a125976", "425003213a0b",
               "a884bf729ff7", "b89a94306ce8", "340f5adbf881",
               "c35a1c9902fe")       # never appear in by-checkpoint
UNKNOWN_CK = "ffffffffffff"          # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"              # the live production suite (10 runs)
POLICY = "m9-live-policy"            # the live production policy (1 gate)
SAMPLE = "f8e66f9c7b50"              # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")    # ASCENDING (created, id)
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m24-smoke-baseline-inventory.json")

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
    print("== LIVE TEST A: baseline audit + saved inventory ==")
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
    check("A3 both models resolve; the 3 known checkpoints register",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_c == 200
          and {c["checkpoint_id"] for c in cks} == set(CKPTS))
    code, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_o, evals_o = call_json("GET",
                                f"{BASE}/models/{OTHER_MODEL}/evaluations")
    ck_grouped = [e for e in evals if e["checkpoint_id"] is not None]
    cur_grouped = [e for e in evals if e["checkpoint_id"] is None]
    check("A4 M4 listing pre-state: 16 evaluations (9 checkpoint-grouped "
          "under 3 checkpoints, 7 current-state); other model none",
          code == 200 and len(evals) == 16 and code_o == 200
          and evals_o == [] and len(ck_grouped) == 9
          and len(cur_grouped) == 7
          and {e["checkpoint_id"] for e in ck_grouped} == set(CKPTS)
          and sorted(e["eval_id"] for e in cur_grouped)
          == sorted(CURRENT_IDS))
    check("A5 M4 listing order is (created_at, eval_id) ASCENDING",
          [(e["created_at"], e["eval_id"]) for e in evals]
          == sorted((e["created_at"], e["eval_id"]) for e in evals))
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  "records")
    code4, bs = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-sample/{SAMPLE}")
    code5, bc = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-checkpoint/0511de4c7372")
    code6, g = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}")
    code7, s = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}/summary")
    code8, gd = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"by-policy/{POLICY}")
    code9, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("A6 M16/M18/M19/M20/M21/M22/M23 pre-state intact",
          code2 == 200 and len(sq) == 2 and code3 == 200 and len(rec) == 2
          and code4 == 200 and len(bs) == 2 and code5 == 200
          and len(bc) == 2 and code6 == 200 and len(g) == 10
          and code7 == 200 and s["total_count"] == 10 and code8 == 200
          and len(gd) == 1 and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A7 M17 dashboard hash begins f48557fe8ab1",
          code9 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])

    print("== LIVE TEST B: first known checkpoint (0511de4c7372) ==")
    first_ck = "0511de4c7372"
    url = f"{BASE}/models/{MODEL}/evaluations/by-checkpoint/{first_ck}"
    code, raw1 = call("GET", url)
    recs = json.loads(raw1)
    check("B1 HTTP 200 with exactly 3 records",
          code == 200 and len(recs) == 3, f"{code}/{len(recs)}")
    check("B2 the exact known eval ids in ASCENDING (created_at, eval_id) "
          "order", [x["eval_id"] for x in recs] == CKPTS[first_ck]
          and [(x["created_at"], x["eval_id"]) for x in recs]
          == sorted((x["created_at"], x["eval_id"]) for x in recs))
    check("B3 every record's persisted identity: model_id + checkpoint_id "
          "as requested",
          all(x["model_id"] == MODEL and x["state_kind"] == "checkpoint"
              and x["checkpoint_id"] == first_ck for x in recs))
    check("B4 parity with the M4 listing filtered by the persisted "
          "checkpoint identity",
          recs == [x for x in evals if x["checkpoint_id"] == first_ck])
    for x in recs:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"{x['eval_id']}")
        check(f"B5 M4 getter parity for {x['eval_id']}",
              c == 200 and one == x)
    check("B6 verbatim EvaluationRecord payloads (same field set as the "
          "M4 listing), metrics present",
          all(set(x) == set(evals[0]) and isinstance(x["loss_nats"], float)
              and x["perplexity"] > 0 and len(x["result_hash"]) == 64
              for x in recs))

    print("== LIVE TEST C: all three checkpoints (3 + 3 + 3 = 9) ==")
    all_ids = []
    for ck, expected in sorted(CKPTS.items()):
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok = c == 200 and [x["eval_id"] for x in body] == expected \
            and all(x["checkpoint_id"] == ck for x in body)
        check(f"C1 checkpoint {ck} -> 200 with exactly its 3 evaluations",
              ok, f"{c}")
        all_ids.extend(x["eval_id"] for x in body)
    check("C2 total across all three checkpoint histories is 9",
          len(all_ids) == 9)
    check("C3 no duplication between checkpoint histories",
          len(set(all_ids)) == 9)
    check("C4 union equals the listing's checkpoint-grouped set",
          set(all_ids) == {e["eval_id"] for e in ck_grouped})

    print("== LIVE TEST D: current-state exclusion ==")
    check("D1 the 7 current-state evaluations appear in NO checkpoint "
          "response", all(cid not in all_ids for cid in CURRENT_IDS)
          and len(CURRENT_IDS) == 7)
    check("D2 9 checkpoint-associated + 7 current-state = 16 total",
          len(all_ids) + len(CURRENT_IDS) == len(evals) == 16)

    print("== LIVE TEST E: deterministic repeat ==")
    code2, raw2 = call("GET", url)
    check("E1 repeated GET raw-byte-identical", code2 == 200 and raw2 == raw1)
    code3, raw3 = call("GET", url)
    check("E2 second repeat raw-byte-identical",
          code3 == 200 and raw3 == raw1)

    print("== LIVE TEST F: valid-empty semantics guarded (no production "
          "mutation) ==")
    check("F1 every production checkpoint responds 200 (none 404); the "
          "200 + [] empty case is proven by the isolated unit/API "
          "fixtures without mutating production",
          all(call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"by-checkpoint/{ck}")[0] == 200
              for ck in CKPTS))

    print("== LIVE TEST G: error semantics ==")
    c1, _ = call_json("GET", f"{BASE}/models/no-such-model-24/"
                             f"evaluations/by-checkpoint/{first_ck}")
    check("G1 unknown model -> 404", c1 == 404, f"{c1}")
    c2, _ = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                             f"by-checkpoint/{UNKNOWN_CK}")
    check("G2 unknown well-formed checkpoint id -> 404", c2 == 404,
          f"{c2}")
    c3, _ = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                             "by-checkpoint/ck%20id%20with%20spaces!!")
    check("G3 malformed unknown checkpoint id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE TEST H: cross-model isolation ==")
    c4, _ = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/evaluations/"
                             f"by-checkpoint/{first_ck}")
    check("H1 other real model + this model's real checkpoint id -> 404 "
          "(model-scoped M3 registry)", c4 == 404, f"{c4}")
    c5, leak = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/evaluations")
    check("H2 no evaluation of 4a0a871886ef is reachable under the other "
          "model", c5 == 200 and leak == []
          and all(eid not in (leak or []) for eid in all_ids))
    c6, one = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                               f"{CKPTS[first_ck][0]}")
    check("H3 by-checkpoint does not shadow the M4 detail getter",
          c6 == 200 and one["eval_id"] == CKPTS[first_ck][0])
    c7, _ = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                             "ghost-eval-24")
    check("H4 unknown eval id via detail getter still 404", c7 == 404,
          f"{c7}")

    print("== LIVE TEST I: regression + final audit (zero storage growth) ==")
    code, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   "records")
    code4, bs2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-sample/{SAMPLE}")
    code5, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/0511de4c7372")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                 f"{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                 f"{SUITE}/summary")
    code8, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"by-policy/{POLICY}")
    check("I1 M18/M19/M20/M21/M22/M23 outputs unchanged",
          code == 200 and sq2 == sq and code3 == 200 and rec2 == rec
          and code4 == 200 and bs2 == bs and code5 == 200 and bc2 == bc
          and code6 == 200 and g2 == g and code7 == 200 and s2 == s
          and code8 == 200 and gd2 == gd)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("I2 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code, evals2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("I3 M4 listing, M3 checkpoint registry and M9 policy registry "
          "byte-identical to pre-state",
          code == 200 and evals2 == evals and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol)
    code, recs2 = call_json("GET", url)
    check("I4 by-checkpoint still deterministic at the end",
          code == 200 and recs2 == recs)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("I5 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("I6 zero new files", new_files == set(), f"{len(new_files)} new")
    check("I7 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M24 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M24 live smoke OK: narrow read-only by-checkpoint grouping of "
          "the M4 evaluation history (exact listing parity for all three "
          "live checkpoints — 3+3+3=9, no duplication —, current-state "
          "evaluations never appear, clean 404s incl. cross-model "
          "isolation through the model-scoped M3 registry, deterministic "
          "byte-identical repeats, M4/M18/M19/M20/M21/M22/M23 surfaces + "
          "dashboard hash + registries unchanged, ZERO production storage "
          "growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
