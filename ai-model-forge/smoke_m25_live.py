"""M25 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M25 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/suite-runs/by-checkpoint/{checkpoint_id},
answering "which immutable M10 suite runs executed against this
checkpoint state?" — the model's authoritative M10 listing filtered by
the persisted run state recorded in each SuiteRunRecord
(state.state_kind="checkpoint" + state.checkpoint_id), after checkpoint
ownership is validated through the M3 checkpoint registry (the same
model-scoped path M20/M24 use). The smoke must prove exact M10-listing
parity for the live checkpoint, 200 + [] for the two valid-but-empty
checkpoints, partition integrity (10 + 0 + 0 = 10, no duplication),
clean 404s (unknown model / unknown checkpoint / cross-model),
byte-identical repeats, unchanged M10/M11/M18-M24 surfaces + dashboard
hash + registries, and ZERO production storage growth.

Production facts (audited at LIVE A): model 4a0a871886ef owns 3
checkpoints (025e6d8d8f15, 0511de4c7372, 30a8bc5b82ab) and 10 suite
runs, ALL 10 executed against checkpoint state 0511de4c7372 (ASCENDING
(created_at, suite_run_id) order aae8e8a8ba9c, bb114d2ce42e,
8f8aee834c9f, d9742459017b, 5684649d0ced, 2af508035191, 4cb388a21ce2,
2dc9f7dff400, 2d203398304b, e4b1c2a7fb2d); checkpoints 025e6d8d8f15 and
30a8bc5b82ab are valid with ZERO suite runs -> natural live 200 + []
cases; model b5bc905326b6 exists with NO checkpoints and NO suite runs
(cross-model probe); M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file SHA256
inventory saved to /tmp/m25-smoke-baseline-inventory.json + model,
checkpoint & suite registries + M10 listing / M11 workflows / M18-M24 /
dashboard pre-state.

LIVE B (known checkpoint): by-checkpoint of 0511de4c7372 -> 200 with
EXACTLY 10 records — all 10 known ids in the ASCENDING M10 order; every
record's persisted model identity + checkpoint state verified against
the authoritative M10 listing.

LIVE C (empty checkpoint 1): 025e6d8d8f15 -> 200 + [].

LIVE D (empty checkpoint 2): 30a8bc5b82ab -> 200 + [].

LIVE E (partition): 10 + 0 + 0 = 10 and no duplicate suite-run ids
across checkpoint results; the union equals the full M10 listing.

LIVE F (deterministic repeat): repeated GETs raw-byte-identical.

LIVE G (unknown model): -> 404.

LIVE H (unknown checkpoint): well-formed and malformed unknown ids ->
404.

LIVE I (cross-model isolation): b5bc905326b6 + 0511de4c7372 (owned by
4a0a871886ef) -> 404 via the model-scoped M3 registry (b5bc905326b6 has
no checkpoints); no run id leaks.

LIVE J (regression + final audit): M10 list/get, M11 workflow history,
M21 by-suite, M22 summary, M24 evaluations by-checkpoint, M23
by-policy, M18-M20 sample-quality, dashboard hash, policy registry,
checkpoint registry — all unchanged; inventory diff — every pre-existing
file byte-identical, ZERO new files, zero .tmp, zero storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8746 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8746/api/v1"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO checkpoints/suite runs
CKPT = "0511de4c7372"                 # the checkpoint ALL 10 runs used
EMPTY_CKPTS = ("025e6d8d8f15", "30a8bc5b82ab")  # valid, zero suite runs
RUN_IDS = ("aae8e8a8ba9c", "bb114d2ce42e", "8f8aee834c9f", "d9742459017b",
           "5684649d0ced", "2af508035191", "4cb388a21ce2", "2dc9f7dff400",
           "2d203398304b", "e4b1c2a7fb2d")   # ASCENDING (created_at, id)
UNKNOWN_CK = "ffffffffffff"          # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"              # the live production suite
POLICY = "m9-live-policy"            # the live production policy (1 gate)
SAMPLE = "f8e66f9c7b50"              # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")    # ASCENDING (created, id)
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m25-smoke-baseline-inventory.json")

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
          == {CKPT, *EMPTY_CKPTS}
          and call_json("GET", f"{BASE}/models/{OTHER_MODEL}/checkpoints")
          [1] == [])
    code, runs = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    code_o, runs_o = call_json("GET",
                               f"{BASE}/models/{OTHER_MODEL}/suite-runs")
    ck_runs = [r for r in runs if r["state"]["state_kind"] == "checkpoint"]
    check("A4 M10 listing pre-state: exactly the 10 known runs, ALL "
          "against checkpoint state 0511de4c7372; other model none",
          code == 200 and len(runs) == 10 and code_o == 200
          and runs_o == []
          and [r["suite_run_id"] for r in runs] == list(RUN_IDS)
          and len(ck_runs) == 10
          and {r["state"]["checkpoint_id"] for r in ck_runs} == {CKPT})
    code_w, wfs = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
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
    code8, ev = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                 f"by-checkpoint/{CKPT}")
    code9, gd = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"by-policy/{POLICY}")
    code10, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("A5 M11/M16/M18-M24 pre-state intact (workflows 200, 2 sample "
          "records, 10 by-suite, summary 10, 3 checkpoint evals, 1 gate)",
          code_w == 200 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(rec) == 2 and code4 == 200
          and len(bs) == 2 and code5 == 200 and len(bc) == 2
          and code6 == 200 and len(g) == 10 and code7 == 200
          and s["total_count"] == 10 and code8 == 200 and len(ev) == 3
          and code9 == 200 and len(gd) == 1 and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A6 M17 dashboard hash equals the full known value",
          code10 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])

    print("== LIVE B: known checkpoint (0511de4c7372) ==")
    url = f"{BASE}/models/{MODEL}/suite-runs/by-checkpoint/{CKPT}"
    code, raw1 = call("GET", url)
    grouped = json.loads(raw1)
    check("B1 HTTP 200 with exactly 10 records",
          code == 200 and len(grouped) == 10, f"{code}/{len(grouped)}")
    check("B2 the exact known run ids in ASCENDING (created_at, "
          "suite_run_id) M10 order",
          [x["suite_run_id"] for x in grouped] == list(RUN_IDS)
          and [(x["created_at"], x["suite_run_id"]) for x in grouped]
          == sorted((x["created_at"], x["suite_run_id"]) for x in grouped))
    check("B3 every record's persisted identity: model + checkpoint "
          "state as requested",
          all(x["model_id"] == MODEL and x["state"]["state_kind"]
              == "checkpoint" and x["state"]["checkpoint_id"] == CKPT
              for x in grouped))
    check("B4 parity with the M10 listing filtered by the persisted run "
          "state", grouped == runs)
    for x in grouped[:3] + grouped[-1:]:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                  f"{x['suite_run_id']}")
        check(f"B5 M10 getter parity for {x['suite_run_id']}",
              c == 200 and one == x)
    check("B6 verbatim SuiteRunRecord payloads (same field set as the "
          "M10 listing)",
          all(set(x) == set(runs[0]) and x["status"] == "completed"
              and len(x["result_hash"]) == 64 for x in grouped))

    print("== LIVE C/D: the two valid empty checkpoints ==")
    for ck in EMPTY_CKPTS:
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                   f"by-checkpoint/{ck}")
        check(f"{'C1' if ck == EMPTY_CKPTS[0] else 'D1'} valid checkpoint "
              f"{ck} with no suite runs -> 200 + []",
              c == 200 and body == [], f"{c}")

    print("== LIVE E: partition verification ==")
    all_ids = []
    for ck in (CKPT, *EMPTY_CKPTS):
        _, body = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                   f"by-checkpoint/{ck}")
        all_ids.extend(x["suite_run_id"] for x in body)
    check("E1 10 + 0 + 0 = 10 across the three checkpoint histories",
          len(all_ids) == 10)
    check("E2 no duplicate suite-run ids across checkpoint results",
          len(set(all_ids)) == 10)
    check("E3 union equals the full M10 listing of the model",
          set(all_ids) == {r["suite_run_id"] for r in runs})

    print("== LIVE F: deterministic repeat ==")
    code2, raw2 = call("GET", url)
    check("F1 repeated GET raw-byte-identical", code2 == 200 and raw2 == raw1)
    code3, raw3 = call("GET", url)
    check("F2 second repeat raw-byte-identical",
          code3 == 200 and raw3 == raw1)

    print("== LIVE G: unknown model ==")
    c1, _ = call_json("GET", f"{BASE}/models/no-such-model-25/"
                             f"suite-runs/by-checkpoint/{CKPT}")
    check("G1 unknown model -> 404", c1 == 404, f"{c1}")

    print("== LIVE H: unknown checkpoint ==")
    c2, _ = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                             f"by-checkpoint/{UNKNOWN_CK}")
    check("H1 unknown well-formed checkpoint id -> 404", c2 == 404,
          f"{c2}")
    c3, _ = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                             "by-checkpoint/ck%20id%20with%20spaces!!")
    check("H2 malformed unknown checkpoint id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE I: cross-model isolation ==")
    c4, _ = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/suite-runs/"
                             f"by-checkpoint/{CKPT}")
    check("I1 other real model + this model's real checkpoint id -> 404 "
          "(model-scoped M3 registry; the other model owns no "
          "checkpoints)", c4 == 404, f"{c4}")
    c5, leak = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/suite-runs")
    check("I2 none of the 10 run ids is reachable under the other model",
          c5 == 200 and leak == []
          and all(rid not in (leak or []) for rid in RUN_IDS))
    c6, one = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                               f"{RUN_IDS[0]}")
    check("I3 by-checkpoint does not shadow the M10 detail getter",
          c6 == 200 and one["suite_run_id"] == RUN_IDS[0])
    c7, _ = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                             "ghost-run-25")
    check("I4 unknown run id via detail getter still 404", c7 == 404,
          f"{c7}")

    print("== LIVE J: regression + final audit (zero storage growth) ==")
    code, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   "records")
    code4, bs2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-sample/{SAMPLE}")
    code5, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/{CKPT}")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                 f"{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                 f"{SUITE}/summary")
    code8, ev2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"by-checkpoint/{CKPT}")
    code9, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"by-policy/{POLICY}")
    check("J1 M18/M19/M20/M21/M22/M23/M24 outputs unchanged",
          code == 200 and sq2 == sq and code3 == 200 and rec2 == rec
          and code4 == 200 and bs2 == bs and code5 == 200 and bc2 == bc
          and code6 == 200 and g2 == g and code7 == 200 and s2 == s
          and code8 == 200 and ev2 == ev and code9 == 200 and gd2 == gd)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("J2 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code, runs2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("J3 M10 listing, M11 workflow history, M3 checkpoint registry "
          "and M9 policy registry byte-identical to pre-state",
          code == 200 and runs2 == runs and code_w2 == 200
          and wfs2 == wfs and code_c2 == 200 and cks2 == cks
          and code_p2 == 200 and pol2 == pol)
    code, grouped2 = call_json("GET", url)
    check("J4 by-checkpoint still deterministic at the end",
          code == 200 and grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("J5 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("J6 zero new files", new_files == set(), f"{len(new_files)} new")
    check("J7 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M25 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M25 live smoke OK: narrow read-only by-checkpoint grouping of "
          "the M10 suite-run history (exact listing parity for the live "
          "checkpoint — all 10 runs —, 200 + [] for the two valid empty "
          "checkpoints, partition 10+0+0=10 with no duplication, clean "
          "404s incl. cross-model isolation through the model-scoped M3 "
          "registry, deterministic byte-identical repeats, "
          "M10/M11/M18-M24 surfaces + dashboard hash + registries "
          "unchanged, ZERO production storage growth) verified live on "
          "production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
