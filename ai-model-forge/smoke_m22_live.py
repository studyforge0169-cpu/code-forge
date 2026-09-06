"""M22 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M22 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/suite-runs/by-suite/{suite_id}/summary,
answering "how many suite runs exist for this model and suite, which
ones, and when did they run?" — a pure derived counting view over the
M21 grouping (model + M9 suite-registry validation, persisted suite_id
filter, deterministic M10 order). The smoke must prove exact M21-listing
parity for the live suite (count, ordered ids, earliest/latest
timestamps), a ZERO summary for a valid suite with no runs under another
model, clean 404s (unknown model / unknown suite), cross-model isolation
(no counted/leaked run ids), byte-identical repeats, unchanged
M10/M18/M19/M20/M21 surfaces + dashboard hash, and ZERO production
storage growth.

Production facts (audited at Phase A): model 4a0a871886ef owns all 10
suite-run records (suite m9-live-suite, status completed, probe_count 2,
recorded under checkpoint 0511de4c7372; ASCENDING (created_at,
suite_run_id) order aae8e8a8ba9c, bb114d2ce42e, 8f8aee834c9f,
d9742459017b, 5684649d0ced, 2af508035191, 4cb388a21ce2, 2dc9f7dff400,
2d203398304b, e4b1c2a7fb2d; earliest created_at
2026-09-04T09:21:11.493507Z, latest 2026-09-04T10:46:30.123758Z); model
b5bc905326b6 exists but owns NO suite-run records (cross-model /
zero-summary probe); suite m9-live-suite is the single registered M9
probe suite; M16-era sample-quality facts (evals 8ff910cf2a9e /
31a283413c75 under checkpoint 0511de4c7372 of sample f8e66f9c7b50,
M17 dashboard result_hash f48557fe8ab1...) — 96 files / 4,002,745 B /
0 .tmp.

Phase A (baseline audit): exact 96 / 4,002,745 / 0 audit + full per-file
hash inventory saved to /tmp/m22-smoke-baseline-inventory.json + model &
suite registry facts + M10 listing / M21 by-suite / M18 records / M19
by-sample / M20 by-checkpoint / dashboard pre-state.

Phase B (new endpoint): summary of m9-live-suite on 4a0a871886ef -> 200
with total_count EXACTLY 10, run_ids equal to the M21 by-suite listing
in the ASCENDING order, earliest/latest timestamps equal to the
listing's min/max created_at (and to the known production timestamps);
exact bookkeeping-only field set; raw-response byte-identical repeat;
M21 by-suite unchanged by the reads.

Phase C (404 semantics, cross-model isolation, storage-neutral): unknown
model summary -> 404; unknown well-formed suite id -> 404; malformed
(percent-encoded) suite id -> 404 (no crash); CROSS-MODEL probe — model
b5bc905326b6 (real) + the real suite m9-live-suite -> 200 + ZERO summary
(total_count 0, empty run_ids, null timestamps) and none of the 10 run
ids is counted or leaked; unknown suite under the other model -> 404;
the summary route does not shadow the M21 by-suite records route or the
M10 detail getter.

Phase D (final audit): M18 records / M19 by-sample / M20 by-checkpoint
output unchanged; M17 dashboard result_hash unchanged; M10 listing and
M21 by-suite byte-identical to pre-state; summary still deterministic at
the end; inventory diff vs Phase A — every pre-existing file
byte-identical, ZERO new files, zero .tmp, zero storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8743 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8743/api/v1"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, owns NO suite-run records
SUITE = "m9-live-suite"               # the live production suite (10 runs)
UNKNOWN_SUITE = "m-ghost-suite-22"    # well-formed, nonexistent
RUN_IDS = ("aae8e8a8ba9c", "bb114d2ce42e", "8f8aee834c9f", "d9742459017b",
           "5684649d0ced", "2af508035191", "4cb388a21ce2", "2dc9f7dff400",
           "2d203398304b", "e4b1c2a7fb2d")   # ASCENDING (created_at, id)
EARLIEST = "2026-09-04T09:21:11.493507Z"     # aae8e8a8ba9c
LATEST = "2026-09-04T10:46:30.123758Z"       # e4b1c2a7fb2d
CKPT = "0511de4c7372"                 # every run's recorded checkpoint
SAMPLE = "f8e66f9c7b50"               # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")     # ASCENDING (created, id)
SUMMARY_FIELDS = {"model_id", "suite_id", "total_count", "run_ids",
                  "earliest_created_at", "latest_created_at"}
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m22-smoke-baseline-inventory.json")

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
    code_s, suite = call_json("GET", f"{BASE}/probe-suites/{SUITE}")
    check("A3 both models + the suite resolve in the registries",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_s == 200
          and suite["suite_id"] == SUITE)
    code, runs = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    code_g, grouped = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                       f"by-suite/{SUITE}")
    check("A4 M10 listing + M21 by-suite pre-state: exactly the 10 known "
          "runs in ASCENDING order",
          code == 200 and len(runs) == 10 and code_g == 200
          and grouped == runs
          and [x["suite_run_id"] for x in grouped] == list(RUN_IDS))
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  "records")
    code4, bs = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-sample/{SAMPLE}")
    code5, bc = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-checkpoint/{CKPT}")
    code6, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A5 M16/M18/M19/M20 pre-state intact (2 records, 200s)",
          code2 == 200 and len(sq) == 2 and code3 == 200 and len(rec) == 2
          and code4 == 200 and len(bs) == 2 and code5 == 200
          and len(bc) == 2 and {x["evaluation_id"] for x in sq}
          == set(M16_EVAL_IDS))
    check("A6 M17 dashboard hash begins f48557fe8ab1",
          code6 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])

    print("== Phase B: new endpoint /suite-runs/by-suite/summary ==")
    url = f"{BASE}/models/{MODEL}/suite-runs/by-suite/{SUITE}/summary"
    code, raw1 = call("GET", url)
    s = json.loads(raw1)
    check("B1 HTTP 200 with exactly the bookkeeping field set",
          code == 200 and set(s) == SUMMARY_FIELDS, f"{code}")
    check("B2 total_count is EXACTLY 10", s.get("total_count") == 10,
          str(s.get("total_count")))
    check("B3 run_ids equal the M21 by-suite listing in the ASCENDING "
          "order", s.get("run_ids") == [x["suite_run_id"] for x in grouped]
          == list(RUN_IDS))
    check("B4 earliest/latest equal the listing's min/max created_at "
          "(and the known production timestamps)",
          s.get("earliest_created_at")
          == min(x["created_at"] for x in grouped) == EARLIEST
          and s.get("latest_created_at")
          == max(x["created_at"] for x in grouped) == LATEST)
    check("B5 summary agrees with a manual count of the M21 listing",
          s["total_count"] == len(grouped)
          and s["run_ids"] == [x["suite_run_id"] for x in grouped])
    check("B6 identity fields carry the requested model + suite",
          s["model_id"] == MODEL and s["suite_id"] == SUITE)
    code2, raw2 = call("GET", url)
    check("B7 repeated GET raw-byte-identical", code2 == 200 and raw2 == raw1)
    code3, raw3 = call("GET", url)
    check("B8 second repeat raw-byte-identical",
          code3 == 200 and raw3 == raw1)
    check("B9 M21 by-suite records unchanged by the reads",
          call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                           f"{SUITE}")[1] == grouped)

    print("== Phase C: 404s + cross-model isolation (zero summary) ==")
    c1, _ = call_json("GET", f"{BASE}/models/no-such-model-22/"
                             f"suite-runs/by-suite/{SUITE}/summary")
    check("C1 unknown model summary -> 404", c1 == 404, f"{c1}")
    c2, _ = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                             f"{UNKNOWN_SUITE}/summary")
    check("C2 unknown well-formed suite id -> 404", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                             "suite%20id%20with%20spaces!!/summary")
    check("C3 malformed unknown suite id -> 404 (no crash)",
          c3 == 404, f"{c3}")
    c4, zero = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/suite-runs/"
                                f"by-suite/{SUITE}/summary")
    check("C4 cross-model isolation: other real model + the real suite -> "
          "200 + ZERO summary",
          c4 == 200 and zero == {"model_id": OTHER_MODEL, "suite_id": SUITE,
                                 "total_count": 0, "run_ids": [],
                                 "earliest_created_at": None,
                                 "latest_created_at": None}, f"{c4}")
    check("C5 none of the 10 run ids is counted or leaked",
          all(rid not in (zero or {}).get("run_ids", []) for rid in RUN_IDS)
          and zero["total_count"] == 0)
    c6, _ = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/suite-runs/"
                             f"by-suite/{UNKNOWN_SUITE}/summary")
    check("C6 unknown suite under the other model -> 404", c6 == 404,
          f"{c6}")
    c7, recs = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                f"by-suite/{SUITE}")
    check("C7 summary route does not shadow the M21 records route",
          c7 == 200 and len(recs) == 10
          and [x["suite_run_id"] for x in recs] == list(RUN_IDS))
    c8, one = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                               f"{RUN_IDS[0]}")
    check("C8 M10 detail getter still resolves a real run id",
          c8 == 200 and one["suite_run_id"] == RUN_IDS[0])
    c9, _ = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                             "ghost-run-22")
    check("C9 unknown run id via detail getter still 404", c9 == 404,
          f"{c9}")

    print("== Phase D: prior endpoints + final audit (zero storage growth) ==")
    code, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code2, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   "records")
    code3, bs2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-sample/{SAMPLE}")
    code4, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/{CKPT}")
    check("D1 M18 records / M19 by-sample / M20 by-checkpoint unchanged",
          code == 200 and sq2 == sq and code2 == 200 and rec2 == rec
          and code3 == 200 and bs2 == bs and code4 == 200 and bc2 == bc)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("D2 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code, runs2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    code_g2, grouped2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                         f"by-suite/{SUITE}")
    check("D3 M10 listing + M21 by-suite byte-identical to pre-state",
          code == 200 and runs2 == runs and code_g2 == 200
          and grouped2 == grouped)
    code, s2 = call_json("GET", url)
    check("D4 summary still deterministic at the end",
          code == 200 and s2 == s)
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
        print(f"M22 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M22 live smoke OK: narrow read-only per-suite bookkeeping "
          "summary (exact M21-listing parity for the live suite — 10 runs, "
          "ordered ids, earliest/latest timestamps —, ZERO summary for a "
          "valid suite without runs under another model, clean 404s incl. "
          "cross-model isolation, deterministic byte-identical repeats, "
          "M10/M18/M19/M20/M21 surfaces + dashboard hash unchanged, ZERO "
          "production storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
