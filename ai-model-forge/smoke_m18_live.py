"""M18 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M18 is READ-ONLY: it adds one dedicated
listing route, GET /models/{id}/sample-quality/records, returning the FULL
immutable M16 SampleEvaluationRecord payloads (loss_nats/perplexity
included) in the authoritative (created_at, evaluation_id) ASCENDING
order — the smoke must prove exact M16-record parity and ZERO production
storage growth.

Production facts (audited at Phase A): model 4a0a871886ef (vocab 640,
context 64), checkpoint 0511de4c7372, tokenizer 99106e3255c5 (vocab 320),
4 samples, 2 sample-evaluations of sample f8e66f9c7b50 (evaluation ids
8ff910cf2a9e and 31a283413c75, result_hash cb8e29f2...), 16 M4
evaluations, 13 workflows, 10 suite-runs, 7 recipes; M17 dashboard
result_hash f48557fe8ab1... — 96 files / 4,002,745 B / 0 .tmp.

Phase A (baseline audit): exact 96 / 4,002,745 / 0 audit + full per-file
hash inventory saved to /tmp/m18-smoke-baseline-inventory.json + artifact
counts + dashboard pre-state (hash starts f48557fe8ab1).

Phase B (new endpoint): GET /models/4a0a871886ef/sample-quality/records ->
200 with EXACTLY the two M16 records ordered by (created_at,
evaluation_id) ASCENDING (8ff910cf2a9e first, 31a283413c75 second);
payload parity with the individual M16 getter responses; metric values
present (loss_nats 6.191012, perplexity 488.340243); byte-identical
repeat; parity with the authoritative M16 reference listing; payloads
carry no derived statistics.

Phase C (unknown model): GET the endpoint with an unknown model -> 404,
zero storage change.

Phase D (final audit): inventory diff vs Phase A — every pre-existing file
byte-identical, ZERO new files, zero .tmp, zero storage growth; sample /
sample-evaluation / M4 evaluation / checkpoint / tokenizer / dataset /
workflow / recipe / policy counts unchanged; M17 dashboard result_hash
still f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838.

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"
RECORDS_URL = (f"{BASE}/models/{MODEL}/sample-quality/records")
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")     # ASCENDING (created_at, id)
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m18-smoke-baseline-inventory.json")

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
    code, smp = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, ev = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code4, wf = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code5, sr = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    code6, rec = call_json("GET", f"{BASE}/workflows/recipes")
    check("A3 counts: 4 samples / 2 sample-evaluations / 16 evals / 13 "
          "workflows / 10 suite-runs / 7 recipes",
          code == 200 and len(smp) == 4 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(ev) == 16 and code4 == 200
          and len(wf) == 13 and code5 == 200 and len(sr) == 10
          and code6 == 200 and len(rec) == 7)
    check("A4 sample-evaluations are the known two of f8e66f9c7b50",
          {x["evaluation_id"] for x in sq} == set(EVAL_IDS)
          and {x["sample_id"] for x in sq} == {"f8e66f9c7b50"})
    code, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A5 M17 dashboard hash begins f48557fe8ab1",
          code == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])

    print("== Phase B: new endpoint /sample-quality/records ==")
    code, recs = call_json("GET", RECORDS_URL)
    check("B1 HTTP 200 with exactly the two M16 records",
          code == 200 and len(recs) == 2, f"{code}")
    check("B2 ids + ASCENDING (created_at, evaluation_id) order",
          [x["evaluation_id"] for x in recs] == list(EVAL_IDS)
          and [(x["created_at"], x["evaluation_id"]) for x in recs]
          == sorted((x["created_at"], x["evaluation_id"]) for x in recs))
    # payload parity with the authoritative M16 listing + individual getters
    check("B3 records == M16 reference listing payloads",
          recs == sq)
    for x in recs:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"{x['evaluation_id']}")
        check(f"B4 getter parity for {x['evaluation_id']}",
              c == 200 and one == x)
    # exact metric values of the known production measurement
    check("B5 metric values present and exact (loss 6.191012 / "
          "ppl 488.340243)",
          all(x["loss_nats"] == 6.191012 and x["perplexity"] == 488.340243
              for x in recs))
    check("B6 no derived statistics in the payload",
          all(set(x) == {"evaluation_id", "model_id", "sample_id",
                         "sample_result_hash", "token_sequence_sha256",
                         "checkpoint_id", "checkpoint_weights_sha256",
                         "tokenizer_id", "tokenizer_hash",
                         "prompt_token_count", "generated_token_count",
                         "evaluated_token_count", "context_length",
                         "window_token_count", "window_rule", "loss_nats",
                         "perplexity", "result_hash", "hardware",
                         "created_at", "duration_seconds", "schema_version"}
              for x in recs))
    code2, recs2 = call_json("GET", RECORDS_URL)
    check("B7 repeated GET byte-identical",
          code2 == 200 and recs2 == recs)
    check("B8 M16/M17 surfaces unchanged by the read",
          call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")[1] == sq
          and call_json("GET", f"{BASE}/models/{MODEL}/dashboard")[1]
          ["result_hash"] == DASH_HASH)

    print("== Phase C: unknown model (zero storage change) ==")
    c1, _ = call_json("GET", f"{BASE}/models/no-such-model-18/"
                             "sample-quality/records")
    check("C1 unknown model records -> 404", c1 == 404, f"{c1}")

    print("== Phase D: final audit (zero production storage growth) ==")
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    check("D1 every pre-existing file byte-identical", changed == [],
          f"{len(changed)} changed")
    check("D2 zero new files", new_files == set(), f"{len(new_files)} new")
    check("D3 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")
    code, smp2 = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code2, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, ev2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code4, wf2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code5, sr2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs")
    code6, rec2 = call_json("GET", f"{BASE}/workflows/recipes")
    check("D4 all artifact families unchanged",
          code == 200 and len(smp2) == 4 and code2 == 200 and len(sq2) == 2
          and code3 == 200 and len(ev2) == 16 and code4 == 200
          and len(wf2) == 13 and code5 == 200 and len(sr2) == 10
          and code6 == 200 and len(rec2) == 7)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("D5 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH
          and d2 == d0 and d2["diagnostics"] == [])

    print()
    if FAILURES:
        print(f"M18 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M18 live smoke OK: dedicated read-only sample-quality records "
          "endpoint (exact M16-record parity, metric values exposed, "
          "deterministic repeats, ZERO production storage growth) verified "
          "live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
