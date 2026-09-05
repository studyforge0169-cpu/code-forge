"""M19 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M19 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/sample-quality/by-sample/{sample_id},
exposing M16 sample-quality history grouped by ONE known generated sample
of the model — full verbatim SampleEvaluationRecord payloads in the
authoritative M16/M18 (created_at, evaluation_id) ASCENDING order. The
smoke must prove exact M16/M18 parity for the measured sample, 200 + []
for a valid unmeasured sample, clean 404s (unknown model / unknown
sample / another model's sample), byte-identical repeats and ZERO
production storage growth.

Production facts (audited at Phase A): model 4a0a871886ef owns all 4
samples (05820e8bc68a, 4e8463e0df06, e2b5166fa549, f8e66f9c7b50) and the
2 M16 sample-evaluations of sample f8e66f9c7b50 (evaluation ids
8ff910cf2a9e then 31a283413c75 ASCENDING; loss 6.191012 / perplexity
488.340243; result_hash cb8e29f2...); model b5bc905326b6 exists but owns
no samples (cross-model isolation probe); M17 dashboard result_hash
f48557fe8ab1... — 96 files / 4,002,745 B / 0 .tmp.

Phase A (baseline audit): exact 96 / 4,002,745 / 0 audit + full per-file
hash inventory saved to /tmp/m19-smoke-baseline-inventory.json + artifact
counts + M16 listing / M18 records listing / dashboard pre-state.

Phase B (new endpoint): by-sample of f8e66f9c7b50 -> 200 with EXACTLY
the two records in ASCENDING order; payload parity with the M18 records
listing and the authoritative M16 listing filtered to that sample; getter
parity per record; metric values present (loss 6.191012, perplexity
488.340243); raw-response byte-identical repeat; valid unmeasured sample
05820e8bc68a (and the two other unmeasured ones) -> 200 + []; no derived
statistics in payloads.

Phase C (404 semantics, storage-neutral): unknown model by-sample -> 404;
unknown well-formed sample id -> 404; CROSS-MODEL probe — model
b5bc905326b6 with the real sample id f8e66f9c7b50 (owned by another
model) -> 404; M18 records endpoint unknown model still 404.

Phase D (final audit): inventory diff vs Phase A — every pre-existing
file byte-identical, ZERO new files, zero .tmp, zero storage growth;
artifact counts unchanged; M17 dashboard result_hash unchanged; M16
listing and M18 records listing unchanged; M15 sample listing unchanged.

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
OTHER_MODEL = "b5bc905326b6"          # exists, owns NO samples
SAMPLE = "f8e66f9c7b50"               # the measured production sample
EMPTY_SAMPLES = ("05820e8bc68a", "4e8463e0df06", "e2b5166fa549")
UNKNOWN_SAMPLE = "ffffffffffff"       # well-formed 12-hex, nonexistent
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")     # ASCENDING (created_at, id)
RECORD_FIELDS = {"evaluation_id", "model_id", "sample_id", "sample_result_hash",
                 "token_sequence_sha256", "checkpoint_id",
                 "checkpoint_weights_sha256", "tokenizer_id", "tokenizer_hash",
                 "prompt_token_count", "generated_token_count",
                 "evaluated_token_count", "context_length",
                 "window_token_count", "window_rule", "loss_nats",
                 "perplexity", "result_hash", "hardware", "created_at",
                 "duration_seconds", "schema_version"}
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m19-smoke-baseline-inventory.json")

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
    code3, rec = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  "records")
    code4, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A3 counts: 4 samples / 2 sample-evaluations / 2 M18 records",
          code == 200 and len(smp) == 4 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(rec) == 2)
    check("A4 evals are the known two of sample f8e66f9c7b50",
          {x["evaluation_id"] for x in sq} == set(EVAL_IDS)
          and {x["sample_id"] for x in sq} == {SAMPLE})
    check("A5 M18 records listing == M16 listing (payload parity)",
          rec == sq)
    check("A6 M17 dashboard hash begins f48557fe8ab1",
          code4 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])

    print("== Phase B: new endpoint /sample-quality/by-sample ==")
    url = (f"{BASE}/models/{MODEL}/sample-quality/by-sample/{SAMPLE}")
    code, raw1 = call("GET", url)
    recs = json.loads(raw1)
    check("B1 HTTP 200 with exactly the two M16 records",
          code == 200 and len(recs) == 2, f"{code}")
    check("B2 ids + ASCENDING (created_at, evaluation_id) order",
          [x["evaluation_id"] for x in recs] == list(EVAL_IDS)
          and [(x["created_at"], x["evaluation_id"]) for x in recs]
          == sorted((x["created_at"], x["evaluation_id"]) for x in recs))
    check("B3 parity with the M18 records listing filtered to the sample",
          recs == [x for x in rec if x["sample_id"] == SAMPLE])
    check("B4 parity with the authoritative M16 listing filtered to the "
          "sample", recs == [x for x in sq if x["sample_id"] == SAMPLE])
    for x in recs:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"{x['evaluation_id']}")
        check(f"B5 M16 getter parity for {x['evaluation_id']}",
              c == 200 and one == x)
    check("B6 metric values present and exact (loss 6.191012 / "
          "ppl 488.340243)",
          all(x["loss_nats"] == 6.191012 and x["perplexity"] == 488.340243
              for x in recs))
    check("B7 full verbatim payloads, no derived statistics",
          all(set(x) == RECORD_FIELDS and
              set(x) == set(sq[0]) and set(x) == set(rec[0])
              for x in recs))
    code2, raw2 = call("GET", url)
    check("B8 repeated GET raw-byte-identical",
          code2 == 200 and raw2 == raw1)
    code3, raw3 = call("GET", url)
    check("B9 second repeat raw-byte-identical", code3 == 200 and raw3 == raw1)
    for sid in EMPTY_SAMPLES:
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   f"by-sample/{sid}")
        check(f"B10 valid unmeasured sample {sid} -> 200 + []",
              c == 200 and body == [], f"{c}")
    check("B11 M16/M17/M18 surfaces unchanged by the reads",
          call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")[1] == sq
          and call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                               "records")[1] == rec
          and call_json("GET", f"{BASE}/models/{MODEL}/dashboard")[1]
          ["result_hash"] == DASH_HASH)

    print("== Phase C: 404 semantics (storage-neutral) ==")
    c1, _ = call_json("GET", f"{BASE}/models/no-such-model-19/"
                             "sample-quality/by-sample/05820e8bc68a")
    check("C1 unknown model by-sample -> 404", c1 == 404, f"{c1}")
    c2, _ = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                             f"by-sample/{UNKNOWN_SAMPLE}")
    check("C2 unknown well-formed sample id -> 404", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/sample-quality/"
                             f"by-sample/{SAMPLE}")
    check("C3 cross-model isolation: another model + this forge's real "
          "sample id -> 404", c3 == 404, f"{c3}")
    c4, _ = call_json("GET", f"{BASE}/models/no-such-model-19/"
                             "sample-quality/records")
    check("C4 M18 records endpoint unknown model still 404", c4 == 404,
          f"{c4}")
    c5, _ = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                             "by-sample/sample-000000000000000000000000")
    check("C5 over-long unknown sample id -> 404 (no id-format crash)",
          c5 == 404, f"{c5}")

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
    code3, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                    "records")
    check("D4 all artifact families unchanged",
          code == 200 and len(smp2) == 4 and code2 == 200 and len(sq2) == 2
          and code3 == 200 and len(rec2) == 2 and smp2 == smp)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("D5 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH
          and d2 == d0 and d2["diagnostics"] == [])
    check("D6 M16 listing and M18 records listing byte-identical to "
          "pre-state", sq2 == sq and rec2 == rec)

    print()
    if FAILURES:
        print(f"M19 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M19 live smoke OK: narrow read-only by-sample access path "
          "(exact M16-record parity for the measured sample, 200 + [] for "
          "unmeasured samples, clean 404s incl. cross-model isolation, "
          "deterministic byte-identical repeats, ZERO production storage "
          "growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
