"""M20 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M20 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/sample-quality/by-checkpoint/{checkpoint_id},
answering "which immutable sample-quality evaluations belong to this
checkpoint?" — the model's authoritative M16 listing filtered by the
persisted checkpoint_id recorded in each SampleEvaluationRecord, after
ownership validation through the model's M3 checkpoint registry. The smoke
must prove exact M16/M18/M19 parity for the measured checkpoint, 200 + []
for valid unmeasured checkpoints, clean 404s (unknown model / unknown
checkpoint / another model's checkpoint), byte-identical repeats, M19
by-sample output unchanged, and ZERO production storage growth.

Production facts (audited at Phase A): model 4a0a871886ef owns checkpoints
025e6d8d8f15, 0511de4c7372, 30a8bc5b82ab and the 2 M16 sample-evaluations
(evaluation ids 8ff910cf2a9e then 31a283413c75 ASCENDING; both recorded
under checkpoint 0511de4c7372 of sample f8e66f9c7b50; loss 6.191012 /
perplexity 488.340243; result_hash cb8e29f2...); model b5bc905326b6 exists
but owns no checkpoints (cross-model isolation probe); M17 dashboard
result_hash f48557fe8ab1... — 96 files / 4,002,745 B / 0 .tmp.

Phase A (baseline audit): exact 96 / 4,002,745 / 0 audit + full per-file
hash inventory saved to /tmp/m20-smoke-baseline-inventory.json + artifact
counts + M16 listing / M18 records / M19 by-sample / dashboard pre-state.

Phase B (new endpoint): by-checkpoint of 0511de4c7372 -> 200 with EXACTLY
the two records in ASCENDING order; payload parity with the M18 records
listing, the authoritative M16 listing and the M19 by-sample response
(filtered by the recorded checkpoint identity); metric values present
(loss 6.191012, perplexity 488.340243); raw-response byte-identical
repeat; valid unmeasured checkpoints 025e6d8d8f15 and 30a8bc5b82ab ->
200 + []; no derived statistics in payloads.

Phase C (404 semantics, storage-neutral): unknown model by-checkpoint ->
404; unknown well-formed checkpoint id -> 404; CROSS-MODEL probe — model
b5bc905326b6 with the real checkpoint id 0511de4c7372 (owned by another
model) -> 404; M19 by-sample unknown model still 404.

Phase D (final audit): inventory diff vs Phase A — every pre-existing file
byte-identical, ZERO new files, zero .tmp, zero storage growth; artifact
counts unchanged; M17 dashboard result_hash unchanged; M16 listing, M18
records listing and M19 by-sample output unchanged.

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
OTHER_MODEL = "b5bc905326b6"          # exists, owns NO checkpoints
CKPT = "0511de4c7372"                 # the measured production checkpoint
EMPTY_CKPTS = ("025e6d8d8f15", "30a8bc5b82ab")   # valid, zero measurements
SAMPLE = "f8e66f9c7b50"               # the measured production sample
UNKNOWN_CKPT = "ffffffffffff"         # well-formed 12-hex, nonexistent
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
INVENTORY = Path("/tmp/m20-smoke-baseline-inventory.json")

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
    code4, bs = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-sample/{SAMPLE}")
    code5, cks = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code6, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A3 counts: 4 samples / 2 sample-evaluations / 2 M18 records / "
          "3 checkpoints", code == 200 and len(smp) == 4 and code2 == 200
          and len(sq) == 2 and code3 == 200 and len(rec) == 2
          and code5 == 200 and len(cks) == 3)
    check("A4 evals are the known two, both recorded under 0511de4c7372 "
          "of sample f8e66f9c7b50",
          {x["evaluation_id"] for x in sq} == set(EVAL_IDS)
          and {x["checkpoint_id"] for x in sq} == {CKPT}
          and {x["sample_id"] for x in sq} == {SAMPLE})
    check("A5 M18 records == M16 listing; M19 by-sample == same payloads",
          rec == sq and bs == rec and len(bs) == 2)
    check("A6 M17 dashboard hash begins f48557fe8ab1",
          code6 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])

    print("== Phase B: new endpoint /sample-quality/by-checkpoint ==")
    url = (f"{BASE}/models/{MODEL}/sample-quality/by-checkpoint/{CKPT}")
    code, raw1 = call("GET", url)
    cks2 = json.loads(raw1)
    check("B1 HTTP 200 with exactly the two M16 records",
          code == 200 and len(cks2) == 2, f"{code}")
    check("B2 ids + ASCENDING (created_at, evaluation_id) order",
          [x["evaluation_id"] for x in cks2] == list(EVAL_IDS)
          and [(x["created_at"], x["evaluation_id"]) for x in cks2]
          == sorted((x["created_at"], x["evaluation_id"]) for x in cks2))
    check("B3 every record carries the requested persisted checkpoint",
          all(x["checkpoint_id"] == CKPT for x in cks2))
    check("B4 parity with the M18 records listing filtered by the "
          "checkpoint", cks2 == [x for x in rec if x["checkpoint_id"] == CKPT])
    check("B5 parity with the authoritative M16 listing filtered by the "
          "checkpoint", cks2 == [x for x in sq if x["checkpoint_id"] == CKPT])
    check("B6 parity with M19 by-sample (one checkpoint, one sample)",
          cks2 == bs)
    for x in cks2:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"{x['evaluation_id']}")
        check(f"B7 M16 getter parity for {x['evaluation_id']}",
              c == 200 and one == x)
    check("B8 metric values present and exact (loss 6.191012 / "
          "ppl 488.340243)",
          all(x["loss_nats"] == 6.191012 and x["perplexity"] == 488.340243
              for x in cks2))
    check("B9 full verbatim payloads, no derived statistics",
          all(set(x) == RECORD_FIELDS and
              set(x) == set(sq[0]) and set(x) == set(rec[0])
              and set(x) == set(bs[0]) for x in cks2))
    code2, raw2 = call("GET", url)
    check("B10 repeated GET raw-byte-identical", code2 == 200 and raw2 == raw1)
    code3, raw3 = call("GET", url)
    check("B11 second repeat raw-byte-identical", code3 == 200 and raw3 == raw1)
    for cid in EMPTY_CKPTS:
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   f"by-checkpoint/{cid}")
        check(f"B12 valid unmeasured checkpoint {cid} -> 200 + []",
              c == 200 and body == [], f"{c}")
    check("B13 M16/M17/M18/M19 surfaces unchanged by the reads",
          call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")[1] == sq
          and call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                               "records")[1] == rec
          and call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                               f"by-sample/{SAMPLE}")[1] == bs
          and call_json("GET", f"{BASE}/models/{MODEL}/dashboard")[1]
          ["result_hash"] == DASH_HASH)

    print("== Phase C: 404 semantics (storage-neutral) ==")
    c1, _ = call_json("GET", f"{BASE}/models/no-such-model-20/"
                             "sample-quality/by-checkpoint/0511de4c7372")
    check("C1 unknown model by-checkpoint -> 404", c1 == 404, f"{c1}")
    c2, _ = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                             f"by-checkpoint/{UNKNOWN_CKPT}")
    check("C2 unknown well-formed checkpoint id -> 404", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/sample-quality/"
                             f"by-checkpoint/{CKPT}")
    check("C3 cross-model isolation: another model + this forge's real "
          "checkpoint id -> 404", c3 == 404, f"{c3}")
    c4, _ = call_json("GET", f"{BASE}/models/no-such-model-20/sample-quality/"
                             f"by-sample/{SAMPLE}")
    check("C4 M19 by-sample unknown model still 404", c4 == 404, f"{c4}")
    c5, _ = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                             "by-checkpoint/ckpt-0000000000000000000000")
    check("C5 malformed unknown checkpoint id -> 404 (no crash)",
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
    code, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code2, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                    "records")
    code3, bs2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-sample/{SAMPLE}")
    check("D4 M16 listing, M18 records and M19 by-sample byte-identical "
          "to pre-state", code == 200 and sq2 == sq and code2 == 200
          and rec2 == rec and code3 == 200 and bs2 == bs)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("D5 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH
          and d2 == d0 and d2["diagnostics"] == [])

    print()
    if FAILURES:
        print(f"M20 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M20 live smoke OK: narrow read-only by-checkpoint access path "
          "(exact M16-record parity for the measured checkpoint, 200 + [] "
          "for unmeasured checkpoints, clean 404s incl. cross-model "
          "isolation, deterministic byte-identical repeats, M19 by-sample "
          "unchanged, ZERO production storage growth) verified live on "
          "production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
