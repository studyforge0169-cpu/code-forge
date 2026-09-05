"""M17 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M17 is a READ-ONLY dashboard
observability milestone: GET /dashboard gains a deterministic
``sample_quality`` section derived live from the immutable M16
``sample-evaluations/`` records — the smoke must prove ZERO production
storage growth.

Production facts (audited at Phase A): model 4a0a871886ef (config vocab
640, context 64), checkpoint 0511de4c7372, the ONLY stored tokenizer
99106e3255c5 (actual vocab 320), 4 immutable M15 samples under
samples/4a0a871886ef/ (including f8e66f9c7b50), 2 immutable M16
sample-evaluations of sample f8e66f9c7b50 (evaluation-8ff910cf2a9e and
evaluation-31a283413c75, both result_hash cb8e29f2...), 16 M4 evaluations,
13 workflows, 10 suite-runs, 7 recipes — 96 files / 4,002,745 B / 0 .tmp.

Phase A (baseline audit): exact storage audit (96 files / 4,002,745 B /
0 .tmp — the M16 certified handoff), full per-file hash inventory saved to
/tmp/m17-smoke-baseline-inventory.json, artifact counts verified over the
API, dashboard pre-state captured (result_hash + sample_quality section
present, total_count == 2).

Phase B (dashboard sample-quality verification): GET the production
dashboard; sample_quality.total_count == 2; both production evaluation ids
represented; by_sample/latest correspond to sample f8e66f9c7b50 and the
correct history; by_sample and latest match the live M16 listing exactly
(the M16 SampleQualityEngine listing through GET sample-quality). Repeated
GET /dashboard must be byte-identical with an identical result_hash;
read-only proof by hash inventory.

Phase C (read-only failure test): unknown model -> 404 with zero storage
change (GET /models/no-such-model/dashboard). No production artifact is
corrupted or modified.

Phase D (final audit): inventory diff vs Phase A — every pre-existing file
byte-identical; ZERO new files; zero .tmp; zero storage growth; samples,
sample-evaluations, M4 evaluations, workflows, recipes, checkpoints,
tokenizers and datasets unchanged; dashboard still deterministic with the
same result_hash.

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
MODEL = "4a0a871886ef"          # production model with the full history
SAMPLE = "f8e66f9c7b50"         # the sampled production sample (M15/M16)
EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")   # both M16 records (Phase A
#                                               verifies them before use)
CKPT = "0511de4c7372"           # its kept checkpoint (suite evidence)
TOK = "99106e3255c5"            # the ONLY stored tokenizer (vocab 320)
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m17-smoke-baseline-inventory.json")

PRE_FILES, PRE_BYTES, PRE_TMP = 96, 4_002_745, 0

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def call(method: str, path: str, body=None):
    req = urllib.request.Request(BASE + path, method=method)
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
    code, m = call_json("GET", f"/models/{MODEL}")
    check("A3 model exists with vocab 640 / context 64",
          code == 200 and m["config"]["vocab_size"] == 640
          and m["config"]["context_length"] == 64, f"{code}")
    code, smp = call_json("GET", f"/models/{MODEL}/samples")
    check("A4 4 samples present incl. f8e66f9c7b50",
          code == 200 and len(smp) == 4
          and SAMPLE in {s["sample_id"] for s in smp})
    code, sq = call_json("GET", f"/models/{MODEL}/sample-quality")
    check("A5 2 sample-evaluations present with the known ids/hash",
          code == 200 and len(sq) == 2
          and {x["evaluation_id"] for x in sq} == set(EVAL_IDS)
          and {x["result_hash"] for x in sq}
          == {"cb8e29f263bfaf95d05cbba43c73fb3f46acb427bbdd86433fd5fc325107c248"}
          and {x["sample_id"] for x in sq} == {SAMPLE})
    code, ev = call_json("GET", f"/models/{MODEL}/evaluations")
    code2, wf = call_json("GET", f"/models/{MODEL}/workflows")
    code3, sr = call_json("GET", f"/models/{MODEL}/suite-runs")
    code4, rec = call_json("GET", "/workflows/recipes")
    check("A6 counts: 16 evals / 13 workflows / 10 suite-runs / 7 recipes",
          code == 200 and len(ev) == 16 and code2 == 200 and len(wf) == 13
          and code3 == 200 and len(sr) == 10 and code4 == 200
          and len(rec) == 7, f"evals={len(ev)} workflows={len(wf)}")
    code, d0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A7 dashboard pre-state: 200, section present, total_count == 2",
          code == 200 and d0["sample_quality"]["total_count"] == 2
          and len(d0["result_hash"]) == 64, f"{code}")

    print("== Phase B: dashboard sample-quality verification ==")
    sq_sec = d0["sample_quality"]
    # both production evaluation ids represented in `latest`
    check("B1 both evaluation ids in the section's latest list",
          {x["evaluation_id"] for x in sq_sec["latest"]} == set(EVAL_IDS)
          and {x["sample_id"] for x in sq_sec["latest"]} == {SAMPLE})
    # the section mirrors the M16 engine listing exactly (newest-first)
    engine_ids = [x["evaluation_id"] for x in sorted(
        sq, key=lambda r: (r["created_at"], r["evaluation_id"]),
        reverse=True)]
    check("B2 latest order == live M16 listing (created_at, id) desc",
          [x["evaluation_id"] for x in sq_sec["latest"]] == engine_ids,
          f"{len(sq_sec['latest'])} records")
    # by_sample: one row for the sampled sample with the correct history
    check("B3 by_sample matches sample f8e66f9c7b50's history exactly",
          len(sq_sec["by_sample"]) == 1
          and sq_sec["by_sample"][0]["sample_id"] == SAMPLE
          and sq_sec["by_sample"][0]["evaluation_count"] == 2
          and sq_sec["by_sample"][0]["latest_evaluation_id"] in EVAL_IDS
          and len(sq_sec["by_sample"][0]["sample_result_hash"]) == 64)
    # latest references resolve through GET sample-quality/<id>
    for x in sq_sec["latest"]:
        c, rec1 = call_json("GET", f"/models/{MODEL}/sample-quality/"
                                   f"{x['evaluation_id']}")
        check(f"B4 reference {x['evaluation_id']} resolves via M16 getter",
              c == 200 and rec1["sample_id"] == x["sample_id"]
              and rec1["created_at"] == x["created_at"])
    # determinism + read-only: repeated dashboard byte-identical, hash same
    code, d1 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("B5 repeated dashboard byte-identical, hash identical",
          code == 200 and d1 == d0
          and d1["result_hash"] == d0["result_hash"])
    # section carries references/counts only — no metric payload
    blob = json.dumps(sq_sec).lower()
    check("B6 section exposes history structure only (no metrics)",
          "loss" not in blob and "perplexity" not in blob
          and "score" not in blob and "rank" not in blob)

    print("== Phase C: read-only failure test (zero storage change) ==")
    c1, _ = call_json("GET", "/models/no-such-model/dashboard")
    check("C1 unknown model dashboard -> 404", c1 == 404, f"{c1}")

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
    code, smp2 = call_json("GET", f"/models/{MODEL}/samples")
    code2, ev2 = call_json("GET", f"/models/{MODEL}/evaluations")
    code3, sq2 = call_json("GET", f"/models/{MODEL}/sample-quality")
    check("D4 samples 4 / M4 evals 16 / sample-evaluations 2 unchanged",
          code == 200 and len(smp2) == 4 and code2 == 200
          and len(ev2) == 16 and code3 == 200 and len(sq2) == 2)
    code, d2 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("D5 dashboard still deterministic with the same hash",
          code == 200 and d2 == d0 and d2["result_hash"]
          == d0["result_hash"] and d2["diagnostics"] == [])

    print()
    if FAILURES:
        print(f"M17 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M17 live smoke OK: read-only sample-quality observability on the "
          "production dashboard (exact M16-history mirror, byte-deterministic "
          "repeats, ZERO production storage growth) verified live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
