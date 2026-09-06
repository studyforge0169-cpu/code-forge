"""M27 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M27 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/samples/by-checkpoint/{checkpoint_id},
answering "which immutable M15 samples were generated from this
checkpoint state?" — the model's authoritative M15 listing filtered by
the persisted checkpoint identity recorded in every SampleRecord
(M15 generation always binds ONE explicit verified checkpoint; the
field is required and non-nullable; membership never comes from
filenames, timestamps or hashes), after checkpoint ownership is
validated through the M3 checkpoint registry (the same model-scoped
path M20/M24/M25/M26 use). The smoke must DISCOVER the authoritative
sample distribution from the live M15 listing (not assume it from an
old report), prove exact listing parity for the checkpoint holding
samples, 200 + [] for the two valid-but-empty checkpoints, partition
integrity, clean 404s (unknown model / unknown checkpoint /
cross-model), byte-identical repeats (x3), unchanged M3/M15/M16/
M17/M18-M26 surfaces + dashboard hash + registries + OpenAPI 59, and
ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
source is the M15 listing): model 4a0a871886ef owns 3 checkpoints
(025e6d8d8f15, 0511de4c7372, 30a8bc5b82ab) and 4 samples, ALL
generated from checkpoint 0511de4c7372 in ASCENDING (created_at,
sample_id) order f8e66f9c7b50 (greedy), e2b5166fa549 (greedy),
05820e8bc68a (temperature), 4e8463e0df06 (temperature); checkpoints
025e6d8d8f15 and 30a8bc5b82ab are valid with ZERO samples -> natural
live 200 + [] cases; model b5bc905326b6 exists with NO checkpoints and
NO samples (cross-model probe); M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m27-smoke-baseline-inventory.json +
model/checkpoint registries + M15 listing + M16/M18-M26 pre-state +
dashboard + OpenAPI 59 pre-state.

LIVE B (discover authoritative sample distribution): derive totals,
ids, per-checkpoint counts and strategies from the live M15 listing;
print them; cross-check against the audited values.

LIVE C (known checkpoint 0511de4c7372): by-checkpoint -> 200 with the
EXACT discovered ids in the ASCENDING M15 order; every record's
persisted checkpoint identity verified; parity with the filtered live
listing; verbatim detail-getter parity for every record.

LIVE D (valid empty checkpoints): 025e6d8d8f15 and 30a8bc5b82ab ->
200 + [] each.

LIVE E (deterministic repeats): three GETs per checkpoint,
raw-byte-identical.

LIVE F (unknown model): -> 404 (even with a real checkpoint id).

LIVE G (unknown checkpoint): well-formed and malformed unknown ids ->
404.

LIVE H (cross-model isolation): b5bc905326b6 checkpoint registry is
empty (authoritative; no guessed statuses) and b5bc905326b6 +
0511de4c7372 (owned by 4a0a871886ef) -> 404 via the model-scoped M3
registry; no sample id leaks under the other model; the sample detail
getter is not shadowed (real id resolves verbatim, ghost id 404s).

LIVE I (regression + final audit): M3 checkpoint registry, M15
list/get, M16 sample quality, M17 dashboard hash, M18 records, M19
by-sample, M20 by-checkpoint, M21 by-suite, M22 summary, M23
by-policy, M24 evaluations by-checkpoint, M25 suite-runs
by-checkpoint, M26 comparisons by-checkpoint (6/5/1), OpenAPI 59 with
the new path exactly once — all unchanged; inventory diff — every
pre-existing file byte-identical, ZERO new files, zero .tmp, zero
storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8748 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8748/api/v1"
SITE = "http://127.0.0.1:8748"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO checkpoints/samples
CK_KNOWN = "0511de4c7372"             # the checkpoint ALL 4 samples used
EMPTY_CKPTS = ("025e6d8d8f15", "30a8bc5b82ab")  # valid, zero samples
CKPTS = (CK_KNOWN, *EMPTY_CKPTS)      # registry order for the model
KNOWN_IDS = ("f8e66f9c7b50", "e2b5166fa549", "05820e8bc68a",
             "4e8463e0df06")   # ASCENDING (created_at, sample_id)
UNKNOWN_CK = "ffffffffffff"          # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"              # the live production suite
POLICY = "m9-live-policy"            # the live production policy (1 gate)
SAMPLE = "f8e66f9c7b50"              # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")    # ASCENDING (created, id)
M26_COUNTS = {"025e6d8d8f15": 6, "30a8bc5b82ab": 5, "0511de4c7372": 1}
NEW_PATH = "/api/v1/models/{model_id}/samples/by-checkpoint/{checkpoint_id}"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m27-smoke-baseline-inventory.json")

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
    return f"{BASE}/models/{model}/samples/by-checkpoint/{ck}"


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
    code_co, cks_o = call_json("GET",
                               f"{BASE}/models/{OTHER_MODEL}/checkpoints")
    check("A3 both models resolve; the 3 known checkpoints register; the "
          "other model's checkpoint registry is EMPTY (authoritative)",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_c == 200
          and {c["checkpoint_id"] for c in cks} == set(CKPTS)
          and code_co == 200 and cks_o == [])
    code_w, wfs = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code2, sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  "records")
    code4, bs = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-sample/{SAMPLE}")
    code5, bc = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                 f"by-checkpoint/{CK_KNOWN}")
    code6, g = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}")
    code7, s = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                                f"{SUITE}/summary")
    code8, ev = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                 f"by-checkpoint/{CK_KNOWN}")
    code9, gd = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"by-policy/{POLICY}")
    code10, srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                    f"by-checkpoint/{CK_KNOWN}")
    code11, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M11/M16/M18-M26 pre-state intact (workflows 200, 2 sample "
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
    check("A5 M17 dashboard hash equals the full known value",
          code11 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 59 paths, the new by-checkpoint path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 59
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["sampling"])

    print("== LIVE B: discover the authoritative sample distribution ==")
    code, samples = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code_o, samples_o = call_json("GET",
                                  f"{BASE}/models/{OTHER_MODEL}/samples")
    dist: dict[str, list[str]] = {}
    for x in samples:
        dist.setdefault(x["checkpoint_id"], []).append(x["sample_id"])
    print(f"    discovered: {len(samples)} samples total, "
          f"{len(dist)} checkpoint(s): "
          + ", ".join(f"{ck} -> {len(ids)} {ids}"
                      for ck, ids in sorted(dist.items())))
    check("B1 M15 listing pre-state: 4 samples, ALL from checkpoint "
          "0511de4c7372, in ASCENDING (created_at, sample_id) order; "
          "other model has none",
          code == 200 and len(samples) == 4 and code_o == 200
          and samples_o == []
          and [x["sample_id"] for x in samples] == list(KNOWN_IDS)
          and [(x["created_at"], x["sample_id"]) for x in samples]
          == sorted((x["created_at"], x["sample_id"]) for x in samples)
          and set(dist) == {CK_KNOWN}
          and {x["checkpoint_id"] for x in samples} == {CK_KNOWN})
    check("B2 discovered strategies match the audited values (2 greedy "
          "+ 2 temperature, per persisted record)",
          [x["strategy"] for x in samples].count("greedy") == 2
          and [x["strategy"] for x in samples].count("temperature") == 2)

    print("== LIVE C: known checkpoint 0511de4c7372 (4 samples) ==")
    url_c = byck(CK_KNOWN)
    code, raw1 = call("GET", url_c)
    grouped = json.loads(raw1)
    check("C1 HTTP 200 with exactly 4 records", code == 200
          and len(grouped) == 4, f"{code}/{len(grouped)}")
    check("C2 the exact discovered ids in ASCENDING (created_at, "
          "sample_id) M15 order",
          [x["sample_id"] for x in grouped] == list(KNOWN_IDS)
          and [(x["created_at"], x["sample_id"]) for x in grouped]
          == sorted((x["created_at"], x["sample_id"]) for x in grouped))
    check("C3 every record's persisted identity: model + checkpoint id "
          "exactly as requested",
          all(x["model_id"] == MODEL and x["checkpoint_id"] == CK_KNOWN
              for x in grouped))
    check("C4 parity with the M15 listing filtered by the persisted "
          "checkpoint identity",
          grouped == [x for x in samples
                      if x["checkpoint_id"] == CK_KNOWN]
          and grouped == samples)
    for x in grouped:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                  f"{x['sample_id']}")
        if c != 200 or one != x:
            check("C5 verbatim payload parity with the M15 detail getter",
                  False, x["sample_id"])
            break
    else:
        check("C5 verbatim payload parity with the M15 detail getter for "
              "all 4 records (prompt/token ids/output/result_hash)", True)

    print("== LIVE D: the two valid empty checkpoints ==")
    for ck in EMPTY_CKPTS:
        c, body = call_json("GET", byck(ck))
        check(f"{'D1' if ck == EMPTY_CKPTS[0] else 'D2'} valid checkpoint "
              f"{ck} with no samples -> 200 + []",
              c == 200 and body == [], f"{c}")

    print("== LIVE E: deterministic repeats (x3 per checkpoint) ==")
    for ck in CKPTS:
        _, r1 = call("GET", byck(ck))
        _, r2 = call("GET", byck(ck))
        _, r3 = call("GET", byck(ck))
        check(f"E {ck}: three GETs raw-byte-identical",
              r1 == r2 == r3 and r1 == raw1 if ck == CK_KNOWN
              else r1 == r2 == r3)

    print("== LIVE F: unknown model ==")
    c1, _ = call_json("GET", byck(CK_KNOWN, "no-such-model-27"))
    check("F1 unknown model -> 404 (even with a real checkpoint id)",
          c1 == 404, f"{c1}")

    print("== LIVE G: unknown checkpoint ==")
    c2, _ = call_json("GET", byck(UNKNOWN_CK))
    check("G1 unknown well-formed checkpoint id -> 404 (valid model, "
          "empty history is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{byck('ck%20id%20with%20spaces!!')}")
    check("G2 malformed unknown checkpoint id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE H: cross-model isolation + no shadowing ==")
    c4, _ = call_json("GET", byck(CK_KNOWN, OTHER_MODEL))
    check("H1 other real model + this model's real registry-valid "
          "checkpoint id -> 404 (model-scoped M3 registry; the other "
          "model's registry is empty)", c4 == 404, f"{c4}")
    c5, leak = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/samples")
    check("H2 none of the 4 sample ids is reachable under the other "
          "model (its history is [])",
          c5 == 200 and leak == []
          and all(sid not in (leak or []) for sid in KNOWN_IDS))
    c6, one = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                               f"{KNOWN_IDS[0]}")
    check("H3 by-checkpoint does not shadow the M15 detail getter "
          "(f8e66f9c7b50 resolves verbatim)",
          c6 == 200 and one == samples[0])
    c7, _ = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                             "ghost-sample-27")
    check("H4 unknown sample id via detail getter still 404", c7 == 404,
          f"{c7}")

    print("== LIVE I: regression + final audit (zero storage growth) ==")
    code, samples2 = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    check("I1 M15 listing unchanged", code == 200 and samples2 == samples)
    ok_detail = True
    for x in samples:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                  f"{x['sample_id']}")
        ok_detail = ok_detail and c == 200 and one == x
    check("I2 M15 detail getter unchanged for all 4 samples", ok_detail)
    code2, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code3, rec2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                   "records")
    code4, bs2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-sample/{SAMPLE}")
    code5, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/{CK_KNOWN}")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}/summary")
    code8, ev2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"by-checkpoint/{CK_KNOWN}")
    code9, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"by-policy/{POLICY}")
    code10, srck2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                     f"by-checkpoint/{CK_KNOWN}")
    check("I3 M16/M18/M19/M20/M21/M22/M23/M24/M25 outputs unchanged",
          code2 == 200 and sq2 == sq and code3 == 200 and rec2 == rec
          and code4 == 200 and bs2 == bs and code5 == 200 and bc2 == bc
          and code6 == 200 and g2 == g and code7 == 200 and s2 == s
          and code8 == 200 and ev2 == ev and code9 == 200 and gd2 == gd
          and code10 == 200 and srck2 == srck)
    ok_m26 = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_m26 = ok_m26 and c == 200 and len(body) == n
    check("I4 M26 comparisons by-checkpoint unchanged (6/5/1)",
          ok_m26)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("I5 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    check("I6 M11 workflow history, M3 checkpoint registry and M9 "
          "policy registry byte-identical to pre-state",
          code_w2 == 200 and wfs2 == wfs and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol)
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    check("I7 OpenAPI still 59 paths with the new path exactly once "
          "(no drift during the smoke)",
          code_sp2 == 200 and len(spec2["paths"]) == 59
          and set(spec2["paths"][NEW_PATH]) == {"get"})
    ok_again = True
    for ck in CKPTS:
        _, body = call_json("GET", byck(ck))
        expect = grouped if ck == CK_KNOWN else []
        ok_again = ok_again and body == expect
    check("I8 by-checkpoint still deterministic at the end (all three "
          "checkpoints)", ok_again)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("I9 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("I10 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("I11 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M27 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M27 live smoke OK: narrow read-only by-checkpoint grouping "
          "of the M15 sample history (distribution DISCOVERED live: 4 "
          "samples, all from 0511de4c7372 — exact listing parity with "
          "the exact audited ids —, 200 + [] for the two valid empty "
          "checkpoints, partition 4+0+0=4, clean 404s incl. cross-model "
          "isolation through the model-scoped M3 registry, "
          "deterministic byte-identical x3 repeats, M3/M15/M16/"
          "M17/M18-M26 surfaces + dashboard hash + registries + "
          "OpenAPI 59 unchanged, ZERO production storage growth) "
          "verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
