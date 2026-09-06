"""M28 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M28 is READ-ONLY: it adds one narrow
access path, GET /models/{id}/evaluations/by-dataset/{dataset_id},
answering "which immutable M4 evaluations of this model measured this
dataset?" — the model's authoritative M4 listing filtered by the
persisted dataset identity recorded in each EvaluationRecord
(top-level dataset_id; the persisted dataset_version travels VERBATIM
inside every record — versions never collapsed, resolved or rewritten),
after the dataset is validated through the M2 registry (datasets are
GLOBAL; model scoping comes from the model's own M4 listing, exactly
like M24/M27). The smoke must DISCOVER the authoritative dataset /
evaluation distribution from the live M2 registry and M4 listing (not
assume it from an old report), prove exact listing parity for the
known dataset, the natural model-scoped empty case, version
integrity, clean 404s, byte-identical repeats (x3), unchanged
M2/M4/M20-M27 surfaces + dashboard hash + registries + OpenAPI 60,
and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M2 registry and the M4 listing): exactly ONE dataset
exists, ee1a716c4573 (v1); model 4a0a871886ef owns 16 evaluations,
ALL persisted with dataset_id=ee1a716c4573 and dataset_version=1
(14 validation + 2 train splits), in ASCENDING (created_at, eval_id)
order a439eb92f9cd, b0502d871114, 7a16eaa12120, 0cc96a125976,
425003213a0b, a884bf729ff7, b89a94306ce8, 75835b64d6af, 90aa392b9a2b,
340f5adbf881, c739c66638e9, c35a1c9902fe, fe7b42cdb411, ebb9b7eccbe3,
0704399fea7b, 75a23351e91c; model b5bc905326b6 exists with NO
evaluations (natural model-scoped empty case: valid global dataset +
model with empty history -> 200 + []); M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m28-smoke-baseline-inventory.json +
model/dataset/checkpoint registries + M4 listing + M16/M18-M27
pre-state + dashboard + OpenAPI 60 pre-state.

LIVE B (authoritative dataset discovery): from the M2 dataset registry
and the live M4 listing, discover dataset ids/versions, evaluation
counts per dataset and the owning model; print the values; cross-check
against the audited facts.

LIVE C (known dataset): 4a0a871886ef + ee1a716c4573 -> 200 with the
EXACT discovered count (16) and ids in the ASCENDING M4 order;
persisted dataset identity on every record; parity with the filtered
live M4 listing; verbatim detail-getter parity for every record.

LIVE D (exact parity): the response equals GET /models/4a0a871886ef/
evaluations filtered locally by persisted dataset identity — no
missing, no extra, no duplicate evaluations.

LIVE E (version integrity): every returned record's persisted
dataset_version equals the version in the M4 listing record verbatim
(no normalization, no rewriting).

LIVE F (natural empty model case): b5bc905326b6 (valid model, NO
evaluations) + the registry-valid ee1a716c4573 -> 200 + [].

LIVE G (deterministic repeats): three GETs of the known-dataset
request raw-byte-identical.

LIVE H (unknown model): -> 404 (even with the real dataset id).

LIVE I (unknown dataset): well-formed and malformed unknown ids -> 404.

LIVE J (cross-model isolation): the dataset is registry-valid and
b5bc905326b6's evaluation listing is EMPTY (both verified from the
authoritative registries first — nothing hard-coded), so the by-dataset
answer for b5bc905326b6 is 200 + [] and none of the 16 evaluation ids
leaks into it.

LIVE K (regression + final audit): M2 dataset registry +
GET /datasets/{id}, M3 checkpoint registry, M4 list/get, M20
sample-quality by-checkpoint, M21 by-suite, M22 summary, M23
by-policy, M24 evaluations by-checkpoint (3/3/3), M25 suite-runs
by-checkpoint (10), M26 comparisons by-checkpoint (6/5/1), M27
samples by-checkpoint (4/0/0), M17 dashboard hash, policies, probe
suites, checkpoints, OpenAPI 60 with the new path exactly once — all
unchanged; inventory diff — every pre-existing file byte-identical,
ZERO new files, zero .tmp, zero storage growth.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8749 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8749/api/v1"
SITE = "http://127.0.0.1:8749"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO evaluations
DATASET = "ee1a716c4573"              # the ONE production dataset (v1)
EVAL_IDS = ("a439eb92f9cd", "b0502d871114", "7a16eaa12120",
            "0cc96a125976", "425003213a0b", "a884bf729ff7",
            "b89a94306ce8", "75835b64d6af", "90aa392b9a2b",
            "340f5adbf881", "c739c66638e9", "c35a1c9902fe",
            "fe7b42cdb411", "ebb9b7eccbe3", "0704399fea7b",
            "75a23351e91c")   # ASCENDING (created_at, eval_id)
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
UNKNOWN_DS = "ffffffffffff"          # well-formed 12-hex, nonexistent
SUITE = "m9-live-suite"              # the live production suite
POLICY = "m9-live-policy"            # the live production policy (1 gate)
SAMPLE = "f8e66f9c7b50"              # the measured production sample
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")    # ASCENDING (created, id)
M24_COUNTS = {CK_0511: 3, CK_025E: 3, CK_30A8: 3}
M26_COUNTS = {CK_025E: 6, CK_30A8: 5, CK_0511: 1}
M27_COUNTS = {CK_0511: 4, CK_025E: 0, CK_30A8: 0}
NEW_PATH = "/api/v1/models/{model_id}/evaluations/by-dataset/{dataset_id}"
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m28-smoke-baseline-inventory.json")

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


def byds(ds: str, model: str = MODEL) -> str:
    return f"{BASE}/models/{model}/evaluations/by-dataset/{ds}"


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
          == {CK_0511, CK_025E, CK_30A8}
          and call_json("GET", f"{BASE}/models/{OTHER_MODEL}/checkpoints")
          [1] == [])
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
    code8, evck = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{CK_0511}")
    code9, gd = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                 f"by-policy/{POLICY}")
    code10, srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                    f"by-checkpoint/{CK_0511}")
    code11, smp = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{CK_0511}")
    code12, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    code_p, pol = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr, probes = call_json("GET", f"{BASE}/probe-suites")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M11/M16/M18-M27 pre-state intact (workflows 200, 2 sample "
          "records, 10 by-suite, summary 10, 3 checkpoint evals, 1 "
          "gate, 10 by-checkpoint suite runs, 4 by-checkpoint samples)",
          code_w == 200 and code2 == 200 and len(sq) == 2
          and code3 == 200 and len(rec) == 2 and code4 == 200
          and len(bs) == 2 and code5 == 200 and len(bc) == 2
          and code6 == 200 and len(g) == 10 and code7 == 200
          and s["total_count"] == 10 and code8 == 200 and len(evck) == 3
          and code9 == 200 and len(gd) == 1 and code10 == 200
          and len(srck) == 10 and code11 == 200 and len(smp) == 4
          and code_p == 200
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))
    check("A5 M17 dashboard hash equals the full known value",
          code12 == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    check("A6 OpenAPI pre-state: 60 paths, the new by-dataset path "
          "registered exactly once with only GET",
          code_sp == 200 and len(paths) == 60
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["evaluation"])

    print("== LIVE B: authoritative dataset discovery ==")
    code_ds, datasets = call_json("GET", f"{BASE}/datasets")
    code_dsv, dsv = call_json("GET", f"{BASE}/datasets/{DATASET}")
    code, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_oe, evals_o = call_json("GET",
                                 f"{BASE}/models/{OTHER_MODEL}/evaluations")
    dist: dict[tuple, list[str]] = {}
    for x in evals:
        dist.setdefault((x["dataset_id"], x["dataset_version"]),
                        []).append(x["eval_id"])
    print(f"    discovered: {len(datasets)} dataset(s): "
          + ", ".join(f"{d['id']} (versions {d['versions']})"
                      for d in datasets))
    print(f"    discovered: {len(evals)} evaluations of {MODEL} over "
          + ", ".join(f"({ds} v{v}) -> {len(ids)}"
                      for (ds, v), ids in sorted(dist.items())))
    check("B1 M2 registry holds exactly the ONE known dataset (v1)",
          code_ds == 200 and len(datasets) == 1
          and datasets[0]["id"] == DATASET
          and datasets[0]["versions"] == [1]
          and code_dsv == 200 and dsv["dataset"]["id"] == DATASET
          and [v["version"] for v in dsv["versions"]] == [1])
    check("B2 M4 listing pre-state: 16 evaluations, ALL persisted with "
          "dataset ee1a716c4573 v1, in ASCENDING (created_at, eval_id) "
          "order; other model has none",
          code == 200 and len(evals) == 16 and code_oe == 200
          and evals_o == []
          and [x["eval_id"] for x in evals] == list(EVAL_IDS)
          and [(x["created_at"], x["eval_id"]) for x in evals]
          == sorted((x["created_at"], x["eval_id"]) for x in evals)
          and set(dist) == {(DATASET, 1)})
    check("B3 split mix as audited (14 validation + 2 train)",
          [x["split"] for x in evals].count("validation") == 14
          and [x["split"] for x in evals].count("train") == 2)

    print("== LIVE C: known dataset (16 evaluations) ==")
    url_c = byds(DATASET)
    code, raw1 = call("GET", url_c)
    grouped = json.loads(raw1)
    check("C1 HTTP 200 with exactly the discovered 16 records",
          code == 200 and len(grouped) == 16, f"{code}/{len(grouped)}")
    check("C2 the exact discovered ids in ASCENDING (created_at, "
          "eval_id) M4 order",
          [x["eval_id"] for x in grouped] == list(EVAL_IDS)
          and [(x["created_at"], x["eval_id"]) for x in grouped]
          == sorted((x["created_at"], x["eval_id"]) for x in grouped))
    check("C3 every record's persisted identity: model + dataset id "
          "exactly as requested",
          all(x["model_id"] == MODEL and x["dataset_id"] == DATASET
              for x in grouped))
    for x in grouped:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"{x['eval_id']}")
        if c != 200 or one != x:
            check("C4 verbatim payload parity with the M4 detail getter",
                  False, x["eval_id"])
            break
    else:
        check("C4 verbatim payload parity with the M4 detail getter for "
              "all 16 records (loss/perplexity/state identity)", True)

    print("== LIVE D: exact parity with the M4 listing ==")
    filtered = [x for x in evals if x["dataset_id"] == DATASET]
    ids = [x["eval_id"] for x in grouped]
    check("D1 response == M4 listing filtered locally by persisted "
          "dataset identity (no missing / extra / duplicate)",
          grouped == filtered and len(ids) == len(set(ids))
          == len(filtered))

    print("== LIVE E: version integrity ==")
    check("E1 every returned dataset_version equals the persisted M4 "
          "listing value VERBATIM (all v1 here; no normalization)",
          all(x["dataset_version"] == y["dataset_version"]
              for x, y in zip(grouped, filtered))
          and {x["dataset_version"] for x in grouped} == {1})

    print("== LIVE F: natural empty model case ==")
    c, body = call_json("GET", byds(DATASET, OTHER_MODEL))
    check("F1 b5bc905326b6 (valid model, empty history) + the "
          "registry-valid dataset -> 200 + []",
          c == 200 and body == [], f"{c}")

    print("== LIVE G: deterministic repeats (x3) ==")
    _, r1 = call("GET", url_c)
    _, r2 = call("GET", url_c)
    _, r3 = call("GET", url_c)
    check("G1 three GETs raw-byte-identical", r1 == r2 == r3 == raw1)

    print("== LIVE H: unknown model ==")
    c1, _ = call_json("GET", byds(DATASET, "no-such-model-28"))
    check("H1 unknown model -> 404 (even with the real dataset id)",
          c1 == 404, f"{c1}")

    print("== LIVE I: unknown dataset ==")
    c2, _ = call_json("GET", byds(UNKNOWN_DS))
    check("I1 unknown well-formed dataset id -> 404 (valid model, "
          "empty history is NEVER 404)", c2 == 404, f"{c2}")
    c3, _ = call_json("GET", f"{byds('ds%20id%20with%20spaces!!')}")
    check("I2 malformed unknown dataset id -> 404 (no crash)",
          c3 == 404, f"{c3}")

    print("== LIVE J: cross-model isolation ==")
    # verify from the authoritative registries FIRST (nothing assumed):
    # the dataset is registry-valid and the other model's M4 listing
    # is empty -> the only correct by-dataset answer is 200 + []
    ok_pre = code_dsv == 200 and code_oe == 200 and evals_o == []
    c4, leak = call_json("GET", byds(DATASET, OTHER_MODEL))
    check("J1 registry-verified: dataset valid + other model's M4 "
          "listing empty -> 200 + [] (model-scoped history)",
          ok_pre and c4 == 200 and leak == [], f"{c4}")
    check("J2 none of the 16 evaluation ids leaks into the other "
          "model's response",
          all(eid not in (leak or []) for eid in EVAL_IDS))

    print("== LIVE K: regression + final audit (zero storage growth) ==")
    code, evals2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    check("K1 M4 listing unchanged", code == 200 and evals2 == evals)
    ok_detail = True
    for eid in EVAL_IDS:
        c, one = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                  f"{eid}")
        ok_detail = ok_detail and c == 200 \
            and one == evals[EVAL_IDS.index(eid)]
    check("K2 M4 detail getter unchanged for all 16 records", ok_detail)
    code_ds2, datasets2 = call_json("GET", f"{BASE}/datasets")
    code_dsv2, dsv2 = call_json("GET", f"{BASE}/datasets/{DATASET}")
    check("K3 M2 dataset registry + GET /datasets/{id} unchanged",
          code_ds2 == 200 and datasets2 == datasets
          and code_dsv2 == 200 and dsv2 == dsv)
    ok_m = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_m = ok_m and c == 200 and len(body) == n
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_m = ok_m and c == 200 and len(body) == n
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_m = ok_m and c == 200 and len(body) == n
    check("K4 M24 (3/3/3) / M26 (6/5/1) / M27 (4/0/0) by-checkpoint "
          "surfaces unchanged", ok_m)
    code2, sq2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")
    code5, bc2 = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                                  f"by-checkpoint/{CK_0511}")
    code6, g2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}")
    code7, s2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                 f"by-suite/{SUITE}/summary")
    code9, gd2 = call_json("GET", f"{BASE}/models/{MODEL}/gates/decisions/"
                                  f"by-policy/{POLICY}")
    code10, srck2 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                     f"by-checkpoint/{CK_0511}")
    code11, smp2 = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                    f"by-checkpoint/{CK_0511}")
    check("K5 M16/M18-M20/M21/M22/M23/M25/M27 outputs unchanged",
          code2 == 200 and sq2 == sq and code5 == 200 and bc2 == bc
          and code6 == 200 and g2 == g and code7 == 200 and s2 == s
          and code9 == 200 and gd2 == gd and code10 == 200
          and srck2 == srck and code11 == 200 and smp2 == smp)
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("K6 M17 dashboard result_hash preserved",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    code_w2, wfs2 = call_json("GET", f"{BASE}/models/{MODEL}/workflows")
    code_c2, cks2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    code_p2, pol2 = call_json("GET", f"{BASE}/policies/{POLICY}")
    code_pr2, probes2 = call_json("GET", f"{BASE}/probe-suites")
    check("K7 M11 workflow history, M3 checkpoint registry, M9 policy "
          "registry and M9 probe-suite registry unchanged",
          code_w2 == 200 and wfs2 == wfs and code_c2 == 200
          and cks2 == cks and code_p2 == 200 and pol2 == pol
          and code_pr2 == 200 and probes2 == probes)
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    generic = "/api/v1/models/{model_id}/evaluations/{eval_id}"
    check("K8 OpenAPI still 60 paths with the new path exactly once "
          "(no drift during the smoke)",
          code_sp2 == 200 and len(spec2["paths"]) == 60
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(generic))
    _, grouped2 = call_json("GET", url_c)
    check("K9 by-dataset still deterministic at the end",
          grouped2 == grouped)
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("K10 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("K11 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("K12 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M28 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M28 live smoke OK: narrow read-only by-dataset grouping of "
          "the M4 evaluation history (distribution DISCOVERED live: one "
          "dataset ee1a716c4573 v1, 16 evaluations all over it with the "
          "exact audited ids — exact listing parity, versions VERBATIM "
          "(all v1) —, natural model-scoped empty case 200 + [] for "
          "b5bc905326b6, clean 404s, deterministic byte-identical x3 "
          "repeats, M2/M4/M20-M27 surfaces + dashboard hash + "
          "registries + OpenAPI 60 unchanged, ZERO production storage "
          "growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
