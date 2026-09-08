"""M46 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M46 is READ-ONLY: it adds one narrow
MODEL-SCOPED access path,
GET /models/{id}/checkpoints/by-run/{run_id},
answering "which immutable checkpoints did ONE training run of this
model produce?" — the model's authoritative M3 checkpoint listing
(deterministic (step, created_at) ASCENDING order) filtered VERBATIM
by each checkpoint's own persisted REQUIRED run_id (lineage
within/across runs), with run OWNERSHIP validated against the model's
OWN manifest training_provenance (RunProvenance.run_id): an unknown
model, an unknown run or a run id belonging to another model is 404
(model-scoped ownership — exactly the M19 sample-ownership
precedent); membership is NEVER derived from checkpoint directories,
steps, epochs, timestamps, losses, parent relationships or any
provenance field other than the checkpoint's own run_id (the
provenance list is used ONLY for existence validation — the 404 —
never for building the response). Because run_id is REQUIRED, the
groups form a TRUE disjoint partition of the listing with no None
case. NO separate training-run registry is introduced — the model
manifest IS the registry. Engine reality (verified live + in the M46
tests): steps >= 1 and the FINAL step is always an eval point
(training.py builds eval_points as range(eval_every, total+1,
eval_every) | {total_steps}), so a HEALTHY registered run always owns
>= 1 checkpoint; the zero-checkpoint 200 [] path is engine-real
through the M3 listing's corruption resilience (unreadable manifests
are skipped) and is covered by the M46 engine/API tests — in
production it is NOT reachable without writes, which this smoke never
performs. The smoke must DISCOVER the authoritative run distribution
from the live M3 listing + the live model manifest training_provenance
(not assume it from an old report), prove exact listing parity for
ALL groups, clean 404 separation (unknown model / unknown run /
another model's run), byte-identical repeats (x3), unchanged M3-M45
surfaces + dashboard hash + registries + OpenAPI 78, and ZERO
production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
sources are the M3 checkpoint listing and the model manifest
training_provenance):
model 4a0a871886ef owns 3 checkpoints in (step, created_at) ASC order
with persisted run_id distribution 291a16d755fc -> 2 (30a8 step 10,
025e step 20) and 85438934f86a -> 1 (0511 step 16) — a TRUE disjoint
partition of all 3; its manifest training_provenance registers
EXACTLY those two run ids (total coverage). Model b5bc905326b6 has 0
provenance entries and 0 checkpoints (every run id 404s under it).
M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m46-smoke-baseline-inventory.json +
model/tokenizer/policy/probe-suite/recipe registries + M3-M45
pre-state (3 checkpoints with verbatim detail payloads, 11 gate
decisions, 8 comparisons, 16 evaluations, 13 workflows, 10
suite-runs, 4 samples) + dashboard + OpenAPI 78 pre-state + the
known pair returns 200.

LIVE B (authoritative distribution discovery): from the live M3
listing and the live model manifest training_provenance, discover the
run distribution; print it; cross-check against the audited facts
(2/1 partition of 3; every checkpoint's run_id registered; every
registered run non-empty in production).

LIVE C (A — known runs): BOTH production runs -> 200 with the EXACT
discovered records; parity with the M3 listing filtered locally by the
persisted run_id; verbatim detail-getter payloads for all 3
checkpoints; (step, created_at) ASC preserved per group.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — empty boundaries without fabrication): b5bc905326b6 (0
provenance, 0 checkpoints) -> its checkpoint listing is 200 [] and
every run id (including 4a0a871886ef's REAL run ids) 404s under it
(ownership validation); in production EVERY registered run of
4a0a871886ef owns >= 1 checkpoint (the healthy-run invariant).

LIVE F (D — unknown ids): unknown model + the REAL run id -> 404;
valid model + unknown run -> 404; another model's run -> 404; the
run_id is a persisted identifier, not an enum (no 422 path — every
 syntactically valid id is either 200 or 404).

LIVE G (E — true partition): the run groups form a TRUE disjoint
partition of the FULL M3 listing (2 + 1 == 3; every record EXACTLY
ONCE; no record in two groups; no None case — run_id is required);
every checkpoint's run_id is registered in the manifest
training_provenance (total coverage).

LIVE H (F — ordering): every group's order equals the authoritative
M3 (step, created_at) ASCENDING order restricted to that run.

LIVE I (G — M3/M4 regressions): the checkpoint listing is identical
to LIVE A; every detail getter verbatim; the provenance registry
unchanged (exactly the two run ids).

LIVE J (H — M24-M45 regressions): evaluations 3/3/3 by-checkpoint +
16 by-dataset + 16 by-tokenizer + by-split validation->14 / train->2
/ test->0 + by-state-kind checkpoint->9 / current->7; comparisons
6/5/1 by-checkpoint + 8 by-dataset + 8 by-tokenizer + by-split
validation->8 / train->0 / test->0 + by-verdict improved->3 /
unchanged->3 / regressed->2 + by-state-kind checkpoint->8 / current->2
(either-side); samples 4/0/0 by-checkpoint + 4 by-tokenizer +
by-strategy greedy->2 / temperature->2; suite runs 10 by-suite + 10
by-checkpoint + summary total 10; sample-quality 2 by-tokenizer;
workflows 13 total with by-status completed->9 / failed->3 /
stopped->1 + 2 by-recipe (m12-live-suite); gates 11 with
by-decision passed->7 / failed->4 + by-verdict improved->4 /
regressed->3 / unchanged->2 + by-baseline-type checkpoint->7 /
current->2 / minimum_loss->2 / evaluation_result_hash->0 + 4
by-comparison + 1 by-policy.

LIVE K (I — M12/M16/M35 regressions): recipe registry 7 verbatim;
GLOBAL /workflows/recipes/m12-live-suite/runs -> 2; the 4 zero-run
recipes -> 200 + []; unknown recipe 404; M16 evaluation ids still
paired in sample-quality.

LIVE L (J — dashboard; K — registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/tokenizer/model
registries unchanged (2 models with the known ids).

LIVE M (L — OpenAPI + storage zero drift): OpenAPI exactly 78 paths,
new path once (after the checkpoint listing, before the generic
checkpoint-detail route; the M45/M44/M43/M42/M41 routes still present
exactly once); every pre-existing file byte-identical, ZERO new
files, zero .tmp, totals unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \\\
        app.api:app --host 127.0.0.1 --port 8767 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8767/api/v1"
SITE = "http://127.0.0.1:8767"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, 0 provenance + 0 checkpoints
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
DATASET = "ee1a716c4573"
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
RUN_A = "291a16d755fc"                # -> 2 checkpoints (30a8, 025e)
RUN_B = "85438934f86a"                # -> 1 checkpoint (0511)
POLICY = "m9-live-policy"
SUITE = "m9-live-suite"
KNOWN_RECIPE = "m12-live-suite"       # M35: 2 runs
ZERO_RUN_RECIPES = ("m14-base", "m14-chain-a", "m14-chain-b",
                    "m14-chain-c")
KNOWN_COMP = "fc379bfcb50f"           # M34: 4 decisions
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")
M24_COUNTS = {CK_0511: 3, CK_025E: 3, CK_30A8: 3}
M26_COUNTS = {CK_025E: 6, CK_30A8: 5, CK_0511: 1}
M27_COUNTS = {CK_0511: 4, CK_025E: 0, CK_30A8: 0}
M25_COUNT = 10                        # suite runs under m9-live-suite
M28_COUNT = 16                        # evaluations under ee1a716c4573
M29_COUNT = 8                         # comparisons under ee1a716c4573
M30_COUNT = 16                        # evaluations under 99106e3255c5
M31_COUNT = 8                         # comparisons under 99106e3255c5
M32_COUNT = 4                         # samples under 99106e3255c5
M33_COUNT = 2                         # sample-quality under 99106e3255c5
M34_COUNT = 4                         # gate decisions under fc379bfcb50f
M23_COUNT = 1                         # gate decisions under m9-live-policy
M36_DIST = {"validation": 14, "train": 2, "test": 0}
M37_COUNT = 8                         # comparisons with split=validation
M38_DIST = {"checkpoint": 9, "current": 7}
M39_DIST = {"improved": 3, "unchanged": 3, "regressed": 2}
M40_DIST = {"greedy": 2, "temperature": 2}
M41_DIST = {"passed": 7, "failed": 4}
M42_DIST = {"completed": 9, "failed": 3, "stopped": 1}
M43_DIST = {"improved": 4, "regressed": 3, "unchanged": 2}
M44_DIST = {"checkpoint": 8, "current": 2}
M45_DIST = {"checkpoint": 7, "current": 2, "minimum_loss": 2,
            "evaluation_result_hash": 0}
M46_DIST = {RUN_A: 2, RUN_B: 1}
EXPECTED_RUNS = (RUN_A, RUN_B)
CKPT_LIST_PATH = "/api/v1/models/{model_id}/checkpoints"
NEW_PATH = "/api/v1/models/{model_id}/checkpoints/by-run/{run_id}"
GENERIC_PATH = "/api/v1/models/{model_id}/checkpoints/{checkpoint_id}"
M45_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-baseline-type/"
            "{baseline_type}")
M44_PATH = ("/api/v1/models/{model_id}/comparisons/by-state-kind/"
            "{state_kind}")
M43_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-verdict/"
            "{verdict}")
M42_PATH = "/api/v1/models/{model_id}/workflows/by-status/{status}"
M41_PATH = ("/api/v1/models/{model_id}/gates/decisions/by-decision/"
            "{decision}")
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m46-smoke-baseline-inventory.json")

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


def byrun(run_id: str, model: str = MODEL) -> str:
    return (f"{BASE}/models/{model}/checkpoints/by-run/{run_id}")


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
    code_ck, ckpts = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    check("A3 both models resolve; the 3 known checkpoints register "
          "(other model has none)",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_ck == 200
          and {c["checkpoint_id"] for c in ckpts}
          == {CK_0511, CK_025E, CK_30A8}
          and call_json("GET", f"{BASE}/models/{OTHER_MODEL}/checkpoints")
          [1] == [])
    code_ev, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_cp, comps = call_json("GET", f"{BASE}/models/{MODEL}/"
                                      "comparisons")
    code_gd, gdecisions = call_json("GET", f"{BASE}/models/{MODEL}/"
                                           "gates/decisions")
    code_wf, workflows = call_json("GET", f"{BASE}/models/{MODEL}/"
                                          "workflows")
    code_sm, samples = call_json("GET", f"{BASE}/models/{MODEL}/samples")
    code_sr, runs10 = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                                       f"by-suite/{SUITE}")
    code_sp, spec = call_json("GET", f"{SITE}/openapi.json")
    paths = spec.get("paths", {}) if code_sp == 200 else {}
    new_ops = paths.get(NEW_PATH, {})
    check("A4 M4-M45 pre-state intact (3 checkpoints, 16 evaluations, 8 "
          "comparisons, 11 gate decisions, 13 workflows, 4 samples, 10 "
          "suite runs, 7 recipes, OpenAPI 78 with the by-run path "
          "registered exactly once, GET-only, tag training)",
          code_ev == 200 and len(evals) == 16 and code_cp == 200
          and len(comps) == 8 and code_gd == 200
          and len(gdecisions) == 11 and code_wf == 200
          and len(workflows) == 13 and code_sm == 200 and len(samples) == 4
          and code_sr == 200 and len(runs10) == 10
          and call_json("GET", f"{BASE}/workflows/recipes")[1].__len__()
          == 7
          and code_sp == 200 and len(paths) == 78
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["training"])
    code_d, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A5 M17 dashboard hash equals the full known value",
          code_d == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    code_a, raw_a = call("GET", byrun(RUN_A))
    check("A6 the known pair (4a0a871886ef + 291a16d755fc) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    prov_ids = [p["run_id"] for p in model["training_provenance"]]
    dist = {r: sum(1 for c in ckpts if c["run_id"] == r)
            for r in prov_ids}
    dist_str = ", ".join(f"{k} -> {v}" for k, v in sorted(dist.items()))
    print(f"    discovered: {len(ckpts)} checkpoints of {MODEL} per "
          f"persisted run_id: {dist_str}")
    print(f"    discovered: manifest training_provenance run ids: "
          f"{prov_ids}")
    check("B1 M3 listing pre-state: 3 checkpoints in (step, created_at) "
          "ASCENDING order; persisted run_id distribution "
          "291a16d755fc -> 2 / 85438934f86a -> 1 (TRUE disjoint "
          "partition of 3); manifest training_provenance registers "
          "EXACTLY those two run ids; other model: 0 provenance + 0 "
          "checkpoints",
          code_ck == 200 and len(ckpts) == 3
          and [(c["step"], c["created_at"]) for c in ckpts]
          == sorted((c["step"], c["created_at"]) for c in ckpts)
          and dist == dict(M46_DIST)
          and set(prov_ids) == set(EXPECTED_RUNS)
          and all(c["run_id"] in prov_ids for c in ckpts)
          and (other["training_provenance"] or []) == [])

    print("== LIVE C: (A) known runs — exact responses ==")
    groups: dict[str, list] = {}
    ok_groups = True
    for run_id in EXPECTED_RUNS:
        code_g, raw_g = call("GET", byrun(run_id))
        groups[run_id] = json.loads(raw_g)
        ok_groups = ok_groups and code_g == 200 and groups[run_id] == \
            [c for c in ckpts if c["run_id"] == run_id]
    check("C1/A1 BOTH run responses == M3 listing filtered locally by "
          "the checkpoints' own persisted run_id (no missing / extra / "
          "duplicate; every record EXACTLY ONCE; membership NEVER "
          "derived from directories, steps, timestamps or parent ids; "
          "the provenance registry is used ONLY for the 404, never "
          "for building the response)",
          ok_groups
          and all(len(groups[r]) == M46_DIST[r]
                  and all(c["model_id"] == MODEL and c["run_id"] == r
                          for c in groups[r])
                  for r in EXPECTED_RUNS),
          "/".join(str(len(groups[r])) for r in EXPECTED_RUNS))
    ok_detail = True
    for c in ckpts:
        code_one, one = call_json("GET", f"{BASE}/models/{MODEL}/"
                                        f"checkpoints/{c['checkpoint_id']}")
        if code_one != 200 or one != c:
            ok_detail = False
            check("C2/A1 verbatim payload parity with the M3 detail "
                  "getter", False, c["checkpoint_id"])
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M3 detail getter "
              "for all 3 checkpoints (losses, lineage, weights hash "
              "included)", True)

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", byrun(RUN_A))
    _, r2 = call("GET", byrun(RUN_A))
    _, r3 = call("GET", byrun(RUN_A))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw_a)

    print("== LIVE E: (C) empty boundaries without fabrication ==")
    c_o, listing_o = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/"
                                      "checkpoints")
    ok_404s = True
    for run_id in EXPECTED_RUNS:
        c_r, _ = call_json("GET", byrun(run_id, OTHER_MODEL))
        ok_404s = ok_404s and c_r == 404
    check("E1/C1 b5bc905326b6: checkpoint listing 200 [] and BOTH real "
          "run ids of 4a0a871886ef -> 404 (ownership: its manifest "
          "registers no runs — model-scoped provenance validation)",
          c_o == 200 and listing_o == [] and ok_404s)
    check("E2/C1 healthy-run invariant: EVERY registered production run "
          "owns >= 1 checkpoint (the final step is always an eval "
          "point), so no production run yields []; the zero-checkpoint "
          "200 [] path is engine-real through the M3 listing's "
          "corruption resilience and covered by the M46 engine/API "
          "tests — never fabricated here",
          all(dist[r] >= 1 for r in prov_ids)
          and sum(dist.values()) == 3)

    print("== LIVE F: (D) unknown ids ==")
    c1, _ = call_json("GET", byrun(RUN_A, "no-such-model-46"))
    c2, _ = call_json("GET", byrun("no-such-run-46"))
    check("F1/D1 unknown model + the REAL run id -> 404; valid model + "
          "unknown run -> 404; run_id is a persisted identifier, not "
          "an enum — no 422 path exists",
          c1 == 404 and c2 == 404, f"{c1}/{c2}")

    print("== LIVE G: (E) true partition ==")
    ids_by = {}
    ok_groups2 = True
    for run_id in prov_ids:
        _, body = call_json("GET", byrun(run_id))
        ids = [c["checkpoint_id"] for c in body]
        ok_groups2 = ok_groups2 and len(ids) == len(set(ids)) \
            and all(c["run_id"] == run_id for c in body)
        ids_by[run_id] = set(ids)
    all_ids = {c["checkpoint_id"] for c in ckpts}
    pairwise = all(ids_by[a].isdisjoint(ids_by[b])
                   for i, a in enumerate(prov_ids) for b in prov_ids[i + 1:])
    check("G1/E1 the run groups form a TRUE disjoint partition of the "
          "FULL M3 listing (2 + 1 == 3; every record EXACTLY ONCE; no "
          "record in two groups; no None case — run_id is required) "
          "and every checkpoint's run_id is registered in the manifest "
          "training_provenance (total coverage)",
          ok_groups2 and pairwise
          and set().union(*ids_by.values()) == all_ids
          and len(set().union(*ids_by.values())) == 3
          and {c["run_id"] for c in ckpts} == set(prov_ids))

    print("== LIVE H: (F) ordering ==")
    ok_order = True
    for run_id in prov_ids:
        _, body = call_json("GET", byrun(run_id))
        expected = [c for c in ckpts if c["run_id"] == run_id]
        ok_order = ok_order and body == expected and \
            [(c["step"], c["created_at"]) for c in body] == \
            sorted((c["step"], c["created_at"]) for c in body)
    check("H1/F1 every group preserves the authoritative M3 (step, "
          "created_at) ASCENDING order exactly (the M4 ordering, not "
          "replaced)", ok_order)

    print("== LIVE I: (G) M3/M4 regressions ==")
    _, ckpts2 = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
    _, model2 = call_json("GET", f"{BASE}/models/{MODEL}")
    ok_detail2 = all(
        call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/"
                  f"{c['checkpoint_id']}")[1] == c for c in ckpts2)
    check("I1/G1 M3 checkpoint surfaces unchanged (listing identical "
          "to LIVE A, every detail getter verbatim, manifest "
          "training_provenance identical — exactly the two run ids)",
          ckpts2 == ckpts and ok_detail2
          and [p["run_id"] for p in model2["training_provenance"]]
          == prov_ids)

    print("== LIVE J: (H) M24-M45 regressions ==")
    ok_evck = True
    for ck, n in M24_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/"
                                   f"by-checkpoint/{ck}")
        ok_evck = ok_evck and c == 200 and len(body) == n
    evd = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/by-dataset/"
                    f"{DATASET}")[1]
    evt = call_json("GET", f"{BASE}/models/{MODEL}/evaluations/by-tokenizer/"
                    f"{TOKENIZER}")[1]
    evs_by = {sp: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"evaluations/by-split/{sp}")[1]
              for sp in M36_DIST}
    stk_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"evaluations/by-state-kind/{k}")[1]
              for k in M38_DIST}
    check("J1/H1 M24/M28/M30/M36/M38 evaluation histories unchanged "
          "(3/3/3 by-checkpoint, 16 by-dataset, 16 by-tokenizer, "
          "by-split 14/2/0, by-state-kind checkpoint->9 / current->7, "
          "all with listing parity)",
          ok_evck and len(evd) == M28_COUNT
          and evd == [x for x in evals if x["dataset_id"] == DATASET]
          and len(evt) == M30_COUNT
          and evt == [x for x in evals if x["tokenizer_id"] == TOKENIZER]
          and all(len(evs_by[sp]) == n
                  and evs_by[sp] == [x for x in evals if x["split"] == sp]
                  for sp, n in M36_DIST.items())
          and all(len(stk_by[k]) == n
                  and stk_by[k] == [x for x in evals
                                    if x["state_kind"] == k]
                  for k, n in M38_DIST.items()))
    ok_cmp = True
    for ck, n in M26_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                                   f"by-checkpoint/{ck}")
        ok_cmp = ok_cmp and c == 200 and len(body) == n
    cmpd = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/by-dataset/"
                     f"{DATASET}")[1]
    cmpt = call_json("GET", f"{BASE}/models/{MODEL}/comparisons/"
                     f"by-tokenizer/{TOKENIZER}")[1]
    cps_by = {sp: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"comparisons/by-split/{sp}")[1]
              for sp in M36_DIST}
    vds_by = {v: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"comparisons/by-verdict/{v}")[1]
              for v in M39_DIST}
    c44_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/"
                           f"comparisons/by-state-kind/{k}")[1]
              for k in M44_DIST}
    check("J2/H1 M26/M29/M31/M37/M39/M44 comparison histories unchanged "
          "(025e->6 / 30a8->5 / 0511->1 by-checkpoint, 8 by-dataset, 8 "
          "by-tokenizer, by-split validation->8 / train->0 / test->0, "
          "by-verdict improved->3 / unchanged->3 / regressed->2, "
          "by-state-kind checkpoint->8 / current->2 either-side, all "
          "with listing parity)",
          ok_cmp and len(cmpd) == M29_COUNT
          and cmpd == [x for x in comps if x["dataset_id"] == DATASET]
          and len(cmpt) == M31_COUNT
          and cmpt == [x for x in comps
                       if x["tokenizer_id"] == TOKENIZER]
          and cps_by["validation"] == [x for x in comps
                                       if x["split"] == "validation"]
          and len(cps_by["validation"]) == M37_COUNT
          and cps_by["train"] == [] and cps_by["test"] == []
          and all(len(vds_by[v]) == n
                  and vds_by[v] == [x for x in comps
                                    if x["verdict"] == v]
                  for v, n in M39_DIST.items())
          and all(len(c44_by[k]) == n
                  and c44_by[k] == [x for x in comps
                                    if k in (x["state_a"]["state_kind"],
                                             x["state_b"]["state_kind"])]
                  for k, n in M44_DIST.items()))
    ok_smp = True
    for ck, n in M27_COUNTS.items():
        c, body = call_json("GET", f"{BASE}/models/{MODEL}/samples/"
                                   f"by-checkpoint/{ck}")
        ok_smp = ok_smp and c == 200 and len(body) == n
    smpt = call_json("GET", f"{BASE}/models/{MODEL}/samples/by-tokenizer/"
                     f"{TOKENIZER}")[1]
    sty_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"samples/by-strategy/{st}")[1]
              for st in M40_DIST}
    check("J3/H1 M27/M32/M40 sample histories unchanged (4/0/0 "
          "by-checkpoint, 4 by-tokenizer, by-strategy greedy->2 / "
          "temperature->2, all with listing parity)",
          ok_smp and len(smpt) == M32_COUNT
          and smpt == [x for x in samples
                       if x["tokenizer_id"] == TOKENIZER]
          and all(len(sty_by[st]) == n
                  and sty_by[st] == [x for x in samples
                                     if x["strategy"] == st]
                  for st, n in M40_DIST.items()))
    sqt = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality/"
                    f"by-tokenizer/{TOKENIZER}")[1]
    check("J4/H1 M33 sample-quality by-tokenizer unchanged (2)",
          len(sqt) == M33_COUNT
          and sqt == [x for x in call_json(
              "GET", f"{BASE}/models/{MODEL}/sample-quality")[1]
              if x["tokenizer_id"] == TOKENIZER])
    s_sum = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                      f"{SUITE}/summary")[1]
    srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                     f"by-checkpoint/{CK_0511}")[1]
    check("J5/H1 M22/M25 suite-run histories unchanged (10 by-suite + "
          "10 by-checkpoint + summary total 10)",
          len(runs10) == M25_COUNT and srck == runs10
          and s_sum["total_count"] == M25_COUNT)
    wfs_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"workflows/by-status/{st}")[1]
              for st in M42_DIST}
    check("J6/H1 M42/M35 workflow histories unchanged (by-status "
          "completed->9 / failed->3 / stopped->1 with listing parity, "
          "13 total, 2 by-recipe with listing parity)",
          all(len(wfs_by[st]) == n
              and wfs_by[st] == [x for x in workflows
                                 if x["status"] == st]
              for st, n in M42_DIST.items())
          and call_json("GET", f"{BASE}/models/{MODEL}/workflows")[1]
          == workflows
          and len(call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                            f"by-recipe/{KNOWN_RECIPE}")[1]) == 2)
    g41_by = {k: call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                           f"decisions/by-decision/{k}")[1]
              for k in M41_DIST}
    g43_by = {v: call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                           f"decisions/by-verdict/{v}")[1]
              for v in M43_DIST}
    g45_by = {bt: call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                            f"decisions/by-baseline-type/{bt}")[1]
              for bt in M45_DIST}
    check("J7/H1 M23/M34/M41/M43/M45 gate histories unchanged "
          "(by-decision passed->7 / failed->4, by-verdict "
          "improved->4 / regressed->3 / unchanged->2, by-baseline-type "
          "checkpoint->7 / current->2 / minimum_loss->2 / "
          "evaluation_result_hash->0, 4 by-comparison, 1 by-policy, "
          "all with listing parity)",
          all(len(g41_by[k]) == n
              and g41_by[k] == [d for d in gdecisions
                                if d["decision"] == k]
              for k, n in M41_DIST.items())
          and all(len(g43_by[v]) == n
                  and g43_by[v] == [d for d in gdecisions
                                    if d["verdict"] == v]
                  for v, n in M43_DIST.items())
          and all(len(g45_by[bt]) == n
                  and g45_by[bt] == [d for d in gdecisions
                                     if d["policy"]["baseline_type"] == bt]
                  for bt, n in M45_DIST.items())
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                            f"decisions/by-comparison/{KNOWN_COMP}")[1])
          == M34_COUNT
          and len(call_json("GET", f"{BASE}/models/{MODEL}/gates/"
                            f"decisions/by-policy/{POLICY}")[1])
          == M23_COUNT)

    print("== LIVE K: (I) M12/M16/M35 regressions ==")
    global_runs = call_json("GET", f"{BASE}/workflows/recipes/"
                            f"{KNOWN_RECIPE}/runs")[1]
    recipes = call_json("GET", f"{BASE}/workflows/recipes")[1]
    c_unk = call_json("GET", f"{BASE}/workflows/recipes/"
                      "ghost-recipe-46")[0]
    ok_zero = all(call_json("GET", f"{BASE}/models/{MODEL}/workflows/"
                            f"by-recipe/{r}")[1] == []
                  for r in ZERO_RUN_RECIPES)
    sq = call_json("GET", f"{BASE}/models/{MODEL}/sample-quality")[1]
    check("K1/I1 M12/M35 workflow surfaces unchanged (registry 7 "
          "verbatim, GLOBAL m12-live-suite runs 2, by-recipe 2, the 4 "
          "zero-run recipes -> 200 + [], unknown recipe 404); M16 "
          "evaluation ids still paired in sample-quality",
          len(global_runs) == 2 and len(recipes) == 7 and ok_zero
          and c_unk == 404
          and {x["evaluation_id"] for x in sq} == set(M16_EVAL_IDS))

    print("== LIVE L: (J) dashboard; (K) registries ==")
    code, d2 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("L1/J1 M17 dashboard result_hash + output unchanged",
          code == 200 and d2["result_hash"] == DASH_HASH and d2 == d0
          and d2["diagnostics"] == [])
    toks = call_json("GET", f"{BASE}/tokenizers")[1]
    models_reg = call_json("GET", f"{BASE}/models")[1]
    check("L2/K1 registries unchanged (M2 tokenizer registry exactly "
          "the ONE entry; M9 policy + probe-suite; M12/M14 recipe "
          "registry; M1 model registry exactly the TWO known models)",
          len(toks) == 1 and toks[0]["id"] == TOKENIZER
          and {m["id"] for m in models_reg} == {MODEL, OTHER_MODEL}
          and call_json("GET", f"{BASE}/policies/{POLICY}")[0] == 200
          and call_json("GET", f"{BASE}/probe-suites")[0] == 200
          and len(recipes) == 7)

    print("== LIVE M: OpenAPI + final storage audit (zero drift) ==")
    code_sp2, spec2 = call_json("GET", f"{SITE}/openapi.json")
    check("M1 OpenAPI exactly 78 paths, the new path exactly once "
          "(after the checkpoint listing, before the generic "
          "checkpoint-detail route; the M45/M44/M43/M42/M41 routes "
          "still present exactly once; model_id + run_id parameters; "
          "array-of-object response like the M3 family)",
          code_sp2 == 200 and len(spec2["paths"]) == 78
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and [p["name"] for p in
               spec2["paths"][NEW_PATH]["get"]["parameters"]]
          == ["model_id", "run_id"]
          and list(spec2["paths"]).index(CKPT_LIST_PATH)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(GENERIC_PATH)
          and list(spec2["paths"]).count(M45_PATH) == 1
          and list(spec2["paths"]).count(M44_PATH) == 1
          and list(spec2["paths"]).count(M43_PATH) == 1
          and list(spec2["paths"]).count(M42_PATH) == 1
          and list(spec2["paths"]).count(M41_PATH) == 1)
    _, grouped2 = call_json("GET", byrun(RUN_A))
    check("M2 by-run still deterministic at the end",
          grouped2 == groups[RUN_A])
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    missing = set(pre_snap) - set(f_snap)
    check("M3 every pre-existing file byte-identical", changed == []
          and missing == set(),
          f"{len(changed)} changed / {len(missing)} missing")
    check("M4 zero new files", new_files == set(),
          f"{len(new_files)} new")
    check("M5 totals unchanged 96/4,002,745/0", f_n == 96
          and f_bytes == 4_002_745 and f_tmp == 0,
          f"{f_n}/{f_bytes}/{f_tmp}")

    print()
    if FAILURES:
        print(f"M46 live smoke FAILED: {len(FAILURES)} check(s) "
              "failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M46 live smoke OK: narrow read-only MODEL-SCOPED by-run "
          "grouping of the M3 checkpoint history over the persisted "
          "REQUIRED run_id, ownership validated against the model's "
          "OWN manifest training_provenance (distribution DISCOVERED "
          "live: 291a16d755fc -> 2 / 85438934f86a -> 1 among 3 "
          "checkpoints — exact listing parity, membership NEVER "
          "derived from directories, steps, timestamps or parent ids, "
          "the provenance registry used ONLY for the 404, authoritative "
          "(step, created_at) order preserved, every record exactly "
          "once, TRUE disjoint partition with total provenance "
          "coverage and no None case, no separate run registry — the "
          "model manifest IS the registry), clean 404 separation "
          "(unknown model / unknown run / another model's run; "
          "run_id is a persisted identifier, not an enum — no 422 "
          "path), healthy-run invariant verified live (every "
          "registered production run owns >= 1 checkpoint; the "
          "zero-checkpoint 200 [] path is engine-real through the M3 "
          "listing's corruption resilience and covered by the M46 "
          "tests — never fabricated), deterministic byte-identical "
          "x3 repeats, M3/M4 + M22-M45 surfaces + dashboard hash + "
          "registries + OpenAPI 78 unchanged, ZERO production "
          "storage growth) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
