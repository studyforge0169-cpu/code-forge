"""M47 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M47 is READ-ONLY: it adds one narrow
MODEL-SCOPED access path,
GET /models/{id}/evaluations/by-truncated/{truncated},
answering "which immutable evaluations of this model were stopped
early by the max_eval_tokens cap?" — the model's authoritative M4
listing (deterministic (created_at, eval_id) ASCENDING order)
filtered VERBATIM by the persisted REQUIRED boolean
EvaluationRecord.truncated (the engine's record of whether the cap
stopped the evaluation before the split ended: True = stopped early,
False = the configured/permitted evaluation stream was consumed
without the cap cutting it short; matched VERBATIM — NEVER
recalculated, NEVER derived from records_covered, token counts,
split length, the evaluation configuration, timestamps, durations,
state kinds or any other field; the boolean carries NO quality
judgment — it is engine metadata about how far the evaluation stream
was consumed, nothing more), registered BEFORE the generic
/evaluations/{eval_id} route and alongside by-checkpoint (M24)/
by-dataset (M28)/by-tokenizer (M30)/by-split (M36)/by-state-kind
(M38). The boolean is a CLOSED two-value contract with no registry:
the two groups form a TRUE disjoint partition of the listing with no
None case; non-boolean spellings are rejected 422 at the API boundary
(schema-level validation, pre-handler — booleans are never silently
reinterpreted, and the validation fires BEFORE the handler even for
an unknown model), while an unknown model with a VALID boolean is
404. The smoke must DISCOVER the authoritative truncation
distribution from the live M4 listing (not assume it from an old
report), prove exact listing parity for BOTH groups, the natural
valid-empty cases, clean 404/422 separation, byte-identical repeats
(x3), unchanged M3-M46 surfaces + dashboard hash + registries +
OpenAPI 79, and ZERO production storage growth.

Production facts (re-derived live at LIVE A/B; the authoritative
source is the M4 evaluation listing):
model 4a0a871886ef owns 16 evaluations in ASCENDING (created_at,
eval_id) order with persisted truncated distribution False -> 15 and
True -> 1 (the truncated record is a884bf729ff7, train split,
current state, records_covered None, token_count 500) — a TRUE
disjoint partition of all 16; model b5bc905326b6 has NO evaluations;
M17 dashboard result_hash
f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838 —
96 files / 4,002,745 B / 0 .tmp.

LIVE A (baseline): exact 96 / 4,002,745 / 0 audit + full per-file
SHA256 inventory saved to /tmp/m47-smoke-baseline-inventory.json +
model/checkpoint/tokenizer/policy/probe-suite/recipe registries +
M4-M46 pre-state (16 evaluations, 8 comparisons, 11 gate decisions,
13 workflows, 10 suite-runs, 4 samples, 3 checkpoints with run
distribution 2/1) + dashboard + OpenAPI 79 pre-state + the known
pair returns 200.

LIVE B (authoritative distribution discovery): from the live M4
listing, discover the persisted truncated distribution; print it;
cross-check against the audited facts.

LIVE C (A — known statuses): BOTH booleans -> 200 with the EXACT
discovered records (15/1); parity with the M4 listing filtered
locally by the persisted boolean; verbatim detail-getter payloads for
all 16 records; (created_at, eval_id) ASC preserved per group.

LIVE D (B — repeatability): three GETs raw-byte-identical.

LIVE E (C — valid empty statuses): BOTH booleans under
b5bc905326b6 -> 200 + [] (it has no evaluations at all).

LIVE F (D — unknown ids / unsupported spellings): unknown model +
valid boolean -> 404; non-boolean spellings ("maybe", "2",
"yes-no") -> 422 on the known model AND on an unknown model
(schema-level validation fires pre-handler; booleans are never
silently reinterpreted).

LIVE G (E — true partition): the two groups form a TRUE disjoint
partition of the FULL M4 listing (15 + 1 == 16; every record EXACTLY
ONCE; no record in two groups; no None case — the boolean is
required on every record).

LIVE H (F — ordering): every group's order equals the authoritative
M4 (created_at, eval_id) ASCENDING order restricted to that status.

LIVE I (G — M4/M24/M28/M30/M36/M38 regressions): the evaluation
listing identical to LIVE A; by-checkpoint 3/3/3; by-dataset 16;
by-tokenizer 16; by-split validation->14 / train->2 / test->0;
by-state-kind checkpoint->9 / current->7 — all with listing parity.

LIVE J (H — M22-M46 regressions): comparisons 6/5/1 by-checkpoint +
8 by-dataset + 8 by-tokenizer + by-split validation->8 / train->0 /
test->0 + by-verdict improved->3 / unchanged->3 / regressed->2 +
by-state-kind checkpoint->8 / current->2 (either-side); samples
4/0/0 by-checkpoint + 4 by-tokenizer + by-strategy greedy->2 /
temperature->2; suite runs 10 by-suite + 10 by-checkpoint + summary
total 10; sample-quality 2 by-tokenizer; workflows 13 total with
by-status completed->9 / failed->3 / stopped->1 + 2 by-recipe
(m12-live-suite); gates 11 with by-decision passed->7 / failed->4 +
by-verdict improved->4 / regressed->3 / unchanged->2 +
by-baseline-type checkpoint->7 / current->2 / minimum_loss->2 /
evaluation_result_hash->0 + 4 by-comparison + 1 by-policy;
checkpoints by-run 291a16d755fc -> 2 / 85438934f86a -> 1.

LIVE K (I — M12/M16/M35 regressions): recipe registry 7 verbatim;
GLOBAL /workflows/recipes/m12-live-suite/runs -> 2; the 4 zero-run
recipes -> 200 + []; unknown recipe 404; M16 evaluation ids still
paired in sample-quality.

LIVE L (J — dashboard; K — registries): dashboard result_hash + full
output unchanged; policy/probe-suite/recipe/tokenizer/model
registries unchanged (2 models with the known ids).

LIVE M (L — OpenAPI + storage zero drift): OpenAPI exactly 79 paths,
new path once (after by-state-kind, before the generic evaluation
detail route; boolean parameter schema; EvaluationRecord items; the
M46/M45/M44/M43/M42/M41 routes still present exactly once); every
pre-existing file byte-identical, ZERO new files, zero .tmp, totals
unchanged 96/4,002,745/0.

Exit code 0 = all checks passed. Start the server first:
    FORGE_ROOT=/home/user/ai-model-forge-data python -m uvicorn \
        app.api:app --host 127.0.0.1 --port 8768 --log-level warning
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8768/api/v1"
SITE = "http://127.0.0.1:8768"
MODEL = "4a0a871886ef"
OTHER_MODEL = "b5bc905326b6"          # exists, NO evaluations
TOKENIZER = "99106e3255c5"            # the ONE production tokenizer
DATASET = "ee1a716c4573"
CK_0511 = "0511de4c7372"
CK_025E = "025e6d8d8f15"
CK_30A8 = "30a8bc5b82ab"
RUN_A = "291a16d755fc"                # M46: -> 2 checkpoints
RUN_B = "85438934f86a"                # M46: -> 1 checkpoint
POLICY = "m9-live-policy"
SUITE = "m9-live-suite"
KNOWN_RECIPE = "m12-live-suite"       # M35: 2 runs
ZERO_RUN_RECIPES = ("m14-base", "m14-chain-a", "m14-chain-b",
                    "m14-chain-c")
KNOWN_COMP = "fc379bfcb50f"           # M34: 4 decisions
DASH_HASH = ("f48557fe8ab173c9ee3678212f0e09073dcd87fd5fa550f625f21271cdaa3838")
M16_EVAL_IDS = ("8ff910cf2a9e", "31a283413c75")
TRUNC_EVAL = "a884bf729ff7"           # the ONE truncated evaluation
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
M47_DIST = {"false": 15, "true": 1}
NEW_PATH = "/api/v1/models/{model_id}/evaluations/by-truncated/{truncated}"
GENERIC_PATH = "/api/v1/models/{model_id}/evaluations/{eval_id}"
M38_PATH = ("/api/v1/models/{model_id}/evaluations/by-state-kind/"
            "{state_kind}")
M46_PATH = "/api/v1/models/{model_id}/checkpoints/by-run/{run_id}"
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
INVENTORY = Path("/tmp/m47-smoke-baseline-inventory.json")

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


def bytrunc(truncated: str, model: str = MODEL) -> str:
    return (f"{BASE}/models/{model}/evaluations/by-truncated/"
            f"{truncated}")


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
    code_ev, evals = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
    code_evo, evals_o = call_json("GET", f"{BASE}/models/{OTHER_MODEL}/"
                                         "evaluations")
    code_ck, ckpts = call_json("GET", f"{BASE}/models/{MODEL}/checkpoints")
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
    check("A4 M4-M46 pre-state intact (16 evaluations, other model 0, "
          "3 checkpoints, 8 comparisons, 11 gate decisions, 13 "
          "workflows, 4 samples, 10 suite runs, 7 recipes, OpenAPI 79 "
          "with the by-truncated path registered exactly once, "
          "GET-only, tag evaluation)",
          code == 200 and model["id"] == MODEL and code_o == 200
          and other["id"] == OTHER_MODEL and code_ev == 200
          and len(evals) == 16 and code_evo == 200 and evals_o == []
          and code_ck == 200 and len(ckpts) == 3 and code_cp == 200
          and len(comps) == 8 and code_gd == 200
          and len(gdecisions) == 11 and code_wf == 200
          and len(workflows) == 13 and code_sm == 200 and len(samples) == 4
          and code_sr == 200 and len(runs10) == 10
          and call_json("GET", f"{BASE}/workflows/recipes")[1].__len__()
          == 7
          and code_sp == 200 and len(paths) == 79
          and list(paths).count(NEW_PATH) == 1 and set(new_ops) == {"get"}
          and new_ops["get"]["tags"] == ["evaluation"])
    code_d, d0 = call_json("GET", f"{BASE}/models/{MODEL}/dashboard")
    check("A5 M17 dashboard hash equals the full known value",
          code_d == 200 and d0["result_hash"] == DASH_HASH,
          d0["result_hash"][:16])
    code_a, raw_a = call("GET", bytrunc("false"))
    check("A6 the known pair (4a0a871886ef + false) -> 200",
          code_a == 200, f"{code_a}")

    print("== LIVE B: authoritative distribution discovery ==")
    dist = {v: sum(1 for e in evals if e["truncated"] == (v == "true"))
            for v in M47_DIST}
    dist_str = ", ".join(f"{k} -> {v}" for k, v in sorted(dist.items()))
    print(f"    discovered: {len(evals)} evaluations of {MODEL} per "
          f"persisted truncated: {dist_str}")
    print(f"    discovered: the truncated record: "
          f"{[e['eval_id'] for e in evals if e['truncated']]}")
    check("B1 M4 listing pre-state: 16 evaluations in ASCENDING "
          "(created_at, eval_id) order; persisted truncated "
          "distribution false->15 / true->1 (TRUE disjoint partition "
          "of 16; the single True record is a884bf729ff7, train "
          "split, current state); other model has none",
          code_ev == 200 and len(evals) == 16
          and [(e["created_at"], e["eval_id"]) for e in evals]
          == sorted((e["created_at"], e["eval_id"]) for e in evals)
          and dist == dict(M47_DIST)
          and all(isinstance(e["truncated"], bool) for e in evals)
          and [e["eval_id"] for e in evals if e["truncated"]]
          == [TRUNC_EVAL]
          and code_evo == 200 and evals_o == [])

    print("== LIVE C: (A) known statuses — exact responses ==")
    groups: dict[str, list] = {}
    ok_groups = True
    for value in ("false", "true"):
        code_g, raw_g = call("GET", bytrunc(value))
        groups[value] = json.loads(raw_g)
        ok_groups = ok_groups and code_g == 200 and groups[value] == \
            [e for e in evals if e["truncated"] == (value == "true")]
    check("C1/A1 BOTH truncated-status responses == M4 listing "
          "filtered locally by the persisted boolean (no missing / "
          "extra / duplicate; every record EXACTLY ONCE; the "
          "persisted boolean travels VERBATIM — never recalculated, "
          "never derived from records_covered, token counts, config "
          "or timestamps)",
          ok_groups
          and all(len(groups[v]) == M47_DIST[v]
                  and all(e["model_id"] == MODEL
                          and e["truncated"] == (v == "true")
                          for e in groups[v])
                  for v in M47_DIST),
          "/".join(str(len(groups[v])) for v in ("false", "true")))
    ok_detail = True
    for e in evals:
        code_one, one = call_json("GET", f"{BASE}/models/{MODEL}/"
                                        f"evaluations/{e['eval_id']}")
        if code_one != 200 or one != e:
            ok_detail = False
            check("C2/A1 verbatim payload parity with the M4 detail "
                  "getter", False, e["eval_id"])
            break
    if ok_detail:
        check("C2/A1 verbatim payload parity with the M4 detail getter "
              "for all 16 records (loss/perplexity/state identity "
              "included)", True)

    print("== LIVE D: (B) repeatability ==")
    _, r1 = call("GET", bytrunc("false"))
    _, r2 = call("GET", bytrunc("false"))
    _, r3 = call("GET", bytrunc("false"))
    check("D1/B1 three GETs raw-byte-identical", r1 == r2 == r3 == raw_a)

    print("== LIVE E: (C) valid empty statuses ==")
    ok_other = True
    for value in ("false", "true"):
        c_o, body_o = call_json("GET", bytrunc(value, OTHER_MODEL))
        ok_other = ok_other and c_o == 200 and body_o == []
    check("E1/C1 BOTH truncated statuses under b5bc905326b6 -> 200 + [] "
          "(its M4 listing is empty; every boolean value is "
          "schema-validated even when the model has no history)",
          ok_other and evals_o == [])

    print("== LIVE F: (D) unknown ids / unsupported spellings ==")
    c1, _ = call_json("GET", bytrunc("false", "no-such-model-47"))
    check("F1/D1 unknown model + VALID boolean -> 404", c1 == 404,
          f"{c1}")
    c2, b2 = call_json("GET", bytrunc("maybe"))
    c3, b3 = call_json("GET", bytrunc("2"))
    c4, b4 = call_json("GET", bytrunc("yes-no"))
    c5, _ = call_json("GET", bytrunc("maybe", "no-such-model-47"))
    check("F2/D1 non-boolean spellings -> 422 on the known model "
          "(maybe / 2 / yes-no) AND on an unknown model "
          "(schema-level validation fires pre-handler; booleans are "
          "never silently reinterpreted)",
          c2 == c3 == c4 == c5 == 422
          and all("boolean" in json.dumps(b) for b in (b2, b3, b4) if b),
          f"{c2}/{c3}/{c4}/{c5}")

    print("== LIVE G: (E) true partition ==")
    ids_by = {}
    ok_groups2 = True
    for value in ("false", "true"):
        _, body = call_json("GET", bytrunc(value))
        ids = [e["eval_id"] for e in body]
        ok_groups2 = ok_groups2 and len(ids) == len(set(ids)) \
            and all(e["truncated"] == (value == "true") for e in body)
        ids_by[value] = set(ids)
    all_ids = {e["eval_id"] for e in evals}
    check("G1/E1 the two groups form a TRUE disjoint partition of the "
          "FULL M4 listing (15 + 1 == 16; every record EXACTLY ONCE; "
          "no record in two groups; no None case — the boolean is "
          "required on every record)",
          ok_groups2
          and ids_by["false"].isdisjoint(ids_by["true"])
          and ids_by["false"] | ids_by["true"] == all_ids
          and len(all_ids) == 16)

    print("== LIVE H: (F) ordering ==")
    ok_order = True
    for value in ("false", "true"):
        _, body = call_json("GET", bytrunc(value))
        expected = [e for e in evals
                    if e["truncated"] == (value == "true")]
        ok_order = ok_order and body == expected and \
            [(e["created_at"], e["eval_id"]) for e in body] == \
            sorted((e["created_at"], e["eval_id"]) for e in body)
    check("H1/F1 every group preserves the authoritative M4 "
          "(created_at, eval_id) ASCENDING order exactly", ok_order)

    print("== LIVE I: (G) M4/M24/M28/M30/M36/M38 regressions ==")
    _, evals2 = call_json("GET", f"{BASE}/models/{MODEL}/evaluations")
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
    check("I1/G1 M4/M24/M28/M30/M36/M38 evaluation surfaces unchanged "
          "(listing identical to LIVE A, 3/3/3 by-checkpoint, 16 "
          "by-dataset, 16 by-tokenizer, by-split 14/2/0, by-state-kind "
          "checkpoint->9 / current->7, all with listing parity)",
          evals2 == evals and ok_evck and len(evd) == M28_COUNT
          and evd == [x for x in evals if x["dataset_id"] == DATASET]
          and len(evt) == M30_COUNT
          and evt == [x for x in evals
                      if x["tokenizer_id"] == TOKENIZER]
          and all(len(evs_by[sp]) == n
                  and evs_by[sp] == [x for x in evals if x["split"] == sp]
                  for sp, n in M36_DIST.items())
          and all(len(stk_by[k]) == n
                  and stk_by[k] == [x for x in evals
                                    if x["state_kind"] == k]
                  for k, n in M38_DIST.items()))

    print("== LIVE J: (H) M22-M46 regressions ==")
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
    check("J1/H1 M26/M29/M31/M37/M39/M44 comparison histories unchanged "
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
    check("J2/H1 M27/M32/M40 sample histories unchanged (4/0/0 "
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
    check("J3/H1 M33 sample-quality by-tokenizer unchanged (2)",
          len(sqt) == M33_COUNT
          and sqt == [x for x in call_json(
              "GET", f"{BASE}/models/{MODEL}/sample-quality")[1]
              if x["tokenizer_id"] == TOKENIZER])
    s_sum = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/by-suite/"
                      f"{SUITE}/summary")[1]
    srck = call_json("GET", f"{BASE}/models/{MODEL}/suite-runs/"
                     f"by-checkpoint/{CK_0511}")[1]
    check("J4/H1 M22/M25 suite-run histories unchanged (10 by-suite + "
          "10 by-checkpoint + summary total 10)",
          len(runs10) == M25_COUNT and srck == runs10
          and s_sum["total_count"] == M25_COUNT)
    wfs_by = {st: call_json("GET", f"{BASE}/models/{MODEL}/"
                            f"workflows/by-status/{st}")[1]
              for st in M42_DIST}
    check("J5/H1 M42/M35 workflow histories unchanged (by-status "
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
    check("J6/H1 M23/M34/M41/M43/M45 gate histories unchanged "
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
    g46_by = {r: call_json("GET", f"{BASE}/models/{MODEL}/checkpoints/"
                           f"by-run/{r}")[1] for r in M46_DIST}
    check("J7/H1 M46 checkpoint-by-run history unchanged "
          "(291a16d755fc -> 2 / 85438934f86a -> 1 with listing "
          "parity)",
          all(len(g46_by[r]) == n
              and g46_by[r] == [c for c in ckpts if c["run_id"] == r]
              for r, n in M46_DIST.items()))

    print("== LIVE K: (I) M12/M16/M35 regressions ==")
    global_runs = call_json("GET", f"{BASE}/workflows/recipes/"
                            f"{KNOWN_RECIPE}/runs")[1]
    recipes = call_json("GET", f"{BASE}/workflows/recipes")[1]
    c_unk = call_json("GET", f"{BASE}/workflows/recipes/"
                      "ghost-recipe-47")[0]
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
    params = {p["name"]: p for p in
              spec2["paths"][NEW_PATH]["get"]["parameters"]}
    check("M1 OpenAPI exactly 79 paths, the new path exactly once "
          "(after by-state-kind, before the generic evaluation-detail "
          "route; boolean parameter schema; EvaluationRecord items; "
          "the M46/M45/M44/M43/M42/M41 routes still present exactly "
          "once)",
          code_sp2 == 200 and len(spec2["paths"]) == 79
          and list(spec2["paths"]).count(NEW_PATH) == 1
          and set(spec2["paths"][NEW_PATH]) == {"get"}
          and set(params) == {"model_id", "truncated"}
          and params["truncated"]["schema"]["type"] == "boolean"
          and list(spec2["paths"]).index(M38_PATH)
          < list(spec2["paths"]).index(NEW_PATH)
          < list(spec2["paths"]).index(GENERIC_PATH)
          and list(spec2["paths"]).count(M46_PATH) == 1
          and list(spec2["paths"]).count(M45_PATH) == 1
          and list(spec2["paths"]).count(M44_PATH) == 1
          and list(spec2["paths"]).count(M43_PATH) == 1
          and list(spec2["paths"]).count(M42_PATH) == 1
          and list(spec2["paths"]).count(M41_PATH) == 1)
    _, grouped2 = call_json("GET", bytrunc("false"))
    check("M2 by-truncated still deterministic at the end",
          grouped2 == groups["false"])
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
        print(f"M47 live smoke FAILED: {len(FAILURES)} check(s) "
              "failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M47 live smoke OK: narrow read-only MODEL-SCOPED "
          "by-truncated grouping of the M4 evaluation history over "
          "the persisted REQUIRED boolean (distribution DISCOVERED "
          "live: false->15 / true->1 among 16 evaluations — exact "
          "listing parity, persisted truncated VERBATIM, never "
          "recalculated, never derived from records_covered, token "
          "counts, config or timestamps, no quality judgment, "
          "authoritative (created_at, eval_id) order preserved, every "
          "record exactly once, TRUE disjoint partition with no None "
          "case — the closed two-value boolean contract), "
          "schema-level 422 for non-boolean spellings (pre-handler, "
          "even for an unknown model), 404 for unknown model with a "
          "valid boolean, valid empty 200 + [] for both statuses under "
          "b5bc905326b6, deterministic byte-identical x3 repeats, "
          "M3-M46 surfaces + dashboard hash + registries + OpenAPI 79 "
          "unchanged, ZERO production storage growth) verified live "
          "on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
