"""M70 LIVE certification smoke (production touched READ-ONLY — the
only DELETEs issued against production are the ones that must be
REFUSED; success deletions run on a DISPOSABLE COPY, executed exactly
ONCE).

Certifies `DELETE /api/v1/models/{id}/workflows/{workflow_id}` (+ the
3 sibling family routes) and the four family retention views —
explicit verified per-record retention for the four model-owned
evidence families — against the production root. Guard order, the
M69-canonical blocker source, the retention==DELETE==M69 headline
invariant, integrity-first refusals, atomic typed deletions and the
full dependency-unblock chain all proven LIVE.

Certified facts of this production root (byte-identical since the
M68 delta — m70_pre.sha256 == m69_pre.sha256, 52/52):
  1 model (a266c8480cb7, live-model); M69 records usage: 30 records
  (training_run 6, checkpoint 13, workflow 3, evaluation 4,
  comparison 2, gate 2), 57 references, 57 internal, 0 external.
  M70 records: 11 total — 3 workflows (LEAF, deletable), 4
  evaluations, 2 comparisons, 2 gates (all 8 BLOCKED). Blocker graph:
  gate 5117c7a5f895 <- workflow edcb773ef326; gate 983c7d65fab5 <-
  workflow 28c7631e1152; comparison 71224941a968 <- gate 5117c7a5f895;
  comparison 894e77ab88bc <- gate 983c7d65fab5; eval 0a5a1e331f90 <-
  comparison 894e77ab88bc + gate 983c7d65fab5; eval d44e1566c239 <-
  comparison 71224941a968 + gate 5117c7a5f895; eval 7afcaa09abdb <-
  workflow edcb773ef326; eval bcfaf78933a5 <- workflow 28c7631e1152.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8796"
API = BASE + "/api/v1"

MODEL_ID = "a266c8480cb7"

FAMILIES = {  # category -> (route family segment, id field)
    "workflow": ("workflows", "workflow_id"),
    "evaluation": ("evaluations", "eval_id"),
    "comparison": ("comparisons", "comparison_id"),
    "gate": ("gates/decisions", "decision_id"),
}
DIRS = {"workflow": "workflows", "evaluation": "evaluations",
        "comparison": "comparisons", "gate": "gates"}
PREF = {"workflow": "workflow", "evaluation": "eval",
        "comparison": "comp", "gate": "gate"}
RECORDS = {  # the 11 certified M70 records of this production root
    "workflow": ["28c7631e1152", "8f022ff69681", "edcb773ef326"],
    "evaluation": ["0a5a1e331f90", "7afcaa09abdb",
                   "bcfaf78933a5", "d44e1566c239"],
    "comparison": ["71224941a968", "894e77ab88bc"],
    "gate": ["5117c7a5f895", "983c7d65fab5"],
}
EXPECTED_BLOCKERS = {  # M69 non-lineage references, canonical order
    ("workflow", "28c7631e1152"): [],
    ("workflow", "8f022ff69681"): [],
    ("workflow", "edcb773ef326"): [],
    ("evaluation", "0a5a1e331f90"): [("comparison", "894e77ab88bc"),
                                      ("gate", "983c7d65fab5")],
    ("evaluation", "7afcaa09abdb"): [("workflow", "edcb773ef326")],
    ("evaluation", "bcfaf78933a5"): [("workflow", "28c7631e1152")],
    ("evaluation", "d44e1566c239"): [("comparison", "71224941a968"),
                                     ("gate", "5117c7a5f895")],
    ("comparison", "71224941a968"): [("gate", "5117c7a5f895")],
    ("comparison", "894e77ab88bc"): [("gate", "983c7d65fab5")],
    ("gate", "5117c7a5f895"): [("workflow", "edcb773ef326")],
    ("gate", "983c7d65fab5"): [("workflow", "28c7631e1152")],
}
# the disposable-copy unblock chain: round -> deletable set
CHAIN_ROUNDS = [
    {("workflow", "28c7631e1152"), ("workflow", "8f022ff69681"),
     ("workflow", "edcb773ef326")},
    {("gate", "5117c7a5f895"), ("gate", "983c7d65fab5"),
     ("evaluation", "7afcaa09abdb"), ("evaluation", "bcfaf78933a5")},
    {("comparison", "71224941a968"), ("comparison", "894e77ab88bc")},
    {("evaluation", "0a5a1e331f90"), ("evaluation", "d44e1566c239")},
]
TOT_RECS, TOT_REFS, INT_REFS, EXT_REFS = 30, 57, 57, 0
M69_COUNTS = {"training_run": 6, "checkpoint": 13, "workflow": 3,
              "evaluation": 4, "comparison": 2, "gate": 2}
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m70_pre.sha256"
PROD_ROOT = Path(__import__("os").environ.get(
    "FORGE_SMOKE_ROOT", "/home/user/ai-model-forge-data"))
LINEAGE = ("checkpoint", "run_provenance")

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((name, bool(cond), extra))
    print(f"{'PASS' if cond else 'FAIL'} {name} {extra}")


def jget(path: str):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        assert r.status == 200, (path, r.status)
        return json.loads(r.read()), r.read()


def jdelete(path: str):
    req = urllib.request.Request(API + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def inventory(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


def m69_flat(overview: dict) -> dict:
    return {(c["category"], r["record_id"]):
            [(x["category"], x["reference_id"]) for x in r["references"]]
            for c in overview["categories"] for r in c["records"]}


def retention_path(category: str, rid: str) -> str:
    seg, _ = FAMILIES[category]
    return f"/models/{MODEL_ID}/{seg}/{rid}/retention"


def delete_path(category: str, rid: str) -> str:
    seg, _ = FAMILIES[category]
    return f"/models/{MODEL_ID}/{seg}/{rid}"


# ============================ prelude (read-only) ========================== #
project, _ = jget("/project")
root = Path(project["storage_root"])
check("P1 project readable", project["model_count"] == 1
      and str(root) == str(PROD_ROOT))

pre_disk = inventory(root)
pre_ref = {}
for line in PRE_INVENTORY.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    pre_ref[rel.removeprefix("./")] = digest
check("P2 disk == m70_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

models, _ = jget("/models")
check("P3 model registry", [m["id"] for m in models] == [MODEL_ID])

ov, _ = jget(f"/models/{MODEL_ID}/records/usage")
counts = {c["category"]: len(c["records"]) for c in ov["categories"]}
check("P4 M69 usage certified counts (30 records / 57 refs / 57 int / 0 ext)",
      counts == M69_COUNTS and ov["total_records"] == TOT_RECS
      and ov["total_references"] == TOT_REFS
      and ov["internal_references"] == INT_REFS
      and ov["external_references"] == EXT_REFS)
m69 = m69_flat(ov)

# ============== the M70 retention views + the headline invariant =========== #
views: dict[tuple[str, str], dict] = {}
ok_v1, ok_v2, ok_v3, ok_v4 = True, True, True, True
n_blocked = 0
for category, ids in RECORDS.items():
    for rid in ids:
        v, raw = jget(retention_path(category, rid))
        views[(category, rid)] = v
        key = (category, rid)
        # identity + integrity + view shape
        if not (v["model_id"] == MODEL_ID and v["category"] == category
                and v["record_id"] == rid and v["created_at"]
                and v["integrity_verified"] is True
                and isinstance(v["files"], list) and v["files"]
                and v["size_bytes"] > 0
                and set(v) == {"model_id", "category", "record_id",
                               "created_at", "files", "size_bytes",
                               "integrity_verified", "deletable",
                               "blockers"}):
            ok_v1 = False
        # files/bytes == the real directory (the M63-boundary walk)
        rdir = (root / "models" / MODEL_ID / DIRS[category]
                / f"{PREF[category]}-{rid}")
        real = sorted(p.name for p in rdir.iterdir() if p.is_file())
        if not (v["files"] == real
                and v["size_bytes"] == sum(p.stat().st_size
                                           for p in rdir.iterdir()
                                           if p.is_file())):
            ok_v2 = False
        # HEADLINE: blockers == M69 non-lineage references (exact order)
        expect = [(c, i) for c, i in m69.get(key, []) if c not in LINEAGE]
        got = [(b["category"], b["reference_id"]) for b in v["blockers"]]
        if got != expect or got != EXPECTED_BLOCKERS[key]:
            ok_v3 = False
        # HEADLINE: deletable == no blockers (integrity already True)
        if v["deletable"] != (not v["blockers"]):
            ok_v4 = False
        n_blocked += bool(v["blockers"])
check("V1 all 11 retention views: identity + integrity + exact shape",
      ok_v1)
check("V2 view files/bytes == the real record directories", ok_v2)
check("V3 HEADLINE blockers == M69 non-lineage refs == the certified "
      "graph (8 blocked / 3 leaf)", ok_v3, f"{n_blocked} blocked")
check("V4 HEADLINE deletable == blockers empty, all 11", ok_v4)

# determinism: byte-identical repeat
_, raw1 = jget(retention_path("comparison", "894e77ab88bc"))
_, raw2 = jget(retention_path("comparison", "894e77ab88bc"))
check("V5 deterministic repeat (byte-identical)", raw1 == raw2)

# ============ the REFUSED deletions (the 8 blocked records) ================= #
ok_d1, ok_d2, ok_d3 = True, True, True
for (category, rid), v in views.items():
    if not v["blockers"]:
        continue  # the 3 leaf workflows: NEVER deleted on production
    status, body = jdelete(delete_path(category, rid))
    detail = body.get("detail", {})
    if status != 409:
        ok_d1 = False
        continue
    if not (set(detail) == {"message", "model_id", "category",
                            "record_id", "protected", "blockers"}
            and detail["model_id"] == MODEL_ID
            and detail["category"] == category
            and detail["record_id"] == rid
            and detail["protected"] is True
            and isinstance(detail["message"], str) and detail["message"]):
        ok_d2 = False
    if [(b["category"], b["reference_id"]) for b in detail["blockers"]] != \
            [(b["category"], b["reference_id"]) for b in v["blockers"]]:
        ok_d3 = False
check("D1 all 8 blocked records: DELETE -> 409", ok_d1)
check("D2 structured 409 shape (message/model_id/category/record_id/"
      "protected/blockers)", ok_d2)
check("D3 409 blockers == retention blockers (view == guard)", ok_d3)
check("D4 zero mutation after the 8 refusals", inventory(root) == pre_disk)

# the 3 leaf workflows are deletable — certified VIEW-ONLY here (the
# success path runs on the disposable copy below; production stays
# byte-identical)
check("D5 the 3 leaf workflows deletable=True (view-level; not "
      "deleted on production)",
      all(views[("workflow", w)]["deletable"] is True
          for w in RECORDS["workflow"]))

# ============================ scope 404s (read-only) ======================= #
ok404 = True
for category in RECORDS:
    seg, idf = FAMILIES[category]
    # unknown record: retention GET + resource DELETE, all 4 families
    try:
        urllib.request.urlopen(
            API + f"/models/{MODEL_ID}/{seg}/no-such-record/retention",
            timeout=30)
        ok404 = False  # must not 200
    except urllib.error.HTTPError as e:
        ok404 = ok404 and e.code == 404
    if jdelete(f"/models/{MODEL_ID}/{seg}/no-such-record")[0] != 404:
        ok404 = False
try:
    urllib.request.urlopen(
        API + "/models/no-m70/workflows/x/retention", timeout=30)
    ok404 = False
except urllib.error.HTTPError as e:
    ok404 = ok404 and e.code == 404
if jdelete("/models/no-m70/workflows/x")[0] != 404:
    ok404 = False
check("S1 unknown record / unknown model -> 404 (GET + DELETE, all "
      "four families)", ok404)

# ============================ OpenAPI (read-only) ========================== #
with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
del_paths = sorted(p for p, ops in spec["paths"].items() if "delete" in ops)
ret_paths = [f"/api/v1/models/{{model_id}}/{FAMILIES[c][0]}/"
             f"{{{FAMILIES[c][1]}}}/retention" for c in FAMILIES]
ok_oa = len(spec["paths"]) == 101 and len(del_paths) == 11 and all(
    set(spec["paths"][p].keys()) == {"get"} for p in ret_paths)
for category in FAMILIES:
    seg, idf = FAMILIES[category]
    p = f"/api/v1/models/{{model_id}}/{seg}/{{{idf}}}"
    ok_oa = ok_oa and set(spec["paths"][p].keys()) >= {"get", "delete"}
    ok_oa = ok_oa and spec["paths"][p]["delete"]["responses"]["409"][
        "content"]["application/json"]["schema"].get(
            "$ref") == "#/components/schemas/ModelRecordDeletionBlocked"
ok_oa = ok_oa and all(
    s in spec["components"]["schemas"] for s in (
        "ModelRecordDeletionBlocker", "ModelRecordDeletionResult",
        "ModelRecordDeletionBlocked", "ModelRecordRetentionOverview"))
check("O1 OpenAPI: 101 paths / 11 deletes / 4 GET-only retention "
      "routes / 4 new DELETE ops / 4 schemas / 409 $ref", ok_oa)

check("A1 production byte-identical after all read-only checks",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()),
      f"{len(pre_disk)} files")

# =========== disposable copy: integrity refusal + THE FULL CHAIN =========== #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.engine import ModelForge  # noqa: E402

copy_root = Path(tempfile.mkdtemp(prefix="m70-copy-"))
try:
    shutil.copytree(PROD_ROOT, copy_root, dirs_exist_ok=True)
    forge = ModelForge(root=copy_root)
    cpre = inventory(copy_root)

    # C1: the copy reproduces every production retention view exactly
    ok_c1 = True
    for (category, rid), v in views.items():
        cv = forge.model_record_retention_overview(
            MODEL_ID, category, rid).model_dump(mode="json")
        if cv != v:
            ok_c1 = False
    check("C1 copy: all 11 retention views == production (determinism "
          "over identical state)", ok_c1)

    # C2: integrity-first refusal, live — tamper ONE gate's
    # result_hash, DELETE must refuse WITHOUT the blocker analysis
    # changing anything on disk
    gdir = copy_root / "models" / MODEL_ID / "gates" / "gate-983c7d65fab5"
    gman = gdir / "manifest.json"
    original = gman.read_bytes()
    tampered = json.loads(original)
    tampered["result_hash"] = "0" * 64
    gman.write_text(json.dumps(tampered, indent=2, sort_keys=True))
    rv = forge.model_record_retention_overview(
        MODEL_ID, "gate", "983c7d65fab5")
    check("C2a copy: tampered record -> integrity_verified False, "
          "deletable False, blockers still reported",
          rv.integrity_verified is False and rv.deletable is False
          and [(b.category, b.reference_id) for b in rv.blockers] ==
          [("workflow", "28c7631e1152")])
    try:
        forge.delete_model_record(MODEL_ID, "gate", "983c7d65fab5")
        refused = False
    except RuntimeError:
        refused = True
    except ValueError:
        refused = False  # wrong refusal class: integrity must come FIRST
    check("C2b copy: tampered DELETE -> RuntimeError (integrity "
          "FIRST, before blockers)", refused and gman.exists())
    # restore, integrity recovers
    gman.write_bytes(original)
    rv = forge.model_record_retention_overview(
        MODEL_ID, "gate", "983c7d65fab5")
    check("C2c copy: restored -> integrity_verified True again",
          rv.integrity_verified is True and rv.deletable is False
          and inventory(copy_root) == cpre)

    # C3: engine scope contract — checkpoints / training runs are
    # out of M70 scope (ValueError), unknown records FileNotFoundError
    ok_c3 = True
    for bad_cat in ("checkpoint", "training_run", "dataset"):
        try:
            forge.delete_model_record(MODEL_ID, bad_cat, "whatever")
            ok_c3 = False
        except ValueError:
            pass
        except Exception:
            ok_c3 = False
    try:
        forge.delete_model_record(MODEL_ID, "workflow", "no-such")
        ok_c3 = False
    except FileNotFoundError:
        pass
    check("C3 copy: out-of-scope category -> ValueError; unknown "
          "record -> FileNotFoundError", ok_c3)

    # C4: THE FULL UNBLOCK CHAIN — delete-all-deletable per round;
    # every round's deletable set must match the certified graph
    deleted: list[tuple[str, str]] = []
    ok_c4a, ok_c4b, ok_c4c = True, True, True
    expected_rounds = [set(r) for r in CHAIN_ROUNDS]
    for rnd, expected in enumerate(expected_rounds, 1):
        current = {}
        for category, ids in RECORDS.items():
            for rid in ids:
                if (category, rid) in deleted:
                    continue  # already gone (404 now)
                try:
                    v = forge.model_record_retention_overview(
                        MODEL_ID, category, rid)
                except FileNotFoundError:
                    continue
                current[(category, rid)] = v
        deletable = {k for k, v in current.items() if v.deletable}
        if deletable != expected:
            ok_c4a = False
        # one still-blocked record refuses with exactly its view blockers
        blocked_here = [k for k, v in current.items() if not v.deletable]
        if blocked_here:
            k = blocked_here[0]
            try:
                forge.delete_model_record(MODEL_ID, k[0], k[1])
                ok_c4b = False
            except ValueError:
                bl = forge.model_record_deletion_blockers(
                    MODEL_ID, k[0], k[1])
                if [(b.category, b.reference_id) for b in bl] != \
                        [(b.category, b.reference_id)
                         for b in current[k].blockers]:
                    ok_c4b = False
        for k in sorted(expected):
            v = current[k]
            res = forge.delete_model_record(MODEL_ID, k[0], k[1])
            if not (res.model_id == MODEL_ID and res.category == k[0]
                    and res.record_id == k[1]
                    and res.files_removed == 1
                    and res.bytes_reclaimed == v.size_bytes):
                ok_c4c = False
            deleted.append(k)
    # nothing left of the four families
    leftovers = []
    for category, ids in RECORDS.items():
        for rid in ids:
            try:
                forge.model_record_retention_overview(
                    MODEL_ID, category, rid)
                leftovers.append((category, rid))
            except FileNotFoundError:
                pass
    check("C4a copy: every chain round's deletable set == the "
          "certified graph (3 -> 4 -> 2 -> 2)", ok_c4a)
    check("C4b copy: blocked records keep refusing with EXACTLY the "
          "view blockers (engine view == guard)", ok_c4b and not leftovers)
    check("C4c copy: 11 atomic deletions, typed results, bytes == the "
          "retention views", ok_c4c and len(deleted) == 11,
          f"{len(deleted)} deleted")

    # C5: the post-chain M69 view == the pre view minus the 11 records
    # minus every reference FROM a deleted record (exact, per record,
    # per order) — the ONE scanner stayed consistent through it all
    deleted_ids = {rid for _, rid in deleted}
    post = forge.model_records_usage_overview(MODEL_ID)
    expected_refs = {
        k: [(c, i) for c, i in refs if i not in deleted_ids]
        for k, refs in m69.items() if k not in deleted}
    got_refs = m69_flat(post.model_dump(mode="json"))
    post_counts = {c.category: len(c.records) for c in post.categories}
    check("C5 copy: post-chain M69 == pre minus the 11 records minus "
          "their outgoing refs (19 records; families empty)",
          got_refs == expected_refs
          and post_counts == {"training_run": 6, "checkpoint": 13,
                              "workflow": 0, "evaluation": 0,
                              "comparison": 0, "gate": 0}
          and post.total_records == 19)

    # C6: upstream M62 shrink — every checkpoint's blockers are now
    # model-pointer only (all eval/comparison/gate/workflow refs died
    # with the four families); M62 agrees with M69 per record
    ret62 = forge.checkpoint_retention_overview(MODEL_ID)
    m62 = {e.checkpoint_id: e for e in ret62.checkpoints}
    post_refs = got_refs
    ok_c6 = ret62.total_checkpoints == 13
    for (cat, rid), refs in post_refs.items():
        if cat != "checkpoint":
            continue
        exp_deletable = not any(c not in LINEAGE for c, _ in refs)
        if m62[rid].deletable != exp_deletable:
            ok_c6 = False
        if any(c in ("evaluation", "comparison", "gate", "workflow")
               for c, _ in refs):
            ok_c6 = False
    n_ptr = len({i for (cat, _), refs in post_refs.items()
                 if cat == "checkpoint" for c, i in refs if c == "model"})
    check("C6 copy: M62 upstream shrink — checkpoint blockers now "
          "model-pointer only; deletable == no non-lineage refs",
          ok_c6 and ret62.deletable_checkpoints == 13 - n_ptr
          and ret62.protected_checkpoints == n_ptr,
          f"{ret62.deletable_checkpoints} deletable / "
          f"{ret62.protected_checkpoints} protected")

    # C7: exact storage effect — ONLY the 11 record manifests left;
    # every other byte identical; total reclaimed == the summed views
    cpost = inventory(copy_root)
    removed = sorted(set(cpre) - set(cpost))
    removed_root = {f"models/{MODEL_ID}/{DIRS[c]}/"
                    f"{PREF[c]}-{rid}/manifest.json"
                    for c, rid in deleted}
    ok_c7 = (set(removed) == removed_root and len(removed) == 11
             and {k: v for k, v in cpost.items()} ==
             {k: v for k, v in cpre.items() if k not in removed_root}
             and not (set(cpost) - set(cpre)))
    reclaimed = sum(v["size_bytes"] for v in views.values())
    check("C7 copy: exactly the 11 record manifests removed, all "
          "other bytes identical", ok_c7,
          f"11 files / {reclaimed} bytes reclaimed")

    # C8: after deletion — GET-one 404, retention 404, repeat DELETE 404
    ok_c8 = True
    for category, rid in deleted[:2] + deleted[-2:]:
        try:
            forge.model_record_retention_overview(MODEL_ID, category, rid)
            ok_c8 = False
        except FileNotFoundError:
            pass
        try:
            forge.delete_model_record(MODEL_ID, category, rid)
            ok_c8 = False
        except FileNotFoundError:
            pass
    check("C8 copy: deleted records -> retention 404 + repeat DELETE "
          "404 (no resurrection)", ok_c8)
finally:
    shutil.rmtree(copy_root, ignore_errors=True)
check("C9 copy discarded", not copy_root.exists())

# final production audit
check("A2 production byte-identical, final (52/52)",
      inventory(root) == pre_disk and not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M70 CHECKS PASSED "
      "(success deletions on a discarded copy; production untouched)")
sys.exit(0 if n_pass == len(results) else 1)
