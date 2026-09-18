"""M71 LIVE certification smoke (production touched READ-ONLY except
ONE DELETE that must be REFUSED — the recipe is referenced by two
production workflow runs, so a 409 is GUARANTEED; success deletions
run on a DISPOSABLE COPY, executed exactly ONCE).

Certifies `DELETE /api/v1/workflows/recipes/{recipe_id}` +
`DELETE /api/v1/policies/{policy_id}` +
`DELETE /api/v1/probe-suites/{suite_id}` (+ the three read-only
`.../retention` views) — explicit verified DEFINITION retention for
the three root-level definition families — against the production
root. Guard order, the raw-manifest blocker source, the
retention == DELETE == usage headline invariant, the §7
model-unblock coupling (M66 external reference shrinks live; the
LAST one flips the untouched M67 model retention state), the §8
dependent-unblock chains (M70 records, M68 suite runs, structural
recipe references) and integrity-first refusals all proven LIVE.

Certified facts of this production root (byte-identical since the
M68 delta — m71_pre.sha256 == m70_pre.sha256, 52/52):
  1 model (a266c8480cb7, live-model); 1 workflow recipe
  (m62-live-loop: train/evaluate/gate/publish, inline gate policy,
  NOT composite, own manifest 3,685 B) referenced by workflow runs
  28c7631e1152 and edcb773ef326 (8f022ff69681 is ad-hoc, recipe_id
  None); ZERO registered policies; ZERO registered probe suites
  (the families are exercised on the disposable copy, where a policy
  and a suite are registered for the chain proofs).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8796"
API = BASE + "/api/v1"

MODEL_ID = "a266c8480cb7"
RECIPE_ID = "m62-live-loop"
# the two production workflow runs carrying the recipe's provenance
# (reference_id is the bare workflow_id, no directory prefix)
RECIPE_BLOCKERS = [("workflow", "28c7631e1152"),
                   ("workflow", "edcb773ef326")]
RECIPE_FILES = ["manifest.json"]
RECIPE_BYTES = 3685
DS_ID = "5062cc7e0954"
TOK_ID = "f24b699890eb"
CKPT = "402883c5dc37"      # a certified production checkpoint id
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m71_pre.sha256"
PROD_ROOT = Path(os.environ.get("FORGE_SMOKE_ROOT",
                                "/home/user/ai-model-forge-data"))

FAMILY_ROUTES = {  # family -> resource path template (no API prefix)
    "workflow_recipe": "/workflows/recipes/{definition_id}",
    "gate_policy": "/policies/{definition_id}",
    "probe_suite": "/probe-suites/{definition_id}",
}
FAMILY_DIRS = {"workflow_recipe": "workflow-recipes",
               "gate_policy": "policies",
               "probe_suite": "probe-suites"}

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((name, bool(cond), extra))
    print(f"{'PASS' if cond else 'FAIL'} {name} {extra}")


def jget(path: str):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        assert r.status == 200, (path, r.status)
        return json.loads(r.read()), r.read()


def jpost(path: str, body: dict):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


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


def pairs(blockers: list) -> list:
    """(category, reference_id) pairs — dict blockers (API) and
    pydantic blockers (engine) alike."""
    return [(b["category"] if isinstance(b, dict) else b.category,
             b["reference_id"] if isinstance(b, dict) else b.reference_id)
            for b in blockers]


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
check("P2 disk == m71_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

models, _ = jget("/models")
check("P3 model registry", [m["id"] for m in models] == [MODEL_ID])

recipes, _ = jget("/workflows/recipes")
policies, _ = jget("/policies")
suites, _ = jget("/probe-suites")
check("P4 definition registries: 1 recipe / 0 policies / 0 suites",
      [r["recipe_id"] for r in recipes] == [RECIPE_ID]
      and policies == [] and suites == [])

# ============== the M71 retention view + the headline invariant ============ #
v, raw1 = jget(f"/workflows/recipes/{RECIPE_ID}/retention")
check("V1 recipe retention: identity + integrity + exact shape",
      set(v) == {"family", "definition_id", "created_at", "model_ids",
                 "files", "size_bytes", "integrity_verified",
                 "deletable", "blockers"}
      and v["family"] == "workflow_recipe"
      and v["definition_id"] == RECIPE_ID and v["created_at"]
      and v["integrity_verified"] is True
      and v["files"] == RECIPE_FILES and v["size_bytes"] == RECIPE_BYTES
      and v["model_ids"] == [MODEL_ID])
check("V2 HEADLINE blockers == the certified raw-manifest dependents "
      "(2 workflow runs)", pairs(v["blockers"]) == RECIPE_BLOCKERS)
check("V3 HEADLINE deletable == blockers empty", v["deletable"] is False)
_, raw2 = jget(f"/workflows/recipes/{RECIPE_ID}/retention")
check("V4 deterministic repeat (byte-identical)", raw1 == raw2)

# the §7 coupling, view-level: the recipe IS the model's ONE M66
# external definition reference
usage, _ = jget(f"/models/{MODEL_ID}/usage")
cats = {c["category"]: c["references"] for c in usage["categories"]}
check("V5 M66 coupling: workflow_recipe == [m62-live-loop], policy == []",
      cats.get("workflow_recipe") == [RECIPE_ID]
      and cats.get("policy") == [])
m67_pre, _ = jget(f"/models/{MODEL_ID}/retention")
check("V6 M67 model retention blocked by exactly the recipe",
      m67_pre["deletable"] is False
      and pairs(m67_pre["blockers"]) == [("workflow_recipe", RECIPE_ID)])

# ============ the REFUSED deletion (guaranteed 409) ======================== #
status, body = jdelete(f"/workflows/recipes/{RECIPE_ID}")
detail = body.get("detail", {})
check("D1 production recipe DELETE -> guaranteed 409 (2 dependents)",
      status == 409, f"status={status}")
check("D2 structured 409 shape (message/family/definition_id/"
      "protected/blockers)",
      set(detail) == {"message", "family", "definition_id", "protected",
                      "blockers"}
      and detail["family"] == "workflow_recipe"
      and detail["definition_id"] == RECIPE_ID
      and detail["protected"] is True
      and isinstance(detail["message"], str) and detail["message"])
check("D3 409 blockers == retention blockers (view == guard)",
      pairs(detail["blockers"]) == pairs(v["blockers"]) == RECIPE_BLOCKERS)
check("D4 zero mutation after the refusal", inventory(root) == pre_disk)

# ============================ scope 404s (read-only) ======================= #
ok404 = True
for family, tpl in FAMILY_ROUTES.items():
    for suffix in ("", "/retention"):
        try:
            urllib.request.urlopen(
                API + tpl.format(definition_id="no-such-def") + suffix,
                timeout=30)
            ok404 = False  # must not 200
        except urllib.error.HTTPError as e:
            ok404 = ok404 and e.code == 404
    if jdelete(tpl.format(definition_id="no-such-def"))[0] != 404:
        ok404 = False
check("S1 unknown recipe / policy / suite -> 404 (retention GET + "
      "DELETE, all three families)", ok404)

# ============================ OpenAPI (read-only) ========================== #
with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as r:
    spec = json.loads(r.read())
del_paths = sorted(p for p, ops in spec["paths"].items()
                   if "delete" in ops)
# the REAL OpenAPI path parameter names per family
OPENAPI_PATHS = {
    "workflow_recipe":
        "/api/v1/workflows/recipes/{recipe_id}",
    "gate_policy": "/api/v1/policies/{policy_id}",
    "probe_suite": "/api/v1/probe-suites/{suite_id}",
}
ok_oa = len(spec["paths"]) == 104 and len(del_paths) == 14
for family, p in OPENAPI_PATHS.items():
    ok_oa = ok_oa and set(spec["paths"][p].keys()) == {"get", "delete"}
    ok_oa = ok_oa and set(spec["paths"][p + "/retention"].keys()) == {"get"}
    ok_oa = ok_oa and spec["paths"][p]["delete"]["responses"]["409"][
        "content"]["application/json"]["schema"].get(
            "$ref") == "#/components/schemas/DefinitionDeletionBlocked"
ok_oa = ok_oa and all(
    s in spec["components"]["schemas"] for s in (
        "DefinitionDeletionBlocker", "DefinitionDeletionResult",
        "DefinitionDeletionBlocked", "DefinitionRetentionOverview"))
check("O1 OpenAPI: 104 paths / 14 deletes / 3 GET-only retention "
      "routes / 3 new DELETE ops / 4 schemas / 409 $ref", ok_oa)

check("A1 production byte-identical after all checks + the refused "
      "DELETE", inventory(root) == pre_disk
      and not list((root / "tmp").iterdir()), f"{len(pre_disk)} files")

# =========== disposable copy: integrity, chains, §7, §8, success =========== #
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.engine import ModelForge  # noqa: E402
from app.schemas import (  # noqa: E402
    ComparisonState,
    EvalStateKind,
    GatePolicy,
    GateRequest,
    PolicyCreateRequest,
    ProbeSuiteCreateRequest,
    SuiteProbe,
    SuiteRunRequest,
)

copy_root = Path(tempfile.mkdtemp(prefix="m71-copy-"))
try:
    shutil.copytree(PROD_ROOT, copy_root, dirs_exist_ok=True)
    forge = ModelForge(root=copy_root)

    # C1: the copy reproduces the production retention view exactly
    cv = forge.definition_retention_overview(
        "workflow_recipe", RECIPE_ID).model_dump(mode="json")
    check("C1 copy: recipe retention view == production (determinism "
          "over identical state)", cv == v)

    # C2: integrity-first refusal — tamper the recipe's stage seed,
    # the config_hash no longer reproduces; DELETE must refuse
    rdir = copy_root / "workflow-recipes" / RECIPE_ID
    original = (rdir / "manifest.json").read_bytes()
    tampered = json.loads(original)
    tampered["stages"][1]["evaluation"]["config"]["seed"] = 99
    (rdir / "manifest.json").write_text(
        json.dumps(tampered, indent=2, sort_keys=True))
    tv = forge.definition_retention_overview(
        "workflow_recipe", RECIPE_ID)
    ok_c2 = (tv.integrity_verified is False and tv.deletable is False)
    try:
        forge.delete_definition("workflow_recipe", RECIPE_ID)
        ok_c2 = False  # must refuse
    except RuntimeError as exc:
        ok_c2 = ok_c2 and "integrity" in str(exc)
    check("C2 integrity-first: tampered recipe never deletable "
          "(RuntimeError, 409 at the API)", ok_c2)
    (rdir / "manifest.json").write_bytes(original)
    check("C2b restore: integrity passes again",
          forge.definition_retention_overview(
              "workflow_recipe", RECIPE_ID).integrity_verified is True)

    # C3: §8 recipe chain — delete the two referencing workflow runs
    # (M70), the recipe becomes deletable, DELETE succeeds atomically
    for wid in ("28c7631e1152", "edcb773ef326"):
        forge.delete_model_record(MODEL_ID, "workflow", wid)
    rv = forge.definition_retention_overview(
        "workflow_recipe", RECIPE_ID)
    ok_c3 = rv.deletable is True and rv.blockers == []
    res = forge.delete_definition("workflow_recipe", RECIPE_ID)
    ok_c3 = (res.family == "workflow_recipe"
             and res.definition_id == RECIPE_ID
             and res.files_removed == len(rv.files)
             and res.bytes_reclaimed == rv.size_bytes
             and not rdir.exists())
    check("C3 recipe chain: 2 M70 record deletions unblock -> atomic "
          "DELETE (1 file / 3,685 B)", ok_c3,
          f"{res.files_removed}f/{res.bytes_reclaimed}B")
    try:
        forge.definition_retention_overview("workflow_recipe", RECIPE_ID)
        ok404_after = False
    except FileNotFoundError:
        ok404_after = True
    check("C3b recipe unknown after deletion (scope -> 404)",
          ok404_after)

    # C4: §7 live — EXACTLY the M66 workflow_recipe reference vanished
    # (no other category changed) and the LAST external reference
    # flipped the UNTOUCHED M67 model retention state
    usage_after = forge.model_usage_overview(MODEL_ID)
    cats_after = {c.category: c.references
                  for c in usage_after.categories}
    ok_c4 = cats_after["workflow_recipe"] == [] \
        and cats_after["policy"] == []
    for cat in cats:
        if cat in ("workflow_recipe", "policy"):
            continue
        # the two deleted workflow runs left the INTERNAL workflow
        # category (the test's own M70 deletions); everything else
        # must be untouched
        if cat == "workflow":
            ok_c4 = ok_c4 and sorted(
                set(cats[cat]) - {"28c7631e1152", "edcb773ef326"}) == \
                sorted(cats_after[cat])
        else:
            ok_c4 = ok_c4 and cats[cat] == cats_after[cat]
    m67_post = forge.model_retention_overview(MODEL_ID)
    ok_c4 = ok_c4 and m67_post.deletable is True \
        and m67_post.blockers == []
    check("C4 §7 live: M66 external ref gone (nothing else vanished); "
          "M67 flips to deletable with ZERO M67 code changes", ok_c4)

    # C5: §8 policy chain — register a policy (production has none),
    # run a gate with it, the decision blocks; delete the decision
    # (M70), the policy deletes atomically; the M66 policy category
    # grows and shrinks live
    pol = forge.register_policy(PolicyCreateRequest(
        policy_id="m71-smoke-policy",
        description="M71 smoke policy",
        policy=GatePolicy(
            name="m71-smoke-policy", model_id=MODEL_ID, dataset_id=DS_ID,
            tokenizer_id=TOK_ID, split="validation", batch_size=8,
            max_seq_len=32, seed=11, baseline_type="checkpoint",
            baseline_checkpoint_id=CKPT, tolerance=1.0)))
    gate = forge.run_gate(GateRequest(
        model_id=MODEL_ID, policy_id="m71-smoke-policy",
        candidate=ComparisonState(state_kind=EvalStateKind.CHECKPOINT,
                                  checkpoint_id=CKPT)))
    usage_pol = forge.model_usage_overview(MODEL_ID)
    cats_pol = {c.category: c.references for c in usage_pol.categories}
    ok_c5 = cats_pol["policy"] == ["m71-smoke-policy"]
    blockers = forge.definition_deletion_blockers(
        "gate_policy", "m71-smoke-policy")
    ok_c5 = ok_c5 and pairs(blockers) == [("gate", gate.decision_id)]
    try:
        forge.delete_definition("gate_policy", "m71-smoke-policy")
        ok_c5 = False  # must refuse
    except ValueError:
        pass
    forge.delete_model_record(MODEL_ID, "gate", gate.decision_id)
    pv = forge.definition_retention_overview(
        "gate_policy", "m71-smoke-policy")
    res = forge.delete_definition("gate_policy", "m71-smoke-policy")
    ok_c5 = ok_c5 and pv.deletable is True and res.files_removed >= 1 \
        and not (copy_root / "policies" / "m71-smoke-policy").exists()
    cats_pol2 = {c.category: c.references
                 for c in forge.model_usage_overview(
                     MODEL_ID).categories}
    ok_c5 = ok_c5 and cats_pol2["policy"] == []
    check("C5 §8 policy chain: gate decision blocks -> M70 deletion "
          "unblocks -> atomic DELETE; M66 policy category grows/"
          "shrinks live", ok_c5)

    # C6: §8 suite chain — register a suite (production has none), run
    # it (cross-checked: the run blocks), delete the suite run (M68),
    # the suite deletes atomically; the dataset blocker surface is
    # unchanged BY THE SUITE DELETION (suites are not M64 categories)
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m71-smoke-suite",
        description="M71 smoke suite",
        probes=[SuiteProbe(dataset_id=DS_ID, split="validation",
                           tokenizer_id=TOK_ID, batch_size=8,
                           max_seq_len=32, seed=42)]))
    srun = forge.run_suite(SuiteRunRequest(
        model_id=MODEL_ID, suite_id="m71-smoke-suite",
        state=ComparisonState(state_kind=EvalStateKind.CURRENT)))
    blockers = forge.definition_deletion_blockers(
        "probe_suite", "m71-smoke-suite")
    ok_c6 = pairs(blockers) == [("suite_run", srun.suite_run_id)]
    sv = forge.definition_retention_overview(
        "probe_suite", "m71-smoke-suite")
    ok_c6 = ok_c6 and sv.model_ids == [] and sv.deletable is False
    try:
        forge.delete_definition("probe_suite", "m71-smoke-suite")
        ok_c6 = False  # must refuse
    except ValueError:
        pass
    ds_pre = [(b.reason, b.detail) for b in
              forge.dataset_deletion_blockers(DS_ID)]
    forge.delete_suite_run(MODEL_ID, srun.suite_run_id)
    ds_mid = [(b.reason, b.detail) for b in
              forge.dataset_deletion_blockers(DS_ID)]
    sv = forge.definition_retention_overview(
        "probe_suite", "m71-smoke-suite")
    ok_c6 = ok_c6 and sv.deletable is True
    res = forge.delete_definition("probe_suite", "m71-smoke-suite")
    ok_c6 = ok_c6 and res.files_removed == len(sv.files) \
        and res.bytes_reclaimed == sv.size_bytes \
        and not (copy_root / "probe-suites" / "m71-smoke-suite").exists()
    ds_post = [(b.reason, b.detail) for b in
               forge.dataset_deletion_blockers(DS_ID)]
    ok_c6 = ok_c6 and ds_post == ds_mid
    check("C6 §8 suite chain: suite run blocks -> M68 deletion "
          "unblocks -> atomic DELETE; NO dataset blocker change from "
          "the suite deletion", ok_c6)

    # C7: the structural directions — a recipe whose gate stage names
    # a policy / whose suite-run stage names a suite blocks that
    # definition even with NO evidence records
    forge.register_policy(PolicyCreateRequest(
        policy_id="m71-struct-policy",
        policy=GatePolicy(
            name="m71-struct-policy", model_id=MODEL_ID,
            dataset_id=DS_ID, tokenizer_id=TOK_ID, split="validation",
            batch_size=8, max_seq_len=32, seed=12,
            baseline_type="checkpoint", baseline_checkpoint_id=CKPT,
            tolerance=1.0)))
    from app.schemas import (  # noqa: E402
        StageStateRef,
        StageType,
        WorkflowGateStage,
        WorkflowRecipeCreateRequest,
        WorkflowStage,
        WorkflowSuiteRunStage,
    )
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-struct-recipe",
        stages=[WorkflowStage(
            stage_id="gt", type=StageType.GATE,
            gate=WorkflowGateStage(
                policy_id="m71-struct-policy",
                candidate=StageStateRef(
                    state_kind=EvalStateKind.CHECKPOINT,
                    checkpoint_id=CKPT)))]))
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m71-struct-suite",
        probes=[SuiteProbe(dataset_id=DS_ID, split="validation",
                           tokenizer_id=TOK_ID, batch_size=8,
                           max_seq_len=32, seed=44)]))
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-struct-recipe-2",
        stages=[WorkflowStage(
            stage_id="sr", type=StageType.SUITE_RUN,
            suite_run=WorkflowSuiteRunStage(
                suite_id="m71-struct-suite",
                state=StageStateRef(
                    state_kind=EvalStateKind.CHECKPOINT,
                    checkpoint_id=CKPT)))]))
    ok_c7 = True
    for family, did, expect in (
            ("gate_policy", "m71-struct-policy",
             [("workflow_recipe", "m71-struct-recipe")]),
            ("probe_suite", "m71-struct-suite",
             [("workflow_recipe", "m71-struct-recipe-2")])):
        blockers = pairs(forge.definition_deletion_blockers(family, did))
        ok_c7 = ok_c7 and blockers == expect
        try:
            forge.delete_definition(family, did)
            ok_c7 = False  # must refuse
        except ValueError:
            pass
    # deleting the referencing recipes unblocks both definitions
    forge.delete_definition("workflow_recipe", "m71-struct-recipe")
    forge.delete_definition("workflow_recipe", "m71-struct-recipe-2")
    ok_c7 = ok_c7 and forge.definition_retention_overview(
        "gate_policy", "m71-struct-policy").deletable is True
    ok_c7 = ok_c7 and forge.definition_retention_overview(
        "probe_suite", "m71-struct-suite").deletable is True
    forge.delete_definition("gate_policy", "m71-struct-policy")
    forge.delete_definition("probe_suite", "m71-struct-suite")
    # the families' directories end EMPTY (the registry may keep the
    # empty parent dir; no definition file may remain)
    for fam_dir in ("policies", "probe-suites"):
        d = copy_root / fam_dir
        ok_c7 = ok_c7 and (not d.exists()
                           or not any(d.rglob("*")))
    check("C7 structural directions: recipes' gate/suite-run stages "
          "block policies/suites; recipe deletion unblocks; the "
          "families' directories end EMPTY", ok_c7)

    # C8: the composite direction — a composite recipe blocks its base
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-comp-base",
        stages=[WorkflowStage(
            stage_id="sr", type=StageType.SUITE_RUN,
            suite_run=WorkflowSuiteRunStage(
                suite_id="m71-comp-suite",
                state=StageStateRef(
                    state_kind=EvalStateKind.CHECKPOINT,
                    checkpoint_id=CKPT)))]))
    forge.register_probe_suite(ProbeSuiteCreateRequest(
        suite_id="m71-comp-suite",
        probes=[SuiteProbe(dataset_id=DS_ID, split="validation",
                           tokenizer_id=TOK_ID, batch_size=8,
                           max_seq_len=32, seed=45)]))
    forge.register_workflow_recipe(WorkflowRecipeCreateRequest(
        recipe_id="m71-comp",
        stages=[WorkflowStage(
            stage_id="call", type=StageType.RECIPE,
            recipe=__import__("app.schemas", fromlist=[
                "WorkflowRecipeCallStage"]).WorkflowRecipeCallStage(
                recipe_id="m71-comp-base"))]))
    ok_c8 = pairs(forge.definition_deletion_blockers(
        "workflow_recipe", "m71-comp-base")) == [
        ("workflow_recipe", "m71-comp")]
    try:
        forge.delete_definition("workflow_recipe", "m71-comp-base")
        ok_c8 = False
    except ValueError:
        pass
    forge.delete_definition("workflow_recipe", "m71-comp")
    ok_c8 = ok_c8 and forge.definition_retention_overview(
        "workflow_recipe", "m71-comp-base").deletable is True
    forge.delete_definition("workflow_recipe", "m71-comp-base")
    forge.delete_definition("probe_suite", "m71-comp-suite")
    check("C8 composite direction: composition references block the "
          "base recipe; composite deletion unblocks", ok_c8)

    # C9: no residue — only the intended files were removed from the
    # copy (the 2 workflow records + the recipe) and the definition
    # families' directories are gone/empty; the model/dataset/
    # tokenizer dirs still exist untouched-by-deletion
    cpost = inventory(copy_root)
    expected_gone = {
        "workflow-recipes/m62-live-loop/manifest.json",
        "models/a266c8480cb7/workflows/workflow-28c7631e1152/"
        "manifest.json",
        "models/a266c8480cb7/workflows/workflow-edcb773ef326/"
        "manifest.json",
    }
    gone = set(pre_disk) - set(cpost)
    added = set(cpost) - set(pre_disk)
    # added: the M66/M67 usage is computed live (no storage); the gate
    # + suite run + their evaluation records added files; the recipe/
    # policy/suite definitions were all deleted again
    check("C9 copy delta: exactly the intended removals (recipe + 2 "
          "workflow records) + the exercise artifacts (gate/suite "
          "run/evals, all later deleted); definitions dirs empty",
          expected_gone <= gone
          and not any(p.startswith(("workflow-recipes/", "policies/",
                                    "probe-suites/")) for p in cpost)
          and all(p.startswith("models/") or p.startswith("tmp/")
                  for p in added), f"gone={len(gone)} added={len(added)}")
finally:
    shutil.rmtree(copy_root, ignore_errors=True)

# ============================== summary ==================================== #
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print(f"\nM71 LIVE SMOKE: {n_pass}/{len(results)} checks passed")
for name, ok, extra in results:
    if not ok:
        print(f"  FAILED: {name} {extra}")
sys.exit(1 if n_fail else 0)
