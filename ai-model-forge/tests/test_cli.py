"""Milestone 74 tests: the READ-ONLY retention & impact CLI.

The CLI is a THIN ADAPTER — every test proves it: the engine/
facade is called DIRECTLY for the expected result, the CLI runs as
a real subprocess (exit codes, stdout/stderr), and the CLI's output
is compared against the direct engine result (the independent
output oracle — never the CLI's own helpers against themselves).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# reuse the M73 multi-family fixture builder (no second fixture)
from test_deletion_impact import _build

REPO = Path(__file__).resolve().parent.parent
FAMILIES_14 = ("model", "dataset", "tokenizer", "workflow_recipe",
               "gate_policy", "probe_suite", "checkpoint", "workflow",
               "evaluation", "comparison", "gate", "suite_run",
               "sample", "sample_quality")
MODEL_SCOPED_8 = ("checkpoint", "workflow", "evaluation", "comparison",
                  "gate", "suite_run", "sample", "sample_quality")


def run_cli(args, root):
    """Run the CLI as a REAL subprocess against the given root."""
    env = {**os.environ, "FORGE_ROOT": str(root)}
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *args],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=120)


def _inventory(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] != "tmp"
    }


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("m74-env"))


@pytest.fixture(scope="module")
def ids(env):
    """One artifact id per family + the model ids, discovered via the
    engine listings (the same discovery the M73 smoke used)."""
    model_a = env.model
    model_d = env.model_d
    gate = env.gw
    comp = env.cg
    return {
        "model": (model_a, None),
        "dataset": (env.ds, None),
        "tokenizer": (env.tok, None),
        "workflow_recipe": ("m73-recipe", None),
        "gate_policy": (env.policy, None),
        "probe_suite": ("m73-suite", None),
        "checkpoint": (env.ck[0], model_a),
        "workflow": (env.w2.workflow_id, model_a),
        "evaluation": (env.direct_eval.eval_id, model_a),
        "comparison": (comp, model_a),
        "gate": (gate, model_a),
        "suite_run": (env.sr.suite_run_id, model_a),
        "sample": (env.sample.sample_id, model_a),
        "sample_quality": (env.quality.evaluation_id, model_a),
        "model_d": model_d,
    }


def _engine_retention_view(forge, family, artifact_id, model_id):
    """The DIRECT engine result for one retention command (the
    oracle) — the exact facade methods the CLI routes to."""
    if family == "model":
        return forge.model_retention_overview(artifact_id)
    if family == "dataset":
        return forge.dataset_retention_overview(artifact_id)
    if family == "tokenizer":
        return forge.tokenizer_retention_overview(artifact_id)
    if family in ("workflow_recipe", "gate_policy", "probe_suite"):
        return forge.definition_retention_overview(family, artifact_id)
    if family == "checkpoint":
        ov = forge.checkpoint_retention_overview(model_id)
        return next(e for e in ov.checkpoints
                    if e.checkpoint_id == artifact_id)
    if family in ("workflow", "evaluation", "comparison", "gate"):
        return forge.model_record_retention_overview(
            model_id, family, artifact_id)
    if family == "suite_run":
        return forge.suite_run_retention_overview(model_id, artifact_id)
    if family == "sample":
        return forge.sample_retention_overview(model_id, artifact_id)
    return forge.sample_evaluation_retention_overview(
        model_id, artifact_id)


# --------------------------------------------------------------------------- #
# A + B + J: help, project retention, JSON oracle
# --------------------------------------------------------------------------- #

def test_m74_help(env):
    for args in (["--help"], ["retention", "--help"],
                 ["impact", "--help"]):
        r = run_cli(args, env.root)
        assert r.returncode == 0, (args, r.stderr)
        assert r.stdout.strip()


def test_m74_project_retention_and_json(env):
    forge = env.forge
    expected = forge.project_retention_overview().model_dump(
        mode="json")
    r = run_cli(["retention", "project"], env.root)
    assert r.returncode == 0, r.stderr
    # human output carries the deterministic headline numbers
    human = r.stdout
    assert f"reclaimable_bytes: {expected['reclaimable_bytes']}" in human
    assert f"total_count: {expected['total_count']}" in human
    assert f"total_files: {expected['total_files']}" in human
    rj = run_cli(["retention", "project", "--json"], env.root)
    assert rj.returncode == 0, rj.stderr
    assert json.loads(rj.stdout) == expected  # the independent oracle


# --------------------------------------------------------------------------- #
# C + D + E: every family, retention AND impact, engine-vs-CLI oracle
# --------------------------------------------------------------------------- #

def test_m74_retention_every_family(env, ids):
    forge = env.forge
    for family in FAMILIES_14:
        artifact_id, model_id = ids[family]
        args = ["retention", family, artifact_id]
        if model_id is not None:
            args += ["--model-id", model_id]
        expected = _engine_retention_view(
            forge, family, artifact_id, model_id).model_dump(mode="json")
        rj = run_cli([*args, "--json"], env.root)
        assert rj.returncode == 0, (family, rj.stderr)
        assert json.loads(rj.stdout) == expected, family
        rh = run_cli(args, env.root)
        assert rh.returncode == 0, (family, rh.stderr)
        # human output carries the family's deterministic identity
        assert f"deletable: {str(expected['deletable']).lower()}" \
            in rh.stdout, family
        assert f"size_bytes: {expected['size_bytes']}" in rh.stdout, \
            family


def test_m74_impact_every_family(env, ids):
    forge = env.forge
    for family in FAMILIES_14:
        artifact_id, model_id = ids[family]
        args = ["impact", family, artifact_id]
        if model_id is not None:
            args += ["--model-id", model_id]
        expected = forge.deletion_impact_preview(
            family, artifact_id, model_id).model_dump(mode="json")
        rj = run_cli([*args, "--json"], env.root)
        assert rj.returncode == 0, (family, rj.stderr)
        assert json.loads(rj.stdout) == expected, family
        rh = run_cli(args, env.root)
        assert rh.returncode == 0, (family, rh.stderr)
        assert f"executable: {str(expected['executable']).lower()}" \
            in rh.stdout, family
        assert (f"project_reclaimable_delta: "
                f"{expected['project_reclaimable_delta']}") in rh.stdout


def test_m74_model_scoped_positions(env, ids):
    """The M73 route-parameter bug class: --model-id must land in the
    MODEL position, never the artifact position. Swapped arguments
    must FAIL, not succeed."""
    # a MODEL id in the artifact position with the eval's model as
    # --model-id -> unknown evaluation (not the model's view)
    r = run_cli(["retention", "evaluation", ids["model"][0],
                 "--model-id", ids["evaluation"][1]], env.root)
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    # a checkpoint id from model A asked under model D -> not found
    r = run_cli(["retention", "checkpoint", ids["checkpoint"][0],
                 "--model-id", ids["model_d"]], env.root)
    assert r.returncode == 3
    r = run_cli(["impact", "checkpoint", ids["checkpoint"][0],
                 "--model-id", ids["model_d"]], env.root)
    assert r.returncode == 3
    # the CORRECT invocation still succeeds (positions verified)
    r = run_cli(["impact", "checkpoint", ids["checkpoint"][0],
                 "--model-id", ids["checkpoint"][1]], env.root)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# F + G: unknown ids, training_run, invalid families, usage errors
# --------------------------------------------------------------------------- #

def test_m74_unknown_ids(env):
    for command in ("retention", "impact"):
        r = run_cli([command, "model", "no-such-model"], env.root)
        assert r.returncode == 3, (command, r.stderr)
        assert "no-such-model" in r.stderr, command
        r = run_cli([command, "dataset", "no-such-ds"], env.root)
        assert r.returncode == 3, (command, r.stderr)
        assert "no-such-ds" in r.stderr, command
        r = run_cli([command, "workflow", "no-such-wf",
                     "--model-id", env.model], env.root)
        assert r.returncode == 3, (command, r.stderr)
        assert "no-such-wf" in r.stderr, command
        r = run_cli([command, "gate_policy", "no-such-pol"], env.root)
        assert r.returncode == 3, (command, r.stderr)
        assert r.stdout == ""  # no fabricated result


def test_m74_training_run_lifecycle(env):
    for command in ("retention", "impact"):
        r = run_cli([command, "training_run", "whatever"], env.root)
        assert r.returncode == 5, (command, r.stderr)
        assert "training_run" in r.stderr
        assert "no deletion lifecycle" in r.stderr, command
        assert r.stdout == ""  # no fabricated retention/impact result


def test_m74_invalid_family_and_usage(env):
    for command in ("retention", "impact"):
        r = run_cli([command, "bogus_family", "x"], env.root)
        assert r.returncode == 4, (command, r.stderr)
        assert "unknown family 'bogus_family'" in r.stderr
    # model-scoped family without --model-id -> usage error
    r = run_cli(["retention", "evaluation", "x"], env.root)
    assert r.returncode == 2 and "--model-id is required" in r.stderr
    r = run_cli(["impact", "workflow", "x"], env.root)
    assert r.returncode == 2 and "--model-id is required" in r.stderr
    # --model-id on a non-model-scoped family -> usage error
    r = run_cli(["retention", "dataset", env.ds,
                 "--model-id", env.model], env.root)
    assert r.returncode == 2 and "not model-scoped" in r.stderr
    # 'project' takes no artifact id
    r = run_cli(["retention", "project", env.ds], env.root)
    assert r.returncode == 2 and "takes no artifact id" in r.stderr
    # no family at all
    r = run_cli(["retention"], env.root)
    assert r.returncode == 2


# --------------------------------------------------------------------------- #
# H: determinism (stdout AND stderr AND exit, twice)
# --------------------------------------------------------------------------- #

def test_m74_determinism(env, ids):
    probes = [
        ["retention", "project"],
        ["retention", "project", "--json"],
        ["retention", "model", ids["model"][0]],
        ["retention", "checkpoint", ids["checkpoint"][0],
         "--model-id", ids["checkpoint"][1], "--json"],
        ["impact", "workflow", ids["workflow"][0],
         "--model-id", ids["workflow"][1]],
        ["impact", "model", ids["model"][0], "--json"],
        ["impact", "training_run", "x"],
        ["retention", "dataset", "no-such"],
    ]
    for args in probes:
        one = run_cli(args, env.root)
        two = run_cli(args, env.root)
        assert one.returncode == two.returncode, args
        assert one.stdout == two.stdout, args
        assert one.stderr == two.stderr, args


# --------------------------------------------------------------------------- #
# I: no mutation (the module fixture ran every other command first)
# --------------------------------------------------------------------------- #

def test_m74_no_mutation(env):
    """The fixture root's inventory is compared against the manifest
    captured BEFORE any CLI command ran (the module-scoped autouse
    _before fixture — it instantiates before the first test in this
    module, so every CLI subprocess ran between the two snapshots)."""
    assert _inventory(env.root) == env.inventory_before
    tmp = env.root / "tmp"
    assert not (tmp.exists() and any(tmp.iterdir()))


@pytest.fixture(scope="module", autouse=True)
def _before(env):
    """Inventory of the fixture root BEFORE any CLI command runs."""
    env.inventory_before = _inventory(env.root)
    return env.inventory_before


# --------------------------------------------------------------------------- #
# Registry no-drift guard + human-field oracle for a blocked artifact
# --------------------------------------------------------------------------- #

def test_m74_cli_table_matches_engine_registry(env):
    from app import cli
    forge = env.forge
    assert set(cli._RETENTION_DISPATCH) == set(forge.IMPACT_FAMILIES)
    assert len(cli.RETENTION_FAMILIES) == 14
    assert set(cli.MODEL_SCOPED_FAMILIES) == set(MODEL_SCOPED_8)
    assert "training_run" in forge.PROJECT_RETENTION_FAMILIES


def test_m74_blocked_artifact_is_not_a_failure(env, ids):
    """A blocked artifact's retention view is a SUCCESSFUL command
    describing the blocked state (never flattened into an error)."""
    forge = env.forge
    # the sample is blocked by its quality measurement (M68)
    view = forge.sample_retention_overview(
        ids["sample"][1], ids["sample"][0])
    assert view.deletable is False and view.blockers
    r = run_cli(["retention", "sample", ids["sample"][0],
                 "--model-id", ids["sample"][1]], env.root)
    assert r.returncode == 0, r.stderr
    assert "deletable: false" in r.stdout
    assert "blockers:" in r.stdout
    rj = run_cli(["retention", "sample", ids["sample"][0],
                  "--model-id", ids["sample"][1], "--json"], env.root)
    assert rj.returncode == 0
    assert json.loads(rj.stdout)["deletable"] is False


# =========================================================================== #
# M75: read-only usage & storage commands
# =========================================================================== #

USAGE_KINDS = ("dataset", "tokenizer", "model", "records", "definition")
DEFINITION_FAMILIES = ("workflow_recipe", "gate_policy", "probe_suite")


def test_m75_help(env):
    for args in (
            ["--help"], ["storage", "--help"], ["usage", "--help"],
            ["usage", "dataset", "--help"],
            ["usage", "tokenizer", "--help"],
            ["usage", "model", "--help"], ["usage", "records", "--help"],
            ["usage", "definition", "--help"]):
        r = run_cli(args, env.root)
        assert r.returncode == 0, (args, r.stderr)
        assert r.stdout.strip(), args


def test_m75_malformed_invocations(env):
    # missing required positionals / subcommands -> argparse exit 2
    for args in (["usage"], ["usage", "dataset"], ["usage", "model"],
                 ["usage", "definition"], ["storage", "unexpected-arg"],
                 ["usage", "bogus", "some-id"]):
        r = run_cli(args, env.root)
        assert r.returncode == 2, (args, r.returncode, r.stdout)
        assert r.stdout == "", args
    # invalid family routed INTO the definition command -> exit 4
    r = run_cli(["usage", "definition", "bogus-family", "x"], env.root)
    assert r.returncode == 4, r.stderr
    assert "unknown family 'bogus-family'" in r.stderr
    assert r.stdout == ""
    # a lifecycle-less family -> the M74 exit-5 semantics
    r = run_cli(["usage", "definition", "training_run", "x"], env.root)
    assert r.returncode == 5, r.stderr
    assert "training_run" in r.stderr and "no deletion lifecycle" \
        in r.stderr
    assert r.stdout == ""


def test_m75_storage(env):
    forge = env.forge
    expected = forge.project_storage_overview().model_dump(mode="json")
    rj = run_cli(["storage", "--json"], env.root)
    assert rj.returncode == 0, rj.stderr
    assert json.loads(rj.stdout) == expected  # the independent oracle
    rh = run_cli(["storage"], env.root)
    assert rh.returncode == 0, rh.stderr
    assert f"total_files: {expected['total_files']}" in rh.stdout
    assert f"total_bytes: {expected['total_bytes']}" in rh.stdout
    assert f"model_count: {expected['model_count']}" in rh.stdout
    # deterministic repeat
    assert run_cli(["storage"], env.root).stdout == rh.stdout
    assert run_cli(["storage", "--json"], env.root).stdout == rj.stdout


def test_m75_usage_dataset_and_tokenizer(env):
    forge = env.forge
    for kind, artifact_id, id_field in (
            ("dataset", env.ds, "dataset_id"),
            ("tokenizer", env.tok, "tokenizer_id")):
        method = (forge.dataset_usage_overview
                  if kind == "dataset" else forge.tokenizer_usage_overview)
        expected = method(artifact_id).model_dump(mode="json")
        rj = run_cli(["usage", kind, artifact_id, "--json"], env.root)
        assert rj.returncode == 0, (kind, rj.stderr)
        assert json.loads(rj.stdout) == expected, kind
        rh = run_cli(["usage", kind, artifact_id], env.root)
        assert rh.returncode == 0, (kind, rh.stderr)
        assert f"{id_field}: {artifact_id}" in rh.stdout, kind
        assert f"total_references: {expected['total_references']}" \
            in rh.stdout, kind
        assert run_cli(["usage", kind, artifact_id],
                       env.root).stdout == rh.stdout, kind
        # unknown id -> exit 3, canonical error, empty stdout
        r = run_cli(["usage", kind, "no-such-id"], env.root)
        assert r.returncode == 3, (kind, r.stderr)
        assert "no-such-id" in r.stderr and r.stdout == ""


def test_m75_usage_model_and_records(env):
    forge = env.forge
    expected_model = forge.model_usage_overview(
        env.model).model_dump(mode="json")
    rj = run_cli(["usage", "model", env.model, "--json"], env.root)
    assert rj.returncode == 0, rj.stderr
    assert json.loads(rj.stdout) == expected_model
    rh = run_cli(["usage", "model", env.model], env.root)
    assert rh.returncode == 0, rh.stderr
    assert f"model_id: {env.model}" in rh.stdout
    assert f"total_references: {expected_model['total_references']}" \
        in rh.stdout
    assert run_cli(["usage", "model", env.model],
                   env.root).stdout == rh.stdout

    expected_records = forge.model_records_usage_overview(
        env.model).model_dump(mode="json")
    rj = run_cli(["usage", "records", env.model, "--json"], env.root)
    assert rj.returncode == 0, rj.stderr
    assert json.loads(rj.stdout) == expected_records
    rh = run_cli(["usage", "records", env.model], env.root)
    assert rh.returncode == 0, rh.stderr
    assert f"total_references: {expected_records['total_references']}" \
        in rh.stdout
    assert run_cli(["usage", "records", env.model],
                   env.root).stdout == rh.stdout
    # the two views are DISTINCT surfaces (no accidental routing)
    assert run_cli(["usage", "model", env.model], env.root).stdout != \
        run_cli(["usage", "records", env.model], env.root).stdout
    # unknown model -> exit 3, canonical error, empty stdout
    for kind in ("model", "records"):
        r = run_cli(["usage", kind, "no-such-model"], env.root)
        assert r.returncode == 3, (kind, r.stderr)
        assert "no-such-model" in r.stderr and r.stdout == ""


def test_m75_usage_definition_every_family(env):
    forge = env.forge
    known = {"workflow_recipe": "m73-recipe",
             "gate_policy": env.policy,
             "probe_suite": "m73-suite"}
    assert set(known) == set(forge.DEFINITION_DELETABLE_FAMILIES)
    for family, definition_id in known.items():
        expected = forge.definition_retention_overview(
            family, definition_id).model_dump(mode="json")
        rj = run_cli(["usage", "definition", family, definition_id,
                      "--json"], env.root)
        assert rj.returncode == 0, (family, rj.stderr)
        assert json.loads(rj.stdout) == expected, family
        rh = run_cli(["usage", "definition", family, definition_id],
                     env.root)
        assert rh.returncode == 0, (family, rh.stderr)
        assert f"family: {family}" in rh.stdout, family
        assert f"deletable: {str(expected['deletable']).lower()}" \
            in rh.stdout, family
        assert f"size_bytes: {expected['size_bytes']}" in rh.stdout, \
            family
        assert run_cli(["usage", "definition", family, definition_id],
                       env.root).stdout == rh.stdout, family
        # unknown definition -> exit 3, canonical error, empty stdout
        r = run_cli(["usage", "definition", family, "no-such-def"],
                    env.root)
        assert r.returncode == 3, (family, r.stderr)
        assert "no-such-def" in r.stderr and r.stdout == "", family


def test_m75_swapped_id_positions(env, ids):
    """The M74 argument-position discipline, M75 edition: an id
    routed into the WRONG command must FAIL (exit 3), never
    succeed by accident."""
    model_a = ids["model"][0]
    ds = ids["dataset"][0]
    # a dataset id is not a model / record-owner / definition id
    for args in (["usage", "model", ds],
                 ["usage", "records", ds],
                 ["usage", "definition", "gate_policy", model_a],
                 ["usage", "dataset", model_a],
                 ["usage", "tokenizer", model_a]):
        r = run_cli(args, env.root)
        assert r.returncode == 3, (args, r.returncode, r.stdout)
        assert r.stdout == "", args


def test_m75_registry_no_drift(env):
    """The CLI definition surface must track the engine's OWN
    registry — a missing or extra family is caught here, before any
    engine call."""
    forge = env.forge
    assert set(forge.DEFINITION_DELETABLE_FAMILIES) == \
        set(DEFINITION_FAMILIES)
    assert len(DEFINITION_FAMILIES) == 3
    # every engine definition family is accepted by the CLI (routed
    # to the real facade — exit 3 on a missing id, never 4)
    for family in forge.DEFINITION_DELETABLE_FAMILIES:
        r = run_cli(["usage", "definition", family, "no-such-def"],
                    env.root)
        assert r.returncode == 3, family
    # training_run stays lifecycle-less everywhere (M74 semantics)
    assert "training_run" in forge.PROJECT_RETENTION_FAMILIES
    assert "training_run" not in forge.DEFINITION_DELETABLE_FAMILIES
    assert "training_run" not in forge.IMPACT_FAMILIES
