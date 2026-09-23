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


# =========================================================================== #
# M76: read-only DASHBOARDS & HISTORY CLI
# =========================================================================== #

HISTORY_DIMENSIONS = {
    "checkpoint": {"run": "run_id"},
    "evaluation": {
        "checkpoint": "ck0", "dataset": "ds", "tokenizer": "tok",
        "split": "validation", "state_kind": "checkpoint",
        "truncated": "false", "seed": "2"},
    "comparison": {
        "checkpoint": "ck0", "dataset": "ds", "tokenizer": "tok",
        "split": "validation", "verdict": "improved",
        "state_kind": "checkpoint", "seed": "2"},
    "gate": {
        "policy": "policy_b", "comparison": "comp",
        "decision": "passed", "verdict": "improved",
        "baseline_type": "checkpoint"},
    "workflow": {"recipe": "recipe", "status": "completed"},
    "suite_run": {
        "suite": "suite", "suite_summary": "suite",
        "checkpoint": "ck0", "reused_count": "1"},
    "sample": {
        "checkpoint": "ck0", "tokenizer": "tok",
        "strategy": "greedy"},
    "sample_quality": {
        "sample": "sample_id", "checkpoint": "ck0",
        "tokenizer": "tok"},
}


def _m76_values(env):
    """One real dimension VALUE per (family, dimension) — ids from
    the fixture, enum names / ints / bools as strings."""
    ck = sorted(c.checkpoint_id for c in
                env.forge.list_checkpoints(env.model))
    return {
        "ck0": ck[0], "ds": env.ds, "tok": env.tok,
        "policy_b": env.policy, "comp": env.cg, "recipe": "m73-recipe",
        "suite": "m73-suite", "sample_id": env.sample.sample_id,
        "run_id": env.forge.list_checkpoints(env.model)[0].run_id,
    }


def test_m76_help():
    for args in (["dashboard", "--help"], ["history", "--help"]):
        r = run_cli(args, "/nonexistent-root")  # no engine needed
        assert r.returncode == 0, (args, r.stderr)
        assert r.stdout.strip()


def test_m76_dashboard(env):
    forge = env.forge
    expected = forge.get_dashboard(env.model).model_dump(mode="json")
    rj = run_cli(["dashboard", env.model, "--json"], env.root)
    assert rj.returncode == 0, rj.stderr
    assert json.loads(rj.stdout) == expected  # the adapter oracle
    rh = run_cli(["dashboard", env.model], env.root)
    assert rh.returncode == 0, rh.stderr
    assert f"model_id: {env.model}" in rh.stdout
    # unknown model -> exit 3, canonical error, empty stdout
    r = run_cli(["dashboard", "no-such-model"], env.root)
    assert r.returncode == 3 and r.stdout == "" and r.stderr.strip()


def test_m76_history_every_family_and_dimension(env):
    """The FULL matrix: every routed (family, dimension) pair plus
    best_checkpoint, each against the direct facade result."""
    from app import cli as cli_mod
    forge = env.forge
    vals = _m76_values(env)
    routed = 0
    for family, dims in HISTORY_DIMENSIONS.items():
        for dimension, vkey in dims.items():
            method, marker = cli_mod._HISTORY_DISPATCH[family][dimension]
            # the PYTHON value built independently of the CLI (the
            # oracle must not reuse the CLI's coercion helper)
            if marker == "str":
                value = vals[vkey]
            elif marker == "int":
                value = int(vkey)
            elif marker == "bool":
                value = vkey == "true"
            else:
                from app import schemas as _schemas
                value = getattr(_schemas,
                                marker.split(":", 1)[1])(vkey)
            direct = getattr(forge, method)(env.model, value)
            expected = ([r.model_dump(mode="json") for r in direct]
                        if isinstance(direct, list)
                        else direct.model_dump(mode="json"))
            args = ["history", family, dimension, str(vkey) if
                    marker != "str" else vals[vkey],
                    "--model-id", env.model]
            rj = run_cli([*args, "--json"], env.root)
            assert rj.returncode == 0, (family, dimension, rj.stderr)
            assert json.loads(rj.stdout) == expected, (family,
                                                       dimension)
            rh = run_cli(args, env.root)
            assert rh.returncode == 0 and rh.stdout.strip(), \
                (family, dimension)
            routed += 1
    assert routed == 32  # every by-X surface is covered
    # best_checkpoint (M59): no dimension
    expected = forge.best_checkpoint_history(
        env.model).model_dump(mode="json")
    rj = run_cli(["history", "best_checkpoint", "--model-id",
                  env.model, "--json"], env.root)
    assert rj.returncode == 0, rj.stderr
    assert json.loads(rj.stdout) == expected
    # a list surface prints its deterministic count line
    rh = run_cli(["history", "evaluation", "checkpoint", vals["ck0"],
                  "--model-id", env.model], env.root)
    assert rh.returncode == 0
    assert rh.stdout.startswith("count: ")


def test_m76_history_errors(env):
    m = env.model
    probes = [
        (["history", "bogus", "x", "y", "--model-id", m], 4),
        (["history", "training_run", "x", "--model-id", m], 5),
        (["history", "evaluation", "bogus_dim", "y", "--model-id", m],
         4),
        (["history", "evaluation", "checkpoint", "--model-id", m], 2),
        (["history", "evaluation", "--model-id", m], 2),
        (["history", "evaluation", "split", "bogus", "--model-id", m],
         2),
        (["history", "evaluation", "seed", "abc", "--model-id", m], 2),
        (["history", "evaluation", "truncated", "yes", "--model-id",
          m], 2),
        (["history", "gate", "decision", "bogus", "--model-id", m],
         2),
        (["history", "best_checkpoint", "x", "y", "--model-id", m],
         2),
        (["history", "evaluation", "split", "validation"], 2),
        (["history", "evaluation", "checkpoint", "no-such",
          "--model-id", m], 3),
        (["history", "evaluation", "dataset", "no-such-ds",
          "--model-id", m], 3),
        (["history", "workflow", "recipe", "no-such-recipe",
          "--model-id", m], 3),
        (["history", "suite_run", "suite", "no-such-suite",
          "--model-id", m], 3),
        (["history", "sample_quality", "sample", "no-such-sample",
          "--model-id", m], 3),
        (["history", "checkpoint", "run", "no-such-run",
          "--model-id", m], 3),
    ]
    for args, code in probes:
        r = run_cli(args, env.root)
        assert r.returncode == code, (args, r.returncode, r.stderr)
        assert r.stdout == "", args  # no fabricated results


def test_m76_history_id_positions(env, ids):
    """The model-argument-position bug class: a MODEL id in the
    value position must fail (or return nothing fabricated), and a
    checkpoint of ANOTHER model must be rejected by the engine."""
    # model A's id offered as a checkpoint id under model A -> 404
    r = run_cli(["history", "evaluation", "checkpoint", ids["model"][0],
                 "--model-id", ids["model"][0]], env.root)
    assert r.returncode == 3, (r.returncode, r.stderr)
    # model A's checkpoint asked under model D -> cross-model 404
    r = run_cli(["history", "evaluation", "checkpoint",
                 ids["checkpoint"][0], "--model-id", ids["model_d"]],
                env.root)
    assert r.returncode == 3
    r = run_cli(["history", "sample", "checkpoint",
                 ids["checkpoint"][0], "--model-id", ids["model_d"]],
                env.root)
    assert r.returncode == 3
    # the CORRECT invocation succeeds (positions verified)
    r = run_cli(["history", "evaluation", "checkpoint",
                 ids["checkpoint"][0], "--model-id",
                 ids["checkpoint"][1]], env.root)
    assert r.returncode == 0, r.stderr


def test_m76_determinism(env):
    probes = [
        ["dashboard", env.model],
        ["dashboard", env.model, "--json"],
        ["history", "best_checkpoint", "--model-id", env.model],
        ["history", "evaluation", "checkpoint", env.ck[0],
         "--model-id", env.model, "--json"],
        ["history", "suite_run", "suite_summary", "m73-suite",
         "--model-id", env.model],
        ["history", "gate", "baseline_type", "checkpoint",
         "--model-id", env.model, "--json"],
    ]
    for args in probes:
        one = run_cli(args, env.root)
        two = run_cli(args, env.root)
        assert one.returncode == two.returncode == 0, args
        assert one.stdout == two.stdout, args
        assert one.stderr == two.stderr, args


def test_m76_registry_no_drift(env):
    """The CLI history table must match the ENGINE facade exactly:
    every routed method exists, the value coercion matches the
    facade's own parameter annotation, and the routed set IS the
    complete set of by-X facade methods (no missing surface, no
    extra)."""
    import inspect
    import typing
    from app import cli as cli_mod
    from app.engine import ModelForge
    routed = set()
    for family, dims in cli_mod._HISTORY_DISPATCH.items():
        for dimension, (method, marker) in dims.items():
            assert hasattr(ModelForge, method), (family, dimension)
            func = getattr(ModelForge, method)
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            assert len(params) == 3, method  # self, model_id, value
            # engine.py uses deferred annotations -> resolve them
            hints = typing.get_type_hints(func)
            annotation = hints[params[2].name]
            if marker == "str":
                assert annotation is str, (method, annotation)
            elif marker == "int":
                assert annotation is int, (method, annotation)
            elif marker == "bool":
                assert annotation is bool, (method, annotation)
            else:  # enum:<Name> — the schemas enum class itself
                from app import schemas
                assert annotation is getattr(
                    schemas, marker.split(":", 1)[1]), \
                    (method, annotation, marker)
            routed.add(method)
    facade_by_x = {name for name in dir(ModelForge)
                   if name.startswith("list_") and "_for_" in name}
    assert routed == facade_by_x, \
        (routed - facade_by_x, facade_by_x - routed)
    assert len(routed) == 32
    assert set(cli_mod.HISTORY_FAMILIES) == set(
        cli_mod._HISTORY_DISPATCH)
    # best_checkpoint history routes to the M59 facade verbatim
    assert hasattr(ModelForge, "best_checkpoint_history")


# =========================================================================== #
# M77: read-only LISTING CLI
# =========================================================================== #

M77_GLOBAL = ("models", "datasets", "tokenizers", "recipes",
              "policies", "suites")
M77_SCOPED = ("checkpoints", "evaluations", "comparisons", "gates",
              "workflows", "suite_runs", "samples", "sample_quality")


def test_m77_help():
    r = run_cli(["list", "--help"], "/nonexistent-root")
    assert r.returncode == 0, r.stderr
    assert "listing family" in r.stdout
    for family in M77_GLOBAL + M77_SCOPED:
        assert family in r.stdout


def test_m77_list_every_family(env):
    """The FULL 14-family matrix against DIRECT facade calls (the
    independent oracle — the expected values never come from the
    CLI's own dispatch table)."""
    forge = env.forge
    direct_global = {
        "models": forge.list_models(),
        "datasets": forge.list_datasets(),
        "tokenizers": forge.list_tokenizers(),
        "recipes": forge.list_workflow_recipes(),
        "policies": forge.list_policies(),
        "suites": forge.list_probe_suites(),
    }
    direct_scoped = {
        "checkpoints": forge.list_checkpoints(env.model),
        "evaluations": forge.list_evaluations(env.model),
        "comparisons": forge.list_comparisons(env.model),
        "gates": forge.list_gate_decisions(env.model),
        "workflows": forge.list_workflows(env.model),
        "suite_runs": forge.list_suite_runs(env.model),
        "samples": forge.list_samples(env.model),
        "sample_quality": forge.list_sample_evaluations(env.model),
    }
    def _dump(records):
        # the facade contract is mixed: most listings return
        # Pydantic records; list_datasets/list_tokenizers return
        # PRE-serialized dicts (passed through verbatim)
        return [r.model_dump(mode="json")
                if hasattr(r, "model_dump") else r for r in records]

    for family, records in direct_global.items():
        expected = _dump(records)
        rj = run_cli(["list", family, "--json"], env.root)
        assert rj.returncode == 0, (family, rj.stderr)
        assert json.loads(rj.stdout) == expected, family
        rh = run_cli(["list", family], env.root)
        assert rh.returncode == 0, family
        assert f"count: {len(expected)}" in rh.stdout, family
    for family, records in direct_scoped.items():
        expected = _dump(records)
        rj = run_cli(["list", family, "--model-id", env.model,
                      "--json"], env.root)
        assert rj.returncode == 0, (family, rj.stderr)
        assert json.loads(rj.stdout) == expected, family
        rh = run_cli(["list", family, "--model-id", env.model],
                     env.root)
        assert rh.returncode == 0, family
        assert f"count: {len(expected)}" in rh.stdout, family
    # an EMPTY listing is a valid success (model D has no samples)
    r = run_cli(["list", "samples", "--model-id", env.model_d],
                env.root)
    assert r.returncode == 0 and "count: 0" in r.stdout
    rj = run_cli(["list", "samples", "--model-id", env.model_d,
                  "--json"], env.root)
    assert rj.returncode == 0 and json.loads(rj.stdout) == []


def test_m77_list_errors(env):
    probes = [
        (["list", "bogus"], 4),
        (["list", "training_run"], 5),  # lifecycle-less family
        (["list", "checkpoints"], 2),  # missing --model-id
        (["list", "evaluations"], 2),
        (["list", "models", "--model-id", env.model], 2),  # stray
        (["list", "recipes", "--model-id", env.model], 2),
        (["list", "checkpoints", "SOME_ID", "--model-id",
          env.model], 2),  # misplaced positional
        (["list", "checkpoints", "--model-id", "no-such"], 3),
        (["list", "workflows", "--model-id", "no-such"], 3),
        (["list"], 2),  # missing family
    ]
    for args, code in probes:
        r = run_cli(args, env.root)
        assert r.returncode == code, (args, r.returncode, r.stderr)
        assert r.stdout == "", args  # no fabricated results


def test_m77_determinism(env):
    probes = [
        ["list", "models"],
        ["list", "models", "--json"],
        ["list", "policies", "--json"],
        ["list", "checkpoints", "--model-id", env.model],
        ["list", "sample_quality", "--model-id", env.model, "--json"],
        ["list", "samples", "--model-id", env.model_d],
    ]
    for args in probes:
        one = run_cli(args, env.root)
        two = run_cli(args, env.root)
        assert one.returncode == two.returncode == 0, args
        assert one.stdout == two.stdout, args
        assert one.stderr == two.stderr, args


def test_m77_registry_no_drift(env):
    """The CLI listing table must equal the ENGINE's actual facade
    listing surface BY SIGNATURE SHAPE: the routed global set IS the
    set of zero-argument ModelForge.list_* methods, the routed
    scoped set IS the set of (model_id)-shaped ones — no missing
    surface, no extra, no renamed method, no wrong scope."""
    import inspect
    from app import cli as cli_mod
    from app.engine import ModelForge
    zero, one = set(), set()
    for name in dir(ModelForge):
        if not name.startswith("list_"):
            continue
        params = [p for p in
                  inspect.signature(getattr(ModelForge, name))
                  .parameters if p != "self"]
        if len(params) == 0:
            zero.add(name)
        elif params == ["model_id"]:
            one.add(name)
    routed_global = {m for m, _s in
                     cli_mod._LIST_DISPATCH.values() if not _s}
    routed_scoped = {m for m, s in
                     cli_mod._LIST_DISPATCH.values() if s}
    assert routed_global == zero, (routed_global - zero,
                                   zero - routed_global)
    assert routed_scoped == one, (routed_scoped - one,
                                  one - routed_scoped)
    assert set(cli_mod.LIST_FAMILIES) == set(cli_mod._LIST_DISPATCH)
    assert len(cli_mod._LIST_DISPATCH) == 14
    # every routed method exists on the facade (no fictional names)
    for method, _s in cli_mod._LIST_DISPATCH.values():
        assert callable(getattr(ModelForge, method, None)), method


# =========================================================================== #
# M78: read-only single-artifact SHOW CLI
# =========================================================================== #

def test_m78_help():
    r = run_cli(["show", "--help"], "/nonexistent-root")
    assert r.returncode == 0, r.stderr
    assert "artifact family" in r.stdout
    for family in ("model", "dataset", "tokenizer", "workflow_recipe",
                   "gate_policy", "probe_suite", "checkpoint",
                   "workflow", "evaluation", "comparison", "gate",
                   "suite_run", "sample", "sample_quality"):
        assert family in r.stdout


def test_m78_show_every_family(env, ids):
    """The FULL 14-family matrix against DIRECT facade getters (the
    independent oracle — expected values never come from the CLI's
    own dispatch table; the dataset getter's pre-serialized dict
    contract is handled explicitly)."""
    forge = env.forge

    def _dump(record):
        return (record.model_dump(mode="json")
                if hasattr(record, "model_dump") else record)

    direct_global = {
        "model": forge.get_model(ids["model"][0]),
        "dataset": forge.get_dataset(ids["dataset"][0]),
        "tokenizer": forge.get_tokenizer(ids["tokenizer"][0]),
        "workflow_recipe": forge.get_workflow_recipe(
            ids["workflow_recipe"][0]),
        "gate_policy": forge.get_policy(ids["gate_policy"][0]),
        "probe_suite": forge.get_probe_suite(ids["probe_suite"][0]),
    }
    direct_scoped = {
        "checkpoint": forge.get_checkpoint(
            ids["checkpoint"][1], ids["checkpoint"][0]),
        "workflow": forge.get_workflow(
            ids["workflow"][1], ids["workflow"][0]),
        "evaluation": forge.get_evaluation(
            ids["evaluation"][1], ids["evaluation"][0]),
        "comparison": forge.get_comparison(
            ids["comparison"][1], ids["comparison"][0]),
        "gate": forge.get_gate_decision(ids["gate"][1], ids["gate"][0]),
        "suite_run": forge.get_suite_run(
            ids["suite_run"][1], ids["suite_run"][0]),
        "sample": forge.get_sample(ids["sample"][1], ids["sample"][0]),
        "sample_quality": forge.get_sample_evaluation(
            ids["sample_quality"][1], ids["sample_quality"][0]),
    }
    for family, record in direct_global.items():
        expected = _dump(record)
        rj = run_cli(["show", family, ids[family][0], "--json"],
                     env.root)
        assert rj.returncode == 0, (family, rj.stderr)
        assert json.loads(rj.stdout) == expected, family
        rh = run_cli(["show", family, ids[family][0]], env.root)
        assert rh.returncode == 0 and rh.stdout.strip(), family
    for family, record in direct_scoped.items():
        expected = _dump(record)
        args = ["show", family, ids[family][0],
                "--model-id", ids[family][1]]
        rj = run_cli([*args, "--json"], env.root)
        assert rj.returncode == 0, (family, rj.stderr)
        assert json.loads(rj.stdout) == expected, family
        rh = run_cli(args, env.root)
        assert rh.returncode == 0 and rh.stdout.strip(), family
    # the dataset getter's dict result renders (human) too
    rh = run_cli(["show", "dataset", ids["dataset"][0]], env.root)
    assert "dataset_id" in rh.stdout


def test_m78_show_errors(env, ids):
    probes = [
        (["show", "bogus", "x"], 4),
        (["show", "training_run", "x"], 5),  # lifecycle-less
        (["show", "checkpoint", "x"], 2),  # missing --model-id
        (["show", "evaluation", "x"], 2),
        (["show", "model", "x", "--model-id", env.model], 2),  # stray
        (["show", "dataset", "x", "--model-id", env.model], 2),
        (["show", "model", "no-such"], 3),
        (["show", "dataset", "no-such"], 3),
        (["show", "tokenizer", "no-such"], 3),
        (["show", "workflow_recipe", "no-such"], 3),
        (["show", "gate_policy", "no-such"], 3),
        (["show", "probe_suite", "no-such"], 3),
        (["show", "checkpoint", "no-such", "--model-id", env.model], 3),
        (["show", "workflow", "no-such", "--model-id", env.model], 3),
        (["show", "suite_run", "no-such", "--model-id", env.model], 3),
        (["show", "sample", "no-such", "--model-id", env.model], 3),
        # the model-argument-position bug class: a MODEL id in the
        # artifact position is an unknown artifact (never routed to
        # another family's view)
        (["show", "evaluation", ids["model"][0],
          "--model-id", ids["model"][0]], 3),
        # cross-model: model A's checkpoint under model D -> 3
        (["show", "checkpoint", ids["checkpoint"][0],
          "--model-id", ids["model_d"]], 3),
        (["show"], 2),  # missing family
        (["show", "model"], 2),  # missing id
    ]
    for args, code in probes:
        r = run_cli(args, env.root)
        assert r.returncode == code, (args, r.returncode, r.stderr)
        assert r.stdout == "", args  # no fabricated results


def test_m78_determinism(env, ids):
    probes = [
        ["show", "model", ids["model"][0]],
        ["show", "model", ids["model"][0], "--json"],
        ["show", "dataset", ids["dataset"][0], "--json"],
        ["show", "checkpoint", ids["checkpoint"][0],
         "--model-id", ids["checkpoint"][1]],
        ["show", "gate", ids["gate"][0], "--model-id",
         ids["gate"][1], "--json"],
        ["show", "sample_quality", ids["sample_quality"][0],
         "--model-id", ids["sample_quality"][1], "--json"],
    ]
    for args in probes:
        one = run_cli(args, env.root)
        two = run_cli(args, env.root)
        assert one.returncode == two.returncode == 0, args
        assert one.stdout == two.stdout, args
        assert one.stderr == two.stderr, args


def test_m78_registry_no_drift(env):
    """The CLI show table must equal the ENGINE's actual getter
    surface BY SIGNATURE SHAPE: the routed global set IS the set of
    single-parameter ModelForge.get_* methods MINUS get_dashboard
    (already routed by 'forge dashboard'); the routed scoped set IS
    the set of (model_id, artifact)-shaped getters — no missing
    surface, no extra, no renamed method, no wrong scope."""
    import inspect
    from app import cli as cli_mod
    from app.engine import ModelForge
    one_param, two_param = set(), set()
    for name in dir(ModelForge):
        if not name.startswith("get_"):
            continue
        params = [p for p in
                  inspect.signature(getattr(ModelForge, name))
                  .parameters if p != "self"]
        if len(params) == 1:
            one_param.add(name)
        elif len(params) == 2:
            two_param.add(name)
    routed_global = {m for m, _s in
                     cli_mod._SHOW_DISPATCH.values() if not _s}
    routed_scoped = {m for m, s in
                     cli_mod._SHOW_DISPATCH.values() if s}
    assert routed_global == one_param - {"get_dashboard"}, (
        routed_global - (one_param - {"get_dashboard"}),
        (one_param - {"get_dashboard"}) - routed_global)
    assert routed_scoped == two_param, (
        routed_scoped - two_param, two_param - routed_scoped)
    assert set(cli_mod.SHOW_FAMILIES) == set(cli_mod._SHOW_DISPATCH)
    assert len(cli_mod._SHOW_DISPATCH) == 14
    for method, _s in cli_mod._SHOW_DISPATCH.values():
        assert callable(getattr(ModelForge, method, None)), method
