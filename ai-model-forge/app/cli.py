"""READ-ONLY retention & deletion-impact command line (M74).

A THIN ADAPTER, never a second implementation: every command parses
argv, calls the ONE existing engine facade (the same methods the API
routes call), and formats the engine's own Pydantic result — no
second scanner, no second blocker engine, no second impact engine,
no second project-retention implementation. Strictly read-only:
zero storage, zero mutation of ``FORGE_ROOT``, zero deletion
capability; unknown artifacts surface the engine's canonical
errors with non-zero exit codes.

Commands (``FORGE_ROOT`` selects the data root, exactly as for the
API):

``forge retention project [--json]``
    the M72 project retention inventory
    (``project_retention_overview``);
``forge retention <family> <id> [--model-id M] [--json]``
    the family's EXISTING retention view (one command per deletable
    family); the eight model-scoped families require ``--model-id``;
``forge impact <family> <id> [--model-id M] [--json]``
    the M73 FIRST-LEVEL deletion impact preview
    (``deletion_impact_preview``);
``forge storage [--json]``
    the M63 PHYSICAL project storage overview
    (``project_storage_overview`` — the ONE storage walk);
``forge usage dataset <id> [--json]`` /
``forge usage tokenizer <id> [--json]``
    the M64 usage overviews (``dataset_usage_overview`` /
    ``tokenizer_usage_overview``);
``forge usage model <id> [--json]``
    the model usage overview (``model_usage_overview``);
``forge usage records <model_id> [--json]``
    the model-owned-record usage overview
    (``model_records_usage_overview``);
``forge usage definition <family> <id> [--json]``
    the M71 definition retention overview
    (``definition_retention_overview``) for the three definition
    families (workflow_recipe / gate_policy / probe_suite);
``forge dashboard <model_id> [--json]``
    the ONE dashboard surface (``get_dashboard``);
``forge history <family> [<dimension> <value>] --model-id M
[--json]``
    the M19-M60 by-X history groupings plus the M59 best-checkpoint
    history (family ``best_checkpoint`` — no dimension), ONE facade
    method per (family, dimension) pair, always model-scoped.
``forge list <family> [--model-id M] [--json]``
    the existing facade LISTINGS — six global families (models,
    datasets, tokenizers, recipes, policies, suites) and eight
    model-scoped families (checkpoints, evaluations, comparisons,
    gates, workflows, suite_runs, samples, sample_quality);
    ``--model-id`` exactly where the facade requires it, empty
    listings are valid successes.
``forge show <family> <id> [--model-id M] [--json]``
    the existing facade GETTERS — one single-artifact view per
    family (the SAME 14 families as retention/list); unknown or
    registry-invisible artifact -> exit 3; the dataset getter
    returns a pre-serialized dict (passed through verbatim).

``training_run`` remains lifecycle-less (M68/M70/M72/M73): both
commands refuse it with the canonical lifecycle error and a non-zero
exit — no fabricated retention or impact result. Exit codes:
0 success; 2 usage; 3 unknown/registry-invisible artifact; 4
unknown family; 5 lifecycle-less family; 1 any other engine error.
A blocked or integrity-failed artifact is NOT a command failure —
its view describes the state faithfully (``deletable: false`` /
``integrity_verified: false``) and the command succeeds.
"""
from __future__ import annotations

import argparse
import json
import sys

# The retention dispatch table: family -> (facade method name,
# whether the family is MODEL-SCOPED). The keys are exactly the 14
# deletable families of the engine's own registries (a test asserts
# no drift against ``ModelForge.IMPACT_FAMILIES``); the table only
# ROUTES to the existing facade methods — it holds no domain logic.
_RETENTION_DISPATCH = {
    "model": ("model_retention_overview", False),
    "dataset": ("dataset_retention_overview", False),
    "tokenizer": ("tokenizer_retention_overview", False),
    "workflow_recipe": ("definition_retention_overview", False),
    "gate_policy": ("definition_retention_overview", False),
    "probe_suite": ("definition_retention_overview", False),
    "checkpoint": ("checkpoint_retention_overview", True),
    "workflow": ("model_record_retention_overview", True),
    "evaluation": ("model_record_retention_overview", True),
    "comparison": ("model_record_retention_overview", True),
    "gate": ("model_record_retention_overview", True),
    "suite_run": ("suite_run_retention_overview", True),
    "sample": ("sample_retention_overview", True),
    "sample_quality": ("sample_evaluation_retention_overview", True),
}
RETENTION_FAMILIES = tuple(_RETENTION_DISPATCH)
MODEL_SCOPED_FAMILIES = frozenset(
    f for f, (_m, scoped) in _RETENTION_DISPATCH.items() if scoped)

# The M76 history dispatch table: family -> dimension -> (facade
# method name, value marker). Derived ONE-to-ONE from the engine's
# own by-X facade methods (the M19-M60 read-only groupings plus the
# M59 best-checkpoint history); the keys mirror the API's by-<X>
# segments and the retention family vocabulary. Value markers:
# "str" (an artifact id), "int", "bool", "enum:<SchemaEnum>" — the
# CLI only COERCES the argv string and routes; the engine does the
# canonical validation (unknown ids -> FileNotFoundError -> exit 3).
# A drift test asserts this table matches the engine facade exactly.
_HISTORY_DISPATCH = {
    "checkpoint": {
        "run": ("list_checkpoints_for_run", "str"),
    },
    "evaluation": {
        "checkpoint": ("list_evaluations_for_checkpoint", "str"),
        "dataset": ("list_evaluations_for_dataset", "str"),
        "tokenizer": ("list_evaluations_for_tokenizer", "str"),
        "split": ("list_evaluations_for_split",
                  "enum:EvaluationSplit"),
        "state_kind": ("list_evaluations_for_state_kind",
                       "enum:EvalStateKind"),
        "truncated": ("list_evaluations_for_truncated", "bool"),
        "seed": ("list_evaluations_for_seed", "int"),
    },
    "comparison": {
        "checkpoint": ("list_comparisons_for_checkpoint", "str"),
        "dataset": ("list_comparisons_for_dataset", "str"),
        "tokenizer": ("list_comparisons_for_tokenizer", "str"),
        "split": ("list_comparisons_for_split",
                  "enum:EvaluationSplit"),
        "verdict": ("list_comparisons_for_verdict",
                    "enum:ComparisonVerdict"),
        "state_kind": ("list_comparisons_for_state_kind",
                       "enum:EvalStateKind"),
        "seed": ("list_comparisons_for_seed", "int"),
    },
    "gate": {
        "policy": ("list_gate_decisions_for_policy", "str"),
        "comparison": ("list_gate_decisions_for_comparison", "str"),
        "decision": ("list_gate_decisions_for_decision",
                     "enum:GateDecisionResult"),
        "verdict": ("list_gate_decisions_for_verdict",
                    "enum:ComparisonVerdict"),
        "baseline_type": ("list_gate_decisions_for_baseline_type",
                          "enum:GateBaselineType"),
    },
    "workflow": {
        "recipe": ("list_workflows_for_recipe", "str"),
        "status": ("list_workflows_for_status",
                   "enum:WorkflowStatus"),
    },
    "suite_run": {
        "suite": ("list_suite_runs_for_suite", "str"),
        "suite_summary": ("list_suite_run_summary_for_suite", "str"),
        "checkpoint": ("list_suite_runs_for_checkpoint", "str"),
        "reused_count": ("list_suite_runs_for_reused_count", "int"),
    },
    "sample": {
        "checkpoint": ("list_samples_for_checkpoint", "str"),
        "tokenizer": ("list_samples_for_tokenizer", "str"),
        "strategy": ("list_samples_for_strategy",
                     "enum:SampleStrategy"),
    },
    "sample_quality": {
        "sample": ("list_sample_evaluations_for_sample", "str"),
        "checkpoint": ("list_sample_evaluations_for_checkpoint",
                       "str"),
        "tokenizer": ("list_sample_evaluations_for_tokenizer",
                      "str"),
    },
    # the M59 best-checkpoint selection history: NO dimension — the
    # model id alone (still --model-id, like every history surface)
    "best_checkpoint": {},
}
HISTORY_FAMILIES = tuple(_HISTORY_DISPATCH)

# The M77 listing dispatch table: family -> (facade method name,
# model-scoped). The keys are the repository's family vocabulary in
# the plural (a LIST command names the collection); every entry is
# ONE existing ModelForge facade method — the SIX zero-argument
# global listings and the EIGHT model-scoped record listings. A
# drift test pins this table to the engine's actual list_* facade
# set by signature shape (no missing surface, no extra, no wrong
# scope); the CLI never constructs a listing itself.
_LIST_DISPATCH = {
    "models": ("list_models", False),
    "datasets": ("list_datasets", False),
    "tokenizers": ("list_tokenizers", False),
    "recipes": ("list_workflow_recipes", False),
    "policies": ("list_policies", False),
    "suites": ("list_probe_suites", False),
    "checkpoints": ("list_checkpoints", True),
    "evaluations": ("list_evaluations", True),
    "comparisons": ("list_comparisons", True),
    "gates": ("list_gate_decisions", True),
    "workflows": ("list_workflows", True),
    "suite_runs": ("list_suite_runs", True),
    "samples": ("list_samples", True),
    "sample_quality": ("list_sample_evaluations", True),
}
LIST_FAMILIES = tuple(_LIST_DISPATCH)

# The M78 single-artifact dispatch table: family -> (facade getter
# name, model-scoped). The keys are the repository's retention
# family vocabulary (the SAME 14 families as the M74 retention and
# M77 listing surfaces); every entry is ONE existing ModelForge
# getter — the SIX global getters and the EIGHT model-scoped ones.
# A drift test pins this table to the engine's actual get_* facade
# set by signature shape (get_dashboard is deliberately absent —
# it is already routed by 'forge dashboard'); the CLI never
# constructs a record itself.
_SHOW_DISPATCH = {
    "model": ("get_model", False),
    "dataset": ("get_dataset", False),
    "tokenizer": ("get_tokenizer", False),
    "workflow_recipe": ("get_workflow_recipe", False),
    "gate_policy": ("get_policy", False),
    "probe_suite": ("get_probe_suite", False),
    "checkpoint": ("get_checkpoint", True),
    "workflow": ("get_workflow", True),
    "evaluation": ("get_evaluation", True),
    "comparison": ("get_comparison", True),
    "gate": ("get_gate_decision", True),
    "suite_run": ("get_suite_run", True),
    "sample": ("get_sample", True),
    "sample_quality": ("get_sample_evaluation", True),
}
SHOW_FAMILIES = tuple(_SHOW_DISPATCH)

EXIT_OK = 0
EXIT_ENGINE_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_UNKNOWN_FAMILY = 4
EXIT_NO_LIFECYCLE = 5


class CliError(Exception):
    """One CLI-level routing error with its dedicated exit code."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


def build_parser() -> argparse.ArgumentParser:
    """The ONE argument parser (stdlib argparse only, no CLI
    framework). ``--help`` at every level exits 0."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Read-only retention & deletion-impact CLI "
                    "(a thin adapter over the engine; never deletes "
                    "anything).")
    sub = parser.add_subparsers(dest="command", required=True)

    ret = sub.add_parser(
        "retention", help="read-only retention views")
    ret.add_argument(
        "family", nargs="?", metavar="family",
        help="'project' or a deletable family "
             f"({', '.join(RETENTION_FAMILIES)})")
    ret.add_argument(
        "artifact_id", nargs="?", metavar="id",
        help="the artifact's id (not for 'project')")
    ret.add_argument(
        "--model-id", metavar="M", dest="model_id",
        help="the owning model's id (required for the model-scoped "
             "families: " + ", ".join(sorted(MODEL_SCOPED_FAMILIES))
             + ")")
    ret.add_argument(
        "--json", action="store_true",
        help="machine-readable JSON (the engine model's "
             "model_dump(mode='json')) instead of human text")

    imp = sub.add_parser(
        "impact", help="read-only FIRST-LEVEL deletion impact "
                       "preview (M73)")
    imp.add_argument(
        "family", metavar="family",
        help="a deletable family (" + ", ".join(RETENTION_FAMILIES)
             + ")")
    imp.add_argument(
        "artifact_id", metavar="id", help="the artifact's id")
    imp.add_argument(
        "--model-id", metavar="M", dest="model_id",
        help="the owning model's id (required for the model-scoped "
             "families: " + ", ".join(sorted(MODEL_SCOPED_FAMILIES))
             + ")")
    imp.add_argument(
        "--json", action="store_true",
        help="machine-readable JSON (the engine model's "
             "model_dump(mode='json')) instead of human text")

    sto = sub.add_parser(
        "storage", help="read-only PHYSICAL project storage overview "
                        "(the ONE M63 storage walk)")
    sto.add_argument(
        "--json", action="store_true",
        help="machine-readable JSON (the engine model's "
             "model_dump(mode='json')) instead of human text")

    # the ONE shared --json flag for every usage subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true",
        help="machine-readable JSON (the engine model's "
             "model_dump(mode='json')) instead of human text")
    use = sub.add_parser(
        "usage", help="read-only usage overviews (M64 dataset/tokenizer, "
                      "model usage, M69 model-owned records, M71 "
                      "definitions)")
    u_sub = use.add_subparsers(dest="kind", required=True)
    u_sub.add_parser(
        "dataset", parents=[common],
        help="M64 dataset usage overview").add_argument(
            "dataset_id", metavar="id", help="the dataset's id")
    u_sub.add_parser(
        "tokenizer", parents=[common],
        help="M64 tokenizer usage overview").add_argument(
            "tokenizer_id", metavar="id", help="the tokenizer's id")
    u_sub.add_parser(
        "model", parents=[common],
        help="model usage overview").add_argument(
            "model_id", metavar="id", help="the model's id")
    u_sub.add_parser(
        "records", parents=[common],
        help="M69 model-owned-record usage overview").add_argument(
            "model_id", metavar="model_id", help="the owning model's id")
    udef = u_sub.add_parser(
        "definition", parents=[common],
        help="M71 definition retention overview (workflow_recipe / "
             "gate_policy / probe_suite)")
    udef.add_argument(
        "family", metavar="family",
        help="a definition family (workflow_recipe, gate_policy, "
             "probe_suite)")
    udef.add_argument(
        "definition_id", metavar="id", help="the definition's id")

    dash = sub.add_parser(
        "dashboard", parents=[common],
        help="read-only model dashboard (the ONE dashboard surface)")
    dash.add_argument(
        "model_id", metavar="model_id", help="the model's id")

    hist = sub.add_parser(
        "history", parents=[common],
        help="read-only history groupings (the M19-M60 by-X "
             "surfaces + the M59 best-checkpoint history)")
    hist.add_argument(
        "family", metavar="family",
        help="a history family (" + ", ".join(HISTORY_FAMILIES) + ")")
    hist.add_argument(
        "dimension", nargs="?", metavar="dimension",
        help="the by-X grouping dimension (family-specific; "
             "run 'forge history --help' per family via the error "
             "messages)")
    hist.add_argument(
        "value", nargs="?", metavar="value",
        help="the dimension's value (an artifact id, enum name, "
             "integer or true/false)")
    hist.add_argument(
        "--model-id", metavar="M", dest="model_id", required=True,
        help="the owning model's id (every history surface is "
             "model-scoped)")
    lst = sub.add_parser(
        "list", parents=[common],
        help="read-only listings (the existing facade listing "
             "methods)")
    lst.add_argument(
        "family", metavar="family",
        help="a listing family (" + ", ".join(LIST_FAMILIES) + ")")
    lst.add_argument(
        "--model-id", metavar="M", dest="model_id",
        help="the owning model's id (required for the model-scoped "
             "families: checkpoints, evaluations, comparisons, "
             "gates, workflows, suite_runs, samples, "
             "sample_quality; rejected for the global families)")
    sho = sub.add_parser(
        "show", parents=[common],
        help="read-only single-artifact views (the existing facade "
             "getters)")
    sho.add_argument(
        "family", metavar="family",
        help="an artifact family (" + ", ".join(SHOW_FAMILIES) + ")")
    sho.add_argument(
        "artifact_id", metavar="id", help="the artifact's id")
    sho.add_argument(
        "--model-id", metavar="M", dest="model_id",
        help="the owning model's id (required for the model-scoped "
             "families: checkpoint, workflow, evaluation, "
             "comparison, gate, suite_run, sample, sample_quality; "
             "rejected for the global families)")
    return parser


def _validate_family(family: str, supported, all_families) -> None:
    """Refuse a family exactly the way the engine's registries
    classify it: a KNOWN family without a deletion lifecycle gets the
    canonical lifecycle error; anything else is unknown. Never a
    fabricated result."""
    if family in supported:
        return
    if family in all_families:
        raise CliError(
            f"family '{family}' has no deletion lifecycle (families: "
            f"{', '.join(supported)})", EXIT_NO_LIFECYCLE)
    raise CliError(
        f"unknown family '{family}' (families: {', '.join(supported)})",
        EXIT_UNKNOWN_FAMILY)


def _require_model_id(args) -> None:
    """Enforce the engine's own signature shape at the CLI boundary:
    the model-scoped families take ``--model-id`` (never guessed,
    never passed in the artifact position); every other family takes
    NONE (a stray ``--model-id`` is a usage error)."""
    scoped = args.family in MODEL_SCOPED_FAMILIES
    if scoped and not args.model_id:
        raise CliError(
            f"family '{args.family}' is model-scoped: --model-id is "
            f"required", EXIT_USAGE)
    if not scoped and args.model_id:
        raise CliError(
            f"family '{args.family}' is not model-scoped: --model-id "
            f"is not accepted", EXIT_USAGE)


def _run_retention(forge, args):
    """Route ONE retention command to the EXISTING facade method and
    return its Pydantic result unchanged. Routing only — no
    recomputation of files, bytes, blockers, deletability or
    reclaimable storage."""
    if args.family == "project":
        if args.artifact_id or args.model_id:
            raise CliError("'retention project' takes no artifact id "
                           "or --model-id", EXIT_USAGE)
        return forge.project_retention_overview()
    _validate_family(args.family, RETENTION_FAMILIES,
                     forge.PROJECT_RETENTION_FAMILIES)
    _require_model_id(args)
    method_name, _scoped = _RETENTION_DISPATCH[args.family]
    method = getattr(forge, method_name)
    if args.family in ("workflow_recipe", "gate_policy", "probe_suite"):
        return method(args.family, args.artifact_id)
    if args.family in ("workflow", "evaluation", "comparison", "gate"):
        return method(args.model_id, args.family, args.artifact_id)
    if args.family == "checkpoint":
        overview = method(args.model_id)
        for entry in overview.checkpoints:
            if entry.checkpoint_id == args.artifact_id:
                return entry
        raise FileNotFoundError(
            f"checkpoint '{args.artifact_id}' not found for model "
            f"'{args.model_id}'")
    if args.family in ("model", "dataset", "tokenizer"):
        return method(args.artifact_id)
    if args.family in ("suite_run", "sample", "sample_quality"):
        return method(args.model_id, args.artifact_id)
    raise CliError(f"unrouted family '{args.family}'", EXIT_USAGE)


def _run_impact(forge, args):
    """Route ONE impact command to the ONE M73 facade
    (``deletion_impact_preview``) and return its result unchanged —
    no impact recomputation, no cascade, no simulation."""
    _validate_family(args.family, forge.IMPACT_FAMILIES,
                     forge.PROJECT_RETENTION_FAMILIES)
    _require_model_id(args)
    return forge.deletion_impact_preview(
        args.family, args.artifact_id, args.model_id)


def _run_storage(forge, args):
    """Route ONE storage command to the ONE M63 facade
    (``project_storage_overview``) and return its result unchanged —
    never a second filesystem walker."""
    return forge.project_storage_overview()


def _run_usage(forge, args):
    """Route ONE usage command to the EXISTING facade method and
    return its Pydantic result unchanged. Routing only — no second
    reference discovery, no second usage aggregation, no second
    definition dependency scanner."""
    if args.kind == "dataset":
        return forge.dataset_usage_overview(args.dataset_id)
    if args.kind == "tokenizer":
        return forge.tokenizer_usage_overview(args.tokenizer_id)
    if args.kind == "model":
        return forge.model_usage_overview(args.model_id)
    if args.kind == "records":
        return forge.model_records_usage_overview(args.model_id)
    # definition: validated against the engine's OWN registry (a
    # lifecycle-less family such as training_run is refused with the
    # M74 semantics; unknown families with the M74 unknown-family
    # error) — never a fabricated usage result
    _validate_family(args.family, forge.DEFINITION_DELETABLE_FAMILIES,
                     forge.PROJECT_RETENTION_FAMILIES)
    return forge.definition_retention_overview(
        args.family, args.definition_id)


def _coerce_history_value(dimension: str, raw: str, marker: str):
    """Coerce ONE argv string to the facade's exact parameter type
    (id / int / bool / a schemas enum). Malformed values are USAGE
    errors raised BEFORE the engine is called — the engine never
    receives garbage; well-formed but unknown ids stay the engine's
    own canonical errors (FileNotFoundError -> exit 3)."""
    if marker == "str":
        return raw
    if marker == "int":
        try:
            return int(raw)
        except ValueError:
            raise CliError(
                f"invalid integer '{raw}' for dimension "
                f"'{dimension}'", EXIT_USAGE)
    if marker == "bool":
        if raw in ("true", "True"):
            return True
        if raw in ("false", "False"):
            return False
        raise CliError(
            f"invalid boolean '{raw}' for dimension '{dimension}' "
            f"(true/false)", EXIT_USAGE)
    from app import schemas  # deferred: --help stays fast
    enum_cls = getattr(schemas, marker.split(":", 1)[1])
    try:
        return enum_cls(raw)
    except ValueError:
        raise CliError(
            f"invalid value '{raw}' for dimension '{dimension}' "
            f"(valid: {', '.join(v.value for v in enum_cls)})",
            EXIT_USAGE)


def _run_dashboard(forge, args):
    """Route ONE dashboard command to the ONE existing dashboard
    facade (``get_dashboard``) and return its result unchanged."""
    return forge.get_dashboard(args.model_id)


def _run_history(forge, args):
    """Route ONE history command to the EXISTING by-X facade method
    and return its result unchanged. Routing + argv coercion only —
    no second history engine, no second grouping, no aggregation."""
    _validate_family(args.family, HISTORY_FAMILIES,
                     forge.PROJECT_RETENTION_FAMILIES)
    if args.family == "best_checkpoint":
        if args.dimension is not None or args.value is not None:
            raise CliError(
                "history best_checkpoint takes no dimension or "
                "value", EXIT_USAGE)
        return forge.best_checkpoint_history(args.model_id)
    if args.dimension is None or args.value is None:
        raise CliError(
            f"history {args.family} requires a dimension and a "
            f"value (dimensions: "
            f"{', '.join(_HISTORY_DISPATCH[args.family])})",
            EXIT_USAGE)
    dims = _HISTORY_DISPATCH[args.family]
    if args.dimension not in dims:
        raise CliError(
            f"unsupported dimension '{args.dimension}' for family "
            f"'{args.family}' (dimensions: {', '.join(dims)})",
            EXIT_UNKNOWN_FAMILY)
    method_name, marker = dims[args.dimension]
    value = _coerce_history_value(args.dimension, args.value, marker)
    return getattr(forge, method_name)(args.model_id, value)


def _run_list(forge, args):
    """Route ONE listing command to the EXISTING facade listing
    method and return its result unchanged. Routing only — never a
    second listing engine, registry or scanner; empty listings are
    valid successful results. ``--model-id`` is enforced to match
    the facade's own signature shape exactly (required for the
    eight model-scoped families, rejected for the six global ones)
    and never lands in any other argument position."""
    _validate_family(args.family, LIST_FAMILIES,
                     forge.PROJECT_RETENTION_FAMILIES)
    method_name, scoped = _LIST_DISPATCH[args.family]
    if scoped and not args.model_id:
        raise CliError(
            f"family '{args.family}' is model-scoped: --model-id "
            f"is required", EXIT_USAGE)
    if not scoped and args.model_id:
        raise CliError(
            f"family '{args.family}' is not model-scoped: "
            f"--model-id is not accepted", EXIT_USAGE)
    method = getattr(forge, method_name)
    if scoped:
        return method(args.model_id)
    return method()


def _run_show(forge, args):
    """Route ONE show command to the EXISTING facade getter and
    return its result unchanged. Routing only — never a second
    record construction; the getter's own canonical validation
    stands (unknown or registry-invisible artifact ->
    FileNotFoundError). ``--model-id`` is enforced to match the
    getter's own signature shape exactly (required for the eight
    model-scoped families, rejected for the six global ones) and
    never lands in the artifact position."""
    _validate_family(args.family, SHOW_FAMILIES,
                     forge.PROJECT_RETENTION_FAMILIES)
    method_name, scoped = _SHOW_DISPATCH[args.family]
    if scoped and not args.model_id:
        raise CliError(
            f"family '{args.family}' is model-scoped: --model-id "
            f"is required", EXIT_USAGE)
    if not scoped and args.model_id:
        raise CliError(
            f"family '{args.family}' is not model-scoped: "
            f"--model-id is not accepted", EXIT_USAGE)
    method = getattr(forge, method_name)
    if scoped:
        return method(args.model_id, args.artifact_id)
    return method(args.artifact_id)


def _render(value, indent: int = 0) -> list[str]:
    """Deterministic human-readable lines for ONE ``model_dump``
    value: fields in the model's declaration order, nested dicts
    indented, list items dashed — no timestamps, ids or order beyond
    what the engine result itself carries."""
    pad = "  " * indent
    lines: list[str] = []
    for key, item in value.items():
        if isinstance(item, dict) and item:
            lines.append(f"{pad}{key}:")
            lines.extend(_render(item, indent + 1))
        elif isinstance(item, list) and item:
            lines.append(f"{pad}{key}:")
            for element in item:
                if isinstance(element, dict):
                    body = _render(element, indent + 1)
                    first = body[0].replace(
                        f"{pad}  ", f"{pad}- ", 1)
                    lines.append(first)
                    lines.extend(body[1:])
                else:
                    lines.append(f"{pad}- {_scalar(element)}")
        else:
            lines.append(f"{pad}{key}: {_scalar(item)}")
    return lines


def _scalar(value) -> str:
    """One scalar in stable JSON-ish spelling (None -> null, booleans
    lowercase)."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _emit(result, as_json: bool) -> None:
    """Format ONE engine result (a Pydantic model, or a LIST of
    models — the M76 by-X surfaces) to stdout: byte-identical on
    repeats against the same state. JSON mode is exactly the
    ``model_dump(mode='json')`` of the result (an array for lists,
    sorted keys) — never a second schema."""
    if isinstance(result, list):
        # some facade listings (list_datasets / list_tokenizers)
        # return PRE-SERIALIZED dicts — passed through verbatim,
        # never a second serialization
        data = [r.model_dump(mode="json")
                if hasattr(r, "model_dump") else r for r in result]
        if as_json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(f"count: {len(data)}")
            for i, item in enumerate(data):
                print(f"- [{i}]")
                print("\n".join(_render(item, 1)))
        return
    if hasattr(result, "model_dump"):
        data = result.model_dump(mode="json")
    else:  # a pre-serialized dict facade result (get_dataset) —
        # passed through verbatim, never a second serialization
        data = result
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print("\n".join(_render(data)))


def main(argv: list[str] | None = None) -> int:
    """The CLI entry point: argv -> parse -> existing facade ->
    format -> stdout. Returns the process exit code (0 success; 2
    usage; 3 unknown artifact; 4 unknown family; 5 lifecycle-less
    family; 1 other engine errors — the canonical engine message
    always printed to stderr)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    from app.engine import get_forge  # deferred: --help stays fast
    try:
        if args.command == "storage":
            result = _run_storage(get_forge(), args)
        elif args.command == "usage":
            result = _run_usage(get_forge(), args)
        elif args.command == "dashboard":
            result = _run_dashboard(get_forge(), args)
        elif args.command == "history":
            result = _run_history(get_forge(), args)
        elif args.command == "list":
            result = _run_list(get_forge(), args)
        elif args.command == "show":
            result = _run_show(get_forge(), args)
        elif args.command == "retention":
            if args.family is None:
                raise CliError(
                    "retention requires 'project' or a family and an "
                    "artifact id", EXIT_USAGE)
            result = _run_retention(get_forge(), args)
        else:
            if not args.artifact_id:
                raise CliError("impact requires a family and an "
                               "artifact id", EXIT_USAGE)
            result = _run_impact(get_forge(), args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENGINE_ERROR
    except Exception as exc:  # never swallow engine errors silently
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENGINE_ERROR
    _emit(result, args.json)
    return EXIT_OK


if __name__ == "__main__":  # python -m app.cli ...
    raise SystemExit(main())
