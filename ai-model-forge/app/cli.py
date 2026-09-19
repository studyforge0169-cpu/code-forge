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
    (``deletion_impact_preview``).

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
    """Format ONE engine result to stdout: byte-identical on repeats
    against the same state. JSON mode is exactly the engine model's
    ``model_dump(mode='json')`` (sorted keys) — never a second
    schema."""
    data = result.model_dump(mode="json")
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
        if args.command == "retention":
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
