"""CLI implementation layer for public entry points.

Library users should import from ``ga_lab.api`` instead of depending on this module directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ga_lab import __version__
from ga_lab.api import (
    list_demos,
    list_presets,
    recommend_preset,
    recommend_solver,
    run_config,
    run_demo,
    run_preset,
)
from ga_lab.config import load_config
from ga_lab.resources import split_resource_uri


def _print_payload(payload: dict[str, Any], output_format: str) -> None:
    if output_format == "text":
        lines = [str(value) for value in payload.values() if isinstance(value, str) and value]
        print("\n".join(lines))
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _run_request_from_args(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    overrides: dict[str, Any] = {}
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.seed is not None:
        overrides["seed"] = args.seed

    if args.config:
        config = load_config(args.config).to_dict()
        config.update(overrides)
        return "config", config
    if args.preset:
        return "preset", {"name": args.preset, "overrides": overrides}
    if args.resource:
        kind, name = split_resource_uri(args.resource)
        if kind != "preset":
            raise ValueError(
                f"Unsupported resource kind '{kind}' for ga-lab-run. "
                "Use a preset resource such as 'preset:onemax_small'."
            )
        return "preset", {"name": name, "overrides": overrides}
    raise ValueError("Provide one of --config, --preset, or --resource.")


def _sync_ops_if_requested(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.sync_ops:
        return
    try:
        from ga_ops.ingestion import sync_results_dir
        from ga_ops.settings import OpsSettings
    except ImportError as exc:  # pragma: no cover
        raise ValueError(
            "ga-lab-run --sync-ops requires repo dev mode with "
            "the local services/ package available."
        ) from exc

    settings = OpsSettings.from_env(project_root=Path.cwd())
    if args.ops_db_path is not None:
        settings.db_path = Path(args.ops_db_path)
    if args.object_store_root is not None:
        settings.object_store_root = Path(args.object_store_root)
    if args.object_store_provider is not None:
        settings.object_store_provider = args.object_store_provider
    sync_results_dir(
        output_dir,
        settings=settings,
        actor=args.ops_actor,
        source_collection="single_run",
    )


def run_experiment_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a GA experiment from a JSON config or builtin packaged preset."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="Path to an external JSON config file.")
    parser.add_argument("--preset", help="Builtin preset name, for example 'onemax_small'.")
    parser.add_argument(
        "--resource",
        help="Builtin resource reference, for example 'preset:onemax_small'.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Directory where run outputs will be stored.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the run_name field.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the seed field.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List builtin packaged presets and exit.",
    )
    parser.add_argument(
        "--sync-ops",
        action="store_true",
        help=(
            "Repo dev mode only: sync the generated run into "
            "the ops metadata DB and object store."
        ),
    )
    parser.add_argument("--ops-actor", default="run_experiment")
    parser.add_argument("--ops-db-path", default=None)
    parser.add_argument("--object-store-root", default=None)
    parser.add_argument("--object-store-provider", default=None)
    args = parser.parse_args(argv)

    if args.list_presets:
        payload = {
            "builtin_presets": [
                {
                    "name": preset.name,
                    "problem": preset.problem,
                    "tier": preset.tier,
                    "resource_uri": preset.resource_uri,
                    "portable_command_hint": preset.portable_command_hint,
                }
                for preset in list_presets()
            ]
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    selected_sources = [bool(args.config), bool(args.preset), bool(args.resource)]
    if sum(selected_sources) > 1:
        parser.error("use only one of --config, --preset, or --resource")
    if not any(selected_sources):
        parser.error(
            "one of --config, --preset, or --resource is required unless --list-presets is used"
        )

    try:
        source_kind, payload = _run_request_from_args(args)
        if source_kind == "preset":
            result = run_preset(
                payload["name"],
                output_root=args.output_root,
                overrides=payload["overrides"],
            )
        else:
            result = run_config(payload, output_root=args.output_root)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if result.output_dir is None:
        print("Run did not produce an output directory", file=sys.stderr)
        return 2

    try:
        _sync_ops_if_requested(args, Path(result.output_dir))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(json.dumps(result.raw_summary, indent=2, ensure_ascii=False))
    print(f"\nOutput directory: {result.output_dir}")
    return 0


def recommend_preset_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend a validated preset by problem and size."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--problem",
        required=True,
        choices=("knapsack", "onemax", "tsp", "zdt1"),
    )
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--priority", default="default")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    try:
        payload = recommend_preset(args.problem, args.size, args.priority, format="dict")
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    _print_payload(payload, args.format)
    return 0


def recommend_solver_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend a validated solver family and config by problem and size."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--problem",
        required=True,
        choices=("knapsack", "onemax", "tsp", "zdt1"),
    )
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--priority", default="default")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    try:
        payload = recommend_solver(args.problem, args.size, args.priority, format="dict")
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    _print_payload(payload, args.format)
    return 0


def demo_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one-command baseline, pure-GA, hybrid-GA, or NSGA-II demos."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    demo_names = [demo.name for demo in list_demos()]
    parser.add_argument(
        "demo_name",
        nargs="?",
        choices=[*demo_names, "all"],
        default=None,
        help="Optional positional demo selector, for example 'baseline' or 'nsga2'.",
    )
    parser.add_argument(
        "--demo",
        choices=[*demo_names, "all"],
        default=None,
        help="Demo path to run. Use 'all' to execute every demo in sequence.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/demo",
        help="Directory where demo outputs will be written.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available demos and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        payload = {
            "available_demos": [
                {
                    "demo": demo.name,
                    "mode": demo.mode,
                    "solver_family": demo.solver_family,
                    "title": demo.title,
                    "focus": demo.focus,
                    "resource_uri": demo.resource_uri,
                }
                for demo in list_demos()
            ]
        }
        _print_payload(payload, args.format)
        return 0

    selected_demo = args.demo or args.demo_name or "baseline"
    if args.demo and args.demo_name and args.demo != args.demo_name:
        parser.error("use either positional demo name or --demo, not both with different values")

    output_root = Path(args.output_root).resolve()
    if selected_demo == "all":
        payload = {
            "demo_suite": [
                run_demo(demo_name, output_root).raw_summary
                for demo_name in ("baseline", "pure-ga", "hybrid", "nsga2")
            ]
        }
        _print_payload(payload, args.format)
        return 0

    payload = run_demo(selected_demo, output_root).raw_summary
    _print_payload(payload, args.format)
    return 0
