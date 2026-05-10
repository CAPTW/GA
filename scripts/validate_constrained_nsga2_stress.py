"""Run constrained NSGA-II seed/budget stress validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ga_lab.experiment.constrained_nsga2_stress import (
    ConstrainedNSGA2StressConfig,
    run_constrained_nsga2_stress,
)


def _comma_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated item")
    return items


def _int_list(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate constrained NSGA-II stress behavior on constrained MO toys.",
    )
    parser.add_argument(
        "--problems",
        type=_comma_list,
        default=("constrained_zdt_box_toy", "constrained_dtlz_box_toy"),
        help="Comma-separated constrained problem names.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Seed count, interpreted as seeds 0..N-1.",
    )
    parser.add_argument(
        "--budgets",
        type=_int_list,
        default=(760, 1500),
        help="Comma-separated evaluation budgets.",
    )
    parser.add_argument(
        "--strategies",
        type=_comma_list,
        default=("constrained_nsga2_constraint_domination", "random_pareto_archive"),
        help="Comma-separated strategy names.",
    )
    parser.add_argument(
        "--zdt-dimension",
        type=int,
        default=6,
        help="Dimension for constrained_zdt_box_toy.",
    )
    parser.add_argument(
        "--dtlz-dimension",
        type=int,
        default=7,
        help="Dimension for constrained_dtlz_box_toy.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Preferred constrained NSGA-II population size.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Constraint feasibility tolerance.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="constrained_nsga2_stress1",
        help="Fresh artifact suffix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Existing stress JSON artifact to use for skip-completed runner-level resume.",
    )
    parser.add_argument(
        "--resume-mode",
        default="skip-completed",
        choices=("skip-completed",),
        help="Runner-level resume mode.",
    )
    parser.add_argument(
        "--resume-report-suffix",
        default=None,
        help="Optional suffix for the runner-level resume report.",
    )
    parser.add_argument(
        "--row-execution-backend",
        default="serial",
        choices=("serial", "thread", "process"),
        help="Explicit opt-in row execution backend.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker count for thread/process row execution.",
    )
    parser.add_argument(
        "--parallel-fail-fast",
        action="store_true",
        help="Stop row execution after the first parallel row failure when possible.",
    )
    parser.add_argument(
        "--parallel-timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for parallel row execution.",
    )
    parser.add_argument(
        "--allow-process-backend",
        action="store_true",
        help="Required safety opt-in for process backend.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dimensions = {
        "constrained_zdt_box_toy": args.zdt_dimension,
        "constrained_dtlz_box_toy": args.dtlz_dimension,
    }
    config = ConstrainedNSGA2StressConfig(
        problems=args.problems,
        dimensions=dimensions,
        seeds=args.seeds,
        budgets=args.budgets,
        strategies=args.strategies,
        population_size=args.population_size,
        tolerance=args.tolerance,
        artifact_suffix=args.artifact_suffix,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        resume_mode=args.resume_mode,
        resume_report_suffix=args.resume_report_suffix,
        row_execution_backend=args.row_execution_backend,
        row_execution_workers=args.workers,
        parallel_fail_fast=args.parallel_fail_fast,
        parallel_timeout_seconds=args.parallel_timeout,
        allow_process_backend=args.allow_process_backend,
    )
    result = run_constrained_nsga2_stress(config)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "artifact_suffix": result["command_metadata"]["artifact_suffix"],
                "artifacts": result["artifacts"],
                "failure_count": len(result["failures"]),
                "warning_count": len(result["warnings"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
