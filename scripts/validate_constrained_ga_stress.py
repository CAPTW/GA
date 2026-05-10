# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.experiment.constrained_ga_stress import (  # noqa: E402
    ConstrainedGAStressConfig,
    run_constrained_ga_stress,
)


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("budgets cannot be empty")
    return budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained GA seed/budget stress validation.")
    parser.add_argument(
        "--problem",
        choices=("constrained_sphere", "constrained_box_quadratic"),
        default="constrained_sphere",
    )
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budgets", type=_parse_budgets, default=[300, 1000])
    parser.add_argument("--constraint-budget", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--artifact-suffix", default="constrained_ga_stress1")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-mode", default="skip-completed", choices=("skip-completed",))
    parser.add_argument("--resume-report-suffix", default=None)
    parser.add_argument(
        "--row-execution-backend",
        default="serial",
        choices=("serial", "thread", "process"),
        help="Runner-level row execution backend. Serial is the default.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Worker count for row execution.")
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
        help="Allow guarded process backend execution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = run_constrained_ga_stress(
        ConstrainedGAStressConfig(
            problem=args.problem,
            dimension=args.dimension,
            seeds=args.seeds,
            budgets=tuple(args.budgets),
            constraint_budget=args.constraint_budget,
            tolerance=args.tolerance,
            population_size=args.population_size,
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
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
