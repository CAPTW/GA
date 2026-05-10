# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.experiment.comparison import collect_comparison_rows
from ga_lab.experiment.retention import (
    apply_retention_plan,
    build_retention_plan,
    write_retention_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or apply retention rules for experiment results."
    )
    parser.add_argument("--results-dir", default="outputs", help="Directory containing run outputs")
    parser.add_argument(
        "--results-name",
        default=None,
        help="Optional label for the retention plan",
    )
    parser.add_argument(
        "--mode",
        choices=("history-only", "run-dirs"),
        default="history-only",
        help="Whether to prune only history.csv files or delete whole run directories",
    )
    parser.add_argument(
        "--keep-best-runs",
        type=int,
        default=2,
        help="How many top runs per label to keep fully",
    )
    parser.add_argument(
        "--keep-latest-runs",
        type=int,
        default=1,
        help="How many latest runs per label to keep fully",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the retention plan after writing it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    rows = collect_comparison_rows(results_dir)
    if not rows:
        raise SystemExit("No summary.json files found.")

    results_name = args.results_name or results_dir.name
    plan = build_retention_plan(
        results_name,
        rows,
        mode=args.mode,
        keep_best_runs=args.keep_best_runs,
        keep_latest_runs=args.keep_latest_runs,
    )
    bundle_paths = write_retention_bundle(results_dir, plan)
    applied_counts = apply_retention_plan(plan, apply=args.apply)

    print(f"Retention plan: {bundle_paths['retention_plan']}")
    print(f"Retention report: {bundle_paths['retention_markdown']}")
    if args.apply:
        print(f"Pruned history.csv files: {applied_counts['pruned_history_count']}")
        print(f"Deleted run directories: {applied_counts['deleted_run_dir_count']}")
    else:
        print("Dry run only. Re-run with --apply to prune artifacts.")


if __name__ == "__main__":
    main()
