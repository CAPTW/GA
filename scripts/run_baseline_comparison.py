# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_lab.experiment.budget_baseline_comparison import run_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run budget-matched preset-vs-baseline comparisons."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Comparison manifest path. Pass multiple times to merge suites into one summary.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark_summary",
        help="Directory where suite outputs and merged summaries will be written.",
    )
    parser.add_argument(
        "--summary-stem",
        default="baseline_comparison_summary",
        help="Filename stem for merged summary artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_manifests(
        args.manifest,
        output_root=args.output_root,
        summary_stem=args.summary_stem,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
