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

from ga_lab.tsp_policy_router import run_tsp_router_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze TSP hard-case study outputs and derive a simple "
            "instance-aware local router."
        )
    )
    parser.add_argument(
        "--train-study-dir",
        required=True,
        help="Path to the TSP router train study output directory.",
    )
    parser.add_argument(
        "--holdout-study-dir",
        required=True,
        help="Path to the TSP router holdout study output directory.",
    )
    parser.add_argument(
        "--budget-study-dir",
        help="Optional path to a small TSP budget-band study output directory.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/local_studies",
        help="Directory where router analysis bundles will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_tsp_router_analysis(
        train_study_dir=args.train_study_dir,
        holdout_study_dir=args.holdout_study_dir,
        budget_study_dir=args.budget_study_dir,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
