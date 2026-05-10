from __future__ import annotations

import argparse
import json
from pathlib import Path

from ga_lab.local_stress_refresh import discover_latest_study_dirs
from ga_lab.local_stress_target_reduction import build_stress_target_reduction_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the future optimization target registry from the latest stress-target reduction outputs."
    )
    parser.add_argument(
        "--previous-registry",
        default="outputs/local_studies/future_optimization_targets.json",
        help="Existing future target registry JSON to update in place.",
    )
    parser.add_argument(
        "--tsp-study-name",
        default="tsp_stress_target_reduction_confirm",
        help="Study name used to resolve the latest TSP reduction confirm directory.",
    )
    parser.add_argument(
        "--zdt1-study-name",
        default="zdt1_stress_target_reduction_confirm",
        help="Study name used to resolve the latest ZDT1 reduction confirm directory.",
    )
    parser.add_argument(
        "--search-root",
        default="outputs/local_studies",
        help="Root directory used to resolve study names.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/local_studies",
        help="Directory where refreshed target registry artifacts should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    search_root = Path(args.search_root).resolve()
    tsp_dir, zdt1_dir = discover_latest_study_dirs(
        search_root,
        [args.tsp_study_name, args.zdt1_study_name],
    )
    payload = build_stress_target_reduction_registry(
        previous_registry_path=Path(args.previous_registry).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        tsp_study_dir=tsp_dir,
        zdt1_study_dir=zdt1_dir,
        include_knapsack_freeze=True,
        include_onemax_freeze=True,
    )
    payload["source_study_dirs"] = [str(tsp_dir), str(zdt1_dir)]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
