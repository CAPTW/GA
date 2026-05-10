from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.local_failure_trace import build_failure_hypothesis_registry
from ga_lab.local_stress_refresh import discover_latest_study_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the failure-hypothesis registry from the latest failure-trace study outputs."
    )
    parser.add_argument(
        "--previous-registry",
        default="outputs/local_studies/future_optimization_targets.json",
        help="Existing future-target registry JSON to refresh in place.",
    )
    parser.add_argument(
        "--tsp-study-name",
        default="tsp_failure_trace_suite",
        help="Study name used to resolve the latest TSP failure-trace directory.",
    )
    parser.add_argument(
        "--zdt1-study-name",
        default="zdt1_failure_trace_suite",
        help="Study name used to resolve the latest ZDT1 failure-trace directory.",
    )
    parser.add_argument(
        "--zdt1-spread-study-name",
        default="",
        help="Optional ZDT1 spread-target study name to merge into the registry refresh.",
    )
    parser.add_argument(
        "--zdt1-joint-study-name",
        default="",
        help="Optional ZDT1 joint-target study name to merge into the registry refresh.",
    )
    parser.add_argument(
        "--search-root",
        default="outputs/local_studies",
        help="Root directory used to resolve study names.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/local_studies",
        help="Directory where refreshed registry artifacts should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    search_root = Path(args.search_root).resolve()
    tsp_dir = discover_latest_study_dirs(search_root, [args.tsp_study_name])[0]
    zdt1_dir: Path
    additional_zdt1_dirs: list[Path] = []
    if args.zdt1_spread_study_name or args.zdt1_joint_study_name:
        spread_name = args.zdt1_spread_study_name or args.zdt1_study_name
        zdt1_dir = discover_latest_study_dirs(search_root, [spread_name])[0]
        if args.zdt1_joint_study_name:
            additional_zdt1_dirs.append(
                discover_latest_study_dirs(search_root, [args.zdt1_joint_study_name])[0]
            )
    else:
        zdt1_dir = discover_latest_study_dirs(search_root, [args.zdt1_study_name])[0]
    payload = build_failure_hypothesis_registry(
        previous_registry_path=Path(args.previous_registry).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        tsp_study_dir=tsp_dir,
        zdt1_study_dir=zdt1_dir,
        additional_zdt1_study_dirs=additional_zdt1_dirs,
        include_knapsack_freeze=True,
        include_onemax_freeze=True,
    )
    payload["source_study_dirs"] = [str(tsp_dir), str(zdt1_dir), *[str(path) for path in additional_zdt1_dirs]]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
