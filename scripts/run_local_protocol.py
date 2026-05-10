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

from ga_lab.local_protocol import run_local_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen local operating protocol helper for TSP, ZDT1, knapsack, or OneMax."
    )
    parser.add_argument(
        "--problem",
        required=True,
        choices=["tsp", "zdt1", "knapsack", "onemax"],
        help="Problem family to read through the frozen local protocol.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="Protocol mode, for example explore, compare, final, sanity, or control.",
    )
    parser.add_argument(
        "--case-group",
        default=None,
        help="Optional case-group hint such as rescue_target, anti_case, or tight_capacity_small.",
    )
    parser.add_argument(
        "--quality-sensitive",
        action="store_true",
        help="For TSP compare, skip the fast rescue path and recommend the Q final directly.",
    )
    parser.add_argument(
        "--anti-case-suspected",
        action="store_true",
        help="For TSP compare/final, treat the run as anti-case aware and move directly to Q final.",
    )
    parser.add_argument(
        "--final-safety",
        action="store_true",
        help="For ZDT1 compare, move directly to the Q final-safety path.",
    )
    parser.add_argument(
        "--borderline",
        action="store_true",
        help="For knapsack sanity, use the longer 5-seed borderline path.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="Optional manual seed-count override for the protocol helper.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the recommended local study slice and attach output paths to the decision bundle.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the longer rationale instead of the shortened one-line explanation.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/local_protocols",
        help="Directory where protocol decision bundles will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_local_protocol(
        problem=args.problem,
        mode=args.mode,
        output_root=args.output_root,
        case_group=args.case_group,
        quality_sensitive=args.quality_sensitive,
        anti_case_suspected=args.anti_case_suspected,
        final_safety=args.final_safety,
        borderline=args.borderline,
        seed_count=args.seed_count,
        execute=args.execute,
        explain=args.explain,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
