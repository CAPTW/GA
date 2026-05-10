from __future__ import annotations

import argparse
from pathlib import Path

from ga_lab.experiment.external_benchmark_suite import run_manifests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the external benchmark suite with matched evaluation budgets."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        help=(
            "Path to an external benchmark manifest. Defaults to "
            "configs/benchmarks/external_suite_manifest.json."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "benchmark_summary"),
        help="Directory for summary outputs.",
    )
    parser.add_argument(
        "--summary-stem",
        default="external_benchmark_summary",
        help="Basename for summary output files.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(PROJECT_ROOT / "benchmarks"),
        help="Benchmark cache root. Defaults to benchmarks/ under the project root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = args.manifests or [
        str(PROJECT_ROOT / "configs" / "benchmarks" / "external_suite_manifest.json")
    ]
    run_manifests(
        manifests,
        output_root=args.output_root,
        summary_stem=args.summary_stem,
        cache_root=args.cache_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
