from __future__ import annotations

import argparse
import json
from pathlib import Path

from ga_lab.benchmarks.external import (
    BENCHMARK_INSTANCES,
    ensure_benchmark_files,
    write_metadata_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch external benchmark cache files for the canonical suite."
    )
    parser.add_argument(
        "--instance",
        action="append",
        dest="instances",
        help=(
            "Benchmark instance id to fetch. May be repeated. "
            "Defaults to all known file-backed instances."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without downloading files.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format for the fetch plan.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(PROJECT_ROOT / "benchmarks"),
        help="Benchmark cache root. Defaults to benchmarks/ under the project root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = Path(args.cache_root).resolve()
    selected = args.instances or sorted(BENCHMARK_INSTANCES)
    plan = ensure_benchmark_files(selected, cache_root=cache_root, dry_run=args.dry_run)
    metadata_path = write_metadata_file(cache_root / "metadata.json")
    payload = {
        "cache_root": str(cache_root),
        "metadata_path": str(metadata_path),
        "instances": selected,
        "plan": plan,
    }
    if args.format == "text":
        for row in plan:
            print(f"{row['instance_id']}: {row['action']} -> {row['cache_path']}")
        print(f"metadata: {metadata_path}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
