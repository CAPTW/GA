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

from ga_lab.local_baseline import build_local_baseline_snapshot, check_local_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the local optimization baseline snapshot."
    )
    parser.add_argument(
        "--write-snapshot",
        action="store_true",
        help="Refresh artifacts/local_baseline_snapshot.{json,md} before checking.",
    )
    parser.add_argument(
        "--snapshot",
        default="artifacts/local_baseline_snapshot.json",
        help="Path to the baseline snapshot JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for local_baseline_check.{json,md}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_path = (PROJECT_ROOT / args.snapshot).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_json = output_dir / "local_baseline_check.json"
    output_md = output_dir / "local_baseline_check.md"

    if args.write_snapshot:
        build_local_baseline_snapshot(
            project_root=PROJECT_ROOT,
            snapshot_path=snapshot_path,
            markdown_path=snapshot_path.with_suffix(".md"),
        )

    report = check_local_baseline(
        project_root=PROJECT_ROOT,
        snapshot_path=snapshot_path,
        output_json_path=output_json,
        output_md_path=output_md,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
