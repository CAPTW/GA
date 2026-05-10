# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_ops.reporting import generate_weekly_report
from ga_ops.settings import OpsSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a weekly ops report from the metadata DB."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional override for the ops SQLite database path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for the generated report bundle",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="How many days of data to summarize",
    )
    parser.add_argument(
        "--actor",
        default="weekly_report",
        help="Actor name recorded in ops audit logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = OpsSettings.from_env(project_root=PROJECT_ROOT)
    if args.db_path is not None:
        settings.db_path = Path(args.db_path)
    result = generate_weekly_report(
        settings=settings,
        output_dir=args.output_dir,
        lookback_days=args.lookback_days,
        actor=args.actor,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
