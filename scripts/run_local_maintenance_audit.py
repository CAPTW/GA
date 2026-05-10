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

from ga_lab.local_maintenance_audit import (
    build_local_maintenance_audit,
    build_local_reopen_criteria_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the no-op local maintenance audit for the frozen cycle-1 baseline."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory holding baseline, candidate, reopen, and maintenance artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    reopen = build_local_reopen_criteria_check(
        project_root=PROJECT_ROOT,
        output_json_path=artifacts_dir / "local_reopen_criteria_check.json",
        output_md_path=artifacts_dir / "local_reopen_criteria_check.md",
    )
    audit = build_local_maintenance_audit(
        project_root=PROJECT_ROOT,
        reopen_criteria_check_path=artifacts_dir / "local_reopen_criteria_check.json",
        output_json_path=artifacts_dir / "local_maintenance_audit.json",
        output_md_path=artifacts_dir / "local_maintenance_audit.md",
    )
    print(
        json.dumps(
            {
                "reopen_criteria_check": reopen["status"],
                "maintenance_audit": audit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
