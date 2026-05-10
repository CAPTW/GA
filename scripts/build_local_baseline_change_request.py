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

from ga_lab.local_change_request import build_local_baseline_change_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manual local baseline change-request pack from a candidate report."
    )
    parser.add_argument(
        "--candidate-report",
        required=True,
        help="Path to outputs/local_candidates/.../candidate_report.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/local_change_requests",
        help="Directory where change-request bundles will be written.",
    )
    parser.add_argument(
        "--force-draft",
        action="store_true",
        help="Allow a draft pack even when the candidate has not cleared the local promotion gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_local_baseline_change_request(
        args.candidate_report,
        project_root=PROJECT_ROOT,
        output_root=args.output_root,
        force_draft=args.force_draft,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
