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

from ga_lab.local_candidate_ledger import build_candidate_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the local candidate ledger and summary from candidate reports."
    )
    parser.add_argument(
        "--reports-root",
        default="outputs/local_candidates",
        help="Directory that contains candidate report bundles.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory where the candidate ledger and summary artifacts will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    result = build_candidate_ledger(
        project_root=PROJECT_ROOT,
        reports_root=args.reports_root,
        ledger_json_path=artifacts_dir / "local_candidate_ledger.json",
        ledger_csv_path=artifacts_dir / "local_candidate_ledger.csv",
        ledger_md_path=artifacts_dir / "local_candidate_ledger.md",
        summary_json_path=artifacts_dir / "local_candidate_summary.json",
        summary_md_path=artifacts_dir / "local_candidate_summary.md",
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
