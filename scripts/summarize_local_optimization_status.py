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

from ga_lab.local_optimization_status import build_local_optimization_closeout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local optimization closeout status, reopen criteria, and candidate backlog summary."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for local_optimization_status.*, local_reopen_criteria.json, and local_candidate_backlog_closeout.*",
    )
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory for local_reopen_criteria.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    docs_dir = (PROJECT_ROOT / args.docs_dir).resolve()
    result = build_local_optimization_closeout(
        project_root=PROJECT_ROOT,
        status_json_path=artifacts_dir / "local_optimization_status.json",
        status_md_path=artifacts_dir / "local_optimization_status.md",
        reopen_json_path=artifacts_dir / "local_reopen_criteria.json",
        reopen_md_path=docs_dir / "local_reopen_criteria.md",
        backlog_json_path=artifacts_dir / "local_candidate_backlog_closeout.json",
        backlog_md_path=artifacts_dir / "local_candidate_backlog_closeout.md",
    )
    print(json.dumps(result["status"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
