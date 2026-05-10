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

from ga_lab.local_experiments import rerender_local_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-render local study plots from an existing study directory."
    )
    parser.add_argument("--study-dir", required=True, help="Existing local study directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = rerender_local_study(args.study_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
