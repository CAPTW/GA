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

from ga_lab.local_experiments import run_local_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small local parameter sweep from a local study manifest."
    )
    parser.add_argument(
        "--study",
        required=True,
        help="Study name or path, for example onemax_mutation_study",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/local_studies",
        help="Directory where local study bundles will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_local_study(args.study, output_root=args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
