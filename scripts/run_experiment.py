# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.consumer_cli import run_experiment_main


def main() -> int:
    return run_experiment_main()


if __name__ == "__main__":
    raise SystemExit(main())
