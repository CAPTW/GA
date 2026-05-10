# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_lab.api import recommend_preset as _recommend_public
from ga_lab.consumer_cli import recommend_preset_main


def recommend(problem: str, size: int, priority: str = "default") -> dict[str, object]:
    return _recommend_public(problem, size, priority, format="dict")

__all__ = ["main", "recommend"]


def main() -> int:
    return recommend_preset_main()


if __name__ == "__main__":
    raise SystemExit(main())
