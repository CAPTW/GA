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

from ga_lab.local_experiments import run_local_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick local GA experiment from a preset, demo, or config."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", help="Builtin preset name, for example onemax_small")
    source.add_argument("--demo", help="Builtin demo name, for example baseline or nsga2")
    source.add_argument("--config", help="Path to a JSON config file")
    parser.add_argument(
        "--output-root",
        default="outputs/local_runs",
        help="Directory where local run bundles will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--run-name", default=None, help="Optional run_name override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_local_experiment(
        preset=args.preset,
        demo=args.demo,
        config_path=args.config,
        output_root=args.output_root,
        seed=args.seed,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
