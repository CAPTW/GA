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

from ga_lab.experiment.constrained_equality_tolerance_stress import (  # noqa: E402
    ConstrainedEqualityToleranceStressConfig,
    run_constrained_equality_tolerance_stress,
)


def _parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return items


def _parse_budgets(value: str) -> list[int]:
    try:
        budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from exc
    if not budgets:
        raise argparse.ArgumentTypeError("budgets cannot be empty")
    return budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run constrained equality tolerance stress validation."
    )
    parser.add_argument("--problem", default="constrained_equality_plane_quadratic")
    parser.add_argument("--variants", type=_parse_csv, default=["loose", "default", "strict"])
    parser.add_argument("--dimension", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=_parse_budgets, default=[300, 1000])
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--artifact-suffix", default="equality_tolerance_stress1")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = run_constrained_equality_tolerance_stress(
        ConstrainedEqualityToleranceStressConfig(
            problem=args.problem,
            variants=tuple(args.variants),
            dimension=args.dimension,
            seeds=args.seeds,
            budgets=tuple(args.budgets),
            population_size=args.population_size,
            artifact_suffix=args.artifact_suffix,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
