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

from ga_lab.experiment.constrained_ga_smoke import (
    ConstrainedGASmokeConfig,
    DEFAULT_SMOKE_TOLERANCE,
    EQUALITY_PLANE_SMOKE_TOLERANCE,
    run_constrained_ga_smoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained GA smoke validation.")
    parser.add_argument(
        "--problem",
        choices=[
            "constrained_sphere",
            "constrained_box_quadratic",
            "constrained_equality_plane_quadratic",
        ],
        default="constrained_sphere",
    )
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--constraint-budget", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--artifact-suffix", default="constrained_ga_smoke1")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tolerance = args.tolerance
    if tolerance is None:
        tolerance = (
            EQUALITY_PLANE_SMOKE_TOLERANCE
            if args.problem == "constrained_equality_plane_quadratic"
            else DEFAULT_SMOKE_TOLERANCE
        )
    artifact = run_constrained_ga_smoke(
        ConstrainedGASmokeConfig(
            problem=args.problem,
            dimension=args.dimension,
            seeds=args.seeds,
            budget=args.budget,
            constraint_budget=args.constraint_budget,
            tolerance=tolerance,
            population_size=args.population_size,
            artifact_suffix=args.artifact_suffix,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
