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

from ga_lab.experiment.constrained_box_quadratic_tightness import (  # noqa: E402
    ConstrainedBoxQuadraticTightnessConfig,
    run_constrained_box_quadratic_tightness_stress,
)


def _parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("variants cannot be empty")
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run constrained_box_quadratic tightness stress validation."
    )
    parser.add_argument("--variants", type=_parse_variants, default=["easy", "default", "strict"])
    parser.add_argument("--dimension", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--artifact-suffix", default="tightness_stress1")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = run_constrained_box_quadratic_tightness_stress(
        ConstrainedBoxQuadraticTightnessConfig(
            variants=tuple(args.variants),
            dimension=args.dimension,
            seeds=args.seeds,
            budget=args.budget,
            tolerance=args.tolerance,
            population_size=args.population_size,
            artifact_suffix=args.artifact_suffix,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
