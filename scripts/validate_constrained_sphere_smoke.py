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

from ga_lab.experiment.constrained_sphere_smoke import SmokeConfig, run_constrained_sphere_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained sphere smoke validation.")
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--constraint-budget", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--artifact-suffix", default="constrained_sphere_smoke1")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = run_constrained_sphere_smoke(
        SmokeConfig(
            dimension=args.dimension,
            seeds=args.seeds,
            budget=args.budget,
            constraint_budget=args.constraint_budget,
            tolerance=args.tolerance,
            artifact_suffix=args.artifact_suffix,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
