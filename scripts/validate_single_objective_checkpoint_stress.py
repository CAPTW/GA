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

from ga_lab.experiment.single_objective_checkpoint_stress import (
    SingleObjectiveCheckpointStressConfig,
    run_single_objective_checkpoint_stress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-objective GA checkpoint/resume stress validation."
    )
    parser.add_argument("--problem", choices=("onemax",), default="onemax")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--genome-length", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--interrupt-generation", type=int, default=None)
    parser.add_argument("--artifact-suffix", default="checkpoint_stress1")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--run-compatibility-negative-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = run_single_objective_checkpoint_stress(
        SingleObjectiveCheckpointStressConfig(
            problem=args.problem,
            seeds=args.seeds,
            generations=args.generations,
            population_size=args.population_size,
            genome_length=args.genome_length,
            checkpoint_interval=args.checkpoint_interval,
            interrupt_generation=args.interrupt_generation,
            artifact_suffix=args.artifact_suffix,
            output_dir=args.output_dir,
            run_compatibility_negative_tests=args.run_compatibility_negative_tests,
        )
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
