from __future__ import annotations

import argparse
from pathlib import Path

from ga_lab.experiment.nsga2_checkpoint_stress import (
    Nsga2CheckpointStressConfig,
    run_nsga2_checkpoint_stress,
)


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return budgets


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NSGA-II checkpoint stress")
    parser.add_argument("--problem", default="zdt1")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=_parse_budgets, default=[300, 760])
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--interruption-policy", default="midpoint")
    parser.add_argument("--artifact-suffix", default="nsga2_checkpoint_stress1")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--run-compatibility-negative-tests", action="store_true")
    args = parser.parse_args()

    artifact = run_nsga2_checkpoint_stress(
        Nsga2CheckpointStressConfig(
            problem=args.problem,
            seeds=args.seeds,
            budgets=args.budgets,
            checkpoint_interval=args.checkpoint_interval,
            interruption_policy=args.interruption_policy,
            artifact_suffix=args.artifact_suffix,
            output_dir=Path(args.output_dir),
            run_compatibility_negative_tests=args.run_compatibility_negative_tests,
        )
    )
    print(artifact["artifact_paths"]["json"])
    return 0 if not artifact["aggregate_summary"]["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
