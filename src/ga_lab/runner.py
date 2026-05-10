from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ga_lab.config import GAConfig
from ga_lab.experiment.trackers import build_tracker
from ga_lab.experiment.tracking import write_history_csv, write_json
from ga_lab.factory import (
    build_runtime_context,
)
from ga_lab.governance.run_metadata import build_run_metadata
from ga_lab.utils.seed import make_rng

RUN_SUMMARY_SCHEMA_VERSION = 1


@dataclass(slots=True)
class RunResult:
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    output_dir: Path


def run_experiment(config: GAConfig, output_root: str | Path = "outputs") -> RunResult:
    config.validate()
    runtime = build_runtime_context(config)
    rng = make_rng(config.seed)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / f"{timestamp}_{config.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = build_tracker(config.tracking, run_name=config.run_name)
    tracker.log_config(config.to_dict())

    try:
        start = time.perf_counter()
        algorithm_summary, history = runtime.algorithm_fn(
            config=config,
            problem=runtime.problem,
            selection_fn=runtime.operators.selection_fn,
            crossover_fn=runtime.operators.crossover_fn,
            mutation_fn=runtime.operators.mutation_fn,
            init_fn=runtime.operators.init_fn,
            rng=rng,
        )

        elapsed = time.perf_counter() - start
        summary = {
            "summary_schema_version": RUN_SUMMARY_SCHEMA_VERSION,
            "run_name": config.run_name,
            "problem": config.problem,
            "seed": config.seed,
            "algorithm": config.algorithm,
            "selection": config.selection,
            "crossover": config.crossover,
            "mutation": config.mutation,
            "representation": config.representation,
            "population_size": config.population_size,
            "genome_length": config.genome_length,
            "generations": config.generations,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
            "elitism": config.elitism,
            "tournament_size": config.tournament_size,
            "runtime_seconds": elapsed,
            "log_every": config.log_every,
        }
        summary.update(algorithm_summary)
        run_metadata = build_run_metadata(
            project_root=Path(__file__).resolve().parents[2],
            summary_stem="summary",
            output_root=output_dir,
            manifest_paths=(),
            include_packages=False,
            extra={
                "run_name": config.run_name,
                "problem": config.problem,
                "seed": config.seed,
                "algorithm": config.algorithm,
            },
        )

        write_json(output_dir / "config.json", config.to_dict())
        write_json(output_dir / "config.canonical.json", config.to_canonical_dict())
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "run_metadata.json", run_metadata)
        write_history_csv(output_dir / "history.csv", history)
        tracker.log_summary(summary)
        tracker.log_artifacts(output_dir)
        return RunResult(summary=summary, history=history, output_dir=output_dir)
    finally:
        tracker.finish()
