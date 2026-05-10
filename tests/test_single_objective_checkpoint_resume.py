from __future__ import annotations

import random

import pytest

from ga_lab.algorithms.single_objective import run_single_objective_ga
from ga_lab.config import GAConfig
from ga_lab.experiment.algorithm_checkpoint import CheckpointConfig, load_checkpoint
from ga_lab.factory import build_runtime_context


def _config(**overrides) -> GAConfig:
    payload = {
        "run_name": "single_objective_checkpoint",
        "problem": "onemax",
        "population_size": 8,
        "genome_length": 10,
        "generations": 5,
        "crossover_rate": 0.8,
        "mutation_rate": 0.05,
        "elitism": 1,
        "tournament_size": 3,
        "seed": 31,
        "maximize": True,
        "target_fitness": None,
        "log_every": 1,
    }
    payload.update(overrides)
    return GAConfig(**payload)


def _run(config: GAConfig, *, checkpoint_config: CheckpointConfig | None = None):
    runtime = build_runtime_context(config)
    return run_single_objective_ga(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(config.seed),
        checkpoint_config=checkpoint_config,
    )


def test_checkpoint_disabled_default_output_remains_unchanged(tmp_path) -> None:
    config = _config()

    default_summary, default_history = _run(config)
    disabled_summary, disabled_history = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=False,
            output_dir=tmp_path,
            run_id="disabled",
        ),
    )

    assert disabled_summary == default_summary
    assert disabled_history == default_history
    assert "checkpointing" not in disabled_summary


def test_checkpoint_enabled_writes_checkpoint_file(tmp_path) -> None:
    config = _config(generations=3)

    summary, _history = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="write_test",
            interval_generations=1,
            write_latest=True,
        ),
    )

    checkpoint_path = tmp_path / "write_test" / "checkpoint_gen_0.json"
    latest_path = tmp_path / "write_test" / "latest.json"
    assert checkpoint_path.exists()
    assert latest_path.exists()
    assert summary["checkpointing"]["enabled"] is True
    assert summary["checkpointing"]["last_checkpoint_path"].endswith("checkpoint_gen_2.json")


def test_resume_from_checkpoint_reaches_uninterrupted_result(tmp_path) -> None:
    config = _config(generations=5)
    uninterrupted_summary, uninterrupted_history = _run(config)

    _stopped_summary, _stopped_history = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="interrupted",
            interval_generations=1,
            stop_after_checkpoint_generation=1,
        ),
    )
    checkpoint_path = tmp_path / "interrupted" / "checkpoint_gen_1.json"

    resumed_summary, resumed_history = _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="resumed",
            interval_generations=1,
            resume_from=checkpoint_path,
        ),
    )

    assert resumed_summary["best_fitness"] == uninterrupted_summary["best_fitness"]
    assert resumed_summary["best_genome"] == uninterrupted_summary["best_genome"]
    assert resumed_summary["actual_evaluations_used"] == uninterrupted_summary["actual_evaluations_used"]
    assert resumed_history == uninterrupted_history
    assert resumed_summary["checkpointing"]["resumed"] is True


def test_resume_with_config_mismatch_raises(tmp_path) -> None:
    config = _config()
    _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="mismatch_source",
            interval_generations=1,
            stop_after_checkpoint_generation=1,
        ),
    )

    with pytest.raises(ValueError, match="compatibility"):
        _run(
            _config(mutation_rate=0.2),
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="mismatch_resume",
                resume_from=tmp_path / "mismatch_source" / "checkpoint_gen_1.json",
            ),
        )


def test_resume_with_smaller_budget_raises(tmp_path) -> None:
    config = _config(generations=5)
    _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="budget_source",
            interval_generations=1,
            stop_after_checkpoint_generation=2,
        ),
    )

    with pytest.raises(ValueError, match="budget"):
        _run(
            _config(generations=0),
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="budget_resume",
                resume_from=tmp_path / "budget_source" / "checkpoint_gen_2.json",
            ),
        )


class _NonFiniteProblem:
    name = "onemax"

    def fitness(self, _genome):
        return float("inf")


def test_non_finite_fitness_after_resume_still_raises(tmp_path) -> None:
    config = _config()
    _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="finite_source",
            interval_generations=1,
            stop_after_checkpoint_generation=1,
        ),
    )

    runtime = build_runtime_context(config)
    with pytest.raises(ValueError, match="Non-finite fitness"):
        run_single_objective_ga(
            config=config,
            problem=_NonFiniteProblem(),
            selection_fn=runtime.operators.selection_fn,
            crossover_fn=runtime.operators.crossover_fn,
            mutation_fn=runtime.operators.mutation_fn,
            init_fn=runtime.operators.init_fn,
            rng=random.Random(config.seed),
            checkpoint_config=CheckpointConfig(
                enabled=True,
                output_dir=tmp_path,
                run_id="non_finite_resume",
                resume_from=tmp_path / "finite_source" / "checkpoint_gen_1.json",
            ),
        )


def test_loaded_checkpoint_contains_population_and_budget_state(tmp_path) -> None:
    config = _config(generations=3)
    _run(
        config,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="loadable",
            interval_generations=1,
            stop_after_checkpoint_generation=1,
        ),
    )

    checkpoint = load_checkpoint(tmp_path / "loadable" / "checkpoint_gen_1.json")

    assert checkpoint.population.population_size == config.population_size
    assert checkpoint.budget_state.actual_evaluations == checkpoint.metadata.actual_evaluations
    assert checkpoint.resume_generation == 2
