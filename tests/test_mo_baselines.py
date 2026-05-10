from __future__ import annotations

import pytest

from ga_lab.config import GAConfig
from ga_lab.experiment.mo_baselines import (
    evaluate_objective_vector,
    run_random_pareto_archive,
    run_weighted_sum_random_archive,
)


def _zdt1_config() -> GAConfig:
    return GAConfig(
        run_name="test_mo_baseline",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=20,
        genome_length=6,
        generations=12,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elitism=1,
        tournament_size=2,
        seed=21,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"hypervolume_reference_point": [1.05, 10.5]},
    )


def test_random_pareto_archive_is_reproducible() -> None:
    config = _zdt1_config()
    left = run_random_pareto_archive(config, seed=123, budget=32)
    right = run_random_pareto_archive(config, seed=123, budget=32)

    assert left.evaluations == 32
    assert right.evaluations == 32
    assert left.nondominated_objective_vectors == right.nondominated_objective_vectors


def test_random_pareto_archive_respects_budget() -> None:
    config = _zdt1_config()
    result = run_random_pareto_archive(config, seed=456, budget=25)

    assert result.evaluations == 25
    assert result.error_message is None
    assert result.nondominated_count >= 1


def test_weighted_sum_random_archive_respects_budget() -> None:
    config = _zdt1_config()
    result = run_weighted_sum_random_archive(config, seed=789, budget=40)

    assert result.evaluations == 40
    assert result.error_message is None
    assert result.metadata["evaluated_candidates"] == 40


def test_evaluate_objective_vector_rejects_non_finite_values() -> None:
    class _BadProblem:
        def fitness(self, genome):
            return (float("inf"), 0.0)

    with pytest.raises(ValueError, match="Non-finite fitness detected"):
        evaluate_objective_vector(
            _BadProblem(),
            [0.1, 0.2],
            problem_name="bad_problem",
            location="test",
            evaluation_index=0,
        )
