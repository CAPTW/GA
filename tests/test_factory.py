from __future__ import annotations

import random

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.algorithms.single_objective import run_single_objective_ga
from ga_lab.config import GAConfig
from ga_lab.factory import (
    build_algorithm_fn,
    build_operator_bundle,
    build_problem_from_name,
    build_problem_instance,
    build_runtime_context,
)


def test_factory_builds_single_objective_components() -> None:
    config = GAConfig(
        run_name="factory_onemax",
        problem="onemax",
        population_size=12,
        genome_length=8,
        generations=4,
        crossover_rate=0.9,
        mutation_rate=0.05,
        elitism=1,
        tournament_size=3,
    )

    problem = build_problem_instance(config)
    operators = build_operator_bundle(config)
    algorithm_fn = build_algorithm_fn(config)

    assert problem.name == "onemax"
    assert algorithm_fn is run_single_objective_ga
    genome = operators.init_fn(random.Random(3), config.genome_length)
    assert len(genome) == config.genome_length


def test_factory_builds_nsga2_algorithm() -> None:
    config = GAConfig(
        run_name="factory_nsga2",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=12,
        genome_length=6,
        generations=4,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        maximize=False,
        objective_directions=[False, False],
    )

    assert build_algorithm_fn(config) is run_nsga2


def test_factory_builds_problem_from_registry() -> None:
    problem = build_problem_from_name(
        "tsp",
        {
            "num_cities": 4,
            "coordinates": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
        },
    )

    assert problem.name == "tsp"


def test_runtime_context_rejects_incompatible_problem_representation() -> None:
    config = GAConfig(
        run_name="bad_onemax_real",
        problem="onemax",
        representation="real",
        selection="rank",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=12,
        genome_length=8,
        generations=4,
        crossover_rate=0.9,
        mutation_rate=0.1,
    )

    try:
        build_runtime_context(config)
    except ValueError as exc:
        assert "requires one of representations" in str(exc)
    else:
        raise AssertionError(
            "build_runtime_context should reject incompatible problem/representation"
        )


def test_runtime_context_rejects_problem_genome_length_mismatch() -> None:
    config = GAConfig(
        run_name="bad_tsp_length",
        problem="tsp",
        representation="permutation",
        selection="tournament",
        crossover="order",
        mutation="swap",
        population_size=12,
        genome_length=5,
        generations=4,
        crossover_rate=0.9,
        mutation_rate=0.1,
        problem_options={"num_cities": 4},
    )

    try:
        build_runtime_context(config)
    except ValueError as exc:
        assert "requires genome_length=4" in str(exc)
    else:
        raise AssertionError("build_runtime_context should reject genome length mismatch")
