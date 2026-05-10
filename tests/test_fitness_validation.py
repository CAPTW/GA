from __future__ import annotations

import random

import pytest

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.algorithms.single_objective import run_single_objective_ga
from ga_lab.config import GAConfig
from ga_lab.problems.base import ProblemMetadata


class _NonFiniteSingleObjectiveProblem:
    name = "non_finite_single"

    def __init__(self, value: float) -> None:
        self.value = value

    def fitness(self, genome):
        return self.value

    def optimal_fitness(self, genome_length: int):
        return None

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=("bit",),
            exact_genome_length=4,
            default_objective_directions=(True,),
        )


class _FiniteSingleObjectiveProblem:
    name = "finite_single"

    def fitness(self, genome):
        return sum(genome)

    def optimal_fitness(self, genome_length: int):
        return float(genome_length)

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=("bit",),
            exact_genome_length=4,
            default_objective_directions=(True,),
        )


class _NonFiniteMultiObjectiveProblem:
    name = "non_finite_multi"

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def fitness(self, genome):
        return self.values

    def optimal_fitness(self, genome_length: int):
        return None

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=("real",),
            exact_genome_length=4,
            default_objective_directions=(False, False),
        )


def _selection_stub(population, _state, _rng):
    return population[0][:]


def _crossover_stub(parent_a, parent_b, _rng):
    return parent_a[:], parent_b[:]


def _bit_init(_rng, genome_length: int):
    return [0.0] * genome_length


def _real_init(_rng, genome_length: int):
    return [0.0] * genome_length


def _identity_mutation(genome, _rng):
    return genome[:]


def _single_objective_config(*, mutation_rate: float = 0.0) -> GAConfig:
    return GAConfig(
        run_name="fitness_validation_single",
        problem="onemax",
        population_size=4,
        genome_length=4,
        generations=1,
        crossover_rate=0.0,
        mutation_rate=mutation_rate,
        elitism=0,
        seed=11,
    )


def _multiobjective_config() -> GAConfig:
    return GAConfig(
        run_name="fitness_validation_multi",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=4,
        genome_length=4,
        generations=1,
        crossover_rate=0.0,
        mutation_rate=0.0,
        elitism=0,
        tournament_size=2,
        seed=17,
        maximize=False,
        objective_directions=[False, False],
    )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_single_objective_rejects_non_finite_fitness(invalid_value: float) -> None:
    config = _single_objective_config()

    with pytest.raises(ValueError, match="Non-finite fitness") as exc_info:
        run_single_objective_ga(
            config=config,
            problem=_NonFiniteSingleObjectiveProblem(invalid_value),
            selection_fn=_selection_stub,
            crossover_fn=_crossover_stub,
            mutation_fn=_identity_mutation,
            init_fn=_bit_init,
            rng=random.Random(config.seed),
        )
    assert "problem=non_finite_single" in str(exc_info.value)
    assert "objective_index=0" in str(exc_info.value)


def test_single_objective_allows_finite_fitness() -> None:
    config = _single_objective_config()

    summary, _history = run_single_objective_ga(
        config=config,
        problem=_FiniteSingleObjectiveProblem(),
        selection_fn=_selection_stub,
        crossover_fn=_crossover_stub,
        mutation_fn=_identity_mutation,
        init_fn=_bit_init,
        rng=random.Random(config.seed),
    )

    assert summary["best_fitness"] == 0.0


@pytest.mark.parametrize(
    "invalid_values",
    ([float("nan"), 0.0], [0.0, float("inf")], [0.0, float("-inf")]),
)
def test_nsga2_rejects_non_finite_objective_values(invalid_values: list[float]) -> None:
    config = _multiobjective_config()

    with pytest.raises(ValueError, match="Non-finite fitness") as exc_info:
        run_nsga2(
            config=config,
            problem=_NonFiniteMultiObjectiveProblem(invalid_values),
            selection_fn=_selection_stub,
            crossover_fn=_crossover_stub,
            mutation_fn=_identity_mutation,
            init_fn=_real_init,
            rng=random.Random(config.seed),
        )
    assert "problem=non_finite_multi" in str(exc_info.value)
    assert "objective_index=" in str(exc_info.value)
