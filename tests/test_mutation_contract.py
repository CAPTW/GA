from __future__ import annotations

import random

from ga_lab.algorithms.nsga2 import run_nsga2
from ga_lab.algorithms.single_objective import run_single_objective_ga
from ga_lab.config import GAConfig
from ga_lab.factory import build_runtime_context
from ga_lab.problems.base import ProblemMetadata


class _SingleObjectiveProblem:
    name = "mutation_contract_single"

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


class _MultiObjectiveProblem:
    name = "mutation_contract_multi"

    def fitness(self, genome):
        return [sum(genome), 10.0 - sum(genome)]

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


def test_single_objective_uses_injected_custom_mutation() -> None:
    config = GAConfig(
        run_name="mutation_contract_single",
        problem="onemax",
        population_size=4,
        genome_length=4,
        generations=1,
        crossover_rate=0.0,
        mutation_rate=1.0,
        elitism=0,
        seed=7,
    )
    call_rates: list[float] = []

    def custom_mutation(genome, _rng):
        call_rates.append(config.mutation_rate)
        mutated = genome[:]
        mutated[0] = 1.0
        return mutated

    summary, _history = run_single_objective_ga(
        config=config,
        problem=_SingleObjectiveProblem(),
        selection_fn=_selection_stub,
        crossover_fn=_crossover_stub,
        mutation_fn=custom_mutation,
        init_fn=_bit_init,
        rng=random.Random(config.seed),
    )

    assert call_rates
    assert all(rate == 1.0 for rate in call_rates)
    assert summary["best_fitness"] == 1.0


def test_single_objective_mutation_rate_zero_is_visible_to_injected_mutation() -> None:
    config = GAConfig(
        run_name="mutation_contract_single_zero",
        problem="onemax",
        population_size=4,
        genome_length=4,
        generations=1,
        crossover_rate=0.0,
        mutation_rate=0.0,
        elitism=0,
        seed=9,
    )
    call_rates: list[float] = []

    def custom_mutation(genome, _rng):
        call_rates.append(config.mutation_rate)
        if config.mutation_rate <= 0.0:
            return genome[:]
        mutated = genome[:]
        mutated[0] = 1.0
        return mutated

    summary, _history = run_single_objective_ga(
        config=config,
        problem=_SingleObjectiveProblem(),
        selection_fn=_selection_stub,
        crossover_fn=_crossover_stub,
        mutation_fn=custom_mutation,
        init_fn=_bit_init,
        rng=random.Random(config.seed),
    )

    assert call_rates
    assert all(rate == 0.0 for rate in call_rates)
    assert summary["best_fitness"] == 0.0


def test_single_objective_uses_factory_mutation_function() -> None:
    config = GAConfig(
        run_name="mutation_contract_factory",
        problem="onemax",
        population_size=6,
        genome_length=8,
        generations=1,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elitism=1,
        tournament_size=3,
        seed=3,
    )
    runtime = build_runtime_context(config)
    call_count = 0
    base_mutation = runtime.operators.mutation_fn

    def counted_mutation(genome, rng):
        nonlocal call_count
        call_count += 1
        return base_mutation(genome, rng)

    summary, _history = runtime.algorithm_fn(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=counted_mutation,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(config.seed),
    )

    assert call_count > 0
    assert summary["best_fitness"] >= 0.0


def test_nsga2_uses_injected_custom_mutation() -> None:
    config = GAConfig(
        run_name="mutation_contract_nsga2",
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
        mutation_rate=1.0,
        elitism=0,
        tournament_size=2,
        seed=5,
        maximize=False,
        objective_directions=[False, False],
    )
    call_rates: list[float] = []

    def custom_mutation(genome, _rng):
        call_rates.append(config.mutation_rate)
        mutated = genome[:]
        mutated[0] = 1.0
        return mutated

    summary, _history = run_nsga2(
        config=config,
        problem=_MultiObjectiveProblem(),
        selection_fn=_selection_stub,
        crossover_fn=_crossover_stub,
        mutation_fn=custom_mutation,
        init_fn=_real_init,
        rng=random.Random(config.seed),
    )

    assert call_rates
    assert all(rate == 1.0 for rate in call_rates)
    assert summary["is_nsga2"] is True
