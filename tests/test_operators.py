from __future__ import annotations

import random

import pytest

from ga_lab.config import GAConfig
from ga_lab.runner import run_experiment
from ga_lab.core.operators import (
    SelectionState,
    bit_flip_mutation,
    build_crossover_fn,
    build_mutation_fn,
    build_representation_adapter,
    build_selection_fn,
    crowded_tournament_select,
    gaussian_mutation,
    inversion_mutation,
    one_point_crossover,
    order_crossover,
    roulette_wheel_select,
    random_genome_from_config,
    swap_mutation,
    tournament_select,
)
from ga_lab.core.selection import _roulette_weights


def test_one_point_crossover_preserves_length() -> None:
    rng = random.Random(3)
    a = [0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 1]
    child_a, child_b = one_point_crossover(a, b, rng)
    assert len(child_a) == len(a)
    assert len(child_b) == len(b)
    assert child_a != a or child_b != b


def test_bit_flip_mutation_can_flip_all_bits() -> None:
    rng = random.Random(4)
    genome = [0, 1, 0, 1]
    mutated = bit_flip_mutation(genome, mutation_rate=1.0, rng=rng)
    assert mutated == [1, 0, 1, 0]


def test_tournament_select_returns_member_of_population() -> None:
    rng = random.Random(5)
    population = [[0, 0], [1, 1], [1, 0]]
    fitnesses = [0.0, 2.0, 1.0]
    winner = tournament_select(population, fitnesses, tournament_size=2, maximize=True, rng=rng)
    assert winner in population


def test_build_operators_bit() -> None:
    config = GAConfig(
        run_name="op_factory_bit",
        problem="onemax",
        population_size=20,
        genome_length=8,
        generations=1,
        crossover_rate=0.9,
        mutation_rate=0.1,
        selection="tournament",
        crossover="one_point",
        mutation="bit_flip",
        representation="bit",
    )
    rng = random.Random(7)
    init = random_genome_from_config(rng, config.genome_length, config)
    assert len(init) == 8
    assert all(gene in {0.0, 1.0} for gene in init)

    select = build_selection_fn(config)
    pop = [init, [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    selected = select(pop, SelectionState.from_fitnesses([0.0, 1.0], config.maximize), rng)
    assert selected in pop


def test_gaussian_mutation_respects_bounds() -> None:
    rng = random.Random(9)
    genome = [0.5, 0.5, 0.5]
    mutated = gaussian_mutation(genome, mutation_rate=1.0, rng=rng, sigma=0.2, low=0.0, high=1.0)
    assert all(0.0 <= value <= 1.0 for value in mutated)


def test_permutation_order_crossover_and_mutations() -> None:
    rng = random.Random(11)
    parent_a = [0.0, 1.0, 2.0, 3.0, 4.0]
    parent_b = [4.0, 3.0, 2.0, 1.0, 0.0]
    child_a, child_b = order_crossover(parent_a, parent_b, rng)
    assert sorted(child_a) == sorted(parent_a)
    assert sorted(child_b) == sorted(parent_b)

    swapped = swap_mutation(parent_a[:], mutation_rate=1.0, rng=rng)
    assert sorted(swapped) == sorted(parent_a)
    inverted = inversion_mutation(parent_a[:], mutation_rate=1.0, rng=rng)
    assert sorted(inverted) == sorted(parent_a)


def test_real_initializer_and_gaussian_pipeline() -> None:
    rng = random.Random(13)
    config = GAConfig(
        run_name="op_factory_real",
        problem="onemax",
        population_size=10,
        genome_length=6,
        generations=1,
        crossover_rate=0.9,
        mutation_rate=0.25,
        selection="rank",
        crossover="arithmetic",
        mutation="gaussian",
        representation="real",
        representation_options={"low": -1.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
    )
    chromosome = random_genome_from_config(rng, config.genome_length, config)
    assert len(chromosome) == 6
    assert all(-1.0 <= gene <= 1.0 for gene in chromosome)
    crossover = build_crossover_fn(config)
    child_a, child_b = crossover(chromosome, chromosome[::-1], rng)
    assert len(child_a) == len(chromosome)
    assert len(child_b) == len(chromosome)
    mutate = build_mutation_fn(config)
    mutated = mutate(chromosome, rng)
    assert len(mutated) == len(chromosome)


def test_runtime_triplet_bit_one_point_bit_flip(tmp_path) -> None:
    config = GAConfig(
        run_name="runtime_bit_combo",
        problem="onemax",
        population_size=20,
        genome_length=10,
        generations=4,
        crossover_rate=0.9,
        mutation_rate=0.05,
        seed=17,
        elitism=1,
        tournament_size=3,
    )
    result = run_experiment(config, output_root=tmp_path)
    assert result.summary["problem"] == "onemax"
    assert result.summary["selection"] == "tournament"


def test_runtime_triplet_bit_uniform_bit_flip(tmp_path) -> None:
    config = GAConfig(
        run_name="runtime_bit_uniform_combo",
        problem="onemax",
        population_size=20,
        genome_length=12,
        generations=4,
        crossover="uniform",
        crossover_rate=0.9,
        mutation_rate=0.05,
        seed=19,
        elitism=1,
        tournament_size=3,
    )
    result = run_experiment(config, output_root=tmp_path)
    assert result.summary["problem"] == "onemax"
    assert result.summary["crossover"] == "uniform"


def test_runtime_triplet_real_arithmetic_gaussian(tmp_path) -> None:
    config = GAConfig(
        run_name="runtime_real_combo",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=14,
        genome_length=6,
        generations=5,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elitism=1,
        tournament_size=2,
        seed=9,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
    )
    result = run_experiment(config, output_root=tmp_path)
    assert result.summary["problem"] == "zdt1"
    assert result.summary["is_nsga2"] is True


def test_runtime_triplet_permutation_order_swap(tmp_path) -> None:
    config = GAConfig(
        run_name="runtime_tsp_order_swap",
        problem="tsp",
        problem_options={"num_cities": 6, "seed": 1, "return_to_start": True},
        representation="permutation",
        selection="tournament",
        crossover="order",
        mutation="swap",
        population_size=16,
        genome_length=6,
        generations=6,
        crossover_rate=0.9,
        mutation_rate=0.4,
        elitism=1,
        tournament_size=3,
        seed=5,
        maximize=True,
        log_every=2,
    )
    result = run_experiment(config, output_root=tmp_path)
    assert result.summary["problem"] == "tsp"
    assert result.summary["best_route"] is not None


def test_runtime_triplet_permutation_order_inversion(tmp_path) -> None:
    config = GAConfig(
        run_name="runtime_tsp_order_inversion",
        problem="tsp",
        problem_options={"num_cities": 6, "seed": 2, "return_to_start": True},
        representation="permutation",
        selection="tournament",
        crossover="order",
        mutation="inversion",
        population_size=16,
        genome_length=6,
        generations=6,
        crossover_rate=0.9,
        mutation_rate=0.4,
        elitism=1,
        tournament_size=3,
        seed=7,
        maximize=True,
        log_every=2,
    )
    result = run_experiment(config, output_root=tmp_path)
    assert result.summary["problem"] == "tsp"
    assert isinstance(result.summary["best_route_distance"], float)


def test_invalid_crossover_with_bit_representation_is_rejected(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="Crossover 'order' is incompatible with representation 'bit'",
    ):
        run_experiment(
            GAConfig(
                run_name="invalid_bit_order",
                problem="onemax",
                population_size=12,
                genome_length=8,
                generations=1,
                crossover="order",
                mutation="bit_flip",
                crossover_rate=0.9,
                mutation_rate=0.05,
                seed=1,
            ),
            output_root=tmp_path,
        )


def test_invalid_mutation_with_real_representation_is_rejected(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="Mutation 'bit_flip' is incompatible with representation 'real'",
    ):
        run_experiment(
            GAConfig(
                run_name="invalid_real_bitflip",
                problem="zdt1",
                algorithm="nsga2",
                representation="real",
                selection="tournament",
                crossover="arithmetic",
                mutation="bit_flip",
                population_size=12,
                genome_length=6,
                generations=1,
                crossover_rate=0.9,
                mutation_rate=0.1,
                seed=2,
                maximize=False,
                objective_directions=[False, False],
                representation_options={"low": 0.0, "high": 1.0},
            ),
            output_root=tmp_path,
        )


def test_invalid_crossover_with_permutation_representation_is_rejected(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="Crossover 'uniform' is incompatible with representation 'permutation'",
    ):
        run_experiment(
                GAConfig(
                    run_name="invalid_perm_uniform",
                    problem="tsp",
                    representation="permutation",
                    selection="tournament",
                    crossover="uniform",
                    mutation="swap",
                    population_size=12,
                    genome_length=20,
                    generations=1,
                    crossover_rate=0.9,
                    mutation_rate=0.05,
                    seed=3,
                ),
            output_root=tmp_path,
        )


def test_nsga2_selection_uses_crowded_tournament() -> None:
    rng = random.Random(17)
    config = GAConfig(
        run_name="nsga2_selection",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=4,
        genome_length=4,
        generations=1,
        crossover_rate=0.9,
        mutation_rate=0.1,
        tournament_size=4,
        maximize=False,
        objective_directions=[False, False],
    )
    population = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]
    ranks = [2, 0, 1, 0]
    crowding = [0.1, 0.2, 0.3, 0.9]

    select = build_selection_fn(config)
    selected = select(population, SelectionState.from_pareto(ranks, crowding), rng)

    assert selected == population[3]


def test_crowded_tournament_select_returns_best_rank_then_crowding() -> None:
    rng = random.Random(19)
    population = [[0.0], [1.0], [2.0], [3.0]]
    selected = crowded_tournament_select(
        population,
        ranks=[1, 0, 0, 2],
        crowding=[0.1, 0.3, 0.9, 1.0],
        tournament_size=4,
        rng=rng,
    )

    assert selected == [2.0]


def test_representation_adapter_repairs_bit_and_permutation_genomes() -> None:
    bit_config = GAConfig(
        run_name="bit_adapter",
        problem="onemax",
        population_size=8,
        genome_length=4,
        generations=1,
        crossover_rate=0.9,
        mutation_rate=0.1,
    )
    bit_adapter = build_representation_adapter(bit_config)
    assert bit_adapter.repair([0.2, 0.8, -3.0, 7.0], 4) == [0.0, 1.0, 0.0, 1.0]

    permutation_config = GAConfig(
        run_name="perm_adapter",
        problem="tsp",
        representation="permutation",
        selection="tournament",
        crossover="order",
        mutation="swap",
        population_size=8,
        genome_length=5,
        generations=1,
        crossover_rate=0.9,
        mutation_rate=0.1,
    )
    permutation_adapter = build_representation_adapter(permutation_config)
    repaired = permutation_adapter.repair([0.0, 0.0, 4.0, 9.0, 2.0], 5)
    assert sorted(int(value) for value in repaired) == [0, 1, 2, 3, 4]


def test_roulette_weights_all_zero_fitness_returns_uniform_positive_weights() -> None:
    population = [[0.0], [1.0], [2.0]]
    weights = _roulette_weights([0.0, 0.0, 0.0], maximize=True)
    assert weights == [1.0, 1.0, 1.0]
    selected = roulette_wheel_select(population, [0.0, 0.0, 0.0], 1, True, random.Random(0))
    assert selected in population


def test_roulette_weights_handles_negative_and_zero_fitness() -> None:
    maximize_weights = _roulette_weights([-4.0, -1.0, 0.0, 2.0], maximize=True)
    assert maximize_weights[3] > maximize_weights[0]

    minimize_weights = _roulette_weights([-4.0, -1.0, 0.0, 2.0], maximize=False)
    assert minimize_weights[0] > minimize_weights[3]
    assert all(value > 0.0 for value in minimize_weights)


def test_roulette_selection_biases_better_individual_for_maximize() -> None:
    population = [[0.0], [1.0], [2.0]]
    rng = random.Random(10)
    winners = {
        tuple(roulette_wheel_select(population, [0.0, 1.0, 10.0], 1, True, rng))
        for _ in range(50)
    }
    assert (2.0,) in winners
