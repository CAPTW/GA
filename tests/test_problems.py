from __future__ import annotations

from ga_lab.problems import build_problem


def test_tsp_fitness_uses_permutation_genome() -> None:
    problem = build_problem(
        "tsp",
        {
            "num_cities": 4,
            "coordinates": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            "return_to_start": True,
        },
    )
    fitness = problem.fitness([0, 1, 2, 3])
    assert fitness < 0


def test_tsp_fitness_is_negative_distance() -> None:
    problem = build_problem(
        "tsp",
        {
            "num_cities": 4,
            "coordinates": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            "return_to_start": True,
        },
    )
    route = [0, 1, 2, 3]
    assert problem.fitness(route) == -problem.route_distance(route)


def test_knapsack_penalizes_overweight() -> None:
    problem = build_problem(
        "knapsack",
        {
            "num_items": 3,
            "weights": [6.0, 6.0, 6.0],
            "values": [10.0, 10.0, 10.0],
            "capacity": 10.0,
            "penalty_factor": 100.0,
        },
    )
    under = problem.fitness([1, 0, 0])
    over = problem.fitness([1, 1, 1])
    assert over < under


def test_knapsack_penalty_factor_scales_by_violation() -> None:
    problem = build_problem(
        "knapsack",
        {
            "num_items": 2,
            "weights": [8.0, 12.0],
            "values": [10.0, 7.0],
            "capacity": 10.0,
            "penalty_factor": 5.0,
        },
    )
    under = problem.fitness([1, 0])
    over = problem.fitness([1, 1])
    # Overweight by 10.0 with penalty_factor 5.0 => subtraction 50.0 from value 17.0
    assert over == 17.0 - 5.0 * 10.0
    assert over < under


def test_zdt1_returns_two_objectives() -> None:
    problem = build_problem("zdt1", {})
    fitness = problem.fitness([0.2, 0.4, 0.6, 0.8])
    assert len(fitness) == 2


def test_onemax_supports_leading_ones_family() -> None:
    problem = build_problem("onemax", {"family": "leading_ones"})
    assert problem.fitness([1, 1, 1, 0, 1]) == 3.0


def test_onemax_supports_deceptive_trap_family() -> None:
    problem = build_problem("onemax", {"family": "trap", "trap_block_size": 4})
    assert problem.fitness([0, 0, 0, 0]) == 3.0
    assert problem.fitness([1, 1, 1, 1]) == 4.0


def test_onemax_supports_jump_family() -> None:
    problem = build_problem("onemax", {"family": "jump", "jump_k": 2})
    assert problem.fitness([1, 1, 1, 1]) == 4.0
    assert problem.fitness([1, 1, 0, 0]) == 2.0
    assert problem.fitness([1, 1, 1, 0]) == 1.0


def test_tsp_supports_distance_matrix() -> None:
    problem = build_problem(
        "tsp",
        {
            "num_cities": 3,
            "distance_matrix": [
                [0.0, 2.0, 4.0],
                [2.0, 0.0, 3.0],
                [4.0, 3.0, 0.0],
            ],
        },
    )
    assert problem.route_distance([0, 1, 2]) == 9.0


def test_zdt1_supports_zdt3_family() -> None:
    problem = build_problem("zdt1", {"family": "zdt3"})
    fitness = problem.fitness([0.2, 0.4, 0.6, 0.8])
    assert len(fitness) == 2
