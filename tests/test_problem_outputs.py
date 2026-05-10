from __future__ import annotations

from ga_lab.config import GAConfig
from ga_lab.runner import run_experiment


def test_tsp_summary_includes_decoded_route(tmp_path) -> None:
    config = GAConfig(
        run_name="test_tsp_outputs",
        problem="tsp",
        algorithm="ga",
        representation="permutation",
        selection="tournament",
        crossover="order",
        mutation="swap",
        population_size=12,
        genome_length=4,
        generations=6,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elitism=1,
        tournament_size=3,
        seed=2,
        maximize=True,
        log_every=2,
        problem_options={
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
    result = run_experiment(config, output_root=tmp_path)
    summary = result.summary

    assert sorted(summary["best_route"]) == [0, 1, 2, 3]
    assert summary["best_route_cycle"][0] == summary["best_route_cycle"][-1]
    assert abs(summary["best_route_distance"] + summary["best_fitness"]) < 1e-9


def test_knapsack_summary_includes_constraint_metrics(tmp_path) -> None:
    config = GAConfig(
        run_name="test_knapsack_outputs",
        problem="knapsack",
        algorithm="ga",
        representation="bit",
        selection="tournament",
        crossover="one_point",
        mutation="bit_flip",
        population_size=16,
        genome_length=4,
        generations=6,
        crossover_rate=0.9,
        mutation_rate=0.05,
        elitism=1,
        tournament_size=3,
        seed=7,
        maximize=True,
        log_every=2,
        problem_options={
            "num_items": 4,
            "weights": [6.0, 4.0, 3.0, 2.0],
            "values": [12.0, 8.0, 7.0, 4.0],
            "capacity": 10.0,
            "penalty_factor": 20.0,
        },
    )
    result = run_experiment(config, output_root=tmp_path)
    summary = result.summary

    assert summary["capacity"] == 10.0
    assert len(summary["best_selected_mask"]) == 4
    assert 0.0 <= summary["best_constraint_violation_rate"]
    assert 0.0 <= summary["population_constraint_violation_ratio"] <= 1.0
    assert "population_constraint_violation_ratio" in result.history[0]
