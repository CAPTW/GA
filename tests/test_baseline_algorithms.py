from __future__ import annotations

from pathlib import Path

from ga_lab.config import GAConfig
from ga_lab.experiment.baseline_protocol import (
    normalize_run_row,
    run_sphere_hill_climb,
    run_sphere_random_search,
    summarize_paired_comparison,
)
from ga_lab.experiment.budget_baseline_comparison import load_comparison_manifest


def _sphere_config() -> GAConfig:
    return GAConfig(
        run_name="test_sphere",
        problem="sphere",
        algorithm="ga",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=32,
        genome_length=8,
        generations=40,
        crossover_rate=0.9,
        mutation_rate=0.12,
        elitism=1,
        tournament_size=3,
        maximize=False,
        target_fitness=1e-2,
        representation_options={"low": -5.0, "high": 5.0},
        mutation_options={"sigma": 0.2},
    )


def test_sphere_random_search_is_reproducible() -> None:
    config = _sphere_config()
    left = run_sphere_random_search(config, seed=1234, budget=64)
    right = run_sphere_random_search(config, seed=1234, budget=64)

    assert left.evaluations == 64
    assert right.evaluations == 64
    assert left.best_fitness == right.best_fitness
    assert left.best_solution == right.best_solution


def test_sphere_hill_climb_respects_budget() -> None:
    config = _sphere_config()
    result = run_sphere_hill_climb(config, seed=2345, budget=32)

    assert result.evaluations == 32
    assert result.best_fitness is not None
    assert result.error_message is None
    assert result.metadata["accepted_moves"] >= 0


def test_local_baseline_comparison_manifest_loads() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "configs" / "baselines" / "local_ga_baseline_comparison.json"
    manifest, entries = load_comparison_manifest(manifest_path)

    assert manifest["suite_name"] == "local_ga_baseline_comparison"
    assert [entry.problem for entry in entries] == ["onemax", "knapsack", "tsp"]


def test_paired_summary_uses_problem_direction() -> None:
    ga_row = normalize_run_row(
        {
            "problem": "tsp",
            "label": "recommended_preset",
            "seed": 1,
            "configured_evaluation_budget": 100,
            "actual_evaluations": 100,
            "runtime_seconds": 0.1,
            "final_best_distance": 10.0,
        }
    )
    baseline_row = normalize_run_row(
        {
            "problem": "tsp",
            "label": "random_tours",
            "seed": 1,
            "configured_evaluation_budget": 100,
            "actual_evaluations": 100,
            "runtime_seconds": 0.1,
            "final_best_distance": 12.0,
        }
    )

    paired = summarize_paired_comparison(
        problem="tsp",
        ga_rows=[ga_row],
        baseline_rows=[baseline_row],
    )

    assert paired["ga_win"] == 1
    assert paired["ga_loss"] == 0
    assert paired["mean_delta"] > 0
