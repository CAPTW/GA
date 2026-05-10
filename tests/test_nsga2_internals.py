from __future__ import annotations

from ga_lab.algorithms.nsga2 import (
    _crowding_distance,
    _nondominated_sort,
    _select_by_rank_and_crowding,
)
from ga_lab.config import GAConfig
from ga_lab.core.selection import crowded_tournament_select
from ga_lab.experiment.mo_metrics import dominates
from ga_lab.factory import build_runtime_context
from ga_lab.problems.base import as_fitness_vector
from ga_lab.runner import run_experiment
from ga_lab.utils.seed import make_rng


def _nsga2_test_config(*, run_name: str) -> GAConfig:
    return GAConfig(
        run_name=run_name,
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=12,
        genome_length=6,
        generations=6,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elitism=1,
        tournament_size=2,
        seed=17,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": 0.0, "high": 1.0},
        mutation_options={"sigma": 0.2},
        algorithm_options={"_return_final_population": True},
        log_every=3,
    )


def test_nondominated_sort_assigns_expected_ranks() -> None:
    adjusted_vectors = [
        [3.0, 1.0],
        [2.0, 2.0],
        [1.0, 3.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ]

    fronts, ranks = _nondominated_sort(adjusted_vectors)

    assert set(fronts[0]) == {0, 1, 2}
    assert ranks[3] == 1
    assert ranks[4] == 2


def test_crowding_distance_marks_boundary_points_infinite() -> None:
    objective_vectors = [
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
    ]

    distances = _crowding_distance([0, 1, 2], objective_vectors)

    assert distances[0] == float("inf")
    assert distances[2] == float("inf")
    assert 0.0 < distances[1] < float("inf")


def test_survival_selection_prefers_rank_then_crowding() -> None:
    fronts = [[0], [1, 2]]
    crowding = [0.0, 0.2, 10.0]
    ranks = [0, 1, 1]

    selected = _select_by_rank_and_crowding(2, fronts, crowding, ranks)

    assert 0 in selected
    assert 2 in selected
    assert 1 not in selected


def test_partial_front_objective_dedup_prefers_unique_points_when_front_overflows() -> None:
    fronts = [[0, 1, 2]]
    crowding = [float("inf"), 5.0, 4.0]
    ranks = [0, 0, 0]
    population = [[0.0], [1.0], [2.0]]
    objective_vectors = [[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]]

    selected = _select_by_rank_and_crowding(
        2,
        fronts,
        crowding,
        ranks,
        population=population,
        objective_vectors=objective_vectors,
        partial_front_dedup_mode="objective",
    )

    assert 0 in selected
    assert len(selected) == 2
    assert {tuple(objective_vectors[idx]) for idx in selected} == {(0.0, 1.0), (0.5, 0.5)}


def test_boundary_preservation_light_prefers_near_boundary_point_with_same_rank() -> None:
    fronts = [[0, 2, 1, 3]]
    crowding = [float("inf"), 0.6, 0.6, float("inf")]
    ranks = [0, 0, 0, 0]
    objective_vectors = [
        [0.0, 1.0],
        [0.2, 0.8],
        [0.5, 0.5],
        [1.0, 0.0],
    ]

    selected = _select_by_rank_and_crowding(
        3,
        fronts,
        crowding,
        ranks,
        objective_vectors=objective_vectors,
        partial_front_strategy="boundary_preservation_light",
    )

    assert 0 in selected
    assert 3 in selected
    assert 1 in selected
    assert 2 not in selected


def test_crowded_tournament_parent_selection_uses_rank_then_crowding() -> None:
    population = [[0.0], [1.0], [2.0]]
    ranks = [1, 0, 0]
    crowding = [999.0, 0.1, 10.0]

    winner = crowded_tournament_select(
        population,
        ranks,
        crowding,
        tournament_size=3,
        rng=make_rng(123),
    )

    assert winner == [2.0]


def test_nsga2_final_population_stays_within_bounds(tmp_path) -> None:
    config = _nsga2_test_config(run_name="test_nsga2_bounds")

    result = run_experiment(config, output_root=tmp_path)

    final_population = result.summary["final_population"]
    assert final_population
    assert len(final_population) == config.population_size
    for genome in final_population:
        assert all(0.0 <= float(gene) <= 1.0 for gene in genome)


def test_nsga2_summary_front_matches_final_population_nondominated_set(tmp_path) -> None:
    config = _nsga2_test_config(run_name="test_nsga2_final_front")

    result = run_experiment(config, output_root=tmp_path)
    runtime = build_runtime_context(config)
    directions = [bool(value) for value in result.summary["objective_directions"]]
    final_population = result.summary["final_population"]
    final_vectors = [list(as_fitness_vector(runtime.problem.fitness(genome))) for genome in final_population]

    expected_front = {
        tuple(candidate)
        for candidate in final_vectors
        if not any(
            candidate != other and dominates(other, candidate, directions)
            for other in final_vectors
        )
    }
    summary_front = {tuple(vector) for vector in result.summary["pareto_front_vectors"]}

    assert summary_front == expected_front
