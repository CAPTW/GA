from __future__ import annotations

import importlib.util

import pytest

from ga_lab.config import GAConfig
from ga_lab.experiment.external_mo_comparators import (
    METRIC_SPECS,
    optional_library_status,
    paired_metric_summary,
    result_to_front_row,
    run_deap_nsga2,
    run_pymoo_nsga2,
)


def _zdt1_config() -> GAConfig:
    return GAConfig(
        run_name="test_external_mo",
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


def test_optional_library_status_reports_missing_pymoo() -> None:
    status = optional_library_status("pymoo")

    if importlib.util.find_spec("pymoo") is None:
        assert status.installed is False
        assert status.reason == "pymoo is not installed"
    else:
        assert status.installed is True


def test_pymoo_missing_returns_skipped_result() -> None:
    config = _zdt1_config()
    result = run_pymoo_nsga2(config, seed=123, budget=60)

    if importlib.util.find_spec("pymoo") is None:
        assert result.status == "skipped"
        assert result.success is False
        assert result.error_message == "pymoo is not installed"
    else:
        assert result.status in {"success", "failed"}


def test_deap_missing_returns_skipped_result() -> None:
    config = _zdt1_config()
    result = run_deap_nsga2(config, seed=123, budget=60)

    if importlib.util.find_spec("deap") is None:
        assert result.status == "skipped"
        assert result.success is False
        assert result.error_message == "deap is not installed"
    else:
        assert result.status in {"success", "failed"}


def test_result_to_front_row_preserves_failure_schema() -> None:
    config = _zdt1_config()
    result = run_pymoo_nsga2(config, seed=123, budget=60)
    row = result_to_front_row(
        result,
        reference_front=[[0.0, 1.0], [1.0, 0.0]],
        reference_point=[1.1, 1.1],
    )

    assert row["status"] in {"success", "skipped", "failed"}
    assert "hypervolume_2d" in row
    if not row["success"]:
        assert row["hypervolume_2d"] is None


def test_paired_metric_summary_counts_wins_and_losses() -> None:
    internal_rows = [
        {
            "seed": 1,
            "success": True,
            "hypervolume_2d": 0.8,
            "reference_front_distance": 0.2,
            "generational_distance": 0.2,
            "inverted_generational_distance": 0.3,
            "spacing": 0.1,
            "nondominated_count": 6,
            "metadata": {"objective_directions": [False, False]},
            "nondominated_objective_vectors": [[0.1, 0.9], [0.2, 0.8]],
        },
        {
            "seed": 2,
            "success": True,
            "hypervolume_2d": 0.7,
            "reference_front_distance": 0.3,
            "generational_distance": 0.3,
            "inverted_generational_distance": 0.4,
            "spacing": 0.2,
            "nondominated_count": 5,
            "metadata": {"objective_directions": [False, False]},
            "nondominated_objective_vectors": [[0.15, 0.85], [0.25, 0.75]],
        },
    ]
    comparator_rows = [
        {
            "seed": 1,
            "success": True,
            "hypervolume_2d": 0.6,
            "reference_front_distance": 0.4,
            "generational_distance": 0.4,
            "inverted_generational_distance": 0.5,
            "spacing": 0.3,
            "nondominated_count": 4,
            "metadata": {"objective_directions": [False, False]},
            "nondominated_objective_vectors": [[0.3, 0.95]],
        },
        {
            "seed": 2,
            "success": True,
            "hypervolume_2d": 0.75,
            "reference_front_distance": 0.2,
            "generational_distance": 0.2,
            "inverted_generational_distance": 0.35,
            "spacing": 0.25,
            "nondominated_count": 5,
            "metadata": {"objective_directions": [False, False]},
            "nondominated_objective_vectors": [[0.12, 0.88], [0.23, 0.78]],
        },
    ]

    hv_summary = paired_metric_summary(
        internal_rows=internal_rows,
        comparator_rows=comparator_rows,
        metric_name="hypervolume_2d",
    )
    spacing_summary = paired_metric_summary(
        internal_rows=internal_rows,
        comparator_rows=comparator_rows,
        metric_name="spacing",
    )

    assert set(METRIC_SPECS) >= {
        "hypervolume_2d",
        "reference_front_distance",
        "generational_distance",
        "inverted_generational_distance",
        "spacing",
        "nondominated_count",
        "coverage_indicator",
    }
    assert hv_summary["internal_win"] == 1
    assert hv_summary["external_win"] == 1
    assert spacing_summary["internal_win"] == 2


@pytest.mark.skipif(importlib.util.find_spec("pymoo") is None, reason="pymoo optional dependency not installed")
def test_pymoo_smoke_if_installed() -> None:
    config = _zdt1_config()
    result = run_pymoo_nsga2(config, seed=123, budget=60)

    assert result.status == "success"
    assert result.success is True
    assert result.objective_vectors
    assert result.nondominated_objective_vectors
    assert result.evaluations > 0
    assert result.runtime_seconds >= 0.0
    assert result.metadata["library_status"]["version"] not in {None, ""}


@pytest.mark.skipif(importlib.util.find_spec("deap") is None, reason="deap optional dependency not installed")
def test_deap_smoke_if_installed() -> None:
    config = _zdt1_config()
    result = run_deap_nsga2(config, seed=123, budget=60)

    assert result.status == "success"
    assert result.success is True
    assert result.objective_vectors
    assert result.nondominated_objective_vectors
    assert result.evaluations > 0
    assert result.runtime_seconds >= 0.0
    assert result.metadata["library_status"]["version"] not in {None, ""}
