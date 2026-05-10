from __future__ import annotations

import pytest

from ga_lab.experiment.diversity_diagnostics import (
    archive_duplicate_rate,
    boundary_point_count,
    crowding_distance_summary,
    duplicate_rate,
    evaluate_diversity_diagnostics,
    raw_nondominated_vectors,
    unique_vector_count,
)


def test_duplicate_helpers_count_duplicates_consistently() -> None:
    vectors = [[0.1, 0.9], [0.1, 0.9], [0.2, 0.8]]

    assert unique_vector_count(vectors) == 2
    assert duplicate_rate(vectors) == pytest.approx(1.0 / 3.0)


def test_archive_duplicate_rate_uses_raw_nondominated_vectors() -> None:
    objective_vectors = [[0.1, 0.9], [0.1, 0.9], [0.2, 0.8], [0.3, 0.95]]
    front = raw_nondominated_vectors(objective_vectors, [False, False])

    assert len(front) == 3
    assert archive_duplicate_rate(objective_vectors, [False, False]) == pytest.approx(1.0 / 3.0)


def test_diversity_diagnostics_report_boundary_and_crowding_summary() -> None:
    objective_vectors = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    diagnostics = evaluate_diversity_diagnostics(
        objective_vectors,
        directions=[False, False],
        decision_vectors=[[0.0], [0.5], [1.0]],
    )

    assert diagnostics["boundary_point_count"] == 2
    assert diagnostics["decision_duplicate_rate"] == 0.0
    assert diagnostics["unique_objective_count"] == 3
    assert diagnostics["crowding_distance_infinite_count"] == 2
    assert diagnostics["crowding_distance_finite_mean"] is not None


def test_crowding_distance_summary_handles_empty_archive() -> None:
    summary = crowding_distance_summary([], [False, False])

    assert summary["crowding_distance_infinite_count"] == 0
    assert summary["crowding_distance_finite_mean"] is None


def test_boundary_point_count_is_zero_for_empty_archive() -> None:
    assert boundary_point_count([], [False, False]) == 0
