from __future__ import annotations

import math

import pytest

from ga_lab.experiment.mo_metrics import (
    coverage_indicator,
    evaluate_front_metrics,
    hypervolume_2d,
    nondominated_vectors,
    validate_objective_vector,
    zdt1_reference_front,
)


def test_nondominated_vectors_filters_dominated_points() -> None:
    vectors = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.95]]
    front = nondominated_vectors(vectors, [False, False])

    assert front == [[0.1, 0.9], [0.2, 0.8]]


def test_hypervolume_2d_matches_simple_rectangle_case() -> None:
    hv = hypervolume_2d([[0.5, 0.5]], [False, False], [1.0, 1.0])

    assert hv == pytest.approx(0.25)


def test_evaluate_front_metrics_returns_expected_fields() -> None:
    metrics = evaluate_front_metrics(
        [[0.1, 0.9], [0.2, 0.8]],
        directions=[False, False],
        reference_front=zdt1_reference_front(11),
        reference_point=[1.1, 1.1],
    )

    assert metrics["archive_size"] == 2
    assert metrics["nondominated_count"] == 2
    assert isinstance(metrics["objective_min_max"], dict)
    assert math.isfinite(float(metrics["hypervolume_2d"]))


def test_validate_objective_vector_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="Non-finite objective"):
        validate_objective_vector([float("nan"), 1.0], label="test")


def test_coverage_indicator_counts_dominated_fraction() -> None:
    left = [[0.1, 0.9], [0.2, 0.8]]
    right = [[0.3, 0.95], [0.4, 0.85]]

    coverage = coverage_indicator(left, right, [False, False])

    assert coverage == pytest.approx(1.0)
