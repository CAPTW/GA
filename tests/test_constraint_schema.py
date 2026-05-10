from __future__ import annotations

import json

import pytest

from ga_lab.constraints import (
    DEFAULT_EQUALITY_TOLERANCE,
    evaluate_constraint_violations,
    is_feasible,
)


def test_inequality_value_at_or_below_zero_has_zero_violation() -> None:
    evaluation = evaluate_constraint_violations(inequality_values=[-1.5, 0.0])

    assert evaluation.inequality_violations == [0.0, 0.0]
    assert evaluation.total_violation == 0.0
    assert evaluation.violation_count == 0
    assert evaluation.feasible is True


def test_inequality_value_above_zero_uses_value_as_violation() -> None:
    evaluation = evaluate_constraint_violations(inequality_values=[0.25, 1.5])

    assert evaluation.inequality_violations == [0.25, 1.5]
    assert evaluation.total_violation == pytest.approx(1.75)
    assert evaluation.max_violation == pytest.approx(1.5)


def test_equality_value_within_tolerance_has_zero_violation() -> None:
    evaluation = evaluate_constraint_violations(
        equality_values=[DEFAULT_EQUALITY_TOLERANCE / 2.0],
        equality_tolerance=DEFAULT_EQUALITY_TOLERANCE,
    )

    assert evaluation.equality_violations == [0.0]
    assert evaluation.feasible is True


def test_equality_value_outside_tolerance_uses_abs_minus_tolerance() -> None:
    evaluation = evaluate_constraint_violations(
        equality_values=[-0.2],
        equality_tolerance=0.05,
    )

    assert evaluation.equality_violations == [pytest.approx(0.15)]
    assert evaluation.total_violation == pytest.approx(0.15)
    assert evaluation.feasible is False


def test_mixed_constraint_violations_compute_total_max_and_count() -> None:
    evaluation = evaluate_constraint_violations(
        inequality_values=[-1.0, 2.0],
        equality_values=[0.02, -0.20],
        equality_tolerance=0.05,
    )

    assert evaluation.inequality_violations == [0.0, 2.0]
    assert evaluation.equality_violations == [0.0, pytest.approx(0.15)]
    assert evaluation.total_violation == pytest.approx(2.15)
    assert evaluation.max_violation == pytest.approx(2.0)
    assert evaluation.violation_count == 2
    assert is_feasible(evaluation) is False


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_constraint_value_raises_value_error(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite constraint value"):
        evaluate_constraint_violations(inequality_values=[invalid_value])


def test_non_numeric_constraint_value_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Non-numeric constraint value"):
        evaluate_constraint_violations(inequality_values=["bad"])


def test_constraint_evaluation_to_dict_is_json_serializable() -> None:
    evaluation = evaluate_constraint_violations(
        inequality_values=[0.25],
        equality_values=[0.0],
        metadata={"problem": "toy"},
        warnings=["note"],
    )

    payload = evaluation.to_dict()

    assert payload["metadata"]["problem"] == "toy"
    assert payload["warnings"] == ["note"]
    json.dumps(payload)
