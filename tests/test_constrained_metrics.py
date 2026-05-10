from __future__ import annotations

import json

import pytest

from ga_lab.constraints import evaluate_constraint_violations
from ga_lab.experiment.constrained_metrics import (
    best_feasible_objective,
    best_feasible_objectives,
    feasible_rate,
    summarize_constrained_population,
)


def _feasible_evaluation():
    return evaluate_constraint_violations(inequality_values=[-1.0])


def _infeasible_evaluation(value: float):
    return evaluate_constraint_violations(inequality_values=[value])


def test_all_feasible_population_summary() -> None:
    evaluations = [_feasible_evaluation(), _feasible_evaluation(), _feasible_evaluation()]

    summary = summarize_constrained_population(evaluations)

    assert summary["feasible_count"] == 3
    assert summary["infeasible_count"] == 0
    assert summary["feasible_rate"] == pytest.approx(1.0)
    assert summary["all_feasible"] is True
    assert summary["all_infeasible"] is False


def test_all_infeasible_population_summary() -> None:
    evaluations = [_infeasible_evaluation(1.0), _infeasible_evaluation(2.0)]

    summary = summarize_constrained_population(evaluations)

    assert summary["feasible_count"] == 0
    assert summary["infeasible_count"] == 2
    assert summary["feasible_rate"] == pytest.approx(0.0)
    assert summary["all_feasible"] is False
    assert summary["all_infeasible"] is True


def test_mixed_population_summary() -> None:
    evaluations = [_feasible_evaluation(), _infeasible_evaluation(2.0), _feasible_evaluation()]

    summary = summarize_constrained_population(evaluations)

    assert summary["feasible_count"] == 2
    assert summary["infeasible_count"] == 1
    assert summary["feasible_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["any_feasible"] is True
    assert summary["violation_count_total"] == 1


def test_best_feasible_objective_returns_best_minimization_value() -> None:
    evaluations = [_feasible_evaluation(), _infeasible_evaluation(1.0), _feasible_evaluation()]

    best = best_feasible_objective([5.0, 1.0, 3.0], evaluations)

    assert best == pytest.approx(3.0)


def test_best_feasible_objective_returns_none_when_no_feasible_solution_exists() -> None:
    evaluations = [_infeasible_evaluation(1.0), _infeasible_evaluation(2.0)]

    assert best_feasible_objective([5.0, 3.0], evaluations) is None


def test_best_feasible_objectives_returns_componentwise_summary_for_feasible_vectors() -> None:
    evaluations = [_feasible_evaluation(), _infeasible_evaluation(1.0), _feasible_evaluation()]

    summary = best_feasible_objectives(
        [[2.0, 5.0], [0.5, 1.0], [1.5, 3.0]],
        evaluations,
        [False, False],
    )

    assert summary == [pytest.approx(1.5), pytest.approx(3.0)]


def test_feasible_rate_helper_matches_summary_rate() -> None:
    evaluations = [_feasible_evaluation(), _infeasible_evaluation(1.0)]

    assert feasible_rate(evaluations) == pytest.approx(0.5)


def test_violation_summary_is_json_serializable() -> None:
    evaluations = [_feasible_evaluation(), _infeasible_evaluation(1.0)]

    summary = summarize_constrained_population(evaluations)

    json.dumps(summary)
