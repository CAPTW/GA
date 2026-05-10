from __future__ import annotations

from ga_lab.experiment.constrained_fairness import evaluate_constrained_fairness


def _expected_contract() -> dict[str, object]:
    return {
        "problem": "constrained_sphere",
        "dimension": 5,
        "bounds": [(-5.0, 5.0)] * 5,
        "objective_directions": [False],
        "constraint_count": 1,
        "inequality_count": 1,
        "equality_count": 0,
        "tolerance": 1e-8,
        "feasibility_policy": "feasibility_first",
        "violation_aggregation": "total_violation",
        "requested_budget": 300,
        "non_finite_constraint_fail_fast_policy": "value_error",
        "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
    }


def _observed_row(**overrides) -> dict[str, object]:
    row = {
        "strategy": "random_search_feasibility_first",
        "problem": "constrained_sphere",
        "dimension": 5,
        "bounds": [(-5.0, 5.0)] * 5,
        "objective_directions": [False],
        "constraint_count": 1,
        "inequality_count": 1,
        "equality_count": 0,
        "tolerance": 1e-8,
        "feasibility_policy": "feasibility_first",
        "violation_aggregation": "total_violation",
        "requested_budget": 300,
        "actual_evaluations": 300,
        "runtime_seconds": 0.1,
        "non_finite_constraint_fail_fast_policy": "value_error",
        "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
    }
    row.update(overrides)
    return row


def test_matching_constrained_configs_pass() -> None:
    payload = evaluate_constrained_fairness(
        expected_contract=_expected_contract(),
        observed_rows=[_observed_row()],
    )

    assert payload["status"] == "pass"


def test_constraint_count_mismatch_fails() -> None:
    payload = evaluate_constrained_fairness(
        expected_contract=_expected_contract(),
        observed_rows=[_observed_row(constraint_count=2)],
    )

    assert payload["status"] == "fail"
    assert any(issue["issue_type"] == "constraint_count_mismatch" for issue in payload["issues"])


def test_tolerance_mismatch_fails() -> None:
    payload = evaluate_constrained_fairness(
        expected_contract=_expected_contract(),
        observed_rows=[_observed_row(tolerance=1e-6)],
    )

    assert payload["status"] == "fail"
    assert any(issue["issue_type"] == "tolerance_mismatch" for issue in payload["issues"])


def test_objective_direction_mismatch_fails() -> None:
    payload = evaluate_constrained_fairness(
        expected_contract=_expected_contract(),
        observed_rows=[_observed_row(objective_directions=[True])],
    )

    assert payload["status"] == "fail"
    assert any(issue["issue_type"] == "objective_direction_mismatch" for issue in payload["issues"])


def test_actual_evaluations_mismatch_is_handled() -> None:
    payload = evaluate_constrained_fairness(
        expected_contract=_expected_contract(),
        observed_rows=[_observed_row(actual_evaluations=285)],
    )

    assert any(issue["issue_type"] in {"actual_evaluations_warning", "actual_evaluations_mismatch"} for issue in payload["issues"])


def test_missing_fail_fast_policy_fails() -> None:
    payload = evaluate_constrained_fairness(
        expected_contract=_expected_contract(),
        observed_rows=[_observed_row(non_finite_constraint_fail_fast_policy=None)],
    )

    assert payload["status"] == "fail"
    assert any(
        issue["issue_type"] == "non_finite_constraint_fail_fast_policy_missing_or_mismatch"
        for issue in payload["issues"]
    )


def test_runtime_difference_does_not_fail() -> None:
    payload = evaluate_constrained_fairness(
        expected_contract=_expected_contract(),
        observed_rows=[_observed_row(runtime_seconds=99.0)],
    )

    assert not any(
        issue["issue_type"] == "runtime_difference_not_checked" and issue["status"] == "fail"
        for issue in payload["issues"]
    )
