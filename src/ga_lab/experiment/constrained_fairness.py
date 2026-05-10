from __future__ import annotations

from typing import Any, Sequence


PASS = "pass"
WARNING = "warning"
FAIL = "fail"


def _severity_for_status(status: str) -> str:
    if status == FAIL:
        return "high"
    if status == WARNING:
        return "medium"
    return "info"


def _normalize_bounds(value: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    normalized: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
            return None
        normalized.append((float(item[0]), float(item[1])))
    return tuple(normalized)


def _append_issue(
    issues: list[dict[str, Any]],
    *,
    status: str,
    issue_type: str,
    message: str,
    recommended_action: str,
) -> None:
    issues.append(
        {
            "status": status,
            "issue_type": issue_type,
            "severity": _severity_for_status(status),
            "message": message,
            "recommended_action": recommended_action,
        }
    )


def _status_from_issues(issues: Sequence[dict[str, Any]]) -> str:
    if any(issue["status"] == FAIL for issue in issues):
        return FAIL
    if any(issue["status"] == WARNING for issue in issues):
        return WARNING
    return PASS


def summarize_constrained_fairness(issues: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        PASS: sum(1 for issue in issues if issue["status"] == PASS),
        WARNING: sum(1 for issue in issues if issue["status"] == WARNING),
        FAIL: sum(1 for issue in issues if issue["status"] == FAIL),
    }


def evaluate_constrained_fairness(
    *,
    expected_contract: dict[str, Any],
    observed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    for row in observed_rows:
        row_name = str(row.get("strategy", row.get("algorithm", "unknown_strategy")))

        if row.get("problem") == expected_contract.get("problem"):
            _append_issue(
                issues,
                status=PASS,
                issue_type="problem_match",
                message=f"{row_name}: problem matches expected contract",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="problem_mismatch",
                message=(
                    f"{row_name}: problem mismatch observed={row.get('problem')} "
                    f"expected={expected_contract.get('problem')}"
                ),
                recommended_action="runner problem binding을 동일하게 맞춘다",
            )

        if int(row.get("dimension", -1)) == int(expected_contract.get("dimension", -2)):
            _append_issue(
                issues,
                status=PASS,
                issue_type="dimension_match",
                message=f"{row_name}: dimension matches expected contract",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="dimension_mismatch",
                message=(
                    f"{row_name}: dimension mismatch observed={row.get('dimension')} "
                    f"expected={expected_contract.get('dimension')}"
                ),
                recommended_action="problem dimension과 runner metadata를 다시 정렬한다",
            )

        observed_bounds = _normalize_bounds(row.get("bounds"))
        expected_bounds = _normalize_bounds(expected_contract.get("bounds"))
        if observed_bounds == expected_bounds:
            _append_issue(
                issues,
                status=PASS,
                issue_type="bounds_match",
                message=f"{row_name}: bounds match expected contract",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="bounds_mismatch",
                message=(
                    f"{row_name}: bounds mismatch observed={observed_bounds} "
                    f"expected={expected_bounds}"
                ),
                recommended_action="problem bounds metadata를 동일하게 맞춘다",
            )

        if list(row.get("objective_directions", [])) == list(expected_contract.get("objective_directions", [])):
            _append_issue(
                issues,
                status=PASS,
                issue_type="objective_direction_match",
                message=f"{row_name}: objective direction matches",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="objective_direction_mismatch",
                message=(
                    f"{row_name}: objective direction mismatch "
                    f"observed={row.get('objective_directions')} "
                    f"expected={expected_contract.get('objective_directions')}"
                ),
                recommended_action="objective direction contract를 동일하게 맞춘다",
            )

        for key, issue_type in (
            ("constraint_count", "constraint_count_mismatch"),
            ("inequality_count", "inequality_count_mismatch"),
            ("equality_count", "equality_count_mismatch"),
        ):
            if int(row.get(key, -1)) == int(expected_contract.get(key, -2)):
                _append_issue(
                    issues,
                    status=PASS,
                    issue_type=f"{key}_match",
                    message=f"{row_name}: {key} matches expected contract",
                    recommended_action="none",
                )
            else:
                _append_issue(
                    issues,
                    status=FAIL,
                    issue_type=issue_type,
                    message=(
                        f"{row_name}: {key} mismatch observed={row.get(key)} "
                        f"expected={expected_contract.get(key)}"
                    ),
                    recommended_action=f"{key} metadata를 동일하게 맞춘다",
                )

        observed_tolerance = float(row.get("tolerance", -1.0))
        expected_tolerance = float(expected_contract.get("tolerance", -2.0))
        if observed_tolerance == expected_tolerance:
            _append_issue(
                issues,
                status=PASS,
                issue_type="tolerance_match",
                message=f"{row_name}: tolerance matches expected contract",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="tolerance_mismatch",
                message=(
                    f"{row_name}: tolerance mismatch observed={observed_tolerance} "
                    f"expected={expected_tolerance}"
                ),
                recommended_action="constraint tolerance를 동일하게 맞춘다",
            )

        for key, issue_type in (
            ("feasibility_policy", "feasibility_policy_mismatch"),
            ("violation_aggregation", "violation_aggregation_mismatch"),
            ("feasible_only_metric_policy", "feasible_only_metric_policy_mismatch"),
        ):
            if row.get(key) == expected_contract.get(key):
                _append_issue(
                    issues,
                    status=PASS,
                    issue_type=f"{key}_match",
                    message=f"{row_name}: {key} matches expected contract",
                    recommended_action="none",
                )
            else:
                _append_issue(
                    issues,
                    status=FAIL,
                    issue_type=issue_type,
                    message=(
                        f"{row_name}: {key} mismatch observed={row.get(key)} "
                        f"expected={expected_contract.get(key)}"
                    ),
                    recommended_action=f"{key}를 동일하게 맞춘다",
                )

        fail_fast_policy = row.get("non_finite_constraint_fail_fast_policy")
        if fail_fast_policy == expected_contract.get("non_finite_constraint_fail_fast_policy"):
            _append_issue(
                issues,
                status=PASS,
                issue_type="non_finite_constraint_fail_fast_policy_match",
                message=f"{row_name}: non-finite constraint fail-fast policy recorded",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="non_finite_constraint_fail_fast_policy_missing_or_mismatch",
                message=(
                    f"{row_name}: non-finite constraint fail-fast policy mismatch "
                    f"observed={fail_fast_policy} "
                    f"expected={expected_contract.get('non_finite_constraint_fail_fast_policy')}"
                ),
                recommended_action="constraint fail-fast policy를 명시적으로 기록한다",
            )

        requested_budget = row.get("requested_budget")
        actual_evaluations = row.get("actual_evaluations")
        if isinstance(requested_budget, int | float) and isinstance(actual_evaluations, int | float):
            if float(requested_budget) <= 0:
                _append_issue(
                    issues,
                    status=FAIL,
                    issue_type="requested_budget_invalid",
                    message=f"{row_name}: requested budget must be > 0",
                    recommended_action="runner budget를 양수로 설정한다",
                )
            else:
                mismatch_rate = abs(float(actual_evaluations) - float(requested_budget)) / float(requested_budget)
                if mismatch_rate == 0.0:
                    _append_issue(
                        issues,
                        status=PASS,
                        issue_type="actual_evaluations_match",
                        message=(
                            f"{row_name}: actual evaluations match requested budget "
                            f"({actual_evaluations}/{requested_budget})"
                        ),
                        recommended_action="none",
                    )
                elif mismatch_rate <= 0.05:
                    _append_issue(
                        issues,
                        status=WARNING,
                        issue_type="actual_evaluations_warning",
                        message=(
                            f"{row_name}: actual evaluations differ slightly from requested budget "
                            f"({actual_evaluations}/{requested_budget})"
                        ),
                        recommended_action="termination contract를 보수적으로 해석한다",
                    )
                else:
                    _append_issue(
                        issues,
                        status=FAIL,
                        issue_type="actual_evaluations_mismatch",
                        message=(
                            f"{row_name}: actual evaluations differ materially from requested budget "
                            f"({actual_evaluations}/{requested_budget})"
                        ),
                        recommended_action="budget control을 다시 맞춘다",
                    )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="actual_evaluations_missing",
                message=f"{row_name}: requested_budget or actual_evaluations is missing",
                recommended_action="artifact schema에 budget/evaluation fields를 채운다",
            )

        _append_issue(
            issues,
            status=PASS,
            issue_type="runtime_difference_not_checked",
            message=f"{row_name}: runtime difference is recorded but not treated as fairness failure",
            recommended_action="none",
        )

    return {
        "status": _status_from_issues(issues),
        "issues": issues,
        "summary_counts": summarize_constrained_fairness(issues),
    }


__all__ = [
    "evaluate_constrained_fairness",
    "summarize_constrained_fairness",
]
