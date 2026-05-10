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


def summarize_constrained_mo_fairness(issues: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        PASS: sum(1 for issue in issues if issue["status"] == PASS),
        WARNING: sum(1 for issue in issues if issue["status"] == WARNING),
        FAIL: sum(1 for issue in issues if issue["status"] == FAIL),
    }


def evaluate_constrained_mo_fairness(
    *,
    expected_contract: dict[str, Any],
    observed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    for row in observed_rows:
        row_name = str(row.get("strategy", row.get("algorithm", "unknown_strategy")))

        checks = (
            ("problem", "problem_mismatch"),
            ("objective_count", "objective_count_mismatch"),
            ("dimension", "dimension_mismatch"),
            ("constraint_count", "constraint_count_mismatch"),
            ("inequality_count", "inequality_count_mismatch"),
            ("equality_count", "equality_count_mismatch"),
            ("tolerance", "tolerance_mismatch"),
            ("constraint_policy", "constraint_policy_mismatch"),
            ("violation_aggregation", "violation_aggregation_mismatch"),
            ("feasible_only_metric_policy", "feasible_only_metric_policy_mismatch"),
            (
                "non_finite_constraint_fail_fast_policy",
                "non_finite_constraint_fail_fast_policy_mismatch",
            ),
        )
        for key, issue_type in checks:
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
                    recommended_action=f"Align {key} metadata with constrained MO contract",
                )

        if list(row.get("objective_directions", [])) == list(
            expected_contract.get("objective_directions", [])
        ):
            _append_issue(
                issues,
                status=PASS,
                issue_type="objective_direction_match",
                message=f"{row_name}: objective directions match expected contract",
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
                recommended_action="Preserve objective direction metadata for constrained MO runs",
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
                recommended_action="Align constrained MO bounds metadata",
            )

        reference_checks = (
            ("reference_front_policy", "reference_front_policy_mismatch"),
            ("reference_point", "reference_point_mismatch"),
        )
        for key, issue_type in reference_checks:
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
                    status=WARNING,
                    issue_type=issue_type,
                    message=(
                        f"{row_name}: {key} mismatch observed={row.get(key)} "
                        f"expected={expected_contract.get(key)}"
                    ),
                    recommended_action=f"Check {key} metadata before broader constrained MO claims",
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
                    recommended_action="Use a positive requested budget",
                )
            else:
                mismatch_rate = abs(float(actual_evaluations) - float(requested_budget)) / float(
                    requested_budget
                )
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
                        recommended_action="Review constrained MO budget accounting",
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
                        recommended_action="Fix constrained MO budget accounting before review",
                    )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="actual_evaluations_missing",
                message=f"{row_name}: requested_budget or actual_evaluations missing",
                recommended_action="Record requested_budget and actual_evaluations on every row",
            )

    return {
        "status": _status_from_issues(issues),
        "summary_counts": summarize_constrained_mo_fairness(issues),
        "issues": issues,
    }


__all__ = [
    "FAIL",
    "PASS",
    "WARNING",
    "evaluate_constrained_mo_fairness",
    "summarize_constrained_mo_fairness",
]
