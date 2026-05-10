from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


METRIC_POSTPROCESSING_ID = "ga_lab.experiment.mo_metrics.evaluate_front_metrics"

PASS = "pass"
WARNING = "warning"
FAIL = "fail"


@dataclass(frozen=True, slots=True)
class FairnessIssue:
    status: str
    issue_type: str
    algorithm: str
    problem: str
    message: str
    severity: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fairness_policy_rows() -> list[dict[str, str]]:
    return [
        {
            "fairness_item": "actual_evaluations mismatch",
            "pass_rule": "requested budget와 actual evaluations가 동일하다",
            "warning_rule": "mismatch rate가 5% 이하이다",
            "fail_rule": "mismatch rate가 5%를 초과한다",
        },
        {
            "fairness_item": "problem/objective/bounds consistency",
            "pass_rule": "problem, objective count, variable count, bounds가 benchmark 정의와 일치한다",
            "warning_rule": "해당 없음",
            "fail_rule": "problem dimension 또는 bounds가 benchmark 정의와 다르다",
        },
        {
            "fairness_item": "metric post-processing",
            "pass_rule": f"모든 알고리즘이 `{METRIC_POSTPROCESSING_ID}`를 사용한다",
            "warning_rule": "해당 없음",
            "fail_rule": "metric post-processing 함수 경로가 다르다",
        },
        {
            "fairness_item": "candidate default_changed",
            "pass_rule": "candidate metadata에서 default_changed=false다",
            "warning_rule": "해당 없음",
            "fail_rule": "candidate metadata에서 default_changed=true다",
        },
        {
            "fairness_item": "external operator family difference",
            "pass_rule": "internal과 같은 operator family다",
            "warning_rule": "external comparator라 operator family가 다르지만 budget/problem/metric contract는 유지된다",
            "fail_rule": "해당 없음",
        },
    ]


def _normalize_bounds(value: Any) -> tuple[float, float] | None:
    if isinstance(value, tuple | list) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _normalize_vector(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, tuple | list):
        converted: list[float] = []
        for item in value:
            try:
                converted.append(float(item))
            except (TypeError, ValueError):
                return None
        return tuple(converted)
    return None


def _append_issue(
    issues: list[FairnessIssue],
    *,
    status: str,
    issue_type: str,
    algorithm: str,
    problem: str,
    message: str,
    recommended_action: str,
) -> None:
    issues.append(
        FairnessIssue(
            status=status,
            issue_type=issue_type,
            algorithm=algorithm,
            problem=problem,
            message=message,
            severity=status,
            recommended_action=recommended_action,
        )
    )


def _benchmark_map(benchmark_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["problem"]): row for row in benchmark_rows}


def evaluate_parameter_fairness(
    raw_rows: list[dict[str, Any]],
    *,
    benchmark_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    issues: list[FairnessIssue] = []
    benchmarks = _benchmark_map(benchmark_rows)
    candidate_rows = candidate_rows or []

    for candidate in candidate_rows:
        candidate_id = str(candidate.get("candidate_id", "unknown_candidate"))
        default_changed = candidate.get("default_changed")
        if default_changed is True:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="candidate_default_changed",
                algorithm=candidate_id,
                problem="*",
                message="candidate metadata reports default_changed=true",
                recommended_action="candidate metadata를 false로 되돌리고 opt-in 경로로만 유지한다",
            )
        else:
            _append_issue(
                issues,
                status=PASS,
                issue_type="candidate_default_changed",
                algorithm=candidate_id,
                problem="*",
                message="candidate metadata keeps default_changed=false",
                recommended_action="none",
            )

    for row in raw_rows:
        algorithm = str(row.get("algorithm", "unknown"))
        problem = str(row.get("problem", "unknown"))
        metadata = dict(row.get("metadata", {}))
        benchmark = benchmarks.get(problem)
        if benchmark is None:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="missing_benchmark_definition",
                algorithm=algorithm,
                problem=problem,
                message="benchmark definition is missing for this problem row",
                recommended_action="runner benchmark_rows와 selected problems를 다시 맞춘다",
            )
            continue

        observed_objectives = metadata.get("objective_count", row.get("problem_objectives"))
        expected_objectives = benchmark.get("objectives")
        if observed_objectives == expected_objectives:
            _append_issue(
                issues,
                status=PASS,
                issue_type="objective_count_match",
                algorithm=algorithm,
                problem=problem,
                message=f"objective_count matches benchmark ({observed_objectives})",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="objective_count_mismatch",
                algorithm=algorithm,
                problem=problem,
                message=(
                    f"objective_count mismatch: observed={observed_objectives}, "
                    f"expected={expected_objectives}"
                ),
                recommended_action="problem definition과 runner metadata를 다시 정렬한다",
            )

        observed_variables = metadata.get("variable_count", row.get("problem_variables"))
        expected_variables = benchmark.get("variables")
        if observed_variables == expected_variables:
            _append_issue(
                issues,
                status=PASS,
                issue_type="variable_count_match",
                algorithm=algorithm,
                problem=problem,
                message=f"variable_count matches benchmark ({observed_variables})",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="variable_count_mismatch",
                algorithm=algorithm,
                problem=problem,
                message=(
                    f"variable_count mismatch: observed={observed_variables}, "
                    f"expected={expected_variables}"
                ),
                recommended_action="problem adapter와 config genome_length를 다시 확인한다",
            )

        observed_bounds = _normalize_bounds(metadata.get("bounds", row.get("problem_bounds")))
        expected_bounds = _normalize_bounds(benchmark.get("bounds"))
        if observed_bounds == expected_bounds:
            _append_issue(
                issues,
                status=PASS,
                issue_type="bounds_match",
                algorithm=algorithm,
                problem=problem,
                message=f"bounds match benchmark {observed_bounds}",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="bounds_mismatch",
                algorithm=algorithm,
                problem=problem,
                message=f"bounds mismatch: observed={observed_bounds}, expected={expected_bounds}",
                recommended_action="internal/external problem bounds를 동일하게 맞춘다",
            )

        observed_hv = _normalize_vector(
            metadata.get("hypervolume_reference_point", row.get("hv_reference_point"))
        )
        expected_hv = _normalize_vector(benchmark.get("hv_reference_point"))
        if observed_hv == expected_hv:
            _append_issue(
                issues,
                status=PASS,
                issue_type="hv_reference_match",
                algorithm=algorithm,
                problem=problem,
                message=f"hypervolume reference point matches benchmark {observed_hv}",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="hv_reference_mismatch",
                algorithm=algorithm,
                problem=problem,
                message=(
                    f"hypervolume reference mismatch: observed={observed_hv}, "
                    f"expected={expected_hv}"
                ),
                recommended_action="metric post-processing reference point를 동일하게 맞춘다",
            )

        observed_front = metadata.get("reference_front_source", row.get("reference_front_source"))
        expected_front = benchmark.get("reference_front")
        if observed_front == expected_front:
            _append_issue(
                issues,
                status=PASS,
                issue_type="reference_front_match",
                algorithm=algorithm,
                problem=problem,
                message=f"reference front source matches benchmark ({observed_front})",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="reference_front_mismatch",
                algorithm=algorithm,
                problem=problem,
                message=(
                    f"reference front mismatch: observed={observed_front}, "
                    f"expected={expected_front}"
                ),
                recommended_action="reference front source를 같은 helper로 통일한다",
            )

        metric_postprocessing = metadata.get(
            "metric_postprocessing",
            row.get("metric_postprocessing"),
        )
        if metric_postprocessing == METRIC_POSTPROCESSING_ID:
            _append_issue(
                issues,
                status=PASS,
                issue_type="metric_postprocessing_match",
                algorithm=algorithm,
                problem=problem,
                message="metric post-processing function matches the shared contract",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="metric_postprocessing_mismatch",
                algorithm=algorithm,
                problem=problem,
                message=(
                    f"metric post-processing mismatch: observed={metric_postprocessing}, "
                    f"expected={METRIC_POSTPROCESSING_ID}"
                ),
                recommended_action="모든 알고리즘 결과를 shared mo_metrics helper로 후처리한다",
            )

        requested_budget = row.get("requested_budget")
        actual_evaluations = row.get("actual_evaluations")
        if isinstance(requested_budget, int | float) and isinstance(actual_evaluations, int | float):
            if float(requested_budget) <= 0:
                mismatch_rate = None
            else:
                mismatch_rate = abs(float(actual_evaluations) - float(requested_budget)) / float(
                    requested_budget
                )
            if mismatch_rate is None or mismatch_rate == 0.0:
                _append_issue(
                    issues,
                    status=PASS,
                    issue_type="evaluation_budget_match",
                    algorithm=algorithm,
                    problem=problem,
                    message=(
                        f"actual evaluations match requested budget "
                        f"({actual_evaluations}/{requested_budget})"
                    ),
                    recommended_action="none",
                )
            elif mismatch_rate <= 0.05:
                _append_issue(
                    issues,
                    status=WARNING,
                    issue_type="evaluation_budget_warning",
                    algorithm=algorithm,
                    problem=problem,
                    message=(
                        f"actual evaluations differ slightly from budget "
                        f"({actual_evaluations}/{requested_budget}, mismatch_rate={mismatch_rate:.4f})"
                    ),
                    recommended_action="paired comparison 해석을 보수적으로 유지한다",
                )
            else:
                _append_issue(
                    issues,
                    status=FAIL,
                    issue_type="evaluation_budget_fail",
                    algorithm=algorithm,
                    problem=problem,
                    message=(
                        f"actual evaluations differ materially from budget "
                        f"({actual_evaluations}/{requested_budget}, mismatch_rate={mismatch_rate:.4f})"
                    ),
                    recommended_action="budget control 또는 termination contract를 다시 맞춘다",
                )
        else:
            _append_issue(
                issues,
                status=FAIL,
                issue_type="evaluation_budget_missing",
                algorithm=algorithm,
                problem=problem,
                message="requested_budget or actual_evaluations is missing",
                recommended_action="runner result schema에 budget/evaluation fields를 채운다",
            )

        runtime_seconds = row.get("runtime_seconds")
        if isinstance(runtime_seconds, int | float):
            _append_issue(
                issues,
                status=PASS,
                issue_type="runtime_available",
                algorithm=algorithm,
                problem=problem,
                message=f"runtime is recorded ({float(runtime_seconds):.6f}s)",
                recommended_action="none",
            )
        else:
            _append_issue(
                issues,
                status=WARNING,
                issue_type="runtime_missing",
                algorithm=algorithm,
                problem=problem,
                message="runtime_seconds is missing",
                recommended_action="runtime measurement를 result schema에 유지한다",
            )

        operator_family = str(metadata.get("operator_family", "unknown"))
        if row.get("library") in {"pymoo", "deap"}:
            _append_issue(
                issues,
                status=WARNING,
                issue_type="external_operator_family_difference",
                algorithm=algorithm,
                problem=problem,
                message=f"external comparator uses a different operator family ({operator_family})",
                recommended_action="strict operator parity로 해석하지 말고 budget/problem/metric fairness만 주장한다",
            )
        else:
            _append_issue(
                issues,
                status=PASS,
                issue_type="operator_family_recorded",
                algorithm=algorithm,
                problem=problem,
                message=f"operator family recorded as {operator_family}",
                recommended_action="none",
            )

        default_changed = metadata.get("default_changed")
        candidate_id = metadata.get("candidate_id")
        if candidate_id is not None:
            if default_changed is True:
                _append_issue(
                    issues,
                    status=FAIL,
                    issue_type="runtime_candidate_default_changed",
                    algorithm=algorithm,
                    problem=problem,
                    message=f"candidate runtime metadata leaked default_changed=true for {candidate_id}",
                    recommended_action="candidate runner를 opt-in only로 되돌린다",
                )
            else:
                _append_issue(
                    issues,
                    status=PASS,
                    issue_type="runtime_candidate_default_unchanged",
                    algorithm=algorithm,
                    problem=problem,
                    message=f"candidate runtime metadata keeps default_changed={default_changed}",
                    recommended_action="none",
                )

    summary_counts = {PASS: 0, WARNING: 0, FAIL: 0}
    for issue in issues:
        summary_counts[issue.status] += 1
    overall_status = FAIL if summary_counts[FAIL] else WARNING if summary_counts[WARNING] else PASS
    return {
        "status": overall_status,
        "issues": [issue.to_dict() for issue in issues],
        "summary_counts": summary_counts,
        "policy_rows": fairness_policy_rows(),
    }
