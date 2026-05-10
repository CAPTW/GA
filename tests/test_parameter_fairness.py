from __future__ import annotations

from ga_lab.experiment.parameter_fairness import (
    FAIL,
    METRIC_POSTPROCESSING_ID,
    WARNING,
    evaluate_parameter_fairness,
)


def _benchmark_rows() -> list[dict[str, object]]:
    return [
        {
            "problem": "zdt1",
            "objectives": 2,
            "variables": 6,
            "bounds": [0.0, 1.0],
            "reference_front": "analytic_zdt1",
            "hv_reference_point": [1.05, 10.5],
        }
    ]


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "problem": "zdt1",
        "algorithm": "internal_nsga2",
        "library": "internal",
        "requested_budget": 760,
        "actual_evaluations": 760,
        "runtime_seconds": 0.5,
        "problem_objectives": 2,
        "problem_variables": 6,
        "problem_bounds": [0.0, 1.0],
        "metadata": {
            "objective_count": 2,
            "variable_count": 6,
            "bounds": [0.0, 1.0],
            "hypervolume_reference_point": [1.05, 10.5],
            "reference_front_source": "analytic_zdt1",
            "metric_postprocessing": METRIC_POSTPROCESSING_ID,
            "operator_family": "internal_nsga2_arithmetic_gaussian",
        },
    }
    row.update(overrides)
    return row


def test_parameter_fairness_passes_when_core_contract_matches() -> None:
    result = evaluate_parameter_fairness(
        [_base_row()],
        benchmark_rows=_benchmark_rows(),
        candidate_rows=[{"candidate_id": "candidate_j_h_lite_retry2", "default_changed": False}],
    )

    assert result["status"] == "pass"
    assert result["summary_counts"][FAIL] == 0
    assert result["summary_counts"][WARNING] == 0


def test_parameter_fairness_flags_bounds_mismatch_as_fail() -> None:
    row = _base_row()
    row["metadata"] = dict(row["metadata"]) | {"bounds": [0.0, 2.0]}

    result = evaluate_parameter_fairness([row], benchmark_rows=_benchmark_rows())

    assert result["status"] == "fail"
    assert any(issue["issue_type"] == "bounds_mismatch" for issue in result["issues"])


def test_parameter_fairness_marks_eval_mismatch_above_threshold_as_fail() -> None:
    row = _base_row(actual_evaluations=820)

    result = evaluate_parameter_fairness([row], benchmark_rows=_benchmark_rows())

    assert result["status"] == "fail"
    assert any(issue["issue_type"] == "evaluation_budget_fail" for issue in result["issues"])


def test_parameter_fairness_warns_for_external_operator_family_difference() -> None:
    row = _base_row(algorithm="pymoo_nsga2", library="pymoo")
    row["metadata"] = dict(row["metadata"]) | {"operator_family": "pymoo_standard_sbx_pm"}

    result = evaluate_parameter_fairness([row], benchmark_rows=_benchmark_rows())

    assert result["status"] == "warning"
    assert any(
        issue["issue_type"] == "external_operator_family_difference" and issue["severity"] == WARNING
        for issue in result["issues"]
    )


def test_parameter_fairness_fails_when_candidate_default_changed_is_true() -> None:
    result = evaluate_parameter_fairness(
        [_base_row()],
        benchmark_rows=_benchmark_rows(),
        candidate_rows=[{"candidate_id": "candidate_bad", "default_changed": True}],
    )

    assert result["status"] == "fail"
    assert any(issue["issue_type"] == "candidate_default_changed" for issue in result["issues"])
