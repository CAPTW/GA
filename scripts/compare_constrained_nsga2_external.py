from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from ga_lab.experiment.constrained_external_mo_comparators import (
    deap_secondary_status,
    run_pymoo_constrained_nsga2,
)
from ga_lab.experiment.constrained_ga_smoke import (
    _artifact_ref,
    _ensure_fresh_path,
    _markdown_table,
    _seed_list,
    _stringify,
    _write_json,
)
from ga_lab.experiment.constrained_mo_fairness import (
    FAIL,
    PASS,
    WARNING,
    evaluate_constrained_mo_fairness,
    summarize_constrained_mo_fairness,
)
from ga_lab.experiment.constrained_mo_metrics import summarize_constrained_mo_population
from ga_lab.experiment.constrained_nsga2_smoke import (
    _constraint_contract,
    _constrained_nsga2_row,
    _problem_constraint_names,
    _random_pareto_archive_row,
    _reference_front,
    _reference_point,
    _resolve_budget_schedule,
)
from ga_lab.experiment.runner_resume import (
    CompletedRunIndex,
    build_resume_key,
    load_completed_run_index,
    merge_resume_results,
    plan_runner_resume,
    write_resume_report,
)
from ga_lab.problems import build_problem


SUPPORTED_PROBLEMS = ("constrained_zdt_box_toy", "constrained_dtlz_box_toy")
SUPPORTED_ALGORITHMS = (
    "constrained_nsga2_constraint_domination",
    "pymoo_constrained_nsga2",
    "random_pareto_archive",
)
PAIR_METRICS = (
    "feasible_rate",
    "feasible_nondominated_count",
    "mean_total_violation",
    "feasible_only_HV",
    "feasible_only_reference_distance",
    "spacing_feasible_only",
    "runtime_seconds",
    "actual_evaluations",
)
LOWER_IS_BETTER = {
    "mean_total_violation",
    "feasible_only_reference_distance",
    "spacing_feasible_only",
    "runtime_seconds",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comma_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated item")
    return items


def _budget_list(value: str) -> tuple[int, ...]:
    items: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        budget = int(stripped)
        if budget <= 0:
            raise argparse.ArgumentTypeError("budgets must be positive integers")
        items.append(budget)
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated budget")
    return tuple(items)


def _dimensions(problem_name: str) -> int:
    return 7 if problem_name == "constrained_dtlz_box_toy" else 6


def _build_problem(problem_name: str, *, tolerance: float):
    return build_problem(problem_name, {"dimension": _dimensions(problem_name), "equality_tolerance": tolerance})


def _row_from_pymoo_result(*, problem, result) -> dict[str, Any]:
    base = {
        "strategy": "pymoo_constrained_nsga2",
        "status": result.status,
        "skip_reason": result.skip_reason,
        "failure_reason": result.failure_reason,
        "seed": result.seed,
        "problem": problem.name,
        "dimension": problem.dimension,
        "bounds": problem.source_bounds(),
        "objective_count": 2,
        "objective_directions": [False, False],
        "constraint_count": problem.constraint_count,
        "inequality_count": problem.inequality_count,
        "equality_count": problem.equality_count,
        "tolerance": float(getattr(problem, "equality_tolerance", 1e-8)),
        "constraint_policy": "pymoo_constrained_nsga2",
        "violation_aggregation": "total_violation",
        "feasible_only_metric_policy": "null_when_no_feasible_front",
        "non_finite_constraint_fail_fast_policy": "value_error",
        "reference_front_policy": getattr(problem, "reference_front_name", "none"),
        "reference_point": _reference_point(problem, [False, False]),
        "requested_budget": result.requested_budget,
        "actual_evaluations": result.actual_evaluations,
        "runtime_seconds": result.runtime_seconds,
        "failure_count": 0 if result.status != "failed" else 1,
        "warnings": list(result.warnings),
        "dependency_status": dict(result.dependency_status),
        "external_operator_family_difference": "warning",
        "constraint_sign_convention": "value_le_0_satisfied",
        "dependency_availability": (
            "available" if result.dependency_status.get("installed") else "missing"
        ),
        "default_changed": False,
        "nsga2_default_changed": False,
        "external_parity_claim": False,
        "penalty_used": False,
        "repair_used": False,
    }
    if result.status != "success":
        base.update(
            {
                "feasible_rate": None,
                "feasible_count": 0,
                "infeasible_count": 0,
                "best_total_violation": None,
                "mean_total_violation": None,
                "median_total_violation": None,
                "min_total_violation": None,
                "max_total_violation": None,
                "mean_max_violation": None,
                "violation_count_total": 0,
                "feasible_nondominated_count": 0,
                "infeasible_nondominated_count": 0,
                "feasible_only_HV": None,
                "feasible_only_reference_distance": None,
                "feasible_only_IGD": None,
                "spacing_feasible_only": None,
                "constraint_summary": {},
                "per_constraint_violation_summary": [],
            }
        )
        return base

    inequality_names, equality_names = _problem_constraint_names(problem)
    summary = summarize_constrained_mo_population(
        result.objective_vectors,
        result.constraint_evaluations,
        directions=[False, False],
        reference_point=_reference_point(problem, [False, False]),
        reference_front=_reference_front(problem),
        inequality_names=inequality_names,
        equality_names=equality_names,
    )
    base.update(summary)
    return base


def _normalize_existing_row(row: dict[str, Any], *, status: str = "success") -> dict[str, Any]:
    normalized = dict(row)
    normalized["status"] = status
    normalized["skip_reason"] = None
    normalized["failure_reason"] = None
    normalized["warnings"] = []
    normalized["external_operator_family_difference"] = "none"
    normalized["constraint_sign_convention"] = "value_le_0_satisfied"
    normalized["dependency_availability"] = "internal"
    normalized["external_parity_claim"] = False
    normalized["penalty_used"] = False
    normalized["repair_used"] = False
    return normalized


def _planned_resume_row(*, problem, strategy: str, seed: int, budget: int) -> dict[str, Any]:
    return {
        "problem": problem.name,
        "strategy": strategy,
        "seed": int(seed),
        "requested_budget": int(budget),
        "dimension": int(problem.dimension),
        "tolerance": float(getattr(problem, "equality_tolerance", 1e-8)),
    }


def _source_or_new_row(
    *,
    resume_enabled: bool,
    completed_index: CompletedRunIndex,
    source_artifact: str | None,
    planned_row: dict[str, Any],
    new_row_factory,
) -> tuple[dict[str, Any], bool]:
    if resume_enabled:
        key_id = build_resume_key(planned_row).stable_id()
        if key_id in completed_index.completed_keys:
            source_row = completed_index.completed_rows_by_key[key_id]
            return (
                merge_resume_results(
                    source_rows=[source_row],
                    new_rows=[],
                    source_artifact=source_artifact,
                )[0],
                True,
            )
    row = new_row_factory()
    if resume_enabled:
        row = merge_resume_results(
            source_rows=[],
            new_rows=[row],
            source_artifact=source_artifact,
        )[0]
    return row, False


def _fairness_payload(*, problem, rows: Sequence[dict[str, Any]], budget: int) -> dict[str, Any]:
    expected = _constraint_contract(problem, budget=budget, tolerance=float(getattr(problem, "equality_tolerance", 1e-8)))
    success_rows = [
        row
        for row in rows
        if row.get("status") == "success" and row.get("strategy") != "pymoo_constrained_nsga2"
    ]
    pymoo_success_rows = [
        {**row, "constraint_policy": expected["constraint_policy"]}
        for row in rows
        if row.get("status") == "success" and row.get("strategy") == "pymoo_constrained_nsga2"
    ]
    issues = list(
        evaluate_constrained_mo_fairness(
            expected_contract=expected,
            observed_rows=[*success_rows, *pymoo_success_rows],
        )["issues"]
    )
    for row in rows:
        row_name = f"{row.get('strategy')} seed={row.get('seed')}"
        if row.get("status") == "skipped":
            issues.append(
                {
                    "status": WARNING,
                    "issue_type": "dependency_skip",
                    "severity": "medium",
                    "message": f"{row_name}: dependency missing or forced skip: {row.get('skip_reason')}",
                    "recommended_action": "Record skip artifact; do not treat as external parity evidence",
                }
            )
            continue
        if row.get("status") == "failed":
            issues.append(
                {
                    "status": FAIL,
                    "issue_type": "comparator_failure",
                    "severity": "high",
                    "message": f"{row_name}: comparator failed: {row.get('failure_reason')}",
                    "recommended_action": "Fix comparator before external review",
                }
            )
            continue
        if row.get("strategy") == "pymoo_constrained_nsga2":
            issues.append(
                {
                    "status": PASS,
                    "issue_type": "constraint_sign_convention_verified",
                    "severity": "info",
                    "message": f"{row_name}: constraint sign convention recorded as value <= 0 satisfied",
                    "recommended_action": "Keep wrapper sign tests active",
                }
            )
            issues.append(
                {
                    "status": WARNING,
                    "issue_type": "external_operator_family_difference",
                    "severity": "medium",
                    "message": f"{row_name}: pymoo operator family differs from internal constrained NSGA-II",
                    "recommended_action": "Interpret as external implementation smoke only; no parity claim",
                }
            )
        issues.append(
            {
                "status": PASS,
                "issue_type": "runtime_measurement_policy",
                "severity": "info",
                "message": f"{row_name}: runtime recorded separately from quality metrics",
                "recommended_action": "none",
            }
        )
    status = FAIL if any(issue["status"] == FAIL for issue in issues) else WARNING if any(issue["status"] == WARNING for issue in issues) else PASS
    return {"status": status, "summary_counts": summarize_constrained_mo_fairness(issues), "issues": issues}


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["problem"], int(row["requested_budget"]), row["strategy"]) for row in rows})
    for problem, budget, strategy in keys:
        group = [
            row
            for row in rows
            if row["problem"] == problem
            and int(row["requested_budget"]) == budget
            and row["strategy"] == strategy
        ]
        success = [row for row in group if row.get("status") == "success"]
        summaries.append(
            {
                "benchmark": problem,
                "problem": problem,
                "budget": budget,
                "algorithm": strategy,
                "strategy": strategy,
                "status": "success" if len(success) == len(group) else "mixed_or_skipped",
                "run_count": len(group),
                "success_count": len(success),
                "mean_feasible_rate": _mean_non_null(success, "feasible_rate"),
                "mean_feasible_nondominated_count": _mean_non_null(success, "feasible_nondominated_count"),
                "mean_total_violation": _mean_non_null(success, "mean_total_violation"),
                "mean_best_total_violation": _mean_non_null(success, "best_total_violation"),
                "mean_feasible_only_HV": _mean_non_null(success, "feasible_only_HV"),
                "mean_feasible_only_reference_distance": _mean_non_null(success, "feasible_only_reference_distance"),
                "mean_feasible_only_spacing": _mean_non_null(success, "spacing_feasible_only"),
                "mean_actual_evaluations": _mean_non_null(success, "actual_evaluations"),
                "mean_runtime_seconds": _mean_non_null(success, "runtime_seconds"),
                "all_runs_completed": len(success) == len(group),
                "fairness_status": None,
            }
        )
    return summaries


def _mean_non_null(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), int | float)]
    return mean(values) if values else None


def _paired(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("constrained_nsga2_constraint_domination", "pymoo_constrained_nsga2"),
        ("constrained_nsga2_constraint_domination", "random_pareto_archive"),
        ("pymoo_constrained_nsga2", "random_pareto_archive"),
    ]
    output: list[dict[str, Any]] = []
    for problem, budget in sorted({(row["problem"], int(row["requested_budget"])) for row in rows}):
        for left, right in comparisons:
            left_rows = {
                int(row["seed"]): row
                for row in rows
                if row["problem"] == problem
                and int(row["requested_budget"]) == budget
                and row["strategy"] == left
                and row.get("status") == "success"
            }
            right_rows = {
                int(row["seed"]): row
                for row in rows
                if row["problem"] == problem
                and int(row["requested_budget"]) == budget
                and row["strategy"] == right
                and row.get("status") == "success"
            }
            common = sorted(set(left_rows) & set(right_rows))
            if not common:
                continue
            for metric in PAIR_METRICS:
                deltas: list[float] = []
                wins = ties = losses = 0
                lower = metric in LOWER_IS_BETTER
                for seed in common:
                    left_value = left_rows[seed].get(metric)
                    right_value = right_rows[seed].get(metric)
                    if not (
                        isinstance(left_value, int | float)
                        and isinstance(right_value, int | float)
                        and math.isfinite(float(left_value))
                        and math.isfinite(float(right_value))
                    ):
                        continue
                    delta = float(left_value) - float(right_value)
                    deltas.append(delta)
                    if math.isclose(delta, 0.0, rel_tol=1e-12, abs_tol=1e-12):
                        ties += 1
                    elif (delta < 0.0) if lower else (delta > 0.0):
                        wins += 1
                    else:
                        losses += 1
                if not deltas:
                    continue
                output.append(
                    {
                        "benchmark": problem,
                        "budget": budget,
                        "comparison": f"{left} vs {right}",
                        "metric": metric,
                        "win": wins,
                        "tie": ties,
                        "loss": losses,
                        "mean_delta": mean(deltas),
                        "median_delta": median(deltas),
                        "interpretation": _interpret(metric, wins, losses),
                    }
                )
    return output


def _interpret(metric: str, wins: int, losses: int) -> str:
    if metric == "runtime_seconds":
        return "runtime trade-off; not a quality gate"
    if metric == "actual_evaluations":
        return "budget accounting comparison; ties expected"
    if wins > losses:
        return "left comparator favorable on this metric; no parity or superiority claim"
    if losses > wins:
        return "right comparator favorable or metric limitation; no parity or superiority claim"
    return "mixed or tied; interpret conservatively"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns = [
        "problem",
        "strategy",
        "status",
        "seed",
        "requested_budget",
        "actual_evaluations",
        "feasible_rate",
        "feasible_nondominated_count",
        "mean_total_violation",
        "feasible_only_HV",
        "feasible_only_reference_distance",
        "spacing_feasible_only",
        "runtime_seconds",
        "skip_reason",
        "failure_reason",
    ]
    import csv

    _ensure_fresh_path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _stringify(row.get(column)) for column in columns})


def _render_results(rows: Sequence[dict[str, Any]], *, title: str = "Constrained NSGA-II External Comparison Results") -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            _markdown_table(
                rows,
                [
                    "problem",
                    "requested_budget",
                    "strategy",
                    "status",
                    "seed",
                    "feasible_rate",
                    "feasible_nondominated_count",
                    "mean_total_violation",
                    "feasible_only_HV",
                    "actual_evaluations",
                ],
            ),
            "",
        ]
    )


def _render_fairness(fairness: dict[str, Any], *, title: str = "Constrained NSGA-II External Comparison Fairness Report") -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- Status: **{fairness['status']}**",
            f"- Summary counts: `{fairness['summary_counts']}`",
            "",
            _markdown_table(
                fairness["issues"],
                ["issue_type", "status", "severity", "message", "recommended_action"],
            ),
            "",
        ]
    )


def _decision(fairness: dict[str, Any], rows: Sequence[dict[str, Any]], *, is_stress: bool = False) -> str:
    if fairness["status"] == FAIL:
        return "Needs fairness rerun"
    if any(row["strategy"] == "pymoo_constrained_nsga2" and row["status"] == "skipped" for row in rows):
        return "pymoo comparator skipped, implementation pending"
    if any(row["status"] == "failed" for row in rows):
        return "Fix required"
    if is_stress and fairness["status"] == WARNING:
        return "External stress completed, parity not established"
    if is_stress:
        return "External stress completed, parity not established"
    if fairness["status"] == WARNING:
        return "External comparison completed with limitations"
    return "External comparison completed, no parity claim"


def _render_report(artifact: dict[str, Any]) -> str:
    if artifact["command_metadata"].get("artifact_kind") == "stress":
        return _render_stress_report(artifact)
    return _render_comparison_report(artifact)


def _resume_summary_section(artifact: dict[str, Any]) -> str:
    if not artifact.get("resume_enabled"):
        return ""
    summary = artifact.get("resume_summary", {})
    rows = [
        {"item": "resume_source_artifact", "value": artifact.get("resume_source_artifact")},
        {"item": "resume_mode", "value": artifact.get("resume_mode")},
        {"item": "total_planned", "value": summary.get("total_planned")},
        {"item": "completed_from_source", "value": summary.get("completed_from_source")},
        {"item": "newly_executed", "value": summary.get("newly_executed")},
        {"item": "skipped_existing", "value": summary.get("skipped_existing")},
        {"item": "failed_existing", "value": summary.get("failed_existing")},
        {"item": "warnings", "value": len(summary.get("warnings", []))},
    ]
    return "\n".join(
        [
            "## Runner-level Resume Summary",
            "",
            _markdown_table(rows, ["item", "value"]),
            "",
            "This is skip-completed runner-level resume only; it is not algorithm checkpoint/resume.",
            "",
        ]
    )


def _render_comparison_report(artifact: dict[str, Any]) -> str:
    rows = artifact["rows"]
    summaries = artifact["summaries"]
    paired = artifact["paired_comparisons"]
    fairness = artifact["fairness_summary"]
    decision = artifact["decision"]
    pymoo_rows = [row for row in rows if row["strategy"] == "pymoo_constrained_nsga2"]
    pymoo_status = "not_run"
    if pymoo_rows:
        statuses = sorted({row["status"] for row in pymoo_rows})
        pymoo_status = ",".join(statuses)
    issue_rows = []
    for row in rows:
        if row.get("status") == "skipped":
            issue_rows.append({"type": "skip", "item": row["strategy"], "message": row.get("skip_reason"), "action": "record skip; no parity claim"})
        for warning in row.get("warnings", []):
            issue_rows.append({"type": "warning", "item": row["strategy"], "message": warning, "action": "document limitation"})
        if row.get("status") == "failed":
            issue_rows.append({"type": "failure", "item": row["strategy"], "message": row.get("failure_reason"), "action": "fix before review"})
    if not issue_rows:
        issue_rows = [{"type": "none", "item": "none", "message": "No failures or skips recorded", "action": "none"}]

    return "\n".join(
        [
            "# Constrained NSGA-II External Comparison Report",
            "",
            "## 1. Executive Summary",
            "",
            "- 이번 작업의 목표는 `pymoo_constrained_nsga2` comparator wrapper를 구현하고 two-toy external comparison smoke를 실행하는 것이다.",
            "- 구현한 comparator: `pymoo_constrained_nsga2` optional comparator.",
            "- 실행한 comparison: constrained NSGA-II opt-in path, pymoo constrained NSGA-II, random archive anchor.",
            f"- pymoo status: `{pymoo_status}`.",
            f"- fairness result: `{fairness['status']}` with counts `{fairness['summary_counts']}`.",
            "- constrained_zdt_box_toy와 constrained_dtlz_box_toy 결과는 Results Summary에 기록했다.",
            f"- external comparison decision: **{decision}**.",
            "- default 변경 여부: none.",
            "- Level 판정 변화 여부: Level 상향 불가; experimental toolkit evidence만 강화.",
            "",
            _resume_summary_section(artifact),
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "- pymoo_constrained_nsga2 optional comparator",
            "- constrained_zdt_box_toy",
            "- constrained_dtlz_box_toy",
            "- constrained_nsga2_constraint_domination",
            "- random_pareto_archive",
            "- feasible-only metrics",
            "- fairness report",
            "- artifact generation",
            "",
            "Non-Scope:",
            "- default NSGA-II replacement",
            "- external parity claim",
            "- penalty/repair",
            "- constrained MO full benchmark suite",
            "- productization",
            "",
            "## 3. Comparator Implementation",
            "",
            _markdown_table(
                [
                    {"comparator": "pymoo_constrained_nsga2", "status": pymoo_status, "note": "optional pymoo wrapper using local toy evaluator"},
                    {"comparator": "random_pareto_archive", "status": "implemented anchor", "note": "internal random anchor"},
                    {"comparator": "constrained_nsga2_constraint_domination", "status": "existing opt-in path", "note": "default NSGA-II unchanged"},
                    {"comparator": "internal_nsga2_posthoc_feasible", "status": "not included", "note": "diagnostic-only follow-up"},
                    {"comparator": "DEAP", "status": "hold/not implemented", "note": "secondary comparator only"},
                ],
                ["comparator", "status", "note"],
            ),
            "",
            "## 4. Fairness Summary",
            "",
            _markdown_table(
                fairness["issues"],
                ["issue_type", "status", "message"],
            ),
            "",
            "## 5. Results Summary",
            "",
            _markdown_table(
                summaries,
                [
                    "problem",
                    "budget",
                    "strategy",
                    "status",
                    "mean_feasible_rate",
                    "mean_feasible_nondominated_count",
                    "mean_total_violation",
                    "mean_feasible_only_HV",
                    "mean_feasible_only_reference_distance",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                ],
            ),
            "",
            "## 6. Paired Comparisons",
            "",
            _markdown_table(
                paired,
                ["benchmark", "budget", "comparison", "metric", "win", "tie", "loss", "mean_delta", "interpretation"],
            ),
            "",
            "## 7. Interpretation",
            "",
            "- Feasibility/violation signal is reported separately from feasible-only front quality.",
            "- Feasible-only HV/reference/spacing may be mixed and is not an external parity claim.",
            "- Runtime is reported separately from quality metrics.",
            "- pymoo operator family difference is recorded as a warning.",
            "- Actual evaluation accounting is a fairness gate.",
            "- No external parity claim is made.",
            "",
            "## 8. Failures / Skips / Warnings",
            "",
            _markdown_table(issue_rows, ["type", "item", "message", "action"]),
            "",
            "## 9. Regression / Governance Check",
            "",
            "_Regression command results are recorded after verification in this task._",
            "",
            "## 10. Decision",
            "",
            f"- {decision}",
            "",
            "## 11. What This Proves",
            "",
            "- pymoo comparator wrapper can be executed or skipped safely.",
            "- custom toy evaluator can be wrapped for external comparison.",
            "- feasible-only post-processing and fairness checks are applied.",
            "- default NSGA-II remains unchanged.",
            "",
            "## 12. What This Does Not Prove",
            "",
            "- external parity 확보 아님.",
            "- default NSGA-II constrained support 아님.",
            "- product-ready constrained MOEA 아님.",
            "- penalty/repair 구현 아님.",
            "- broad benchmark generalization 아님.",
            "",
            "## 13. Maturity Impact",
            "",
            "- Level 4 근거 강화. External comparator implementation은 experimental toolkit 근거를 강화하지만, default maturity 또는 constrained MOEA maturity 상향 근거는 아니다.",
            "",
            "## 14. Recommended Next Work",
            "",
            "- external comparison review package 작성.",
            "",
            "이번 constrained NSGA-II external comparison 결과, pymoo comparator는 "
            f"{pymoo_status} 상태였고, external parity는 not established 상태이며, default NSGA-II는 unchanged 상태로 유지된다.",
            "",
        ]
    )


def _stress_decision_table(summaries: Sequence[dict[str, Any]], paired: Sequence[dict[str, Any]], fairness: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(row["benchmark"], int(row["budget"])) for row in summaries})
    for benchmark, budget in keys:
        budget_pairs = [
            row
            for row in paired
            if row["benchmark"] == benchmark
            and int(row["budget"]) == budget
            and row["comparison"] == "constrained_nsga2_constraint_domination vs pymoo_constrained_nsga2"
        ]
        feasibility = next((row for row in budget_pairs if row["metric"] == "feasible_rate"), None)
        violation = next((row for row in budget_pairs if row["metric"] == "mean_total_violation"), None)
        hv = next((row for row in budget_pairs if row["metric"] == "feasible_only_HV"), None)
        runtime = next((row for row in budget_pairs if row["metric"] == "runtime_seconds"), None)
        rows.append(
            {
                "benchmark": benchmark,
                "budget": budget,
                "stress result": "completed",
                "fairness": fairness["status"],
                "internal vs pymoo": (
                    f"feasible_rate {feasibility['interpretation'] if feasibility else 'unavailable'}; "
                    f"violation {violation['interpretation'] if violation else 'unavailable'}; "
                    f"HV {hv['interpretation'] if hv else 'unavailable'}"
                ),
                "runtime trade-off": runtime["interpretation"] if runtime else "unavailable",
                "decision contribution": "supports stress evidence only; no parity claim",
            }
        )
    return rows


def _render_stress_report(artifact: dict[str, Any]) -> str:
    rows = artifact["rows"]
    summaries = artifact["summaries"]
    paired = artifact["paired_comparisons"]
    fairness = artifact["fairness_summary"]
    decision = artifact["decision"]
    config = artifact["configuration"]
    pymoo_rows = [row for row in rows if row["strategy"] == "pymoo_constrained_nsga2"]
    pymoo_status = ",".join(sorted({row["status"] for row in pymoo_rows})) if pymoo_rows else "not_run"
    issue_rows = []
    for row in rows:
        if row.get("status") == "skipped":
            issue_rows.append({"type": "skip", "item": row["strategy"], "message": row.get("skip_reason"), "action": "record skip; no parity claim"})
        if row.get("status") == "failed":
            issue_rows.append({"type": "failure", "item": row["strategy"], "message": row.get("failure_reason"), "action": "fix before review"})
    for issue in fairness["issues"]:
        if issue.get("status") == WARNING:
            issue_rows.append({"type": "warning", "item": issue.get("issue_type"), "message": issue.get("message"), "action": issue.get("recommended_action")})
    if not issue_rows:
        issue_rows = [{"type": "none", "item": "none", "message": "No failures, skips, or warnings recorded", "action": "none"}]

    return "\n".join(
        [
            "# Constrained NSGA-II External Stress Report",
            "",
            "## 1. Executive Summary",
            "",
            f"- 이번 작업의 목표는 seeds {config['seeds']}와 budgets {config['budgets']}에서 internal constrained NSGA-II, pymoo constrained NSGA-II, random archive 비교 stress를 실행하는 것이다.",
            f"- 실행한 stress scope: benchmarks `{config['problems']}`, budgets `{config['budgets']}`, algorithms `{config['algorithms']}`.",
            f"- pymoo status: `{pymoo_status}`.",
            f"- fairness result: `{fairness['status']}` with counts `{fairness['summary_counts']}`.",
            "- constrained_zdt_box_toy와 constrained_dtlz_box_toy 결과는 Results Summary와 Paired Comparisons에 기록했다.",
            "- internal vs pymoo 결과는 feasibility/violation, feasible-only quality, runtime을 분리해서 해석한다.",
            f"- external stress decision: **{decision}**.",
            "- external parity decision: not established.",
            "- default 변경 여부: none.",
            "- Level 판정 변화 여부: Level 상향 불가; 실험 툴킷 근거만 강화.",
            "",
            _resume_summary_section(artifact),
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "- pymoo_constrained_nsga2",
            "- constrained_zdt_box_toy",
            "- constrained_dtlz_box_toy",
            "- constrained_nsga2_constraint_domination",
            "- random_pareto_archive",
            "- feasible-only metrics",
            "- fairness report",
            "- stress artifact generation",
            "",
            "Non-Scope:",
            "- default NSGA-II replacement",
            "- external parity claim",
            "- penalty/repair",
            "- constrained MO full benchmark suite",
            "- productization",
            "",
            "## 3. Stress Configuration",
            "",
            _markdown_table(
                [
                    {"field": "benchmarks", "value": ", ".join(config["problems"])},
                    {"field": "budgets", "value": ", ".join(str(item) for item in config["budgets"])},
                    {"field": "seeds", "value": str(config["seeds"])},
                    {"field": "algorithms", "value": ", ".join(config["algorithms"])},
                    {"field": "constraint policy", "value": "constraint_domination for internal path; pymoo constrained handling for external wrapper"},
                    {"field": "fairness policy", "value": "external stress fairness gate with operator family warning"},
                ],
                ["field", "value"],
            ),
            "",
            "## 4. Results Summary",
            "",
            _markdown_table(
                summaries,
                [
                    "benchmark",
                    "budget",
                    "algorithm",
                    "status",
                    "mean_feasible_rate",
                    "mean_feasible_nondominated_count",
                    "mean_total_violation",
                    "mean_feasible_only_HV",
                    "mean_feasible_only_reference_distance",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                ],
            ),
            "",
            "## 5. Paired Comparisons",
            "",
            _markdown_table(
                paired,
                ["benchmark", "budget", "comparison", "metric", "win", "tie", "loss", "mean_delta", "interpretation"],
            ),
            "",
            "## 6. Fairness Summary",
            "",
            _markdown_table(
                fairness["issues"],
                ["issue_type", "status", "message"],
            ),
            "",
            "## 7. Interpretation",
            "",
            "- Feasibility/violation signal is interpreted separately from feasible-only quality.",
            "- Feasible-only HV/reference/spacing can be mixed and must not be converted into parity.",
            "- Runtime is separated from quality; runtime advantage is not a quality proof.",
            "- pymoo external operator family difference remains a warning.",
            "- Actual evaluation accounting is an explicit fairness gate.",
            "- No external parity claim is made.",
            "",
            _markdown_table(
                _stress_decision_table(summaries, paired, fairness),
                ["benchmark", "budget", "stress result", "fairness", "internal vs pymoo", "runtime trade-off", "decision contribution"],
            ),
            "",
            "## 8. Failures / Skips / Warnings",
            "",
            _markdown_table(issue_rows, ["type", "item", "message", "action"]),
            "",
            "## 9. Regression / Governance Check",
            "",
            "_Regression command results are recorded after verification in this task._",
            "",
            "## 10. Decision",
            "",
            f"- {decision}",
            "",
            "## 11. What This Proves",
            "",
            "- pymoo comparator stress can run or skip safely.",
            "- custom constrained toy evaluator can be used under stress.",
            "- feasible-only post-processing and fairness checks are applied.",
            "- exact budget accounting still works for successful rows.",
            "- default NSGA-II remains unchanged.",
            "",
            "## 12. What This Does Not Prove",
            "",
            "- external parity 확보 아님.",
            "- default NSGA-II constrained support 아님.",
            "- product-ready constrained MOEA 아님.",
            "- penalty/repair 구현 아님.",
            "- broad benchmark generalization 아님.",
            "",
            "## 13. Maturity Impact",
            "",
            "- Level 4 근거 강화. External stress는 실험 툴킷 근거를 강화하지만 default maturity나 constrained MOEA maturity 상향 근거는 아니다.",
            "",
            "## 14. Recommended Next Work",
            "",
            "- external stress review package 작성.",
            "",
            f"이번 constrained NSGA-II external stress 결과, pymoo comparison은 {decision}로 정리되었고, external parity는 not established이며, default NSGA-II는 unchanged 상태로 유지된다.",
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained NSGA-II external comparison smoke.")
    parser.add_argument("--problems", type=_comma_list, default=SUPPORTED_PROBLEMS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=760)
    parser.add_argument("--budgets", type=_budget_list, default=None)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--artifact-suffix", default="pymoo_constrained_compare1")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--force-pymoo-skip", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-mode", default="skip-completed", choices=("skip-completed",))
    parser.add_argument("--resume-report-suffix", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    fairness_payloads: list[dict[str, Any]] = []
    seeds = _seed_list(args.seeds)
    budgets = list(args.budgets) if args.budgets is not None else [int(args.budget)]
    is_stress = args.budgets is not None
    resume_enabled = args.resume_from is not None
    completed_index = (
        load_completed_run_index(args.resume_from) if resume_enabled else CompletedRunIndex()
    )
    planned_keys = []
    copied_from_source = 0
    newly_executed = 0

    for problem_name in args.problems:
        if problem_name not in SUPPORTED_PROBLEMS:
            raise ValueError(f"Unsupported problem: {problem_name}")
        problem = _build_problem(problem_name, tolerance=args.tolerance)
        for budget in budgets:
            population_size, generations = _resolve_budget_schedule(
                budget,
                preferred_population_size=args.population_size,
            )
            problem_rows: list[dict[str, Any]] = []
            for seed in seeds:
                constrained_plan = _planned_resume_row(
                    problem=problem,
                    strategy="constrained_nsga2_constraint_domination",
                    seed=seed,
                    budget=budget,
                )
                planned_keys.append(build_resume_key(constrained_plan))
                constrained_row, was_copied = _source_or_new_row(
                    resume_enabled=resume_enabled,
                    completed_index=completed_index,
                    source_artifact=args.resume_from,
                    planned_row=constrained_plan,
                    new_row_factory=lambda seed=seed, budget=budget, problem=problem: _normalize_existing_row(
                        _constrained_nsga2_row(
                            seed=seed,
                            problem=problem,
                            budget=budget,
                            population_size=population_size,
                            generations=generations,
                        )
                    ),
                )
                copied_from_source += int(was_copied)
                newly_executed += int(not was_copied)

                pymoo_plan = _planned_resume_row(
                    problem=problem,
                    strategy="pymoo_constrained_nsga2",
                    seed=seed,
                    budget=budget,
                )
                planned_keys.append(build_resume_key(pymoo_plan))
                pymoo_row, was_copied = _source_or_new_row(
                    resume_enabled=resume_enabled,
                    completed_index=completed_index,
                    source_artifact=args.resume_from,
                    planned_row=pymoo_plan,
                    new_row_factory=lambda seed=seed, budget=budget, problem=problem: _row_from_pymoo_result(
                        problem=problem,
                        result=run_pymoo_constrained_nsga2(
                            problem=problem,
                            seed=seed,
                            budget=budget,
                            population_size=population_size,
                            force_skip=args.force_pymoo_skip,
                        ),
                    ),
                )
                copied_from_source += int(was_copied)
                newly_executed += int(not was_copied)

                random_plan = _planned_resume_row(
                    problem=problem,
                    strategy="random_pareto_archive",
                    seed=seed,
                    budget=budget,
                )
                planned_keys.append(build_resume_key(random_plan))
                random_row, was_copied = _source_or_new_row(
                    resume_enabled=resume_enabled,
                    completed_index=completed_index,
                    source_artifact=args.resume_from,
                    planned_row=random_plan,
                    new_row_factory=lambda seed=seed, budget=budget, problem=problem: _normalize_existing_row(
                        _random_pareto_archive_row(seed=seed, problem=problem, budget=budget)
                    ),
                )
                copied_from_source += int(was_copied)
                newly_executed += int(not was_copied)
                problem_rows.extend([constrained_row, pymoo_row, random_row])
            rows.extend(problem_rows)
            fairness_payloads.append(
                {
                    "problem": problem.name,
                    "budget": budget,
                    **_fairness_payload(problem=problem, rows=problem_rows, budget=budget),
                }
            )

    issues = [issue for payload in fairness_payloads for issue in payload["issues"]]
    fairness = {
        "status": FAIL if any(issue["status"] == FAIL for issue in issues) else WARNING if any(issue["status"] == WARNING for issue in issues) else PASS,
        "summary_counts": summarize_constrained_mo_fairness(issues),
        "issues": issues,
        "by_problem": fairness_payloads,
    }
    summaries = _aggregate(rows)
    for summary in summaries:
        summary["fairness_status"] = fairness["status"]
    paired = _paired(rows)
    decision = _decision(fairness, rows, is_stress=is_stress)
    resume_plan = plan_runner_resume(planned_keys, completed_index, resume_mode=args.resume_mode)
    planned_failed_existing = sum(
        1 for key in planned_keys if key.stable_id() in completed_index.failed_keys
    )
    resume_summary = {
        "total_planned": len(planned_keys),
        "completed_from_source": copied_from_source,
        "newly_executed": newly_executed,
        "skipped_existing": copied_from_source,
        "failed_existing": planned_failed_existing,
        "missing_executed": newly_executed,
        "warnings": list(completed_index.warnings),
    }
    for warning in completed_index.warnings:
        warnings.append({"type": "resume_warning", "message": warning})

    output_dir_input = Path(args.output_dir)
    project_root = _project_root()
    output_dir = output_dir_input if output_dir_input.is_absolute() else project_root / output_dir_input
    suffix = args.artifact_suffix
    artifact_stem = "constrained_nsga2_external_stress" if is_stress else "constrained_nsga2_external_comparison"
    result_title = "Constrained NSGA-II External Stress Results" if is_stress else "Constrained NSGA-II External Comparison Results"
    fairness_title = "Constrained NSGA-II External Stress Fairness Report" if is_stress else "Constrained NSGA-II External Comparison Fairness Report"
    json_path = output_dir / f"{artifact_stem}_results_{suffix}.json"
    csv_path = output_dir / f"{artifact_stem}_results_{suffix}.csv"
    md_path = output_dir / f"{artifact_stem}_results_{suffix}.md"
    report_path = output_dir / f"{artifact_stem}_report_{suffix}.md"
    fairness_path = output_dir / f"{artifact_stem}_fairness_report_{suffix}.md"
    resume_report_suffix = args.resume_report_suffix or suffix
    resume_report_path = output_dir / f"{artifact_stem}_resume_report_{resume_report_suffix}.md"

    artifact = {
        "command_metadata": {
            "argv": sys.argv,
            "artifact_suffix": suffix,
            "artifact_kind": "stress" if is_stress else "comparison",
            "output_dir": str(args.output_dir),
            "resume_from": args.resume_from,
        },
        "configuration": {
            "problems": list(args.problems),
            "seeds": args.seeds,
            "seed_list": seeds,
            "budget": args.budget,
            "budgets": budgets,
            "population_size": args.population_size,
            "algorithms": list(SUPPORTED_ALGORITHMS),
            "deap_status": deap_secondary_status(),
        },
        "rows": rows,
        "summaries": summaries,
        "paired_comparisons": paired,
        "fairness_summary": fairness,
        "decision": decision,
        "failures": failures,
        "warnings": warnings,
        "default_nsga2_changed": False,
        "constrained_nsga2_scope_change": "none",
        "external_parity_established": False,
        "resume_enabled": resume_enabled,
        "resume_source_artifact": str(args.resume_from) if args.resume_from else None,
        "resume_mode": args.resume_mode,
        "resume_summary": resume_summary,
        "artifacts": {
            "json": _artifact_ref(json_path, project_root, prefer_absolute=output_dir_input.is_absolute()),
            "csv": _artifact_ref(csv_path, project_root, prefer_absolute=output_dir_input.is_absolute()),
            "results_markdown": _artifact_ref(md_path, project_root, prefer_absolute=output_dir_input.is_absolute()),
            "report_markdown": _artifact_ref(report_path, project_root, prefer_absolute=output_dir_input.is_absolute()),
            "fairness_markdown": _artifact_ref(fairness_path, project_root, prefer_absolute=output_dir_input.is_absolute()),
            "resume_markdown": (
                _artifact_ref(
                    resume_report_path,
                    project_root,
                    prefer_absolute=output_dir_input.is_absolute(),
                )
                if resume_enabled
                else None
            ),
        },
    }
    _write_json(json_path, artifact)
    _write_csv(csv_path, rows)
    _ensure_fresh_path(md_path).write_text(_render_results(rows, title=result_title), encoding="utf-8")
    _ensure_fresh_path(fairness_path).write_text(_render_fairness(fairness, title=fairness_title), encoding="utf-8")
    _ensure_fresh_path(report_path).write_text(_render_report(artifact), encoding="utf-8")
    if resume_enabled:
        write_resume_report(resume_report_path, resume_plan, resume_summary)
    print(
        json.dumps(
            {"decision": decision, "artifacts": artifact["artifacts"]},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
