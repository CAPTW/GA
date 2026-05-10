from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from ga_lab.experiment.constrained_ga_smoke import (
    _artifact_ref,
    _build_failures_and_warnings_rows,
    _build_fairness_rows,
    _ensure_fresh_path,
    _markdown_table,
    _seed_list,
    _stringify,
    _write_csv,
    _write_json,
)
from ga_lab.experiment.constrained_mo_fairness import evaluate_constrained_mo_fairness
from ga_lab.experiment.constrained_nsga2_smoke import (
    DEFAULT_STRATEGIES,
    SUPPORTED_PROBLEMS,
    SUPPORTED_STRATEGIES,
    _constrained_nsga2_row,
    _constraint_contract,
    _random_pareto_archive_row,
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
from ga_lab.experiment.runner_parallel import (
    ParallelExecutionConfig,
    execute_runner_rows,
    parallel_summary_to_dict,
)
from ga_lab.problems import build_problem


DEFAULT_DIMENSIONS = {
    "constrained_zdt_box_toy": 6,
    "constrained_dtlz_box_toy": 7,
}

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


@dataclass(slots=True)
class ConstrainedNSGA2StressConfig:
    problems: tuple[str, ...] = ("constrained_zdt_box_toy", "constrained_dtlz_box_toy")
    dimensions: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_DIMENSIONS))
    seeds: int | Sequence[int] = 20
    budgets: tuple[int, ...] = (760, 1500)
    tolerance: float = 1e-8
    population_size: int | None = None
    artifact_suffix: str = "constrained_nsga2_stress1"
    output_dir: str = "artifacts"
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    constraint_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    resume_from: str | Path | None = None
    resume_mode: str = "skip-completed"
    resume_report_suffix: str | None = None
    row_execution_backend: str = "serial"
    row_execution_workers: int = 1
    parallel_fail_fast: bool = False
    parallel_timeout_seconds: float | None = None
    allow_process_backend: bool = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_problem_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _dimension_for_problem(config: ConstrainedNSGA2StressConfig, problem_name: str) -> int:
    if problem_name in config.dimensions:
        return int(config.dimensions[problem_name])
    return int(DEFAULT_DIMENSIONS[problem_name])


def _build_problem_for_stress(
    config: ConstrainedNSGA2StressConfig,
    problem_name: str,
):
    options: dict[str, Any] = {
        "dimension": _dimension_for_problem(config, problem_name),
        "equality_tolerance": config.tolerance,
    }
    options.update(config.constraint_overrides.get(problem_name, {}))
    return build_problem(problem_name, options)


def _non_null_mean(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return mean(numeric)


def _row_per_constraint_fields(row: dict[str, Any]) -> None:
    satisfaction: dict[str, float | None] = {}
    mean_violation: dict[str, float | None] = {}
    max_violation: dict[str, float | None] = {}
    for item in row.get("per_constraint_violation_summary", []):
        constraint = str(item.get("constraint", item.get("constraint_name", "unknown")))
        satisfaction[constraint] = item.get("satisfaction_rate")
        mean_violation[constraint] = item.get("mean_violation")
        max_violation[constraint] = item.get("max_violation")
    row["per_constraint_satisfaction_rate"] = satisfaction
    row["per_constraint_mean_violation"] = mean_violation
    row["per_constraint_max_violation"] = max_violation


def _run_strategy(
    *,
    strategy: str,
    seed: int,
    problem,
    budget: int,
    population_size: int,
    generations: int,
) -> dict[str, Any]:
    if strategy == "constrained_nsga2_constraint_domination":
        row = _constrained_nsga2_row(
            seed=seed,
            problem=problem,
            budget=budget,
            population_size=population_size,
            generations=generations,
        )
    elif strategy == "random_pareto_archive":
        row = _random_pareto_archive_row(seed=seed, problem=problem, budget=budget)
    else:
        raise ValueError(f"Unsupported constrained NSGA-II stress strategy: {strategy}")
    row.setdefault("status", "success")
    row.setdefault("skip_reason", None)
    row.setdefault("failure_reason", None)
    row.setdefault("warnings", [])
    row["benchmark"] = row["problem"]
    _row_per_constraint_fields(row)
    return row


def _stress_task_key(task: dict[str, Any]) -> str:
    return str(task["resume_key_id"])


def _execute_stress_task(task: dict[str, Any]) -> dict[str, Any]:
    problem_name = str(task["problem_name"])
    task_config = ConstrainedNSGA2StressConfig(
        problems=(problem_name,),
        dimensions={problem_name: int(task["dimension"])},
        seeds=(int(task["seed"]),),
        budgets=(int(task["budget"]),),
        tolerance=float(task["tolerance"]),
        population_size=int(task["population_size"]),
        strategies=(str(task["strategy"]),),
        constraint_overrides=dict(task.get("constraint_overrides", {})),
    )
    problem = _build_problem_for_stress(task_config, problem_name)
    return _run_strategy(
        strategy=str(task["strategy"]),
        seed=int(task["seed"]),
        problem=problem,
        budget=int(task["budget"]),
        population_size=int(task["population_size"]),
        generations=int(task["generations"]),
    )


def _planned_resume_row(
    *,
    problem,
    strategy: str,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    return {
        "problem": problem.name,
        "benchmark": problem.name,
        "strategy": strategy,
        "seed": int(seed),
        "requested_budget": int(budget),
        "dimension": int(problem.dimension),
        "tolerance": float(getattr(problem, "equality_tolerance", 1e-8)),
    }


def _resume_source_or_run(
    *,
    planned_row: dict[str, Any],
    completed_index: CompletedRunIndex,
    resume_enabled: bool,
    source_artifact: str | Path | None,
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


def _fairness_by_benchmark_budget(
    *,
    config: ConstrainedNSGA2StressConfig,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    for problem_name in config.problems:
        normalized_problem = _normalize_problem_name(problem_name)
        problem = _build_problem_for_stress(config, normalized_problem)
        for budget in config.budgets:
            group_rows = [
                row
                for row in rows
                if row.get("benchmark") == normalized_problem
                and int(row.get("requested_budget", -1)) == int(budget)
            ]
            fairness = evaluate_constrained_mo_fairness(
                expected_contract=_constraint_contract(
                    problem,
                    budget=int(budget),
                    tolerance=config.tolerance,
                ),
                observed_rows=group_rows,
            )
            payload = {
                "benchmark": normalized_problem,
                "budget": int(budget),
                "status": fairness["status"],
                "summary_counts": fairness["summary_counts"],
                "issues": fairness["issues"],
            }
            payloads.append(payload)
            all_issues.extend(fairness["issues"])
    status = "pass"
    if any(item["status"] == "fail" for item in all_issues):
        status = "fail"
    elif any(item["status"] == "warning" for item in all_issues):
        status = "warning"
    return {
        "status": status,
        "summary_counts": {
            "pass": sum(1 for item in all_issues if item["status"] == "pass"),
            "warning": sum(1 for item in all_issues if item["status"] == "warning"),
            "fail": sum(1 for item in all_issues if item["status"] == "fail"),
        },
        "issues": all_issues,
        "by_benchmark_budget": payloads,
    }


def _aggregate_rows(
    *,
    config: ConstrainedNSGA2StressConfig,
    rows: Sequence[dict[str, Any]],
    fairness_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seed_count = len(_seed_list(config.seeds))
    fairness_by_key = {
        (item["benchmark"], int(item["budget"])): item["status"]
        for item in fairness_payload.get("by_benchmark_budget", [])
    }
    for benchmark in config.problems:
        normalized_benchmark = _normalize_problem_name(benchmark)
        for budget in config.budgets:
            for strategy in config.strategies:
                group = [
                    row
                    for row in rows
                    if row["benchmark"] == normalized_benchmark
                    and int(row["requested_budget"]) == int(budget)
                    and row["strategy"] == strategy
                ]
                if not group:
                    continue
                summaries.append(
                    {
                        "benchmark": normalized_benchmark,
                        "strategy": strategy,
                        "budget": int(budget),
                        "mean_feasible_rate": mean(float(row["feasible_rate"]) for row in group),
                        "median_feasible_rate": median(float(row["feasible_rate"]) for row in group),
                        "mean_feasible_nondominated_count": mean(
                            float(row["feasible_nondominated_count"]) for row in group
                        ),
                        "mean_total_violation": mean(
                            float(row["mean_total_violation"] or 0.0) for row in group
                        ),
                        "mean_feasible_only_HV": _non_null_mean(
                            [row.get("feasible_only_HV") for row in group]
                        ),
                        "mean_feasible_only_reference_distance": _non_null_mean(
                            [row.get("feasible_only_reference_distance") for row in group]
                        ),
                        "mean_feasible_only_spacing": _non_null_mean(
                            [row.get("spacing_feasible_only") for row in group]
                        ),
                        "actual_evaluations": group[0]["actual_evaluations"],
                        "mean_runtime_seconds": mean(float(row["runtime_seconds"]) for row in group),
                        "all_runs_completed": len(group) == seed_count
                        and all(int(row["failure_count"]) == 0 for row in group),
                        "fairness_status": fairness_by_key.get(
                            (normalized_benchmark, int(budget)),
                            "unknown",
                        ),
                    }
                )
    return summaries


def _compare_values(metric: str, constrained: Any, comparator: Any) -> int:
    if constrained is None or comparator is None:
        return 0
    left = float(constrained)
    right = float(comparator)
    if abs(left - right) <= 1e-12:
        return 0
    if metric == "actual_evaluations":
        return 0
    if metric in LOWER_IS_BETTER:
        return 1 if left < right else -1
    return 1 if left > right else -1


def _paired_comparisons(
    *,
    config: ConstrainedNSGA2StressConfig,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for benchmark in config.problems:
        normalized_benchmark = _normalize_problem_name(benchmark)
        for budget in config.budgets:
            constrained_rows = {
                int(row["seed"]): row
                for row in rows
                if row["benchmark"] == normalized_benchmark
                and int(row["requested_budget"]) == int(budget)
                and row["strategy"] == "constrained_nsga2_constraint_domination"
            }
            comparator_rows = {
                int(row["seed"]): row
                for row in rows
                if row["benchmark"] == normalized_benchmark
                and int(row["requested_budget"]) == int(budget)
                and row["strategy"] == "random_pareto_archive"
            }
            paired_seeds = sorted(set(constrained_rows) & set(comparator_rows))
            for metric in PAIR_METRICS:
                deltas: list[float] = []
                constrained_wins = 0
                comparator_wins = 0
                ties = 0
                for seed in paired_seeds:
                    constrained_value = constrained_rows[seed].get(metric)
                    comparator_value = comparator_rows[seed].get(metric)
                    result = _compare_values(metric, constrained_value, comparator_value)
                    if result > 0:
                        constrained_wins += 1
                    elif result < 0:
                        comparator_wins += 1
                    else:
                        ties += 1
                    if constrained_value is not None and comparator_value is not None:
                        deltas.append(float(constrained_value) - float(comparator_value))
                mean_delta = mean(deltas) if deltas else None
                median_delta = median(deltas) if deltas else None
                comparisons.append(
                    {
                        "benchmark": normalized_benchmark,
                        "budget": int(budget),
                        "metric": metric,
                        "constrained_nsga2_win": constrained_wins,
                        "tie": ties,
                        "comparator_win": comparator_wins,
                        "mean_delta": mean_delta,
                        "median_delta": median_delta,
                        "interpretation": _comparison_interpretation(
                            metric,
                            constrained_wins,
                            comparator_wins,
                            ties,
                        ),
                    }
                )
    return comparisons


def _comparison_interpretation(
    metric: str,
    constrained_wins: int,
    comparator_wins: int,
    ties: int,
) -> str:
    if metric == "actual_evaluations":
        return "budget accounting comparison; ties are expected"
    if metric == "runtime_seconds":
        return "runtime trade-off; lower runtime is not a quality gate"
    if constrained_wins > comparator_wins:
        return "constrained NSGA-II favorable signal for this metric"
    if comparator_wins > constrained_wins:
        return "comparator favorable or feasible-only limitation to document"
    if ties > 0:
        return "tied or unavailable paired signal"
    return "no paired signal"


def _stress_decision(
    *,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: Sequence[dict[str, Any]],
) -> str:
    if failures:
        return "Fix required"
    if fairness_payload.get("status") == "fail":
        return "Needs fairness rerun"
    if any(int(row["actual_evaluations"]) != int(row["requested_budget"]) for row in rows):
        return "Fix required"
    required_fields = (
        "feasible_only_HV",
        "feasible_only_reference_distance",
        "spacing_feasible_only",
        "constraint_summary",
        "per_constraint_violation_summary",
    )
    if any(field not in row for row in rows for field in required_fields):
        return "Fix required"
    constrained_summaries = [
        item
        for item in summaries
        if item["strategy"] == "constrained_nsga2_constraint_domination"
    ]
    positive_feasibility = all(
        float(item["mean_feasible_rate"]) >= 0.95
        and float(item["mean_total_violation"]) <= 1e-12
        for item in constrained_summaries
    )
    if positive_feasibility:
        return "Stress passed, constrained NSGA-II restricted opt-in evidence strengthened"
    return "Stress passed with limitations"


def _render_results_markdown(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Constrained NSGA-II Stress Results",
            "",
            _markdown_table(
                rows,
                [
                    "benchmark",
                    "strategy",
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
                    "failure_count",
                ],
            ),
            "",
        ]
    )


def _render_fairness_markdown(fairness_payload: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for item in fairness_payload.get("by_benchmark_budget", []):
        counts = item.get("summary_counts", {})
        rows.append(
            {
                "benchmark": item.get("benchmark"),
                "budget": item.get("budget"),
                "status": item.get("status"),
                "pass": counts.get("pass", 0),
                "warning": counts.get("warning", 0),
                "fail": counts.get("fail", 0),
            }
        )
    return "\n".join(
        [
            "# Constrained NSGA-II Stress Fairness Report",
            "",
            f"- Status: **{fairness_payload['status']}**",
            f"- Summary counts: `{fairness_payload.get('summary_counts', {})}`",
            "",
            _markdown_table(rows, ["benchmark", "budget", "status", "pass", "warning", "fail"]),
            "",
            "## Issues",
            "",
            _markdown_table(
                list(fairness_payload.get("issues", [])),
                ["issue_type", "status", "severity", "message", "recommended_action"],
            ),
            "",
        ]
    )


def _render_stress_report(
    *,
    config: ConstrainedNSGA2StressConfig,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
    decision: str,
    resume_summary: dict[str, Any] | None = None,
    resume_source_artifact: str | None = None,
    resume_mode: str | None = None,
    parallel_summary: dict[str, Any] | None = None,
) -> str:
    issue_rows = _build_failures_and_warnings_rows(failures, warnings)
    if not issue_rows:
        issue_rows = [{"type": "none", "message": "no runner failures or warnings", "action": "none"}]
    fairness_rows = _build_fairness_rows(fairness_payload)
    result_rows = [
        {
            "benchmark": item["benchmark"],
            "budget": item["budget"],
            "strategy": item["strategy"],
            "mean feasible_rate": item["mean_feasible_rate"],
            "mean feasible_nondominated_count": item["mean_feasible_nondominated_count"],
            "mean_total_violation": item["mean_total_violation"],
            "mean feasible_only_HV": item["mean_feasible_only_HV"],
            "mean feasible_only_reference_distance": item[
                "mean_feasible_only_reference_distance"
            ],
            "actual_evaluations": item["actual_evaluations"],
            "mean runtime": item["mean_runtime_seconds"],
        }
        for item in summaries
    ]
    comparison_rows = [
        {
            "benchmark": item["benchmark"],
            "budget": item["budget"],
            "metric": item["metric"],
            "constrained_nsga2 win": item["constrained_nsga2_win"],
            "tie": item["tie"],
            "comparator win": item["comparator_win"],
            "mean delta": item["mean_delta"],
            "interpretation": item["interpretation"],
        }
        for item in paired_comparisons
    ]
    decision_rows = [
        {
            "benchmark": item["benchmark"],
            "budget": item["budget"],
            "stress result": "completed" if item["all_runs_completed"] else "incomplete",
            "fairness": item["fairness_status"],
            "constrained_nsga2 vs baseline": _summary_interpretation(item, summaries),
            "runtime trade-off": "documented",
            "decision contribution": "supports restricted opt-in evidence"
            if item["strategy"] == "constrained_nsga2_constraint_domination"
            else "baseline comparator",
        }
        for item in summaries
    ]
    resume_section = ""
    if resume_summary is not None:
        resume_rows = [
            {"item": "resume_source_artifact", "value": resume_source_artifact},
            {"item": "resume_mode", "value": resume_mode},
            {"item": "total_planned", "value": resume_summary.get("total_planned")},
            {"item": "completed_from_source", "value": resume_summary.get("completed_from_source")},
            {"item": "newly_executed", "value": resume_summary.get("newly_executed")},
            {"item": "skipped_existing", "value": resume_summary.get("skipped_existing")},
            {"item": "failed_existing", "value": resume_summary.get("failed_existing")},
            {"item": "warnings", "value": len(resume_summary.get("warnings", []))},
        ]
        resume_section = "\n".join(
            [
                "## Runner-level Resume Summary",
                "",
                _markdown_table(resume_rows, ["item", "value"]),
                "",
                "This is skip-completed runner-level resume only; it is not algorithm checkpoint/resume.",
                "",
            ]
        )
    parallel_section = ""
    if parallel_summary is not None:
        parallel_rows = [
            {"item": "parallel_enabled", "value": parallel_summary.get("parallel_enabled")},
            {"item": "backend", "value": parallel_summary.get("backend")},
            {"item": "workers", "value": parallel_summary.get("workers")},
            {"item": "total_submitted", "value": parallel_summary.get("total_submitted")},
            {
                "item": "skipped_before_parallel",
                "value": parallel_summary.get("skipped_before_parallel"),
            },
            {"item": "success_count", "value": parallel_summary.get("success_count")},
            {"item": "failure_count", "value": parallel_summary.get("failure_count")},
            {
                "item": "deterministic_merge_order",
                "value": parallel_summary.get("deterministic_merge_order"),
            },
        ]
        parallel_section = "\n".join(
            [
                "## Runner-level Parallel Summary",
                "",
                _markdown_table(parallel_rows, ["item", "value"]),
                "",
                "This is runner-level row execution only. It is not algorithm-internal evaluation parallelism.",
                "",
            ]
        )
    return "\n".join(
        [
            "# Constrained NSGA-II Stress Report",
            "",
            "## 1. Executive Summary",
            "",
            "- 이번 작업의 목표는 constrained NSGA-II opt-in path의 two-toy seed+budget stress를 실행하는 것이다.",
            f"- 실행한 stress scope는 benchmarks={list(config.problems)}, budgets={list(config.budgets)}, seeds={len(_seed_list(config.seeds))}, strategies={list(config.strategies)}이다.",
            "- benchmark별 결과는 Results Summary와 Paired Comparison에 기록했다.",
            f"- fairness 결과는 `{fairness_payload['status']}`이며 summary_counts={fairness_payload.get('summary_counts', {})}이다.",
            f"- stress decision: **{decision}**",
            "- 기본값 변경 여부: 없음.",
            "- default NSGA-II 영향 여부: 없음.",
            "- Level 판정 변화 여부: default maturity 상향 없음, Level 4 근거 강화.",
            "",
            resume_section,
            parallel_section,
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "",
            "- constrained_zdt_box_toy",
            "- constrained_dtlz_box_toy",
            "- constrained_nsga2_constraint_domination",
            "- random_pareto_archive",
            "- feasible-only constrained MO metrics",
            "- constrained fairness",
            "- stress artifact generation",
            "",
            "Non-Scope:",
            "",
            "- default NSGA-II replacement",
            "- penalty/repair",
            "- constrained MO full benchmark suite",
            "- productization",
            "",
            "## 3. Stress Configuration",
            "",
            _markdown_table(
                [
                    {"field": "benchmarks", "value": list(config.problems)},
                    {"field": "dimensions", "value": dict(config.dimensions)},
                    {"field": "budgets", "value": list(config.budgets)},
                    {"field": "seeds", "value": len(_seed_list(config.seeds))},
                    {"field": "strategies", "value": list(config.strategies)},
                    {"field": "constraint policy", "value": "constraint_domination"},
                ],
                ["field", "value"],
            ),
            "",
            "## 4. Results Summary",
            "",
            _markdown_table(
                result_rows,
                [
                    "benchmark",
                    "budget",
                    "strategy",
                    "mean feasible_rate",
                    "mean feasible_nondominated_count",
                    "mean_total_violation",
                    "mean feasible_only_HV",
                    "mean feasible_only_reference_distance",
                    "actual_evaluations",
                    "mean runtime",
                ],
            ),
            "",
            "## 5. Paired Comparison",
            "",
            _markdown_table(
                comparison_rows,
                [
                    "benchmark",
                    "budget",
                    "metric",
                    "constrained_nsga2 win",
                    "tie",
                    "comparator win",
                    "mean delta",
                    "interpretation",
                ],
            ),
            "",
            "## 6. Fairness Summary",
            "",
            _markdown_table(fairness_rows, ["item", "status", "message"]),
            "",
            "## 7. Limitations",
            "",
            "- toy benchmark limitation: two local toy benchmarks only.",
            "- no external constrained comparator.",
            "- feasible-only metric can be mixed and must not be hidden.",
            "- no penalty/repair.",
            "- no default NSGA-II constrained support.",
            "",
            "## 8. Failures and Warnings",
            "",
            _markdown_table(issue_rows, ["type", "message", "action"]),
            "",
            "## 9. Regression Check",
            "",
            "_Regression command results are recorded in the final task report after the stress run._",
            "",
            "## 10. Decision",
            "",
            f"- {decision}",
            "",
            "## 10a. Decision Gate by Benchmark/Budget",
            "",
            _markdown_table(
                decision_rows,
                [
                    "benchmark",
                    "budget",
                    "stress result",
                    "fairness",
                    "constrained_nsga2 vs baseline",
                    "runtime trade-off",
                    "decision contribution",
                ],
            ),
            "",
            "## 11. What This Proves",
            "",
            "- constrained NSGA-II opt-in path can run on two constrained MO toys.",
            "- feasible-only metrics and violation summary are generated under stress.",
            "- fairness and exact budget accounting still work when fairness status is pass.",
            "- default NSGA-II unchanged.",
            "",
            "## 12. What This Does Not Prove",
            "",
            "- default NSGA-II가 constrained optimizer가 됐다는 뜻은 아님.",
            "- product-ready constrained MOEA라는 뜻은 아님.",
            "- penalty/repair가 구현됐다는 뜻은 아님.",
            "- broad constrained MO benchmark coverage가 확보됐다는 뜻은 아님.",
            "- external constrained parity가 확보됐다는 뜻은 아님.",
            "",
            "## 13. Maturity Impact",
            "",
            "- Level 4 근거 강화.",
            "",
            "Constrained NSGA-II stress는 실험 툴킷 근거를 강화한다. Default NSGA-II가 바뀌지 않았으므로 default algorithm maturity 상향은 금지한다. Constrained MO toy 2개 + stress 수준이므로 범용 constrained MOEA 주장은 금지한다.",
            "",
            "## 14. Recommended Next Work",
            "",
            "- constrained NSGA-II broader review update",
            "",
            "이번 constrained NSGA-II stress 결과, 저장소는 two constrained MO toys seed+budget stress artifact generation까지 검증했지만, default NSGA-II는 unchanged 상태이며, 다음 단계는 constrained NSGA-II broader review update이다.",
            "",
        ]
    )


def _summary_interpretation(
    item: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
) -> str:
    if item["strategy"] != "constrained_nsga2_constraint_domination":
        return "baseline"
    baseline = next(
        (
            row
            for row in summaries
            if row["benchmark"] == item["benchmark"]
            and row["budget"] == item["budget"]
            and row["strategy"] == "random_pareto_archive"
        ),
        None,
    )
    if baseline is None:
        return "no baseline"
    feasibility_good = float(item["mean_feasible_rate"]) >= float(baseline["mean_feasible_rate"])
    violation_good = float(item["mean_total_violation"]) <= float(baseline["mean_total_violation"])
    if feasibility_good and violation_good:
        return "feasibility/violation positive signal"
    return "mixed signal"


def run_constrained_nsga2_stress(config: ConstrainedNSGA2StressConfig) -> dict[str, Any]:
    normalized_problems = tuple(_normalize_problem_name(problem) for problem in config.problems)
    for problem_name in normalized_problems:
        if problem_name not in SUPPORTED_PROBLEMS:
            raise ValueError(f"Unsupported constrained NSGA-II stress problem: {problem_name}")
    for strategy in config.strategies:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported constrained NSGA-II stress strategy: {strategy}")

    seed_list = _seed_list(config.seeds)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    resume_enabled = config.resume_from is not None
    completed_index = (
        load_completed_run_index(config.resume_from) if resume_enabled else CompletedRunIndex()
    )
    planned_keys = []
    planned_order: list[str] = []
    rows_by_key: dict[str, dict[str, Any]] = {}
    missing_tasks: list[dict[str, Any]] = []
    copied_from_source = 0
    newly_executed = 0
    parallel_config = ParallelExecutionConfig(
        backend=config.row_execution_backend,  # type: ignore[arg-type]
        workers=config.row_execution_workers,
        fail_fast=config.parallel_fail_fast,
        timeout_seconds=config.parallel_timeout_seconds,
        allow_process_backend=config.allow_process_backend,
    )

    for problem_name in normalized_problems:
        for budget in config.budgets:
            problem = _build_problem_for_stress(config, problem_name)
            try:
                population_size, generations = _resolve_budget_schedule(
                    int(budget),
                    preferred_population_size=config.population_size,
                )
            except Exception as exc:
                failures.append(
                    {
                        "type": "budget_schedule_failure",
                        "message": f"{problem_name} budget={budget}: {exc}",
                    }
                )
                continue
            for strategy in config.strategies:
                for seed in seed_list:
                    planned_row = _planned_resume_row(
                        problem=problem,
                        strategy=strategy,
                        seed=seed,
                        budget=int(budget),
                    )
                    resume_key = build_resume_key(planned_row)
                    key_id = resume_key.stable_id()
                    planned_keys.append(resume_key)
                    planned_order.append(key_id)
                    if resume_enabled and key_id in completed_index.completed_keys:
                        source_row = completed_index.completed_rows_by_key[key_id]
                        row = merge_resume_results(
                            source_rows=[source_row],
                            new_rows=[],
                            source_artifact=config.resume_from,
                        )[0]
                        row["row_execution_origin"] = "resume_source_artifact"
                        rows_by_key[key_id] = row
                        copied_from_source += 1
                        continue
                    missing_tasks.append(
                        {
                            "problem_name": problem_name,
                            "dimension": int(problem.dimension),
                            "tolerance": float(
                                getattr(problem, "equality_tolerance", config.tolerance)
                            ),
                            "constraint_overrides": config.constraint_overrides,
                            "strategy": strategy,
                            "seed": int(seed),
                            "budget": int(budget),
                            "population_size": int(population_size),
                            "generations": int(generations),
                            "resume_key_id": key_id,
                        }
                    )

    try:
        parallel_execution = execute_runner_rows(
            missing_tasks,
            _execute_stress_task,
            config=parallel_config,
            row_key_fn=_stress_task_key,
        )
    except Exception as exc:
        parallel_execution = None
        failures.append(
            {
                "type": "parallel_configuration_failure",
                "message": (
                    f"backend={config.row_execution_backend} workers={config.row_execution_workers}: {exc}"
                ),
            }
        )
    if parallel_execution is not None:
        for result in parallel_execution.successful_results:
            row = dict(result.row or {})
            if resume_enabled:
                row = merge_resume_results(
                    source_rows=[],
                    new_rows=[row],
                    source_artifact=config.resume_from,
                )[0]
            row["row_execution_origin"] = (
                "serial"
                if parallel_execution.backend == "serial"
                else f"parallel_{parallel_execution.backend}"
            )
            rows_by_key[result.row_key] = row
            newly_executed += 1
        for result in parallel_execution.failed_results:
            if result.failure is None:
                continue
            failures.append(
                {
                    "type": "parallel_row_failure",
                    "message": result.failure["message"],
                    "row_key": result.failure["row_key"],
                    "backend": result.failure["backend"],
                    "exception_type": result.failure["exception_type"],
                    "order_index": result.failure["order_index"],
                }
            )

    rows = [rows_by_key[key_id] for key_id in planned_order if key_id in rows_by_key]
    for row_order_index, row in enumerate(rows):
        row["row_order_index"] = row_order_index

    if parallel_execution is None:
        parallel_summary = {
            "parallel_enabled": config.row_execution_backend != "serial",
            "backend": config.row_execution_backend,
            "workers": config.row_execution_workers,
            "total_submitted": len(missing_tasks),
            "success_count": 0,
            "failure_count": len(missing_tasks),
            "skipped_before_parallel": copied_from_source,
            "deterministic_merge_order": True,
            "warnings": [],
            "failures": failures[:],
        }
    else:
        parallel_summary = parallel_summary_to_dict(parallel_execution)
        parallel_summary["skipped_before_parallel"] = copied_from_source
    if config.row_execution_backend == "serial":
        parallel_summary["parallel_enabled"] = False
    parallel_summary["total_rows"] = len(rows)

    resume_plan = plan_runner_resume(
        planned_keys,
        completed_index,
        resume_mode=config.resume_mode,
    )
    failed_existing = sum(
        1 for key in planned_keys if key.stable_id() in completed_index.failed_keys
    )
    resume_summary = {
        "total_planned": len(planned_keys),
        "completed_from_source": copied_from_source,
        "newly_executed": newly_executed,
        "skipped_existing": copied_from_source,
        "failed_existing": failed_existing,
        "missing_executed": newly_executed,
        "warnings": list(completed_index.warnings),
    }
    warnings.extend(
        {"type": "resume_warning", "message": warning}
        for warning in completed_index.warnings
    )

    fairness_payload = _fairness_by_benchmark_budget(config=config, rows=rows)
    summaries = _aggregate_rows(config=config, rows=rows, fairness_payload=fairness_payload)
    paired_comparisons = _paired_comparisons(config=config, rows=rows)
    decision = _stress_decision(
        rows=rows,
        summaries=summaries,
        fairness_payload=fairness_payload,
        failures=failures,
    )

    artifact = {
        "command_metadata": {
            "argv": sys.argv,
            "artifact_suffix": config.artifact_suffix,
            "output_dir": str(config.output_dir),
            "row_execution_backend": config.row_execution_backend,
            "row_execution_workers": config.row_execution_workers,
        },
        "configuration": {
            "problems": list(normalized_problems),
            "dimensions": {
                problem: _dimension_for_problem(config, problem)
                for problem in normalized_problems
            },
            "budgets": [int(budget) for budget in config.budgets],
            "seeds": config.seeds,
            "seed_list": seed_list,
            "tolerance": config.tolerance,
            "population_size": config.population_size,
            "constraint_policy": "constraint_domination",
            "violation_aggregation": "total_violation",
            "strategies": list(config.strategies),
        },
        "rows": rows,
        "benchmark_budget_summaries": summaries,
        "paired_comparisons": paired_comparisons,
        "fairness_summary": fairness_payload,
        "failures": failures,
        "warnings": warnings,
        "decision": decision,
        "default_changed": False,
        "nsga2_default_changed": False,
        "nsga2_constraint_domination_done": True,
        "penalty_repair_done": False,
        "resume_enabled": resume_enabled,
        "resume_source_artifact": str(config.resume_from) if config.resume_from else None,
        "resume_mode": config.resume_mode,
        "resume_summary": resume_summary if resume_enabled else None,
        "parallel_enabled": config.row_execution_backend != "serial",
        "parallel_backend": config.row_execution_backend,
        "parallel_workers": config.row_execution_workers,
        "parallel_summary": parallel_summary,
        "algorithm_internal_parallel_evaluation_done": False,
        "production_scalability_claim": False,
    }

    project_root = _project_root()
    output_dir_input = Path(config.output_dir)
    output_dir = output_dir_input if output_dir_input.is_absolute() else project_root / output_dir_input
    prefer_absolute_artifact_paths = output_dir_input.is_absolute()
    suffix = config.artifact_suffix

    json_path = output_dir / f"constrained_nsga2_stress_results_{suffix}.json"
    csv_path = output_dir / f"constrained_nsga2_stress_results_{suffix}.csv"
    md_results_path = output_dir / f"constrained_nsga2_stress_results_{suffix}.md"
    report_path = output_dir / f"constrained_nsga2_stress_report_{suffix}.md"
    fairness_path = output_dir / f"constrained_nsga2_stress_fairness_report_{suffix}.md"
    resume_report_suffix = config.resume_report_suffix or suffix
    resume_report_path = output_dir / f"constrained_nsga2_stress_resume_report_{resume_report_suffix}.md"

    artifact["artifacts"] = {
        "json": _artifact_ref(json_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "csv": _artifact_ref(csv_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "results_markdown": _artifact_ref(
            md_results_path,
            project_root,
            prefer_absolute=prefer_absolute_artifact_paths,
        ),
        "report_markdown": _artifact_ref(
            report_path,
            project_root,
            prefer_absolute=prefer_absolute_artifact_paths,
        ),
        "fairness_markdown": _artifact_ref(
            fairness_path,
            project_root,
            prefer_absolute=prefer_absolute_artifact_paths,
        ),
        "resume_markdown": (
            _artifact_ref(
                resume_report_path,
                project_root,
                prefer_absolute=prefer_absolute_artifact_paths,
            )
            if resume_enabled
            else None
        ),
    }

    _write_json(json_path, artifact)
    _write_stress_csv(csv_path, rows)
    _ensure_fresh_path(md_results_path).write_text(
        _render_results_markdown(rows),
        encoding="utf-8",
    )
    _ensure_fresh_path(fairness_path).write_text(
        _render_fairness_markdown(fairness_payload),
        encoding="utf-8",
    )
    _ensure_fresh_path(report_path).write_text(
        _render_stress_report(
            config=config,
            rows=rows,
            summaries=summaries,
            paired_comparisons=paired_comparisons,
            fairness_payload=fairness_payload,
            failures=failures,
            warnings=warnings,
            decision=decision,
            resume_summary=resume_summary if resume_enabled else None,
            resume_source_artifact=str(config.resume_from) if config.resume_from else None,
            resume_mode=config.resume_mode,
            parallel_summary=parallel_summary,
        ),
        encoding="utf-8",
    )
    if resume_enabled:
        write_resume_report(resume_report_path, resume_plan, resume_summary)
    return artifact


def _write_stress_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fieldnames = [
        "benchmark",
        "strategy",
        "seed",
        "requested_budget",
        "actual_evaluations",
        "feasible_rate",
        "feasible_count",
        "infeasible_count",
        "feasible_nondominated_count",
        "infeasible_nondominated_count",
        "best_total_violation",
        "mean_total_violation",
        "median_total_violation",
        "min_total_violation",
        "max_total_violation",
        "feasible_only_HV",
        "feasible_only_reference_distance",
        "spacing_feasible_only",
        "per_constraint_satisfaction_rate",
        "per_constraint_mean_violation",
        "per_constraint_max_violation",
        "runtime_seconds",
        "failure_count",
        "row_execution_origin",
        "row_order_index",
    ]
    _ensure_fresh_path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _stringify(row.get(field, "")) for field in fieldnames})
    return path


__all__ = [
    "ConstrainedNSGA2StressConfig",
    "DEFAULT_DIMENSIONS",
    "run_constrained_nsga2_stress",
]
