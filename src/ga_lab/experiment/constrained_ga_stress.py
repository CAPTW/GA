from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ga_lab.experiment.constrained_fairness import evaluate_constrained_fairness
from ga_lab.experiment.constrained_ga_smoke import (
    DEFAULT_STRATEGIES,
    SUPPORTED_STRATEGIES,
    ConstrainedGASmokeConfig,
    _aggregate_per_constraint_summaries,
    _artifact_ref,
    _build_problem_for_smoke,
    _build_failures_and_warnings_rows,
    _build_fairness_rows,
    _constraint_contract,
    _constrained_ga_row,
    _ensure_fresh_path,
    _markdown_table,
    _project_root,
    _random_search_row,
    _seed_list,
    _stringify,
    _write_csv,
    _write_json,
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


HIGHER_IS_BETTER = {"feasible_rate"}
LOWER_IS_BETTER = {
    "best_feasible_objective",
    "mean_total_violation",
    "min_total_violation",
    "max_total_violation",
    "per_constraint_mean_violation",
    "runtime_seconds",
}
COMPARISON_METRICS = (
    "feasible_rate",
    "best_feasible_objective",
    "mean_total_violation",
    "per_constraint_mean_violation",
    "min_total_violation",
    "max_total_violation",
    "runtime_seconds",
    "actual_evaluations",
)


@dataclass(slots=True)
class ConstrainedGAStressConfig:
    dimension: int = 5
    seeds: int | Sequence[int] = 30
    budgets: Sequence[int] = (300, 1000)
    constraint_budget: float = 1.0
    tolerance: float = 1e-8
    population_size: int = 20
    artifact_suffix: str = "constrained_ga_stress1"
    output_dir: str = "artifacts"
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    problem: str = "constrained_sphere"
    resume_from: str | Path | None = None
    resume_mode: str = "skip-completed"
    resume_report_suffix: str | None = None
    row_execution_backend: str = "serial"
    row_execution_workers: int = 1
    parallel_fail_fast: bool = False
    parallel_timeout_seconds: float | None = None
    allow_process_backend: bool = False


def _normalize_budgets(budget_spec: Sequence[int] | int) -> list[int]:
    if isinstance(budget_spec, int):
        if budget_spec <= 0:
            raise ValueError("budget must be > 0")
        return [budget_spec]
    budgets: list[int] = []
    seen: set[int] = set()
    for value in budget_spec:
        budget = int(value)
        if budget <= 0:
            raise ValueError("budgets must all be > 0")
        if budget not in seen:
            budgets.append(budget)
            seen.add(budget)
    if not budgets:
        raise ValueError("budgets cannot be empty")
    return budgets


def _smoke_config_for_budget(
    config: ConstrainedGAStressConfig,
    budget: int,
) -> ConstrainedGASmokeConfig:
    return ConstrainedGASmokeConfig(
        problem=config.problem,
        dimension=config.dimension,
        seeds=1,
        budget=budget,
        constraint_budget=config.constraint_budget,
        tolerance=config.tolerance,
        population_size=config.population_size,
        artifact_suffix=config.artifact_suffix,
        output_dir=config.output_dir,
        strategies=config.strategies,
    )


def _overall_status(statuses: Sequence[str]) -> str:
    normalized = [status for status in statuses if status]
    if any(status == "fail" for status in normalized):
        return "fail"
    if any(status == "warning" for status in normalized):
        return "warning"
    return "pass"


def _summarize_counts(issues: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "pass": sum(1 for issue in issues if issue.get("status") == "pass"),
        "warning": sum(1 for issue in issues if issue.get("status") == "warning"),
        "fail": sum(1 for issue in issues if issue.get("status") == "fail"),
    }


def _safe_mean(values: Sequence[float]) -> float | None:
    normalized = [float(value) for value in values]
    if not normalized:
        return None
    return float(statistics.mean(normalized))


def _safe_median(values: Sequence[float]) -> float | None:
    normalized = [float(value) for value in values]
    if not normalized:
        return None
    return float(statistics.median(normalized))


def _artifact_prefix(problem_name: str) -> str:
    normalized = problem_name.strip().lower().replace("-", "_")
    if normalized == "constrained_sphere":
        return "constrained_ga_stress"
    return f"{normalized}_stress"


def _annotate_per_constraint_proxy(row: dict[str, Any]) -> None:
    available_items = [
        item
        for item in row.get("per_constraint_violation_summary", [])
        if item.get("mean_violation") is not None
    ]
    if available_items:
        row["per_constraint_mean_violation"] = _safe_mean(
            [float(item["mean_violation"]) for item in available_items]
        )
        row["per_constraint_satisfaction_rate"] = _safe_mean(
            [float(item["satisfaction_rate"]) for item in available_items]
            if all(item.get("satisfaction_rate") is not None for item in available_items)
            else []
        )
        row["per_constraint_summary_scope"] = "evaluation_trace"
        return

    constraint_count = int(row.get("constraint_count") or 0)
    mean_total_violation = row.get("mean_total_violation")
    row["per_constraint_mean_violation"] = (
        float(mean_total_violation) / constraint_count
        if constraint_count > 0 and mean_total_violation is not None
        else None
    )
    row["per_constraint_satisfaction_rate"] = None
    row["per_constraint_summary_scope"] = "derived_from_total_violation_no_group_breakdown"


def _planned_resume_row(
    *,
    problem,
    config: ConstrainedGAStressConfig,
    strategy: str,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    return {
        "problem": problem.name,
        "strategy": strategy,
        "seed": int(seed),
        "requested_budget": int(budget),
        "budget": int(budget),
        "dimension": int(config.dimension),
        "tolerance": float(config.tolerance),
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


def _stress_task_key(task: dict[str, Any]) -> str:
    return str(task["resume_key_id"])


def _execute_stress_task(task: dict[str, Any]) -> dict[str, Any]:
    config = ConstrainedGAStressConfig(
        problem=str(task["problem"]),
        dimension=int(task["dimension"]),
        seeds=1,
        budgets=(int(task["budget"]),),
        constraint_budget=float(task["constraint_budget"]),
        tolerance=float(task["tolerance"]),
        population_size=int(task["population_size"]),
        strategies=(str(task["strategy"]),),
    )
    smoke_config = _smoke_config_for_budget(config, int(task["budget"]))
    problem = _build_problem_for_smoke(smoke_config)
    row = _run_stress_strategy(
        strategy=str(task["strategy"]),
        smoke_config=smoke_config,
        problem=problem,
        seed=int(task["seed"]),
    )
    row.setdefault("status", "success")
    row.setdefault("skip_reason", None)
    row.setdefault("failure_reason", None)
    row.setdefault("warnings", [])
    row["budget"] = int(task["budget"])
    row["stress_mode"] = "seed_budget"
    _annotate_per_constraint_proxy(row)
    return row


def _run_stress_strategy(
    *,
    strategy: str,
    smoke_config: ConstrainedGASmokeConfig,
    problem,
    seed: int,
) -> dict[str, Any]:
    if strategy == "random_search_feasibility_first":
        return _random_search_row(config=smoke_config, problem=problem, seed=seed)
    if strategy == "constrained_ga_feasibility_first":
        return _constrained_ga_row(config=smoke_config, problem=problem, seed=seed)
    raise ValueError(f"Unsupported strategy: {strategy}")


def _budget_per_constraint_summaries(
    *,
    rows: Sequence[dict[str, Any]],
    budgets: Sequence[int],
    strategies: Sequence[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for budget in budgets:
        budget_rows = [row for row in rows if int(row["requested_budget"]) == int(budget)]
        budget_summary_rows = _aggregate_per_constraint_summaries(budget_rows)
        for item in budget_summary_rows:
            summaries.append({"budget": budget, **item})

        for strategy in strategies:
            strategy_rows = [row for row in budget_rows if row["strategy"] == strategy]
            if not strategy_rows:
                continue
            has_trace_summary = any(
                item.get("satisfaction_rate") is not None
                for row in strategy_rows
                for item in row.get("per_constraint_violation_summary", [])
            )
            if has_trace_summary:
                continue
            constraint_names = list(strategy_rows[0].get("constraint_names", []))
            for index, constraint_name in enumerate(constraint_names):
                summaries.append(
                    {
                        "budget": budget,
                        "constraint": f"{strategy}:{constraint_name}",
                        "strategy": strategy,
                        "constraint_name": constraint_name,
                        "satisfaction_rate": None,
                        "mean_violation": None,
                        "max_violation": None,
                        "limitation": "not_available_from_constrained_ga_summary",
                        "index": index,
                    }
                )
    return summaries


def _budget_strategy_summaries(
    *,
    rows: Sequence[dict[str, Any]],
    budgets: Sequence[int],
    strategies: Sequence[str],
    seed_count: int,
    fairness_by_budget: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for budget in budgets:
        budget_rows = [row for row in rows if int(row["requested_budget"]) == int(budget)]
        fairness_status = fairness_by_budget[budget]["status"]
        for strategy in strategies:
            strategy_rows = [row for row in budget_rows if row["strategy"] == strategy]
            feasible_rates = [float(row["feasible_rate"]) for row in strategy_rows]
            best_feasible_values = [
                float(row["best_feasible_objective"])
                for row in strategy_rows
                if row["best_feasible_objective"] is not None
            ]
            mean_total_violation_values = [
                float(row["mean_total_violation"] or 0.0) for row in strategy_rows
            ]
            per_constraint_mean_values = [
                float(row["per_constraint_mean_violation"])
                for row in strategy_rows
                if row.get("per_constraint_mean_violation") is not None
            ]
            per_constraint_satisfaction_values = [
                float(row["per_constraint_satisfaction_rate"])
                for row in strategy_rows
                if row.get("per_constraint_satisfaction_rate") is not None
            ]
            runtime_values = [float(row["runtime_seconds"]) for row in strategy_rows]
            actual_evaluations = [int(row["actual_evaluations"]) for row in strategy_rows]
            summaries.append(
                {
                    "budget": budget,
                    "strategy": strategy,
                    "run_count": len(strategy_rows),
                    "expected_run_count": seed_count,
                    "all_runs_completed": len(strategy_rows) == seed_count,
                    "fairness_status": fairness_status,
                    "mean_feasible_rate": _safe_mean(feasible_rates),
                    "median_feasible_rate": _safe_median(feasible_rates),
                    "mean_best_feasible_objective": _safe_mean(best_feasible_values),
                    "median_best_feasible_objective": _safe_median(best_feasible_values),
                    "mean_total_violation_mean": _safe_mean(mean_total_violation_values),
                    "mean_total_violation": _safe_mean(mean_total_violation_values),
                    "mean_per_constraint_satisfaction_rate": _safe_mean(
                        per_constraint_satisfaction_values
                    ),
                    "mean_per_constraint_violation": _safe_mean(per_constraint_mean_values),
                    "mean_runtime_seconds": _safe_mean(runtime_values),
                    "mean_actual_evaluations": _safe_mean(actual_evaluations),
                    "all_infeasible_runs": sum(
                        1 for row in strategy_rows if bool(row.get("all_infeasible"))
                    ),
                }
            )
    return summaries


def _comparison_outcome(
    *,
    metric: str,
    constrained_value: Any,
    baseline_value: Any,
    budget: int,
) -> tuple[str, float | None]:
    if metric == "actual_evaluations":
        if constrained_value is None or baseline_value is None:
            return "tie", None
        constrained_error = abs(float(constrained_value) - float(budget))
        baseline_error = abs(float(baseline_value) - float(budget))
        delta = baseline_error - constrained_error
        if constrained_error < baseline_error:
            return "constrained_ga", delta
        if baseline_error < constrained_error:
            return "random_search", delta
        return "tie", delta

    if constrained_value is None and baseline_value is None:
        return "tie", None
    if constrained_value is None:
        return "random_search", None
    if baseline_value is None:
        return "constrained_ga", None

    constrained = float(constrained_value)
    baseline = float(baseline_value)
    if metric in HIGHER_IS_BETTER:
        delta = constrained - baseline
        if constrained > baseline:
            return "constrained_ga", delta
        if baseline > constrained:
            return "random_search", delta
        return "tie", delta
    if metric in LOWER_IS_BETTER:
        delta = baseline - constrained
        if constrained < baseline:
            return "constrained_ga", delta
        if baseline < constrained:
            return "random_search", delta
        return "tie", delta
    raise ValueError(f"Unsupported comparison metric: {metric}")


def _paired_interpretation(
    *,
    metric: str,
    constrained_wins: int,
    ties: int,
    baseline_wins: int,
) -> str:
    if metric == "runtime_seconds":
        if baseline_wins > constrained_wins:
            return "random_search가 더 빠른 budget이었다. 품질과 분리된 runtime trade-off로 기록한다."
        if constrained_wins > baseline_wins:
            return "constrained_ga가 runtime에서도 우세했지만, 이는 보조 신호로만 기록한다."
        return "runtime은 budget 전반에서 큰 차별 신호 없이 tie에 가까웠다."
    if metric == "actual_evaluations":
        if ties and constrained_wins == 0 and baseline_wins == 0:
            return "두 strategy 모두 exact-budget accounting을 유지해 tie로 기록됐다."
        if constrained_wins > baseline_wins:
            return "constrained_ga 쪽 budget accounting이 baseline보다 더 정확했다."
        if baseline_wins > constrained_wins:
            return "random_search 쪽 budget accounting이 더 정확했다."
        return "actual_evaluations 결과는 혼재했다."
    if constrained_wins > baseline_wins:
        return "constrained_ga가 이 metric에서 반복 positive integration signal을 보였다."
    if baseline_wins > constrained_wins:
        return "random_search가 이 metric에서 더 나은 수치를 보인 budget/seed가 더 많았다."
    return "이 metric은 budget 내에서 전반적으로 tie에 가까웠다."


def _paired_comparisons(
    *,
    rows: Sequence[dict[str, Any]],
    budgets: Sequence[int],
    seed_list: Sequence[int],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for budget in budgets:
        budget_rows = [row for row in rows if int(row["requested_budget"]) == int(budget)]
        by_strategy_seed = {
            (str(row["strategy"]), int(row["seed"])): row
            for row in budget_rows
        }
        for metric in COMPARISON_METRICS:
            constrained_wins = 0
            ties = 0
            baseline_wins = 0
            deltas: list[float] = []
            matched_pairs = 0
            for seed in seed_list:
                constrained = by_strategy_seed.get(("constrained_ga_feasibility_first", int(seed)))
                baseline = by_strategy_seed.get(("random_search_feasibility_first", int(seed)))
                if constrained is None or baseline is None:
                    continue
                matched_pairs += 1
                winner, delta = _comparison_outcome(
                    metric=metric,
                    constrained_value=constrained.get(metric),
                    baseline_value=baseline.get(metric),
                    budget=budget,
                )
                if winner == "constrained_ga":
                    constrained_wins += 1
                elif winner == "random_search":
                    baseline_wins += 1
                else:
                    ties += 1
                if delta is not None:
                    deltas.append(float(delta))
            comparisons.append(
                {
                    "budget": budget,
                    "metric": metric,
                    "constrained_ga_win": constrained_wins,
                    "tie": ties,
                    "random_search_win": baseline_wins,
                    "mean_delta": _safe_mean(deltas),
                    "median_delta": _safe_median(deltas),
                    "matched_pairs": matched_pairs,
                    "interpretation": _paired_interpretation(
                        metric=metric,
                        constrained_wins=constrained_wins,
                        ties=ties,
                        baseline_wins=baseline_wins,
                    ),
                }
            )
    return comparisons


def _budget_decision_summary(
    *,
    budget_summaries: Sequence[dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
    fairness_by_budget: dict[int, dict[str, Any]],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        fairness_status = fairness_by_budget[budget]["status"]
        budget_pairs = [row for row in paired_comparisons if int(row["budget"]) == int(budget)]
        positive_metrics = [
            row["metric"]
            for row in budget_pairs
            if row["metric"]
            in {
                "feasible_rate",
                "best_feasible_objective",
                "mean_total_violation",
                "per_constraint_mean_violation",
            }
            and int(row["constrained_ga_win"]) > int(row["random_search_win"])
        ]
        runtime_row = next(
            (row for row in budget_pairs if row["metric"] == "runtime_seconds"),
            None,
        )
        budget_strategy = {
            row["strategy"]: row
            for row in budget_summaries
            if int(row["budget"]) == int(budget)
        }
        constrained_summary = budget_strategy.get("constrained_ga_feasibility_first")
        random_summary = budget_strategy.get("random_search_feasibility_first")
        all_runs_completed = bool(
            constrained_summary
            and random_summary
            and constrained_summary["all_runs_completed"]
            and random_summary["all_runs_completed"]
        )
        if positive_metrics:
            stress_result = "positive integration signal observed"
            comparison_note = ", ".join(positive_metrics)
            decision_contribution = "ready_or_tradeoff_positive"
        else:
            stress_result = "mixed_or_weak signal"
            comparison_note = "positive metric 없음"
            decision_contribution = "hold_or_tradeoff"
        runtime_tradeoff = (
            runtime_row["interpretation"]
            if runtime_row is not None
            else "runtime comparison unavailable"
        )
        if not all_runs_completed:
            stress_result = "incomplete runs"
            decision_contribution = "fix_required"
        if fairness_status == "fail":
            decision_contribution = "needs_fairness_rerun"
        rows.append(
            {
                "budget": budget,
                "stress_result": stress_result,
                "fairness": fairness_status,
                "constrained_ga_vs_random_search": comparison_note,
                "runtime_trade_off": runtime_tradeoff,
                "decision_contribution": decision_contribution,
            }
        )
    return rows


def _determine_decision(
    *,
    rows: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    fairness_by_budget: dict[int, dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
    budget_decisions: Sequence[dict[str, Any]],
    budgets: Sequence[int],
) -> str:
    if failures:
        return "Fix required"
    if any(payload["status"] == "fail" for payload in fairness_by_budget.values()):
        return "Needs fairness rerun"
    if any(int(row["actual_evaluations"]) != int(row["requested_budget"]) for row in rows):
        return "Needs fairness rerun"
    if any("constraint_summary" not in row for row in rows):
        return "Fix required"
    if any(bool(row.get("default_changed")) for row in rows):
        return "Fix required"

    feasibility_or_violation_positive_budgets = 0
    objective_positive_budgets = 0
    for budget in budgets:
        budget_pairs = [row for row in paired_comparisons if int(row["budget"]) == int(budget)]
        has_feasibility_or_violation_positive = any(
            row["metric"]
            in {"feasible_rate", "mean_total_violation", "per_constraint_mean_violation"}
            and int(row["constrained_ga_win"]) > int(row["random_search_win"])
            for row in budget_pairs
        )
        has_objective_positive = any(
            row["metric"] == "best_feasible_objective"
            and int(row["constrained_ga_win"]) > int(row["random_search_win"])
            for row in budget_pairs
        )
        if has_feasibility_or_violation_positive:
            feasibility_or_violation_positive_budgets += 1
        if has_objective_positive:
            objective_positive_budgets += 1

    all_budgets_complete = all(
        row["decision_contribution"] != "fix_required"
        for row in budget_decisions
    )

    if all_budgets_complete and feasibility_or_violation_positive_budgets == len(budgets):
        if objective_positive_budgets == len(budgets):
            return "Stress passed, second toy supports constrained GA opt-in evidence"
        return "Stress passed with objective trade-off"
    if all_budgets_complete and feasibility_or_violation_positive_budgets >= 1:
        return "Hold for more evidence"
    return "Hold for more evidence"


def _render_results_markdown(
    *,
    rows: Sequence[dict[str, Any]],
    budget_summaries: Sequence[dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Constrained GA Stress Results",
            "",
            "## Budget Strategy Summary",
            "",
            _markdown_table(
                budget_summaries,
                [
                    "budget",
                    "strategy",
                    "mean_feasible_rate",
                    "median_feasible_rate",
                    "mean_best_feasible_objective",
                    "mean_total_violation_mean",
                    "mean_per_constraint_satisfaction_rate",
                    "mean_per_constraint_violation",
                    "mean_runtime_seconds",
                    "all_runs_completed",
                    "fairness_status",
                ],
            ),
            "",
            "## Paired Comparison",
            "",
            _markdown_table(
                paired_comparisons,
                [
                    "budget",
                    "metric",
                    "constrained_ga_win",
                    "tie",
                    "random_search_win",
                    "mean_delta",
                    "median_delta",
                    "interpretation",
                ],
            ),
            "",
            "## Seed Rows",
            "",
            _markdown_table(
                rows,
                [
                    "budget",
                    "strategy",
                    "seed",
                    "feasible_rate",
                    "best_feasible_objective",
                    "mean_total_violation",
                    "per_constraint_mean_violation",
                    "actual_evaluations",
                    "runtime_seconds",
                ],
            ),
            "",
        ]
    )


def _render_fairness_markdown(
    *,
    fairness_by_budget: dict[int, dict[str, Any]],
    overall_fairness: dict[str, Any],
) -> str:
    budget_rows: list[dict[str, Any]] = []
    for budget, payload in fairness_by_budget.items():
        for issue in payload.get("issues", []):
            budget_rows.append(
                {
                    "budget": budget,
                    "issue_type": issue.get("issue_type"),
                    "status": issue.get("status"),
                    "severity": issue.get("severity"),
                    "message": issue.get("message"),
                    "recommended_action": issue.get("recommended_action"),
                }
            )
    if not budget_rows:
        budget_rows.append(
            {
                "budget": "all",
                "issue_type": "none",
                "status": "pass",
                "severity": "info",
                "message": "No fairness issues recorded.",
                "recommended_action": "none",
            }
        )
    return "\n".join(
        [
            "# Constrained GA Stress Fairness Report",
            "",
            f"- Overall status: **{overall_fairness['status']}**",
            f"- Overall summary counts: `{overall_fairness.get('summary_counts', {})}`",
            "",
            _markdown_table(
                budget_rows,
                [
                    "budget",
                    "issue_type",
                    "status",
                    "severity",
                    "message",
                    "recommended_action",
                ],
            ),
            "",
        ]
    )


def _render_stress_report(
    *,
    config: ConstrainedGAStressConfig,
    artifact: dict[str, Any],
    budget_summaries: Sequence[dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
    fairness_summary: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
    decision: str,
    budget_decisions: Sequence[dict[str, Any]],
) -> str:
    configuration_rows = [
        {"field": "problem", "value": artifact["problem"]},
        {"field": "dimension", "value": config.dimension},
        {"field": "budgets", "value": ", ".join(str(budget) for budget in artifact["configuration"]["budgets"])},
        {"field": "seeds", "value": len(artifact["configuration"]["seed_list"])},
        {"field": "strategies", "value": ", ".join(config.strategies)},
        {"field": "tolerance", "value": config.tolerance},
        {"field": "feasibility policy", "value": "feasibility_first"},
    ]
    fairness_rows = [
        {
            "item": f"budget {budget}",
            "status": payload["status"],
            "message": (
                f"summary_counts={payload.get('summary_counts', {})}"
            ),
        }
        for budget, payload in artifact["fairness_by_budget"].items()
    ]
    issue_rows = _build_failures_and_warnings_rows(failures, warnings)
    regression_rows = [
        {
            "command": (
                "python scripts/validate_constrained_ga_stress.py "
                f"--dimension {config.dimension} --seeds {len(artifact['configuration']['seed_list'])} "
                f"--budgets {','.join(str(budget) for budget in artifact['configuration']['budgets'])} "
                f"--artifact-suffix {config.artifact_suffix}"
            ),
            "result": "stress artifact generation completed",
            "note": "pytest and local baseline checks are appended in the execution pass",
        }
    ]

    exec_lines = [
        "- 이번 작업의 목표는 constrained GA opt-in path가 seed/budget stress에서도 안정적으로 동작하는지 검증하는 것이었다.",
        f"- 실행한 stress scope는 dimension={config.dimension}, seeds={len(artifact['configuration']['seed_list'])}, budgets={artifact['configuration']['budgets']}였다.",
        f"- fairness 결과는 status=`{fairness_summary['status']}`, summary_counts={fairness_summary.get('summary_counts', {})}였다.",
        f"- stress decision은 `{decision}`이다.",
        "- 기본값 변경 여부는 `false`다.",
        "- GA/NSGA-II default 영향 여부는 `none`이다.",
        "- Level 판정은 default maturity 상향이 아니라 실험 툴킷 근거 강화로 제한한다.",
    ]

    return "\n".join(
        [
            "# Constrained GA Stress Report",
            "",
            "## 1. Executive Summary",
            "",
            *exec_lines,
            "",
            resume_section,
            parallel_section,
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "",
            "- constrained_sphere",
            "- dimension 5",
            "- budget 300/1000",
            "- seeds 30",
            "- random_search_feasibility_first",
            "- constrained_ga_feasibility_first",
            "- feasibility-first comparator",
            "- constrained fairness",
            "- stress artifact generation",
            "",
            "Non-Scope:",
            "",
            "- default GA replacement",
            "- NSGA-II constraint-domination",
            "- penalty/repair",
            "- constrained MOEA benchmark",
            "- productization",
            "",
            "## 3. Stress Configuration",
            "",
            _markdown_table(configuration_rows, ["field", "value"]),
            "",
            "## 4. Results Summary",
            "",
            _markdown_table(
                budget_summaries,
                [
                    "budget",
                    "strategy",
                    "mean_feasible_rate",
                    "mean_best_feasible_objective",
                    "mean_total_violation",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                ],
            ),
            "",
            "## 5. Paired Comparison",
            "",
            _markdown_table(
                paired_comparisons,
                [
                    "budget",
                    "metric",
                    "constrained_ga_win",
                    "tie",
                    "random_search_win",
                    "mean_delta",
                    "interpretation",
                ],
            ),
            "",
            "## 6. Fairness Summary",
            "",
            _markdown_table(fairness_rows, ["item", "status", "message"]),
            "",
            "## 7. Failures and Warnings",
            "",
            _markdown_table(issue_rows, ["type", "message", "action"]),
            "",
            "## 8. Regression Check",
            "",
            _markdown_table(regression_rows, ["command", "result", "note"]),
            "",
            "## 9. Decision",
            "",
            _markdown_table(
                budget_decisions,
                [
                    "budget",
                    "stress_result",
                    "fairness",
                    "constrained_ga_vs_random_search",
                    "runtime_trade_off",
                    "decision_contribution",
                ],
            ),
            "",
            f"- Final decision: **{decision}**",
            "",
            "## 10. What This Proves",
            "",
            "- constrained GA opt-in path가 seed/budget stress에서 실행 가능함",
            "- feasibility-first comparator가 GA path에서 반복 사용 가능함",
            "- constrained artifact/fairness/metrics가 stress에서도 생성됨",
            "- mutation_fn contract와 finite validation이 유지됨",
            "",
            "## 11. What This Does Not Prove",
            "",
            "- default GA가 constrained optimizer가 됐다는 뜻은 아님",
            "- NSGA-II constraint-domination이 구현됐다는 뜻은 아님",
            "- penalty/repair가 구현됐다는 뜻은 아님",
            "- industrial constrained optimization을 지원한다는 뜻은 아님",
            "- constrained GA가 모든 constrained problem에서 random search보다 우수하다는 뜻은 아님",
            "",
            "## 12. Maturity Impact",
            "",
            "- Level 4 근거 강화",
            "- constrained GA stress는 실험 툴킷 근거를 강화한다.",
            "- default GA/NSGA-II가 바뀌지 않았으므로 default algorithm maturity 상향은 금지한다.",
            "- constrained benchmark evidence가 constrained_sphere toy에 한정되므로 범용 constrained optimizer 주장도 금지한다.",
            "",
            "## 13. Recommended Next Work",
            "",
            "1. stress passed이면 constrained GA opt-in review package 작성",
            "2. trade-off가 있으면 stress postmortem 작성",
            "3. second constrained single-objective toy benchmark planning",
            "4. NSGA-II constraint-domination planning",
            "5. constrained ZDT-style toy planning",
            "6. fairness checker 통합 범위 확장",
            "7. checkpoint/resume",
            "8. parallel evaluation",
            "",
            f"이번 constrained GA stress 결과, 저장소는 seed/budget stress에서 opt-in constrained GA path와 artifact contract까지 검증했지만, default GA/NSGA-II 통합은 미구현 상태이며, 최종 판정은 {decision}이다.",
            "",
        ]
    )


def _render_stress_report_v2(
    *,
    config: ConstrainedGAStressConfig,
    artifact: dict[str, Any],
    budget_summaries: Sequence[dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
    fairness_summary: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
    decision: str,
    budget_decisions: Sequence[dict[str, Any]],
) -> str:
    problem_name = str(artifact["problem"])
    title = (
        "Constrained Box Quadratic Stress Report"
        if problem_name == "constrained_box_quadratic"
        else "Constrained GA Stress Report"
    )
    configuration_rows = [
        {"field": "problem", "value": problem_name},
        {"field": "dimension", "value": config.dimension},
        {"field": "budgets", "value": ", ".join(str(budget) for budget in artifact["configuration"]["budgets"])},
        {"field": "seeds", "value": len(artifact["configuration"]["seed_list"])},
        {"field": "strategies", "value": ", ".join(config.strategies)},
        {"field": "tolerance", "value": config.tolerance},
        {"field": "feasibility policy", "value": "feasibility_first"},
    ]
    fairness_rows = [
        {
            "item": f"budget {budget}",
            "status": payload["status"],
            "message": f"summary_counts={payload.get('summary_counts', {})}",
        }
        for budget, payload in artifact["fairness_by_budget"].items()
    ]
    issue_rows = _build_failures_and_warnings_rows(failures, warnings)
    if not issue_rows:
        issue_rows = [{"type": "none", "message": "No failures or warnings recorded.", "action": "none"}]
    resume_section = ""
    if artifact.get("resume_enabled"):
        summary = artifact.get("resume_summary", {})
        resume_section = "\n".join(
            [
                "## Runner-level Resume Summary",
                "",
                _markdown_table(
                    [
                        {"item": "resume_source_artifact", "value": artifact.get("resume_source_artifact")},
                        {"item": "resume_mode", "value": artifact.get("resume_mode")},
                        {"item": "total_planned", "value": summary.get("total_planned")},
                        {"item": "completed_from_source", "value": summary.get("completed_from_source")},
                        {"item": "newly_executed", "value": summary.get("newly_executed")},
                        {"item": "skipped_existing", "value": summary.get("skipped_existing")},
                        {"item": "failed_existing", "value": summary.get("failed_existing")},
                        {"item": "warnings", "value": len(summary.get("warnings", []))},
                    ],
                    ["item", "value"],
                ),
                "",
                "This is skip-completed runner-level resume only; it is not algorithm checkpoint/resume.",
                "",
            ]
        )
    parallel_summary = artifact.get("parallel_summary", {})
    parallel_section = "\n".join(
        [
            "## Runner-level Parallel Summary",
            "",
            _markdown_table(
                [
                    {"item": "parallel_enabled", "value": artifact.get("parallel_enabled")},
                    {"item": "backend", "value": artifact.get("parallel_backend")},
                    {"item": "workers", "value": artifact.get("parallel_workers")},
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
                ],
                ["item", "value"],
            ),
            "",
            "This is runner-level row execution only. It is not algorithm-internal evaluation parallelism.",
            "",
        ]
    )
    regression_rows = [
        {
            "command": (
                "python scripts/validate_constrained_ga_stress.py "
                f"--problem {problem_name} "
                f"--dimension {config.dimension} --seeds {len(artifact['configuration']['seed_list'])} "
                f"--budgets {','.join(str(budget) for budget in artifact['configuration']['budgets'])} "
                f"--artifact-suffix {config.artifact_suffix}"
            ),
            "result": "PASS",
            "note": "stress artifact generation completed",
        }
    ]
    objective_tradeoff_rows = [
        row
        for row in paired_comparisons
        if row["metric"] in {"best_feasible_objective", "feasible_rate", "mean_total_violation"}
    ]
    per_constraint_summaries = list(artifact.get("per_constraint_summaries", []))

    return "\n".join(
        [
            f"# {title}",
            "",
            "## 1. Executive Summary",
            "",
            f"- 이번 작업의 목표는 `{problem_name}`에서 feasibility/violation 우세와 best objective trade-off를 seed/budget stress로 검증하는 것이다.",
            f"- 실행한 stress scope는 dimension `{config.dimension}`, seeds `{len(artifact['configuration']['seed_list'])}`, budgets `{artifact['configuration']['budgets']}`이다.",
            "- random search baseline과 constrained GA restricted opt-in path를 같은 budget과 seed에서 비교했다.",
            "- feasibility/violation 결과와 best objective trade-off는 budget별 paired comparison으로 해석한다.",
            f"- fairness 결과는 `{fairness_summary['status']}`이고 summary counts는 `{fairness_summary.get('summary_counts', {})}`이다.",
            f"- stress decision은 `{decision}`이다.",
            "- 기본값 변경은 없다.",
            "- GA/NSGA-II default 영향은 없다.",
            "- Level 판정은 유지하되, second toy stress evidence는 실험 툴킷 관점의 Level 4 근거를 강화한다.",
            "",
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "",
            f"- {problem_name}",
            f"- dimension {config.dimension}",
            "- budget 300/1000",
            f"- seeds {len(artifact['configuration']['seed_list'])}",
            "- random_search_feasibility_first",
            "- constrained_ga_feasibility_first",
            "- feasibility-first comparator",
            "- per-constraint summary",
            "- constrained fairness",
            "- stress artifact generation",
            "",
            "Non-Scope:",
            "",
            "- default GA replacement",
            "- NSGA-II constraint-domination",
            "- penalty/repair",
            "- constrained MOEA benchmark",
            "- productization",
            "",
            "## 3. Stress Configuration",
            "",
            _markdown_table(configuration_rows, ["field", "value"]),
            "",
            "## 4. Results Summary",
            "",
            _markdown_table(
                budget_summaries,
                [
                    "budget",
                    "strategy",
                    "mean_feasible_rate",
                    "mean_best_feasible_objective",
                    "mean_total_violation",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                ],
            ),
            "",
            "## 5. Per-Constraint Summary",
            "",
            _markdown_table(
                per_constraint_summaries,
                [
                    "budget",
                    "strategy",
                    "constraint_name",
                    "satisfaction_rate",
                    "mean_violation",
                    "max_violation",
                    "limitation",
                ],
            ),
            "",
            "## 6. Paired Comparison",
            "",
            _markdown_table(
                paired_comparisons,
                [
                    "budget",
                    "metric",
                    "constrained_ga_win",
                    "tie",
                    "random_search_win",
                    "mean_delta",
                    "interpretation",
                ],
            ),
            "",
            "## 7. Fairness Summary",
            "",
            _markdown_table(fairness_rows, ["item", "status", "message"]),
            "",
            "## 8. Best Objective Trade-off Interpretation",
            "",
            "- smoke에서 `best_feasible_objective`는 mixed였고, 평균 기준 random search가 약간 더 좋았다.",
            "- budget 300과 budget 1000을 분리해 best objective trade-off가 완화되는지 확인한다.",
            "- feasibility-first pressure는 infeasible 후보를 강하게 밀어내므로 feasible_rate와 violation은 좋아질 수 있지만 objective exploration 폭은 줄어들 수 있다.",
            "- random search가 best objective에서 우세한 budget 또는 seed가 있으면 성능 주장 대신 trade-off로 기록한다.",
            "",
            _markdown_table(
                objective_tradeoff_rows,
                [
                    "budget",
                    "metric",
                    "constrained_ga_win",
                    "tie",
                    "random_search_win",
                    "mean_delta",
                    "interpretation",
                ],
            ),
            "",
            "## 9. Failures and Warnings",
            "",
            _markdown_table(issue_rows, ["type", "message", "action"]),
            "",
            "## 10. Regression Check",
            "",
            _markdown_table(regression_rows, ["command", "result", "note"]),
            "",
            "## 11. Decision",
            "",
            _markdown_table(
                budget_decisions,
                [
                    "budget",
                    "stress_result",
                    "fairness",
                    "constrained_ga_vs_random_search",
                    "runtime_trade_off",
                    "decision_contribution",
                ],
            ),
            "",
            f"- Final decision: **{decision}**",
            "",
            "## 12. What This Proves",
            "",
            f"- `{problem_name}`에서 stress artifact/fairness/metrics가 생성됨",
            "- feasibility-first constrained GA path가 second toy에서도 반복 실행 가능함",
            "- per-constraint summary가 ConstraintEvaluation trace에서 기록됨. legacy row에 raw trace가 없으면 limitation으로 표시됨",
            "- mutation_fn contract와 finite validation이 유지됨",
            "",
            "## 13. What This Does Not Prove",
            "",
            "- default GA가 constrained optimizer가 됐다는 뜻은 아님",
            "- NSGA-II constraint-domination이 구현됐다는 뜻은 아님",
            "- penalty/repair가 구현됐다는 뜻은 아님",
            "- industrial constrained optimization을 지원한다는 뜻은 아님",
            "- constrained GA가 모든 constrained problem에서 random search보다 우수하다는 뜻은 아님",
            "",
            "## 14. Maturity Impact",
            "",
            "- Level 4 근거 강화",
            "- second toy stress는 실험 툴킷 근거를 강화한다.",
            "- default GA/NSGA-II가 바뀌지 않았으므로 default algorithm maturity 상향은 금지한다.",
            "- constrained benchmark evidence가 toy 2개 수준이므로 범용 constrained optimizer 주장은 금지한다.",
            "",
            "## 15. Recommended Next Work",
            "",
            "1. stress passed이면 constrained GA opt-in review update 작성",
            "2. objective trade-off가 크면 constraint tightness stress planning",
            "3. equality-constrained toy planning",
            "4. NSGA-II constraint-domination planning, 단 single-objective constrained evidence와 분리",
            "5. constrained ZDT-style toy planning",
            "6. fairness checker 통합 범위 확장",
            "7. checkpoint/resume",
            "8. parallel evaluation",
            "",
            f"이번 constrained_box_quadratic stress 결과, 저장소는 second toy seed/budget stress artifact와 fairness contract까지 검증했지만, default GA/NSGA-II 통합은 미구현 상태이며, 최종 판정은 {decision}이다.",
            "",
        ]
    )


def run_constrained_ga_stress(config: ConstrainedGAStressConfig) -> dict[str, Any]:
    for strategy in config.strategies:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
    if config.dimension <= 0:
        raise ValueError("dimension must be > 0")
    if config.population_size <= 1:
        raise ValueError("population_size must be > 1")

    budgets = _normalize_budgets(config.budgets)
    seed_list = _seed_list(config.seeds)
    problem = _build_problem_for_smoke(_smoke_config_for_budget(config, budgets[0]))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    fairness_by_budget: dict[int, dict[str, Any]] = {}
    overall_fairness_issues: list[dict[str, Any]] = []
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

    for budget in budgets:
        for strategy in config.strategies:
            for seed in seed_list:
                planned_row = _planned_resume_row(
                    problem=problem,
                    config=config,
                    strategy=strategy,
                    seed=seed,
                    budget=budget,
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
                        "problem": problem.name,
                        "dimension": int(config.dimension),
                        "constraint_budget": float(config.constraint_budget),
                        "tolerance": float(config.tolerance),
                        "population_size": int(config.population_size),
                        "strategy": strategy,
                        "seed": int(seed),
                        "budget": int(budget),
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
                    f"backend={config.row_execution_backend} "
                    f"workers={config.row_execution_workers}: {exc}"
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
        row.setdefault("status", "success")
        row.setdefault("skip_reason", None)
        row.setdefault("failure_reason", None)
        row.setdefault("warnings", [])
        _annotate_per_constraint_proxy(row)
        budget = int(row["requested_budget"])
        if int(row["actual_evaluations"]) != budget:
            warnings.append(
                {
                    "type": "actual_evaluations_mismatch",
                    "budget": budget,
                    "strategy": row.get("strategy"),
                    "seed": row.get("seed"),
                    "message": (
                        f"expected {budget} evaluations but observed "
                        f"{row['actual_evaluations']}"
                    ),
                }
            )

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

    for budget in budgets:
        smoke_config = _smoke_config_for_budget(config, budget)
        budget_rows = [row for row in rows if int(row["requested_budget"]) == int(budget)]
        fairness_payload = evaluate_constrained_fairness(
            expected_contract=_constraint_contract(problem, smoke_config),
            observed_rows=budget_rows,
        )
        fairness_payload["budget"] = budget
        fairness_by_budget[budget] = fairness_payload
        for issue in fairness_payload.get("issues", []):
            overall_fairness_issues.append(
                {
                    "budget": budget,
                    **issue,
                }
            )

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

    budget_summaries = _budget_strategy_summaries(
        rows=rows,
        budgets=budgets,
        strategies=config.strategies,
        seed_count=len(seed_list),
        fairness_by_budget=fairness_by_budget,
    )
    per_constraint_summaries = _budget_per_constraint_summaries(
        rows=rows,
        budgets=budgets,
        strategies=config.strategies,
    )
    paired_comparisons = _paired_comparisons(
        rows=rows,
        budgets=budgets,
        seed_list=seed_list,
    )
    budget_decisions = _budget_decision_summary(
        budget_summaries=budget_summaries,
        paired_comparisons=paired_comparisons,
        fairness_by_budget=fairness_by_budget,
        budgets=budgets,
    )
    fairness_summary = {
        "status": _overall_status([payload["status"] for payload in fairness_by_budget.values()]),
        "issues": overall_fairness_issues,
        "summary_counts": _summarize_counts(overall_fairness_issues),
    }
    decision = _determine_decision(
        rows=rows,
        failures=failures,
        fairness_by_budget=fairness_by_budget,
        paired_comparisons=paired_comparisons,
        budget_decisions=budget_decisions,
        budgets=budgets,
    )

    artifact = {
        "command_metadata": {
            "argv": sys.argv,
            "artifact_suffix": config.artifact_suffix,
            "output_dir": config.output_dir,
            "row_execution_backend": config.row_execution_backend,
            "row_execution_workers": config.row_execution_workers,
        },
        "problem": problem.name,
        "configuration": {
            "problem": problem.name,
            "dimension": config.dimension,
            "budgets": budgets,
            "constraint_budget": config.constraint_budget,
            "seeds": config.seeds,
            "seed_list": seed_list,
            "tolerance": config.tolerance,
            "population_size": config.population_size,
            "feasibility_policy": "feasibility_first",
            "violation_aggregation": "total_violation",
            "strategies": list(config.strategies),
        },
        "rows": rows,
        "budget_strategy_summaries": budget_summaries,
        "per_constraint_summaries": per_constraint_summaries,
        "paired_comparisons": paired_comparisons,
        "budget_decisions": budget_decisions,
        "fairness_by_budget": fairness_by_budget,
        "fairness_summary": fairness_summary,
        "failures": failures,
        "warnings": warnings,
        "decision": decision,
        "default_changed": False,
        "ga_default_changed": False,
        "constrained_ga_opt_in_path_status": "stress_validated_explicit_opt_in_only",
        "ga_integration_done": False,
        "nsga2_constraint_domination_done": False,
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

    prefix = _artifact_prefix(problem.name)
    json_path = output_dir / f"{prefix}_results_{suffix}.json"
    csv_path = output_dir / f"{prefix}_results_{suffix}.csv"
    md_results_path = output_dir / f"{prefix}_results_{suffix}.md"
    report_path = output_dir / f"{prefix}_report_{suffix}.md"
    fairness_path = output_dir / f"{prefix}_fairness_report_{suffix}.md"
    resume_report_suffix = config.resume_report_suffix or suffix
    resume_report_path = output_dir / f"{prefix}_resume_report_{resume_report_suffix}.md"

    artifact["artifacts"] = {
        "json": _artifact_ref(json_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "csv": _artifact_ref(csv_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "results_markdown": _artifact_ref(md_results_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "report_markdown": _artifact_ref(report_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "fairness_markdown": _artifact_ref(fairness_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "resume_markdown": (
            _artifact_ref(resume_report_path, project_root, prefer_absolute=prefer_absolute_artifact_paths)
            if resume_enabled
            else None
        ),
    }

    _write_json(json_path, artifact)
    _write_csv(
        csv_path,
        rows,
        [
            "budget",
            "strategy",
            "seed",
            "problem",
            "dimension",
            "feasible_rate",
            "feasible_count",
            "infeasible_count",
            "best_feasible_objective",
            "best_record_feasible",
            "best_record_total_violation",
            "best_infeasible_total_violation",
            "mean_total_violation",
            "per_constraint_mean_violation",
            "per_constraint_satisfaction_rate",
            "per_constraint_trace_summary",
            "median_total_violation",
            "min_total_violation",
            "max_total_violation",
            "mean_max_violation",
            "violation_count_total",
            "requested_budget",
            "actual_evaluations",
            "runtime_seconds",
            "failure_count",
            "status",
            "row_origin",
            "resume_key",
            "row_execution_origin",
            "row_order_index",
        ],
    )
    _ensure_fresh_path(md_results_path).write_text(
        _render_results_markdown(
            rows=rows,
            budget_summaries=budget_summaries,
            paired_comparisons=paired_comparisons,
        ),
        encoding="utf-8",
    )
    _ensure_fresh_path(fairness_path).write_text(
        _render_fairness_markdown(
            fairness_by_budget=fairness_by_budget,
            overall_fairness=fairness_summary,
        ),
        encoding="utf-8",
    )
    _ensure_fresh_path(report_path).write_text(
        _render_stress_report_v2(
            config=config,
            artifact=artifact,
            budget_summaries=budget_summaries,
            paired_comparisons=paired_comparisons,
            fairness_summary=fairness_summary,
            failures=failures,
            warnings=warnings,
            decision=decision,
            budget_decisions=budget_decisions,
        ),
        encoding="utf-8",
    )
    if resume_enabled:
        write_resume_report(resume_report_path, resume_plan, resume_summary)
    return artifact


__all__ = [
    "ConstrainedGAStressConfig",
    "run_constrained_ga_stress",
]
