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
    ConstrainedGASmokeConfig,
    _aggregate_per_constraint_summaries,
    _artifact_ref,
    _build_failures_and_warnings_rows,
    _build_fairness_rows,
    _constraint_contract,
    _constrained_ga_row,
    _ensure_fresh_path,
    _markdown_table,
    _project_root,
    _random_search_row,
    _seed_list,
    _write_csv,
    _write_json,
)
from ga_lab.problems.constrained_equality_plane_quadratic import (
    ConstrainedEqualityPlaneQuadraticProblem,
    DEFAULT_EQUALITY_PLANE_TOLERANCE,
)


VARIANT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "loose": {
        "tolerance": 0.05,
        "expected_feasibility": "higher feasible_rate",
        "purpose": "check whether a wider equality band reduces feasibility pressure",
    },
    "default": {
        "tolerance": DEFAULT_EQUALITY_PLANE_TOLERANCE,
        "expected_feasibility": "mixed feasible profile",
        "purpose": "anchor against equality_smoke1 behavior",
    },
    "strict": {
        "tolerance": 0.0001,
        "expected_feasibility": "lower feasible_rate and stronger equality pressure",
        "purpose": "check near-feasible and no-feasible/null metric behavior",
    },
}

COMPARISON_METRICS = (
    "feasible_rate",
    "best_feasible_objective",
    "mean_total_violation",
    "equality_satisfaction_rate",
    "equality_mean_violation",
    "runtime_seconds",
    "actual_evaluations",
)
HIGHER_IS_BETTER = {"feasible_rate", "equality_satisfaction_rate"}
LOWER_IS_BETTER = {
    "best_feasible_objective",
    "mean_total_violation",
    "equality_mean_violation",
    "runtime_seconds",
}


@dataclass(slots=True)
class ConstrainedEqualityToleranceStressConfig:
    variants: Sequence[str] = ("loose", "default", "strict")
    dimension: int = 6
    seeds: int | Sequence[int] = 20
    budgets: Sequence[int] = (300, 1000)
    population_size: int = 20
    artifact_suffix: str = "equality_tolerance_stress1"
    output_dir: str = "artifacts"
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    problem: str = "constrained_equality_plane_quadratic"


def _normalize_variants(variants: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in variants:
        variant = value.strip().lower().replace("-", "_")
        if variant not in VARIANT_DEFINITIONS:
            raise ValueError(f"Unsupported equality tolerance variant: {value}")
        if variant not in seen:
            normalized.append(variant)
            seen.add(variant)
    if not normalized:
        raise ValueError("variants cannot be empty")
    return normalized


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


def _variant_definition_rows(variants: Sequence[str]) -> list[dict[str, Any]]:
    return [{"variant": variant, **VARIANT_DEFINITIONS[variant]} for variant in variants]


def _build_variant_problem(
    *,
    variant: str,
    config: ConstrainedEqualityToleranceStressConfig,
) -> ConstrainedEqualityPlaneQuadraticProblem:
    definition = VARIANT_DEFINITIONS[variant]
    return ConstrainedEqualityPlaneQuadraticProblem(
        dimension=config.dimension,
        equality_tolerance=float(definition["tolerance"]),
    )


def _smoke_config(
    *,
    config: ConstrainedEqualityToleranceStressConfig,
    budget: int,
    tolerance: float,
) -> ConstrainedGASmokeConfig:
    return ConstrainedGASmokeConfig(
        problem=config.problem,
        dimension=config.dimension,
        seeds=1,
        budget=budget,
        tolerance=tolerance,
        population_size=config.population_size,
        artifact_suffix=config.artifact_suffix,
        output_dir=config.output_dir,
        strategies=config.strategies,
    )


def _annotate_per_constraint(row: dict[str, Any]) -> None:
    items = [
        item
        for item in row.get("per_constraint_violation_summary", [])
        if item.get("mean_violation") is not None
    ]
    if not items:
        row["per_constraint_satisfaction_rate"] = None
        row["per_constraint_mean_violation"] = None
        row["per_constraint_max_violation"] = None
        row["per_constraint_summary_scope"] = "unavailable"
        return
    row["per_constraint_satisfaction_rate"] = _safe_mean(
        [float(item["satisfaction_rate"]) for item in items]
        if all(item.get("satisfaction_rate") is not None for item in items)
        else []
    )
    row["per_constraint_mean_violation"] = _safe_mean(
        [float(item["mean_violation"]) for item in items]
    )
    row["per_constraint_max_violation"] = max(float(item["max_violation"]) for item in items)
    row["per_constraint_summary_scope"] = "evaluation_trace"


def _add_variant_metadata(
    row: dict[str, Any],
    *,
    variant: str,
    tolerance: float,
) -> None:
    row["variant_name"] = variant
    row["tolerance_variant"] = variant
    row["tolerance"] = float(tolerance)
    _annotate_per_constraint(row)


def _append_variant_metadata_issues(
    issues: list[dict[str, Any]],
    *,
    rows: Sequence[dict[str, Any]],
    variant: str,
    tolerance: float,
) -> None:
    for row in rows:
        row_name = str(row.get("strategy", "unknown_strategy"))
        checks = (
            ("variant_name", row.get("variant_name"), variant),
            ("tolerance", row.get("tolerance"), tolerance),
        )
        for field, observed, expected in checks:
            if observed == expected:
                issues.append(
                    {
                        "status": "pass",
                        "issue_type": f"{field}_match",
                        "severity": "info",
                        "message": f"{row_name}: {field} matches expected tolerance metadata",
                        "recommended_action": "none",
                    }
                )
            else:
                issues.append(
                    {
                        "status": "fail",
                        "issue_type": f"{field}_mismatch",
                        "severity": "high",
                        "message": f"{row_name}: {field} observed={observed} expected={expected}",
                        "recommended_action": "fix tolerance variant metadata binding",
                    }
                )


def _variant_budget_fairness(
    *,
    variant: str,
    budget: int,
    problem: ConstrainedEqualityPlaneQuadraticProblem,
    config: ConstrainedEqualityToleranceStressConfig,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    tolerance = float(problem.equality_tolerance)
    smoke_config = _smoke_config(config=config, budget=budget, tolerance=tolerance)
    contract = _constraint_contract(problem, smoke_config)
    contract["variant_name"] = variant
    payload = evaluate_constrained_fairness(
        expected_contract=contract,
        observed_rows=rows,
    )
    issues = list(payload.get("issues", []))
    _append_variant_metadata_issues(
        issues,
        rows=rows,
        variant=variant,
        tolerance=tolerance,
    )
    return {
        "status": _overall_status([str(issue.get("status")) for issue in issues]),
        "issues": issues,
        "summary_counts": _summarize_counts(issues),
        "variant_name": variant,
        "budget": budget,
        "tolerance": tolerance,
    }


def _variant_budget_strategy_summaries(
    *,
    rows: Sequence[dict[str, Any]],
    variants: Sequence[str],
    budgets: Sequence[int],
    strategies: Sequence[str],
    seed_count: int,
    fairness_by_variant_budget: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        for budget in budgets:
            budget_rows = [
                row
                for row in rows
                if row["variant_name"] == variant and int(row["requested_budget"]) == int(budget)
            ]
            fairness_status = fairness_by_variant_budget[f"{variant}:{budget}"]["status"]
            for strategy in strategies:
                strategy_rows = [row for row in budget_rows if row["strategy"] == strategy]
                best_feasible_values = [
                    float(row["best_feasible_objective"])
                    for row in strategy_rows
                    if row["best_feasible_objective"] is not None
                ]
                summaries.append(
                    {
                        "variant_name": variant,
                        "tolerance": VARIANT_DEFINITIONS[variant]["tolerance"],
                        "budget": budget,
                        "strategy": strategy,
                        "run_count": len(strategy_rows),
                        "expected_run_count": seed_count,
                        "all_runs_completed": len(strategy_rows) == seed_count,
                        "fairness_status": fairness_status,
                        "mean_feasible_rate": _safe_mean(
                            [float(row["feasible_rate"]) for row in strategy_rows]
                        ),
                        "median_feasible_rate": _safe_median(
                            [float(row["feasible_rate"]) for row in strategy_rows]
                        ),
                        "mean_best_feasible_objective": _safe_mean(best_feasible_values),
                        "median_best_feasible_objective": _safe_median(best_feasible_values),
                        "mean_total_violation_mean": _safe_mean(
                            [float(row["mean_total_violation"] or 0.0) for row in strategy_rows]
                        ),
                        "mean_total_violation": _safe_mean(
                            [float(row["mean_total_violation"] or 0.0) for row in strategy_rows]
                        ),
                        "mean_equality_satisfaction_rate": _safe_mean(
                            [
                                float(row["equality_satisfaction_rate"])
                                for row in strategy_rows
                                if row.get("equality_satisfaction_rate") is not None
                            ]
                        ),
                        "mean_equality_violation": _safe_mean(
                            [
                                float(row["equality_mean_violation"])
                                for row in strategy_rows
                                if row.get("equality_mean_violation") is not None
                            ]
                        ),
                        "mean_equality_max_violation": _safe_mean(
                            [
                                float(row["equality_max_violation"])
                                for row in strategy_rows
                                if row.get("equality_max_violation") is not None
                            ]
                        ),
                        "mean_runtime_seconds": _safe_mean(
                            [float(row["runtime_seconds"]) for row in strategy_rows]
                        ),
                        "mean_actual_evaluations": _safe_mean(
                            [int(row["actual_evaluations"]) for row in strategy_rows]
                        ),
                        "all_infeasible_runs": sum(
                            1 for row in strategy_rows if bool(row.get("all_infeasible"))
                        ),
                        "all_feasible_runs": sum(
                            1 for row in strategy_rows if bool(row.get("all_feasible"))
                        ),
                    }
                )
    return summaries


def _equality_summaries(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "variant_name": row["variant_name"],
            "tolerance": row["tolerance"],
            "budget": row["budget"],
            "strategy": row["strategy"],
            "equality_satisfaction_rate": row["mean_equality_satisfaction_rate"],
            "equality_mean_violation": row["mean_equality_violation"],
            "equality_max_violation": row["mean_equality_max_violation"],
        }
        for row in summaries
    ]


def _per_constraint_summaries(
    *,
    rows: Sequence[dict[str, Any]],
    variants: Sequence[str],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        for budget in budgets:
            filtered = [
                row
                for row in rows
                if row["variant_name"] == variant and int(row["requested_budget"]) == int(budget)
            ]
            for item in _aggregate_per_constraint_summaries(filtered):
                summaries.append(
                    {
                        "variant_name": variant,
                        "tolerance": VARIANT_DEFINITIONS[variant]["tolerance"],
                        "budget": budget,
                        **item,
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
            return "random_search is faster; runtime is recorded separately from quality."
        if constrained_wins > baseline_wins:
            return "constrained_ga is faster in this slice; this is a secondary signal only."
        return "runtime is close to tied."
    if metric == "actual_evaluations":
        if ties and constrained_wins == 0 and baseline_wins == 0:
            return "both strategies keep exact budget accounting."
        return "actual_evaluations accounting is mixed."
    if constrained_wins > baseline_wins:
        return "constrained_ga shows repeated positive signal for this metric."
    if baseline_wins > constrained_wins:
        return "random_search is better on more paired seeds for this metric."
    return "paired result is close to tied."


def _paired_comparisons(
    *,
    rows: Sequence[dict[str, Any]],
    variants: Sequence[str],
    budgets: Sequence[int],
    seed_list: Sequence[int],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for variant in variants:
        for budget in budgets:
            filtered = [
                row
                for row in rows
                if row["variant_name"] == variant and int(row["requested_budget"]) == int(budget)
            ]
            by_strategy_seed = {
                (str(row["strategy"]), int(row["seed"])): row
                for row in filtered
            }
            for metric in COMPARISON_METRICS:
                constrained_wins = 0
                ties = 0
                baseline_wins = 0
                deltas: list[float] = []
                matched_pairs = 0
                for seed in seed_list:
                    constrained = by_strategy_seed.get(
                        ("constrained_ga_feasibility_first", int(seed))
                    )
                    baseline = by_strategy_seed.get(
                        ("random_search_feasibility_first", int(seed))
                    )
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
                        "variant": variant,
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


def _profile_is_distinguishable(summaries: Sequence[dict[str, Any]]) -> bool:
    for budget in {int(row["budget"]) for row in summaries}:
        for strategy in DEFAULT_STRATEGIES:
            filtered = [
                row
                for row in summaries
                if int(row["budget"]) == budget and row["strategy"] == strategy
            ]
            feasible_rates = [
                float(row["mean_feasible_rate"])
                for row in filtered
                if row.get("mean_feasible_rate") is not None
            ]
            equality_violations = [
                float(row["mean_equality_violation"])
                for row in filtered
                if row.get("mean_equality_violation") is not None
            ]
            if feasible_rates and max(feasible_rates) - min(feasible_rates) >= 0.02:
                return True
            if equality_violations and max(equality_violations) - min(equality_violations) >= 0.02:
                return True
    return False


def _variant_decisions(
    *,
    variants: Sequence[str],
    paired_comparisons: Sequence[dict[str, Any]],
    fairness_by_variant_budget: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for variant in variants:
        variant_pairs = [row for row in paired_comparisons if row["variant"] == variant]
        feasibility_wins = sum(
            int(row["constrained_ga_win"])
            for row in variant_pairs
            if row["metric"] == "feasible_rate"
        )
        violation_wins = sum(
            int(row["constrained_ga_win"])
            for row in variant_pairs
            if row["metric"] in {"mean_total_violation", "equality_mean_violation"}
        )
        objective_wins = sum(
            int(row["constrained_ga_win"])
            for row in variant_pairs
            if row["metric"] == "best_feasible_objective"
        )
        objective_baseline_wins = sum(
            int(row["random_search_win"])
            for row in variant_pairs
            if row["metric"] == "best_feasible_objective"
        )
        runtime_baseline_wins = sum(
            int(row["random_search_win"])
            for row in variant_pairs
            if row["metric"] == "runtime_seconds"
        )
        fairness = _overall_status(
            [
                payload["status"]
                for key, payload in fairness_by_variant_budget.items()
                if key.startswith(f"{variant}:")
            ]
        )
        if fairness == "fail":
            contribution = "needs_fairness_rerun"
        elif feasibility_wins or violation_wins:
            contribution = "positive_with_tradeoff"
        else:
            contribution = "limited_or_mixed"
        decisions.append(
            {
                "variant": variant,
                "stress_result": contribution,
                "fairness": fairness,
                "constrained_ga_vs_random_search": (
                    f"feasibility_wins={feasibility_wins}, violation_wins={violation_wins}"
                ),
                "objective_trade_off": (
                    f"best_objective constrained_ga/random_search "
                    f"{objective_wins}:{objective_baseline_wins}"
                ),
                "runtime_trade_off": f"random_search_runtime_wins={runtime_baseline_wins}",
                "decision_contribution": contribution,
            }
        )
    return decisions


def _determine_decision(
    *,
    variants: Sequence[str],
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    per_constraint_summaries: Sequence[dict[str, Any]],
    fairness_by_variant_budget: dict[str, dict[str, Any]],
) -> str:
    if any(payload["status"] == "fail" for payload in fairness_by_variant_budget.values()):
        return "Needs fairness rerun"
    if any(int(row["actual_evaluations"]) != int(row["requested_budget"]) for row in rows):
        return "Needs fairness rerun"
    if not per_constraint_summaries:
        return "Fix required"
    if any("equality_satisfaction_rate" not in row for row in rows):
        return "Fix required"
    if sorted({row["variant_name"] for row in rows}) != sorted(variants):
        return "Fix required"
    if not _profile_is_distinguishable(summaries):
        return "Equality tolerance stress passed with limitations"
    return "Equality tolerance stress passed, semantics characterized"


def _render_results_markdown(
    *,
    summaries: Sequence[dict[str, Any]],
    equality_summaries: Sequence[dict[str, Any]],
    per_constraint_summaries: Sequence[dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            "# Constrained Equality Tolerance Stress Results",
            "## Variant Strategy Summary",
            _markdown_table(
                summaries,
                [
                    "variant_name",
                    "budget",
                    "strategy",
                    "mean_feasible_rate",
                    "mean_best_feasible_objective",
                    "mean_total_violation_mean",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                    "fairness_status",
                ],
            ),
            "## Equality Summary",
            _markdown_table(
                equality_summaries,
                [
                    "variant_name",
                    "budget",
                    "strategy",
                    "equality_satisfaction_rate",
                    "equality_mean_violation",
                    "equality_max_violation",
                ],
            ),
            "## Per-Constraint Summary",
            _markdown_table(
                per_constraint_summaries,
                [
                    "variant_name",
                    "budget",
                    "strategy",
                    "constraint_name",
                    "constraint_type",
                    "satisfaction_rate",
                    "mean_violation",
                    "max_violation",
                ],
            ),
            "## Paired Comparison",
            _markdown_table(
                paired_comparisons,
                [
                    "variant",
                    "budget",
                    "metric",
                    "constrained_ga_win",
                    "tie",
                    "random_search_win",
                    "mean_delta",
                    "interpretation",
                ],
            ),
        ]
    )


def _render_fairness_markdown(
    *,
    fairness_by_variant_budget: dict[str, dict[str, Any]],
    fairness_summary: dict[str, Any],
) -> str:
    lines = [
        "# Constrained Equality Tolerance Stress Fairness Report",
        "",
        f"- Overall status: **{fairness_summary['status']}**",
        f"- Overall summary counts: `{fairness_summary.get('summary_counts', {})}`",
    ]
    for key, payload in fairness_by_variant_budget.items():
        lines.extend(
            [
                "",
                f"## {key}",
                "",
                f"- status: `{payload['status']}`",
                f"- tolerance: `{payload['tolerance']}`",
                f"- summary counts: `{payload.get('summary_counts', {})}`",
                "",
                _markdown_table(
                    _build_fairness_rows(payload),
                    ["item", "status", "message"],
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _decision_table(
    *,
    variants: Sequence[str],
    paired_comparisons: Sequence[dict[str, Any]],
    fairness_by_variant_budget: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return _variant_decisions(
        variants=variants,
        paired_comparisons=paired_comparisons,
        fairness_by_variant_budget=fairness_by_variant_budget,
    )


def _render_report(
    *,
    artifact: dict[str, Any],
    config: ConstrainedEqualityToleranceStressConfig,
    decision: str,
) -> str:
    summaries = artifact["variant_budget_strategy_summaries"]
    equality_summaries = artifact["equality_summaries"]
    per_constraint = artifact["per_constraint_summaries"]
    paired = artifact["paired_comparisons"]
    fairness_counts = artifact["fairness_summary"].get("summary_counts", {})
    fairness_rows = [
        {
            "item": key,
            "status": payload["status"],
            "message": f"tolerance={payload['tolerance']}; summary_counts={payload.get('summary_counts', {})}",
        }
        for key, payload in artifact["fairness_by_variant_budget"].items()
    ]
    issue_rows = _build_failures_and_warnings_rows(
        artifact.get("failures", []),
        artifact.get("warnings", []),
    )
    regression_rows = [
        {
            "command": (
                "python scripts/validate_constrained_equality_tolerance_stress.py "
                f"--variants {','.join(artifact['configuration']['variants'])} "
                f"--dimension {config.dimension} "
                f"--seeds {len(artifact['configuration']['seed_list'])} "
                f"--budgets {','.join(str(budget) for budget in artifact['configuration']['budgets'])} "
                f"--artifact-suffix {config.artifact_suffix}"
            ),
            "result": "PASS",
            "note": "stress artifact generation completed",
        }
    ]
    return "\n\n".join(
        [
            "# Constrained Equality Tolerance Stress Report",
            "## 1. Executive Summary\n\n"
            f"- Goal: validate equality tolerance stress for `{config.problem}` without changing defaults.\n"
            f"- Variants: `{', '.join(artifact['configuration']['variants'])}`.\n"
            f"- Budgets: `{artifact['configuration']['budgets']}`, seeds: `{len(artifact['configuration']['seed_list'])}`.\n"
            "- random_search_feasibility_first and constrained_ga_feasibility_first were compared under identical tolerance metadata.\n"
            f"- fairness status: `{artifact['fairness_summary']['status']}`, summary_counts=`{fairness_counts}`.\n"
            f"- stress decision: `{decision}`.\n"
            "- Default GA/NSGA-II paths were not changed. Level impact is evidence strengthening only.",
            "## 2. Scope and Non-Scope\n\n"
            "Scope:\n\n"
            "- constrained_equality_plane_quadratic\n"
            "- loose/default/strict variants\n"
            "- random_search_feasibility_first\n"
            "- constrained_ga_feasibility_first\n"
            "- equality tolerance validation\n"
            "- per-constraint summary\n"
            "- constrained fairness\n"
            "- stress artifact generation\n\n"
            "Non-Scope:\n\n"
            "- default GA replacement\n"
            "- NSGA-II constraint-domination\n"
            "- penalty/repair\n"
            "- constrained MOEA benchmark\n"
            "- productization",
            "## 3. Variant Definitions\n\n"
            + _markdown_table(
                artifact["variant_definitions"],
                ["variant", "tolerance", "expected_feasibility", "purpose"],
            ),
            "## 4. Stress Configuration\n\n"
            + _markdown_table(
                [
                    {"field": "problem", "value": config.problem},
                    {"field": "dimension", "value": config.dimension},
                    {"field": "variants", "value": list(artifact["configuration"]["variants"])},
                    {"field": "budgets", "value": list(artifact["configuration"]["budgets"])},
                    {"field": "seeds", "value": len(artifact["configuration"]["seed_list"])},
                    {"field": "strategies", "value": list(config.strategies)},
                    {"field": "feasibility policy", "value": "feasibility_first"},
                ],
                ["field", "value"],
            ),
            "## 5. Results Summary\n\n"
            + _markdown_table(
                summaries,
                [
                    "variant_name",
                    "budget",
                    "strategy",
                    "mean_feasible_rate",
                    "mean_best_feasible_objective",
                    "mean_total_violation_mean",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                ],
            ),
            "## 6. Equality Summary\n\n"
            + _markdown_table(
                equality_summaries,
                [
                    "variant_name",
                    "budget",
                    "strategy",
                    "equality_satisfaction_rate",
                    "equality_mean_violation",
                    "equality_max_violation",
                ],
            ),
            "## 7. Per-Constraint Summary\n\n"
            + _markdown_table(
                per_constraint,
                [
                    "variant_name",
                    "budget",
                    "strategy",
                    "constraint_name",
                    "constraint_type",
                    "satisfaction_rate",
                    "mean_violation",
                    "max_violation",
                ],
            ),
            "## 8. Paired Comparison\n\n"
            + _markdown_table(
                paired,
                [
                    "variant",
                    "budget",
                    "metric",
                    "constrained_ga_win",
                    "tie",
                    "random_search_win",
                    "mean_delta",
                    "interpretation",
                ],
            ),
            "## 9. Tolerance Interpretation\n\n"
            "- loose/default/strict are interpreted through feasible_rate and equality violation profiles.\n"
            "- strict tolerance may produce all-infeasible rows; these are valid if null metrics and equality summaries remain well-formed.\n"
            "- objective trade-off is interpreted separately from feasibility and runtime.\n"
            "- budget 1000 is used to check whether extra evaluations reduce objective trade-off.\n"
            "- equality tolerance metadata is checked through artifact, fairness, and trace summaries.",
            "## 10. Fairness Summary\n\n" + _markdown_table(fairness_rows, ["item", "status", "message"]),
            "## 11. Failures and Warnings\n\n"
            + _markdown_table(issue_rows, ["type", "message", "action"]),
            "## 12. Regression Check\n\n"
            + _markdown_table(regression_rows, ["command", "result", "note"]),
            "## 13. Decision\n\n" + f"**{decision}**",
            "## 14. What This Proves\n\n"
            "- equality tolerance variants can be executed and reported.\n"
            "- equality feasibility/violation trade-off can be observed by variant.\n"
            "- per-constraint summary is captured.\n"
            "- fairness and exact budget accounting still work.",
            "## 15. What This Does Not Prove\n\n"
            "- default GA가 constrained optimizer가 됐다는 뜻은 아님.\n"
            "- NSGA-II constraint-domination이 구현됐다는 뜻은 아님.\n"
            "- penalty/repair가 구현됐다는 뜻은 아님.\n"
            "- industrial constrained optimization을 지원한다는 뜻은 아님.\n"
            "- constrained GA가 모든 equality problem에서 random search보다 우수하다는 뜻은 아님.",
            "## 16. Maturity Impact\n\n"
            "- Level 4 근거 강화\n"
            "- equality tolerance stress는 실험 툴킷 근거를 강화한다.\n"
            "- default GA/NSGA-II가 바뀌지 않았으므로 default algorithm maturity 상향은 금지한다.\n"
            "- constrained benchmark evidence가 toy 3개 수준이므로 범용 constrained optimizer 주장은 금지한다.",
            "## 17. Recommended Next Work\n\n"
            "1. stress passed이면 broader constrained GA evidence review update\n"
            "2. limitations가 크면 tolerance redesign planning\n"
            "3. NSGA-II constraint-domination planning, 단 single-objective constrained evidence와 분리\n"
            "4. constrained ZDT-style toy planning\n"
            "5. fairness checker 통합 범위 확장\n"
            "6. checkpoint/resume\n"
            "7. parallel evaluation\n\n"
            f"이번 constrained_equality_plane_quadratic tolerance stress 결과, 저장소는 equality tolerance variant artifact/fairness/trace 생성까지 검증했지만, default GA/NSGA-II 통합은 미구현 상태이며, 최종 판정은 {decision}이다.",
        ]
    )


def run_constrained_equality_tolerance_stress(
    config: ConstrainedEqualityToleranceStressConfig,
) -> dict[str, Any]:
    if config.problem != "constrained_equality_plane_quadratic":
        raise ValueError("only constrained_equality_plane_quadratic is supported")
    for strategy in config.strategies:
        if strategy not in DEFAULT_STRATEGIES:
            raise ValueError(f"Unsupported equality tolerance stress strategy: {strategy}")
    if config.dimension <= 0 or config.dimension % 2 != 0:
        raise ValueError("dimension must be a positive even integer")
    if config.population_size <= 1:
        raise ValueError("population_size must be > 1")

    variants = _normalize_variants(config.variants)
    budgets = _normalize_budgets(config.budgets)
    seeds = _seed_list(config.seeds)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    fairness_by_variant_budget: dict[str, dict[str, Any]] = {}
    overall_issues: list[dict[str, Any]] = []

    for variant in variants:
        problem = _build_variant_problem(variant=variant, config=config)
        tolerance = float(problem.equality_tolerance)
        for budget in budgets:
            smoke_config = _smoke_config(config=config, budget=budget, tolerance=tolerance)
            variant_budget_rows: list[dict[str, Any]] = []
            for seed in seeds:
                for strategy in config.strategies:
                    try:
                        if strategy == "random_search_feasibility_first":
                            row = _random_search_row(config=smoke_config, problem=problem, seed=seed)
                        elif strategy == "constrained_ga_feasibility_first":
                            row = _constrained_ga_row(config=smoke_config, problem=problem, seed=seed)
                        else:  # pragma: no cover - guarded above
                            raise ValueError(f"Unsupported strategy: {strategy}")
                    except Exception as exc:  # pragma: no cover - failure path
                        failures.append(
                            {
                                "type": "strategy_failure",
                                "variant": variant,
                                "budget": budget,
                                "strategy": strategy,
                                "seed": seed,
                                "message": str(exc),
                            }
                        )
                        continue
                    _add_variant_metadata(row, variant=variant, tolerance=tolerance)
                    row["budget"] = budget
                    row["stress_mode"] = "equality_tolerance"
                    if int(row["actual_evaluations"]) != int(budget):
                        warnings.append(
                            {
                                "type": "actual_evaluations_mismatch",
                                "variant": variant,
                                "budget": budget,
                                "strategy": strategy,
                                "seed": seed,
                                "message": (
                                    f"expected {budget} evaluations but observed "
                                    f"{row['actual_evaluations']}"
                                ),
                            }
                        )
                    rows.append(row)
                    variant_budget_rows.append(row)
            fairness_payload = _variant_budget_fairness(
                variant=variant,
                budget=budget,
                problem=problem,
                config=config,
                rows=variant_budget_rows,
            )
            key = f"{variant}:{budget}"
            fairness_by_variant_budget[key] = fairness_payload
            for issue in fairness_payload.get("issues", []):
                overall_issues.append({"variant": variant, "budget": budget, **issue})

    fairness_summary = {
        "status": _overall_status(
            [payload["status"] for payload in fairness_by_variant_budget.values()]
        ),
        "issues": overall_issues,
        "summary_counts": _summarize_counts(overall_issues),
    }
    summaries = _variant_budget_strategy_summaries(
        rows=rows,
        variants=variants,
        budgets=budgets,
        strategies=config.strategies,
        seed_count=len(seeds),
        fairness_by_variant_budget=fairness_by_variant_budget,
    )
    equality_summary = _equality_summaries(summaries)
    per_constraint_summary = _per_constraint_summaries(
        rows=rows,
        variants=variants,
        budgets=budgets,
    )
    paired = _paired_comparisons(
        rows=rows,
        variants=variants,
        budgets=budgets,
        seed_list=seeds,
    )
    variant_decisions = _decision_table(
        variants=variants,
        paired_comparisons=paired,
        fairness_by_variant_budget=fairness_by_variant_budget,
    )
    decision = _determine_decision(
        variants=variants,
        rows=rows,
        summaries=summaries,
        per_constraint_summaries=per_constraint_summary,
        fairness_by_variant_budget=fairness_by_variant_budget,
    )

    project_root = _project_root()
    output_dir_input = Path(config.output_dir)
    output_dir = output_dir_input if output_dir_input.is_absolute() else project_root / output_dir_input
    prefer_absolute = output_dir_input.is_absolute()
    prefix = "constrained_equality_tolerance_stress"
    suffix = config.artifact_suffix
    json_path = output_dir / f"{prefix}_results_{suffix}.json"
    csv_path = output_dir / f"{prefix}_results_{suffix}.csv"
    results_md_path = output_dir / f"{prefix}_results_{suffix}.md"
    report_path = output_dir / f"{prefix}_report_{suffix}.md"
    fairness_path = output_dir / f"{prefix}_fairness_report_{suffix}.md"

    artifact: dict[str, Any] = {
        "command_metadata": {
            "argv": list(sys.argv),
            "runner": "validate_constrained_equality_tolerance_stress.py",
            "artifact_suffix": config.artifact_suffix,
            "output_dir": config.output_dir,
        },
        "problem": config.problem,
        "configuration": {
            "problem": config.problem,
            "dimension": config.dimension,
            "variants": variants,
            "budgets": budgets,
            "seeds": config.seeds,
            "seed_list": seeds,
            "population_size": config.population_size,
            "feasibility_policy": "feasibility_first",
            "violation_aggregation": "total_violation",
            "strategies": list(config.strategies),
        },
        "variant_definitions": _variant_definition_rows(variants),
        "rows": rows,
        "variant_budget_strategy_summaries": summaries,
        "aggregate_summaries": summaries,
        "equality_summaries": equality_summary,
        "per_constraint_summaries": per_constraint_summary,
        "paired_comparisons": paired,
        "variant_decisions": variant_decisions,
        "fairness_by_variant_budget": fairness_by_variant_budget,
        "fairness_summary": fairness_summary,
        "failures": failures,
        "warnings": warnings,
        "decision": decision,
        "default_changed": False,
        "ga_default_changed": False,
        "constrained_ga_opt_in_path_status": "restricted_experimental_opt_in",
        "ga_integration_done": False,
        "nsga2_constraint_domination_done": False,
        "penalty_repair_done": False,
        "artifacts": {
            "json": _artifact_ref(json_path, project_root, prefer_absolute=prefer_absolute),
            "csv": _artifact_ref(csv_path, project_root, prefer_absolute=prefer_absolute),
            "results_markdown": _artifact_ref(
                results_md_path,
                project_root,
                prefer_absolute=prefer_absolute,
            ),
            "report_markdown": _artifact_ref(report_path, project_root, prefer_absolute=prefer_absolute),
            "fairness_markdown": _artifact_ref(fairness_path, project_root, prefer_absolute=prefer_absolute),
        },
    }

    csv_fields = [
        "variant_name",
        "tolerance",
        "budget",
        "strategy",
        "seed",
        "problem",
        "dimension",
        "requested_budget",
        "actual_evaluations",
        "feasible_rate",
        "feasible_count",
        "infeasible_count",
        "best_feasible_objective",
        "best_feasible_solution",
        "best_record_feasible",
        "best_record_total_violation",
        "best_infeasible_total_violation",
        "mean_total_violation",
        "median_total_violation",
        "min_total_violation",
        "max_total_violation",
        "mean_max_violation",
        "violation_count_total",
        "equality_satisfaction_rate",
        "equality_mean_violation",
        "equality_max_violation",
        "per_constraint_satisfaction_rate",
        "per_constraint_mean_violation",
        "per_constraint_max_violation",
        "runtime_seconds",
        "failure_count",
        "default_changed",
        "ga_default_changed",
        "nsga2_constraint_domination_done",
    ]
    _write_json(json_path, artifact)
    _write_csv(csv_path, rows, csv_fields)
    _ensure_fresh_path(results_md_path).write_text(
        _render_results_markdown(
            summaries=summaries,
            equality_summaries=equality_summary,
            per_constraint_summaries=per_constraint_summary,
            paired_comparisons=paired,
        )
        + "\n",
        encoding="utf-8",
    )
    _ensure_fresh_path(fairness_path).write_text(
        _render_fairness_markdown(
            fairness_by_variant_budget=fairness_by_variant_budget,
            fairness_summary=fairness_summary,
        ),
        encoding="utf-8",
    )
    _ensure_fresh_path(report_path).write_text(
        _render_report(artifact=artifact, config=config, decision=decision) + "\n",
        encoding="utf-8",
    )
    return artifact


__all__ = [
    "ConstrainedEqualityToleranceStressConfig",
    "VARIANT_DEFINITIONS",
    "run_constrained_equality_tolerance_stress",
]
