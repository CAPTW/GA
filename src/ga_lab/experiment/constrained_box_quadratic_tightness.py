from __future__ import annotations

import json
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
    _stringify,
    _write_csv,
    _write_json,
)
from ga_lab.experiment.constrained_ga_stress import (
    COMPARISON_METRICS,
    _annotate_per_constraint_proxy,
    _comparison_outcome,
    _overall_status,
    _paired_interpretation,
    _safe_mean,
    _safe_median,
    _summarize_counts,
)
from ga_lab.problems.constrained_box_quadratic import ConstrainedBoxQuadraticProblem


VARIANT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "easy": {
        "budget_1": 3.0,
        "budget_2": 2.25,
        "expected_feasibility": "higher feasible_rate, lower total violation",
        "purpose": "check objective signal when constraints are looser",
    },
    "default": {
        "budget_1": 2.0,
        "budget_2": 1.5,
        "expected_feasibility": "mixed feasible profile",
        "purpose": "anchor against existing constrained_box_quadratic evidence",
    },
    "strict": {
        "budget_1": 1.2,
        "budget_2": 0.9,
        "expected_feasibility": "lower feasible_rate, higher violation pressure",
        "purpose": "check whether feasibility pressure increases objective trade-off",
    },
}


@dataclass(slots=True)
class ConstrainedBoxQuadraticTightnessConfig:
    variants: Sequence[str] = ("easy", "default", "strict")
    dimension: int = 6
    seeds: int | Sequence[int] = 20
    budget: int = 1000
    tolerance: float = 1e-8
    population_size: int = 20
    artifact_suffix: str = "tightness_stress1"
    output_dir: str = "artifacts"
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES


def _normalize_variants(variants: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in variants:
        variant = value.strip().lower().replace("-", "_")
        if variant not in VARIANT_DEFINITIONS:
            raise ValueError(f"Unsupported constrained_box_quadratic tightness variant: {value}")
        if variant not in seen:
            normalized.append(variant)
            seen.add(variant)
    if not normalized:
        raise ValueError("variants cannot be empty")
    return normalized


def _variant_definition_rows(variants: Sequence[str]) -> list[dict[str, Any]]:
    return [{"variant": variant, **VARIANT_DEFINITIONS[variant]} for variant in variants]


def _build_variant_problem(
    *,
    variant: str,
    config: ConstrainedBoxQuadraticTightnessConfig,
) -> ConstrainedBoxQuadraticProblem:
    definition = VARIANT_DEFINITIONS[variant]
    return ConstrainedBoxQuadraticProblem(
        dimension=config.dimension,
        group1_budget=float(definition["budget_1"]),
        group2_budget=float(definition["budget_2"]),
        equality_tolerance=config.tolerance,
    )


def _smoke_config(config: ConstrainedBoxQuadraticTightnessConfig) -> ConstrainedGASmokeConfig:
    return ConstrainedGASmokeConfig(
        problem="constrained_box_quadratic",
        dimension=config.dimension,
        seeds=1,
        budget=config.budget,
        tolerance=config.tolerance,
        population_size=config.population_size,
        artifact_suffix=config.artifact_suffix,
        output_dir=config.output_dir,
        strategies=config.strategies,
    )


def _add_variant_metadata(row: dict[str, Any], *, variant: str, problem) -> None:
    row["variant_name"] = variant
    row["budget_1"] = float(problem.group1_budget)
    row["budget_2"] = float(problem.group2_budget)
    row["group_budgets"] = [float(problem.group1_budget), float(problem.group2_budget)]
    _annotate_per_constraint_proxy(row)


def _append_variant_metadata_issues(
    issues: list[dict[str, Any]],
    *,
    rows: Sequence[dict[str, Any]],
    variant: str,
    budget_1: float,
    budget_2: float,
) -> None:
    for row in rows:
        row_name = str(row.get("strategy", "unknown_strategy"))
        checks = (
            ("variant_name", row.get("variant_name"), variant),
            ("budget_1", row.get("budget_1"), budget_1),
            ("budget_2", row.get("budget_2"), budget_2),
        )
        for field, observed, expected in checks:
            if observed == expected:
                issues.append(
                    {
                        "status": "pass",
                        "issue_type": f"{field}_match",
                        "severity": "info",
                        "message": f"{row_name}: {field} matches expected tightness variant metadata",
                        "recommended_action": "none",
                    }
                )
            else:
                issues.append(
                    {
                        "status": "fail",
                        "issue_type": f"{field}_mismatch",
                        "severity": "high",
                        "message": (
                            f"{row_name}: {field} mismatch observed={observed} "
                            f"expected={expected}"
                        ),
                        "recommended_action": "fix tightness variant metadata binding",
                    }
                )


def _variant_fairness(
    *,
    variant: str,
    problem,
    config: ConstrainedBoxQuadraticTightnessConfig,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    smoke_config = _smoke_config(config)
    contract = _constraint_contract(problem, smoke_config)
    contract["variant_name"] = variant
    contract["budget_1"] = float(problem.group1_budget)
    contract["budget_2"] = float(problem.group2_budget)
    payload = evaluate_constrained_fairness(
        expected_contract=contract,
        observed_rows=rows,
    )
    issues = list(payload.get("issues", []))
    _append_variant_metadata_issues(
        issues,
        rows=rows,
        variant=variant,
        budget_1=float(problem.group1_budget),
        budget_2=float(problem.group2_budget),
    )
    return {
        "status": _overall_status([str(issue.get("status")) for issue in issues]),
        "issues": issues,
        "summary_counts": _summarize_counts(issues),
        "variant_name": variant,
        "budget_1": float(problem.group1_budget),
        "budget_2": float(problem.group2_budget),
    }


def _variant_per_constraint_summaries(
    *,
    rows: Sequence[dict[str, Any]],
    variants: Sequence[str],
    strategies: Sequence[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant_name"] == variant]
        for item in _aggregate_per_constraint_summaries(variant_rows):
            summaries.append({"variant": variant, **item})

        for strategy in strategies:
            strategy_rows = [row for row in variant_rows if row["strategy"] == strategy]
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
                        "variant": variant,
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


def _variant_strategy_summaries(
    *,
    rows: Sequence[dict[str, Any]],
    variants: Sequence[str],
    strategies: Sequence[str],
    seed_count: int,
    fairness_by_variant: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant_name"] == variant]
        fairness_status = fairness_by_variant[variant]["status"]
        for strategy in strategies:
            strategy_rows = [row for row in variant_rows if row["strategy"] == strategy]
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
                    "variant_name": variant,
                    "budget_1": variant_rows[0].get("budget_1") if variant_rows else None,
                    "budget_2": variant_rows[0].get("budget_2") if variant_rows else None,
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
                    "all_feasible_runs": sum(
                        1 for row in strategy_rows if bool(row.get("all_feasible"))
                    ),
                }
            )
    return summaries


def _paired_comparisons(
    *,
    rows: Sequence[dict[str, Any]],
    variants: Sequence[str],
    seed_list: Sequence[int],
    budget: int,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant_name"] == variant]
        by_strategy_seed = {
            (str(row["strategy"]), int(row["seed"])): row
            for row in variant_rows
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
    for strategy in DEFAULT_STRATEGIES:
        strategy_rows = [row for row in summaries if row["strategy"] == strategy]
        feasible_rates = [
            float(row["mean_feasible_rate"])
            for row in strategy_rows
            if row.get("mean_feasible_rate") is not None
        ]
        violations = [
            float(row["mean_total_violation_mean"])
            for row in strategy_rows
            if row.get("mean_total_violation_mean") is not None
        ]
        if feasible_rates and max(feasible_rates) - min(feasible_rates) >= 0.05:
            return True
        if violations and max(violations) - min(violations) >= 0.05:
            return True
    return False


def _variant_decisions(
    *,
    variants: Sequence[str],
    fairness_by_variant: dict[str, dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for variant in variants:
        pairs = [row for row in paired_comparisons if row["variant"] == variant]
        feasibility_pair = next(row for row in pairs if row["metric"] == "feasible_rate")
        objective_pair = next(row for row in pairs if row["metric"] == "best_feasible_objective")
        violation_pair = next(row for row in pairs if row["metric"] == "mean_total_violation")
        runtime_pair = next(row for row in pairs if row["metric"] == "runtime_seconds")
        if fairness_by_variant[variant]["status"] == "fail":
            contribution = "needs_fairness_rerun"
        elif (
            feasibility_pair["constrained_ga_win"] > feasibility_pair["random_search_win"]
            or violation_pair["constrained_ga_win"] > violation_pair["random_search_win"]
        ):
            contribution = "positive_with_tradeoff"
        else:
            contribution = "limited_or_mixed"
        decisions.append(
            {
                "variant": variant,
                "stress_result": contribution,
                "fairness": fairness_by_variant[variant]["status"],
                "constrained_ga_vs_random_search": (
                    f"feasible_rate {feasibility_pair['constrained_ga_win']}:"
                    f"{feasibility_pair['random_search_win']}, "
                    f"mean_total_violation {violation_pair['constrained_ga_win']}:"
                    f"{violation_pair['random_search_win']}"
                ),
                "objective_trade_off": (
                    f"best_feasible_objective constrained_ga/random_search "
                    f"{objective_pair['constrained_ga_win']}:{objective_pair['random_search_win']}"
                ),
                "runtime_trade_off": (
                    f"runtime constrained_ga/random_search "
                    f"{runtime_pair['constrained_ga_win']}:{runtime_pair['random_search_win']}"
                ),
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
    fairness_by_variant: dict[str, dict[str, Any]],
    budget: int,
) -> str:
    if any(payload["status"] == "fail" for payload in fairness_by_variant.values()):
        return "Needs fairness rerun"
    if any(int(row["actual_evaluations"]) != int(budget) for row in rows):
        return "Needs fairness rerun"
    if not per_constraint_summaries:
        return "Fix required"
    if sorted({row["variant_name"] for row in rows}) != sorted(variants):
        return "Fix required"
    if not _profile_is_distinguishable(summaries):
        return "Tightness stress passed with limitations"
    return "Tightness stress passed, trade-off characterized"


def _render_results_markdown(
    *,
    summaries: Sequence[dict[str, Any]],
    paired_comparisons: Sequence[dict[str, Any]],
    per_constraint_summaries: Sequence[dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            "# Constrained Box Quadratic Tightness Results",
            "## Variant Strategy Summary",
            _markdown_table(
                summaries,
                [
                    "variant_name",
                    "strategy",
                    "mean_feasible_rate",
                    "mean_best_feasible_objective",
                    "mean_total_violation_mean",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                    "fairness_status",
                ],
            ),
            "## Per-Constraint Summary",
            _markdown_table(
                per_constraint_summaries,
                [
                    "variant",
                    "strategy",
                    "constraint_name",
                    "satisfaction_rate",
                    "mean_violation",
                    "max_violation",
                    "limitation",
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
    fairness_by_variant: dict[str, dict[str, Any]],
    fairness_summary: dict[str, Any],
) -> str:
    lines = [
        "# Constrained Box Quadratic Tightness Fairness Report",
        "",
        f"- Overall status: **{fairness_summary['status']}**",
        f"- Overall summary counts: `{fairness_summary.get('summary_counts', {})}`",
    ]
    for variant, payload in fairness_by_variant.items():
        lines.extend(
            [
                "",
                f"## Variant {variant}",
                "",
                f"- status: `{payload['status']}`",
                f"- summary counts: `{payload.get('summary_counts', {})}`",
                "",
                _markdown_table(
                    _build_fairness_rows(payload),
                    ["item", "status", "message"],
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _render_report(
    *,
    artifact: dict[str, Any],
    config: ConstrainedBoxQuadraticTightnessConfig,
    decision: str,
) -> str:
    summaries = artifact["variant_strategy_summaries"]
    per_constraint = artifact["per_constraint_summaries"]
    paired = artifact["paired_comparisons"]
    fairness_counts = artifact["fairness_summary"].get("summary_counts", {})
    fairness_rows = [
        {
            "item": f"{variant}:fairness_summary",
            "status": payload["status"],
            "message": f"summary_counts={payload.get('summary_counts', {})}",
        }
        for variant, payload in artifact["fairness_by_variant"].items()
    ]
    return "\n\n".join(
        [
            "# Constrained Box Quadratic Tightness Stress Report",
            "## 1. Executive Summary\n\n"
            f"- 이번 작업의 목표: `constrained_box_quadratic` easy/default/strict constraint tightness가 feasibility, violation, best objective, runtime에 미치는 영향을 확인한다.\n"
            f"- 실행한 variants: `{', '.join(artifact['configuration']['variants'])}`.\n"
            "- random search baseline 결과와 constrained GA 결과는 variant별 paired comparison으로 분리해 기록한다.\n"
            "- feasibility/violation 결과와 best objective trade-off는 performance superiority claim 없이 해석한다.\n"
            f"- fairness 결과는 `{artifact['fairness_summary']['status']}`이고 summary counts는 `{fairness_counts}`이다.\n"
            f"- tightness stress decision은 `{decision}`이다.\n"
            "- 기본값 변경 없음, GA/NSGA-II default 영향 없음.\n"
            "- Level 판정은 유지하되, 실험 툴킷 관점의 Level 4 근거는 강화한다.",
            "## 2. Scope and Non-Scope\n\n"
            "Scope:\n\n"
            "- constrained_box_quadratic\n"
            "- easy/default/strict variants\n"
            "- random_search_feasibility_first\n"
            "- constrained_ga_feasibility_first\n"
            "- feasibility-first comparator\n"
            "- per-constraint summary\n"
            "- constrained fairness\n"
            "- tightness stress artifact generation\n\n"
            "Non-Scope:\n\n"
            "- default GA replacement\n"
            "- NSGA-II constraint-domination\n"
            "- penalty/repair\n"
            "- constrained MOEA benchmark\n"
            "- productization",
            "## 3. Variant Definitions\n\n"
            + _markdown_table(
                artifact["variant_definitions"],
                ["variant", "budget_1", "budget_2", "expected_feasibility", "purpose"],
            ),
            "## 4. Stress Configuration\n\n"
            + _markdown_table(
                [
                    {"field": "problem", "value": "constrained_box_quadratic"},
                    {"field": "dimension", "value": config.dimension},
                    {"field": "variants", "value": list(artifact["configuration"]["variants"])},
                    {"field": "budget", "value": config.budget},
                    {"field": "seeds", "value": config.seeds},
                    {"field": "strategies", "value": list(config.strategies)},
                    {"field": "tolerance", "value": config.tolerance},
                    {"field": "feasibility policy", "value": "feasibility_first"},
                ],
                ["field", "value"],
            ),
            "## 5. Results Summary\n\n"
            + _markdown_table(
                summaries,
                [
                    "variant_name",
                    "strategy",
                    "mean_feasible_rate",
                    "mean_best_feasible_objective",
                    "mean_total_violation_mean",
                    "mean_actual_evaluations",
                    "mean_runtime_seconds",
                ],
            ),
            "## 6. Per-Constraint Summary\n\n"
            + _markdown_table(
                per_constraint,
                [
                    "variant",
                    "strategy",
                    "constraint_name",
                    "satisfaction_rate",
                    "mean_violation",
                    "max_violation",
                ],
            ),
            "## 7. Paired Comparison\n\n"
            + _markdown_table(
                paired,
                [
                    "variant",
                    "metric",
                    "constrained_ga_win",
                    "tie",
                    "random_search_win",
                    "mean_delta",
                    "interpretation",
                ],
            ),
            "## 8. Tightness Interpretation\n\n"
            "- easy/default/strict variants are interpreted by feasible_rate and violation profile differences.\n"
            "- objective trade-off is reported per variant and must not be hidden when mixed.\n"
            "- this run uses budget 1000, matching the follow-up phase from the plan; budget 300 interpretation remains in prior stress artifacts.\n"
            "- group1/group2 bottleneck is read from per-constraint trace summaries where available, with legacy limitation rows marked if trace is absent.\n"
            "- all-feasible or all-infeasible outcomes, if present, are valid stress outcomes and not automatic failures.",
            "## 9. Fairness Summary\n\n" + _markdown_table(fairness_rows, ["item", "status", "message"]),
            "## 10. Failures and Warnings\n\n"
            + _markdown_table(
                _build_failures_and_warnings_rows(
                    artifact.get("failures", []),
                    artifact.get("warnings", []),
                ),
                ["type", "message", "action"],
            ),
            "## 11. Regression Check\n\n"
            "| command | result | note |\n"
            "|---|---|---|\n"
            "| python -m pytest tests/test_constrained_box_quadratic.py tests/test_constrained_box_quadratic_smoke_runner.py tests/test_constrained_box_quadratic_tightness_runner.py -q | pending | filled after command execution |\n"
            "| python -m json.tool artifacts/hypotheses/constrained_optimization_contract_backlog.json | pending | filled after command execution |\n"
            "| python scripts/check_local_baseline.py --output-dir artifacts/constrained_box_quadratic_tightness_guard | pending | filled after command execution |",
            "## 12. Decision\n\n" + f"**{decision}**",
            "## 13. What This Proves\n\n"
            "- constraint tightness variants can be executed and reported.\n"
            "- feasibility/objective trade-off can be observed by variant.\n"
            "- per-constraint summary is captured from ConstraintEvaluation trace where available, with legacy raw-trace limitations explicitly marked where present.\n"
            "- fairness and exact budget accounting still work.",
            "## 14. What This Does Not Prove\n\n"
            "- default GA가 constrained optimizer가 됐다는 뜻은 아님.\n"
            "- NSGA-II constraint-domination이 구현됐다는 뜻은 아님.\n"
            "- penalty/repair가 구현됐다는 뜻은 아님.\n"
            "- industrial constrained optimization을 지원한다는 뜻은 아님.\n"
            "- constrained GA가 모든 constrained problem에서 random search보다 우수하다는 뜻은 아님.",
            "## 15. Maturity Impact\n\n"
            "- Level 4 근거 강화\n"
            "- tightness stress는 실험 툴킷 근거를 강화한다.\n"
            "- default GA/NSGA-II가 바뀌지 않았으므로 default algorithm maturity 상향은 금지한다.\n"
            "- constrained benchmark evidence가 toy 2개 수준이므로 범용 constrained optimizer 주장은 금지한다.",
            "## 16. Recommended Next Work\n\n"
            "1. tightness stress passed이면 broader constrained GA evidence review update\n"
            "2. limitations가 크면 variant redesign planning\n"
            "3. equality-constrained toy planning\n"
            "4. NSGA-II constraint-domination planning, 단 single-objective constrained evidence와 분리\n"
            "5. constrained ZDT-style toy planning\n"
            "6. fairness checker 통합 범위 확장\n"
            "7. checkpoint/resume\n"
            "8. parallel evaluation\n\n"
            f"이번 constrained_box_quadratic tightness stress 결과, 저장소는 tightness variant artifact/fairness/metric contract까지 검증했지만, default GA/NSGA-II 통합은 미구현 상태이며, 최종 판정은 {decision}이다.",
        ]
    )


def run_constrained_box_quadratic_tightness_stress(
    config: ConstrainedBoxQuadraticTightnessConfig,
) -> dict[str, Any]:
    variants = _normalize_variants(config.variants)
    seeds = _seed_list(config.seeds)
    if config.budget <= 0:
        raise ValueError("budget must be > 0")
    unsupported = [strategy for strategy in config.strategies if strategy not in DEFAULT_STRATEGIES]
    if unsupported:
        raise ValueError(f"Unsupported constrained tightness strategy: {unsupported}")

    smoke_config = _smoke_config(config)
    rows: list[dict[str, Any]] = []
    fairness_by_variant: dict[str, dict[str, Any]] = {}
    overall_issues: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    variant_problems: dict[str, ConstrainedBoxQuadraticProblem] = {}

    for variant in variants:
        problem = _build_variant_problem(variant=variant, config=config)
        variant_problems[variant] = problem
        variant_rows: list[dict[str, Any]] = []
        for seed in seeds:
            for strategy in config.strategies:
                if strategy == "random_search_feasibility_first":
                    row = _random_search_row(config=smoke_config, problem=problem, seed=seed)
                elif strategy == "constrained_ga_feasibility_first":
                    row = _constrained_ga_row(config=smoke_config, problem=problem, seed=seed)
                else:  # pragma: no cover - guarded above
                    raise ValueError(f"Unsupported constrained tightness strategy: {strategy}")
                _add_variant_metadata(row, variant=variant, problem=problem)
                rows.append(row)
                variant_rows.append(row)
        fairness_payload = _variant_fairness(
            variant=variant,
            problem=problem,
            config=config,
            rows=variant_rows,
        )
        fairness_by_variant[variant] = fairness_payload
        for issue in fairness_payload.get("issues", []):
            overall_issues.append({"variant": variant, **issue})

    fairness_summary = {
        "status": _overall_status([payload["status"] for payload in fairness_by_variant.values()]),
        "issues": overall_issues,
        "summary_counts": _summarize_counts(overall_issues),
    }
    variant_summaries = _variant_strategy_summaries(
        rows=rows,
        variants=variants,
        strategies=config.strategies,
        seed_count=len(seeds),
        fairness_by_variant=fairness_by_variant,
    )
    per_constraint_summaries = _variant_per_constraint_summaries(
        rows=rows,
        variants=variants,
        strategies=config.strategies,
    )
    paired_comparisons = _paired_comparisons(
        rows=rows,
        variants=variants,
        seed_list=seeds,
        budget=config.budget,
    )
    variant_decisions = _variant_decisions(
        variants=variants,
        fairness_by_variant=fairness_by_variant,
        paired_comparisons=paired_comparisons,
    )
    decision = _determine_decision(
        variants=variants,
        rows=rows,
        summaries=variant_summaries,
        per_constraint_summaries=per_constraint_summaries,
        fairness_by_variant=fairness_by_variant,
        budget=config.budget,
    )

    project_root = _project_root()
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    suffix = config.artifact_suffix
    prefix = "constrained_box_quadratic_tightness"
    json_path = output_dir / f"{prefix}_results_{suffix}.json"
    csv_path = output_dir / f"{prefix}_results_{suffix}.csv"
    results_md_path = output_dir / f"{prefix}_results_{suffix}.md"
    report_path = output_dir / f"{prefix}_report_{suffix}.md"
    fairness_path = output_dir / f"{prefix}_fairness_report_{suffix}.md"
    prefer_absolute = output_dir.resolve() != (project_root / "artifacts").resolve()

    artifact: dict[str, Any] = {
        "command_metadata": {
            "argv": list(sys.argv),
            "runner": "validate_constrained_box_quadratic_tightness_stress.py",
        },
        "problem": "constrained_box_quadratic",
        "configuration": {
            "variants": variants,
            "dimension": config.dimension,
            "seeds": seeds,
            "budget": config.budget,
            "tolerance": config.tolerance,
            "population_size": config.population_size,
            "strategies": list(config.strategies),
            "feasibility_policy": "feasibility_first",
            "violation_aggregation": "total_violation",
        },
        "variant_definitions": _variant_definition_rows(variants),
        "rows": rows,
        "variant_strategy_summaries": variant_summaries,
        "per_constraint_summaries": per_constraint_summaries,
        "paired_comparisons": paired_comparisons,
        "variant_decisions": variant_decisions,
        "fairness_by_variant": fairness_by_variant,
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
            "results_markdown": _artifact_ref(results_md_path, project_root, prefer_absolute=prefer_absolute),
            "report_markdown": _artifact_ref(report_path, project_root, prefer_absolute=prefer_absolute),
            "fairness_markdown": _artifact_ref(fairness_path, project_root, prefer_absolute=prefer_absolute),
        },
    }

    row_fieldnames = [
        "variant_name",
        "budget_1",
        "budget_2",
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
        "per_constraint_satisfaction_rate",
        "per_constraint_mean_violation",
        "per_constraint_summary_scope",
        "per_constraint_trace_summary",
        "runtime_seconds",
        "failure_count",
        "default_changed",
        "ga_default_changed",
        "nsga2_constraint_domination_done",
    ]
    _write_json(json_path, artifact)
    _write_csv(csv_path, rows, row_fieldnames)
    _ensure_fresh_path(results_md_path).write_text(
        _render_results_markdown(
            summaries=variant_summaries,
            paired_comparisons=paired_comparisons,
            per_constraint_summaries=per_constraint_summaries,
        )
        + "\n",
        encoding="utf-8",
    )
    _ensure_fresh_path(fairness_path).write_text(
        _render_fairness_markdown(
            fairness_by_variant=fairness_by_variant,
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
    "ConstrainedBoxQuadraticTightnessConfig",
    "VARIANT_DEFINITIONS",
    "run_constrained_box_quadratic_tightness_stress",
]
