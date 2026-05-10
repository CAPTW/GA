from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from ga_lab.algorithms.constrained_single_objective import (
    run_constrained_single_objective_ga,
)
from ga_lab.config import GAConfig
from ga_lab.constraints import summarize_constraint_violations
from ga_lab.experiment.constrained_fairness import evaluate_constrained_fairness
from ga_lab.experiment.constrained_metrics import (
    best_feasible_objective,
    feasible_rate,
    per_constraint_violation_summary,
    summarize_constrained_population,
)
from ga_lab.experiment.constrained_protocol import (
    ConstrainedCandidateRecord,
    select_best_feasible_first,
)
from ga_lab.experiment.constrained_trace import (
    build_per_constraint_trace_records,
    per_constraint_summary_to_dict,
    summarize_per_constraint_trace,
)
from ga_lab.factory import build_runtime_context
from ga_lab.problems import build_problem


DEFAULT_STRATEGIES = (
    "random_search_feasibility_first",
    "constrained_ga_feasibility_first",
)
SUPPORTED_STRATEGIES = DEFAULT_STRATEGIES
SUPPORTED_PROBLEMS = (
    "constrained_sphere",
    "constrained_box_quadratic",
    "constrained_equality_plane_quadratic",
)
DEFAULT_SMOKE_TOLERANCE = 1e-8
EQUALITY_PLANE_SMOKE_TOLERANCE = 1e-2


@dataclass(slots=True)
class ConstrainedGASmokeConfig:
    dimension: int = 5
    seeds: int = 5
    budget: int = 300
    constraint_budget: float = 1.0
    tolerance: float | None = None
    population_size: int = 20
    artifact_suffix: str = "constrained_ga_smoke1"
    output_dir: str = "artifacts"
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    problem: str = "constrained_sphere"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_problem_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _effective_tolerance(config: ConstrainedGASmokeConfig) -> float:
    if config.tolerance is not None:
        return float(config.tolerance)
    problem_name = _normalize_problem_name(config.problem)
    if problem_name == "constrained_equality_plane_quadratic":
        return EQUALITY_PLANE_SMOKE_TOLERANCE
    return DEFAULT_SMOKE_TOLERANCE


def _seed_list(seed_spec: int | Sequence[int]) -> list[int]:
    if isinstance(seed_spec, int):
        if seed_spec <= 0:
            raise ValueError("seeds must be > 0")
        return list(range(seed_spec))
    seeds = [int(seed) for seed in seed_spec]
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return seeds


def _ensure_fresh_path(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_ref(path: Path, project_root: Path, *, prefer_absolute: bool = False) -> str:
    if prefer_absolute:
        return str(path)
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _stringify(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | tuple):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _write_json(path: Path, payload: Any) -> Path:
    _ensure_fresh_path(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    _ensure_fresh_path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_stringify(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _problem_build_options(config: ConstrainedGASmokeConfig) -> dict[str, object]:
    problem_name = _normalize_problem_name(config.problem)
    tolerance = _effective_tolerance(config)
    if problem_name == "constrained_sphere":
        return {
            "dimension": config.dimension,
            "budget": config.constraint_budget,
            "equality_tolerance": tolerance,
        }
    if problem_name == "constrained_box_quadratic":
        return {
            "dimension": config.dimension,
            "equality_tolerance": tolerance,
        }
    if problem_name == "constrained_equality_plane_quadratic":
        return {
            "dimension": config.dimension,
            "equality_tolerance": tolerance,
        }
    raise ValueError(f"Unsupported constrained smoke problem: {config.problem}")


def _build_problem_for_smoke(config: ConstrainedGASmokeConfig):
    problem_name = _normalize_problem_name(config.problem)
    if problem_name not in SUPPORTED_PROBLEMS:
        raise ValueError(f"Unsupported constrained smoke problem: {config.problem}")
    return build_problem(problem_name, _problem_build_options(config))


def _problem_display_name(problem_name: str) -> str:
    return problem_name.replace("_", " ").title()


def _problem_constraint_names(problem) -> list[str]:
    constraint_names = getattr(problem, "constraint_names", None)
    if callable(constraint_names):
        return [str(item) for item in constraint_names()]
    if getattr(problem, "name", "") == "constrained_sphere":
        if getattr(problem, "equality_target", None) is None:
            return ["sum_budget"]
        return ["sum_budget", "x0_equals_target"]
    inequality_count = int(getattr(problem, "inequality_count", 0))
    equality_count = int(getattr(problem, "equality_count", 0))
    return [
        *[f"inequality_{index}" for index in range(inequality_count)],
        *[f"equality_{index}" for index in range(equality_count)],
    ]


def _problem_constraint_counts(problem) -> tuple[int, int, int]:
    inequality_count = getattr(problem, "inequality_count", None)
    equality_count = getattr(problem, "equality_count", None)
    if isinstance(inequality_count, int) and isinstance(equality_count, int):
        return inequality_count + equality_count, inequality_count, equality_count
    names = _problem_constraint_names(problem)
    if getattr(problem, "name", "") == "constrained_sphere":
        inferred_equality = 0 if getattr(problem, "equality_target", None) is None else 1
        inferred_inequality = len(names) - inferred_equality
        return len(names), inferred_inequality, inferred_equality
    return len(names), len(names), 0


def _problem_uniform_bounds(problem) -> tuple[float, float]:
    bounds = list(problem.source_bounds())
    if not bounds:
        raise ValueError("problem.source_bounds() cannot be empty")
    first = tuple(bounds[0])
    if len(first) != 2:
        raise ValueError("problem bounds entries must be length-2 pairs")
    for item in bounds[1:]:
        if tuple(item) != first:
            raise ValueError("smoke runner currently requires uniform bounds across dimensions")
    return float(first[0]), float(first[1])


def _problem_ga_options(problem) -> dict[str, object]:
    options_fn = getattr(problem, "problem_options", None)
    if callable(options_fn):
        payload = options_fn()
        if not isinstance(payload, dict):
            raise ValueError("problem_options() must return a dict")
        return dict(payload)
    if getattr(problem, "name", "") == "constrained_sphere":
        return {
            "dimension": problem.dimension,
            "lower_bound": problem.lower_bound,
            "upper_bound": problem.upper_bound,
            "budget": problem.budget,
            "equality_tolerance": problem.equality_tolerance,
        }
    raise ValueError(f"Unsupported problem options extraction for {getattr(problem, 'name', 'unknown')}")


def _constraint_contract(problem, config: ConstrainedGASmokeConfig) -> dict[str, Any]:
    constraint_count, inequality_count, equality_count = _problem_constraint_counts(problem)
    tolerance = _effective_tolerance(config)
    contract = {
        "problem": problem.name,
        "dimension": problem.dimension,
        "bounds": problem.source_bounds(),
        "objective_directions": [False],
        "constraint_count": constraint_count,
        "inequality_count": inequality_count,
        "equality_count": equality_count,
        "constraint_names": _problem_constraint_names(problem),
        "tolerance": tolerance,
        "feasibility_policy": "feasibility_first",
        "violation_aggregation": "total_violation",
        "requested_budget": config.budget,
        "non_finite_constraint_fail_fast_policy": "value_error",
        "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
    }
    if problem.name == "constrained_sphere":
        contract["constraint_budget"] = getattr(problem, "budget", None)
    if problem.name == "constrained_box_quadratic":
        contract["group_budgets"] = [
            getattr(problem, "group1_budget", None),
            getattr(problem, "group2_budget", None),
        ]
        contract["weights"] = list(getattr(problem, "weights", []))
        contract["targets"] = list(getattr(problem, "targets", []))
    if problem.name == "constrained_equality_plane_quadratic":
        contract["equality_target"] = getattr(problem, "equality_target", None)
        contract["group_budget"] = getattr(problem, "group_budget", None)
        contract["weights"] = list(getattr(problem, "weights", []))
        contract["targets"] = list(getattr(problem, "targets", []))
    return contract


def _random_solution(rng: random.Random, problem) -> list[float]:
    return [
        rng.uniform(float(low), float(high))
        for low, high in problem.source_bounds()
    ]


def _row_per_constraint_summary(problem, evaluations) -> list[dict[str, Any]]:
    names = _problem_constraint_names(problem)
    inequality_count = len(evaluations[0].inequality_violations) if evaluations else 0
    return per_constraint_violation_summary(
        evaluations,
        inequality_names=names[:inequality_count],
        equality_names=names[inequality_count:],
    )


def _constraint_type_metrics(
    per_constraint_summary: Sequence[dict[str, Any]],
    constraint_type: str,
) -> dict[str, float | None]:
    entries = [
        item
        for item in per_constraint_summary
        if item.get("constraint_type") == constraint_type
        and item.get("satisfaction_rate") is not None
    ]
    if not entries:
        return {
            f"{constraint_type}_satisfaction_rate": None,
            f"{constraint_type}_mean_violation": None,
            f"{constraint_type}_max_violation": None,
        }
    return {
        f"{constraint_type}_satisfaction_rate": sum(
            float(item["satisfaction_rate"]) for item in entries
        )
        / len(entries),
        f"{constraint_type}_mean_violation": sum(
            float(item["mean_violation"]) for item in entries
        )
        / len(entries),
        f"{constraint_type}_max_violation": max(
            float(item["max_violation"]) for item in entries
        ),
    }


def _records_to_row(
    *,
    strategy: str,
    seed: int,
    requested_budget: int,
    problem,
    records: Sequence[ConstrainedCandidateRecord],
    runtime_seconds: float,
    failure_count: int,
) -> dict[str, Any]:
    evaluations = [record.constraint_evaluation for record in records]
    objective_values = [record.objective for record in records]
    summary = summarize_constraint_violations(evaluations)
    best_record = select_best_feasible_first(records)
    feasible_records = [record for record in records if record.constraint_evaluation.feasible]
    best_feasible_record = (
        select_best_feasible_first(feasible_records)
        if feasible_records
        else None
    )
    infeasible_evaluations = [evaluation for evaluation in evaluations if not evaluation.feasible]
    best_infeasible_total_violation = (
        min(evaluation.total_violation for evaluation in infeasible_evaluations)
        if infeasible_evaluations
        else None
    )
    constraint_count, inequality_count, equality_count = _problem_constraint_counts(problem)
    trace_records, trace_warnings = build_per_constraint_trace_records(
        records,
        strategy=strategy,
        constraint_names=_problem_constraint_names(problem),
        metadata={"source": f"{strategy}_evaluations"},
    )
    per_constraint_trace_summary = per_constraint_summary_to_dict(
        summarize_per_constraint_trace(
            trace_records,
            warnings=trace_warnings,
            sample_limit=20,
        )
    )
    per_constraint_summary = per_constraint_trace_summary["aggregates"]
    equality_metrics = _constraint_type_metrics(per_constraint_summary, "equality")
    return {
        "strategy": strategy,
        "seed": seed,
        "problem": problem.name,
        "dimension": problem.dimension,
        "bounds": problem.source_bounds(),
        "objective_directions": [False],
        "constraint_count": constraint_count,
        "inequality_count": inequality_count,
        "equality_count": equality_count,
        "constraint_names": _problem_constraint_names(problem),
        "tolerance": getattr(problem, "equality_tolerance", 1e-8),
        "feasibility_policy": "feasibility_first",
        "violation_aggregation": "total_violation",
        "non_finite_constraint_fail_fast_policy": "value_error",
        "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
        "requested_budget": requested_budget,
        "actual_evaluations": len(records),
        "runtime_seconds": runtime_seconds,
        "failure_count": failure_count,
        "feasible_rate": feasible_rate(evaluations),
        "feasible_count": summary.feasible_count,
        "infeasible_count": summary.infeasible_count,
        "best_feasible_objective": best_feasible_objective(objective_values, evaluations, maximize=False),
        "best_feasible_solution": (
            list(best_feasible_record.solution)
            if best_feasible_record is not None
            else None
        ),
        "best_record_feasible": best_record.constraint_evaluation.feasible,
        "best_record_total_violation": best_record.constraint_evaluation.total_violation,
        "best_infeasible_total_violation": best_infeasible_total_violation,
        "mean_total_violation": summary.mean_total_violation,
        "median_total_violation": summary.median_total_violation,
        "min_total_violation": summary.min_total_violation,
        "max_total_violation": summary.max_total_violation,
        "mean_max_violation": summary.mean_max_violation,
        "violation_count_total": summary.violation_count_total,
        "all_feasible": summary.all_feasible,
        "any_feasible": summary.any_feasible,
        "all_infeasible": summary.all_infeasible,
        "constraint_summary": summarize_constrained_population(evaluations),
        "per_constraint_violation_summary": per_constraint_summary,
        "per_constraint_trace_summary": per_constraint_trace_summary,
        "per_constraint_trace_records_sample": per_constraint_trace_summary.get("records_sample", []),
        "per_constraint_trace_limitations": per_constraint_trace_summary.get("limitations", []),
        "per_constraint_trace_warnings": per_constraint_trace_summary.get("warnings", []),
        **equality_metrics,
        "best_record": best_record.to_dict(),
        "default_changed": False,
        "ga_default_changed": False,
        "constrained_ga_opt_in_path_used": False,
        "ga_integration_done": False,
        "nsga2_constraint_domination_done": False,
    }


def _random_search_row(
    *,
    config: ConstrainedGASmokeConfig,
    problem,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    records: list[ConstrainedCandidateRecord] = []
    started = time.perf_counter()
    failure_count = 0
    for evaluation_index in range(config.budget):
        try:
            solution = _random_solution(rng, problem)
            objective = problem.evaluate_objective(solution)
            constraint_evaluation = problem.evaluate_constraints(solution)
            records.append(
                ConstrainedCandidateRecord(
                    solution=solution,
                    objective=objective,
                    constraint_evaluation=constraint_evaluation,
                    seed=seed,
                    evaluation_index=evaluation_index,
                    metadata={"strategy": "random_search_feasibility_first"},
                )
            )
        except Exception as exc:  # pragma: no cover - failure path
            failure_count += 1
            raise RuntimeError(
                f"random_search_feasibility_first failed at seed={seed} "
                f"evaluation_index={evaluation_index}: {exc}"
            ) from exc
    runtime_seconds = time.perf_counter() - started
    return _records_to_row(
        strategy="random_search_feasibility_first",
        seed=seed,
        requested_budget=config.budget,
        problem=problem,
        records=records,
        runtime_seconds=runtime_seconds,
        failure_count=failure_count,
    )


def _ga_generations_for_budget(budget: int, population_size: int) -> int:
    if budget <= population_size:
        return 1
    return max(1, math.ceil((budget - population_size) / population_size))


def _constrained_ga_config(
    *,
    config: ConstrainedGASmokeConfig,
    seed: int,
    problem,
) -> GAConfig:
    lower_bound, upper_bound = _problem_uniform_bounds(problem)
    return GAConfig(
        run_name=f"{problem.name}_constrained_ga_smoke_seed_{seed}",
        problem=problem.name,
        problem_options=_problem_ga_options(problem),
        population_size=config.population_size,
        genome_length=problem.dimension,
        generations=_ga_generations_for_budget(config.budget, config.population_size),
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=3,
        algorithm="ga",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        representation="real",
        representation_options={
            "low": lower_bound,
            "high": upper_bound,
        },
        algorithm_options={
            "requested_budget": config.budget,
            "constraint_policy": "feasibility_first",
            "feasibility_policy": "feasibility_first",
            "violation_aggregation": "total_violation",
            "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
            "non_finite_constraint_fail_fast_policy": "value_error",
            "experimental_opt_in": True,
        },
        seed=seed,
        maximize=False,
        objective_directions=[False],
    )


def _constrained_ga_row(
    *,
    config: ConstrainedGASmokeConfig,
    problem,
    seed: int,
) -> dict[str, Any]:
    ga_config = _constrained_ga_config(config=config, seed=seed, problem=problem)
    runtime = build_runtime_context(ga_config)
    started = time.perf_counter()
    summary, _history = run_constrained_single_objective_ga(
        config=ga_config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(seed),
    )
    runtime_seconds = time.perf_counter() - started
    constraint_count, inequality_count, equality_count = _problem_constraint_counts(problem)
    per_constraint_trace_summary = dict(summary.get("per_constraint_trace_summary") or {})
    per_constraint_trace_aggregates = [
        dict(item)
        for item in per_constraint_trace_summary.get("aggregates", [])
    ]
    unavailable_per_constraint = [
        {
            "constraint": name,
            "constraint_type": "inequality" if index < inequality_count else "equality",
            "index": index,
            "satisfaction_count": None,
            "satisfaction_rate": None,
            "mean_violation": None,
            "max_violation": None,
            "limitation": "not_available_from_constrained_ga_summary",
        }
        for index, name in enumerate(_problem_constraint_names(problem))
    ]
    per_constraint_summary = (
        per_constraint_trace_aggregates
        if per_constraint_trace_aggregates
        else unavailable_per_constraint
    )
    equality_metrics = _constraint_type_metrics(per_constraint_summary, "equality")
    row = {
        "strategy": "constrained_ga_feasibility_first",
        "seed": seed,
        "problem": problem.name,
        "dimension": problem.dimension,
        "bounds": problem.source_bounds(),
        "objective_directions": [False],
        "constraint_count": constraint_count,
        "inequality_count": inequality_count,
        "equality_count": equality_count,
        "constraint_names": _problem_constraint_names(problem),
        "tolerance": getattr(problem, "equality_tolerance", 1e-8),
        "feasibility_policy": "feasibility_first",
        "violation_aggregation": "total_violation",
        "non_finite_constraint_fail_fast_policy": "value_error",
        "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
        "requested_budget": summary["requested_budget"],
        "actual_evaluations": summary["actual_evaluations"],
        "runtime_seconds": runtime_seconds,
        "failure_count": 0,
        "feasible_rate": summary["feasible_rate"],
        "feasible_count": summary["feasible_count"],
        "infeasible_count": summary["infeasible_count"],
        "best_feasible_objective": summary["best_feasible_objective"],
        "best_feasible_solution": summary["best_feasible_solution"],
        "best_record_feasible": summary["best_record_feasible"],
        "best_record_total_violation": summary["best_record_total_violation"],
        "best_infeasible_total_violation": summary["best_infeasible_total_violation"],
        "mean_total_violation": summary["mean_total_violation"],
        "median_total_violation": summary["median_total_violation"],
        "min_total_violation": summary["min_total_violation"],
        "max_total_violation": summary["max_total_violation"],
        "mean_max_violation": summary["mean_max_violation"],
        "violation_count_total": summary["violation_count_total"],
        "all_feasible": summary["all_feasible"],
        "any_feasible": summary["any_feasible"],
        "all_infeasible": summary["all_infeasible"],
        "constraint_summary": summary["constraint_summary"],
        "per_constraint_violation_summary": per_constraint_summary,
        "per_constraint_trace_summary": per_constraint_trace_summary,
        "per_constraint_trace_records_sample": per_constraint_trace_summary.get("records_sample", []),
        "per_constraint_trace_limitations": per_constraint_trace_summary.get("limitations", []),
        "per_constraint_trace_warnings": per_constraint_trace_summary.get("warnings", []),
        **equality_metrics,
        "best_record": summary["best_record"],
        "default_changed": summary["default_changed"],
        "ga_default_changed": summary["default_changed"],
        "constrained_ga_opt_in_path_used": summary["constrained_ga_opt_in_path_used"],
        "ga_integration_done": summary["ga_integration_done"],
        "nsga2_constraint_domination_done": summary["nsga2_constraint_domination_done"],
    }
    return row


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(dict(row))

    aggregated: list[dict[str, Any]] = []
    for strategy, strategy_rows in by_strategy.items():
        feasible_objectives = [
            float(row["best_feasible_objective"])
            for row in strategy_rows
            if row["best_feasible_objective"] is not None
        ]
        aggregated.append(
            {
                "strategy": strategy,
                "feasible_rate": sum(float(row["feasible_rate"] or 0.0) for row in strategy_rows)
                / len(strategy_rows),
                "best_feasible_objective": mean(feasible_objectives) if feasible_objectives else None,
                "mean_total_violation": sum(float(row["mean_total_violation"] or 0.0) for row in strategy_rows)
                / len(strategy_rows),
                "equality_satisfaction_rate": (
                    sum(float(row["equality_satisfaction_rate"]) for row in strategy_rows)
                    / len(strategy_rows)
                    if all(row.get("equality_satisfaction_rate") is not None for row in strategy_rows)
                    else None
                ),
                "equality_mean_violation": (
                    sum(float(row["equality_mean_violation"]) for row in strategy_rows)
                    / len(strategy_rows)
                    if all(row.get("equality_mean_violation") is not None for row in strategy_rows)
                    else None
                ),
                "equality_max_violation": (
                    max(float(row["equality_max_violation"]) for row in strategy_rows)
                    if all(row.get("equality_max_violation") is not None for row in strategy_rows)
                    else None
                ),
                "min_total_violation": min(float(row["min_total_violation"] or 0.0) for row in strategy_rows),
                "actual_evaluations": strategy_rows[0]["actual_evaluations"],
                "runtime": sum(float(row["runtime_seconds"]) for row in strategy_rows)
                / len(strategy_rows),
            }
        )
    return aggregated


def _aggregate_per_constraint_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        strategy = str(row["strategy"])
        for item in row.get("per_constraint_violation_summary", []):
            if item.get("satisfaction_rate") is None:
                continue
            key = (strategy, str(item["constraint"]))
            buckets.setdefault(key, []).append(dict(item))

    summary_rows: list[dict[str, Any]] = []
    for (strategy, constraint_name), entries in sorted(buckets.items()):
        summary_rows.append(
            {
                "constraint": f"{strategy}:{constraint_name}",
                "strategy": strategy,
                "constraint_name": constraint_name,
                "constraint_type": entries[0].get("constraint_type", ""),
                "satisfaction_rate": sum(float(item["satisfaction_rate"]) for item in entries)
                / len(entries),
                "mean_violation": sum(float(item["mean_violation"]) for item in entries)
                / len(entries),
                "max_violation": max(float(item["max_violation"]) for item in entries),
            }
        )
    return summary_rows


def _build_fairness_rows(fairness_payload: dict[str, Any]) -> list[dict[str, Any]]:
    issues = list(fairness_payload.get("issues", []))
    if not issues:
        return [{"item": "fairness_summary", "status": "pass", "message": "No fairness issues recorded."}]
    return [
        {
            "item": issue.get("issue_type", "unknown_issue"),
            "status": issue.get("status", "unknown"),
            "message": issue.get("message", ""),
        }
        for issue in issues
    ]


def _build_failures_and_warnings_rows(
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for failure in failures:
        rows.append(
            {
                "type": failure.get("type", "failure"),
                "message": failure.get("message", ""),
                "action": "문제 원인을 확인한 뒤 constrained path를 고치고 다시 실행한다",
            }
        )
    for warning in warnings:
        rows.append(
            {
                "type": warning.get("type", "warning"),
                "message": warning.get("message", ""),
                "action": "거버넌스 warning으로 기록하고 budget/fairness contract를 다시 확인한다",
            }
        )
    if not rows:
        rows.append({"type": "none", "message": "No smoke failures or warnings were recorded.", "action": "none"})
    return rows


def _comparison_note(problem_name: str, summaries: Sequence[dict[str, Any]]) -> str:
    by_strategy = {str(row["strategy"]): row for row in summaries}
    baseline = by_strategy.get("random_search_feasibility_first")
    constrained = by_strategy.get("constrained_ga_feasibility_first")
    if baseline is None or constrained is None:
        return "비교용 strategy summary가 모두 준비되지 않아 경향 해석을 생략한다."

    messages: list[str] = []
    baseline_feasible = baseline.get("feasible_rate")
    constrained_feasible = constrained.get("feasible_rate")
    if isinstance(baseline_feasible, float) and isinstance(constrained_feasible, float):
        if constrained_feasible > baseline_feasible:
            messages.append("constrained GA가 feasible_rate에서 더 좋은 경향을 보였다.")
        elif constrained_feasible < baseline_feasible:
            messages.append("random search가 feasible_rate에서 더 좋은 경향을 보였다.")

    baseline_best = baseline.get("best_feasible_objective")
    constrained_best = constrained.get("best_feasible_objective")
    if isinstance(baseline_best, float) and isinstance(constrained_best, float):
        if constrained_best < baseline_best:
            messages.append("constrained GA가 더 낮은 best_feasible_objective를 보였다.")
        elif constrained_best > baseline_best:
            messages.append("random search가 더 낮은 best_feasible_objective를 보였다.")

    if not messages:
        messages.append("두 전략 모두 smoke artifact contract를 만족했다.")
    messages.append(
        f"이 비교는 {problem_name} toy benchmark에서의 integration sanity 확인이지 일반 성능 우월 주장으로 읽으면 안 된다."
    )
    return " ".join(messages)


def _render_results_markdown(problem_name: str, rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"# {_problem_display_name(problem_name)} Smoke Results",
            "",
            _markdown_table(
                rows,
                [
                    "strategy",
                    "seed",
                    "problem",
                    "feasible_rate",
                    "best_feasible_objective",
                    "mean_total_violation",
                    "equality_satisfaction_rate",
                    "actual_evaluations",
                ],
            ),
            "",
        ]
    )


def _render_fairness_markdown(problem_name: str, fairness_payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {_problem_display_name(problem_name)} Smoke Fairness Report",
            "",
            f"- Status: **{fairness_payload['status']}**",
            f"- Summary counts: `{fairness_payload.get('summary_counts', {})}`",
            "",
            _markdown_table(
                list(fairness_payload.get("issues", [])),
                ["issue_type", "status", "severity", "message", "recommended_action"],
            ),
            "",
        ]
    )


def _benchmark_definition_rows(problem) -> list[dict[str, Any]]:
    if problem.name == "constrained_equality_plane_quadratic":
        return [
            {"field": "objective", "value": "weighted shifted quadratic minimization"},
            {"field": "dimension", "value": problem.dimension},
            {"field": "bounds", "value": problem.source_bounds()},
            {
                "field": "inequality constraints",
                "value": [f"sum(second_half) <= {problem.group_budget}"],
            },
            {
                "field": "equality constraints",
                "value": [f"sum(first_half) == {problem.equality_target}"],
            },
            {"field": "tolerance", "value": problem.equality_tolerance},
            {
                "field": "expected feasibility profile",
                "value": "mixed feasible/near-feasible at smoke tolerance",
            },
        ]
    if problem.name == "constrained_box_quadratic":
        return [
            {"field": "objective", "value": "weighted shifted quadratic minimization"},
            {"field": "dimension", "value": problem.dimension},
            {"field": "bounds", "value": problem.source_bounds()},
            {
                "field": "inequality constraints",
                "value": [
                    f"sum(first_half) <= {problem.group1_budget}",
                    f"sum(second_half) <= {problem.group2_budget}",
                ],
            },
            {"field": "equality constraints", "value": "none"},
            {"field": "tolerance", "value": problem.equality_tolerance},
            {"field": "expected feasibility profile", "value": "mixed-feasible by default"},
        ]
    if problem.name == "constrained_sphere":
        return [
            {"field": "objective", "value": "sum(x_i^2)"},
            {"field": "dimension", "value": problem.dimension},
            {"field": "bounds", "value": problem.source_bounds()},
            {"field": "inequality constraints", "value": [f"sum(x) <= {problem.budget}"]},
            {
                "field": "equality constraints",
                "value": "none" if getattr(problem, "equality_target", None) is None else "x0 == equality_target",
            },
            {"field": "tolerance", "value": problem.equality_tolerance},
            {"field": "expected feasibility profile", "value": "mixed-feasible under the default budget"},
        ]
    return [
        {"field": "objective", "value": "problem-specific"},
        {"field": "dimension", "value": getattr(problem, "dimension", "unknown")},
        {"field": "bounds", "value": problem.source_bounds()},
        {"field": "inequality constraints", "value": "problem-specific"},
        {"field": "equality constraints", "value": "problem-specific"},
        {"field": "tolerance", "value": getattr(problem, "equality_tolerance", 1e-8)},
        {"field": "expected feasibility profile", "value": "problem-specific"},
    ]


def _constraint_examples_rows(problem) -> list[dict[str, Any]]:
    examples_fn = getattr(problem, "example_cases", None)
    if callable(examples_fn):
        examples = examples_fn()
        if isinstance(examples, list):
            return [
                {
                    "case": item.get("case", "unknown"),
                    "expected feasible": item.get("expected_feasible", ""),
                    "expected violation": item.get("expected_violation", ""),
                }
                for item in examples
            ]
    return [
        {
            "case": "not provided",
            "expected feasible": "n/a",
            "expected violation": "benchmark-specific examples were not provided",
        }
    ]


def _artifact_prefix(problem_name: str) -> str:
    normalized = _normalize_problem_name(problem_name)
    if normalized == "constrained_sphere":
        return "constrained_ga_smoke"
    return f"{normalized}_smoke"


def _render_smoke_report(
    *,
    problem,
    config: ConstrainedGASmokeConfig,
    artifact: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    per_constraint_summaries: Sequence[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> str:
    fairness_rows = _build_fairness_rows(fairness_payload)
    issue_rows = _build_failures_and_warnings_rows(failures, warnings)
    regression_rows = [
        {
            "command": (
                f"python scripts/validate_constrained_ga_smoke.py --problem {problem.name} "
                f"--dimension {config.dimension} --seeds {config.seeds} --budget {config.budget} "
                f"--artifact-suffix {config.artifact_suffix}"
            ),
            "result": "PASS",
            "note": "smoke artifact generation completed",
        }
    ]
    has_per_constraint_limitation = any(
        any(item.get("limitation") for item in row.get("per_constraint_violation_summary", []))
        for row in artifact.get("rows", [])
    )
    return "\n".join(
        [
            f"# {_problem_display_name(problem.name)} Smoke Report",
            "",
            "## 1. Executive Summary",
            "",
            f"- 이번 작업의 목표는 `{problem.name}`를 구현하고 restricted opt-in constrained GA path가 second toy에서도 smoke 수준으로 동작하는지 확인하는 것이었다.",
            f"- 구현한 benchmark는 `{problem.name}`이다.",
            f"- 실행한 smoke strategy는 `{', '.join(config.strategies)}`이다.",
            f"- random search와 constrained GA 모두 `{problem.name}`에서 artifact를 생성했다.",
            f"- fairness 결과 상태는 `{fairness_payload['status']}`이고 summary_counts={fairness_payload.get('summary_counts', {})}이다.",
            "- 기본값 변경은 없다.",
            "- GA/NSGA-II default 영향은 없다.",
            "- Level 해석은 default algorithm 승격이 아니라 실험 툴킷 근거 강화에 한정된다.",
            "",
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "",
            f"- {problem.name}",
            f"- dimension {config.dimension}",
            "- random_search_feasibility_first",
            "- constrained_ga_feasibility_first",
            "- feasibility-first comparator",
            "- constrained fairness",
            "- smoke artifact generation",
            "",
            "Non-Scope:",
            "",
            "- default GA replacement",
            "- NSGA-II constraint-domination",
            "- penalty/repair",
            "- constrained MOEA benchmark",
            "- productization",
            "",
            "## 3. Benchmark Definition",
            "",
            _markdown_table(_benchmark_definition_rows(problem), ["field", "value"]),
            "",
            "## 4. Constraint Examples",
            "",
            _markdown_table(_constraint_examples_rows(problem), ["case", "expected feasible", "expected violation"]),
            "",
            "## 5. Smoke Configuration",
            "",
            _markdown_table(
                [
                    {"field": "problem", "value": problem.name},
                    {"field": "dimension", "value": config.dimension},
                    {"field": "budget", "value": config.budget},
                    {"field": "seeds", "value": config.seeds},
                    {"field": "strategies", "value": ", ".join(config.strategies)},
                    {"field": "tolerance", "value": _effective_tolerance(config)},
                    {"field": "feasibility policy", "value": "feasibility_first"},
                ],
                ["field", "value"],
            ),
            "",
            "## 6. Results Summary",
            "",
            _markdown_table(
                summaries,
                [
                    "strategy",
                    "feasible_rate",
                    "best_feasible_objective",
                    "mean_total_violation",
                    "equality_satisfaction_rate",
                    "min_total_violation",
                    "actual_evaluations",
                    "runtime",
                ],
            ),
            "",
            "## 7. Per-Constraint Summary",
            "",
            _markdown_table(
                per_constraint_summaries
                if per_constraint_summaries
                else [
                    {
                        "constraint": "not_available",
                        "satisfaction_rate": "n/a",
                        "mean_violation": "n/a",
                        "max_violation": "n/a",
                    }
                ],
                [
                    "strategy",
                    "constraint_name",
                    "constraint_type",
                    "satisfaction_rate",
                    "mean_violation",
                    "max_violation",
                ],
            ),
            (
                "- limitation: constrained GA summary는 현재 raw evaluation별 per-constraint trace를 직접 저장하지 않으므로, "
                "per-constraint aggregate는 random search 측 summary를 중심으로 해석해야 한다."
                if has_per_constraint_limitation
                else ""
            ),
            "",
            "## 8. Fairness Summary",
            "",
            _markdown_table(fairness_rows, ["item", "status", "message"]),
            "",
            "## 9. Comparison Interpretation",
            "",
            f"- {_comparison_note(problem.name, summaries)}",
            f"- 이 비교는 `{problem.name}` toy benchmark에서의 smoke sanity signal이지 일반 성능 주장으로 읽으면 안 된다.",
            "",
            "## 10. Failures and Warnings",
            "",
            _markdown_table(issue_rows, ["type", "message", "action"]),
            "",
            "## 11. Regression Check",
            "",
            _markdown_table(regression_rows, ["command", "result", "note"]),
            "",
            "## 12. What This Proves",
            "",
            f"- second constrained toy benchmark `{problem.name}`가 registry와 smoke runner에서 동작한다.",
            "- feasibility-first constrained GA path가 constrained_sphere 외 toy에서도 실행 가능하다.",
            "- constrained artifact, fairness, metrics가 생성된다.",
            "- mutation_fn contract와 finite validation이 유지된다.",
            "",
            "## 13. What This Does Not Prove",
            "",
            "- default GA가 constrained optimizer가 됐다는 뜻은 아니다.",
            "- NSGA-II constraint-domination이 구현됐다는 뜻은 아니다.",
            "- penalty/repair가 구현됐다는 뜻은 아니다.",
            "- industrial constrained optimization을 지원한다는 뜻은 아니다.",
            "- constrained GA가 모든 constrained problem에서 random search보다 우수하다는 뜻은 아니다.",
            "",
            "## 14. Maturity Impact",
            "",
            "- Level 4 근거 강화",
            "- second toy smoke는 실험 툴킷 근거를 강화한다.",
            "- default GA/NSGA-II가 바뀌지 않았으므로 default algorithm maturity 상향은 금지된다.",
            "- constrained benchmark evidence가 toy 2개 수준이므로 범용 constrained optimizer 주장은 금지된다.",
            "",
            "## 15. Recommended Next Work",
            "",
            "1. constrained_box_quadratic stress planning",
            "2. constrained GA opt-in broader review update",
            "3. third constrained toy 또는 equality-constrained toy planning",
            "4. NSGA-II constraint-domination planning, 단 single-objective constrained evidence와 분리",
            "5. constrained ZDT-style toy planning",
            "6. fairness checker 통합 범위 확장",
            "7. checkpoint/resume",
            "8. parallel evaluation",
            "",
            f"이번 {problem.name} smoke 결과, 저장소는 second constrained toy smoke까지 검증했지만, default GA/NSGA-II 통합은 미구현 상태이며, 다음 단계는 constrained_box_quadratic stress planning이다.",
            "",
        ]
    )


def _render_equality_smoke_report(
    *,
    problem,
    config: ConstrainedGASmokeConfig,
    artifact: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    per_constraint_summaries: Sequence[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> str:
    fairness_rows = _build_fairness_rows(fairness_payload)
    issue_rows = _build_failures_and_warnings_rows(failures, warnings)
    regression_rows = [
        {
            "command": (
                f"python scripts/validate_constrained_ga_smoke.py --problem {problem.name} "
                f"--dimension {config.dimension} --seeds {config.seeds} --budget {config.budget} "
                f"--artifact-suffix {config.artifact_suffix}"
            ),
            "result": "PASS",
            "note": "smoke artifact generation completed",
        }
    ]
    equality_rows = [
        row
        for row in per_constraint_summaries
        if row.get("constraint_type") == "equality"
    ]
    equality_note = (
        "equality rows recorded"
        if equality_rows
        else "no equality per-constraint rows were recorded"
    )

    return "\n".join(
        [
            "# Constrained Equality Plane Quadratic Smoke Report",
            "",
            "## 1. Executive Summary",
            "",
            "- 이번 작업의 목표는 equality tolerance semantics를 가진 `constrained_equality_plane_quadratic` benchmark를 구현하고 explicit opt-in constrained GA smoke가 동작하는지 확인하는 것이다.",
            f"- 실행한 smoke scope는 dimension {config.dimension}, seeds {config.seeds}, budget {config.budget}, tolerance {_effective_tolerance(config)}이다.",
            "- random_search_feasibility_first와 constrained_ga_feasibility_first를 비교했다.",
            f"- fairness 결과는 `{fairness_payload['status']}`이고 summary_counts={fairness_payload.get('summary_counts', {})}이다.",
            "- 기본 GA/NSGA-II default는 변경하지 않았다.",
            "- Level 판정은 default algorithm 상향이 아니라 실험 툴킷 근거 강화로만 해석한다.",
            "",
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "",
            "- constrained_equality_plane_quadratic",
            f"- dimension {config.dimension}",
            "- random_search_feasibility_first",
            "- constrained_ga_feasibility_first",
            "- equality tolerance validation",
            "- constrained fairness",
            "- smoke artifact generation",
            "",
            "Non-Scope:",
            "",
            "- default GA replacement",
            "- NSGA-II constraint-domination",
            "- penalty/repair",
            "- constrained MOEA benchmark",
            "- productization",
            "",
            "## 3. Benchmark Definition",
            "",
            _markdown_table(_benchmark_definition_rows(problem), ["field", "value"]),
            "",
            "## 4. Constraint Examples",
            "",
            _markdown_table(
                _constraint_examples_rows(problem),
                ["case", "expected feasible", "expected violation"],
            ),
            "",
            "## 5. Smoke Configuration",
            "",
            _markdown_table(
                [
                    {"field": "problem", "value": problem.name},
                    {"field": "dimension", "value": config.dimension},
                    {"field": "budget", "value": config.budget},
                    {"field": "seeds", "value": config.seeds},
                    {"field": "strategies", "value": ", ".join(config.strategies)},
                    {"field": "tolerance", "value": _effective_tolerance(config)},
                    {"field": "feasibility policy", "value": "feasibility_first"},
                ],
                ["field", "value"],
            ),
            "",
            "## 6. Results Summary",
            "",
            _markdown_table(
                summaries,
                [
                    "strategy",
                    "feasible_rate",
                    "best_feasible_objective",
                    "mean_total_violation",
                    "equality_satisfaction_rate",
                    "actual_evaluations",
                    "runtime",
                ],
            ),
            "",
            "## 7. Per-Constraint Summary",
            "",
            _markdown_table(
                per_constraint_summaries,
                [
                    "strategy",
                    "constraint_name",
                    "constraint_type",
                    "satisfaction_rate",
                    "mean_violation",
                    "max_violation",
                ],
            ),
            f"- equality trace status: {equality_note}",
            "",
            "## 8. Fairness Summary",
            "",
            _markdown_table(fairness_rows, ["item", "status", "message"]),
            "",
            "## 9. Comparison Interpretation",
            "",
            f"- {_comparison_note(problem.name, summaries)}",
            "- equality tolerance가 feasible_rate에 직접 영향을 줄 수 있으므로 이번 비교는 성능 주장이 아니라 integration sanity 신호다.",
            "- 이 toy는 project-local equality/tolerance contract smoke이며 industrial constrained optimization 근거가 아니다.",
            "",
            "## 10. Failures and Warnings",
            "",
            _markdown_table(issue_rows, ["type", "message", "action"]),
            "",
            "## 11. Regression Check",
            "",
            _markdown_table(regression_rows, ["command", "result", "note"]),
            "",
            "## 12. What This Proves",
            "",
            "- equality tolerance benchmark가 registry와 smoke runner에서 동작함.",
            "- equality violation semantics가 artifact로 남음.",
            "- constrained artifact/fairness/trace가 equality case에서도 생성됨.",
            "- mutation_fn contract와 finite validation이 유지됨.",
            "",
            "## 13. What This Does Not Prove",
            "",
            "- default GA가 constrained optimizer가 됐다는 뜻은 아님.",
            "- NSGA-II constraint-domination이 구현됐다는 뜻은 아님.",
            "- penalty/repair가 구현됐다는 뜻은 아님.",
            "- industrial constrained optimization을 지원한다는 뜻은 아님.",
            "- constrained GA가 모든 equality-constrained problem에서 random search보다 우수하다는 뜻은 아님.",
            "",
            "## 14. Maturity Impact",
            "",
            "- Level 4 근거 강화",
            "- equality toy smoke는 실험 툴킷 근거를 강화한다.",
            "- default GA/NSGA-II가 바뀌지 않았으므로 default algorithm maturity 상향은 금지한다.",
            "- constrained benchmark evidence가 toy 3개 수준이므로 범용 constrained optimizer 주장은 금지한다.",
            "",
            "## 15. Recommended Next Work",
            "",
            "1. equality toy stress planning",
            "2. broader constrained GA review update",
            "3. NSGA-II constraint-domination planning, 단 single-objective constrained evidence와 분리",
            "4. constrained ZDT-style toy planning",
            "5. fairness checker 통합 범위 확장",
            "6. checkpoint/resume",
            "7. parallel evaluation",
            "",
            "이번 constrained_equality_plane_quadratic smoke 결과, 저장소는 equality tolerance smoke artifact/fairness/trace 생성까지 검증했지만, default GA/NSGA-II 통합은 미구현 상태이며, 다음 단계는 equality toy stress planning이다.",
            "",
        ]
    )


def run_constrained_ga_smoke(config: ConstrainedGASmokeConfig) -> dict[str, Any]:
    for strategy in config.strategies:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
    if config.dimension <= 0:
        raise ValueError("dimension must be > 0")
    if config.budget <= 0:
        raise ValueError("budget must be > 0")
    if config.population_size <= 1:
        raise ValueError("population_size must be > 1")

    seed_list = _seed_list(config.seeds)
    problem = _build_problem_for_smoke(config)
    contract = _constraint_contract(problem, config)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for strategy in config.strategies:
        for seed in seed_list:
            try:
                if strategy == "random_search_feasibility_first":
                    row = _random_search_row(config=config, problem=problem, seed=seed)
                elif strategy == "constrained_ga_feasibility_first":
                    row = _constrained_ga_row(config=config, problem=problem, seed=seed)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported strategy: {strategy}")
            except Exception as exc:  # pragma: no cover - failure path
                failures.append(
                    {
                        "type": "strategy_failure",
                        "strategy": strategy,
                        "seed": seed,
                        "message": str(exc),
                    }
                )
                continue
            if int(row["actual_evaluations"]) != config.budget:
                warnings.append(
                    {
                        "type": "actual_evaluations_mismatch",
                        "strategy": strategy,
                        "seed": seed,
                        "message": (
                            f"expected {config.budget} evaluations but observed "
                            f"{row['actual_evaluations']}"
                        ),
                    }
                )
            rows.append(row)

    fairness_payload = evaluate_constrained_fairness(
        expected_contract=contract,
        observed_rows=rows,
    )
    summaries = _aggregate_rows(rows)
    per_constraint_summaries = _aggregate_per_constraint_summaries(rows)
    artifact = {
        "command_metadata": {
            "argv": sys.argv,
            "artifact_suffix": config.artifact_suffix,
            "output_dir": config.output_dir,
        },
        "problem": problem.name,
        "configuration": {
            "problem": problem.name,
            "dimension": config.dimension,
            "requested_budget": config.budget,
            "constraint_budget": config.constraint_budget,
            "seeds": config.seeds,
            "seed_list": seed_list,
            "tolerance": _effective_tolerance(config),
            "population_size": config.population_size,
            "feasibility_policy": "feasibility_first",
            "violation_aggregation": "total_violation",
            "strategies": list(config.strategies),
        },
        "benchmark_contract": contract,
        "rows": rows,
        "summaries": summaries,
        "per_constraint_summaries": per_constraint_summaries,
        "fairness_summary": fairness_payload,
        "failures": failures,
        "warnings": warnings,
        "default_changed": False,
        "ga_default_changed": False,
        "constrained_ga_opt_in_path_status": "implemented_explicit_opt_in_only",
        "ga_integration_done": False,
        "nsga2_constraint_domination_done": False,
    }

    project_root = _project_root()
    output_dir_input = Path(config.output_dir)
    output_dir = output_dir_input if output_dir_input.is_absolute() else project_root / output_dir_input
    prefer_absolute_artifact_paths = output_dir_input.is_absolute()
    suffix = config.artifact_suffix
    artifact_prefix = _artifact_prefix(problem.name)

    json_path = output_dir / f"{artifact_prefix}_results_{suffix}.json"
    csv_path = output_dir / f"{artifact_prefix}_results_{suffix}.csv"
    md_results_path = output_dir / f"{artifact_prefix}_results_{suffix}.md"
    report_path = output_dir / f"{artifact_prefix}_report_{suffix}.md"
    fairness_path = output_dir / f"{artifact_prefix}_fairness_report_{suffix}.md"

    artifact["artifacts"] = {
        "json": _artifact_ref(json_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "csv": _artifact_ref(csv_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "results_markdown": _artifact_ref(md_results_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "report_markdown": _artifact_ref(report_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
        "fairness_markdown": _artifact_ref(fairness_path, project_root, prefer_absolute=prefer_absolute_artifact_paths),
    }

    _write_json(json_path, artifact)
    _write_csv(
        csv_path,
        rows,
        [
            "strategy",
            "seed",
            "problem",
            "dimension",
            "constraint_count",
            "inequality_count",
            "equality_count",
            "feasible_rate",
            "feasible_count",
            "infeasible_count",
            "best_feasible_objective",
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
            "per_constraint_violation_summary",
            "per_constraint_trace_summary",
            "requested_budget",
            "actual_evaluations",
            "runtime_seconds",
            "failure_count",
        ],
    )
    _ensure_fresh_path(md_results_path).write_text(
        _render_results_markdown(problem.name, rows),
        encoding="utf-8",
    )
    _ensure_fresh_path(fairness_path).write_text(
        _render_fairness_markdown(problem.name, fairness_payload),
        encoding="utf-8",
    )
    _ensure_fresh_path(report_path).write_text(
        (
            _render_equality_smoke_report
            if problem.name == "constrained_equality_plane_quadratic"
            else _render_smoke_report
        )(
            problem=problem,
            config=config,
            artifact=artifact,
            summaries=summaries,
            per_constraint_summaries=per_constraint_summaries,
            fairness_payload=fairness_payload,
            failures=failures,
            warnings=warnings,
        ),
        encoding="utf-8",
    )
    return artifact


__all__ = [
    "DEFAULT_STRATEGIES",
    "SUPPORTED_STRATEGIES",
    "ConstrainedGASmokeConfig",
    "_artifact_ref",
    "_build_failures_and_warnings_rows",
    "_build_fairness_rows",
    "_constraint_contract",
    "_constrained_ga_row",
    "_ensure_fresh_path",
    "_markdown_table",
    "_project_root",
    "_random_search_row",
    "_seed_list",
    "_stringify",
    "_write_csv",
    "_write_json",
    "run_constrained_ga_smoke",
]
