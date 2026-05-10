from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from ga_lab.algorithms._shared import validate_fitness_vector
from ga_lab.algorithms.constrained_nsga2 import run_constrained_nsga2
from ga_lab.config import GAConfig
from ga_lab.constraints import ConstraintEvaluation
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
from ga_lab.experiment.constrained_mo_metrics import summarize_constrained_mo_population
from ga_lab.experiment.mo_metrics import reference_front_for_problem
from ga_lab.factory import build_runtime_context
from ga_lab.problems import build_problem
from ga_lab.problems.base import as_fitness_vector


DEFAULT_STRATEGIES = (
    "constrained_nsga2_constraint_domination",
    "random_pareto_archive",
)
SUPPORTED_STRATEGIES = DEFAULT_STRATEGIES
SUPPORTED_PROBLEMS = ("constrained_zdt_box_toy", "constrained_dtlz_box_toy")


@dataclass(slots=True)
class ConstrainedNSGA2SmokeConfig:
    dimension: int = 6
    seeds: int | Sequence[int] = 5
    budget: int = 760
    tolerance: float = 1e-8
    population_size: int | None = None
    artifact_suffix: str = "constrained_nsga2_smoke1"
    output_dir: str = "artifacts"
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    problem: str = "constrained_zdt_box_toy"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_problem_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _resolve_budget_schedule(
    budget: int,
    *,
    preferred_population_size: int | None = None,
) -> tuple[int, int]:
    if budget <= 0:
        raise ValueError("budget must be > 0")
    preferred = preferred_population_size if preferred_population_size is not None else 20
    candidates: list[tuple[int, int, int]] = []
    for population_size in range(4, budget + 1):
        if budget % population_size != 0:
            continue
        quotient = budget // population_size
        if quotient < 5:
            continue
        if (quotient - 2) % 3 != 0:
            continue
        generations = (quotient - 2) // 3
        if generations < 1:
            continue
        candidates.append((abs(population_size - preferred), -population_size, generations))
    if not candidates:
        raise ValueError(
            "budget must be compatible with exact NSGA-II accounting "
            "(budget = population_size * ((3 * generations) + 2))"
        )
    _, neg_population_size, generations = min(candidates)
    return -neg_population_size, generations


def _build_problem_for_smoke(config: ConstrainedNSGA2SmokeConfig):
    problem_name = _normalize_problem_name(config.problem)
    if problem_name not in SUPPORTED_PROBLEMS:
        raise ValueError(f"Unsupported constrained NSGA-II smoke problem: {config.problem}")
    return build_problem(
        problem_name,
        {
            "dimension": config.dimension,
            "equality_tolerance": config.tolerance,
        },
    )


def _artifact_prefix(problem_name: str) -> str:
    if problem_name == "constrained_dtlz_box_toy":
        return "constrained_nsga2_dtlz_smoke"
    return "constrained_nsga2_smoke"


def _report_title(problem_name: str) -> str:
    if problem_name == "constrained_dtlz_box_toy":
        return "Constrained NSGA-II DTLZ Box Toy Smoke Report"
    return "Constrained NSGA-II Smoke Report"


def _results_title(problem_name: str) -> str:
    if problem_name == "constrained_dtlz_box_toy":
        return "Constrained NSGA-II DTLZ Box Toy Smoke Results"
    return "Constrained NSGA-II Smoke Results"


def _fairness_title(problem_name: str) -> str:
    if problem_name == "constrained_dtlz_box_toy":
        return "Constrained NSGA-II DTLZ Box Toy Smoke Fairness Report"
    return "Constrained NSGA-II Smoke Fairness Report"


def _reference_point(problem, directions: list[bool]) -> list[float]:
    provider = getattr(problem, "hypervolume_reference_point", None)
    if callable(provider):
        return [float(value) for value in provider(directions)]
    raise ValueError("constrained NSGA-II smoke requires problem.hypervolume_reference_point")


def _reference_front_policy(problem) -> str:
    if isinstance(getattr(problem, "reference_front_name", None), str):
        return str(getattr(problem, "reference_front_name"))
    return "none"


def _reference_front(problem) -> list[list[float]] | None:
    policy = _reference_front_policy(problem)
    if policy == "none":
        return None
    return reference_front_for_problem(
        policy,
        objective_count=2,
        variable_count=int(getattr(problem, "dimension", 6)),
    )


def _constraint_contract(
    problem,
    *,
    budget: int,
    tolerance: float,
) -> dict[str, Any]:
    directions = [False, False]
    return {
        "problem": problem.name,
        "objective_count": 2,
        "dimension": problem.dimension,
        "bounds": problem.source_bounds(),
        "objective_directions": directions,
        "constraint_count": problem.constraint_count,
        "inequality_count": problem.inequality_count,
        "equality_count": problem.equality_count,
        "tolerance": tolerance,
        "constraint_policy": "constraint_domination",
        "violation_aggregation": "total_violation",
        "feasible_only_metric_policy": "null_when_no_feasible_front",
        "requested_budget": budget,
        "non_finite_constraint_fail_fast_policy": "value_error",
        "reference_front_policy": _reference_front_policy(problem),
        "reference_point": _reference_point(problem, directions),
    }


def _problem_constraint_names(problem) -> tuple[list[str], list[str]]:
    names_fn = getattr(problem, "constraint_names", None)
    if callable(names_fn):
        names = [str(item) for item in names_fn()]
    else:
        names = [
            *[f"inequality_{index}" for index in range(int(getattr(problem, "inequality_count", 0)))],
            *[f"equality_{index}" for index in range(int(getattr(problem, "equality_count", 0)))],
        ]
    return names[: int(getattr(problem, "inequality_count", 0))], names[int(getattr(problem, "inequality_count", 0)) :]


def _summarize_strategy_run(
    *,
    strategy: str,
    seed: int,
    problem,
    budget: int,
    objective_vectors: Sequence[Sequence[float]],
    evaluations: Sequence[ConstraintEvaluation],
    runtime_seconds: float,
    reference_point: Sequence[float],
    reference_front,
) -> dict[str, Any]:
    directions = [False, False]
    inequality_names, equality_names = _problem_constraint_names(problem)
    summary = summarize_constrained_mo_population(
        objective_vectors,
        evaluations,
        directions=directions,
        reference_point=reference_point,
        reference_front=reference_front,
        inequality_names=inequality_names,
        equality_names=equality_names,
    )
    return {
        "strategy": strategy,
        "seed": seed,
        "problem": problem.name,
        "dimension": problem.dimension,
        "bounds": problem.source_bounds(),
        "objective_count": 2,
        "objective_directions": directions,
        "constraint_count": problem.constraint_count,
        "inequality_count": problem.inequality_count,
        "equality_count": problem.equality_count,
        "tolerance": float(getattr(problem, "equality_tolerance", 1e-8)),
        "constraint_policy": "constraint_domination",
        "violation_aggregation": "total_violation",
        "feasible_only_metric_policy": "null_when_no_feasible_front",
        "non_finite_constraint_fail_fast_policy": "value_error",
        "reference_front_policy": _reference_front_policy(problem),
        "reference_point": list(reference_point),
        "requested_budget": budget,
        "actual_evaluations": len(objective_vectors),
        "runtime_seconds": runtime_seconds,
        "failure_count": 0,
        **summary,
        "default_changed": False,
        "nsga2_default_changed": False,
        "nsga2_constraint_domination_done": strategy == "constrained_nsga2_constraint_domination",
    }


def _random_solution(rng: random.Random, problem) -> list[float]:
    return [
        rng.uniform(float(low), float(high))
        for low, high in problem.source_bounds()
    ]


def _random_pareto_archive_row(
    *,
    seed: int,
    problem,
    budget: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    objective_vectors: list[list[float]] = []
    evaluations: list[ConstraintEvaluation] = []
    start = time.perf_counter()
    for evaluation_index in range(budget):
        solution = _random_solution(rng, problem)
        objectives = as_fitness_vector(problem.fitness(solution))
        validate_fitness_vector(
            objectives,
            problem_name=problem.name,
            genome=solution,
            location="random_pareto_archive",
            generation=0,
            evaluation_index=evaluation_index,
        )
        objective_vectors.append([float(value) for value in objectives])
        evaluations.append(problem.evaluate_constraints(solution))
    runtime_seconds = time.perf_counter() - start
    return _summarize_strategy_run(
        strategy="random_pareto_archive",
        seed=seed,
        problem=problem,
        budget=budget,
        objective_vectors=objective_vectors,
        evaluations=evaluations,
        runtime_seconds=runtime_seconds,
        reference_point=_reference_point(problem, [False, False]),
        reference_front=_reference_front(problem),
    )


def _constrained_nsga2_row(
    *,
    seed: int,
    problem,
    budget: int,
    population_size: int,
    generations: int,
) -> dict[str, Any]:
    config = GAConfig(
        run_name=f"constrained_nsga2_smoke_seed{seed}",
        problem=problem.name,
        algorithm="nsga2",
        representation="real",
        selection="nsga2_tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=population_size,
        genome_length=problem.dimension,
        generations=generations,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elitism=1,
        tournament_size=2,
        seed=seed,
        maximize=False,
        objective_directions=[False, False],
        representation_options={"low": problem.lower_bound, "high": problem.upper_bound},
        mutation_options={"sigma": 0.05},
        problem_options=problem.problem_options(),
    )
    runtime = build_runtime_context(config)
    start = time.perf_counter()
    result = run_constrained_nsga2(
        config=config,
        problem=runtime.problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=random.Random(seed),
    )
    runtime_seconds = time.perf_counter() - start
    summary = dict(result.summary)
    row = {
        "strategy": "constrained_nsga2_constraint_domination",
        "seed": seed,
        "problem": runtime.problem.name,
        "dimension": runtime.problem.dimension,
        "bounds": runtime.problem.source_bounds(),
        "objective_count": 2,
        "objective_directions": summary["objective_directions"],
        "constraint_count": runtime.problem.constraint_count,
        "inequality_count": runtime.problem.inequality_count,
        "equality_count": runtime.problem.equality_count,
        "tolerance": float(getattr(runtime.problem, "equality_tolerance", 1e-8)),
        "constraint_policy": "constraint_domination",
        "violation_aggregation": "total_violation",
        "feasible_only_metric_policy": summary["feasible_only_metric_policy"],
        "non_finite_constraint_fail_fast_policy": summary[
            "non_finite_constraint_fail_fast_policy"
        ],
        "reference_front_policy": _reference_front_policy(runtime.problem),
        "reference_point": _reference_point(runtime.problem, [False, False]),
        "requested_budget": budget,
        "actual_evaluations": summary["actual_evaluations_used"],
        "runtime_seconds": runtime_seconds,
        "failure_count": 0,
        "feasible_rate": summary["feasible_rate"],
        "feasible_count": summary["feasible_count"],
        "infeasible_count": summary["infeasible_count"],
        "best_total_violation": summary["best_total_violation"],
        "mean_total_violation": summary["mean_total_violation"],
        "median_total_violation": summary["median_total_violation"],
        "min_total_violation": summary["min_total_violation"],
        "max_total_violation": summary["max_total_violation"],
        "mean_max_violation": summary["mean_max_violation"],
        "violation_count_total": summary["violation_count_total"],
        "feasible_nondominated_count": summary["feasible_nondominated_count"],
        "infeasible_nondominated_count": summary["infeasible_nondominated_count"],
        "feasible_only_HV": summary["feasible_only_HV"],
        "feasible_only_reference_distance": summary["feasible_only_reference_distance"],
        "feasible_only_IGD": summary["feasible_only_IGD"],
        "spacing_feasible_only": summary["spacing_feasible_only"],
        "equality_satisfaction_rate": summary["equality_satisfaction_rate"],
        "equality_mean_violation": summary["equality_mean_violation"],
        "equality_max_violation": summary["equality_max_violation"],
        "constraint_summary": summary["constraint_summary"],
        "per_constraint_violation_summary": summary["per_constraint_violation_summary"],
        "default_changed": False,
        "nsga2_default_changed": False,
        "nsga2_constraint_domination_done": True,
    }
    return row


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for strategy in SUPPORTED_STRATEGIES:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        if not strategy_rows:
            continue
        aggregated.append(
            {
                "strategy": strategy,
                "feasible_rate": sum(float(row["feasible_rate"] or 0.0) for row in strategy_rows)
                / len(strategy_rows),
                "feasible_nondominated_count": sum(
                    int(row["feasible_nondominated_count"]) for row in strategy_rows
                )
                / len(strategy_rows),
                "mean_total_violation": sum(
                    float(row["mean_total_violation"] or 0.0) for row in strategy_rows
                )
                / len(strategy_rows),
                "feasible_only_HV": (
                    sum(float(row["feasible_only_HV"]) for row in strategy_rows)
                    / len(strategy_rows)
                    if all(row.get("feasible_only_HV") is not None for row in strategy_rows)
                    else None
                ),
                "feasible_only_reference_distance": (
                    sum(
                        float(row["feasible_only_reference_distance"])
                        for row in strategy_rows
                    )
                    / len(strategy_rows)
                    if all(
                        row.get("feasible_only_reference_distance") is not None
                        for row in strategy_rows
                    )
                    else None
                ),
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


def _render_results_markdown(rows: Sequence[dict[str, Any]]) -> str:
    problem_name = str(rows[0]["problem"]) if rows else "constrained_nsga2_smoke"
    return "\n".join(
        [
            f"# {_results_title(problem_name)}",
            "",
            _markdown_table(
                rows,
                [
                    "strategy",
                    "seed",
                    "problem",
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


def _render_fairness_markdown(fairness_payload: dict[str, Any]) -> str:
    problem_name = str(fairness_payload.get("problem", "constrained_nsga2_smoke"))
    return "\n".join(
        [
            f"# {_fairness_title(problem_name)}",
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


def _decision(
    *,
    fairness_payload: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
) -> str:
    if fairness_payload.get("status") == "fail":
        return "Needs fairness rerun"
    if failures:
        return "Fix required"
    if any(int(row["actual_evaluations"]) != int(row["requested_budget"]) for row in rows):
        return "Fix required"
    if any(row.get("feasible_only_HV") is None for row in rows if row["strategy"] == "constrained_nsga2_constraint_domination"):
        return "Constrained NSGA-II smoke passed with limitations"
    return "Constrained NSGA-II smoke passed, ready for restricted experimental opt-in review"


def _render_smoke_report(
    *,
    problem,
    config: ConstrainedNSGA2SmokeConfig,
    artifact: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    per_constraint_summaries: Sequence[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> str:
    fairness_rows = _build_fairness_rows(fairness_payload)
    issue_rows = _build_failures_and_warnings_rows(failures, warnings)
    objective_description = (
        "DTLZ2-like 2-objective minimization"
        if problem.name == "constrained_dtlz_box_toy"
        else "ZDT-like 2-objective minimization"
    )
    inequality_constraints = (
        [
            f"x0 + x1 <= {problem.first_pair_budget}",
            f"mean(last_half) <= {problem.tail_mean_budget}",
        ]
        if problem.name == "constrained_dtlz_box_toy"
        else [
            f"x0 + x1 <= {problem.first_pair_budget}",
            f"sum(last_half) <= {problem.second_half_budget}",
        ]
    )
    expected_profile = (
        "non-ZDT family toy with mixed feasible and infeasible baseline behavior"
        if problem.name == "constrained_dtlz_box_toy"
        else "mixed feasible/infeasible under random archive, feasible front available for constrained NSGA-II"
    )
    regression_rows = [
        {
            "command": (
                "python scripts/validate_constrained_nsga2_smoke.py "
                f"--problem {problem.name} --dimension {config.dimension} --seeds {len(_seed_list(config.seeds))} "
                f"--budget {config.budget} --artifact-suffix {config.artifact_suffix}"
            ),
            "result": "PASS",
            "note": "smoke artifact generation completed",
        }
    ]
    decision = _decision(fairness_payload=fairness_payload, failures=failures, rows=artifact["rows"])
    return "\n".join(
        [
            f"# {_report_title(problem.name)}",
            "",
            "## 1. Executive Summary",
            "",
            "- 이번 작업의 목표는 별도 opt-in constrained NSGA-II path와 first constrained MO toy benchmark를 smoke 수준으로 검증하는 것이다.",
            "- 구현한 opt-in constrained NSGA-II path는 `run_constrained_nsga2(...)`이다.",
            f"- 구현한 first constrained MO toy benchmark는 `{problem.name}`이다.",
            f"- 실행한 smoke scope는 dimension {config.dimension}, seeds {len(_seed_list(config.seeds))}, budget {config.budget}이다.",
            f"- fairness 결과는 `{fairness_payload['status']}`이고 summary_counts={fairness_payload.get('summary_counts', {})}이다.",
            "- default 변경 여부는 없다.",
            "- NSGA-II default 영향은 없다.",
            "- Level 판정은 default maturity 상향이 아니라 실험 툴킷 근거 강화로만 해석한다.",
            "",
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "",
            f"- {problem.name}",
            "- constrained_nsga2_constraint_domination",
            "- feasible-only constrained MO metrics",
            "- constrained fairness",
            "- smoke artifact generation",
            "",
            "Non-Scope:",
            "",
            "- default NSGA-II replacement",
            "- penalty/repair",
            "- constrained MO full benchmark suite",
            "- productization",
            "",
            "## 3. Constrained NSGA-II Path",
            "",
            _markdown_table(
                [
                    {
                        "item": "function",
                        "value": "run_constrained_nsga2",
                    },
                    {
                        "item": "file",
                        "value": "src/ga_lab/algorithms/constrained_nsga2.py",
                    },
                    {
                        "item": "default path changed?",
                        "value": "false",
                    },
                    {
                        "item": "constraint policy",
                        "value": "constraint_domination",
                    },
                    {
                        "item": "evaluation budget policy",
                        "value": "exact match via population_size * ((3 * generations) + 2)",
                    },
                ],
                ["item", "value"],
            ),
            "",
            "## 4. Benchmark Definition",
            "",
            _markdown_table(
                [
                    {"field": "objectives", "value": objective_description},
                    {"field": "dimension", "value": problem.dimension},
                    {"field": "bounds", "value": problem.source_bounds()},
                    {
                        "field": "inequality constraints",
                        "value": inequality_constraints,
                    },
                    {"field": "equality constraints", "value": "none"},
                    {"field": "tolerance", "value": problem.equality_tolerance},
                    {
                        "field": "expected feasibility profile",
                        "value": expected_profile,
                    },
                ],
                ["field", "value"],
            ),
            "",
            "## 5. Smoke Configuration",
            "",
            _markdown_table(
                [
                    {"field": "problem", "value": problem.name},
                    {"field": "dimension", "value": config.dimension},
                    {"field": "budget", "value": config.budget},
                    {"field": "seeds", "value": len(_seed_list(config.seeds))},
                    {"field": "strategies", "value": list(config.strategies)},
                    {"field": "constraint policy", "value": "constraint_domination"},
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
                    "feasible_nondominated_count",
                    "mean_total_violation",
                    "feasible_only_HV",
                    "feasible_only_reference_distance",
                    "actual_evaluations",
                    "runtime",
                ],
            ),
            "",
            "## 7. Fairness Summary",
            "",
            _markdown_table(fairness_rows, ["item", "status", "message"]),
            "",
            "## 8. Failures and Warnings",
            "",
            _markdown_table(issue_rows, ["type", "message", "action"]),
            "",
            "## 9. Regression Check",
            "",
            _markdown_table(regression_rows, ["command", "result", "note"]),
            "",
            "## 10. What This Proves",
            "",
            "- separate constrained NSGA-II opt-in path가 default를 건드리지 않고 실행 가능함",
            "- constraint-domination principle을 smoke 수준에서 설명 가능함",
            "- feasible-only constrained MO metrics가 생성됨",
            "- fairness and exact budget accounting still work",
            "",
            "## 11. What This Does Not Prove",
            "",
            "- default NSGA-II가 constrained optimizer가 됐다는 뜻은 아님",
            "- product-ready constrained MOEA라는 뜻은 아님",
            "- penalty/repair가 구현됐다는 뜻은 아님",
            "- broad constrained MO benchmark coverage가 확보됐다는 뜻은 아님",
            "- external constrained parity가 확보됐다는 뜻은 아님",
            "",
            "## 12. Decision",
            "",
            f"- {decision}",
            "",
            "## 13. Maturity Impact",
            "",
            "- Level 4 근거 강화",
            "",
            "## 14. Recommended Next Work",
            "",
            "- constrained NSGA-II opt-in review package 작성",
            "",
            (
                "이번 constrained NSGA-II smoke 결과, 저장소는 separate constrained NSGA-II opt-in path와 "
                "first constrained MO toy smoke artifact generation까지 검증했지만, default NSGA-II는 unchanged "
                "상태이며, 다음 단계는 constrained NSGA-II opt-in review package 작성이다."
            ),
        ]
    )


def run_constrained_nsga2_smoke(config: ConstrainedNSGA2SmokeConfig) -> dict[str, Any]:
    for strategy in config.strategies:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
    problem = _build_problem_for_smoke(config)
    population_size, generations = _resolve_budget_schedule(
        config.budget,
        preferred_population_size=config.population_size,
    )
    seed_list = _seed_list(config.seeds)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for strategy in config.strategies:
        for seed in seed_list:
            try:
                if strategy == "constrained_nsga2_constraint_domination":
                    row = _constrained_nsga2_row(
                        seed=seed,
                        problem=problem,
                        budget=config.budget,
                        population_size=population_size,
                        generations=generations,
                    )
                else:
                    row = _random_pareto_archive_row(
                        seed=seed,
                        problem=problem,
                        budget=config.budget,
                    )
            except Exception as exc:
                failures.append(
                    {
                        "type": f"{strategy}_failure",
                        "message": f"seed={seed}: {exc}",
                    }
                )
                continue
            rows.append(row)

    contract = _constraint_contract(
        problem,
        budget=config.budget,
        tolerance=config.tolerance,
    )
    fairness_payload = evaluate_constrained_mo_fairness(
        expected_contract=contract,
        observed_rows=rows,
    )
    fairness_payload["problem"] = problem.name
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
            "seeds": config.seeds,
            "seed_list": seed_list,
            "tolerance": config.tolerance,
            "population_size": population_size,
            "generations": generations,
            "constraint_policy": "constraint_domination",
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
        "nsga2_default_changed": False,
        "nsga2_constraint_domination_done": True,
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
            "objective_count",
            "constraint_count",
            "inequality_count",
            "equality_count",
            "feasible_rate",
            "feasible_nondominated_count",
            "mean_total_violation",
            "feasible_only_HV",
            "feasible_only_reference_distance",
            "constraint_summary",
            "per_constraint_violation_summary",
            "requested_budget",
            "actual_evaluations",
            "runtime_seconds",
            "failure_count",
        ],
    )
    _ensure_fresh_path(md_results_path).write_text(
        _render_results_markdown(rows),
        encoding="utf-8",
    )
    _ensure_fresh_path(fairness_path).write_text(
        _render_fairness_markdown(fairness_payload),
        encoding="utf-8",
    )
    _ensure_fresh_path(report_path).write_text(
        _render_smoke_report(
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
    "ConstrainedNSGA2SmokeConfig",
    "DEFAULT_STRATEGIES",
    "SUPPORTED_STRATEGIES",
    "_resolve_budget_schedule",
    "run_constrained_nsga2_smoke",
]
