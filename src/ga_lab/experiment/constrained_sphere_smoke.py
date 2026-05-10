from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ga_lab.constraints import summarize_constraint_violations
from ga_lab.experiment.constrained_fairness import evaluate_constrained_fairness
from ga_lab.experiment.constrained_metrics import (
    best_feasible_objective,
    feasible_rate,
    summarize_constrained_population,
)
from ga_lab.experiment.constrained_protocol import (
    ConstrainedCandidateRecord,
    select_best_feasible_first,
)
from ga_lab.problems.constrained_sphere import ConstrainedSphereProblem


DEFAULT_STRATEGY = "random_search_feasibility_first"
SUPPORTED_STRATEGIES = (DEFAULT_STRATEGY,)


@dataclass(slots=True)
class SmokeConfig:
    dimension: int = 5
    seeds: int = 5
    budget: int = 300
    constraint_budget: float = 1.0
    tolerance: float = 1e-8
    artifact_suffix: str = "constrained_sphere_smoke1"
    output_dir: str = "artifacts"
    strategies: tuple[str, ...] = (DEFAULT_STRATEGY,)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _constraint_contract(problem: ConstrainedSphereProblem, config: SmokeConfig) -> dict[str, Any]:
    bounds = problem.source_bounds()
    equality_count = 0 if problem.equality_target is None else 1
    return {
        "problem": problem.name,
        "dimension": problem.dimension,
        "bounds": bounds,
        "objective_directions": [False],
        "constraint_count": 1 + equality_count,
        "inequality_count": 1,
        "equality_count": equality_count,
        "tolerance": config.tolerance,
        "feasibility_policy": "feasibility_first",
        "violation_aggregation": "total_violation",
        "requested_budget": config.budget,
        "non_finite_constraint_fail_fast_policy": "value_error",
        "feasible_only_metric_policy": "best_feasible_objective_null_when_absent",
    }


def _random_solution(rng: random.Random, problem: ConstrainedSphereProblem) -> list[float]:
    return [
        rng.uniform(problem.lower_bound, problem.upper_bound)
        for _ in range(problem.dimension)
    ]


def _row_from_run(
    *,
    strategy: str,
    seed: int,
    requested_budget: int,
    problem: ConstrainedSphereProblem,
    records: Sequence[ConstrainedCandidateRecord],
    runtime_seconds: float,
    failure_count: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records cannot be empty")

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

    return {
        "strategy": strategy,
        "seed": seed,
        "problem": problem.name,
        "dimension": problem.dimension,
        "bounds": problem.source_bounds(),
        "objective_directions": [False],
        "constraint_count": len(best_record.constraint_evaluation.inequality_values)
        + len(best_record.constraint_evaluation.equality_values),
        "inequality_count": len(best_record.constraint_evaluation.inequality_values),
        "equality_count": len(best_record.constraint_evaluation.equality_values),
        "tolerance": problem.equality_tolerance,
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
        "best_record": best_record.to_dict(),
        "default_changed": False,
        "ga_integration_done": False,
    }


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
                "best_feasible_objective": min(feasible_objectives) if feasible_objectives else None,
                "mean_total_violation": sum(float(row["mean_total_violation"] or 0.0) for row in strategy_rows)
                / len(strategy_rows),
                "min_total_violation": min(float(row["min_total_violation"] or 0.0) for row in strategy_rows),
                "actual_evaluations": strategy_rows[0]["actual_evaluations"],
                "runtime": sum(float(row["runtime_seconds"]) for row in strategy_rows)
                / len(strategy_rows),
            }
        )
    return aggregated


def _build_fairness_rows(fairness_payload: dict[str, Any]) -> list[dict[str, Any]]:
    issues = list(fairness_payload.get("issues", []))
    if not issues:
        return [
            {
                "item": "fairness_summary",
                "status": "pass",
                "message": "No fairness issues recorded.",
            }
        ]
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
                "action": "inspect failure details and rerun after fixing the evaluator/runner contract",
            }
        )
    for warning in warnings:
        rows.append(
            {
                "type": warning.get("type", "warning"),
                "message": warning.get("message", ""),
                "action": "record as a smoke warning and keep the budget/fairness contract under review",
            }
        )
    if not rows:
        rows.append(
            {
                "type": "none",
                "message": "No smoke failures or warnings were recorded.",
                "action": "none",
            }
        )
    return rows


def _render_results_markdown(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Constrained Sphere Smoke Results",
            "",
            _markdown_table(
                rows,
                [
                    "strategy",
                    "seed",
                    "feasible_rate",
                    "best_feasible_objective",
                    "mean_total_violation",
                    "actual_evaluations",
                ],
            ),
            "",
        ]
    )


def _render_fairness_markdown(fairness_payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Constrained Sphere Smoke Fairness Report",
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


def _render_smoke_report(
    *,
    config: SmokeConfig,
    artifact: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    fairness_payload: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    warnings: Sequence[dict[str, Any]],
) -> str:
    configuration_rows = [
        {"field": "problem", "value": artifact["problem"]},
        {"field": "dimension", "value": config.dimension},
        {"field": "budget", "value": config.budget},
        {"field": "seeds", "value": config.seeds},
        {"field": "feasibility policy", "value": "feasibility_first"},
        {"field": "tolerance", "value": config.tolerance},
        {"field": "strategies", "value": ", ".join(config.strategies)},
    ]
    protocol_rows = [
        {
            "helper": "compare_feasibility_first",
            "behavior": "feasible beats infeasible; feasible ties use objective; infeasible ties use total_violation, max_violation, then objective",
        },
        {
            "helper": "rank_constrained_candidates",
            "behavior": "stable feasibility-first ordering for smoke-only candidate records",
        },
        {
            "helper": "select_best_feasible_first",
            "behavior": "returns the best record under the feasibility-first policy without touching the GA loop",
        },
    ]
    fairness_rows = _build_fairness_rows(fairness_payload)
    issue_rows = _build_failures_and_warnings_rows(failures, warnings)
    regression_rows = [
        {
            "command": "python scripts/validate_constrained_sphere_smoke.py --dimension 5 --seeds 5 --budget 300 --artifact-suffix constrained_sphere_smoke1",
            "result": "runner-side smoke artifact generation is supported",
            "note": "pytest/local baseline results are added in the execution pass after the smoke run",
        }
    ]

    summary_lines = [
        "- 이번 작업의 목표는 `constrained_sphere`에서 feasibility-first smoke artifact와 fairness wiring을 확인하는 것입니다.",
        "- 구현한 runner/helper는 `constrained_protocol`, `constrained_fairness`, `constrained_sphere_smoke`입니다.",
        f"- 실행한 smoke strategy는 `{', '.join(config.strategies)}`입니다.",
        "- feasibility 결과는 seed별 constrained metrics와 violation summary로 artifact에 남았습니다.",
        f"- fairness 결과는 `{fairness_payload['status']}`이며 summary_counts={fairness_payload.get('summary_counts', {})} 입니다.",
        f"- 기본값 변경 여부는 `{artifact['default_changed']}`입니다.",
        "- GA/NSGA-II 통합 여부는 `ga_integration_done=false`, `nsga2_constraint_domination_done=false` 입니다.",
        "- Level 해석은 실험 툴킷 근거 강화로 제한하고, 알고리즘 성숙도 상향은 주장하지 않습니다.",
    ]

    return "\n".join(
        [
            "# Constrained Sphere Smoke Report",
            "",
            "## 1. Executive Summary",
            "",
            *summary_lines,
            "",
            "## 2. Scope and Non-Scope",
            "",
            "Scope:",
            "",
            "- constrained_sphere",
            "- random_search_feasibility_first",
            "- constraint schema",
            "- constrained metrics",
            "- constrained fairness",
            "- smoke artifact generation",
            "",
            "Non-Scope:",
            "",
            "- GA feasibility-first integration",
            "- NSGA-II constraint-domination",
            "- penalty/repair",
            "- constrained MOEA benchmark",
            "- productization",
            "",
            "## 3. Constrained Protocol Helper",
            "",
            _markdown_table(protocol_rows, ["helper", "behavior"]),
            "",
            "## 4. Smoke Configuration",
            "",
            _markdown_table(configuration_rows, ["field", "value"]),
            "",
            "## 5. Results Summary",
            "",
            _markdown_table(
                summaries,
                [
                    "strategy",
                    "feasible_rate",
                    "best_feasible_objective",
                    "mean_total_violation",
                    "min_total_violation",
                    "actual_evaluations",
                    "runtime",
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
            "## 9. What This Proves",
            "",
            "- constraint schema가 smoke runner에서 사용 가능함",
            "- finite validation과 violation summary가 artifact로 남음",
            "- feasibility-first comparator helper가 독립적으로 검증됨",
            "- constrained fairness wiring이 최소 동작함",
            "",
            "## 10. What This Does Not Prove",
            "",
            "- GA가 constrained optimization을 지원한다는 뜻은 아님",
            "- NSGA-II constraint-domination이 구현됐다는 뜻은 아님",
            "- penalty/repair가 구현됐다는 뜻은 아님",
            "- industrial constrained optimization을 지원한다는 뜻은 아님",
            "",
            "## 11. Maturity Impact",
            "",
            "- Level 4 근거 강화",
            "- smoke runner와 fairness wiring은 실험 툴킷 근거를 강화한다.",
            "- 알고리즘 루프에 constraint handling이 통합되지 않았으므로 algorithm maturity 상향은 금지한다.",
            "- constrained benchmark evidence가 toy 수준이므로 범용 constrained optimizer 주장 금지.",
            "",
            "## 12. Recommended Next Work",
            "",
            "1. feasibility-first ranking을 single-objective GA에 experimental opt-in path로 붙일지 planning",
            "2. constrained_sphere GA smoke planning",
            "3. constrained multi-objective contract-domination planning",
            "4. constrained ZDT-style toy planning",
            "5. fairness checker 통합 범위 확장",
            "6. checkpoint/resume",
            "7. parallel evaluation",
            "",
            "이번 constrained sphere smoke 결과, 저장소는 constraint schema, feasibility-first smoke comparator, constrained fairness wiring까지 검증했지만, constrained GA/NSGA-II 통합은 미통합 상태이며, 다음 단계는 single-objective GA opt-in planning과 constrained sphere GA smoke 설계이다.",
            "",
        ]
    )


def run_constrained_sphere_smoke(config: SmokeConfig) -> dict[str, Any]:
    for strategy in config.strategies:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
    if config.dimension <= 0:
        raise ValueError("dimension must be > 0")
    if config.budget <= 0:
        raise ValueError("budget must be > 0")

    seed_list = _seed_list(config.seeds)
    problem = ConstrainedSphereProblem(
        dimension=config.dimension,
        budget=config.constraint_budget,
        equality_tolerance=config.tolerance,
    )
    contract = _constraint_contract(problem, config)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for strategy in config.strategies:
        for seed in seed_list:
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
                            metadata={"strategy": strategy},
                        )
                    )
                except Exception as exc:  # pragma: no cover - failure path
                    failure_count += 1
                    failures.append(
                        {
                            "type": "evaluation_failure",
                            "strategy": strategy,
                            "seed": seed,
                            "evaluation_index": evaluation_index,
                            "message": str(exc),
                        }
                    )
                    break
            runtime_seconds = time.perf_counter() - started
            if len(records) != config.budget:
                warnings.append(
                    {
                        "type": "actual_evaluations_mismatch",
                        "strategy": strategy,
                        "seed": seed,
                        "message": (
                            f"expected {config.budget} evaluations but collected {len(records)}"
                        ),
                    }
                )
            if records:
                rows.append(
                    _row_from_run(
                        strategy=strategy,
                        seed=seed,
                        requested_budget=config.budget,
                        problem=problem,
                        records=records,
                        runtime_seconds=runtime_seconds,
                        failure_count=failure_count,
                    )
                )

    fairness_payload = evaluate_constrained_fairness(
        expected_contract=contract,
        observed_rows=rows,
    )
    summaries = _aggregate_rows(rows)

    artifact = {
        "command_metadata": {
            "argv": sys.argv,
            "artifact_suffix": config.artifact_suffix,
            "output_dir": config.output_dir,
        },
        "problem": problem.name,
        "configuration": {
            "dimension": config.dimension,
            "requested_budget": config.budget,
            "constraint_budget": config.constraint_budget,
            "seeds": config.seeds,
            "seed_list": seed_list,
            "tolerance": config.tolerance,
            "feasibility_policy": "feasibility_first",
            "violation_aggregation": "total_violation",
            "strategies": list(config.strategies),
        },
        "benchmark_contract": contract,
        "rows": rows,
        "summaries": summaries,
        "fairness_summary": fairness_payload,
        "failures": failures,
        "warnings": warnings,
        "default_changed": False,
        "ga_integration_done": False,
        "nsga2_constraint_domination_done": False,
    }

    project_root = _project_root()
    output_dir_input = Path(config.output_dir)
    output_dir = output_dir_input
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    prefer_absolute_artifact_paths = output_dir_input.is_absolute()

    suffix = config.artifact_suffix
    json_path = output_dir / f"constrained_sphere_smoke_results_{suffix}.json"
    csv_path = output_dir / f"constrained_sphere_smoke_results_{suffix}.csv"
    md_results_path = output_dir / f"constrained_sphere_smoke_results_{suffix}.md"
    report_path = output_dir / f"constrained_sphere_smoke_report_{suffix}.md"
    fairness_path = output_dir / f"constrained_sphere_smoke_fairness_report_{suffix}.md"

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
            "feasible_rate",
            "feasible_count",
            "infeasible_count",
            "best_feasible_objective",
            "best_infeasible_total_violation",
            "mean_total_violation",
            "median_total_violation",
            "min_total_violation",
            "max_total_violation",
            "mean_max_violation",
            "violation_count_total",
            "requested_budget",
            "actual_evaluations",
            "runtime_seconds",
            "failure_count",
            "all_feasible",
            "all_infeasible",
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
            config=config,
            artifact=artifact,
            summaries=summaries,
            fairness_payload=fairness_payload,
            failures=failures,
            warnings=warnings,
        ),
        encoding="utf-8",
    )

    return artifact


__all__ = [
    "SmokeConfig",
    "run_constrained_sphere_smoke",
]
