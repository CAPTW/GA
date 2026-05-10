from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ga_lab.algorithms.registry import build_algorithm_from_name
from ga_lab.config import GAConfig
from ga_lab.factory import build_operator_bundle
from ga_lab.problems.base import ProblemMetadata
from ga_lab.utils.seed import make_rng


@dataclass(slots=True)
class BaselineRunResult:
    problem: str
    algorithm: str
    seed: int
    budget: int
    best_fitness: float | None
    best_solution: list[float] | list[int] | None
    evaluations: int
    runtime_seconds: float
    success: bool
    error_message: str | None
    metadata: dict[str, Any]


class SphereProblem:
    name = "sphere"
    compatible_representations = ("real",)
    default_objective_directions = (False,)

    def fitness(self, genome: list[float]) -> float:
        value = float(sum(gene * gene for gene in genome))
        if not math.isfinite(value):
            raise ValueError(f"Sphere fitness must be finite, got {value!r}")
        return value

    def optimal_fitness(self, genome_length: int) -> float | None:
        return 0.0 if genome_length > 0 else None

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            min_genome_length=1,
            default_objective_directions=self.default_objective_directions,
        )


def _is_better(candidate: float, incumbent: float | None, *, maximize: bool) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent if maximize else candidate < incumbent


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _safe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def _safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _sphere_bounds(config: GAConfig) -> tuple[float, float]:
    options = config.representation_options or {}
    low = float(options.get("low", -5.0))
    high = float(options.get("high", 5.0))
    if low >= high:
        raise ValueError(f"Sphere bounds must satisfy low < high, got {(low, high)!r}")
    return low, high


def _sphere_sigma(config: GAConfig) -> float:
    options = config.mutation_options or {}
    sigma = float(options.get("sigma", 0.2))
    if sigma <= 0:
        raise ValueError(f"Sphere sigma must be positive, got {sigma!r}")
    return sigma


def _sphere_random_genome(config: GAConfig, seed: int) -> list[float]:
    low, high = _sphere_bounds(config)
    rng = make_rng(seed)
    return [rng.uniform(low, high) for _ in range(config.genome_length)]


def _sample_real_genome(
    *,
    genome_length: int,
    low: float,
    high: float,
    rng: Any,
) -> list[float]:
    return [float(rng.uniform(low, high)) for _ in range(genome_length)]


def run_sphere_random_search(
    config: GAConfig,
    *,
    seed: int,
    budget: int,
) -> BaselineRunResult:
    problem = SphereProblem()
    low, high = _sphere_bounds(config)
    rng = make_rng(seed)
    best_fitness: float | None = None
    best_solution: list[float] | None = None
    evaluations = 0
    start = time.perf_counter()
    error_message: str | None = None

    try:
        while evaluations < budget:
            genome = _sample_real_genome(
                genome_length=config.genome_length,
                low=low,
                high=high,
                rng=rng,
            )
            fitness = problem.fitness(genome)
            evaluations += 1
            if _is_better(fitness, best_fitness, maximize=config.maximize):
                best_fitness = fitness
                best_solution = list(genome)
    except Exception as exc:  # pragma: no cover - defensive error capture for reports
        error_message = str(exc)

    runtime_seconds = time.perf_counter() - start
    success = (
        best_fitness is not None
        and config.target_fitness is not None
        and best_fitness <= float(config.target_fitness)
    )
    return BaselineRunResult(
        problem="sphere",
        algorithm="random_search",
        seed=seed,
        budget=budget,
        best_fitness=best_fitness,
        best_solution=best_solution,
        evaluations=evaluations,
        runtime_seconds=runtime_seconds,
        success=success,
        error_message=error_message,
        metadata={
            "bounds": {"low": low, "high": high},
            "maximize": config.maximize,
            "target_fitness": config.target_fitness,
        },
    )


def run_sphere_hill_climb(
    config: GAConfig,
    *,
    seed: int,
    budget: int,
) -> BaselineRunResult:
    problem = SphereProblem()
    low, high = _sphere_bounds(config)
    sigma = _sphere_sigma(config)
    rng = make_rng(seed)
    best_fitness: float | None = None
    best_solution: list[float] | None = None
    evaluations = 0
    accepted_moves = 0
    start = time.perf_counter()
    error_message: str | None = None

    try:
        current = _sample_real_genome(
            genome_length=config.genome_length,
            low=low,
            high=high,
            rng=rng,
        )
        current_fitness = problem.fitness(current)
        evaluations += 1
        best_fitness = current_fitness
        best_solution = list(current)

        while evaluations < budget:
            candidate = list(current)
            index = int(rng.randrange(config.genome_length))
            candidate[index] = _clamp(
                float(candidate[index] + rng.gauss(0.0, sigma)),
                low,
                high,
            )
            candidate_fitness = problem.fitness(candidate)
            evaluations += 1
            if _is_better(candidate_fitness, current_fitness, maximize=config.maximize):
                current = candidate
                current_fitness = candidate_fitness
                accepted_moves += 1
            if _is_better(candidate_fitness, best_fitness, maximize=config.maximize):
                best_fitness = candidate_fitness
                best_solution = list(candidate)
    except Exception as exc:  # pragma: no cover - defensive error capture for reports
        error_message = str(exc)

    runtime_seconds = time.perf_counter() - start
    success = (
        best_fitness is not None
        and config.target_fitness is not None
        and best_fitness <= float(config.target_fitness)
    )
    return BaselineRunResult(
        problem="sphere",
        algorithm="hill_climb",
        seed=seed,
        budget=budget,
        best_fitness=best_fitness,
        best_solution=best_solution,
        evaluations=evaluations,
        runtime_seconds=runtime_seconds,
        success=success,
        error_message=error_message,
        metadata={
            "bounds": {"low": low, "high": high},
            "sigma": sigma,
            "accepted_moves": accepted_moves,
            "maximize": config.maximize,
            "target_fitness": config.target_fitness,
        },
    )


def run_sphere_ga(
    config: GAConfig,
    *,
    seed: int,
) -> BaselineRunResult:
    problem = SphereProblem()
    operators = build_operator_bundle(config)
    algorithm_fn = build_algorithm_from_name(config.algorithm)
    rng = make_rng(seed)
    start = time.perf_counter()
    summary, _history = algorithm_fn(
        config=config,
        problem=problem,
        selection_fn=operators.selection_fn,
        crossover_fn=operators.crossover_fn,
        mutation_fn=operators.mutation_fn,
        init_fn=operators.init_fn,
        rng=rng,
    )
    runtime_seconds = time.perf_counter() - start
    best_fitness = float(summary["best_fitness"])
    actual_evaluations = int(summary["actual_evaluations_used"])
    return BaselineRunResult(
        problem="sphere",
        algorithm="ga",
        seed=seed,
        budget=actual_evaluations,
        best_fitness=best_fitness,
        best_solution=None,
        evaluations=actual_evaluations,
        runtime_seconds=runtime_seconds,
        success=(
            summary.get("stop_reason") == "target_fitness_reached"
            or (
                config.target_fitness is not None
                and best_fitness <= float(config.target_fitness)
            )
        ),
        error_message=None,
        metadata={
            "stop_reason": summary.get("stop_reason"),
            "convergence_generation": summary.get("convergence_generation"),
            "configured_evaluation_budget": config.population_size * (config.generations + 2),
        },
    )


def sphere_result_to_row(result: BaselineRunResult) -> dict[str, Any]:
    return {
        "problem": result.problem,
        "label": "recommended_preset" if result.algorithm == "ga" else result.algorithm,
        "algorithm_name": "GA" if result.algorithm == "ga" else result.algorithm,
        "seed": result.seed,
        "configured_evaluation_budget": result.budget,
        "actual_evaluations": result.evaluations,
        "runtime_seconds": result.runtime_seconds,
        "success_to_target": result.success,
        "final_best_fitness": result.best_fitness,
        "error_message": result.error_message,
        "best_solution": result.best_solution,
        "metadata": result.metadata,
    }


def normalize_run_row(row: dict[str, Any]) -> dict[str, Any]:
    problem = str(row["problem"])
    label = str(row.get("label") or row.get("algorithm_name") or "unknown")
    display_name = {
        "recommended_preset": "GA",
        "random_search": "Random Search",
        "random_sampling": "Random Search",
        "hill_climb": "Hill Climbing",
        "greedy_local_search": "Greedy Local Search",
        "random_tours": "Random Search",
        "nearest_neighbor_2opt": "Nearest Neighbor + 2-opt",
    }.get(label, label)
    if problem == "onemax":
        primary_metric = row.get("final_best_fitness")
        primary_metric_name = "best_fitness"
        higher_is_better = True
        success = bool(row.get("success_to_target"))
    elif problem == "sphere":
        primary_metric = row.get("final_best_fitness")
        primary_metric_name = "best_fitness"
        higher_is_better = False
        success = bool(row.get("success_to_target"))
    elif problem == "knapsack":
        primary_metric = row.get("best_feasible_fitness")
        primary_metric_name = "best_feasible_fitness"
        higher_is_better = True
        success = row.get("best_feasible_fitness") is not None
    elif problem == "tsp":
        primary_metric = row.get("final_best_distance")
        primary_metric_name = "best_route_distance"
        higher_is_better = False
        success = isinstance(primary_metric, int | float) and math.isfinite(primary_metric)
    else:
        raise ValueError(f"Unsupported problem for normalized baseline report: {problem}")

    metric_value = float(primary_metric) if isinstance(primary_metric, int | float) else None
    score = None if metric_value is None else metric_value if higher_is_better else -metric_value
    return {
        "problem": problem,
        "label": label,
        "algorithm": display_name,
        "seed": int(row["seed"]),
        "budget": int(row.get("configured_evaluation_budget") or 0),
        "actual_evaluations": int(row.get("actual_evaluations") or 0),
        "runtime_seconds": float(row.get("runtime_seconds") or 0.0),
        "success": bool(success),
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": metric_value,
        "higher_is_better": higher_is_better,
        "score": score,
        "error_message": row.get("error_message"),
        "raw_row": row,
    }


def summarize_problem_algorithm(
    *,
    problem: str,
    algorithm: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [
        float(row["primary_metric_value"])
        for row in rows
        if isinstance(row.get("primary_metric_value"), int | float)
        and math.isfinite(float(row["primary_metric_value"]))
    ]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    success_count = sum(1 for row in rows if row["success"])
    return {
        "problem": problem,
        "algorithm": algorithm,
        "seeds": len(rows),
        "metric": rows[0]["primary_metric_name"] if rows else None,
        "mean_best": _safe_mean(metrics),
        "std": _safe_stdev(metrics),
        "median": _safe_median(metrics),
        "best": _safe_max(metrics) if rows and rows[0]["higher_is_better"] else _safe_min(metrics),
        "worst": _safe_min(metrics) if rows and rows[0]["higher_is_better"] else _safe_max(metrics),
        "success_rate": success_count / len(rows) if rows else None,
        "mean_runtime_seconds": _safe_mean(runtimes),
        "mean_actual_evaluations": _safe_mean(
            [float(row["actual_evaluations"]) for row in rows]
        ),
    }


def _cliffs_delta(left_scores: list[float], right_scores: list[float]) -> float | None:
    if not left_scores or not right_scores:
        return None
    wins = 0
    losses = 0
    for left in left_scores:
        for right in right_scores:
            if left > right:
                wins += 1
            elif left < right:
                losses += 1
    total = len(left_scores) * len(right_scores)
    if total == 0:
        return None
    return (wins - losses) / total


def summarize_paired_comparison(
    *,
    problem: str,
    ga_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ga_by_seed = {int(row["seed"]): row for row in ga_rows}
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    common_seeds = sorted(set(ga_by_seed) & set(baseline_by_seed))
    ga_wins = 0
    ties = 0
    ga_losses = 0
    improvements: list[float] = []
    ga_scores: list[float] = []
    baseline_scores: list[float] = []

    for seed in common_seeds:
        ga_row = ga_by_seed[seed]
        baseline_row = baseline_by_seed[seed]
        ga_metric = ga_row["primary_metric_value"]
        baseline_metric = baseline_row["primary_metric_value"]
        if ga_metric is None or baseline_metric is None:
            continue
        ga_score = float(ga_row["score"])
        baseline_score = float(baseline_row["score"])
        ga_scores.append(ga_score)
        baseline_scores.append(baseline_score)
        improvement = ga_score - baseline_score
        improvements.append(improvement)
        if improvement > 0:
            ga_wins += 1
        elif improvement < 0:
            ga_losses += 1
        else:
            ties += 1

    mean_delta = _safe_mean(improvements)
    median_delta = _safe_median(improvements)
    if ga_wins > ga_losses:
        interpretation = "GA 우세 경향"
    elif ga_losses > ga_wins:
        interpretation = "baseline 우세 또는 최소 동급"
    else:
        interpretation = "혼합 또는 동률 경향"

    return {
        "problem": problem,
        "baseline": baseline_rows[0]["algorithm"] if baseline_rows else None,
        "ga_win": ga_wins,
        "tie": ties,
        "ga_loss": ga_losses,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "cliffs_delta": _cliffs_delta(ga_scores, baseline_scores),
        "paired_seeds": common_seeds,
        "interpretation": interpretation,
    }


def build_results_markdown(
    *,
    generated_at: str,
    command: str,
    scope_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        "# Baseline Comparison Results",
        "",
        f"- Generated at: {generated_at}",
        f"- Command: `{command}`",
        "",
        "## Problem summaries",
        "",
        "| 문제 | 알고리즘 | seeds | mean best | std | median | best | worst | success rate | mean runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scope_rows:
        success_rate = row["success_rate"]
        lines.append(
            "| {problem} | {algorithm} | {seeds} | {mean_best:.4f} | {std:.4f} | {median:.4f} | {best:.4f} | {worst:.4f} | {success} | {runtime:.4f} |".format(
                problem=row["problem"],
                algorithm=row["algorithm"],
                seeds=row["seeds"],
                mean_best=row["mean_best"] or float("nan"),
                std=row["std"] or 0.0,
                median=row["median"] or float("nan"),
                best=row["best"] or float("nan"),
                worst=row["worst"] or float("nan"),
                success=f"{100.0 * success_rate:.1f}%" if success_rate is not None else "n/a",
                runtime=row["mean_runtime_seconds"] or 0.0,
            )
        )

    lines.extend(
        [
            "",
            "## Paired GA vs baseline summary",
            "",
            "| 문제 | baseline | GA win | tie | GA loss | 평균 차이 | 중앙값 차이 | 해석 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {problem} | {baseline} | {ga_win} | {tie} | {ga_loss} | {mean_delta:.4f} | {median_delta:.4f} | {interpretation} |".format(
                problem=row["problem"],
                baseline=row["baseline"],
                ga_win=row["ga_win"],
                tie=row["tie"],
                ga_loss=row["ga_loss"],
                mean_delta=row["mean_delta"] or 0.0,
                median_delta=row["median_delta"] or 0.0,
                interpretation=row["interpretation"],
            )
        )

    lines.extend(["", "## Failures and exceptions", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- `{failure['problem']}` / `{failure['algorithm']}` / seed `{failure['seed']}`: {failure['error_message']}"
            )
    else:
        lines.append("- 없음")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column) for column in columns} for row in rows])


def serialize_baseline_result(result: BaselineRunResult) -> dict[str, Any]:
    return asdict(result)
