from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SERVICES_DIR = PROJECT_ROOT / "services"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from ga_lab.algorithms.registry import build_algorithm_from_name
from ga_lab.config import GAConfig
from ga_lab.factory import build_operator_bundle
from ga_lab.problems.base import ProblemMetadata
from ga_lab.runner import run_experiment
from ga_lab.utils.seed import make_rng


ARTIFACT_JSON = PROJECT_ROOT / "artifacts" / "ga_audit_execution_results.json"
ARTIFACT_MD = PROJECT_ROOT / "artifacts" / "ga_audit_execution_results.md"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ga_audit"


@dataclass(slots=True)
class SeedRun:
    seed: int
    succeeded: bool
    metric_value: float | None
    metric_label: str
    runtime_seconds: float | None
    evaluations: int | None
    summary_excerpt: dict[str, Any]
    error: str | None = None


class SphereProblem:
    name = "sphere"
    compatible_representations = ("real",)
    default_objective_directions = (False,)

    def fitness(self, genome: list[float]) -> float:
        return float(sum(gene * gene for gene in genome))

    def optimal_fitness(self, genome_length: int) -> float | None:
        return 0.0 if genome_length > 0 else None

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            min_genome_length=1,
            default_objective_directions=self.default_objective_directions,
        )

    def solution_metrics(self, genome: list[float]) -> dict[str, float]:
        value = self.fitness(genome)
        return {
            "sphere_value": value,
            "max_abs_gene": max(abs(gene) for gene in genome) if genome else 0.0,
        }


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _serialize_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _run_registered(config: GAConfig) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    result = run_experiment(config, output_root=OUTPUT_ROOT)
    elapsed = time.perf_counter() - start
    summary = dict(result.summary)
    summary["output_dir"] = str(result.output_dir)
    return summary, elapsed


def _run_custom_single_objective(config: GAConfig, problem: object) -> tuple[dict[str, Any], float]:
    operators = build_operator_bundle(config)
    algorithm_fn = build_algorithm_from_name(config.algorithm)
    rng = make_rng(config.seed)
    start = time.perf_counter()
    summary, history = algorithm_fn(
        config=config,
        problem=problem,
        selection_fn=operators.selection_fn,
        crossover_fn=operators.crossover_fn,
        mutation_fn=operators.mutation_fn,
        init_fn=operators.init_fn,
        rng=rng,
    )
    elapsed = time.perf_counter() - start
    return {
        **summary,
        "history_rows": len(history),
    }, elapsed


def _seed_run_from_summary(
    *,
    seed: int,
    metric_label: str,
    metric_value: float | None,
    success: bool,
    runtime_seconds: float,
    evaluations: int | None,
    summary: dict[str, Any],
) -> SeedRun:
    excerpt_keys = (
        "best_fitness",
        "best_route_distance",
        "best_total_value",
        "best_is_feasible",
        "hypervolume",
        "pareto_ratio",
        "spread",
        "stop_reason",
        "configured_evaluation_budget",
        "actual_evaluations_used",
    )
    return SeedRun(
        seed=seed,
        succeeded=success,
        metric_value=_serialize_float(metric_value),
        metric_label=metric_label,
        runtime_seconds=_serialize_float(runtime_seconds),
        evaluations=evaluations,
        summary_excerpt={key: summary.get(key) for key in excerpt_keys if key in summary},
    )


def _run_onemax() -> dict[str, Any]:
    seed_runs: list[SeedRun] = []
    base = GAConfig(
        run_name="audit_onemax",
        problem="onemax",
        population_size=40,
        genome_length=32,
        generations=40,
        crossover_rate=0.9,
        mutation_rate=0.02,
        elitism=1,
        tournament_size=3,
        maximize=True,
        target_fitness=32.0,
        log_every=5,
    )
    for seed in range(101, 111):
        config = GAConfig(**base.to_dict())
        config.seed = seed
        config.run_name = f"audit_onemax_seed{seed}"
        summary, elapsed = _run_registered(config)
        seed_runs.append(
            _seed_run_from_summary(
                seed=seed,
                metric_label="best_fitness",
                metric_value=float(summary["best_fitness"]),
                success=summary.get("stop_reason") == "target_fitness_reached",
                runtime_seconds=elapsed,
                evaluations=int(summary["actual_evaluations_used"]),
                summary=summary,
            )
        )
    return _aggregate_problem(
        problem="OneMax",
        representation="bit",
        run_type="registered",
        population_size=base.population_size,
        generations=base.generations,
        mutation_rate=base.mutation_rate,
        crossover_rate=base.crossover_rate,
        seed_runs=seed_runs,
        success_definition="best_fitness == 32 or target_fitness_reached",
    )


def _run_sphere() -> dict[str, Any]:
    seed_runs: list[SeedRun] = []
    base = GAConfig(
        run_name="audit_sphere",
        problem="sphere",
        algorithm="ga",
        representation="real",
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        population_size=32,
        genome_length=8,
        generations=40,
        crossover_rate=0.9,
        mutation_rate=0.12,
        elitism=1,
        tournament_size=3,
        maximize=False,
        target_fitness=1e-2,
        log_every=5,
        representation_options={"low": -5.0, "high": 5.0},
        mutation_options={"sigma": 0.2},
    )
    problem = SphereProblem()
    for seed in range(201, 211):
        config = GAConfig(**base.to_dict())
        config.seed = seed
        config.run_name = f"audit_sphere_seed{seed}"
        summary, elapsed = _run_custom_single_objective(config, problem)
        best = float(summary["best_fitness"])
        seed_runs.append(
            _seed_run_from_summary(
                seed=seed,
                metric_label="best_fitness",
                metric_value=best,
                success=best <= 1e-2,
                runtime_seconds=elapsed,
                evaluations=int(summary["actual_evaluations_used"]),
                summary=summary,
            )
        )
    return _aggregate_problem(
        problem="Sphere",
        representation="real",
        run_type="custom_single_objective",
        population_size=base.population_size,
        generations=base.generations,
        mutation_rate=base.mutation_rate,
        crossover_rate=base.crossover_rate,
        seed_runs=seed_runs,
        success_definition="best_fitness <= 1e-2",
    )


def _run_knapsack() -> dict[str, Any]:
    seed_runs: list[SeedRun] = []
    base = GAConfig(
        run_name="audit_knapsack",
        problem="knapsack",
        algorithm="ga",
        representation="bit",
        selection="tournament",
        crossover="one_point",
        mutation="bit_flip",
        population_size=40,
        genome_length=16,
        generations=30,
        crossover_rate=0.9,
        mutation_rate=0.05,
        elitism=1,
        tournament_size=3,
        maximize=True,
        log_every=5,
        problem_options={
            "num_items": 16,
            "seed": 9,
            "weight_scale": [1.0, 12.0],
            "value_scale": [1.0, 18.0],
            "capacity": 60.0,
            "penalty_factor": 20.0,
        },
    )
    for seed in range(301, 311):
        config = GAConfig(**base.to_dict())
        config.seed = seed
        config.run_name = f"audit_knapsack_seed{seed}"
        summary, elapsed = _run_registered(config)
        feasible = bool(summary.get("best_is_feasible"))
        metric = (
            float(summary["best_total_value"])
            if feasible and isinstance(summary.get("best_total_value"), int | float)
            else float(summary["best_fitness"])
        )
        seed_runs.append(
            _seed_run_from_summary(
                seed=seed,
                metric_label="best_total_value_if_feasible_else_best_fitness",
                metric_value=metric,
                success=feasible,
                runtime_seconds=elapsed,
                evaluations=int(summary["actual_evaluations_used"]),
                summary=summary,
            )
        )
    return _aggregate_problem(
        problem="Knapsack",
        representation="bit",
        run_type="registered",
        population_size=base.population_size,
        generations=base.generations,
        mutation_rate=base.mutation_rate,
        crossover_rate=base.crossover_rate,
        seed_runs=seed_runs,
        success_definition="best genome is feasible",
    )


def _run_tsp() -> dict[str, Any]:
    seed_runs: list[SeedRun] = []
    base = GAConfig(
        run_name="audit_tsp",
        problem="tsp",
        algorithm="ga",
        representation="permutation",
        selection="tournament",
        crossover="order",
        mutation="swap",
        population_size=30,
        genome_length=10,
        generations=30,
        crossover_rate=0.9,
        mutation_rate=0.25,
        elitism=1,
        tournament_size=3,
        maximize=True,
        log_every=5,
        problem_options={
            "num_cities": 10,
            "seed": 11,
            "bounds": [0.0, 100.0],
            "return_to_start": True,
        },
    )
    for seed in range(401, 411):
        config = GAConfig(**base.to_dict())
        config.seed = seed
        config.run_name = f"audit_tsp_seed{seed}"
        summary, elapsed = _run_registered(config)
        route_distance = float(summary["best_route_distance"])
        seed_runs.append(
            _seed_run_from_summary(
                seed=seed,
                metric_label="best_route_distance",
                metric_value=route_distance,
                success=math.isfinite(route_distance),
                runtime_seconds=elapsed,
                evaluations=int(summary["actual_evaluations_used"]),
                summary=summary,
            )
        )
    return _aggregate_problem(
        problem="TSP",
        representation="permutation",
        run_type="registered",
        population_size=base.population_size,
        generations=base.generations,
        mutation_rate=base.mutation_rate,
        crossover_rate=base.crossover_rate,
        seed_runs=seed_runs,
        success_definition="finite valid route produced",
    )


def _run_zdt1() -> dict[str, Any]:
    seed_runs: list[SeedRun] = []
    base = GAConfig(
        run_name="audit_zdt1",
        problem="zdt1",
        algorithm="nsga2",
        representation="real",
        representation_options={"low": 0.0, "high": 1.0},
        selection="tournament",
        crossover="arithmetic",
        mutation="gaussian",
        mutation_options={"sigma": 0.1},
        population_size=24,
        genome_length=8,
        generations=15,
        crossover_rate=0.9,
        mutation_rate=0.12,
        elitism=1,
        tournament_size=2,
        maximize=False,
        objective_directions=[False, False],
        log_every=5,
    )
    for seed in range(501, 506):
        config = GAConfig(**base.to_dict())
        config.seed = seed
        config.run_name = f"audit_zdt1_seed{seed}"
        summary, elapsed = _run_registered(config)
        hypervolume = summary.get("hypervolume")
        metric = float(hypervolume) if isinstance(hypervolume, int | float) else None
        seed_runs.append(
            _seed_run_from_summary(
                seed=seed,
                metric_label="hypervolume",
                metric_value=metric,
                success=metric is not None and math.isfinite(metric),
                runtime_seconds=elapsed,
                evaluations=int(summary["actual_evaluations_used"]),
                summary=summary,
            )
        )
    return _aggregate_problem(
        problem="ZDT1",
        representation="real_multiobjective",
        run_type="registered",
        population_size=base.population_size,
        generations=base.generations,
        mutation_rate=base.mutation_rate,
        crossover_rate=base.crossover_rate,
        seed_runs=seed_runs,
        success_definition="finite hypervolume produced",
    )


def _aggregate_problem(
    *,
    problem: str,
    representation: str,
    run_type: str,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    seed_runs: list[SeedRun],
    success_definition: str,
) -> dict[str, Any]:
    metric_values = [run.metric_value for run in seed_runs if run.metric_value is not None]
    runtimes = [run.runtime_seconds for run in seed_runs if run.runtime_seconds is not None]
    evaluations = [run.evaluations for run in seed_runs if run.evaluations is not None]
    return {
        "problem": problem,
        "representation": representation,
        "run_type": run_type,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "seed_count": len(seed_runs),
        "metric_label": seed_runs[0].metric_label if seed_runs else None,
        "success_definition": success_definition,
        "best_metric": min(metric_values) if problem in {"Sphere", "TSP"} else max(metric_values)
        if metric_values
        else None,
        "mean_metric": _safe_mean(metric_values),
        "std_metric": _safe_stdev(metric_values),
        "success_rate": (
            sum(1 for run in seed_runs if run.succeeded) / len(seed_runs) if seed_runs else None
        ),
        "mean_runtime_seconds": _safe_mean([value for value in runtimes if value is not None]),
        "mean_actual_evaluations": _safe_mean(
            [float(value) for value in evaluations if value is not None]
        ),
        "runs": [asdict(run) for run in seed_runs],
    }


def _write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# GA Audit Execution Results",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Output root: `{payload['output_root']}`",
        "",
        "| Problem | Seeds | Metric | Best | Mean | Std | Success Rate | Mean Runtime (s) | Mean Evals |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        lines.append(
            "| {problem} | {seed_count} | {metric_label} | {best_metric} | {mean_metric} | "
            "{std_metric} | {success_rate} | {mean_runtime_seconds} | {mean_actual_evaluations} |".format(
                problem=result["problem"],
                seed_count=result["seed_count"],
                metric_label=result["metric_label"],
                best_metric=_format_number(result["best_metric"]),
                mean_metric=_format_number(result["mean_metric"]),
                std_metric=_format_number(result["std_metric"]),
                success_rate=_format_rate(result["success_rate"]),
                mean_runtime_seconds=_format_number(result["mean_runtime_seconds"]),
                mean_actual_evaluations=_format_number(result["mean_actual_evaluations"]),
            )
        )
    lines.append("")
    for result in payload["results"]:
        lines.append(f"## {result['problem']}")
        lines.append("")
        lines.append(
            f"- Setup: pop={result['population_size']}, gen={result['generations']}, "
            f"cx={result['crossover_rate']}, mut={result['mutation_rate']}, "
            f"seeds={result['seed_count']}"
        )
        lines.append(f"- Success rule: {result['success_definition']}")
        lines.append("")
    ARTIFACT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)

    results = [
        _run_onemax(),
        _run_sphere(),
        _run_knapsack(),
        _run_tsp(),
        _run_zdt1(),
    ]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_root": str(OUTPUT_ROOT),
        "results": results,
    }
    ARTIFACT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
