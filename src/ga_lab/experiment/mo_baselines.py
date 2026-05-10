from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from ga_lab.algorithms._shared import resolve_objective_directions, validate_fitness_vector
from ga_lab.config import GAConfig
from ga_lab.core.representation import build_representation_adapter
from ga_lab.experiment.mo_metrics import nondominated_vectors
from ga_lab.factory import build_runtime_context
from ga_lab.problems.base import as_fitness_vector
from ga_lab.utils.seed import make_rng


@dataclass(slots=True)
class MultiObjectiveBaselineResult:
    problem_name: str
    algorithm_name: str
    seed: int
    budget: int
    evaluations: int
    runtime_seconds: float
    success: bool
    error_message: str | None
    objective_vectors: list[list[float]]
    nondominated_objective_vectors: list[list[float]]
    nondominated_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_objective_vector(
    problem: Any,
    genome: list[float] | list[int],
    *,
    problem_name: str,
    location: str,
    evaluation_index: int,
) -> list[float]:
    values = list(as_fitness_vector(problem.fitness(genome)))
    validate_fitness_vector(
        values,
        problem_name=problem_name,
        genome=list(genome),
        location=location,
        evaluation_index=evaluation_index,
    )
    return [float(value) for value in values]


def _default_weight_vectors(count: int = 11) -> list[list[float]]:
    if count < 2:
        raise ValueError("count must be at least 2")
    return [
        [index / float(count - 1), 1.0 - (index / float(count - 1))]
        for index in range(count)
    ]


def _scalarized_min_score(
    values: list[float],
    *,
    directions: list[bool],
    weights: list[float],
) -> float:
    if len(values) != len(weights):
        raise ValueError("weights length must match objective vector length")
    adjusted = [-value if directions[index] else value for index, value in enumerate(values)]
    return sum(
        weight * adjusted_value
        for weight, adjusted_value in zip(weights, adjusted, strict=True)
    )


def run_random_pareto_archive(
    config: GAConfig,
    *,
    seed: int,
    budget: int,
) -> MultiObjectiveBaselineResult:
    runtime = build_runtime_context(config)
    adapter = build_representation_adapter(config)
    rng = make_rng(seed)
    genomes: list[list[float] | list[int]] = []
    objective_vectors: list[list[float]] = []
    directions: list[bool] | None = None
    error_message: str | None = None
    started = time.perf_counter()

    try:
        for evaluation_index in range(budget):
            genome = adapter.initialize(rng, config.genome_length)
            vector = evaluate_objective_vector(
                runtime.problem,
                genome,
                problem_name=config.problem,
                location="random_pareto_archive",
                evaluation_index=evaluation_index,
            )
            if directions is None:
                directions = resolve_objective_directions(len(vector), config, runtime.problem)
            genomes.append(list(genome))
            objective_vectors.append(vector)
    except Exception as exc:  # pragma: no cover - defensive capture for reports
        error_message = str(exc)

    runtime_seconds = time.perf_counter() - started
    resolved_directions = directions or list(config.objective_directions)
    nondominated = (
        nondominated_vectors(objective_vectors, resolved_directions) if resolved_directions else []
    )
    nondominated_genomes: list[list[float] | list[int]] = []
    for index, vector in enumerate(objective_vectors):
        if vector in nondominated:
            nondominated_genomes.append(list(genomes[index]))
    return MultiObjectiveBaselineResult(
        problem_name=config.problem,
        algorithm_name="random_pareto_archive",
        seed=seed,
        budget=budget,
        evaluations=len(objective_vectors),
        runtime_seconds=runtime_seconds,
        success=error_message is None,
        error_message=error_message,
        objective_vectors=objective_vectors,
        nondominated_objective_vectors=nondominated,
        nondominated_count=len(nondominated),
        metadata={
            "objective_directions": resolved_directions,
            "representation_options": dict(config.representation_options),
            "decision_vectors": genomes,
            "front_decision_vectors": nondominated_genomes,
        },
    )


def run_weighted_sum_random_archive(
    config: GAConfig,
    *,
    seed: int,
    budget: int,
    weights: list[list[float]] | None = None,
) -> MultiObjectiveBaselineResult:
    runtime = build_runtime_context(config)
    adapter = build_representation_adapter(config)
    rng = make_rng(seed)
    weight_vectors = weights or _default_weight_vectors()
    selected_vectors: list[list[float]] = []
    directions: list[bool] | None = None
    error_message: str | None = None
    started = time.perf_counter()
    evaluations = 0

    try:
        base_budget = budget // len(weight_vectors)
        remainder = budget % len(weight_vectors)
        for weight_index, weight_vector in enumerate(weight_vectors):
            local_budget = base_budget + (1 if weight_index < remainder else 0)
            best_score: float | None = None
            best_vector: list[float] | None = None
            for _ in range(local_budget):
                genome = adapter.initialize(rng, config.genome_length)
                vector = evaluate_objective_vector(
                    runtime.problem,
                    genome,
                    problem_name=config.problem,
                    location="weighted_sum_random_archive",
                    evaluation_index=evaluations,
                )
                evaluations += 1
                if directions is None:
                    directions = resolve_objective_directions(len(vector), config, runtime.problem)
                score = _scalarized_min_score(
                    vector,
                    directions=directions,
                    weights=weight_vector,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_vector = vector
            if best_vector is not None:
                selected_vectors.append(best_vector)
    except Exception as exc:  # pragma: no cover - defensive capture for reports
        error_message = str(exc)

    runtime_seconds = time.perf_counter() - started
    resolved_directions = directions or list(config.objective_directions)
    nondominated = (
        nondominated_vectors(selected_vectors, resolved_directions) if resolved_directions else []
    )
    return MultiObjectiveBaselineResult(
        problem_name=config.problem,
        algorithm_name="weighted_sum_random",
        seed=seed,
        budget=budget,
        evaluations=evaluations,
        runtime_seconds=runtime_seconds,
        success=error_message is None,
        error_message=error_message,
        objective_vectors=selected_vectors,
        nondominated_objective_vectors=nondominated,
        nondominated_count=len(nondominated),
        metadata={
            "objective_directions": resolved_directions,
            "weight_vectors": weight_vectors,
            "weight_count": len(weight_vectors),
            "evaluated_candidates": evaluations,
            "limitation": "weighted_sum_random is an auxiliary convex-front-biased baseline",
        },
    )
