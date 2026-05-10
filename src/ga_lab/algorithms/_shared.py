from __future__ import annotations

import math
from statistics import mean
from typing import Any

from ga_lab.config import GAConfig
from ga_lab.core.mutation import MutationFn
from ga_lab.core.representation import Genome, Population
from ga_lab.metrics import resolve_reference_point
from ga_lab.problems.base import problem_metadata


def select_log_generation(generation: int, step: int) -> bool:
    return generation % step == 0


def _genome_preview(genome: Genome, *, max_items: int = 8) -> str:
    preview = genome[:max_items]
    suffix = "..." if len(genome) > max_items else ""
    return f"{preview!r}{suffix}"


def validate_fitness_vector(
    values: list[float],
    *,
    problem_name: str,
    genome: Genome,
    location: str,
    generation: int | None = None,
    evaluation_index: int | None = None,
) -> None:
    for objective_index, value in enumerate(values):
        if math.isfinite(value):
            continue
        generation_label = generation if generation is not None else "unknown"
        evaluation_label = evaluation_index if evaluation_index is not None else "unknown"
        raise ValueError(
            "Non-finite fitness detected "
            f"(problem={problem_name}, location={location}, generation={generation_label}, "
            f"evaluation_index={evaluation_label}, objective_index={objective_index}, "
            f"value={value!r}, genome={_genome_preview(genome)})"
        )


def apply_mutation_with_rate(
    config: GAConfig,
    mutation_fn: MutationFn,
    genome: Genome,
    rng,
    *,
    mutation_rate: float,
) -> Genome:
    original_rate = config.mutation_rate
    if mutation_rate == original_rate:
        return mutation_fn(genome, rng)
    config.mutation_rate = mutation_rate
    try:
        return mutation_fn(genome, rng)
    finally:
        config.mutation_rate = original_rate


def _option_int(options: dict[str, Any], key: str, default: int) -> int:
    value = options.get(key, default)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def _option_float(options: dict[str, Any], key: str, default: float) -> float:
    value = options.get(key, default)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def early_stop_decision(
    *,
    generation: int,
    row: dict[str, Any],
    options: dict[str, Any],
) -> tuple[bool, str]:
    policy = options.get("early_stop_policy", "none")
    if not isinstance(policy, str):
        return False, "none"
    normalized_policy = policy.strip() or "none"
    if normalized_policy == "none":
        return False, normalized_policy

    min_generation = max(0, _option_int(options, "early_stop_min_generation", 0))
    if generation < min_generation:
        return False, normalized_policy

    if normalized_policy == "progress_plateau":
        plateau_window = max(1, _option_int(options, "early_stop_window", 10))
        plateau_epsilon = _option_float(options, "early_stop_epsilon", 0.0)
        stagnation = row.get("generations_since_last_improvement")
        improvement = row.get("recent_window_improvement")
        if (
            isinstance(stagnation, int | float)
            and not isinstance(stagnation, bool)
            and isinstance(improvement, int | float)
            and not isinstance(improvement, bool)
            and float(stagnation) >= float(plateau_window)
            and float(improvement) <= plateau_epsilon
        ):
            return True, normalized_policy
        return False, normalized_policy

    raise ValueError(f"Unsupported early_stop_policy: {normalized_policy}")


def log_summary_row(
    generation: int,
    objective_vectors: list[list[float]],
    maximize: bool,
) -> dict[str, float]:
    scalar = [values[0] for values in objective_vectors]
    best_idx = (
        max(range(len(scalar)), key=scalar.__getitem__)
        if maximize
        else min(
            range(len(scalar)),
            key=scalar.__getitem__,
        )
    )
    worst_idx = (
        min(range(len(scalar)), key=scalar.__getitem__)
        if maximize
        else max(
            range(len(scalar)),
            key=scalar.__getitem__,
        )
    )
    best_objectives = objective_vectors[best_idx]
    row: dict[str, float] = {
        "generation": generation,
        "best_fitness": best_objectives[0],
        "mean_fitness": mean(scalar),
        "worst_fitness": scalar[worst_idx],
    }
    for idx, best_value in enumerate(best_objectives):
        row[f"best_objective_{idx}"] = best_value
    return row


def problem_solution_metrics(problem, genome: Genome) -> dict[str, Any]:
    metrics_fn = getattr(problem, "solution_metrics", None)
    if not callable(metrics_fn):
        return {}
    payload = metrics_fn(genome)
    return dict(payload) if isinstance(payload, dict) else {}


def problem_population_metrics(problem, population: Population) -> dict[str, Any]:
    metrics_fn = getattr(problem, "population_metrics", None)
    if not callable(metrics_fn):
        return {}
    payload = metrics_fn(population)
    return dict(payload) if isinstance(payload, dict) else {}


def reference_point_candidates(
    config: GAConfig,
    problem,
    directions: list[bool],
) -> tuple[list[float] | None, list[float] | None]:
    config_reference = config.algorithm_options.get("hypervolume_reference_point")
    normalized_config = (
        [float(value) for value in config_reference] if isinstance(config_reference, list) else None
    )
    provider = getattr(problem, "hypervolume_reference_point", None)
    preset_reference = provider(directions) if callable(provider) else None
    normalized_preset = (
        [float(value) for value in preset_reference] if isinstance(preset_reference, list) else None
    )
    return normalized_preset, normalized_config


def resolve_algorithm_reference_point(
    config: GAConfig,
    problem,
    objective_vectors: list[list[float]],
    directions: list[bool],
) -> tuple[list[float], dict[str, Any]]:
    preset_reference, config_reference = reference_point_candidates(
        config,
        problem,
        directions,
    )
    if config_reference is not None:
        resolved = resolve_reference_point(
            objective_vectors,
            directions,
            explicit=config_reference,
        )
        source = "config_override"
    elif preset_reference is not None:
        resolved = resolve_reference_point(
            objective_vectors,
            directions,
            explicit=preset_reference,
        )
        source = "problem_preset"
    else:
        resolved = resolve_reference_point(objective_vectors, directions)
        source = "auto"
    return resolved, {
        "hypervolume_reference_point": resolved[:],
        "hypervolume_reference_point_preset": preset_reference[:] if preset_reference else None,
        "hypervolume_reference_point_override": config_reference[:] if config_reference else None,
        "hypervolume_reference_point_source": source,
    }


def decorate_history_with_convergence(
    history: list[dict[str, Any]],
    final_hypervolume: float | None,
) -> float | None:
    hypervolumes = [
        float(row["hypervolume"])
        for row in history
        if isinstance(row.get("hypervolume"), int | float)
        and math.isfinite(float(row["hypervolume"]))
    ]
    if isinstance(final_hypervolume, int | float) and math.isfinite(final_hypervolume):
        hypervolumes.append(float(final_hypervolume))
    if not hypervolumes:
        for row in history:
            row["normalized_hypervolume"] = math.nan
            row["convergence_speed"] = math.nan
        return None

    peak_hypervolume = max(hypervolumes)
    if peak_hypervolume <= 0.0:
        for row in history:
            row["normalized_hypervolume"] = math.nan
            row["convergence_speed"] = math.nan
        return None

    speeds: list[float] = []
    previous_generation: int | None = None
    previous_progress = 0.0
    running_best = 0.0
    for row in history:
        generation = int(row["generation"])
        hypervolume = row.get("hypervolume")
        if not (isinstance(hypervolume, int | float) and math.isfinite(float(hypervolume))):
            row["normalized_hypervolume"] = math.nan
            row["convergence_speed"] = math.nan
            continue
        running_best = max(running_best, float(hypervolume))
        progress = min(1.0, running_best / peak_hypervolume)
        row["normalized_hypervolume"] = progress
        if previous_generation is None:
            speed = 0.0
        else:
            delta_generation = max(1, generation - previous_generation)
            speed = max(0.0, progress - previous_progress) / delta_generation
        row["convergence_speed"] = speed
        speeds.append(speed)
        previous_generation = generation
        previous_progress = progress

    return mean(speeds) if speeds else None


def resolve_objective_directions(
    objective_count_: int,
    config: GAConfig,
    problem=None,
) -> list[bool]:
    if not config.objective_directions:
        if problem is not None:
            metadata = problem_metadata(problem)
            if len(metadata.default_objective_directions) == objective_count_:
                return [bool(value) for value in metadata.default_objective_directions]
        if objective_count_ == 1:
            return [config.maximize]
        return [True] * objective_count_
    if len(config.objective_directions) != objective_count_:
        raise ValueError("objective_directions length must match objective count")
    return [bool(value) for value in config.objective_directions]


def maximize_vectors(
    objective_vectors: list[list[float]],
    directions: list[bool],
) -> list[list[float]]:
    return [
        [
            value if direction else -value
            for value, direction in zip(values, directions, strict=True)
        ]
        for values in objective_vectors
    ]
