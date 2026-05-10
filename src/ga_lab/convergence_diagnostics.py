from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from ga_lab.core.representation import Population

_EPSILON = 1e-12


@dataclass(slots=True)
class ProgressState:
    maximize: bool
    recent_window: int = 5
    best_so_far: float | None = None
    last_value: float | None = None
    last_improvement_generation: int = 0
    recent_values: deque[tuple[int, float]] = field(default_factory=deque)
    recent_signal_values: dict[str, deque[tuple[int, float]]] = field(default_factory=dict)
    last_signal_values: dict[str, float] = field(default_factory=dict)


def configured_evaluation_budget(config) -> int:
    if config.algorithm in {"ga", "hybrid_ga", "memetic_ga", "hybrid-ga", "memetic-ga"}:
        return config.population_size * (config.generations + 2)
    if config.algorithm == "nsga2":
        return config.population_size * ((3 * config.generations) + 2)
    raise ValueError(f"Unsupported algorithm for budget accounting: {config.algorithm}")


def progress_metric(
    problem_name: str,
    row: dict[str, Any],
    maximize: bool,
) -> tuple[str, float, bool]:
    if problem_name == "tsp":
        value = row.get("best_route_distance")
        if isinstance(value, int | float):
            return "best_route_distance", float(value), False
    if problem_name == "zdt1":
        value = row.get("hypervolume")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return "hypervolume", float(value), True
    value = row.get("best_fitness")
    if not isinstance(value, int | float):
        raise ValueError(
            "Row must contain a numeric best_fitness or problem-specific progress metric"
        )
    return "best_fitness", float(value), maximize


def update_progress_state(
    state: ProgressState,
    *,
    generation: int,
    value: float,
) -> dict[str, float | int]:
    previous_value = state.last_value
    if previous_value is None:
        improvement_delta = 0.0
    elif state.maximize:
        improvement_delta = value - previous_value
    else:
        improvement_delta = previous_value - value

    improved = (
        state.best_so_far is None
        or (
            value > state.best_so_far + _EPSILON
            if state.maximize
            else value < state.best_so_far - _EPSILON
        )
    )
    if improved:
        state.best_so_far = value
        state.last_improvement_generation = generation

    generations_since_last_improvement = generation - state.last_improvement_generation
    state.recent_values.append((generation, value))
    while len(state.recent_values) > max(2, state.recent_window + 1):
        state.recent_values.popleft()

    if len(state.recent_values) >= 2:
        window_generation, window_value = state.recent_values[0]
        if state.maximize:
            recent_window_improvement = value - window_value
        else:
            recent_window_improvement = window_value - value
        generation_span = max(1, generation - window_generation)
        recent_window_slope = recent_window_improvement / generation_span
    else:
        recent_window_improvement = 0.0
        recent_window_slope = 0.0

    state.last_value = value
    return {
        "improvement_delta": improvement_delta,
        "generations_since_last_improvement": generations_since_last_improvement,
        "recent_window_improvement": recent_window_improvement,
        "recent_window_slope": recent_window_slope,
    }


def update_signal_window(
    state: ProgressState,
    *,
    signal_key: str,
    output_prefix: str,
    generation: int,
    value: float,
) -> dict[str, float]:
    previous_value = state.last_signal_values.get(signal_key)
    delta = 0.0 if previous_value is None else value - previous_value
    history = state.recent_signal_values.setdefault(signal_key, deque())
    history.append((generation, value))
    while len(history) > max(2, state.recent_window + 1):
        history.popleft()
    if len(history) >= 2:
        window_generation, window_value = history[0]
        change = value - window_value
        generation_span = max(1, generation - window_generation)
        slope = change / generation_span
    else:
        change = 0.0
        slope = 0.0
    state.last_signal_values[signal_key] = value
    return {
        f"{output_prefix}_delta": delta,
        f"recent_{output_prefix}_change": change,
        f"recent_{output_prefix}_slope": slope,
    }


def _sample_pairs(size: int, max_pairs: int = 32) -> list[tuple[int, int]]:
    if size < 2 or max_pairs <= 0:
        return []
    pairs: list[tuple[int, int]] = []
    gap_stride = max(1, (size - 1) // max_pairs)
    for gap in range(1, size, gap_stride):
        for left in range(0, size - gap):
            pairs.append((left, left + gap))
            if len(pairs) >= max_pairs:
                return pairs
    for left in range(size - 1):
        for right in range(left + 1, size):
            pair = (left, right)
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _bit_population(population: Population) -> list[list[int]]:
    return [[1 if float(gene) >= 0.5 else 0 for gene in genome] for genome in population]


def _bit_entropy(bits_population: list[list[int]]) -> float:
    if not bits_population:
        return 0.0
    locus_count = len(bits_population[0])
    if locus_count == 0:
        return 0.0
    entropies: list[float] = []
    population_size = len(bits_population)
    for locus in range(locus_count):
        ones = sum(bits[locus] for bits in bits_population)
        probability = ones / population_size
        if probability in {0.0, 1.0}:
            entropies.append(0.0)
            continue
        entropy = -(
            probability * math.log2(probability)
            + (1.0 - probability) * math.log2(1.0 - probability)
        )
        entropies.append(entropy)
    return mean(entropies)


def _sampled_mean_hamming(bits_population: list[list[int]]) -> float:
    if len(bits_population) < 2:
        return 0.0
    length = len(bits_population[0])
    if length == 0:
        return 0.0
    distances = []
    for left, right in _sample_pairs(len(bits_population)):
        mismatch = sum(
            1
            for a_value, b_value in zip(
                bits_population[left],
                bits_population[right],
                strict=True,
            )
            if a_value != b_value
        )
        distances.append(mismatch / length)
    return mean(distances) if distances else 0.0


def _permutation_population(problem, population: Population) -> list[list[int]]:
    if hasattr(problem, "decode_route"):
        return [problem.decode_route(genome) for genome in population]
    return [[int(round(gene)) for gene in genome] for genome in population]


def _sampled_mean_positional_distance(routes: list[list[int]]) -> float:
    if len(routes) < 2:
        return 0.0
    length = len(routes[0])
    if length == 0:
        return 0.0
    distances = []
    for left, right in _sample_pairs(len(routes)):
        mismatch = sum(
            1
            for a_value, b_value in zip(routes[left], routes[right], strict=True)
            if a_value != b_value
        )
        distances.append(mismatch / length)
    return mean(distances) if distances else 0.0


def _edge_diversity_ratio(routes: list[list[int]]) -> float:
    if not routes:
        return 0.0
    total_edges = 0
    unique_edges: set[tuple[int, int]] = set()
    for route in routes:
        if len(route) < 2:
            continue
        for index in range(len(route)):
            edge = tuple(sorted((route[index], route[(index + 1) % len(route)])))
            unique_edges.add(edge)
            total_edges += 1
    if total_edges == 0:
        return 0.0
    return len(unique_edges) / total_edges


def _real_mean_coordinate_variance(population: Population) -> float:
    if not population:
        return 0.0
    dimension = len(population[0])
    if dimension == 0:
        return 0.0
    means = [mean(genome[idx] for genome in population) for idx in range(dimension)]
    variances = []
    for idx in range(dimension):
        variances.append(
            mean((genome[idx] - means[idx]) ** 2 for genome in population)
        )
    return mean(variances) if variances else 0.0


def _real_population_spread(population: Population) -> float:
    if len(population) < 2:
        return 0.0
    distances: list[float] = []
    for left, right in _sample_pairs(len(population)):
        squared = 0.0
        for left_value, right_value in zip(population[left], population[right], strict=True):
            squared += (left_value - right_value) ** 2
        distances.append(math.sqrt(squared))
    return mean(distances) if distances else 0.0


def diversity_metrics(config, problem, population: Population) -> dict[str, float]:
    if config.representation == "bit":
        bits_population = _bit_population(population)
        return {
            "allele_entropy": _bit_entropy(bits_population),
            "sampled_mean_hamming_distance": _sampled_mean_hamming(bits_population),
            "diversity_signal": _bit_entropy(bits_population),
        }
    if config.representation == "permutation":
        routes = _permutation_population(problem, population)
        positional_diversity = _sampled_mean_positional_distance(routes)
        edge_diversity = _edge_diversity_ratio(routes)
        return {
            "positional_diversity": positional_diversity,
            "edge_diversity_ratio": edge_diversity,
            "diversity_signal": edge_diversity,
        }
    if config.representation == "real":
        coordinate_variance = _real_mean_coordinate_variance(population)
        population_spread = _real_population_spread(population)
        return {
            "mean_coordinate_variance": coordinate_variance,
            "population_spread": population_spread,
            "diversity_signal": population_spread,
        }
    return {}


def problem_diagnostics(
    problem_name: str,
    problem,
    population: Population,
    row: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if problem_name == "knapsack" and hasattr(problem, "evaluate_selection"):
        decoded = [problem.decode_items(genome) for genome in population]
        selections = [problem.evaluate_selection(bits) for bits in decoded]
        feasible = [
            1.0 if selection["constraint_violation"] == 0.0 else 0.0
            for selection in selections
        ]
        violations = [float(selection["constraint_violation"]) for selection in selections]
        metrics["feasible_ratio"] = mean(feasible) if feasible else 0.0
        metrics["mean_constraint_violation"] = mean(violations) if violations else 0.0
    if problem_name == "tsp":
        best_fitness = row.get("best_fitness")
        if isinstance(best_fitness, int | float):
            metrics["best_route_distance"] = -float(best_fitness)
    if problem_name == "zdt1":
        metrics["front_size"] = row.get("pareto_front_size")
    return metrics


def build_generation_diagnostics(
    *,
    config,
    problem,
    population: Population,
    scalar_values: list[float],
    row: dict[str, Any],
    progress_state: ProgressState,
    generation: int,
    configured_budget_value: int,
    actual_evaluations_used: int,
    extra_evaluations_from_adaptation: int,
    adaptive_mutation_rate: float,
    adaptive_policy_name: str,
    adaptive_event: str,
    adaptive_event_count: int,
) -> dict[str, Any]:
    metric_name, progress_value, progress_maximize = progress_metric(
        config.problem,
        row,
        config.maximize,
    )
    if progress_state.maximize != progress_maximize:
        progress_state.maximize = progress_maximize
    diagnostics = {
        "median_fitness": median(scalar_values) if scalar_values else 0.0,
        "progress_metric_name": metric_name,
        "configured_budget": configured_budget_value,
        "actual_evaluations_used": actual_evaluations_used,
        "extra_evaluations_from_adaptation": extra_evaluations_from_adaptation,
        "adaptive_mutation_rate": adaptive_mutation_rate,
        "adaptive_policy": adaptive_policy_name,
        "adaptive_event": adaptive_event,
        "adaptive_event_count": adaptive_event_count,
    }
    diagnostics.update(
        update_progress_state(
            progress_state,
            generation=generation,
            value=progress_value,
        )
    )
    representation_metrics = diversity_metrics(config, problem, population)
    diagnostics.update(representation_metrics)
    if isinstance(representation_metrics.get("diversity_signal"), int | float):
        diagnostics.update(
            update_signal_window(
                progress_state,
                signal_key="diversity_signal",
                output_prefix="diversity",
                generation=generation,
                value=float(representation_metrics["diversity_signal"]),
            )
        )
    problem_metrics = problem_diagnostics(config.problem, problem, population, row)
    diagnostics.update(problem_metrics)
    if isinstance(problem_metrics.get("feasible_ratio"), int | float):
        diagnostics.update(
            update_signal_window(
                progress_state,
                signal_key="feasible_ratio",
                output_prefix="feasible_ratio",
                generation=generation,
                value=float(problem_metrics["feasible_ratio"]),
            )
        )
    if isinstance(problem_metrics.get("mean_constraint_violation"), int | float):
        diagnostics.update(
            update_signal_window(
                progress_state,
                signal_key="mean_constraint_violation",
                output_prefix="constraint_violation",
                generation=generation,
                value=float(problem_metrics["mean_constraint_violation"]),
            )
        )
    return diagnostics
