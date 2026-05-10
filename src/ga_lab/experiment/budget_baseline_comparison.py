from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ga_lab.algorithms._shared import resolve_algorithm_reference_point, resolve_objective_directions
from ga_lab.config import GAConfig
from ga_lab.core.mutation import build_mutation_fn
from ga_lab.core.representation import Genome, RepresentationAdapter, build_representation_adapter
from ga_lab.factory import build_runtime_context
from ga_lab.governance.run_metadata import build_run_metadata, write_run_metadata
from ga_lab.metrics import finite_or_none, front_metrics
from ga_lab.problems.base import as_fitness_vector
from ga_lab.utils.seed import make_rng


PROJECT_ROOT = Path(__file__).resolve().parents[3]


VALIDATED_SCOPE: dict[str, dict[str, Any]] = {
    "onemax": {
        "size_key": "genome_length",
        "sizes": {32: "small", 64: "medium", 128: "large"},
        "baselines": ("random_search", "hill_climb"),
    },
    "knapsack": {
        "size_key": "problem_options.num_items",
        "sizes": {20: "small", 30: "medium", 80: "large"},
        "baselines": ("random_sampling", "greedy_local_search"),
    },
    "tsp": {
        "size_key": "problem_options.num_cities",
        "sizes": {10: "small", 20: "medium", 50: "large"},
        "baselines": ("random_tours", "nearest_neighbor_2opt"),
    },
    "zdt1": {
        "size_key": "genome_length",
        "sizes": {10: "small", 20: "medium", 50: "large"},
        "baselines": ("random_archive", "mutation_archive"),
    },
}

RUN_COLUMNS = (
    "suite_name",
    "suite_kind",
    "problem",
    "tier",
    "size",
    "size_key",
    "label",
    "family",
    "seed",
    "preset_path",
    "configured_evaluation_budget",
    "actual_evaluations",
    "runtime_seconds",
    "success_to_target",
    "evaluations_to_target",
    "generations_to_target",
    "final_best_fitness",
    "final_best_distance",
    "best_feasible_fitness",
    "feasible_rate",
    "mean_violation",
    "hypervolume",
    "pareto_ratio",
    "spread",
    "pareto_front_size",
)


@dataclass(slots=True)
class ComparisonEntry:
    suite_name: str
    suite_kind: str
    problem: str
    size: int
    preset_path: Path
    baselines: tuple[str, ...]
    seeds: int
    seed_start: int


class TrackedProblem:
    def __init__(self, inner: object, config: GAConfig) -> None:
        self._inner = inner
        self._config = config
        self.evaluation_count = 0
        self.first_target_evaluation: int | None = None
        self.knapsack_feasible_evaluations = 0
        self.knapsack_total_violation = 0.0
        self.knapsack_best_feasible_value: float | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def fitness(self, genome: Genome):
        value = self._inner.fitness(genome)
        self.evaluation_count += 1
        vector = as_fitness_vector(value)
        scalar = vector[0]
        if self._config.target_fitness is not None and self.first_target_evaluation is None:
            if (
                self._config.maximize
                and scalar >= self._config.target_fitness
                or (not self._config.maximize and scalar <= self._config.target_fitness)
            ):
                self.first_target_evaluation = self.evaluation_count

        if self._config.problem == "knapsack":
            bits = self._inner.decode_items(genome)
            selection = self._inner.evaluate_selection(bits)
            violation = float(selection["constraint_violation"])
            self.knapsack_total_violation += violation
            if violation == 0.0:
                self.knapsack_feasible_evaluations += 1
                feasible_value = float(selection["total_value"])
                if (
                    self.knapsack_best_feasible_value is None
                    or feasible_value > self.knapsack_best_feasible_value
                ):
                    self.knapsack_best_feasible_value = feasible_value
        return value

    def tracked_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "actual_evaluations": self.evaluation_count,
            "evaluations_to_target": self.first_target_evaluation,
        }
        if self._config.problem == "knapsack":
            if self.evaluation_count > 0:
                metrics["feasible_rate"] = (
                    self.knapsack_feasible_evaluations / self.evaluation_count
                )
                metrics["mean_violation"] = self.knapsack_total_violation / self.evaluation_count
            else:
                metrics["feasible_rate"] = None
                metrics["mean_violation"] = None
            metrics["best_feasible_fitness"] = self.knapsack_best_feasible_value
        return metrics


def configured_evaluation_budget(config: GAConfig) -> int:
    population = config.population_size
    generations = config.generations
    if config.algorithm in {"ga", "hybrid_ga", "memetic_ga", "hybrid-ga", "memetic-ga"}:
        return population * (generations + 2)
    if config.algorithm == "nsga2":
        return population * ((3 * generations) + 2)
    raise ValueError(f"Unsupported algorithm for budget comparison: {config.algorithm}")


def budget_formula_text(algorithm: str) -> str:
    if algorithm in {"ga", "hybrid_ga", "memetic_ga", "hybrid-ga", "memetic-ga"}:
        return "population_size * (generations + 2)"
    if algorithm == "nsga2":
        return "population_size * (3 * generations + 2)"
    raise ValueError(f"Unsupported algorithm for budget comparison: {algorithm}")


def _size_value(config: GAConfig) -> int:
    if config.problem in {"onemax", "zdt1"}:
        return config.genome_length
    if config.problem == "knapsack":
        return int(config.problem_options["num_items"])
    if config.problem == "tsp":
        return int(config.problem_options["num_cities"])
    raise ValueError(f"Unsupported problem: {config.problem}")


def _tier_name(problem: str, size: int) -> str:
    return str(VALIDATED_SCOPE[problem]["sizes"][size])


def _comparison_label(_problem: str, family: str) -> str:
    if family == "ga_preset":
        return "recommended_preset"
    return family


def _json_value(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(
            [{column: _json_value(row.get(column)) for column in columns} for row in rows]
        )


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and math.isfinite(value):
            values.append(float(value))
    return values


def _mean_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    if not values:
        return None
    return mean(values)


def _stdev_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return stdev(values)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), bool)]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _group_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["suite_name"],
            row["suite_kind"],
            row["problem"],
            row["tier"],
            row["size"],
            row["label"],
            row["family"],
            row["preset_path"],
            row["configured_evaluation_budget"],
        )
        grouped.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        sample = bucket[0]
        aggregate_rows.append(
            {
                "suite_name": sample["suite_name"],
                "suite_kind": sample["suite_kind"],
                "problem": sample["problem"],
                "tier": sample["tier"],
                "size": sample["size"],
                "size_key": sample["size_key"],
                "label": sample["label"],
                "family": sample["family"],
                "preset_path": sample["preset_path"],
                "configured_evaluation_budget": sample["configured_evaluation_budget"],
                "run_count": len(bucket),
                "mean_actual_evaluations": _mean_or_none(bucket, "actual_evaluations"),
                "mean_runtime_seconds": _mean_or_none(bucket, "runtime_seconds"),
                "stdev_runtime_seconds": _stdev_or_none(bucket, "runtime_seconds"),
                "success_rate": _rate(bucket, "success_to_target"),
                "mean_evaluations_to_target": _mean_or_none(bucket, "evaluations_to_target"),
                "mean_generations_to_target": _mean_or_none(bucket, "generations_to_target"),
                "mean_final_best_fitness": _mean_or_none(bucket, "final_best_fitness"),
                "stdev_final_best_fitness": _stdev_or_none(bucket, "final_best_fitness"),
                "mean_final_best_distance": _mean_or_none(bucket, "final_best_distance"),
                "stdev_final_best_distance": _stdev_or_none(bucket, "final_best_distance"),
                "mean_best_feasible_fitness": _mean_or_none(bucket, "best_feasible_fitness"),
                "stdev_best_feasible_fitness": _stdev_or_none(bucket, "best_feasible_fitness"),
                "mean_feasible_rate": _mean_or_none(bucket, "feasible_rate"),
                "mean_violation": _mean_or_none(bucket, "mean_violation"),
                "mean_hypervolume": _mean_or_none(bucket, "hypervolume"),
                "stdev_hypervolume": _stdev_or_none(bucket, "hypervolume"),
                "mean_pareto_ratio": _mean_or_none(bucket, "pareto_ratio"),
                "mean_spread": _mean_or_none(bucket, "spread"),
                "mean_pareto_front_size": _mean_or_none(bucket, "pareto_front_size"),
            }
        )
    return aggregate_rows


def _lookup_aggregate(
    aggregates: list[dict[str, Any]],
    suite_kind: str,
    problem: str,
    size: int,
    label: str,
) -> dict[str, Any] | None:
    for row in aggregates:
        if (
            row["suite_kind"] == suite_kind
            and row["problem"] == problem
            and row["size"] == size
            and row["label"] == label
        ):
            return row
    return None


def _claim_ranges(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    claim_ranges: dict[str, Any] = {}

    onemax_random_deltas = []
    for size in VALIDATED_SCOPE["onemax"]["sizes"]:
        ga = _lookup_aggregate(aggregates, "comparison", "onemax", size, "recommended_preset")
        random_row = _lookup_aggregate(aggregates, "comparison", "onemax", size, "random_search")
        if ga and random_row and ga["success_rate"] is not None and random_row["success_rate"] is not None:
            onemax_random_deltas.append(float(ga["success_rate"]) - float(random_row["success_rate"]))
    if onemax_random_deltas:
        claim_ranges["onemax_success_vs_random"] = {
            "min_delta": min(onemax_random_deltas),
            "max_delta": max(onemax_random_deltas),
        }

    tsp_random_improvements = []
    tsp_nn_improvements = []
    for size in VALIDATED_SCOPE["tsp"]["sizes"]:
        ga = _lookup_aggregate(aggregates, "comparison", "tsp", size, "recommended_preset")
        random_row = _lookup_aggregate(aggregates, "comparison", "tsp", size, "random_tours")
        nn_row = _lookup_aggregate(
            aggregates,
            "comparison",
            "tsp",
            size,
            "nearest_neighbor_2opt",
        )
        ga_distance = ga["mean_final_best_distance"] if ga else None
        random_distance = random_row["mean_final_best_distance"] if random_row else None
        nn_distance = nn_row["mean_final_best_distance"] if nn_row else None
        if (
            isinstance(ga_distance, int | float)
            and isinstance(random_distance, int | float)
            and random_distance > 0
        ):
            tsp_random_improvements.append(
                100.0 * (float(random_distance) - float(ga_distance)) / float(random_distance)
            )
        if isinstance(ga_distance, int | float) and isinstance(nn_distance, int | float) and nn_distance > 0:
            tsp_nn_improvements.append(
                100.0 * (float(nn_distance) - float(ga_distance)) / float(nn_distance)
            )
    if tsp_random_improvements:
        claim_ranges["tsp_distance_vs_random_pct"] = {
            "min_pct": min(tsp_random_improvements),
            "max_pct": max(tsp_random_improvements),
        }
    if tsp_nn_improvements:
        claim_ranges["tsp_distance_vs_nearest_neighbor_pct"] = {
            "min_pct": min(tsp_nn_improvements),
            "max_pct": max(tsp_nn_improvements),
        }

    zdt1_random_hv_delta = []
    for size in VALIDATED_SCOPE["zdt1"]["sizes"]:
        ga = _lookup_aggregate(aggregates, "comparison", "zdt1", size, "recommended_preset")
        random_row = _lookup_aggregate(aggregates, "comparison", "zdt1", size, "random_archive")
        if ga and random_row:
            ga_hv = ga["mean_hypervolume"]
            random_hv = random_row["mean_hypervolume"]
            if isinstance(ga_hv, int | float) and isinstance(random_hv, int | float):
                zdt1_random_hv_delta.append(float(ga_hv) - float(random_hv))
    if zdt1_random_hv_delta:
        claim_ranges["zdt1_hv_vs_random_archive"] = {
            "min_delta": min(zdt1_random_hv_delta),
            "max_delta": max(zdt1_random_hv_delta),
        }

    return claim_ranges


def _suite_problem_size_rows(
    aggregates: list[dict[str, Any]],
    suite_kind: str,
    problem: str,
    size: int,
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in aggregates
            if row["suite_kind"] == suite_kind and row["problem"] == problem and row["size"] == size
        ],
        key=lambda row: str(row["label"]),
    )


def _problem_size_interpretations(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interpretations: list[dict[str, Any]] = []
    for problem, scope in VALIDATED_SCOPE.items():
        for size, tier in scope["sizes"].items():
            rows = _suite_problem_size_rows(aggregates, "comparison", problem, size)
            if not rows:
                continue
            label_index = {row["label"]: row for row in rows}
            ga = label_index.get("recommended_preset")
            note = "evidence collected"
            outcome = "mixed"
            if ga is not None:
                if problem == "onemax":
                    random_row = label_index.get("random_search")
                    hill_row = label_index.get("hill_climb")
                    ga_success = ga.get("success_rate")
                    hill_success = hill_row.get("success_rate") if hill_row else None
                    random_success = random_row.get("success_rate") if random_row else None
                    ga_eval = ga.get("mean_evaluations_to_target")
                    hill_eval = hill_row.get("mean_evaluations_to_target") if hill_row else None
                    if (
                        isinstance(ga_success, float)
                        and isinstance(random_success, float)
                        and ga_success > random_success
                    ):
                        note = "GA beats random search on target reliability, but hill climbing remains the stronger cheap baseline on this unimodal problem."
                    if (
                        isinstance(ga_success, float)
                        and isinstance(hill_success, float)
                        and ga_success > hill_success
                    ):
                        outcome = "ga_clear_win"
                    elif (
                        isinstance(ga_success, float)
                        and isinstance(hill_success, float)
                        and isinstance(ga_eval, float)
                        and isinstance(hill_eval, float)
                        and ga_success == hill_success
                        and hill_eval < ga_eval
                    ):
                        outcome = "cheap_baseline_competitive"
                    else:
                        outcome = "cheap_baseline_competitive"
                elif problem == "knapsack":
                    greedy_row = label_index.get("greedy_local_search")
                    ga_best = ga.get("mean_best_feasible_fitness")
                    greedy_best = greedy_row.get("mean_best_feasible_fitness") if greedy_row else None
                    if (
                        isinstance(ga_best, float)
                        and isinstance(greedy_best, float)
                        and ga_best > greedy_best * 1.01
                    ):
                        outcome = "ga_clear_win"
                        note = "GA leads greedy local search on feasible-value quality at matched budgets."
                    else:
                        outcome = "cheap_baseline_competitive"
                        note = "Greedy local search is already close enough that GA is not a universal default for value-only knapsack runs."
                elif problem == "tsp":
                    nn_row = label_index.get("nearest_neighbor_2opt")
                    ga_distance = ga.get("mean_final_best_distance")
                    nn_distance = nn_row.get("mean_final_best_distance") if nn_row else None
                    if (
                        isinstance(ga_distance, float)
                        and isinstance(nn_distance, float)
                        and ga_distance < nn_distance * 0.98
                    ):
                        outcome = "ga_clear_win"
                        note = "GA clearly improves route distance over both simple baselines."
                    else:
                        outcome = "cheap_baseline_competitive"
                        note = "Nearest-neighbor plus lightweight 2-opt is already competitive, especially at smaller validated sizes."
                elif problem == "zdt1":
                    random_row = label_index.get("random_archive")
                    mutation_row = label_index.get("mutation_archive")
                    ga_hv = ga.get("mean_hypervolume")
                    random_hv = random_row.get("mean_hypervolume") if random_row else None
                    mutation_hv = mutation_row.get("mean_hypervolume") if mutation_row else None
                    if (
                        isinstance(ga_hv, float)
                        and isinstance(random_hv, float)
                        and isinstance(mutation_hv, float)
                        and ga_hv > random_hv
                        and ga_hv >= mutation_hv
                    ):
                        outcome = "ga_clear_win"
                        note = "NSGA-II leads the random archive baseline and is at least as strong as mutation-only Pareto search on hypervolume."
                    else:
                        outcome = "mixed"
                        note = "ZDT1 remains multi-metric: NSGA-II does not dominate every cheap archive baseline on every metric."
            interpretations.append(
                {
                    "problem": problem,
                    "size": size,
                    "tier": tier,
                    "outcome": outcome,
                    "note": note,
                }
            )
    return interpretations


def _dominates_adjusted(left: list[float], right: list[float]) -> bool:
    better_or_equal = True
    strictly_better = False
    for left_value, right_value in zip(left, right, strict=True):
        if left_value < right_value:
            better_or_equal = False
            break
        if left_value > right_value:
            strictly_better = True
    return better_or_equal and strictly_better


@dataclass(slots=True)
class ArchiveEntry:
    genome: Genome
    objectives: list[float]
    adjusted: list[float]


def _truncate_archive_evenly(
    archive: list[ArchiveEntry],
    capacity: int,
    *,
    maximize_first_objective: bool,
) -> list[ArchiveEntry]:
    if len(archive) <= capacity:
        return archive
    sorted_archive = sorted(
        archive,
        key=lambda entry: entry.objectives[0],
        reverse=maximize_first_objective,
    )
    if capacity <= 1:
        return [sorted_archive[0]]
    selected_indices = []
    last_index = len(sorted_archive) - 1
    for step in range(capacity):
        idx = round((step * last_index) / (capacity - 1))
        if idx not in selected_indices:
            selected_indices.append(idx)
    if len(selected_indices) < capacity:
        for idx in range(len(sorted_archive)):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) == capacity:
                break
    return [sorted_archive[idx] for idx in sorted(selected_indices[:capacity])]


def _update_archive(
    archive: list[ArchiveEntry],
    genome: Genome,
    objectives: list[float],
    directions: list[bool],
    *,
    capacity: int,
) -> list[ArchiveEntry]:
    adjusted = [
        value if direction else -value
        for value, direction in zip(objectives, directions, strict=True)
    ]
    for entry in archive:
        if entry.adjusted == adjusted:
            return archive
        if _dominates_adjusted(entry.adjusted, adjusted):
            return archive
    next_archive = [
        entry for entry in archive if not _dominates_adjusted(adjusted, entry.adjusted)
    ]
    next_archive.append(ArchiveEntry(genome=genome[:], objectives=objectives[:], adjusted=adjusted))
    return _truncate_archive_evenly(
        next_archive,
        capacity,
        maximize_first_objective=directions[0],
    )


def _random_bit_genome(adapter: RepresentationAdapter, rng, length: int) -> Genome:
    return adapter.initialize(rng, length)


def _repair_knapsack(bits: list[int], problem) -> list[int]:
    if sum(problem.weights[idx] for idx, selected in enumerate(bits) if selected) <= problem.capacity:
        return bits
    ratios = [
        (problem.values[idx] / problem.weights[idx], idx)
        for idx, selected in enumerate(bits)
        if selected and problem.weights[idx] > 0
    ]
    for _, idx in sorted(ratios):
        bits[idx] = 0
        if sum(problem.weights[item] for item, selected in enumerate(bits) if selected) <= problem.capacity:
            break
    return bits


def _nearest_neighbor_route(problem, start_city: int) -> list[int]:
    unvisited = set(range(problem.num_cities))
    route = [start_city]
    unvisited.remove(start_city)
    while unvisited:
        current = route[-1]
        next_city = min(unvisited, key=lambda city: problem._distance(current, city))
        route.append(next_city)
        unvisited.remove(next_city)
    return route


def _two_opt_move(route: list[int], rng) -> list[int]:
    if len(route) < 4:
        return route[:]
    start, end = sorted(rng.sample(range(len(route)), 2))
    if start == 0 and end == len(route) - 1:
        return route[:]
    candidate = route[:]
    candidate[start : end + 1] = reversed(candidate[start : end + 1])
    return candidate


def _run_onemax_random_search(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    best_fitness = float("-inf")
    for _ in range(budget):
        genome = _random_bit_genome(adapter, rng, config.genome_length)
        fitness = float(tracked_problem.fitness(genome))
        best_fitness = max(best_fitness, fitness)
    success = (
        tracked_problem.first_target_evaluation is not None
        if config.target_fitness is not None
        else None
    )
    return {
        "success_to_target": success,
        "final_best_fitness": best_fitness,
    }


def _run_onemax_hill_climb(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    current = _random_bit_genome(adapter, rng, config.genome_length)
    current_fitness = float(tracked_problem.fitness(current))
    best_fitness = current_fitness
    while tracked_problem.evaluation_count < budget:
        candidate = current[:]
        index = rng.randrange(config.genome_length)
        candidate[index] = 1.0 - candidate[index]
        candidate_fitness = float(tracked_problem.fitness(candidate))
        best_fitness = max(best_fitness, candidate_fitness)
        if candidate_fitness >= current_fitness:
            current = candidate
            current_fitness = candidate_fitness
    success = (
        tracked_problem.first_target_evaluation is not None
        if config.target_fitness is not None
        else None
    )
    return {
        "success_to_target": success,
        "final_best_fitness": best_fitness,
    }


def _run_knapsack_random_sampling(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    for _ in range(budget):
        genome = adapter.initialize(rng, config.genome_length)
        tracked_problem.fitness(genome)
    return {
        "best_feasible_fitness": tracked_problem.knapsack_best_feasible_value,
        "feasible_rate": tracked_problem.tracked_metrics().get("feasible_rate"),
        "mean_violation": tracked_problem.tracked_metrics().get("mean_violation"),
    }


def _run_knapsack_greedy_local_search(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    _adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    problem = tracked_problem._inner
    item_order = sorted(
        range(problem.num_items),
        key=lambda idx: problem.values[idx] / problem.weights[idx],
        reverse=True,
    )
    current_bits = [0] * problem.num_items
    current_weight = 0.0
    for idx in item_order:
        candidate_weight = current_weight + problem.weights[idx]
        if candidate_weight <= problem.capacity:
            current_bits[idx] = 1
            current_weight = candidate_weight
    current = [float(bit) for bit in current_bits]
    current_fitness = float(tracked_problem.fitness(current))
    current_value = problem.evaluate_selection(current_bits)["total_value"]

    while tracked_problem.evaluation_count < budget:
        candidate_bits = current_bits[:]
        idx = rng.randrange(problem.num_items)
        candidate_bits[idx] = 1 - candidate_bits[idx]
        candidate_bits = _repair_knapsack(candidate_bits, problem)
        candidate = [float(bit) for bit in candidate_bits]
        candidate_fitness = float(tracked_problem.fitness(candidate))
        candidate_value = problem.evaluate_selection(candidate_bits)["total_value"]
        if candidate_fitness > current_fitness or (
            candidate_fitness == current_fitness and candidate_value > current_value
        ):
            current_bits = candidate_bits
            current_fitness = candidate_fitness
            current_value = candidate_value

    tracked_metrics = tracked_problem.tracked_metrics()
    return {
        "best_feasible_fitness": tracked_metrics.get("best_feasible_fitness"),
        "feasible_rate": tracked_metrics.get("feasible_rate"),
        "mean_violation": tracked_metrics.get("mean_violation"),
    }


def _run_tsp_random_tours(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    best_distance = math.inf
    for _ in range(budget):
        genome = adapter.initialize(rng, config.genome_length)
        fitness = float(tracked_problem.fitness(genome))
        best_distance = min(best_distance, -fitness)
    return {
        "final_best_distance": best_distance,
        "final_best_fitness": -best_distance,
    }


def _run_tsp_nearest_neighbor_2opt(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    _adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    problem = tracked_problem._inner
    start_order = list(range(problem.num_cities))
    rng.shuffle(start_order)
    current_route: list[int] | None = None
    current_distance = math.inf
    for start_city in start_order:
        if tracked_problem.evaluation_count >= budget:
            break
        route = _nearest_neighbor_route(problem, start_city)
        distance = -float(tracked_problem.fitness([float(city) for city in route]))
        if distance < current_distance:
            current_route = route
            current_distance = distance
    if current_route is None:
        current_route = list(range(problem.num_cities))
        current_distance = problem.route_distance(current_route)
    best_distance = current_distance
    while tracked_problem.evaluation_count < budget:
        candidate_route = _two_opt_move(current_route, rng)
        candidate_distance = -float(tracked_problem.fitness([float(city) for city in candidate_route]))
        if candidate_distance <= current_distance:
            current_route = candidate_route
            current_distance = candidate_distance
        if candidate_distance < best_distance:
            best_distance = candidate_distance
    return {
        "final_best_distance": best_distance,
        "final_best_fitness": -best_distance,
    }


def _run_zdt1_random_archive(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    directions = resolve_objective_directions(2, config, tracked_problem)
    archive: list[ArchiveEntry] = []
    for _ in range(budget):
        genome = adapter.initialize(rng, config.genome_length)
        objectives = list(as_fitness_vector(tracked_problem.fitness(genome)))
        archive = _update_archive(
            archive,
            genome,
            objectives,
            directions,
            capacity=config.population_size,
        )
    objective_vectors = [entry.objectives[:] for entry in archive]
    if objective_vectors:
        reference_point, reference_metadata = resolve_algorithm_reference_point(
            config,
            tracked_problem,
            objective_vectors,
            directions,
        )
        metrics = front_metrics(
            list(range(len(objective_vectors))),
            objective_vectors,
            directions,
            reference_point,
            config.population_size,
        )
    else:
        reference_point, reference_metadata = resolve_algorithm_reference_point(
            config,
            tracked_problem,
            [[1.0, 1.0]],
            directions,
        )
        metrics = {
            "pareto_front_size": 0.0,
            "pareto_ratio": 0.0,
            "hypervolume": 0.0,
            "spread": math.nan,
        }
    return {
        "hypervolume": finite_or_none(float(metrics["hypervolume"])),
        "pareto_ratio": float(metrics["pareto_ratio"]),
        "spread": finite_or_none(float(metrics["spread"])),
        "pareto_front_size": int(metrics["pareto_front_size"]),
        "objective_directions": directions,
        "hypervolume_reference_point": reference_point,
        "hypervolume_reference_point_source": reference_metadata["hypervolume_reference_point_source"],
    }


def _run_zdt1_mutation_archive(
    config: GAConfig,
    tracked_problem: TrackedProblem,
    adapter: RepresentationAdapter,
    rng,
    budget: int,
) -> dict[str, Any]:
    directions = resolve_objective_directions(2, config, tracked_problem)
    mutation_fn = build_mutation_fn(config)
    archive: list[ArchiveEntry] = []
    warmup = min(max(8, config.population_size // 4), budget)
    for _ in range(warmup):
        genome = adapter.initialize(rng, config.genome_length)
        objectives = list(as_fitness_vector(tracked_problem.fitness(genome)))
        archive = _update_archive(
            archive,
            genome,
            objectives,
            directions,
            capacity=config.population_size,
        )

    while tracked_problem.evaluation_count < budget:
        if archive:
            parent = rng.choice(archive).genome[:]
        else:
            parent = adapter.initialize(rng, config.genome_length)
        child = adapter.repair(mutation_fn(parent, rng), config.genome_length)
        objectives = list(as_fitness_vector(tracked_problem.fitness(child)))
        archive = _update_archive(
            archive,
            child,
            objectives,
            directions,
            capacity=config.population_size,
        )

    objective_vectors = [entry.objectives[:] for entry in archive]
    if objective_vectors:
        reference_point, reference_metadata = resolve_algorithm_reference_point(
            config,
            tracked_problem,
            objective_vectors,
            directions,
        )
        metrics = front_metrics(
            list(range(len(objective_vectors))),
            objective_vectors,
            directions,
            reference_point,
            config.population_size,
        )
    else:
        reference_point, reference_metadata = resolve_algorithm_reference_point(
            config,
            tracked_problem,
            [[1.0, 1.0]],
            directions,
        )
        metrics = {
            "pareto_front_size": 0.0,
            "pareto_ratio": 0.0,
            "hypervolume": 0.0,
            "spread": math.nan,
        }
    return {
        "hypervolume": finite_or_none(float(metrics["hypervolume"])),
        "pareto_ratio": float(metrics["pareto_ratio"]),
        "spread": finite_or_none(float(metrics["spread"])),
        "pareto_front_size": int(metrics["pareto_front_size"]),
        "objective_directions": directions,
        "hypervolume_reference_point": reference_point,
        "hypervolume_reference_point_source": reference_metadata["hypervolume_reference_point_source"],
    }


def _run_baseline_trial(entry: ComparisonEntry, config: GAConfig, family: str, seed: int) -> dict[str, Any]:
    adapter = build_representation_adapter(config)
    problem = build_runtime_context(config).problem
    tracked_problem = TrackedProblem(problem, config)
    rng = make_rng(seed)
    budget = configured_evaluation_budget(config)

    runners = {
        ("onemax", "random_search"): _run_onemax_random_search,
        ("onemax", "hill_climb"): _run_onemax_hill_climb,
        ("knapsack", "random_sampling"): _run_knapsack_random_sampling,
        ("knapsack", "greedy_local_search"): _run_knapsack_greedy_local_search,
        ("tsp", "random_tours"): _run_tsp_random_tours,
        ("tsp", "nearest_neighbor_2opt"): _run_tsp_nearest_neighbor_2opt,
        ("zdt1", "random_archive"): _run_zdt1_random_archive,
        ("zdt1", "mutation_archive"): _run_zdt1_mutation_archive,
    }
    runner = runners[(config.problem, family)]
    started = time.perf_counter()
    payload = runner(config, tracked_problem, adapter, rng, budget)
    elapsed = time.perf_counter() - started
    tracked_metrics = tracked_problem.tracked_metrics()
    return {
        "suite_name": entry.suite_name,
        "suite_kind": entry.suite_kind,
        "problem": entry.problem,
        "tier": _tier_name(entry.problem, entry.size),
        "size": entry.size,
        "size_key": VALIDATED_SCOPE[entry.problem]["size_key"],
        "label": _comparison_label(entry.problem, family),
        "family": family,
        "seed": seed,
        "preset_path": str(entry.preset_path.as_posix()),
        "configured_evaluation_budget": budget,
        "actual_evaluations": tracked_metrics["actual_evaluations"],
        "runtime_seconds": elapsed,
        "success_to_target": payload.get("success_to_target"),
        "evaluations_to_target": tracked_metrics.get("evaluations_to_target"),
        "generations_to_target": None,
        "final_best_fitness": payload.get("final_best_fitness"),
        "final_best_distance": payload.get("final_best_distance"),
        "best_feasible_fitness": payload.get(
            "best_feasible_fitness",
            tracked_metrics.get("best_feasible_fitness"),
        ),
        "feasible_rate": payload.get("feasible_rate", tracked_metrics.get("feasible_rate")),
        "mean_violation": payload.get("mean_violation", tracked_metrics.get("mean_violation")),
        "hypervolume": payload.get("hypervolume"),
        "pareto_ratio": payload.get("pareto_ratio"),
        "spread": payload.get("spread"),
        "pareto_front_size": payload.get("pareto_front_size"),
    }


def _run_ga_trial(entry: ComparisonEntry, config: GAConfig, seed: int) -> dict[str, Any]:
    runtime = build_runtime_context(config)
    tracked_problem = TrackedProblem(runtime.problem, config)
    rng = make_rng(seed)
    started = time.perf_counter()
    algorithm_summary, _history = runtime.algorithm_fn(
        config=config,
        problem=tracked_problem,
        selection_fn=runtime.operators.selection_fn,
        crossover_fn=runtime.operators.crossover_fn,
        mutation_fn=runtime.operators.mutation_fn,
        init_fn=runtime.operators.init_fn,
        rng=rng,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "runtime_seconds": elapsed,
        **algorithm_summary,
    }
    tracked_metrics = tracked_problem.tracked_metrics()
    return {
        "suite_name": entry.suite_name,
        "suite_kind": entry.suite_kind,
        "problem": entry.problem,
        "tier": _tier_name(entry.problem, entry.size),
        "size": entry.size,
        "size_key": VALIDATED_SCOPE[entry.problem]["size_key"],
        "label": "recommended_preset",
        "family": "ga_preset",
        "seed": seed,
        "preset_path": str(entry.preset_path.as_posix()),
        "configured_evaluation_budget": configured_evaluation_budget(config),
        "actual_evaluations": tracked_metrics["actual_evaluations"],
        "runtime_seconds": elapsed,
        "success_to_target": (
            summary["stop_reason"] == "target_fitness_reached"
            if config.target_fitness is not None
            else None
        ),
        "evaluations_to_target": tracked_metrics.get("evaluations_to_target"),
        "generations_to_target": summary.get("convergence_generation"),
        "final_best_fitness": summary.get("best_fitness"),
        "final_best_distance": summary.get("best_route_distance"),
        "best_feasible_fitness": tracked_metrics.get(
            "best_feasible_fitness",
            summary.get("best_total_value") if summary.get("best_is_feasible") else None,
        ),
        "feasible_rate": tracked_metrics.get("feasible_rate"),
        "mean_violation": tracked_metrics.get("mean_violation"),
        "hypervolume": summary.get("hypervolume"),
        "pareto_ratio": summary.get("pareto_ratio"),
        "spread": summary.get("spread"),
        "pareto_front_size": summary.get("pareto_front_size"),
    }


def _validated_entry(problem: str, size: int, baselines: tuple[str, ...]) -> None:
    scope = VALIDATED_SCOPE.get(problem)
    if scope is None:
        raise ValueError(f"Unsupported problem in baseline comparison manifest: {problem}")
    if size not in scope["sizes"]:
        allowed = ", ".join(str(value) for value in sorted(scope["sizes"]))
        raise ValueError(f"Unsupported validated size {size} for {problem}. Allowed: {allowed}")
    unsupported = sorted(set(baselines) - set(scope["baselines"]))
    if unsupported:
        allowed = ", ".join(scope["baselines"])
        raise ValueError(
            f"Unsupported baselines for {problem}: {', '.join(unsupported)}. Allowed: {allowed}"
        )


def load_comparison_manifest(path: str | Path) -> tuple[dict[str, Any], list[ComparisonEntry]]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Baseline comparison manifest must be a JSON object")
    suite_name = str(manifest.get("suite_name", manifest_path.stem))
    suite_kind = str(manifest.get("suite_kind", "comparison"))
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Baseline comparison manifest requires a non-empty entries list")
    entries: list[ComparisonEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each baseline comparison manifest entry must be an object")
        problem = str(raw_entry["problem"])
        size = int(raw_entry["size"])
        baselines = tuple(str(value) for value in raw_entry.get("baselines", []))
        _validated_entry(problem, size, baselines)
        preset_path = Path(raw_entry["preset"])
        if not preset_path.is_absolute():
            candidates = [
                (PROJECT_ROOT / preset_path).resolve(),
                (manifest_path.parent / preset_path).resolve(),
            ]
            preset_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        entries.append(
            ComparisonEntry(
                suite_name=suite_name,
                suite_kind=suite_kind,
                problem=problem,
                size=size,
                preset_path=preset_path,
                baselines=baselines,
                seeds=int(raw_entry.get("seeds", manifest.get("default_seeds", 10))),
                seed_start=int(raw_entry.get("seed_start", 0)),
            )
        )
    return manifest, entries


def _run_manifest(manifest_path: str | Path, output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    manifest, entries = load_comparison_manifest(manifest_path)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suite_dir = output_root / f"{timestamp}_{manifest['suite_name']}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []

    for entry in entries:
        config = GAConfig.from_dict(json.loads(entry.preset_path.read_text(encoding="utf-8")))
        if config.problem != entry.problem:
            raise ValueError(
                f"Preset problem mismatch for {entry.preset_path}: {config.problem} != {entry.problem}"
            )
        if _size_value(config) != entry.size:
            raise ValueError(
                f"Preset size mismatch for {entry.preset_path}: {_size_value(config)} != {entry.size}"
            )
        for offset in range(entry.seeds):
            seed = entry.seed_start + offset
            run_config = GAConfig.from_dict(config.to_dict())
            run_config.seed = seed
            run_rows.append(_run_ga_trial(entry, run_config, seed))
            for family in entry.baselines:
                baseline_config = GAConfig.from_dict(config.to_dict())
                baseline_config.seed = seed
                run_rows.append(_run_baseline_trial(entry, baseline_config, family, seed))

    aggregate_rows = _group_run_rows(run_rows)
    (suite_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (suite_dir / "run_rows.json").write_text(
        json.dumps(run_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (suite_dir / "aggregate_rows.json").write_text(
        json.dumps(aggregate_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if aggregate_rows:
        _write_csv(suite_dir / "aggregate_rows.csv", aggregate_rows, tuple(aggregate_rows[0].keys()))
    return manifest, run_rows, suite_dir


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Baseline comparison summary",
        "",
        "## Scope",
        "",
        "Validated comparison scope only:",
        "",
        "| Problem | Size key | Validated sizes |",
        "| --- | --- | --- |",
    ]
    for problem, scope in VALIDATED_SCOPE.items():
        sizes = " / ".join(str(size) for size in scope["sizes"])
        lines.append(f"| {problem} | `{scope['size_key']}` | `{sizes}` |")

    lines.extend(
        [
            "",
            "## Fairness policy",
            "",
            "- Primary comparison basis: matched function evaluation budget.",
            "- Initial population evaluation is included in the GA budget.",
            "- The runner's final post-loop re-evaluation is also included because it happens in the current code path.",
            "- Wall-clock runtime is reported, but it is secondary to matched-budget quality.",
            "",
            "## Budget definition",
            "",
            "| Algorithm | Configured budget formula | Notes |",
            "| --- | --- | --- |",
            "| `ga` | `population_size * (generations + 2)` | evaluates the population at generations `0..G`, then re-evaluates the final population once more |",
            "| `nsga2` | `population_size * (3 * generations + 2)` | per generation it evaluates the current population and the combined parent+offspring population, then re-evaluates the final population |",
            "",
            "## Comparison snapshot",
            "",
            "| Problem | Size | Label | Key metrics | Runtime |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for row in summary["aggregate_rows"]:
        key_metrics = []
        if row.get("success_rate") is not None:
            key_metrics.append(f"success `{row['success_rate']:.2f}`")
        if row.get("mean_final_best_distance") is not None:
            key_metrics.append(f"distance `{row['mean_final_best_distance']:.2f}`")
        if row.get("mean_best_feasible_fitness") is not None:
            key_metrics.append(f"feasible fitness `{row['mean_best_feasible_fitness']:.2f}`")
        if row.get("mean_hypervolume") is not None:
            key_metrics.append(f"HV `{row['mean_hypervolume']:.4f}`")
        if row.get("mean_pareto_ratio") is not None:
            key_metrics.append(f"pareto_ratio `{row['mean_pareto_ratio']:.4f}`")
        if row.get("mean_final_best_fitness") is not None and row["problem"] == "onemax":
            key_metrics.append(f"best fitness `{row['mean_final_best_fitness']:.2f}`")
        lines.append(
            f"| {row['problem']} | `{row['size']}` | `{row['suite_kind']}:{row['label']}` | "
            f"{', '.join(key_metrics)} | `{row['mean_runtime_seconds']:.3f}s` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "| Problem | Size | Outcome | Note |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in summary["interpretations"]:
        lines.append(
            f"| {row['problem']} | `{row['size']}` | `{row['outcome']}` | {row['note']} |"
        )
    return "\n".join(lines) + "\n"


def run_manifests(
    manifest_paths: list[str | Path],
    *,
    output_root: str | Path,
    summary_stem: str,
) -> dict[str, Any]:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    all_run_rows: list[dict[str, Any]] = []
    suite_inputs: dict[str, str] = {}
    manifests_payload: dict[str, Any] = {}

    for manifest_path in manifest_paths:
        manifest, run_rows, suite_dir = _run_manifest(manifest_path, output_root_path)
        suite_name = str(manifest["suite_name"])
        suite_inputs[suite_name] = str(suite_dir.as_posix())
        manifests_payload[suite_name] = manifest
        all_run_rows.extend(run_rows)

    aggregate_rows = _group_run_rows(all_run_rows)
    aggregate_columns = tuple(aggregate_rows[0].keys()) if aggregate_rows else ()
    run_metadata = build_run_metadata(
        project_root=PROJECT_ROOT,
        summary_stem=summary_stem,
        output_root=output_root_path,
        manifest_paths=manifest_paths,
        extra={
            "suite_kind": "baseline_comparison",
            "suite_names": sorted(suite_inputs),
            "run_row_count": len(all_run_rows),
        },
    )
    summary = {
        "summary_version": 1,
        "summary_schema_version": 2,
        "validated_scope": {
            problem: {
                "size_key": scope["size_key"],
                "sizes": list(scope["sizes"]),
            }
            for problem, scope in VALIDATED_SCOPE.items()
        },
        "suite_inputs": suite_inputs,
        "manifests": manifests_payload,
        "fairness_policy": {
            "primary_basis": "matched_function_evaluation_budget",
            "initial_population_evaluations_included": True,
            "final_post_loop_population_re_evaluation_included": True,
            "runtime_is_secondary": True,
        },
        "budget_definition": {
            "ga": budget_formula_text("ga"),
            "nsga2": budget_formula_text("nsga2"),
        },
        "run_rows": all_run_rows,
        "aggregate_rows": aggregate_rows,
        "claim_ranges": _claim_ranges(aggregate_rows),
        "interpretations": _problem_size_interpretations(aggregate_rows),
        "run_metadata": run_metadata,
    }

    (output_root_path / f"{summary_stem}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_run_metadata(output_root_path / f"{summary_stem}_run_metadata.json", run_metadata)
    if aggregate_rows:
        _write_csv(output_root_path / f"{summary_stem}.csv", aggregate_rows, aggregate_columns)
    (output_root_path / f"{summary_stem}.md").write_text(
        _markdown_summary(summary),
        encoding="utf-8",
    )
    return summary
