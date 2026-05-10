from __future__ import annotations

from statistics import mean
from typing import Any

from ga_lab.algorithms._shared import (
    early_stop_decision,
    log_summary_row,
    problem_population_metrics,
    problem_solution_metrics,
    select_log_generation,
    validate_fitness_vector,
)
from ga_lab.config import GAConfig
from ga_lab.convergence_diagnostics import ProgressState, build_generation_diagnostics
from ga_lab.core.crossover import CrossoverFn
from ga_lab.core.mutation import MutationFn
from ga_lab.core.representation import Genome, InitFn, Population
from ga_lab.core.selection import SelectionFn, SelectionState
from ga_lab.problems.base import as_fitness_vector, best_index


def _sort_population(
    population: Population,
    objective: list[float],
    maximize: bool,
) -> Population:
    return [
        population[idx][:]
        for idx in sorted(range(len(population)), key=objective.__getitem__, reverse=maximize)
    ]


def _get_int(options: dict[str, object], key: str, default: int) -> int:
    value = options.get(key, default)
    return int(value) if isinstance(value, int) else default


def _get_float(options: dict[str, object], key: str, default: float) -> float:
    value = options.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _get_str(options: dict[str, object], key: str, default: str) -> str:
    value = options.get(key, default)
    return str(value).strip() if isinstance(value, str) and value.strip() else default


def _configured_budget(config: GAConfig) -> int:
    return config.population_size * (config.generations + 2)


def _knapsack_item_ratios(problem) -> list[tuple[float, int]]:
    return sorted(
        (
            (problem.values[idx] / problem.weights[idx], idx)
            for idx in range(problem.num_items)
            if problem.weights[idx] > 0
        ),
        reverse=True,
    )


def _knapsack_greedy_bits(problem) -> list[int]:
    bits = [0] * problem.num_items
    current_weight = 0.0
    for _, idx in _knapsack_item_ratios(problem):
        next_weight = current_weight + problem.weights[idx]
        if next_weight <= problem.capacity:
            bits[idx] = 1
            current_weight = next_weight
    return bits


def _knapsack_repair_and_fill(problem, genome: Genome) -> Genome:
    bits = problem.decode_items(genome)
    selected_by_low_ratio = [
        idx
        for _, idx in sorted(
            (
                (problem.values[idx] / problem.weights[idx], idx)
                for idx, selected in enumerate(bits)
                if selected and problem.weights[idx] > 0
            )
        )
    ]
    total_weight = sum(
        problem.weights[idx] for idx, selected in enumerate(bits) if selected == 1
    )
    for idx in selected_by_low_ratio:
        if total_weight <= problem.capacity:
            break
        bits[idx] = 0
        total_weight -= problem.weights[idx]
    for _, idx in _knapsack_item_ratios(problem):
        if bits[idx] == 1:
            continue
        candidate_weight = total_weight + problem.weights[idx]
        if candidate_weight <= problem.capacity:
            bits[idx] = 1
            total_weight = candidate_weight
    return [float(bit) for bit in bits]


def _build_knapsack_seed(problem, rng) -> Genome:
    bits = _knapsack_greedy_bits(problem)
    selected = [idx for idx, value in enumerate(bits) if value == 1]
    if selected:
        drop_cap = max(1, len(selected) // 5)
        drop_count = rng.randint(0, drop_cap)
        for idx in rng.sample(selected, min(drop_count, len(selected))):
            bits[idx] = 0
    repaired = _knapsack_repair_and_fill(problem, [float(bit) for bit in bits])
    bits = [int(value) for value in repaired]
    return [float(bit) for bit in bits]


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


def _two_opt_candidate(route: list[int], start: int, end: int) -> list[int]:
    candidate = route[:]
    candidate[start : end + 1] = reversed(candidate[start : end + 1])
    return candidate


def _build_tsp_seed(problem, rng, seed_index: int) -> Genome:
    start_city = seed_index % problem.num_cities
    route = _nearest_neighbor_route(problem, start_city)
    if len(route) > 4 and rng.random() < 0.5:
        left = rng.randrange(len(route) - 2)
        right = rng.randrange(left + 1, len(route) - 1)
        route = _two_opt_candidate(route, left, right)
    return [float(city) for city in route]


def _initialize_population(
    config: GAConfig,
    problem,
    init_fn: InitFn,
    rng,
    options: dict[str, object],
) -> tuple[Population, int]:
    initial_population = options.get("_initial_population")
    if isinstance(initial_population, list) and len(initial_population) == config.population_size:
        resumed_population: Population = []
        for genome in initial_population:
            if isinstance(genome, list) and len(genome) == config.genome_length:
                resumed_population.append([float(gene) for gene in genome])
            else:
                resumed_population = []
                break
        if len(resumed_population) == config.population_size:
            return resumed_population, 0

    population: Population = []
    seeded = 0
    init_strategy = _get_str(options, "init_strategy", "none")
    seed_fraction = max(0.0, min(1.0, _get_float(options, "seed_fraction", 0.0)))
    seed_count = min(config.population_size, int(round(config.population_size * seed_fraction)))

    if init_strategy == "knapsack_greedy_mix" and config.problem == "knapsack":
        for _ in range(seed_count):
            population.append(_build_knapsack_seed(problem, rng))
        seeded = seed_count
    elif init_strategy == "tsp_nearest_neighbor_mix" and config.problem == "tsp":
        start_order = list(range(problem.num_cities))
        rng.shuffle(start_order)
        for seed_index in range(seed_count):
            start_city = start_order[seed_index % len(start_order)]
            population.append(_build_tsp_seed(problem, rng, start_city))
        seeded = seed_count

    while len(population) < config.population_size:
        genome = init_fn(rng, config.genome_length)
        population.append(_apply_repair(problem, config, genome, options))
    return population, seeded


def _apply_repair(problem, config: GAConfig, genome: Genome, options: dict[str, object]) -> Genome:
    repair_strategy = _get_str(options, "repair_strategy", "none")
    if repair_strategy == "knapsack_greedy_fill" and config.problem == "knapsack":
        return _knapsack_repair_and_fill(problem, genome)
    return genome[:]


def _knapsack_local_search(
    problem,
    genome: Genome,
    current_fitness: float,
    evaluate_scalar,
    budget_remaining: int,
    options: dict[str, object],
) -> tuple[Genome, float, int, bool]:
    if budget_remaining <= 0:
        return genome[:], current_fitness, 0, False

    max_candidate_evals = min(budget_remaining, _get_int(options, "local_search_steps", 8))
    remove_width = max(1, _get_int(options, "local_search_remove_width", 3))
    add_width = max(1, _get_int(options, "local_search_add_width", 5))

    bits = problem.decode_items(genome)
    selected = [
        (problem.values[idx] / problem.weights[idx], idx)
        for idx, value in enumerate(bits)
        if value == 1 and problem.weights[idx] > 0
    ]
    unselected = [
        (problem.values[idx] / problem.weights[idx], idx)
        for idx, value in enumerate(bits)
        if value == 0 and problem.weights[idx] > 0
    ]
    selected_low = [idx for _, idx in sorted(selected)[:remove_width]]
    unselected_high = [idx for _, idx in sorted(unselected, reverse=True)[:add_width]]

    seen: set[tuple[int, ...]] = set()
    best_bits = bits[:]
    best_fitness = current_fitness
    evaluations_used = 0

    def enqueue_candidate(candidate_bits: list[int]) -> list[int] | None:
        key = tuple(candidate_bits)
        if key in seen:
            return None
        seen.add(key)
        return candidate_bits

    candidates: list[list[int]] = []
    for add_idx in unselected_high:
        candidate = bits[:]
        candidate[add_idx] = 1
        repaired = _knapsack_repair_and_fill(problem, [float(bit) for bit in candidate])
        normalized = problem.decode_items(repaired)
        queued = enqueue_candidate(normalized)
        if queued is not None:
            candidates.append(queued)
    for remove_idx in selected_low:
        for add_idx in unselected_high:
            if remove_idx == add_idx:
                continue
            candidate = bits[:]
            candidate[remove_idx] = 0
            candidate[add_idx] = 1
            repaired = _knapsack_repair_and_fill(problem, [float(bit) for bit in candidate])
            normalized = problem.decode_items(repaired)
            queued = enqueue_candidate(normalized)
            if queued is not None:
                candidates.append(queued)

    for candidate_bits in candidates[:max_candidate_evals]:
        candidate_genome = [float(bit) for bit in candidate_bits]
        candidate_fitness = evaluate_scalar(candidate_genome)
        evaluations_used += 1
        if candidate_fitness > best_fitness:
            best_bits = candidate_bits
            best_fitness = candidate_fitness

    return (
        [float(bit) for bit in best_bits],
        best_fitness,
        evaluations_used,
        best_fitness > current_fitness,
    )


def _tsp_local_search(
    problem,
    genome: Genome,
    current_fitness: float,
    evaluate_scalar,
    budget_remaining: int,
    options: dict[str, object],
    rng,
) -> tuple[Genome, float, int, bool]:
    if budget_remaining <= 0:
        return genome[:], current_fitness, 0, False

    route = problem.decode_route(genome)
    current_route = route[:]
    current_route_fitness = current_fitness
    best_route = route[:]
    best_fitness = current_fitness
    evaluations_used = 0
    max_candidate_evals = min(budget_remaining, _get_int(options, "local_search_steps", 12))

    for _ in range(max_candidate_evals):
        left = rng.randrange(len(route) - 2)
        right = rng.randrange(left + 1, len(route) - 1)
        candidate_route = _two_opt_candidate(current_route, left, right)
        candidate_fitness = evaluate_scalar([float(city) for city in candidate_route])
        evaluations_used += 1
        if candidate_fitness >= current_route_fitness:
            current_route = candidate_route
            current_route_fitness = candidate_fitness
        if candidate_fitness > best_fitness:
            best_route = candidate_route
            best_fitness = candidate_fitness

    return (
        [float(city) for city in best_route],
        best_fitness,
        evaluations_used,
        best_fitness > current_fitness,
    )


def run_hybrid_ga(
    config: GAConfig,
    problem,
    selection_fn: SelectionFn,
    crossover_fn: CrossoverFn,
    mutation_fn: MutationFn,
    init_fn: InitFn,
    rng,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    options = dict(config.algorithm_options)
    population, seeded_individuals = _initialize_population(config, problem, init_fn, rng, options)
    history: list[dict[str, Any]] = []
    stop_reason = "max_generations"
    convergence_generation: int | None = None
    last_evaluated_generation = 0
    configured_budget = _configured_budget(config)
    reserve_final_evaluations = config.population_size
    evaluations_used = 0
    hybrid_extra_evaluations = 0
    local_search_applications = 0
    local_search_improvements = 0
    configured_early_stop_policy = (
        str(options.get("early_stop_policy", "none"))
        if isinstance(options.get("early_stop_policy", "none"), str)
        else "none"
    )
    early_stop_generation: int | None = None
    early_stop_triggered = False
    progress_state = ProgressState(maximize=config.maximize)

    def evaluate_scalar(genome: Genome) -> float:
        nonlocal evaluations_used
        values = as_fitness_vector(problem.fitness(genome))
        validate_fitness_vector(
            values,
            problem_name=str(getattr(problem, "name", config.problem)),
            genome=genome,
            location="hybrid_local_evaluation",
            generation=last_evaluated_generation,
            evaluation_index=evaluations_used,
        )
        evaluations_used += 1
        return float(values[0])

    for generation in range(config.generations + 1):
        if (
            evaluations_used + config.population_size + reserve_final_evaluations
            > configured_budget
        ):
            stop_reason = "evaluation_budget_exhausted"
            break

        objective_vectors = []
        for genome_index, genome in enumerate(population):
            values = as_fitness_vector(problem.fitness(genome))
            validate_fitness_vector(
                values,
                problem_name=str(getattr(problem, "name", config.problem)),
                genome=genome,
                location="hybrid_population",
                generation=generation,
                evaluation_index=genome_index,
            )
            objective_vectors.append(values)
            evaluations_used += 1
        scalar = [values[0] for values in objective_vectors]
        selection_state = SelectionState.from_fitnesses(scalar, config.maximize)
        last_evaluated_generation = generation
        best_idx = best_index(objective_vectors, config.maximize)

        row = log_summary_row(generation, objective_vectors, config.maximize)
        row.update(problem_population_metrics(problem, population))
        row.update(problem_solution_metrics(problem, population[best_idx]))
        row.update(
            build_generation_diagnostics(
                config=config,
                problem=problem,
                population=population,
                scalar_values=scalar,
                row=row,
                progress_state=progress_state,
                generation=generation,
                configured_budget_value=configured_budget,
                actual_evaluations_used=evaluations_used,
                extra_evaluations_from_adaptation=hybrid_extra_evaluations,
                adaptive_mutation_rate=config.mutation_rate,
                adaptive_policy_name="none",
                adaptive_event="none",
                adaptive_event_count=0,
            )
        )
        should_stop_early, early_stop_policy = early_stop_decision(
            generation=generation,
            row=row,
            options=options,
        )
        row["early_stop_policy"] = early_stop_policy
        row["early_stop_triggered"] = should_stop_early
        if select_log_generation(generation, config.log_every):
            history.append(row)

        best_fitness = scalar[best_idx]
        if (
            convergence_generation is None
            and config.target_fitness is not None
            and (
                (config.maximize and best_fitness >= config.target_fitness)
                or ((not config.maximize) and best_fitness <= config.target_fitness)
            )
        ):
            convergence_generation = generation
            stop_reason = "target_fitness_reached"
            break

        if generation == config.generations:
            break

        if should_stop_early:
            stop_reason = f"early_stop_{early_stop_policy}"
            early_stop_generation = generation
            early_stop_triggered = True
            break

        ranked = _sort_population(population, scalar, config.maximize)
        next_population = ranked[: config.elitism]
        while len(next_population) < config.population_size:
            parent_a = selection_fn(population, selection_state, rng)
            parent_b = selection_fn(population, selection_state, rng)
            if rng.random() < config.crossover_rate:
                child_a, child_b = crossover_fn(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a[:], parent_b[:]

            child_a = _apply_repair(problem, config, mutation_fn(child_a, rng), options)
            child_b = _apply_repair(problem, config, mutation_fn(child_b, rng), options)

            next_population.append(child_a)
            if len(next_population) < config.population_size:
                next_population.append(child_b)

        local_search_strategy = _get_str(options, "local_search_strategy", "none")
        interval = max(1, _get_int(options, "local_search_interval", 1))
        candidate_count = max(
            0,
            min(
                len(next_population),
                _get_int(options, "local_search_candidates", max(config.elitism, 2)),
            ),
        )
        should_run_local_search = (
            local_search_strategy != "none"
            and candidate_count > 0
            and (generation + 1) % interval == 0
        )
        if should_run_local_search:
            for candidate_idx in range(candidate_count):
                budget_remaining = configured_budget - reserve_final_evaluations - evaluations_used
                if budget_remaining <= 1:
                    break

                base_genome = next_population[candidate_idx][:]
                base_fitness = evaluate_scalar(base_genome)
                hybrid_extra_evaluations += 1
                budget_remaining = configured_budget - reserve_final_evaluations - evaluations_used
                if budget_remaining <= 0:
                    break

                if local_search_strategy == "knapsack_ratio_swap" and config.problem == "knapsack":
                    (
                        improved_genome,
                        _,
                        extra_used,
                        improved,
                    ) = _knapsack_local_search(
                        problem,
                        base_genome,
                        base_fitness,
                        evaluate_scalar,
                        budget_remaining,
                        options,
                    )
                elif local_search_strategy == "tsp_2opt" and config.problem == "tsp":
                    (
                        improved_genome,
                        _,
                        extra_used,
                        improved,
                    ) = _tsp_local_search(
                        problem,
                        base_genome,
                        base_fitness,
                        evaluate_scalar,
                        budget_remaining,
                        options,
                        rng,
                    )
                else:
                    break

                hybrid_extra_evaluations += extra_used
                local_search_applications += 1
                if improved:
                    next_population[candidate_idx] = improved_genome
                    local_search_improvements += 1

        population = next_population

    final_vectors = []
    for genome_index, genome in enumerate(population):
        values = as_fitness_vector(problem.fitness(genome))
        validate_fitness_vector(
            values,
            problem_name=str(getattr(problem, "name", config.problem)),
            genome=genome,
            location="hybrid_final_population",
            generation=last_evaluated_generation,
            evaluation_index=genome_index,
        )
        final_vectors.append(values)
        evaluations_used += 1
    final_scalars = [values[0] for values in final_vectors]
    best_idx = best_index(final_vectors, config.maximize)
    if config.maximize:
        worst_idx = min(range(len(final_scalars)), key=final_scalars.__getitem__)
    else:
        worst_idx = max(range(len(final_scalars)), key=final_scalars.__getitem__)

    summary = {
        "best_fitness": final_scalars[best_idx],
        "mean_fitness": mean(final_scalars),
        "worst_fitness": final_scalars[worst_idx],
        "best_fitness_vector": final_vectors[best_idx],
        "best_genome": population[best_idx][:],
        "convergence_generation": convergence_generation,
        "stop_reason": stop_reason,
        "final_generation": last_evaluated_generation,
        "objective_count": 1,
        "objective_directions": [config.maximize],
        "configured_evaluation_budget": configured_budget,
        "actual_evaluations_used": evaluations_used,
        "hybrid_extra_evaluations": hybrid_extra_evaluations,
        "extra_evaluations_from_adaptation": hybrid_extra_evaluations,
        "hybrid_local_search_applications": local_search_applications,
        "hybrid_local_search_improvements": local_search_improvements,
        "hybrid_seeded_individuals": seeded_individuals,
        "early_stop_policy": configured_early_stop_policy,
        "early_stop_generation": early_stop_generation,
        "early_stop_triggered": early_stop_triggered,
    }
    if bool(options.get("_return_final_population")):
        summary["final_population"] = [genome[:] for genome in population]
    summary.update(problem_solution_metrics(problem, population[best_idx]))
    summary.update(problem_population_metrics(problem, population))
    return summary, history
