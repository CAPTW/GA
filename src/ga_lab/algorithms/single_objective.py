from __future__ import annotations

from statistics import mean
from typing import Any

from ga_lab.adaptive_policies import (
    adaptive_mutation_decision,
    adaptive_policy_name,
    effective_policy_for_mode,
    fixed_policy_mode,
    is_switching_policy,
    maybe_refresh_population,
    online_switch_mode_decision,
)
from ga_lab.algorithms._shared import (
    apply_mutation_with_rate,
    early_stop_decision,
    log_summary_row,
    problem_population_metrics,
    problem_solution_metrics,
    select_log_generation,
    validate_fitness_vector,
)
from ga_lab.config import GAConfig
from ga_lab.convergence_diagnostics import (
    ProgressState,
    build_generation_diagnostics,
    configured_evaluation_budget,
    update_progress_state,
    update_signal_window,
)
from ga_lab.core.crossover import CrossoverFn
from ga_lab.core.mutation import MutationFn
from ga_lab.core.representation import InitFn, Population
from ga_lab.core.selection import SelectionFn, SelectionState
from ga_lab.experiment.algorithm_checkpoint import (
    ALGORITHM_NAME,
    CHECKPOINT_TYPE,
    SCHEMA_VERSION,
    AlgorithmCheckpointState,
    CheckpointConfig,
    CheckpointMetadata,
    EvaluationBudgetState,
    PopulationCheckpoint,
    build_config_hash,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    summarize_operator_signature,
    summarize_problem_signature,
    utc_timestamp,
    validate_resume_compatibility,
    write_checkpoint_atomic,
    write_resume_compatibility_report,
)
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


def run_single_objective_ga(
    config: GAConfig,
    problem,
    selection_fn: SelectionFn,
    crossover_fn: CrossoverFn,
    mutation_fn: MutationFn,
    init_fn: InitFn,
    rng,
    checkpoint_config: CheckpointConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_enabled = bool(checkpoint_config is not None and checkpoint_config.enabled)
    configured_budget = configured_evaluation_budget(config)
    population = [init_fn(rng, config.genome_length) for _ in range(config.population_size)]
    stop_reason = "max_generations"
    convergence_generation: int | None = None
    history: list[dict[str, Any]] = []
    actual_evaluations_used = 0
    start_generation = 0
    resume_metadata: dict[str, Any] | None = None
    checkpoint_paths: list[str] = []
    evaluations_to_target: int | None = None
    extra_evaluations_from_adaptation = 0
    adaptive_event_count = 0
    adaptive_restart_events = 0
    adaptive_diversity_injections = 0
    adaptive_mutation_boost_events = 0
    refresh_event_count = 0
    total_refreshed_individuals = 0
    last_event_generation: int | None = None
    first_trigger_generation: int | None = None
    first_trigger_metric_value: float | None = None
    trigger_event_generations: list[int] = []
    trigger_event_names: list[str] = []
    pending_event_name = "none"
    options = dict(config.algorithm_options)
    policy_name = adaptive_policy_name(options)
    configured_early_stop_policy = (
        str(options.get("early_stop_policy", "none"))
        if isinstance(options.get("early_stop_policy", "none"), str)
        else "none"
    )
    switching_policy = is_switching_policy(policy_name)
    adaptive_mode = "decay" if switching_policy else fixed_policy_mode(policy_name)
    rescue_until_generation: int | None = None
    mode_switch_count = 0
    first_switch_generation: int | None = None
    mode_switch_generations: list[int] = []
    mode_switch_modes: list[str] = []
    time_in_decay_mode = 0
    time_in_trigger_mode = 0
    early_stop_generation: int | None = None
    early_stop_triggered = False
    progress_state = ProgressState(maximize=config.maximize)

    if checkpoint_enabled and checkpoint_config is not None and checkpoint_config.resume_from:
        checkpoint = load_checkpoint(checkpoint_config.resume_from)
        compatibility_report = validate_resume_compatibility(
            checkpoint,
            config=config,
            problem=problem,
            requested_budget=configured_budget,
        )
        suffix = checkpoint_config.artifact_suffix or "phase1_resume"
        write_resume_compatibility_report(
            compatibility_report,
            output_dir=checkpoint_config.output_dir,
            run_id=checkpoint_config.run_id,
            suffix=suffix,
            allow_overwrite=checkpoint_config.allow_overwrite,
        )
        if compatibility_report.decision == "fail":
            failures = "; ".join(compatibility_report.failures)
            raise ValueError(f"Checkpoint resume compatibility failed: {failures}")
        restore_warnings = restore_rng_state(rng, checkpoint.rng)
        population = [list(genome) for genome in checkpoint.population.decision_vectors]
        history = [dict(row) for row in checkpoint.history]
        actual_evaluations_used = checkpoint.budget_state.actual_evaluations
        start_generation = checkpoint.resume_generation
        for row in history:
            if not isinstance(row.get("generation"), int):
                continue
            metric_name = str(row.get("progress_metric_name", "best_fitness"))
            metric_value = row.get(metric_name)
            if isinstance(metric_value, int | float):
                update_progress_state(
                    progress_state,
                    generation=int(row["generation"]),
                    value=float(metric_value),
                )
            diversity_value = row.get("diversity_signal")
            if isinstance(diversity_value, int | float):
                update_signal_window(
                    progress_state,
                    signal_key="diversity_signal",
                    output_prefix="diversity",
                    generation=int(row["generation"]),
                    value=float(diversity_value),
                )
            feasible_ratio = row.get("feasible_ratio")
            if isinstance(feasible_ratio, int | float):
                update_signal_window(
                    progress_state,
                    signal_key="feasible_ratio",
                    output_prefix="feasible_ratio",
                    generation=int(row["generation"]),
                    value=float(feasible_ratio),
                )
            mean_violation = row.get("mean_constraint_violation")
            if isinstance(mean_violation, int | float):
                update_signal_window(
                    progress_state,
                    signal_key="mean_constraint_violation",
                    output_prefix="constraint_violation",
                    generation=int(row["generation"]),
                    value=float(mean_violation),
                )
        resume_metadata = {
            "resumed": True,
            "resume_from": str(checkpoint_config.resume_from),
            "resume_generation": start_generation,
            "compatibility_report": compatibility_report.to_dict(),
            "restore_warnings": restore_warnings,
        }

    def evaluate_population(
        population_slice: Population,
        *,
        generation: int | None,
        location: str,
    ) -> list[list[float]]:
        nonlocal actual_evaluations_used, evaluations_to_target
        objective_vectors: list[list[float]] = []
        for genome_index, genome in enumerate(population_slice):
            values = as_fitness_vector(problem.fitness(genome))
            validate_fitness_vector(
                values,
                problem_name=str(getattr(problem, "name", config.problem)),
                genome=genome,
                location=location,
                generation=generation,
                evaluation_index=genome_index,
            )
            actual_evaluations_used += 1
            objective_vectors.append(values)
            if evaluations_to_target is None and config.target_fitness is not None:
                scalar_value = values[0]
                if (
                    (config.maximize and scalar_value >= config.target_fitness)
                    or ((not config.maximize) and scalar_value <= config.target_fitness)
                ):
                    evaluations_to_target = actual_evaluations_used
        return objective_vectors

    generations_since_last_improvement = 0
    current_mutation_rate = config.mutation_rate

    def build_checkpoint_state(
        *,
        generation: int,
        checkpoint_population: Population,
        scalar_values: list[float],
        best_idx_value: int,
        best_fitness_value: float,
        completed: bool,
    ) -> AlgorithmCheckpointState:
        warnings: list[str] = [
            "Phase 1 checkpoint is explicit opt-in and not production fault tolerance"
        ]
        if resume_metadata is not None:
            warnings.append("checkpoint was written during a resumed run")
        rng_checkpoint = capture_rng_state(rng)
        if rng_checkpoint.rng_warning:
            warnings.append(rng_checkpoint.rng_warning)
        requested_budget = int(configured_budget)
        return AlgorithmCheckpointState(
            metadata=CheckpointMetadata(
                schema_version=SCHEMA_VERSION,
                checkpoint_type=CHECKPOINT_TYPE,
                algorithm=ALGORITHM_NAME,
                problem=str(getattr(problem, "name", config.problem)),
                seed=config.seed,
                created_at=utc_timestamp(),
                generation_index=generation,
                actual_evaluations=actual_evaluations_used,
                requested_budget=requested_budget,
                config_hash=build_config_hash(config),
                operator_signature_summary=summarize_operator_signature(config),
                problem_signature_summary=summarize_problem_signature(config, problem),
                completed=completed,
                warnings=warnings,
            ),
            population=PopulationCheckpoint(
                decision_vectors=[list(genome) for genome in checkpoint_population],
                fitness_values=[float(value) for value in scalar_values],
                population_size=len(checkpoint_population),
                best_index=best_idx_value,
                best_fitness=float(best_fitness_value),
                objective_direction=bool(config.maximize),
                finite_validation_status="pass",
            ),
            rng=rng_checkpoint,
            budget_state=EvaluationBudgetState(
                requested_budget=requested_budget,
                actual_evaluations=actual_evaluations_used,
                remaining_budget=max(0, requested_budget - actual_evaluations_used),
                generation_index=generation,
                evaluations_per_generation=config.population_size,
                budget_policy="configured_evaluation_budget",
            ),
            resume_generation=generation + 1,
            history=[dict(row) for row in history],
        )

    for generation in range(start_generation, config.generations + 1):
        objective_vectors = evaluate_population(
            population,
            generation=generation,
            location="single_objective_population",
        )
        scalar = [values[0] for values in objective_vectors]
        selection_state = SelectionState.from_fitnesses(scalar, config.maximize)

        best_idx = best_index(objective_vectors, config.maximize)
        best_fitness = scalar[best_idx]
        current_row = log_summary_row(generation, objective_vectors, config.maximize)
        current_row.update(problem_population_metrics(problem, population))
        current_row.update(problem_solution_metrics(problem, population[best_idx]))
        diagnostics = build_generation_diagnostics(
            config=config,
            problem=problem,
            population=population,
            scalar_values=scalar,
            row=current_row,
            progress_state=progress_state,
            generation=generation,
            configured_budget_value=configured_budget,
            actual_evaluations_used=actual_evaluations_used,
            extra_evaluations_from_adaptation=extra_evaluations_from_adaptation,
            adaptive_mutation_rate=current_mutation_rate,
            adaptive_policy_name=policy_name,
            adaptive_event=pending_event_name,
            adaptive_event_count=adaptive_event_count,
        )
        current_row.update(diagnostics)
        pending_event_name = "none"
        generations_since_last_improvement = int(
            current_row.get("generations_since_last_improvement", 0)
        )
        reached_target = (
            convergence_generation is None
            and config.target_fitness is not None
            and (
                (config.maximize and best_fitness >= config.target_fitness)
                or ((not config.maximize) and best_fitness <= config.target_fitness)
            )
        )
        reached_limit = generation == config.generations

        mode_switch_event = "none"
        if not reached_target and not reached_limit and switching_policy:
            switch_decision = online_switch_mode_decision(
                policy=policy_name,
                generation=generation,
                current_mode=adaptive_mode,
                rescue_until_generation=rescue_until_generation,
                generations_since_last_improvement=generations_since_last_improvement,
                diversity_signal=(
                    float(current_row["diversity_signal"])
                    if isinstance(current_row.get("diversity_signal"), int | float)
                    else None
                ),
                recent_window_improvement=(
                    float(current_row["recent_window_improvement"])
                    if isinstance(current_row.get("recent_window_improvement"), int | float)
                    else None
                ),
                diversity_recent_slope=(
                    float(current_row["recent_diversity_slope"])
                    if isinstance(current_row.get("recent_diversity_slope"), int | float)
                    else None
                ),
                options=options,
            )
            adaptive_mode = switch_decision.mode
            rescue_until_generation = switch_decision.rescue_until_generation
            if switch_decision.switched:
                mode_switch_count += 1
                mode_switch_event = switch_decision.switch_event
                mode_switch_generations.append(generation)
                mode_switch_modes.append(adaptive_mode)
                if first_switch_generation is None and adaptive_mode == "trigger":
                    first_switch_generation = generation

        current_row["adaptive_mode"] = adaptive_mode
        current_row["mode_switch_event"] = mode_switch_event
        current_row["mode_switch_count"] = mode_switch_count
        current_row["first_switch_generation"] = first_switch_generation
        current_row["time_in_decay_mode"] = time_in_decay_mode
        current_row["time_in_trigger_mode"] = time_in_trigger_mode
        should_stop_early, early_stop_policy = early_stop_decision(
            generation=generation,
            row=current_row,
            options=options,
        )
        current_row["early_stop_policy"] = early_stop_policy
        current_row["early_stop_triggered"] = should_stop_early

        if select_log_generation(generation, config.log_every):
            history.append(current_row)

        if reached_target:
            convergence_generation = generation
            stop_reason = "target_fitness_reached"
            break

        if reached_limit:
            break

        if should_stop_early:
            stop_reason = f"early_stop_{early_stop_policy}"
            early_stop_generation = generation
            early_stop_triggered = True
            break

        effective_options = dict(options)
        effective_policy = effective_policy_for_mode(policy_name, adaptive_mode)
        effective_options["adaptive_policy"] = effective_policy
        if effective_policy == "decay_mutation":
            time_in_decay_mode += 1
        elif effective_policy == "low_diversity_injection":
            time_in_trigger_mode += 1

        mutation_decision = adaptive_mutation_decision(
            config,
            generation=generation,
            generations_since_last_improvement=generations_since_last_improvement,
            diversity_signal=(
                float(current_row["diversity_signal"])
                if isinstance(current_row.get("diversity_signal"), int | float)
                else None
            ),
            recent_window_improvement=(
                float(current_row["recent_window_improvement"])
                if isinstance(current_row.get("recent_window_improvement"), int | float)
                else None
            ),
            diversity_recent_slope=(
                float(current_row["recent_diversity_slope"])
                if isinstance(current_row.get("recent_diversity_slope"), int | float)
                else None
            ),
            feasible_ratio=(
                float(current_row["feasible_ratio"])
                if isinstance(current_row.get("feasible_ratio"), int | float)
                else None
            ),
            mean_constraint_violation=(
                float(current_row["mean_constraint_violation"])
                if isinstance(current_row.get("mean_constraint_violation"), int | float)
                else None
            ),
            recent_constraint_violation_slope=(
                float(current_row["recent_constraint_violation_slope"])
                if isinstance(current_row.get("recent_constraint_violation_slope"), int | float)
                else None
            ),
            last_event_generation=last_event_generation,
            options=effective_options,
        )
        current_mutation_rate = mutation_decision.mutation_rate
        ranked = _sort_population(population, scalar, config.maximize)
        next_population = ranked[: config.elitism]
        while len(next_population) < config.population_size:
            parent_a = selection_fn(population, selection_state, rng)
            parent_b = selection_fn(population, selection_state, rng)

            if rng.random() < config.crossover_rate:
                child_a, child_b = crossover_fn(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a[:], parent_b[:]

            child_a = apply_mutation_with_rate(
                config,
                mutation_fn,
                child_a,
                rng,
                mutation_rate=current_mutation_rate,
            )
            child_b = apply_mutation_with_rate(
                config,
                mutation_fn,
                child_b,
                rng,
                mutation_rate=current_mutation_rate,
            )

            next_population.append(child_a)
            if len(next_population) < config.population_size:
                next_population.append(child_b)

        next_population, adaptive_event, refreshed = maybe_refresh_population(
            next_population,
            config=config,
            init_fn=init_fn,
            rng=rng,
            options=effective_options,
            generations_since_last_improvement=generations_since_last_improvement,
            diversity_signal=(
                float(current_row["diversity_signal"])
                if isinstance(current_row.get("diversity_signal"), int | float)
                else None
            ),
            recent_window_improvement=(
                float(current_row["recent_window_improvement"])
                if isinstance(current_row.get("recent_window_improvement"), int | float)
                else None
            ),
            diversity_recent_slope=(
                float(current_row["recent_diversity_slope"])
                if isinstance(current_row.get("recent_diversity_slope"), int | float)
                else None
            ),
            last_event_generation=last_event_generation,
            generation=generation,
        )
        logged_event = (
            adaptive_event
            if adaptive_event != "none"
            else mutation_decision.trigger_event
        )
        if logged_event != "none":
            adaptive_event_count += 1
            last_event_generation = generation
            pending_event_name = logged_event
            trigger_event_generations.append(generation)
            trigger_event_names.append(logged_event)
            if first_trigger_generation is None:
                first_trigger_generation = generation
                if config.problem == "tsp" and isinstance(best_fitness, int | float):
                    first_trigger_metric_value = float(-best_fitness)
                else:
                    metric_name = str(current_row.get("progress_metric_name", "best_fitness"))
                    metric_value = current_row.get(metric_name)
                    if isinstance(metric_value, int | float):
                        first_trigger_metric_value = float(metric_value)
            if logged_event == "restart":
                adaptive_restart_events += 1
            if logged_event == "diversity_injection":
                adaptive_diversity_injections += 1
            if logged_event == "mutation_boost":
                adaptive_mutation_boost_events += 1
        if refreshed > 0:
            refresh_event_count += 1
            total_refreshed_individuals += refreshed

        population = next_population
        if (
            checkpoint_enabled
            and checkpoint_config is not None
            and generation % checkpoint_config.interval_generations == 0
        ):
            checkpoint_state = build_checkpoint_state(
                generation=generation,
                checkpoint_population=population,
                scalar_values=scalar,
                best_idx_value=best_idx,
                best_fitness_value=best_fitness,
                completed=False,
            )
            checkpoint_path = write_checkpoint_atomic(checkpoint_state, checkpoint_config)
            checkpoint_paths.append(str(checkpoint_path))
            if checkpoint_config.stop_after_checkpoint_generation == generation:
                summary = {
                    "stop_reason": "checkpoint_debug_stop",
                    "final_generation": generation,
                    "objective_count": 1,
                    "objective_directions": [config.maximize],
                    "configured_evaluation_budget": configured_budget,
                    "actual_evaluations_used": actual_evaluations_used,
                    "checkpointing": {
                        "enabled": True,
                        "resumed": resume_metadata is not None,
                        "last_checkpoint_path": str(checkpoint_path),
                        "checkpoint_paths": checkpoint_paths[:],
                        "debug_stop_after_checkpoint_generation": generation,
                        "debug_only": True,
                    },
                }
                return summary, history

    final_vectors = evaluate_population(
        population,
        generation=history[-1]["generation"] if history else 0,
        location="single_objective_final_population",
    )
    final_scalars = [values[0] for values in final_vectors]
    best_idx = best_index(final_vectors, config.maximize)
    if config.maximize:
        worst_idx = min(range(len(final_scalars)), key=final_scalars.__getitem__)
    else:
        worst_idx = max(range(len(final_scalars)), key=final_scalars.__getitem__)

    final_generation = history[-1]["generation"]
    summary = {
        "best_fitness": final_scalars[best_idx],
        "mean_fitness": mean(final_scalars),
        "worst_fitness": final_scalars[worst_idx],
        "best_fitness_vector": final_vectors[best_idx],
        "best_genome": population[best_idx][:],
        "convergence_generation": convergence_generation,
        "stop_reason": stop_reason,
        "final_generation": final_generation,
        "objective_count": 1,
        "objective_directions": [config.maximize],
        "configured_evaluation_budget": configured_budget,
        "actual_evaluations_used": actual_evaluations_used,
        "evaluations_to_target": evaluations_to_target,
        "extra_evaluations_from_adaptation": extra_evaluations_from_adaptation,
        "adaptive_policy": policy_name,
        "early_stop_policy": configured_early_stop_policy,
        "early_stop_generation": early_stop_generation,
        "early_stop_triggered": early_stop_triggered,
        "adaptive_event_count": adaptive_event_count,
        "adaptive_restart_events": adaptive_restart_events,
        "adaptive_diversity_injections": adaptive_diversity_injections,
        "adaptive_mutation_boost_events": adaptive_mutation_boost_events,
        "refresh_event_count": refresh_event_count,
        "total_refreshed_individuals": total_refreshed_individuals,
        "average_refresh_fraction_realized": (
            total_refreshed_individuals / (config.population_size * refresh_event_count)
            if refresh_event_count > 0 and config.population_size > 0
            else 0.0
        ),
        "total_refresh_fraction_realized": (
            total_refreshed_individuals / config.population_size
            if config.population_size > 0
            else 0.0
        ),
        "trigger_fire_count": adaptive_event_count,
        "first_trigger_generation": first_trigger_generation,
        "trigger_event_generations": trigger_event_generations,
        "trigger_event_names": trigger_event_names,
        "mode_switch_count": mode_switch_count,
        "first_switch_generation": first_switch_generation,
        "mode_switch_generations": mode_switch_generations,
        "mode_switch_modes": mode_switch_modes,
        "time_in_decay_mode": time_in_decay_mode,
        "time_in_trigger_mode": time_in_trigger_mode,
    }
    summary.update(problem_solution_metrics(problem, population[best_idx]))
    if (
        first_trigger_metric_value is not None
        and isinstance(summary["best_fitness"], int | float)
    ):
        if config.problem == "tsp":
            final_progress_value = summary.get("best_route_distance")
            if isinstance(final_progress_value, int | float):
                summary["post_trigger_improvement"] = (
                    first_trigger_metric_value - float(final_progress_value)
                )
        elif config.maximize:
            summary["post_trigger_improvement"] = (
                float(summary["best_fitness"]) - first_trigger_metric_value
            )
        else:
            summary["post_trigger_improvement"] = (
                first_trigger_metric_value - float(summary["best_fitness"])
            )
    summary.update(problem_population_metrics(problem, population))
    if checkpoint_enabled and checkpoint_config is not None:
        summary["checkpointing"] = {
            "enabled": True,
            "resumed": resume_metadata is not None,
            "resume_metadata": resume_metadata,
            "last_checkpoint_path": checkpoint_paths[-1] if checkpoint_paths else None,
            "checkpoint_paths": checkpoint_paths[:],
            "phase": "algorithm_checkpoint_phase1_single_objective_ga",
        }
    return summary, history
