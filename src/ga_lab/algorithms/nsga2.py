from __future__ import annotations

import math
from statistics import mean
from typing import Any

from ga_lab.adaptive_policies import (
    adaptive_mutation_decision,
    adaptive_policy_name,
    maybe_refresh_population,
)
from ga_lab.algorithms._shared import (
    apply_mutation_with_rate,
    early_stop_decision,
    decorate_history_with_convergence,
    log_summary_row,
    maximize_vectors,
    problem_population_metrics,
    problem_solution_metrics,
    resolve_algorithm_reference_point,
    resolve_objective_directions,
    select_log_generation,
    validate_fitness_vector,
)
from ga_lab.config import GAConfig
from ga_lab.convergence_diagnostics import (
    ProgressState,
    build_generation_diagnostics,
    configured_evaluation_budget,
    progress_metric,
    update_progress_state,
    update_signal_window,
)
from ga_lab.core.crossover import CrossoverFn
from ga_lab.core.mutation import MutationFn
from ga_lab.core.representation import InitFn, Population
from ga_lab.core.selection import SelectionFn, SelectionState
from ga_lab.experiment.algorithm_checkpoint import CheckpointConfig
from ga_lab.experiment.nsga2_checkpoint import (
    build_nsga2_checkpoint_state_from_run_state,
    load_nsga2_checkpoint,
    restore_nsga2_rng_state,
    validate_nsga2_resume_compatibility,
    write_nsga2_checkpoint_atomic,
)
from ga_lab.experiment.nsga2_diagnostics import (
    Nsga2DiagnosticsConfig,
    Nsga2DiagnosticsRecorder,
)
from ga_lab.metrics import finite_or_none, front_metrics
from ga_lab.problems.base import as_fitness_vector, best_index, objective_count


def _dominates(a: list[float], b: list[float]) -> bool:
    better_or_equal = True
    strictly_better = False
    for a_value, b_value in zip(a, b, strict=True):
        if a_value < b_value:
            better_or_equal = False
            break
        if a_value > b_value:
            strictly_better = True
    return better_or_equal and strictly_better


def _nondominated_sort(population_values: list[list[float]]) -> tuple[list[list[int]], list[int]]:
    size = len(population_values)
    dominates: list[set[int]] = [set() for _ in range(size)]
    dominated_by: list[set[int]] = [set() for _ in range(size)]

    for p_idx in range(size):
        for q_idx in range(size):
            if p_idx == q_idx:
                continue
            if _dominates(population_values[p_idx], population_values[q_idx]):
                dominates[p_idx].add(q_idx)
                dominated_by[q_idx].add(p_idx)
            elif _dominates(population_values[q_idx], population_values[p_idx]):
                dominated_by[p_idx].add(q_idx)

    ranks = [-1] * size
    current_front: list[int] = [
        index for index, dominators in enumerate(dominated_by) if not dominators
    ]
    fronts: list[list[int]] = []
    for index in current_front:
        ranks[index] = 0
    current_rank = 0
    while current_front:
        fronts.append(current_front[:])
        next_front: list[int] = []
        for p_idx in current_front:
            for q_idx in dominates[p_idx]:
                dominated_by[q_idx].discard(p_idx)
                if not dominated_by[q_idx]:
                    ranks[q_idx] = current_rank + 1
                    next_front.append(q_idx)
        current_rank += 1
        current_front = next_front

    if len(fronts) == 0:
        fronts = [list(range(size))]
        ranks = [0] * size
    return fronts, ranks


def _crowding_distance(front: list[int], objective_vectors: list[list[float]]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) == 1:
        return {front[0]: float("inf")}
    if len(front) == 2:
        return {front[0]: float("inf"), front[1]: float("inf")}

    objective_len = len(objective_vectors[0])
    distances: dict[int, float] = {index: 0.0 for index in front}

    for obj_idx in range(objective_len):
        sorted_front = sorted(front, key=lambda idx: objective_vectors[idx][obj_idx])
        min_idx = sorted_front[0]
        max_idx = sorted_front[-1]
        distances[min_idx] = float("inf")
        distances[max_idx] = float("inf")
        min_value = objective_vectors[min_idx][obj_idx]
        max_value = objective_vectors[max_idx][obj_idx]
        if max_value == min_value:
            continue
        scale = max_value - min_value
        for left, center, right in zip(
            sorted_front,
            sorted_front[1:],
            sorted_front[2:],
            strict=False,
        ):
            distances[center] += (
                objective_vectors[right][obj_idx] - objective_vectors[left][obj_idx]
            ) / scale
    return distances


def _vector_signature(vector: list[float] | list[int], *, precision: int = 12) -> tuple[float, ...]:
    return tuple(round(float(value), precision) for value in vector)


def _decision_component_summary(candidate: list[float] | list[int]) -> dict[str, Any]:
    if not candidate:
        return {
            "x0": None,
            "tail_mean": None,
            "tail_min": None,
            "tail_max": None,
            "tail_std": None,
        }

    numeric = [float(value) for value in candidate]
    tail = numeric[1:]
    if not tail:
        return {
            "x0": float(numeric[0]),
            "tail_mean": None,
            "tail_min": None,
            "tail_max": None,
            "tail_std": None,
        }
    tail_mean = float(sum(tail) / len(tail))
    tail_variance = sum((value - tail_mean) ** 2 for value in tail) / len(tail)
    return {
        "x0": float(numeric[0]),
        "tail_mean": tail_mean,
        "tail_min": float(min(tail)),
        "tail_max": float(max(tail)),
        "tail_std": float(math.sqrt(tail_variance)),
    }


def _normalized_problem_name(problem: Any, config: GAConfig) -> str:
    problem_name = getattr(problem, "name", config.problem)
    return str(problem_name).strip().lower()


def _build_low_g_tail_stats(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(settings.get("enabled")),
        "zdt1_only": bool(settings.get("zdt1_only", True)),
        "trigger_probability": float(settings.get("trigger_probability", 0.0)),
        "gene_probability": float(settings.get("gene_probability", 0.0)),
        "step_scale": float(settings.get("step_scale", 0.0)),
        "applications": 0,
        "trigger_count": 0,
        "adjusted_solution_count": 0,
        "adjusted_gene_count": 0,
        "total_step": 0.0,
        "mean_step": None,
        "stage_application_counts": {},
        "stage_trigger_counts": {},
        "stage_adjusted_solution_counts": {},
        "stage_adjusted_gene_counts": {},
        "skipped_non_zdt1_problem_count": 0,
        "skipped_no_tail_count": 0,
        "skipped_non_numeric_bounds_count": 0,
    }


def _build_spread_preserving_stats(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(settings.get("enabled")),
        "zdt1_only": bool(settings.get("zdt1_only", True)),
        "trigger_probability": float(settings.get("trigger_probability", 0.0)),
        "x0_probability": float(settings.get("x0_probability", 0.0)),
        "x0_step_scale": float(settings.get("x0_step_scale", 0.0)),
        "tail_step_scale": float(settings.get("tail_step_scale", 0.0)),
        "applications": 0,
        "trigger_count": 0,
        "adjusted_solution_count": 0,
        "adjusted_gene_count": 0,
        "x0_adjusted_count": 0,
        "tail_adjusted_count": 0,
        "total_abs_step": 0.0,
        "mean_abs_step": None,
        "stage_application_counts": {},
        "stage_trigger_counts": {},
        "stage_adjusted_solution_counts": {},
        "stage_adjusted_gene_counts": {},
        "stage_x0_adjusted_counts": {},
        "stage_tail_adjusted_counts": {},
        "skipped_non_zdt1_problem_count": 0,
        "skipped_no_tail_count": 0,
        "skipped_non_numeric_bounds_count": 0,
    }


def _increment_nested_counter(counter_map: dict[str, Any], key: str) -> None:
    current = counter_map.get(key, 0)
    if isinstance(current, int):
        counter_map[key] = current + 1
    else:
        counter_map[key] = 1


def _apply_low_g_tail_mutation_light(
    candidate: list[float] | list[int],
    *,
    config: GAConfig,
    problem: Any,
    rng,
    settings: dict[str, Any] | None,
    stats: dict[str, Any] | None,
    stage: str,
) -> list[float] | list[int]:
    if not settings or not bool(settings.get("enabled")):
        return candidate

    if stats is not None:
        stats["applications"] = int(stats.get("applications", 0)) + 1
        _increment_nested_counter(
            stats.setdefault("stage_application_counts", {}),
            stage,
        )

    if bool(settings.get("zdt1_only", True)) and _normalized_problem_name(problem, config) != "zdt1":
        if stats is not None:
            stats["skipped_non_zdt1_problem_count"] = (
                int(stats.get("skipped_non_zdt1_problem_count", 0)) + 1
            )
        return candidate

    if len(candidate) <= 1:
        if stats is not None:
            stats["skipped_no_tail_count"] = int(stats.get("skipped_no_tail_count", 0)) + 1
        return candidate

    low = config.representation_options.get("low")
    high = config.representation_options.get("high")
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        if stats is not None:
            stats["skipped_non_numeric_bounds_count"] = (
                int(stats.get("skipped_non_numeric_bounds_count", 0)) + 1
            )
        return candidate

    trigger_probability = float(settings.get("trigger_probability", 0.0))
    gene_probability = float(settings.get("gene_probability", 0.0))
    step_scale = float(settings.get("step_scale", 0.0))
    if trigger_probability <= 0.0 or gene_probability <= 0.0 or step_scale <= 0.0:
        return candidate

    if rng.random() >= trigger_probability:
        return candidate

    if stats is not None:
        stats["trigger_count"] = int(stats.get("trigger_count", 0)) + 1
        _increment_nested_counter(stats.setdefault("stage_trigger_counts", {}), stage)

    lower = float(low)
    upper = float(high)
    span = upper - lower
    if not math.isfinite(span) or span <= 0.0:
        return candidate

    working = candidate[:]
    adjusted_gene_count = 0
    total_step = 0.0
    for index in range(1, len(working)):
        if rng.random() >= gene_probability:
            continue
        value = float(working[index])
        distance_to_low = max(0.0, value - lower)
        if distance_to_low <= 0.0:
            continue
        max_step = min(distance_to_low * 0.5, span * step_scale)
        if max_step <= 0.0 or not math.isfinite(max_step):
            continue
        step = rng.random() * max_step
        if step <= 0.0 or not math.isfinite(step):
            continue
        updated = max(lower, min(upper, value - step))
        if math.isclose(updated, value, rel_tol=1e-12, abs_tol=1e-12):
            continue
        working[index] = updated
        adjusted_gene_count += 1
        total_step += step

    if adjusted_gene_count <= 0:
        return candidate

    if stats is not None:
        stats["adjusted_solution_count"] = int(stats.get("adjusted_solution_count", 0)) + 1
        stats["adjusted_gene_count"] = int(stats.get("adjusted_gene_count", 0)) + adjusted_gene_count
        stats["total_step"] = float(stats.get("total_step", 0.0)) + total_step
        _increment_nested_counter(
            stats.setdefault("stage_adjusted_solution_counts", {}),
            stage,
        )
        stage_gene_counts = stats.setdefault("stage_adjusted_gene_counts", {})
        current_stage_gene_count = stage_gene_counts.get(stage, 0)
        stage_gene_counts[stage] = int(current_stage_gene_count) + adjusted_gene_count
    return working


def _apply_spread_preserving_variation_light(
    candidate: list[float] | list[int],
    *,
    config: GAConfig,
    problem: Any,
    rng,
    settings: dict[str, Any] | None,
    stats: dict[str, Any] | None,
    stage: str,
) -> list[float] | list[int]:
    if not settings or not bool(settings.get("enabled")):
        return candidate

    if stats is not None:
        stats["applications"] = int(stats.get("applications", 0)) + 1
        _increment_nested_counter(
            stats.setdefault("stage_application_counts", {}),
            stage,
        )

    if bool(settings.get("zdt1_only", True)) and _normalized_problem_name(problem, config) != "zdt1":
        if stats is not None:
            stats["skipped_non_zdt1_problem_count"] = (
                int(stats.get("skipped_non_zdt1_problem_count", 0)) + 1
            )
        return candidate

    if len(candidate) <= 1:
        if stats is not None:
            stats["skipped_no_tail_count"] = int(stats.get("skipped_no_tail_count", 0)) + 1
        return candidate

    low = config.representation_options.get("low")
    high = config.representation_options.get("high")
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        if stats is not None:
            stats["skipped_non_numeric_bounds_count"] = (
                int(stats.get("skipped_non_numeric_bounds_count", 0)) + 1
            )
        return candidate

    trigger_probability = float(settings.get("trigger_probability", 0.0))
    x0_probability = float(settings.get("x0_probability", 0.0))
    x0_step_scale = float(settings.get("x0_step_scale", 0.0))
    tail_step_scale = float(settings.get("tail_step_scale", 0.0))
    if trigger_probability <= 0.0 or max(x0_step_scale, tail_step_scale) <= 0.0:
        return candidate

    if rng.random() >= trigger_probability:
        return candidate

    if stats is not None:
        stats["trigger_count"] = int(stats.get("trigger_count", 0)) + 1
        _increment_nested_counter(stats.setdefault("stage_trigger_counts", {}), stage)

    lower = float(low)
    upper = float(high)
    span = upper - lower
    if not math.isfinite(span) or span <= 0.0:
        return candidate

    working = candidate[:]
    adjust_x0 = rng.random() < min(1.0, max(0.0, x0_probability))
    target_index = 0 if adjust_x0 else 1 + int(rng.randrange(len(working) - 1))
    step_scale = x0_step_scale if adjust_x0 else tail_step_scale
    if step_scale <= 0.0:
        return candidate

    current_value = float(working[target_index])
    max_step = span * step_scale
    if max_step <= 0.0 or not math.isfinite(max_step):
        return candidate
    step = (float(rng.random()) * 2.0 - 1.0) * max_step
    if math.isclose(step, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return candidate
    updated = max(lower, min(upper, current_value + step))
    if math.isclose(updated, current_value, rel_tol=1e-12, abs_tol=1e-12):
        return candidate
    working[target_index] = updated

    if stats is not None:
        stats["adjusted_solution_count"] = int(stats.get("adjusted_solution_count", 0)) + 1
        stats["adjusted_gene_count"] = int(stats.get("adjusted_gene_count", 0)) + 1
        stats["total_abs_step"] = float(stats.get("total_abs_step", 0.0)) + abs(step)
        _increment_nested_counter(
            stats.setdefault("stage_adjusted_solution_counts", {}),
            stage,
        )
        _increment_nested_counter(
            stats.setdefault("stage_adjusted_gene_counts", {}),
            stage,
        )
        if adjust_x0:
            stats["x0_adjusted_count"] = int(stats.get("x0_adjusted_count", 0)) + 1
            _increment_nested_counter(
                stats.setdefault("stage_x0_adjusted_counts", {}),
                stage,
            )
        else:
            stats["tail_adjusted_count"] = int(stats.get("tail_adjusted_count", 0)) + 1
            _increment_nested_counter(
                stats.setdefault("stage_tail_adjusted_counts", {}),
                stage,
            )
    return working


def _candidate_signature(
    index: int,
    population: Population | None,
    objective_vectors: list[list[float]] | None,
    mode: str,
) -> tuple[float, ...] | None:
    if mode == "decision" and population is not None:
        return _vector_signature(population[index])
    if mode == "objective" and objective_vectors is not None:
        return _vector_signature(objective_vectors[index])
    return None


def _order_partial_front(
    front: list[int],
    crowding: list[float],
    objective_vectors: list[list[float]] | None,
    selected_indices: list[int],
    strategy: str,
) -> list[int]:
    ordered = sorted(front, key=lambda idx: crowding[idx], reverse=True)
    if objective_vectors is None or len(ordered) <= 2:
        return ordered

    if strategy == "boundary_preservation_light":
        objective_len = len(objective_vectors[0])
        boundary_scores: dict[int, float] = {idx: 0.0 for idx in ordered}
        for obj_idx in range(objective_len):
            values = [objective_vectors[idx][obj_idx] for idx in ordered]
            min_value = min(values)
            max_value = max(values)
            scale = max_value - min_value
            if math.isclose(scale, 0.0, abs_tol=1e-12):
                continue
            for idx in ordered:
                normalized = (objective_vectors[idx][obj_idx] - min_value) / scale
                boundary_scores[idx] += abs((2.0 * normalized) - 1.0)
        return sorted(
            ordered,
            key=lambda idx: (
                0 if math.isinf(crowding[idx]) else 1,
                -(crowding[idx] if math.isfinite(crowding[idx]) else 0.0),
                -boundary_scores[idx],
            ),
        )

    if strategy != "novelty_crowding":
        return ordered

    remaining = ordered[:]
    chosen: list[int] = []
    anchor_indices = selected_indices[:]
    while remaining:
        if not chosen:
            next_index = max(remaining, key=lambda idx: crowding[idx])
        else:
            reference_indices = anchor_indices + chosen

            def _novelty_score(idx: int) -> tuple[float, float]:
                if not reference_indices:
                    return (float("inf"), crowding[idx])
                min_distance = min(
                    math.dist(objective_vectors[idx], objective_vectors[other_idx])
                    for other_idx in reference_indices
                )
                return (min_distance, crowding[idx])

            next_index = max(remaining, key=_novelty_score)
        chosen.append(next_index)
        remaining.remove(next_index)
    return chosen


def _select_by_rank_and_crowding(
    population_size: int,
    fronts: list[list[int]],
    crowding: list[float],
    ranks: list[int],
    *,
    population: Population | None = None,
    objective_vectors: list[list[float]] | None = None,
    partial_front_dedup_mode: str = "none",
    partial_front_strategy: str = "crowding",
) -> list[int]:
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
            continue

        remaining = population_size - len(selected)
        if remaining <= 0:
            break
        ordered_front = _order_partial_front(
            front,
            crowding,
            objective_vectors,
            selected,
            partial_front_strategy,
        )
        if (
            partial_front_dedup_mode in {"decision", "objective"}
            and population is not None
            and objective_vectors is not None
        ):
            selected_signatures = {
                signature
                for idx in selected
                if (signature := _candidate_signature(idx, population, objective_vectors, partial_front_dedup_mode))
                is not None
            }
            chosen: list[int] = []
            deferred: list[int] = []
            for idx in ordered_front:
                signature = _candidate_signature(
                    idx,
                    population,
                    objective_vectors,
                    partial_front_dedup_mode,
                )
                if signature is not None and signature in selected_signatures:
                    deferred.append(idx)
                    continue
                chosen.append(idx)
                if signature is not None:
                    selected_signatures.add(signature)
                if len(chosen) >= remaining:
                    break
            if len(chosen) < remaining:
                chosen.extend(deferred[: remaining - len(chosen)])
            selected.extend(chosen[:remaining])
        else:
            selected.extend(ordered_front[:remaining])
        break
    if len(selected) < population_size:
        fallback = [
            idx
            for idx in sorted(range(len(ranks)), key=lambda idx: (ranks[idx], -crowding[idx]))
            if idx not in selected
        ]
        selected.extend(fallback[: population_size - len(selected)])
    return selected[:population_size]


def _deduplicate_offspring_genome(
    candidate: list[float] | list[int],
    *,
    seen_signatures: set[tuple[float, ...]],
    config: GAConfig,
    problem: Any,
    mutation_fn: MutationFn,
    init_fn: InitFn,
    rng,
    mutation_rate: float,
    retry_count: int,
    retry_mutation_scale: float,
    reinitialize_fallback: bool,
    low_g_tail_settings: dict[str, Any] | None = None,
    low_g_tail_stats: dict[str, Any] | None = None,
    spread_preserving_settings: dict[str, Any] | None = None,
    spread_preserving_stats: dict[str, Any] | None = None,
) -> tuple[list[float] | list[int], dict[str, Any]]:
    initial_signature = _vector_signature(candidate)
    signature = initial_signature
    initial_decision_components = _decision_component_summary(candidate)
    retry_metadata: dict[str, Any] = {
        "duplicate_detected": signature in seen_signatures,
        "retry_attempt_count": 0,
        "retry_success": False,
        "retry_reinitialized": False,
        "decision_changed_after_retry": False,
        "initial_decision_hash": "|".join(f"{value:.12f}" for value in signature),
        "final_decision_hash": "|".join(f"{value:.12f}" for value in signature),
        "initial_decision_components": initial_decision_components,
        "final_decision_components": initial_decision_components,
    }
    if signature not in seen_signatures:
        seen_signatures.add(signature)
        return candidate, retry_metadata

    working = candidate[:]
    for attempt in range(max(0, retry_count)):
        retry_rate = min(1.0, mutation_rate * (retry_mutation_scale ** (attempt + 1)))
        working = apply_mutation_with_rate(
            config,
            mutation_fn,
            working,
            rng,
            mutation_rate=retry_rate,
        )
        working = _apply_low_g_tail_mutation_light(
            working,
            config=config,
            problem=problem,
            rng=rng,
            settings=low_g_tail_settings,
            stats=low_g_tail_stats,
            stage="duplicate_retry",
        )
        working = _apply_spread_preserving_variation_light(
            working,
            config=config,
            problem=problem,
            rng=rng,
            settings=spread_preserving_settings,
            stats=spread_preserving_stats,
            stage="duplicate_retry",
        )
        signature = _vector_signature(working)
        retry_metadata["retry_attempt_count"] = attempt + 1
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            retry_metadata["retry_success"] = True
            retry_metadata["decision_changed_after_retry"] = signature != initial_signature
            retry_metadata["final_decision_hash"] = "|".join(
                f"{value:.12f}" for value in signature
            )
            retry_metadata["final_decision_components"] = _decision_component_summary(working)
            return working, retry_metadata

    if reinitialize_fallback:
        for _ in range(3):
            reinitialized = init_fn(rng, config.genome_length)
            signature = _vector_signature(reinitialized)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                retry_metadata["retry_reinitialized"] = True
                retry_metadata["decision_changed_after_retry"] = signature != initial_signature
                retry_metadata["final_decision_hash"] = "|".join(
                    f"{value:.12f}" for value in signature
                )
                retry_metadata["final_decision_components"] = _decision_component_summary(
                    reinitialized
                )
                return reinitialized, retry_metadata

    seen_signatures.add(signature)
    retry_metadata["decision_changed_after_retry"] = signature != initial_signature
    retry_metadata["final_decision_hash"] = "|".join(f"{value:.12f}" for value in signature)
    retry_metadata["final_decision_components"] = _decision_component_summary(working)
    return working, retry_metadata


def run_nsga2(
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
    resume_enabled = bool(
        checkpoint_enabled and checkpoint_config is not None and checkpoint_config.resume_from is not None
    )
    resume_checkpoint = None
    resume_compatibility_report = None
    resume_rng_warnings: list[str] = []
    resume_cache_warnings: list[str] = []
    resume_metadata: dict[str, Any] | None = None
    options = dict(config.algorithm_options)
    diagnostics_config = Nsga2DiagnosticsConfig.from_algorithm_options(
        options,
        default_run_id=f"{config.run_name}_nsga2_diagnostics",
    )
    diagnostics_recorder = Nsga2DiagnosticsRecorder(
        config=diagnostics_config,
        algorithm=str(config.algorithm),
        problem=str(config.problem),
        seed=int(config.seed) if isinstance(config.seed, int) else None,
        candidate_id=(
            str(options.get("nsga2_trace_candidate_id"))
            if isinstance(options.get("nsga2_trace_candidate_id"), str)
            else None
        ),
    )
    offspring_decision_dedup = bool(options.get("nsga2_offspring_decision_dedup", False))
    duplicate_retry_count = max(
        0,
        int(options.get("nsga2_duplicate_retry_count", 0))
        if isinstance(options.get("nsga2_duplicate_retry_count", 0), int)
        and not isinstance(options.get("nsga2_duplicate_retry_count", 0), bool)
        else 0,
    )
    duplicate_retry_mutation_scale = (
        float(options.get("nsga2_duplicate_retry_mutation_scale", 1.0))
        if isinstance(options.get("nsga2_duplicate_retry_mutation_scale", 1.0), int | float)
        and not isinstance(options.get("nsga2_duplicate_retry_mutation_scale", 1.0), bool)
        else 1.0
    )
    if duplicate_retry_mutation_scale <= 0.0:
        duplicate_retry_mutation_scale = 1.0
    duplicate_reinitialize_fallback = bool(
        options.get("nsga2_duplicate_reinitialize_fallback", False)
    )
    partial_front_dedup_mode = str(
        options.get("nsga2_partial_front_dedup_mode", "none")
    ).strip().lower()
    if partial_front_dedup_mode not in {"none", "decision", "objective"}:
        partial_front_dedup_mode = "none"
    partial_front_strategy = str(
        options.get("nsga2_partial_front_strategy", "crowding")
    ).strip().lower()
    if partial_front_strategy not in {"crowding", "novelty_crowding", "boundary_preservation_light"}:
        partial_front_strategy = "crowding"
    low_g_tail_mutation_enabled = bool(options.get("nsga2_low_g_tail_mutation_light", False))
    low_g_tail_settings = {
        "enabled": low_g_tail_mutation_enabled,
        "trigger_probability": (
            float(options.get("nsga2_low_g_tail_probability", 0.0))
            if isinstance(options.get("nsga2_low_g_tail_probability", 0.0), int | float)
            and not isinstance(options.get("nsga2_low_g_tail_probability", 0.0), bool)
            else 0.0
        ),
        "gene_probability": (
            float(options.get("nsga2_low_g_tail_gene_probability", 0.0))
            if isinstance(options.get("nsga2_low_g_tail_gene_probability", 0.0), int | float)
            and not isinstance(options.get("nsga2_low_g_tail_gene_probability", 0.0), bool)
            else 0.0
        ),
        "step_scale": (
            float(options.get("nsga2_low_g_tail_step_scale", 0.0))
            if isinstance(options.get("nsga2_low_g_tail_step_scale", 0.0), int | float)
            and not isinstance(options.get("nsga2_low_g_tail_step_scale", 0.0), bool)
            else 0.0
        ),
        "zdt1_only": bool(options.get("nsga2_low_g_tail_zdt1_only", True)),
    }
    low_g_tail_stats = (
        _build_low_g_tail_stats(low_g_tail_settings)
        if low_g_tail_mutation_enabled
        else None
    )
    spread_preserving_enabled = bool(
        options.get("nsga2_spread_preserving_variation_light", False)
    )
    spread_preserving_settings = {
        "enabled": spread_preserving_enabled,
        "trigger_probability": (
            float(options.get("nsga2_spread_preserving_probability", 0.0))
            if isinstance(options.get("nsga2_spread_preserving_probability", 0.0), int | float)
            and not isinstance(options.get("nsga2_spread_preserving_probability", 0.0), bool)
            else 0.0
        ),
        "x0_probability": (
            float(options.get("nsga2_spread_preserving_x0_probability", 0.0))
            if isinstance(options.get("nsga2_spread_preserving_x0_probability", 0.0), int | float)
            and not isinstance(options.get("nsga2_spread_preserving_x0_probability", 0.0), bool)
            else 0.0
        ),
        "x0_step_scale": (
            float(options.get("nsga2_spread_preserving_x0_step_scale", 0.0))
            if isinstance(options.get("nsga2_spread_preserving_x0_step_scale", 0.0), int | float)
            and not isinstance(options.get("nsga2_spread_preserving_x0_step_scale", 0.0), bool)
            else 0.0
        ),
        "tail_step_scale": (
            float(options.get("nsga2_spread_preserving_tail_step_scale", 0.0))
            if isinstance(options.get("nsga2_spread_preserving_tail_step_scale", 0.0), int | float)
            and not isinstance(options.get("nsga2_spread_preserving_tail_step_scale", 0.0), bool)
            else 0.0
        ),
        "zdt1_only": bool(options.get("nsga2_spread_preserving_zdt1_only", True)),
    }
    spread_preserving_stats = (
        _build_spread_preserving_stats(spread_preserving_settings)
        if spread_preserving_enabled
        else None
    )
    configured_budget = configured_evaluation_budget(config)
    if resume_enabled and checkpoint_config is not None:
        resume_checkpoint = load_nsga2_checkpoint(checkpoint_config.resume_from)
        resume_directions = resolve_objective_directions(
            resume_checkpoint.metadata.objective_count,
            config,
            problem,
        )
        resume_compatibility_report = validate_nsga2_resume_compatibility(
            resume_checkpoint,
            config=config,
            objective_count=resume_checkpoint.metadata.objective_count,
            objective_directions=resume_directions,
            problem=problem,
            requested_budget=configured_budget,
        )
        if resume_compatibility_report.decision == "fail":
            raise ValueError(
                "NSGA-II checkpoint resume compatibility failed: "
                + "; ".join(resume_compatibility_report.failures)
            )
        resume_rng_warnings = restore_nsga2_rng_state(rng, resume_checkpoint.rng)
        if resume_rng_warnings:
            raise ValueError("NSGA-II checkpoint RNG restore failed: " + "; ".join(resume_rng_warnings))
        population = [list(genome) for genome in resume_checkpoint.population.decision_vectors]
    else:
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
                population = resumed_population
            else:
                population = [init_fn(rng, config.genome_length) for _ in range(config.population_size)]
        else:
            population = [init_fn(rng, config.genome_length) for _ in range(config.population_size)]
    stop_reason = "max_generations"
    convergence_generation: int | None = None
    history: list[dict[str, Any]] = (
        [dict(row) for row in resume_checkpoint.history]
        if resume_checkpoint is not None
        else []
    )
    reference_point: list[float] | None = None
    reference_metadata: dict[str, Any] = {}
    actual_evaluations_used = (
        int(resume_checkpoint.metadata.actual_evaluations)
        if resume_checkpoint is not None
        else 0
    )
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
    policy_name = adaptive_policy_name(options)
    configured_early_stop_policy = (
        str(options.get("early_stop_policy", "none"))
        if isinstance(options.get("early_stop_policy", "none"), str)
        else "none"
    )
    progress_state = ProgressState(maximize=True)
    if resume_enabled and history:
        for restored_row in history:
            restored_generation = restored_row.get("generation")
            if not isinstance(restored_generation, int):
                continue
            metric_name, progress_value, progress_maximize = progress_metric(
                config.problem,
                restored_row,
                config.maximize,
            )
            if progress_state.maximize != progress_maximize:
                progress_state.maximize = progress_maximize
            update_progress_state(
                progress_state,
                generation=restored_generation,
                value=progress_value,
            )
            if isinstance(restored_row.get("diversity_signal"), int | float):
                update_signal_window(
                    progress_state,
                    signal_key="diversity_signal",
                    output_prefix="diversity",
                    generation=restored_generation,
                    value=float(restored_row["diversity_signal"]),
                )
    current_mutation_rate = config.mutation_rate
    generations_since_last_improvement = 0
    early_stop_generation: int | None = None
    early_stop_triggered = False
    checkpoint_paths: list[str] = []
    resume_start_generation = (
        int(resume_checkpoint.metadata.generation_index)
        if resume_checkpoint is not None
        else 0
    )
    resume_objective_vectors = (
        [list(row) for row in resume_checkpoint.population.objective_values]
        if resume_checkpoint is not None
        else None
    )
    resume_source_history_length = len(history)

    def _crowding_cache_as_float(values: list[float | str] | None) -> list[float] | None:
        if values is None:
            return None
        converted: list[float] = []
        for value in values:
            if value == "inf":
                converted.append(math.inf)
            elif value == "-inf":
                converted.append(-math.inf)
            else:
                converted.append(float(value))
        return converted

    def _same_float_sequence(left: list[float], right: list[float]) -> bool:
        if len(left) != len(right):
            return False
        for left_value, right_value in zip(left, right, strict=True):
            if math.isinf(left_value) or math.isinf(right_value):
                if left_value != right_value:
                    return False
            elif not math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-12):
                return False
        return True

    def validate_restored_cache(
        *,
        generation: int,
        fronts: list[list[int]],
        ranks: list[int],
        crowding_values: list[float],
    ) -> None:
        if resume_checkpoint is None or generation != resume_start_generation:
            return
        cached_ranks = resume_checkpoint.population.ranks
        cached_crowding = _crowding_cache_as_float(resume_checkpoint.population.crowding_distances)
        cached_fronts = resume_checkpoint.population.front_indices
        if cached_ranks is not None and len(cached_ranks) != len(ranks):
            raise ValueError("NSGA-II checkpoint cached rank length mismatch")
        if cached_crowding is not None and len(cached_crowding) != len(crowding_values):
            raise ValueError("NSGA-II checkpoint cached crowding length mismatch")
        if cached_fronts is not None and sum(len(front) for front in cached_fronts) > len(ranks):
            raise ValueError("NSGA-II checkpoint cached front index shape mismatch")
        if cached_ranks is not None and cached_ranks != ranks:
            resume_cache_warnings.append("cached ranks differ from recomputed ranks")
        if cached_crowding is not None and not _same_float_sequence(cached_crowding, crowding_values):
            resume_cache_warnings.append("cached crowding differs from recomputed crowding")
        if cached_fronts is not None and cached_fronts != fronts:
            resume_cache_warnings.append("cached fronts differ from recomputed fronts")

    def maybe_write_checkpoint(
        *,
        generation: int,
        objective_vectors: list[list[float]],
        directions: list[bool],
        fronts: list[list[int]],
        ranks: list[int],
        crowding_values: list[float],
        completed: bool = False,
    ) -> str | None:
        if not checkpoint_enabled or checkpoint_config is None:
            return None
        if generation % checkpoint_config.interval_generations != 0:
            return None
        checkpoint_state = build_nsga2_checkpoint_state_from_run_state(
            config=config,
            problem=problem,
            rng=rng,
            generation_index=generation,
            actual_evaluations=actual_evaluations_used,
            requested_budget=configured_budget,
            population=population,
            objective_values=objective_vectors,
            objective_directions=directions,
            ranks=ranks,
            crowding_distances=crowding_values,
            front_indices=fronts,
            history=history,
            completed=completed,
        )
        path = write_nsga2_checkpoint_atomic(
            checkpoint_state,
            output_dir=checkpoint_config.output_dir,
            run_id=checkpoint_config.run_id,
            allow_overwrite=checkpoint_config.allow_overwrite,
            write_latest=checkpoint_config.write_latest,
        )
        checkpoint_paths.append(str(path))
        return str(path)

    def evaluate_population(
        population_slice: Population,
        *,
        generation: int | None,
        location: str,
    ) -> list[list[float]]:
        nonlocal actual_evaluations_used
        vectors: list[list[float]] = []
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
            vectors.append(values)
            actual_evaluations_used += 1
        return vectors

    for generation in range(resume_start_generation, config.generations + 1):
        if (
            resume_objective_vectors is not None
            and generation == resume_start_generation
        ):
            objective_vectors = [list(row) for row in resume_objective_vectors]
        else:
            objective_vectors = evaluate_population(
                population,
                generation=generation,
                location="nsga2_population",
            )
        objective_len = objective_count(objective_vectors[0])
        directions = resolve_objective_directions(objective_len, config, problem)
        if reference_point is None:
            reference_point, reference_metadata = resolve_algorithm_reference_point(
                config,
                problem,
                objective_vectors,
                directions,
            )
        adjusted = maximize_vectors(objective_vectors, directions)
        fronts, ranks = _nondominated_sort(adjusted)
        crowding_values = [0.0] * len(population)
        for front in fronts:
            distances = _crowding_distance(front, adjusted)
            for idx, value in distances.items():
                crowding_values[idx] = value
        validate_restored_cache(
            generation=generation,
            fronts=fronts,
            ranks=ranks,
            crowding_values=crowding_values,
        )

        restored_history_row = (
            history[-1]
            if resume_enabled
            and generation == resume_start_generation
            and bool(history)
            and history[-1].get("generation") == generation
            else None
        )
        if restored_history_row is not None:
            current_row = dict(restored_history_row)
        else:
            current_row = log_summary_row(generation, objective_vectors, directions[0])
            row_metrics = front_metrics(
                fronts[0] if fronts else [],
                objective_vectors,
                directions,
                reference_point,
                len(population),
            )
            current_row["pareto_front_size"] = int(row_metrics["pareto_front_size"])
            current_row["pareto_ratio"] = row_metrics["pareto_ratio"]
            current_row["hypervolume"] = row_metrics["hypervolume"]
            current_row["spread"] = row_metrics["spread"]
            current_row["objective_count"] = objective_len
            current_row.update(problem_population_metrics(problem, population))
            diagnostics = build_generation_diagnostics(
                config=config,
                problem=problem,
                population=population,
                scalar_values=[values[0] for values in objective_vectors],
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
        should_stop_early, early_stop_policy = early_stop_decision(
            generation=generation,
            row=current_row,
            options=options,
        )
        current_row["early_stop_policy"] = early_stop_policy
        current_row["early_stop_triggered"] = should_stop_early
        already_has_resume_row = (
            resume_enabled
            and generation == resume_start_generation
            and bool(history)
            and history[-1].get("generation") == generation
        )
        if select_log_generation(generation, config.log_every) and not already_has_resume_row:
            history.append(current_row)

        checkpoint_path = maybe_write_checkpoint(
            generation=generation,
            objective_vectors=objective_vectors,
            directions=directions,
            fronts=fronts,
            ranks=ranks,
            crowding_values=crowding_values,
            completed=generation == config.generations,
        )
        if (
            checkpoint_enabled
            and checkpoint_config is not None
            and checkpoint_config.stop_after_checkpoint_generation == generation
        ):
            summary = {
                "stop_reason": "checkpoint_debug_stop",
                "final_generation": generation,
                "objective_count": objective_len,
                "is_nsga2": True,
                "configured_evaluation_budget": configured_budget,
                "actual_evaluations_used": actual_evaluations_used,
                "checkpointing": {
                    "enabled": True,
                    "mode": "write_only",
                    "resume_implemented": False,
                    "last_checkpoint_path": checkpoint_path,
                    "checkpoint_paths": checkpoint_paths[:],
                    "debug_stop_after_checkpoint_generation": generation,
                    "debug_only": True,
                },
            }
            return summary, history

        if (
            objective_len == 1
            and convergence_generation is None
            and config.target_fitness is not None
            and (
                (
                    directions[0]
                    and objective_vectors[best_index(adjusted, True)][0] >= config.target_fitness
                )
                or (
                    (not directions[0])
                    and objective_vectors[best_index(adjusted, True)][0] <= config.target_fitness
                )
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

        diagnostics_recorder.begin_generation(
            generation=generation,
            population_size=len(population),
            population=population,
            objective_vectors=objective_vectors,
            fronts=fronts,
            ranks=ranks,
            crowding=crowding_values,
            directions=directions,
            evaluations_so_far=actual_evaluations_used,
        )

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
            options=options,
        )
        current_mutation_rate = mutation_decision.mutation_rate
        offspring: Population = []
        selection_state = SelectionState.from_pareto(
            ranks,
            crowding_values,
            diagnostics=diagnostics_recorder if diagnostics_recorder.trace_enabled else None,
        )
        offspring_signatures = (
            {_vector_signature(genome) for genome in population}
            if offspring_decision_dedup
            else set()
        )
        while len(offspring) < config.population_size:
            parent_event_offset = diagnostics_recorder.parent_event_count
            parent_a = selection_fn(population, selection_state, rng)
            parent_a_events = diagnostics_recorder.parent_events_since(parent_event_offset)
            parent_event_offset = diagnostics_recorder.parent_event_count
            parent_b = selection_fn(population, selection_state, rng)
            parent_b_events = diagnostics_recorder.parent_events_since(parent_event_offset)
            parent_trace_events = parent_a_events[-1:] + parent_b_events[-1:]

            if rng.random() < config.crossover_rate:
                child_a, child_b = crossover_fn(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a[:], parent_b[:]
            mutated_child_a = apply_mutation_with_rate(
                config,
                mutation_fn,
                child_a,
                rng,
                mutation_rate=current_mutation_rate,
            )
            mutated_child_a = _apply_low_g_tail_mutation_light(
                mutated_child_a,
                config=config,
                problem=problem,
                rng=rng,
                settings=low_g_tail_settings,
                stats=low_g_tail_stats,
                stage="offspring_mutation",
            )
            mutated_child_a = _apply_spread_preserving_variation_light(
                mutated_child_a,
                config=config,
                problem=problem,
                rng=rng,
                settings=spread_preserving_settings,
                stats=spread_preserving_stats,
                stage="offspring_mutation",
            )
            child_a_variation_metadata: dict[str, Any] = {
                "duplicate_detected": False,
                "retry_attempt_count": 0,
                "retry_success": False,
                "retry_reinitialized": False,
                "decision_changed_after_retry": False,
            }
            if offspring_decision_dedup:
                mutated_child_a, child_a_variation_metadata = _deduplicate_offspring_genome(
                    mutated_child_a,
                    seen_signatures=offspring_signatures,
                    config=config,
                    problem=problem,
                    mutation_fn=mutation_fn,
                    init_fn=init_fn,
                    rng=rng,
                    mutation_rate=current_mutation_rate,
                    retry_count=duplicate_retry_count,
                    retry_mutation_scale=duplicate_retry_mutation_scale,
                    reinitialize_fallback=duplicate_reinitialize_fallback,
                    low_g_tail_settings=low_g_tail_settings,
                    low_g_tail_stats=low_g_tail_stats,
                    spread_preserving_settings=spread_preserving_settings,
                    spread_preserving_stats=spread_preserving_stats,
                )
            diagnostics_recorder.record_offspring_creation(
                offspring_genome=mutated_child_a,
                parent_events=parent_trace_events,
                variation_metadata=child_a_variation_metadata,
            )
            offspring.append(mutated_child_a)
            if len(offspring) < config.population_size:
                mutated_child_b = apply_mutation_with_rate(
                    config,
                    mutation_fn,
                    child_b,
                    rng,
                    mutation_rate=current_mutation_rate,
                )
                mutated_child_b = _apply_low_g_tail_mutation_light(
                    mutated_child_b,
                    config=config,
                    problem=problem,
                    rng=rng,
                    settings=low_g_tail_settings,
                    stats=low_g_tail_stats,
                    stage="offspring_mutation",
                )
                mutated_child_b = _apply_spread_preserving_variation_light(
                    mutated_child_b,
                    config=config,
                    problem=problem,
                    rng=rng,
                    settings=spread_preserving_settings,
                    stats=spread_preserving_stats,
                    stage="offspring_mutation",
                )
                child_b_variation_metadata: dict[str, Any] = {
                    "duplicate_detected": False,
                    "retry_attempt_count": 0,
                    "retry_success": False,
                    "retry_reinitialized": False,
                    "decision_changed_after_retry": False,
                }
                if offspring_decision_dedup:
                    mutated_child_b, child_b_variation_metadata = _deduplicate_offspring_genome(
                        mutated_child_b,
                        seen_signatures=offspring_signatures,
                        config=config,
                        problem=problem,
                        mutation_fn=mutation_fn,
                        init_fn=init_fn,
                        rng=rng,
                        mutation_rate=current_mutation_rate,
                        retry_count=duplicate_retry_count,
                        retry_mutation_scale=duplicate_retry_mutation_scale,
                        reinitialize_fallback=duplicate_reinitialize_fallback,
                        low_g_tail_settings=low_g_tail_settings,
                        low_g_tail_stats=low_g_tail_stats,
                        spread_preserving_settings=spread_preserving_settings,
                        spread_preserving_stats=spread_preserving_stats,
                    )
                diagnostics_recorder.record_offspring_creation(
                    offspring_genome=mutated_child_b,
                    parent_events=parent_trace_events,
                    variation_metadata=child_b_variation_metadata,
                )
                offspring.append(mutated_child_b)

        offspring, adaptive_event, refreshed = maybe_refresh_population(
            offspring,
            config=config,
            init_fn=init_fn,
            rng=rng,
            options=options,
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
                metric_name = str(current_row.get("progress_metric_name", "hypervolume"))
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

        combined_population = population + offspring
        combined_vectors = evaluate_population(
            combined_population,
            generation=generation,
            location="nsga2_combined_population",
        )
        combined_adjusted = maximize_vectors(combined_vectors, directions)
        combined_fronts, combined_ranks = _nondominated_sort(combined_adjusted)
        combined_crowding = [0.0] * len(combined_population)
        for front in combined_fronts:
            distances = _crowding_distance(front, combined_adjusted)
            for idx, value in distances.items():
                combined_crowding[idx] = value
        survivors = _select_by_rank_and_crowding(
            config.population_size,
            combined_fronts,
            combined_crowding,
            combined_ranks,
            population=combined_population,
            objective_vectors=combined_vectors,
            partial_front_dedup_mode=partial_front_dedup_mode,
            partial_front_strategy=partial_front_strategy,
        )
        next_population = [combined_population[idx] for idx in survivors]
        next_objective_vectors = [combined_vectors[idx] for idx in survivors]
        diagnostics_recorder.record_generation_transition(
            base_population_size=len(population),
            next_population_size=len(next_population),
            next_population=next_population,
            next_objective_vectors=next_objective_vectors,
            combined_population=combined_population,
            combined_fronts=combined_fronts,
            combined_ranks=combined_ranks,
            combined_crowding=combined_crowding,
            combined_objective_vectors=combined_vectors,
            survivor_indices=survivors,
            partial_front_strategy=partial_front_strategy,
            partial_front_dedup_mode=partial_front_dedup_mode,
            evaluations_so_far=actual_evaluations_used,
        )
        population = next_population

    final_objective_vectors = evaluate_population(
        population,
        generation=history[-1]["generation"] if history else 0,
        location="nsga2_final_population",
    )
    directions = resolve_objective_directions(
        objective_count(final_objective_vectors[0]),
        config,
        problem,
    )
    final_adjusted = maximize_vectors(final_objective_vectors, directions)
    final_fronts, _ = _nondominated_sort(final_adjusted)
    if reference_point is None:
        reference_point, reference_metadata = resolve_algorithm_reference_point(
            config,
            problem,
            final_objective_vectors,
            directions,
        )
    final_metrics = front_metrics(
        final_fronts[0] if final_fronts else [],
        final_objective_vectors,
        directions,
        reference_point,
        len(population),
    )
    if directions[0]:
        best_idx = max(
            range(len(final_objective_vectors)),
            key=lambda idx: final_objective_vectors[idx][0],
        )
        worst_idx = min(
            range(len(final_objective_vectors)),
            key=lambda idx: final_objective_vectors[idx][0],
        )
    else:
        best_idx = min(
            range(len(final_objective_vectors)),
            key=lambda idx: final_objective_vectors[idx][0],
        )
        worst_idx = max(
            range(len(final_objective_vectors)),
            key=lambda idx: final_objective_vectors[idx][0],
        )
    final_scalars = [values[0] for values in final_objective_vectors]
    final_generation = history[-1]["generation"]
    mean_convergence_speed = decorate_history_with_convergence(
        history,
        finite_or_none(final_metrics["hypervolume"]),
    )
    summary = {
        "best_fitness": final_scalars[best_idx],
        "mean_fitness": mean(final_scalars),
        "worst_fitness": final_scalars[worst_idx],
        "best_fitness_vector": final_objective_vectors[best_idx],
        "best_genome": population[best_idx][:],
        "convergence_generation": convergence_generation,
        "stop_reason": stop_reason,
        "final_generation": final_generation,
        "pareto_front_size": int(final_metrics["pareto_front_size"]),
        "pareto_front_vectors": [
            list(final_objective_vectors[idx]) for idx in (final_fronts[0] if final_fronts else [])
        ],
        "pareto_ratio": final_metrics["pareto_ratio"],
        "hypervolume": finite_or_none(final_metrics["hypervolume"]),
        "spread": finite_or_none(final_metrics["spread"]),
        "objective_count": len(final_objective_vectors[0]),
        "objective_directions": [bool(value) for value in directions],
        "is_nsga2": True,
        "configured_evaluation_budget": configured_budget,
        "actual_evaluations_used": actual_evaluations_used,
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
    }
    if bool(options.get("_return_final_population")):
        summary["final_population"] = [genome[:] for genome in population]
        summary["pareto_front_genomes"] = [
            population[idx][:] for idx in (final_fronts[0] if final_fronts else [])
        ]
    if mean_convergence_speed is not None:
        summary["mean_convergence_speed"] = mean_convergence_speed
    if (
        first_trigger_metric_value is not None
        and isinstance(summary.get("hypervolume"), int | float)
    ):
        summary["post_trigger_improvement"] = (
            float(summary["hypervolume"]) - first_trigger_metric_value
        )
    diagnostics_payload = diagnostics_recorder.build_payload()
    if diagnostics_payload is not None:
        summary["nsga2_diagnostics"] = diagnostics_payload
    if low_g_tail_stats is not None:
        adjusted_gene_count = int(low_g_tail_stats.get("adjusted_gene_count", 0))
        total_step = float(low_g_tail_stats.get("total_step", 0.0))
        low_g_tail_stats["mean_step"] = (
            total_step / adjusted_gene_count if adjusted_gene_count > 0 else None
        )
        low_g_tail_stats["problem_applied"] = _normalized_problem_name(problem, config)
        summary["low_g_tail_mutation_stats"] = low_g_tail_stats
    if spread_preserving_stats is not None:
        adjusted_gene_count = int(spread_preserving_stats.get("adjusted_gene_count", 0))
        total_abs_step = float(spread_preserving_stats.get("total_abs_step", 0.0))
        spread_preserving_stats["mean_abs_step"] = (
            total_abs_step / adjusted_gene_count if adjusted_gene_count > 0 else None
        )
        spread_preserving_stats["problem_applied"] = _normalized_problem_name(problem, config)
        summary["spread_preserving_variation_stats"] = spread_preserving_stats
    if resume_enabled and checkpoint_config is not None and resume_checkpoint is not None:
        resume_metadata = {
            "enabled": True,
            "mode": "resume_from_checkpoint",
            "resume_source_checkpoint": str(checkpoint_config.resume_from),
            "resumed_from_generation": int(resume_checkpoint.metadata.generation_index),
            "resumed_from_actual_evaluations": int(resume_checkpoint.metadata.actual_evaluations),
            "remaining_budget": (
                int(configured_budget) - int(resume_checkpoint.metadata.actual_evaluations)
                if configured_budget is not None
                else None
            ),
            "history_source": "checkpoint",
            "history_source_length": resume_source_history_length,
            "history_continuation_policy": "append_after_checkpoint_generation_without_duplicate",
            "compatibility_decision": (
                resume_compatibility_report.decision
                if resume_compatibility_report is not None
                else None
            ),
            "compatibility_warnings": (
                resume_compatibility_report.warnings
                if resume_compatibility_report is not None
                else []
            ),
            "rank_crowding_cache_warnings": resume_cache_warnings[:],
            "warnings": resume_rng_warnings[:] + resume_cache_warnings[:],
        }
        summary["resume_metadata"] = resume_metadata
    if checkpoint_enabled and checkpoint_config is not None:
        summary["checkpointing"] = {
            "enabled": True,
            "mode": "resume" if resume_enabled else "write_only",
            "resume_implemented": bool(resume_enabled),
            "run_id": checkpoint_config.run_id,
            "output_dir": str(checkpoint_config.output_dir),
            "checkpoint_paths": checkpoint_paths[:],
            "last_checkpoint_path": checkpoint_paths[-1] if checkpoint_paths else None,
            "artifact_suffix": checkpoint_config.artifact_suffix,
        }
    summary.update(reference_metadata)
    summary.update(problem_solution_metrics(problem, population[best_idx]))
    summary.update(problem_population_metrics(problem, population))
    return summary, history
