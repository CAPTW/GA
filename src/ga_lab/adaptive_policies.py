from __future__ import annotations

from dataclasses import dataclass

from ga_lab.config import GAConfig
from ga_lab.core.mutation import (
    bit_flip_mutation,
    gaussian_mutation,
    inversion_mutation,
    swap_mutation,
)
from ga_lab.core.representation import Genome, InitFn, Population

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class AdaptiveMutationDecision:
    mutation_rate: float
    trigger_event: str = "none"


@dataclass(frozen=True, slots=True)
class AdaptiveModeDecision:
    mode: str
    switched: bool = False
    switch_event: str = "none"
    rescue_until_generation: int | None = None


SWITCHING_POLICIES = {"switch_controller_v1", "switch_controller_v2"}
TRIGGER_POLICIES = {
    "low_diversity_injection",
    "composite_trigger_v1",
    "composite_trigger_v2",
    "composite_trigger_v3",
    "delayed_trigger_v1",
    "delayed_trigger_v2",
    "periodic_refresh_control",
}


def _get_float(options: dict[str, object], key: str, default: float) -> float:
    value = options.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _get_int(options: dict[str, object], key: str, default: int) -> int:
    value = options.get(key, default)
    return int(value) if isinstance(value, int) else default


def _get_str(options: dict[str, object], key: str, default: str) -> str:
    value = options.get(key, default)
    return str(value).strip() if isinstance(value, str) and value.strip() else default


def adaptive_policy_name(options: dict[str, object]) -> str:
    return _get_str(options, "adaptive_policy", "none")


def is_switching_policy(policy: str) -> bool:
    return policy in SWITCHING_POLICIES


def effective_policy_for_mode(policy: str, mode: str) -> str:
    if policy in SWITCHING_POLICIES:
        return "low_diversity_injection" if mode == "trigger" else "decay_mutation"
    return policy


def fixed_policy_mode(policy: str) -> str:
    if policy == "decay_mutation":
        return "decay"
    if policy in TRIGGER_POLICIES:
        return "trigger"
    return "none"


def _cooldown_ready(
    *,
    generation: int,
    last_event_generation: int | None,
    cooldown: int,
) -> bool:
    return last_event_generation is None or generation - last_event_generation >= max(1, cooldown)


def _warmup_ready(
    *,
    generation: int,
    options: dict[str, object],
) -> bool:
    warmup_generation = max(0, _get_int(options, "warmup_generation", 0))
    return generation >= warmup_generation


def _mutation_bounds(config: GAConfig, options: dict[str, object]) -> tuple[float, float]:
    base_rate = config.mutation_rate
    min_rate = max(
        0.0,
        min(
            1.0,
            _get_float(options, "adaptive_mutation_min_rate", base_rate * 0.5),
        ),
    )
    configured_cap = _get_float(
        options,
        "mutation_cap",
        _get_float(options, "adaptive_mutation_max_rate", base_rate * 4.0),
    )
    max_rate = max(
        base_rate,
        min(
            1.0,
            configured_cap,
        ),
    )
    return min_rate, max_rate


def _boosted_rate(config: GAConfig, options: dict[str, object]) -> float:
    _, max_rate = _mutation_bounds(config, options)
    boost_factor = max(1.0, _get_float(options, "stagnation_boost_factor", 1.5))
    boost_factor = max(
        boost_factor,
        _get_float(options, "mutation_boost_multiplier", boost_factor),
    )
    return min(max_rate, config.mutation_rate * boost_factor)


def _low_diversity_trigger(
    *,
    diversity_signal: float | None,
    options: dict[str, object],
) -> bool:
    threshold = _get_float(options, "diversity_threshold", 0.15)
    return diversity_signal is not None and diversity_signal <= threshold


def _delayed_low_diversity_trigger(
    *,
    generation: int,
    diversity_signal: float | None,
    options: dict[str, object],
) -> bool:
    return _warmup_ready(generation=generation, options=options) and _low_diversity_trigger(
        diversity_signal=diversity_signal,
        options=options,
    )


def _composite_trigger_active(
    policy: str,
    *,
    generation: int,
    generations_since_last_improvement: int,
    diversity_signal: float | None,
    recent_window_improvement: float | None,
    diversity_recent_slope: float | None,
    options: dict[str, object],
) -> bool:
    if not _low_diversity_trigger(diversity_signal=diversity_signal, options=options):
        return False

    improvement_epsilon = _get_float(options, "improvement_epsilon", 0.5)
    stagnation_window = max(1, _get_int(options, "stagnation_window", 6))
    no_recent_progress = (
        recent_window_improvement is None or recent_window_improvement <= improvement_epsilon
    )
    if policy == "composite_trigger_v1":
        return no_recent_progress
    if policy == "composite_trigger_v2":
        return generations_since_last_improvement >= stagnation_window
    if policy == "composite_trigger_v3":
        slope_threshold = max(0.0, _get_float(options, "diversity_slope_threshold", 0.0))
        return (
            no_recent_progress
            and diversity_recent_slope is not None
            and diversity_recent_slope < -(slope_threshold + _EPSILON)
        )
    if policy == "delayed_trigger_v1":
        return _delayed_low_diversity_trigger(
            generation=generation,
            diversity_signal=diversity_signal,
            options=options,
        )
    if policy == "delayed_trigger_v2":
        return _delayed_low_diversity_trigger(
            generation=generation,
            diversity_signal=diversity_signal,
            options=options,
        ) and generations_since_last_improvement >= stagnation_window
    return False


def _switch_collapse_active(
    policy: str,
    *,
    generation: int,
    generations_since_last_improvement: int,
    diversity_signal: float | None,
    recent_window_improvement: float | None,
    diversity_recent_slope: float | None,
    options: dict[str, object],
) -> bool:
    del diversity_recent_slope
    if not _warmup_ready(generation=generation, options=options):
        return False
    if not _low_diversity_trigger(diversity_signal=diversity_signal, options=options):
        return False

    improvement_epsilon = _get_float(options, "improvement_epsilon", 0.5)
    no_recent_progress = (
        recent_window_improvement is None or recent_window_improvement <= improvement_epsilon
    )
    if not no_recent_progress:
        return False

    if policy == "switch_controller_v2":
        stagnation_window = max(1, _get_int(options, "stagnation_window", 6))
        return generations_since_last_improvement >= stagnation_window

    return True


def online_switch_mode_decision(
    *,
    policy: str,
    generation: int,
    current_mode: str,
    rescue_until_generation: int | None,
    generations_since_last_improvement: int,
    diversity_signal: float | None,
    recent_window_improvement: float | None,
    diversity_recent_slope: float | None,
    options: dict[str, object],
) -> AdaptiveModeDecision:
    if policy not in SWITCHING_POLICIES:
        return AdaptiveModeDecision(
            mode=current_mode,
            rescue_until_generation=rescue_until_generation,
        )

    if (
        policy == "switch_controller_v2"
        and current_mode == "trigger"
        and rescue_until_generation is not None
        and generation >= rescue_until_generation
    ):
        return AdaptiveModeDecision(
            mode="decay",
            switched=True,
            switch_event="switch_to_decay",
            rescue_until_generation=None,
        )

    collapse_active = _switch_collapse_active(
        policy,
        generation=generation,
        generations_since_last_improvement=generations_since_last_improvement,
        diversity_signal=diversity_signal,
        recent_window_improvement=recent_window_improvement,
        diversity_recent_slope=diversity_recent_slope,
        options=options,
    )

    if current_mode == "decay" and collapse_active:
        rescue_until = rescue_until_generation
        if policy == "switch_controller_v2":
            rescue_window = max(1, _get_int(options, "rescue_window", 5))
            rescue_until = generation + rescue_window
        return AdaptiveModeDecision(
            mode="trigger",
            switched=True,
            switch_event="switch_to_trigger",
            rescue_until_generation=rescue_until,
        )

    return AdaptiveModeDecision(
        mode=current_mode,
        rescue_until_generation=rescue_until_generation,
    )


def adaptive_mutation_decision(
    config: GAConfig,
    *,
    generation: int,
    generations_since_last_improvement: int,
    diversity_signal: float | None,
    recent_window_improvement: float | None,
    diversity_recent_slope: float | None,
    feasible_ratio: float | None,
    mean_constraint_violation: float | None,
    recent_constraint_violation_slope: float | None,
    last_event_generation: int | None,
    options: dict[str, object],
) -> AdaptiveMutationDecision:
    base_rate = config.mutation_rate
    policy = adaptive_policy_name(options)
    min_rate, max_rate = _mutation_bounds(config, options)
    stagnation_window = max(1, _get_int(options, "stagnation_window", 6))

    if policy == "decay_mutation":
        progress = generation / max(1, config.generations)
        power = max(0.1, _get_float(options, "adaptive_decay_power", 1.0))
        rate = base_rate - ((base_rate - min_rate) * (progress**power))
        end_rate = _get_float(options, "decay_end_rate", min_rate)
        return AdaptiveMutationDecision(max(end_rate, min(max_rate, rate)))
    if policy == "stagnation_boost":
        if generations_since_last_improvement >= stagnation_window:
            return AdaptiveMutationDecision(_boosted_rate(config, options))
        return AdaptiveMutationDecision(base_rate)
    if policy == "low_diversity_injection":
        if _low_diversity_trigger(diversity_signal=diversity_signal, options=options):
            return AdaptiveMutationDecision(_boosted_rate(config, options))
        return AdaptiveMutationDecision(base_rate)
    if policy in {
        "composite_trigger_v1",
        "composite_trigger_v2",
        "composite_trigger_v3",
        "delayed_trigger_v1",
        "delayed_trigger_v2",
    }:
        if _composite_trigger_active(
            policy,
            generation=generation,
            generations_since_last_improvement=generations_since_last_improvement,
            diversity_signal=diversity_signal,
            recent_window_improvement=recent_window_improvement,
            diversity_recent_slope=diversity_recent_slope,
            options=options,
        ):
            return AdaptiveMutationDecision(_boosted_rate(config, options))
        return AdaptiveMutationDecision(base_rate)
    if policy in {
        "feasibility_aware_mutation_v1",
        "feasibility_aware_mutation_v2",
        "feasibility_aware_mutation_v3",
    }:
        cooldown = max(1, _get_int(options, "adaptation_cooldown", 5))
        if not _cooldown_ready(
            generation=generation,
            last_event_generation=last_event_generation,
            cooldown=cooldown,
        ):
            return AdaptiveMutationDecision(base_rate)

        feasible_threshold = _get_float(options, "feasible_ratio_threshold", 0.4)
        recovery_threshold = _get_float(
            options,
            "feasible_ratio_recovery_threshold",
            min(1.0, feasible_threshold + 0.2),
        )
        plateau_window = max(1, _get_int(options, "violation_plateau_window", 10))
        plateau_epsilon = _get_float(options, "violation_plateau_epsilon", 0.0025)
        low_feasible_ratio = feasible_ratio is not None and feasible_ratio < feasible_threshold
        violation_plateau = (
            mean_constraint_violation is not None
            and mean_constraint_violation > 0.0
            and recent_constraint_violation_slope is not None
            and abs(recent_constraint_violation_slope) <= plateau_epsilon
            and generations_since_last_improvement >= plateau_window
        )

        should_boost = False
        if policy == "feasibility_aware_mutation_v1":
            should_boost = low_feasible_ratio
        elif policy == "feasibility_aware_mutation_v2":
            should_boost = violation_plateau
        else:
            should_boost = low_feasible_ratio or (
                violation_plateau
                and (feasible_ratio is None or feasible_ratio < recovery_threshold)
            )

        if should_boost:
            return AdaptiveMutationDecision(
                mutation_rate=min(max_rate, _boosted_rate(config, options)),
                trigger_event="mutation_boost",
            )
        return AdaptiveMutationDecision(base_rate)
    return AdaptiveMutationDecision(base_rate)


def apply_adaptive_mutation(
    config: GAConfig,
    genome: Genome,
    rng,
    *,
    mutation_rate: float,
) -> Genome:
    if config.mutation == "bit_flip":
        return bit_flip_mutation(genome, mutation_rate, rng)
    if config.mutation == "gaussian":
        sigma = _get_float(config.mutation_options, "sigma", 0.1)
        low = _get_float(config.representation_options, "low", 0.0)
        high = _get_float(config.representation_options, "high", 1.0)
        return gaussian_mutation(
            genome,
            mutation_rate,
            rng,
            sigma=sigma,
            low=low,
            high=high,
        )
    if config.mutation == "swap":
        return swap_mutation(genome, mutation_rate, rng)
    if config.mutation == "inversion":
        return inversion_mutation(genome, mutation_rate, rng)
    return genome[:]


def maybe_refresh_population(
    population: Population,
    *,
    config: GAConfig,
    init_fn: InitFn,
    rng,
    options: dict[str, object],
    generations_since_last_improvement: int,
    diversity_signal: float | None,
    recent_window_improvement: float | None,
    diversity_recent_slope: float | None,
    last_event_generation: int | None,
    generation: int,
) -> tuple[Population, str, int]:
    policy = adaptive_policy_name(options)
    if policy not in {
        "stagnation_restart",
        "low_diversity_injection",
        "composite_trigger_v1",
        "composite_trigger_v2",
        "composite_trigger_v3",
        "delayed_trigger_v1",
        "delayed_trigger_v2",
        "periodic_refresh_control",
    }:
        return population, "none", 0

    preserve_count = min(config.elitism, len(population))
    refresh_fraction = max(0.0, min(0.9, _get_float(options, "refresh_fraction", 0.25)))
    refresh_count = min(
        len(population) - preserve_count,
        max(0, int(round(config.population_size * refresh_fraction))),
    )
    if refresh_count <= 0:
        return population, "none", 0

    event_name = "none"
    if policy == "stagnation_restart":
        stagnation_window = max(1, _get_int(options, "stagnation_window", 6))
        cooldown = max(1, _get_int(options, "adaptation_cooldown", stagnation_window))
        cooled_down = (
            last_event_generation is None
            or generation - last_event_generation >= cooldown
        )
        if generations_since_last_improvement >= stagnation_window and cooled_down:
            event_name = "restart"
    elif policy == "low_diversity_injection":
        cooldown = max(1, _get_int(options, "adaptation_cooldown", 3))
        cooled_down = _cooldown_ready(
            generation=generation,
            last_event_generation=last_event_generation,
            cooldown=cooldown,
        )
        if (
            _low_diversity_trigger(
                diversity_signal=diversity_signal,
                options=options,
            )
            and cooled_down
        ):
            event_name = "diversity_injection"
    elif policy == "periodic_refresh_control":
        warmup_generation = max(0, _get_int(options, "warmup_generation", 0))
        interval = max(1, _get_int(options, "periodic_refresh_every", 4))
        if generation >= warmup_generation and (generation - warmup_generation) % interval == 0:
            event_name = "periodic_refresh"
    else:
        cooldown = max(1, _get_int(options, "adaptation_cooldown", 5))
        cooled_down = _cooldown_ready(
            generation=generation,
            last_event_generation=last_event_generation,
            cooldown=cooldown,
        )
        if (
            _composite_trigger_active(
                policy,
                generation=generation,
                generations_since_last_improvement=generations_since_last_improvement,
                diversity_signal=diversity_signal,
                recent_window_improvement=recent_window_improvement,
                diversity_recent_slope=diversity_recent_slope,
                options=options,
            )
            and cooled_down
        ):
            event_name = "diversity_injection"

    if event_name == "none":
        return population, "none", 0

    refreshed = [genome[:] for genome in population]
    for index in range(len(refreshed) - refresh_count, len(refreshed)):
        refreshed[index] = init_fn(rng, config.genome_length)
    return refreshed, event_name, refresh_count
