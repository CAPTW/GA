from __future__ import annotations

import random
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ga_lab.core.representation import Genome, Population

if TYPE_CHECKING:
    from ga_lab.config import GAConfig


@dataclass(slots=True)
class SelectionState:
    fitnesses: Sequence[float] | None = None
    maximize: bool = True
    ranks: Sequence[int] | None = None
    crowding: Sequence[float] | None = None
    diagnostics: Any | None = None

    @classmethod
    def from_fitnesses(
        cls,
        fitnesses: Sequence[float],
        maximize: bool,
        diagnostics: Any | None = None,
    ) -> SelectionState:
        return cls(fitnesses=fitnesses, maximize=maximize, diagnostics=diagnostics)

    @classmethod
    def from_pareto(
        cls,
        ranks: Sequence[int],
        crowding: Sequence[float],
        diagnostics: Any | None = None,
    ) -> SelectionState:
        return cls(ranks=ranks, crowding=crowding, diagnostics=diagnostics)


SelectionFn = Callable[[Population, SelectionState, random.Random], Genome]
SelectionBuilder = Callable[["GAConfig"], SelectionFn]


@dataclass(frozen=True, slots=True)
class SelectionPluginSpec:
    name: str
    build_fn: SelectionBuilder
    compatible_algorithms: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def supports_algorithm(self, algorithm: str) -> bool:
        normalized = _normalize_algorithm_name(algorithm)
        return normalized in {
            _normalize_algorithm_name(value) for value in self.compatible_algorithms
        }


def _get_int(value: object, default: int) -> int:
    return int(value) if isinstance(value, int) else default


def _get_float(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _normalize_algorithm_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def tournament_select(
    population: Population,
    fitnesses: Sequence[float],
    tournament_size: int,
    maximize: bool,
    rng: random.Random,
) -> Genome:
    selected_indices = [rng.randrange(len(population)) for _ in range(tournament_size)]
    key_fn = (lambda idx: fitnesses[idx]) if maximize else (lambda idx: -fitnesses[idx])
    winner_idx = max(selected_indices, key=key_fn)
    return population[winner_idx][:]


def rank_select(
    population: Population,
    fitnesses: Sequence[float],
    tournament_size: int,
    maximize: bool,
    rng: random.Random,
) -> Genome:
    if len(population) == 0:
        raise ValueError("population must not be empty")
    if tournament_size <= 0:
        tournament_size = len(population)
    ranked = sorted(range(len(population)), key=fitnesses.__getitem__, reverse=maximize)
    base_weights = [len(population) - idx for idx in range(len(population))]
    candidates = rng.choices(ranked, weights=base_weights, k=tournament_size)
    winner_idx = (
        max(candidates, key=fitnesses.__getitem__)
        if maximize
        else min(candidates, key=fitnesses.__getitem__)
    )
    return population[winner_idx][:]


def roulette_wheel_select(
    population: Population,
    fitnesses: Sequence[float],
    _tournament_size: int,
    maximize: bool,
    rng: random.Random,
) -> Genome:
    if len(population) == 0:
        raise ValueError("population must not be empty")
    weights = _roulette_weights(fitnesses, maximize)
    selected_idx = rng.choices(range(len(population)), weights=weights, k=1)[0]
    return population[selected_idx][:]


def _roulette_weights(fitnesses: Sequence[float], maximize: bool) -> list[float]:
    if not fitnesses:
        raise ValueError("fitnesses must not be empty")
    values: list[float] = []
    for value in fitnesses:
        if not isinstance(value, int | float):
            raise ValueError("roulette selection requires finite numeric fitness values")
        value_float = float(value)
        if not math.isfinite(value_float):
            raise ValueError("roulette selection requires finite numeric fitness values")
        values.append(value_float)

    if maximize:
        shifted = [value - min(values) for value in values]
    else:
        shifted = [max(values) - value for value in values]
    if all(value == 0.0 for value in shifted):
        return [1.0 for _ in shifted]
    return [value + 1e-12 for value in shifted]


def crowded_tournament_select(
    population: Population,
    ranks: Sequence[int],
    crowding: Sequence[float],
    tournament_size: int,
    rng: random.Random,
    diagnostics: Any | None = None,
) -> Genome:
    winner_idx = crowded_tournament_index(
        population,
        ranks,
        crowding,
        tournament_size,
        rng,
        diagnostics=diagnostics,
    )
    return population[winner_idx][:]


def crowded_tournament_index(
    population: Population,
    ranks: Sequence[int],
    crowding: Sequence[float],
    tournament_size: int,
    rng: random.Random,
    *,
    diagnostics: Any | None = None,
) -> int:
    if len(population) == 0:
        raise ValueError("population must not be empty")
    if len(ranks) != len(population) or len(crowding) != len(population):
        raise ValueError("ranks and crowding must match population size")
    if tournament_size <= 0:
        tournament_size = 2
    sample_size = min(len(population), tournament_size)
    candidate_indices = rng.sample(range(len(population)), sample_size)
    winner_idx = min(
        candidate_indices,
        key=lambda idx: (ranks[idx], -crowding[idx]),
    )
    _record_selection_trace(
        diagnostics,
        selection_kind="crowded_tournament",
        winner_index=winner_idx,
        candidate_indices=candidate_indices,
    )
    return winner_idx


def _record_selection_trace(
    diagnostics: Any | None,
    *,
    selection_kind: str,
    winner_index: int,
    candidate_indices: Sequence[int],
    reference_distance: float | None = None,
    bias_applied: bool = False,
) -> None:
    if diagnostics is None:
        return
    record_fn = getattr(diagnostics, "record_parent_selection", None)
    if not callable(record_fn):
        return
    record_fn(
        selection_kind=selection_kind,
        winner_index=winner_index,
        candidate_indices=list(candidate_indices),
        reference_distance=reference_distance,
        bias_applied=bias_applied,
    )


def _euclidean_distance(left: Genome, right: Genome) -> float:
    if len(left) != len(right):
        return 0.0
    squared = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        squared += (left_value - right_value) ** 2
    return math.sqrt(squared)


def _sparse_parent_bias_light_index(
    population: Population,
    ranks: Sequence[int],
    crowding: Sequence[float],
    *,
    candidate_pool_size: int,
    reference_parent: Genome,
    rng: random.Random,
    diagnostics: Any | None = None,
) -> int:
    if len(population) == 0:
        raise ValueError("population must not be empty")
    if len(ranks) != len(population) or len(crowding) != len(population):
        raise ValueError("ranks and crowding must match population size")
    sample_size = min(len(population), max(2, candidate_pool_size))
    candidate_indices = rng.sample(range(len(population)), sample_size)
    best_rank = min(ranks[idx] for idx in candidate_indices)
    best_rank_candidates = [idx for idx in candidate_indices if ranks[idx] == best_rank]
    ordered = sorted(best_rank_candidates, key=lambda idx: crowding[idx], reverse=True)
    if len(ordered) <= 1:
        winner_idx = ordered[0]
        _record_selection_trace(
            diagnostics,
            selection_kind="sparse_parent_bias_light",
            winner_index=winner_idx,
            candidate_indices=candidate_indices,
            reference_distance=_euclidean_distance(population[winner_idx], reference_parent),
            bias_applied=True,
        )
        return winner_idx
    finalists = ordered[: min(2, len(ordered))]
    winner_idx = max(
        finalists,
        key=lambda idx: (
            _euclidean_distance(population[idx], reference_parent),
            crowding[idx],
        ),
    )
    _record_selection_trace(
        diagnostics,
        selection_kind="sparse_parent_bias_light",
        winner_index=winner_idx,
        candidate_indices=candidate_indices,
        reference_distance=_euclidean_distance(population[winner_idx], reference_parent),
        bias_applied=True,
    )
    return winner_idx


def _fitnesses_from_state(state: SelectionState) -> Sequence[float]:
    if state.fitnesses is None:
        raise ValueError("selection strategy requires scalar fitness values")
    return state.fitnesses


def _ranks_and_crowding_from_state(
    state: SelectionState,
) -> tuple[Sequence[int], Sequence[float]]:
    if state.ranks is None or state.crowding is None:
        raise ValueError("selection strategy requires Pareto ranks and crowding distances")
    return state.ranks, state.crowding


def _selection_tournament_size(config: GAConfig) -> int:
    return _get_int(
        config.selection_options.get("tournament_size"),
        config.tournament_size,
    )


def _build_tournament_selection(config: GAConfig) -> SelectionFn:
    tournament_size = _selection_tournament_size(config)
    algorithm = _normalize_algorithm_name(config.algorithm)
    if algorithm in {"nsga2", "nsga-ii"}:
        sparse_parent_bias_light = bool(
            config.selection_options.get("nsga2_sparse_parent_bias_light", False)
        )
        if sparse_parent_bias_light:
            sparse_parent_bias_probability = _get_float(
                config.selection_options.get("nsga2_sparse_parent_bias_probability"),
                0.15,
            )
            sparse_parent_bias_probability = max(0.0, min(1.0, sparse_parent_bias_probability))
            sparse_parent_bias_pool = _get_int(
                config.selection_options.get("nsga2_sparse_parent_bias_pool"),
                max(tournament_size + 1, 3),
            )
            awaiting_partner = False
            previous_parent: Genome | None = None

            def _select_sparse_parent_bias(
                population: Population,
                state: SelectionState,
                rng: random.Random,
            ) -> Genome:
                nonlocal awaiting_partner, previous_parent
                ranks, crowding = _ranks_and_crowding_from_state(state)
                if not awaiting_partner:
                    winner_idx = crowded_tournament_index(
                        population,
                        ranks,
                        crowding,
                        tournament_size,
                        rng,
                        diagnostics=state.diagnostics,
                    )
                    previous_parent = population[winner_idx][:]
                    awaiting_partner = True
                    return population[winner_idx][:]

                awaiting_partner = False
                if (
                    previous_parent is None
                    or rng.random() >= sparse_parent_bias_probability
                ):
                    winner_idx = crowded_tournament_index(
                        population,
                        ranks,
                        crowding,
                        tournament_size,
                        rng,
                        diagnostics=state.diagnostics,
                    )
                    return population[winner_idx][:]

                winner_idx = _sparse_parent_bias_light_index(
                    population,
                    ranks,
                    crowding,
                    candidate_pool_size=sparse_parent_bias_pool,
                    reference_parent=previous_parent,
                    rng=rng,
                    diagnostics=state.diagnostics,
                )
                return population[winner_idx][:]

            return _select_sparse_parent_bias
        return lambda population, state, rng: crowded_tournament_select(
            population,
            *_ranks_and_crowding_from_state(state),
            tournament_size,
            rng,
            state.diagnostics,
        )
    return lambda population, state, rng: tournament_select(
        population,
        _fitnesses_from_state(state),
        tournament_size,
        state.maximize,
        rng,
    )


def _build_rank_selection(config: GAConfig) -> SelectionFn:
    tournament_size = _selection_tournament_size(config)
    return lambda population, state, rng: rank_select(
        population,
        _fitnesses_from_state(state),
        tournament_size,
        state.maximize,
        rng,
    )


def _build_roulette_selection(config: GAConfig) -> SelectionFn:
    tournament_size = _selection_tournament_size(config)
    return lambda population, state, rng: roulette_wheel_select(
        population,
        _fitnesses_from_state(state),
        tournament_size,
        state.maximize,
        rng,
    )


def _build_crowded_selection(config: GAConfig) -> SelectionFn:
    tournament_size = _selection_tournament_size(config)
    return lambda population, state, rng: crowded_tournament_select(
        population,
        *_ranks_and_crowding_from_state(state),
        tournament_size,
        rng,
        state.diagnostics,
    )


SELECTION_SPECS = (
    SelectionPluginSpec(
        name="tournament",
        build_fn=_build_tournament_selection,
        compatible_algorithms=("ga", "hybrid_ga", "memetic_ga", "nsga2", "nsga_ii"),
    ),
    SelectionPluginSpec(
        name="rank",
        build_fn=_build_rank_selection,
        compatible_algorithms=("ga", "hybrid_ga", "memetic_ga"),
    ),
    SelectionPluginSpec(
        name="roulette",
        build_fn=_build_roulette_selection,
        compatible_algorithms=("ga", "hybrid_ga", "memetic_ga"),
    ),
    SelectionPluginSpec(
        name="crowded_tournament",
        build_fn=_build_crowded_selection,
        compatible_algorithms=("nsga2", "nsga_ii"),
    ),
    SelectionPluginSpec(
        name="nsga2_tournament",
        build_fn=_build_crowded_selection,
        compatible_algorithms=("nsga2", "nsga_ii"),
    ),
)


def _build_registry() -> dict[str, SelectionPluginSpec]:
    registry: dict[str, SelectionPluginSpec] = {}
    for spec in SELECTION_SPECS:
        for alias in (spec.name, *spec.aliases):
            registry[_normalize_name(alias)] = spec
    return registry


SELECTION_REGISTRY = _build_registry()


def selection_plugin_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in SELECTION_SPECS)


def get_selection_plugin(name: str) -> SelectionPluginSpec:
    try:
        return SELECTION_REGISTRY[_normalize_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported selection: {name}") from exc


def build_selection_fn(config: GAConfig) -> SelectionFn:
    spec = get_selection_plugin(config.selection)
    if not spec.supports_algorithm(config.algorithm):
        supported = ", ".join(spec.compatible_algorithms)
        raise ValueError(
            f"Selection '{config.selection}' is not compatible with algorithm "
            f"'{config.algorithm}'. Supported algorithms: {supported}"
        )
    return spec.build_fn(config)
