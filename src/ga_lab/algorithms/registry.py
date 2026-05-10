from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ga_lab.core.crossover import CrossoverFn
from ga_lab.core.mutation import MutationFn
from ga_lab.core.representation import InitFn
from ga_lab.core.selection import SelectionFn


class AlgorithmFn(Protocol):
    def __call__(
        self,
        config: Any,
        problem: object,
        selection_fn: SelectionFn,
        crossover_fn: CrossoverFn,
        mutation_fn: MutationFn,
        init_fn: InitFn,
        rng: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]: ...


AlgorithmLoader = Callable[[], AlgorithmFn]


@dataclass(frozen=True, slots=True)
class AlgorithmPluginSpec:
    name: str
    load_fn: AlgorithmLoader
    aliases: tuple[str, ...] = ()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _load_single_objective() -> AlgorithmFn:
    from ga_lab.algorithms.single_objective import run_single_objective_ga

    return run_single_objective_ga


def _load_nsga2() -> AlgorithmFn:
    from ga_lab.algorithms.nsga2 import run_nsga2

    return run_nsga2


def _load_hybrid_ga() -> AlgorithmFn:
    from ga_lab.algorithms.hybrid_ga import run_hybrid_ga

    return run_hybrid_ga


ALGORITHM_SPECS = (
    AlgorithmPluginSpec(name="ga", load_fn=_load_single_objective),
    AlgorithmPluginSpec(name="nsga2", load_fn=_load_nsga2, aliases=("nsga_ii", "nsga-ii")),
    AlgorithmPluginSpec(
        name="hybrid_ga",
        load_fn=_load_hybrid_ga,
        aliases=("memetic_ga", "hybrid-ga", "memetic-ga"),
    ),
)


def _build_registry() -> dict[str, AlgorithmPluginSpec]:
    registry: dict[str, AlgorithmPluginSpec] = {}
    for spec in ALGORITHM_SPECS:
        for alias in (spec.name, *spec.aliases):
            registry[_normalize_name(alias)] = spec
    return registry


ALGORITHM_REGISTRY = _build_registry()


def algorithm_plugin_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in ALGORITHM_SPECS)


def get_algorithm_plugin(name: str) -> AlgorithmPluginSpec:
    try:
        return ALGORITHM_REGISTRY[_normalize_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported algorithm: {name}") from exc


def build_algorithm_from_name(name: str) -> AlgorithmFn:
    return get_algorithm_plugin(name).load_fn()
