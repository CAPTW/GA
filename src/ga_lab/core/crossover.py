from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ga_lab.core.representation import Genome

if TYPE_CHECKING:
    from ga_lab.config import GAConfig


CrossoverFn = Callable[[Genome, Genome, random.Random], tuple[Genome, Genome]]
CrossoverBuilder = Callable[["GAConfig"], CrossoverFn]


@dataclass(frozen=True, slots=True)
class CrossoverPluginSpec:
    name: str
    build_fn: CrossoverBuilder
    compatible_representations: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def supports_representation(self, representation: str) -> bool:
        normalized = _normalize_name(representation)
        return normalized in {_normalize_name(value) for value in self.compatible_representations}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _get_float(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def one_point_crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
) -> tuple[Genome, Genome]:
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")
    if len(parent_a) < 2:
        return parent_a[:], parent_b[:]
    point = rng.randint(1, len(parent_a) - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def uniform_crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
) -> tuple[Genome, Genome]:
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")
    child_a: Genome = []
    child_b: Genome = []
    for value_a, value_b in zip(parent_a, parent_b, strict=True):
        if rng.random() < 0.5:
            child_a.append(value_a)
            child_b.append(value_b)
        else:
            child_a.append(value_b)
            child_b.append(value_a)
    return child_a, child_b


def order_crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
) -> tuple[Genome, Genome]:
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")
    size = len(parent_a)
    if size < 2:
        return parent_a[:], parent_b[:]
    start = rng.randrange(size)
    end = rng.randrange(size)
    if start > end:
        start, end = end, start
    if start == end:
        end = (start + 1) % size
        if end < start:
            start, end = end, start

    def _build_child(first_parent: Genome, second_parent: Genome) -> Genome:
        child: list[float | None] = [None] * size
        segment = first_parent[start : end + 1]
        child[start : end + 1] = segment
        remaining = [value for value in second_parent if value not in segment]
        cursor = 0
        for pos in range(end + 1, end + size + 1):
            actual_pos = pos % size
            if child[actual_pos] is None:
                child[actual_pos] = remaining[cursor]
                cursor += 1
        if any(value is None for value in child):
            raise ValueError("Order crossover failed to fill the child genome")
        return [float(value) for value in child if value is not None]

    return _build_child(parent_a, parent_b), _build_child(parent_b, parent_a)


def arithmetic_crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
    alpha: float = 0.5,
) -> tuple[Genome, Genome]:
    _ = rng
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have the same genome length")
    child_a: Genome = []
    child_b: Genome = []
    for value_a, value_b in zip(parent_a, parent_b, strict=True):
        child_a.append(alpha * value_a + (1.0 - alpha) * value_b)
        child_b.append((1.0 - alpha) * value_a + alpha * value_b)
    return child_a, child_b


def _build_one_point(_: GAConfig) -> CrossoverFn:
    return one_point_crossover


def _build_uniform(_: GAConfig) -> CrossoverFn:
    return uniform_crossover


def _build_order(_: GAConfig) -> CrossoverFn:
    return order_crossover


def _build_arithmetic(config: GAConfig) -> CrossoverFn:
    alpha = _get_float(config.crossover_options.get("alpha"), 0.5)
    return lambda parent_a, parent_b, rng: arithmetic_crossover(
        parent_a,
        parent_b,
        rng,
        alpha=alpha,
    )


CROSSOVER_SPECS = (
    CrossoverPluginSpec(
        name="one_point",
        build_fn=_build_one_point,
        compatible_representations=("bit", "real"),
    ),
    CrossoverPluginSpec(
        name="uniform",
        build_fn=_build_uniform,
        compatible_representations=("bit", "real"),
    ),
    CrossoverPluginSpec(
        name="arithmetic",
        build_fn=_build_arithmetic,
        compatible_representations=("real",),
    ),
    CrossoverPluginSpec(
        name="order",
        build_fn=_build_order,
        compatible_representations=("permutation",),
    ),
)


def _build_registry() -> dict[str, CrossoverPluginSpec]:
    registry: dict[str, CrossoverPluginSpec] = {}
    for spec in CROSSOVER_SPECS:
        for alias in (spec.name, *spec.aliases):
            registry[_normalize_name(alias)] = spec
    return registry


CROSSOVER_REGISTRY = _build_registry()


def crossover_plugin_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in CROSSOVER_SPECS)


def get_crossover_plugin(name: str) -> CrossoverPluginSpec:
    try:
        return CROSSOVER_REGISTRY[_normalize_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported crossover: {name}") from exc


def build_crossover_fn(config: GAConfig) -> CrossoverFn:
    spec = get_crossover_plugin(config.crossover)
    if not spec.supports_representation(config.representation):
        raise ValueError(
            f"Crossover '{config.crossover}' is incompatible with representation "
            f"'{config.representation}'"
        )
    return spec.build_fn(config)
