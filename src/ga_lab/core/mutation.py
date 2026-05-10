from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ga_lab.core.representation import Genome

if TYPE_CHECKING:
    from ga_lab.config import GAConfig


MutationFn = Callable[[Genome, random.Random], Genome]
MutationBuilder = Callable[["GAConfig"], MutationFn]


@dataclass(frozen=True, slots=True)
class MutationPluginSpec:
    name: str
    build_fn: MutationBuilder
    compatible_representations: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def supports_representation(self, representation: str) -> bool:
        normalized = _normalize_name(representation)
        return normalized in {_normalize_name(value) for value in self.compatible_representations}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _get_float(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def bit_flip_mutation(genome: Genome, mutation_rate: float, rng: random.Random) -> Genome:
    return [1.0 - gene if rng.random() < mutation_rate else gene for gene in genome]


def gaussian_mutation(
    genome: Genome,
    mutation_rate: float,
    rng: random.Random,
    sigma: float = 0.1,
    low: float = 0.0,
    high: float = 1.0,
) -> Genome:
    if low > high:
        low, high = high, low
    return [
        max(low, min(high, gene + rng.gauss(0.0, sigma))) if rng.random() < mutation_rate else gene
        for gene in genome
    ]


def swap_mutation(genome: Genome, mutation_rate: float, rng: random.Random) -> Genome:
    mutated = genome[:]
    if len(mutated) < 2:
        return mutated
    if rng.random() >= mutation_rate:
        return mutated
    a, b = rng.sample(range(len(mutated)), 2)
    mutated[a], mutated[b] = mutated[b], mutated[a]
    return mutated


def inversion_mutation(genome: Genome, mutation_rate: float, rng: random.Random) -> Genome:
    mutated = genome[:]
    if len(mutated) < 2:
        return mutated
    if rng.random() >= mutation_rate:
        return mutated
    a, b = sorted(rng.sample(range(len(mutated)), 2))
    mutated[a : b + 1] = reversed(mutated[a : b + 1])
    return mutated


def _build_bit_flip(config: GAConfig) -> MutationFn:
    return lambda genome, rng: bit_flip_mutation(genome, config.mutation_rate, rng)


def _build_gaussian(config: GAConfig) -> MutationFn:
    sigma = _get_float(config.mutation_options.get("sigma"), 0.1)
    low = _get_float(config.representation_options.get("low"), 0.0)
    high = _get_float(config.representation_options.get("high"), 1.0)
    return lambda genome, rng: gaussian_mutation(
        genome,
        config.mutation_rate,
        rng,
        sigma=sigma,
        low=low,
        high=high,
    )


def _build_swap(config: GAConfig) -> MutationFn:
    return lambda genome, rng: swap_mutation(genome, config.mutation_rate, rng)


def _build_inversion(config: GAConfig) -> MutationFn:
    return lambda genome, rng: inversion_mutation(genome, config.mutation_rate, rng)


MUTATION_SPECS = (
    MutationPluginSpec(
        name="bit_flip",
        build_fn=_build_bit_flip,
        compatible_representations=("bit",),
    ),
    MutationPluginSpec(
        name="gaussian",
        build_fn=_build_gaussian,
        compatible_representations=("real",),
    ),
    MutationPluginSpec(
        name="swap",
        build_fn=_build_swap,
        compatible_representations=("permutation",),
    ),
    MutationPluginSpec(
        name="inversion",
        build_fn=_build_inversion,
        compatible_representations=("permutation",),
    ),
)


def _build_registry() -> dict[str, MutationPluginSpec]:
    registry: dict[str, MutationPluginSpec] = {}
    for spec in MUTATION_SPECS:
        for alias in (spec.name, *spec.aliases):
            registry[_normalize_name(alias)] = spec
    return registry


MUTATION_REGISTRY = _build_registry()


def mutation_plugin_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in MUTATION_SPECS)


def get_mutation_plugin(name: str) -> MutationPluginSpec:
    try:
        return MUTATION_REGISTRY[_normalize_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported mutation: {name}") from exc


def build_mutation_fn(config: GAConfig) -> MutationFn:
    spec = get_mutation_plugin(config.mutation)
    if not spec.supports_representation(config.representation):
        raise ValueError(
            f"Mutation '{config.mutation}' is incompatible with representation "
            f"'{config.representation}'"
        )
    return spec.build_fn(config)
