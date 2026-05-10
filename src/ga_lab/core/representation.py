from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ga_lab.config import GAConfig


Genome = list[float]
Population = list[Genome]
InitFn = Callable[[random.Random, int], Genome]
RepairFn = Callable[[Genome, int], Genome]
ValidateFn = Callable[[Genome, int], None]


def _get_float(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


@dataclass(slots=True)
class RepresentationAdapter:
    name: str
    init_fn: InitFn
    repair_fn: RepairFn
    validate_fn: ValidateFn

    def initialize(self, rng: random.Random, length: int) -> Genome:
        genome = self.init_fn(rng, length)
        repaired = self.repair_fn(genome, length)
        self.validate_fn(repaired, length)
        return repaired

    def repair(self, genome: Genome, length: int) -> Genome:
        repaired = self.repair_fn(genome, length)
        self.validate_fn(repaired, length)
        return repaired


@dataclass(frozen=True, slots=True)
class RepresentationPluginSpec:
    name: str
    build_fn: Callable[[GAConfig], RepresentationAdapter]
    aliases: tuple[str, ...] = ()


def random_bit_genome(length: int, rng: random.Random) -> Genome:
    return [float(rng.randint(0, 1)) for _ in range(length)]


def random_real_genome(
    length: int,
    rng: random.Random,
    low: float = 0.0,
    high: float = 1.0,
) -> Genome:
    if low > high:
        low, high = high, low
    return [rng.uniform(low, high) for _ in range(length)]


def random_permutation_genome(length: int, rng: random.Random) -> Genome:
    values = list(range(length))
    rng.shuffle(values)
    return [float(value) for value in values]


def _repair_bit_genome(genome: Genome, _length: int) -> Genome:
    return [1.0 if float(gene) >= 0.5 else 0.0 for gene in genome]


def _validate_bit_genome(genome: Genome, length: int) -> None:
    if len(genome) != length:
        raise ValueError("Bit genome length does not match config.genome_length")
    if any(gene not in {0.0, 1.0} for gene in genome):
        raise ValueError("Bit representation requires genes in {0.0, 1.0}")


def _repair_real_genome(low: float, high: float, genome: Genome, _length: int) -> Genome:
    if low > high:
        low, high = high, low
    return [max(low, min(high, float(gene))) for gene in genome]


def _validate_real_genome(low: float, high: float, genome: Genome, length: int) -> None:
    if low > high:
        low, high = high, low
    if len(genome) != length:
        raise ValueError("Real genome length does not match config.genome_length")
    if any(float(gene) < low or float(gene) > high for gene in genome):
        raise ValueError("Real representation genes must stay inside configured bounds")


def _repair_permutation_genome(genome: Genome, length: int) -> Genome:
    rounded = [int(round(gene)) for gene in genome[:length]]
    seen: set[int] = set()
    repaired: list[int] = []
    for value in rounded:
        if 0 <= value < length and value not in seen:
            repaired.append(value)
            seen.add(value)
    missing = [value for value in range(length) if value not in seen]
    repaired.extend(missing)
    return [float(value) for value in repaired[:length]]


def _validate_permutation_genome(genome: Genome, length: int) -> None:
    if len(genome) != length:
        raise ValueError("Permutation genome length does not match config.genome_length")
    decoded = [int(round(gene)) for gene in genome]
    if sorted(decoded) != list(range(length)):
        raise ValueError("Permutation representation requires a permutation of 0..length-1")


def _build_bit_representation(_: GAConfig) -> RepresentationAdapter:
    return RepresentationAdapter(
        name="bit",
        init_fn=lambda rng, length: random_bit_genome(length, rng),
        repair_fn=_repair_bit_genome,
        validate_fn=_validate_bit_genome,
    )


def _build_real_representation(config: GAConfig) -> RepresentationAdapter:
    low = _get_float(config.representation_options.get("low"), 0.0)
    high = _get_float(config.representation_options.get("high"), 1.0)
    return RepresentationAdapter(
        name="real",
        init_fn=lambda rng, length: random_real_genome(length, rng, low=low, high=high),
        repair_fn=lambda genome, length: _repair_real_genome(low, high, genome, length),
        validate_fn=lambda genome, length: _validate_real_genome(low, high, genome, length),
    )


def _build_permutation_representation(_: GAConfig) -> RepresentationAdapter:
    return RepresentationAdapter(
        name="permutation",
        init_fn=lambda rng, length: random_permutation_genome(length, rng),
        repair_fn=_repair_permutation_genome,
        validate_fn=_validate_permutation_genome,
    )


REPRESENTATION_SPECS = (
    RepresentationPluginSpec(name="bit", build_fn=_build_bit_representation),
    RepresentationPluginSpec(name="real", build_fn=_build_real_representation),
    RepresentationPluginSpec(name="permutation", build_fn=_build_permutation_representation),
)


def _build_registry() -> dict[str, RepresentationPluginSpec]:
    registry: dict[str, RepresentationPluginSpec] = {}
    for spec in REPRESENTATION_SPECS:
        for alias in (spec.name, *spec.aliases):
            registry[_normalize_name(alias)] = spec
    return registry


REPRESENTATION_REGISTRY = _build_registry()


def representation_plugin_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in REPRESENTATION_SPECS)


def get_representation_plugin(name: str) -> RepresentationPluginSpec:
    try:
        return REPRESENTATION_REGISTRY[_normalize_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported representation: {name}") from exc


def random_genome_from_config(rng: random.Random, length: int, config: GAConfig) -> Genome:
    return build_representation_adapter(config).initialize(rng, length)


def build_init_fn(config: GAConfig) -> InitFn:
    adapter = build_representation_adapter(config)
    return adapter.initialize


def build_representation_adapter(config: GAConfig) -> RepresentationAdapter:
    return get_representation_plugin(config.representation).build_fn(config)
