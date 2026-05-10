from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Protocol, cast

Genome = list[float]
Fitness = float | tuple[float, ...] | list[float]


@dataclass(frozen=True, slots=True)
class ProblemMetadata:
    name: str
    compatible_representations: tuple[str, ...] = field(default_factory=tuple)
    exact_genome_length: int | None = None
    min_genome_length: int | None = None
    default_objective_directions: tuple[bool, ...] = field(default_factory=tuple)


class Problem(Protocol):
    name: str

    def fitness(self, genome: Genome) -> Fitness: ...

    def optimal_fitness(self, genome_length: int) -> Fitness | None: ...

    def metadata(self) -> ProblemMetadata: ...


def as_fitness_vector(value: Fitness) -> list[float]:
    if isinstance(value, numbers.Real):
        return [float(value)]
    vector = cast(tuple[float, ...] | list[float], value)
    return [float(item) for item in vector]


def objective_count(value: Fitness) -> int:
    return len(as_fitness_vector(value))


def best_index(values: list[list[float]], maximize: bool) -> int:
    if not values:
        raise ValueError("fitness values cannot be empty")
    objective_zero = [item[0] for item in values]
    return (
        max(range(len(objective_zero)), key=objective_zero.__getitem__)
        if maximize
        else min(range(len(objective_zero)), key=objective_zero.__getitem__)
    )


def problem_metadata(problem) -> ProblemMetadata:
    metadata_fn = getattr(problem, "metadata", None)
    if callable(metadata_fn):
        metadata = metadata_fn()
        if isinstance(metadata, ProblemMetadata):
            return metadata

    compatible = tuple(
        str(value)
        for value in getattr(problem, "compatible_representations", ())
        if isinstance(value, str)
    )
    default_directions = tuple(
        bool(value) for value in getattr(problem, "default_objective_directions", ())
    )
    return ProblemMetadata(
        name=str(getattr(problem, "name", problem.__class__.__name__)),
        compatible_representations=compatible,
        exact_genome_length=getattr(problem, "exact_genome_length", None),
        min_genome_length=getattr(problem, "min_genome_length", None),
        default_objective_directions=default_directions,
    )
