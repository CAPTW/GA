from __future__ import annotations

from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


class ZDT2Problem:
    name = "zdt2"
    compatible_representations = ("real",)
    default_objective_directions = (False, False)

    def __init__(self, name: str = "zdt2") -> None:
        self.name = name

    def fitness(self, genome: Genome) -> Fitness:
        if len(genome) < 2:
            raise ValueError("ZDT2 requires at least 2 real genes")
        f1 = genome[0]
        if f1 < 0:
            f1 = 0.0
        if f1 > 1:
            f1 = 1.0
        tail = genome[1:]
        mean_tail = sum(tail) / len(tail)
        g = 1.0 + (9.0 * mean_tail)
        ratio = f1 / g if g != 0.0 else 0.0
        f2 = g * (1.0 - (ratio * ratio))
        return (f1, f2)

    def optimal_fitness(self, genome_length: int) -> list[float] | None:
        if genome_length < 2:
            return None
        return [0.0, 0.0]

    def hypervolume_reference_point(self, directions: list[bool]) -> list[float]:
        bounds = [(0.0, 1.0), (0.0, 10.0)]
        if len(directions) != len(bounds):
            raise ValueError("directions length must match ZDT2 objective count")
        reference: list[float] = []
        for (low, high), maximize in zip(bounds, directions, strict=True):
            margin = (high - low) * 0.1
            reference.append(low - margin if maximize else high + margin)
        return reference

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            min_genome_length=2,
            default_objective_directions=self.default_objective_directions,
        )
