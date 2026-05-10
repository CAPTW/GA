from __future__ import annotations

from math import cos, pi, sin

from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


class DTLZ3Problem:
    name = "dtlz3"
    compatible_representations = ("real",)

    def __init__(self, objective_count: int = 2) -> None:
        if objective_count < 2:
            raise ValueError("DTLZ3 requires objective_count >= 2")
        self.objective_count = objective_count
        self.default_objective_directions = tuple(False for _ in range(objective_count))

    def fitness(self, genome: Genome) -> Fitness:
        if len(genome) < self.objective_count:
            raise ValueError(
                f"DTLZ3 requires genome_length >= objective_count ({self.objective_count})"
            )
        m = self.objective_count
        tail = genome[m - 1 :]
        g = 100.0 * (
            len(tail)
            + sum((float(value) - 0.5) ** 2 - cos(20.0 * pi * (float(value) - 0.5)) for value in tail)
        )
        base = 1.0 + g
        objectives: list[float] = []
        for objective_index in range(m):
            value = base
            for gene in genome[: m - objective_index - 1]:
                value *= cos(float(gene) * pi / 2.0)
            if objective_index > 0:
                value *= sin(float(genome[m - objective_index - 1]) * pi / 2.0)
            objectives.append(float(value))
        return objectives

    def optimal_fitness(self, genome_length: int) -> list[float] | None:
        if genome_length < self.objective_count:
            return None
        return [0.0 for _ in range(self.objective_count)]

    def hypervolume_reference_point(self, directions: list[bool]) -> list[float]:
        if len(directions) != self.objective_count:
            raise ValueError("directions length must match DTLZ3 objective count")
        reference: list[float] = []
        for maximize in directions:
            reference.append(-0.2 if maximize else 2.0)
        return reference

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            min_genome_length=self.objective_count,
            default_objective_directions=self.default_objective_directions,
        )
