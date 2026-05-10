from __future__ import annotations

from math import cos, pi, sin

from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


class DTLZ4Problem:
    name = "dtlz4"
    compatible_representations = ("real",)

    def __init__(self, objective_count: int = 2, alpha: float = 100.0) -> None:
        if objective_count < 2:
            raise ValueError("DTLZ4 requires objective_count >= 2")
        self.objective_count = objective_count
        self.alpha = float(alpha)
        self.default_objective_directions = tuple(False for _ in range(objective_count))

    def fitness(self, genome: Genome) -> Fitness:
        if len(genome) < self.objective_count:
            raise ValueError(
                f"DTLZ4 requires genome_length >= objective_count ({self.objective_count})"
            )
        m = self.objective_count
        g = sum((float(value) - 0.5) ** 2 for value in genome[m - 1 :])
        base = 1.0 + g
        angular = [float(gene) ** self.alpha for gene in genome[: m - 1]]
        objectives: list[float] = []
        for objective_index in range(m):
            value = base
            for gene in angular[: m - objective_index - 1]:
                value *= cos(gene * pi / 2.0)
            if objective_index > 0:
                value *= sin(angular[m - objective_index - 1] * pi / 2.0)
            objectives.append(float(value))
        return objectives

    def optimal_fitness(self, genome_length: int) -> list[float] | None:
        if genome_length < self.objective_count:
            return None
        return [0.0 for _ in range(self.objective_count)]

    def hypervolume_reference_point(self, directions: list[bool]) -> list[float]:
        if len(directions) != self.objective_count:
            raise ValueError("directions length must match DTLZ4 objective count")
        return [(-0.1 if maximize else 1.1) for maximize in directions]

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            min_genome_length=self.objective_count,
            default_objective_directions=self.default_objective_directions,
        )
