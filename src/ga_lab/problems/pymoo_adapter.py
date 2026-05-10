from __future__ import annotations

import importlib.util
from typing import Any

from ga_lab.experiment.mo_metrics import validate_objective_vector
from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


class PymooBackedProblem:
    compatible_representations = ("real",)

    def __init__(
        self,
        *,
        pymoo_name: str,
        problem_name: str,
        objective_count: int = 2,
        variable_count: int = 6,
    ) -> None:
        if importlib.util.find_spec("pymoo") is None:
            raise ImportError("pymoo is required for pymoo-backed benchmark adapters")
        if objective_count < 2:
            raise ValueError("pymoo-backed benchmark adapters require objective_count >= 2")
        if variable_count < objective_count:
            raise ValueError("variable_count must be >= objective_count")

        from pymoo.problems import get_problem

        self.name = problem_name
        self.pymoo_name = pymoo_name
        self.objective_count = objective_count
        self.variable_count = variable_count
        self.default_objective_directions = tuple(False for _ in range(objective_count))
        self._problem = get_problem(pymoo_name, n_var=variable_count, n_obj=objective_count)
        self.source_lower_bounds = [float(value) for value in self._problem.xl.tolist()]
        self.source_upper_bounds = [float(value) for value in self._problem.xu.tolist()]

    def _denormalize(self, genome: Genome) -> list[float]:
        if len(genome) != self.variable_count:
            raise ValueError(
                f"{self.name} requires genome_length == {self.variable_count}, "
                f"got {len(genome)}"
            )
        denormalized: list[float] = []
        for index, value in enumerate(genome):
            clipped = min(1.0, max(0.0, float(value)))
            lower = self.source_lower_bounds[index]
            upper = self.source_upper_bounds[index]
            denormalized.append(lower + clipped * (upper - lower))
        return denormalized

    def fitness(self, genome: Genome) -> Fitness:
        denormalized = self._denormalize(genome)
        raw = self._problem.evaluate([denormalized], return_values_of=["F"])
        first_row: Any
        if hasattr(raw, "tolist"):
            as_list = raw.tolist()
            first_row = as_list[0] if as_list and isinstance(as_list[0], list) else as_list
        elif isinstance(raw, list):
            first_row = raw[0] if raw and isinstance(raw[0], list) else raw
        else:
            raise ValueError(f"{self.name} returned an unsupported objective payload: {type(raw)!r}")
        values = [float(value) for value in first_row]
        validate_objective_vector(values, label=f"{self.name}.fitness")
        return values

    def optimal_fitness(self, genome_length: int) -> list[float] | None:
        if genome_length != self.variable_count:
            return None
        return None

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.variable_count,
            default_objective_directions=self.default_objective_directions,
        )

    def source_bounds(self) -> list[tuple[float, float]]:
        return list(zip(self.source_lower_bounds, self.source_upper_bounds, strict=True))
